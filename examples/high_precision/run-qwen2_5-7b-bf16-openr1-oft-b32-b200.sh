#!/usr/bin/env bash
# Qwen2.5-7B BF16 + OFT (block-size 32) on PEFT-Arena openr1-50k. B200 variant
# of run-qwen2_5-7b-bf16-openr1-oft-b32.sh.
#
# Recipe (OFT block-size/eps, lr, batch sizes, num-rollout, n-samples, context
# lengths, KL/clip) is IDENTICAL to the H100 script -- only perf/memory knobs
# change to exploit B200's 192 GB HBM and ~2.3x BF16 throughput:
#   * max-tokens-per-gpu 16384 -> 32768 (2x H100)
#   * activation recompute KEPT ON (uniform / 1 layer): a previous draft of
#     this script disabled recompute and OOM'd at Megatron's BF16->FP32 cast
#     of the output logits at end of forward (peak-memory moment). On B200
#     the recompute tax is ~13% wall-clock but saves ~80 GB activations.
#   * 1 SGLang engine per GPU (8 engines instead of 2) for rollout parallelism
#   * sglang-mem-fraction-static 0.75 -> 0.85
#   * sglang-max-running-requests 128 -> 1024
#   * sglang-max-total-tokens 262144 -> 524288
#   * sglang-cuda-graph-max-bs 64 -> 256
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"
source "${ORBIT_ROOT}/scripts/lib/paths.sh"

# === Recipe identity ===
LAUNCHER_NAME=run_qwen25_7b_openr1_oft_b32_bs32_r1000_lr1e5_b200
WANDB_PROJECT=${WANDB_PROJECT:-orbit-release}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log"

# === Paths ===
: "${HF_CKPT:?set HF_CKPT to a Hugging Face checkpoint path}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to a Megatron torch_dist checkpoint path}"
SAVE_DIR="${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/exp_ckpt/${LAUNCHER_NAME#run_}_$(date +%Y%m%d_%H%M%S)}"
: "${TRAIN_JSONL:?set TRAIN_JSONL to a training jsonl path}"
EVAL_DATA_DIR=${EVAL_DATA_DIR:-}
EVAL_ORBIT_DIR=${EVAL_ORBIT_DIR:-${EVAL_DATA_DIR}}

# === Local checkpoint staging (Lustre -> NVMe) ===
LOCAL_STAGE_ROOT=${LOCAL_STAGE_ROOT:-${ORBIT_CACHE_DIR:-${HOME}/.cache/orbit}/stage}
STAGE_HF_CKPT_TO=${STAGE_HF_CKPT_TO-${LOCAL_STAGE_ROOT}/Qwen2.5-7B}
STAGE_MEGATRON_CKPT_TO=${STAGE_MEGATRON_CKPT_TO-${LOCAL_STAGE_ROOT}/Megatron-Bridge/checkpoints/Qwen2.5-7B}

# === Resources ===
GPUS_PER_NODE=8
RAY_NUM_CPUS=16

# === Model args ===
source "${ORBIT_ROOT}/orbit_plugins/model_args/qwen2.5-7B.sh"   # provides MODEL_ARGS=(...)

# === Training schedule (identical to H100 recipe) ===
TOTAL_EPOCHS=1000
NUM_ROLLOUT=1000
ROLLOUT_BATCH_SIZE=32
N_SAMPLES_PER_PROMPT=8
GLOBAL_BATCH_SIZE=256

# === ARGS arrays ===
COLOCATE_ARGS=( --colocate )

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${MEGATRON_LOAD}"
    --save "${SAVE_DIR}"
    --save-interval 20
    --no-save-optim
    --no-save-rng
    --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_JSONL}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --rollout-shuffle
    --rm-type custom
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len 8192
    --rollout-temperature 1.0
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --custom-rm-path orbit.peft.rewards.peft_arena_reward.peft_arena_reward
    --reward-key score
    --eval-reward-key score
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-5
    --lr-decay-style constant
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.999
)

RL_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.001
    --kl-loss-type low_var_kl
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.2
)

LOSS_ARGS=(
    --calculate-per-token-loss
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --disable-wandb-random-suffix
)

# B200: 2x H100 tokens-per-gpu, but keep activation recompute on. Dropping
# recompute at 40960 tokens hits OOM at the FP32 cast of model output logits
# (the peak-memory moment of forward) -- activations across 28 layers of 7B
# at 40k tokens are ~80-120 GB, which leaves no headroom for the FP32 cast.
PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 32768
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --sequence-parallel
)

EVAL_ARGS=(
    --eval-interval 20
    --eval-prompt-data math500 "${EVAL_ORBIT_DIR}/math500.jsonl" \
                       aime24  "${EVAL_ORBIT_DIR}/aime24.jsonl" \
                       amc23   "${EVAL_ORBIT_DIR}/amc23.jsonl"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len 8192
    --eval-top-k 1
    --skip-eval-before-train
    --eval-pass-k-values 1 2 4 8 16
)

# B200: 7B BF16 weights (~14 GB) fit per GPU with huge KV headroom, so use
# 1 GPU per SGLang engine (8 engines) and a much larger KV pool / running-
# requests budget. mem-fraction-static raised to 0.85 since OFT training
# memory is tiny once offloaded.
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.85
    --rollout-num-gpus 0
    --sglang-max-running-requests 1024
    --sglang-max-total-tokens 524288
    --sglang-cuda-graph-max-bs 256
    --router-disable-circuit-breaker
    --sglang-router-policy round_robin
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --offload-train
    --offload-train-async
    --offload-rollout
    --cuda-graph-impl local
    --cuda-graph-scope full_iteration
    --te-rng-tracker
    --no-check-for-nan-in-loss-and-grad
)

DEBUG_ARGS=(
    --log-passrate
    --log-reward-category acc
)

PEFT_ARGS=(
    --peft-method oft
    --peft-variant standard
    --oft-type canonical_oft
    --oft-block-size 32
    --oft-eps 6e-5
    --target-modules all-linear
)

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
