#!/usr/bin/env bash
# Qwen2.5-7B BF16 + OFT on PEFT-Arena openr1-50k. Self-contained launcher.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"
source "${ORBIT_ROOT}/scripts/lib/paths.sh"

# === Recipe identity ===
LAUNCHER_NAME=run_qwen25_7b_openr1_oft_all_b32_bs32_r1000_lr1e5
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

# === Training schedule ===
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

PERF_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 16384
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

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 4
    --sglang-mem-fraction-static 0.75
    --rollout-num-gpus 0
    # --sglang-server-concurrency 128
    --sglang-max-running-requests 128
    --sglang-max-total-tokens 262144
    --sglang-cuda-graph-max-bs 64
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
    --target-modules all
)

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
