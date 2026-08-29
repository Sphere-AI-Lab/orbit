#!/usr/bin/env bash
# Qwen3-0.6B BF16 full-finetune GRPO smoke for the true-on-policy ladder.
# TOP=1 adds --true-on-policy (contract qwen3_dense_true_on_policy_v1):
# deterministic sglang rollout + prefill recompute + batch-invariant Megatron
# kernels + fusion bans + bf16 log-prob pipeline. TP=2 exercises the Phase-4
# TP-correct full-vocab gather. This measures the remaining cross-kernel gap;
# exact parity waits for Phase 5. Compare train_rollout_logprob_abs_diff against
# a TOP=0 run of this same script.
# NOTE: no --sequence-parallel (the contract rejects it); --attention-backend
# flash is required by batch-invariant mode.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"

# === Recipe identity ===
LAUNCHER_NAME=smoke_qwen3_06b_top
WANDB_PROJECT=${WANDB_PROJECT:-orbit-release}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log"

# === Paths ===
: "${HF_CKPT:?set HF_CKPT to the Qwen3-0.6B Hugging Face checkpoint path}"
# HF-dir loading: --load falls through to _load_checkpoint_hf (bridge mode).
MEGATRON_LOAD="${MEGATRON_LOAD:-${HF_CKPT}}"
SAVE_DIR="${ORBIT_ROOT}/orbit_ckpts/Qwen3-0.6B_top_smoke"
: "${TRAIN_JSONL:?set TRAIN_JSONL to a training jsonl path}"

# === Resources ===
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-2}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"

# === Model args ===
source "${ORBIT_ROOT}/miles_plugins/model_args/qwen3-0.6B.sh"   # provides MODEL_ARGS=(...)

# === Training schedule ===
NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-512}"
TP="${TP:-1}"

# === ARGS arrays ===
COLOCATE_ARGS=()

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${MEGATRON_LOAD}"
    --save "${SAVE_DIR}/actor"
    --save-interval 200
    --no-save-optim
    --no-save-rng
    --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_JSONL}"
    --input-key prompt
    --label-key label
    --apply-chat-template
    --apply-chat-template-kwargs '{"enable_thinking": false}'
    --rollout-shuffle
    --rm-type math
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    --rollout-temperature 1.0
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.999
)

RL_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.0
    --kl-loss-type k1
    --kl-coef 0.0
    --entropy-coef 0.0
    --eps-clip 0.2
    --normalize-advantages
)
if [ "${TOP:-0}" = "1" ]; then
    RL_ARGS+=(--true-on-policy)
fi

LOSS_ARGS=(
    --calculate-per-token-loss
)

WANDB_ARGS=()

PERF_ARGS=(
    --tensor-model-parallel-size "${TP}"
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
)

EVAL_ARGS=()

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.60}"
    --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-1024}"
    --sglang-force-native-ops
    --sglang-attention-backend triton
    --sglang-sampling-backend pytorch
    --router-disable-circuit-breaker
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --no-gradient-accumulation-fusion
    --no-offload-train
    --no-offload-train-async
    --no-offload-rollout
    --cuda-graph-impl local
    --cuda-graph-scope full_iteration
    --te-rng-tracker
    --no-check-for-nan-in-loss-and-grad
)

DEBUG_ARGS=(
    --log-passrate
)

PEFT_ARGS=()

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
