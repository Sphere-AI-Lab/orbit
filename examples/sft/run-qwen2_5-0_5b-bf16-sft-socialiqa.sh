#!/usr/bin/env bash
# SFT on allenai/social_i_qa converted with tools/convert_sft_dataset_to_orbit.py.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"

SFT_DATA_ROOT="${SFT_DATA_ROOT:-${ORBIT_ROOT}/data/sft}"
SFT_DATASET_NAME="socialiqa"
SFT_TRAIN_JSONL_DEFAULT="${SFT_DATA_ROOT}/socialiqa/train.jsonl"
SFT_SAVE_DIR_SUFFIX="socialiqa"
SFT_TOTAL_EPOCHS="${SFT_TOTAL_EPOCHS:-3}"

SFT_DATASET_SAFE="${SFT_DATASET_NAME//[^a-zA-Z0-9]/_}"

# === Recipe identity ===
LAUNCHER_NAME=${LAUNCHER_NAME:-run_qwen25_05b_bf16_sft_${SFT_DATASET_SAFE}}
WANDB_PROJECT=${WANDB_PROJECT:-orbit-release}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${RUN_LOG:-${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log}"

# === Paths ===
: "${HF_CKPT:?set HF_CKPT to a Hugging Face checkpoint path}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to a Megatron torch_dist checkpoint path}"
SAVE_DIR="${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/Qwen2.5-0.5B-Instruct_sft_${SFT_SAVE_DIR_SUFFIX}}"
TRAIN_JSONL="${TRAIN_JSONL:-${SFT_TRAIN_JSONL_DEFAULT}}"
: "${TRAIN_JSONL:?set TRAIN_JSONL or SFT_TRAIN_JSONL_DEFAULT to a chat-format training jsonl path}"

# === Resources ===
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-32}

# === Model args ===
source "${ORBIT_ROOT}/orbit_plugins/model_args/qwen2.5-0.5B.sh"   # provides MODEL_ARGS=(...)

# === Training schedule ===
TOTAL_EPOCHS="${TOTAL_EPOCHS:-${SFT_TOTAL_EPOCHS:-3}}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-${SFT_ROLLOUT_BATCH_SIZE:-256}}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${SFT_GLOBAL_BATCH_SIZE:-64}}"
TRAIN_ROWS=${TRAIN_ROWS:-$(wc -l < "${TRAIN_JSONL}")}
NUM_ROLLOUT=${NUM_ROLLOUT:-$(( (TRAIN_ROWS * TOTAL_EPOCHS + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE ))}

# === ARGS arrays ===
COLOCATE_ARGS=( --colocate )

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${MEGATRON_LOAD}"
    --save "${SAVE_DIR}"
    --save-interval "${SAVE_INTERVAL:-200}"
    --no-save-optim
    --no-save-rng
    --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_JSONL}"
    --input-key messages
    --rollout-shuffle
    --rollout-function-path orbit.rollout.sft_rollout.generate_rollout
    --loss-mask-type "${LOSS_MASK_TYPE:-qwen}"
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt 1
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-1e-5}"
    --lr-decay-style "${LR_DECAY_STYLE:-cosine}"
    --weight-decay "${WEIGHT_DECAY:-0.01}"
    --adam-beta1 "${ADAM_BETA1:-0.9}"
    --adam-beta2 "${ADAM_BETA2:-0.999}"
)

RL_ARGS=()

LOSS_ARGS=(
    --training-mode sft
    --loss-type sft_loss
    --disable-compute-advantages-and-returns
    --calculate-per-token-loss
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --disable-wandb-random-suffix
)

PERF_ARGS=(
    --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE:-1}"
    --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE:-1}"
    --context-parallel-size "${CONTEXT_PARALLEL_SIZE:-1}"
    --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE:-1}"
    --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE:-1}"
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-1}"
    --sequence-parallel
)

EVAL_ARGS=()
SGLANG_ARGS=()

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --no-offload-train
    --no-offload-train-async
    --cuda-graph-impl local
    --cuda-graph-scope full_iteration
    --te-rng-tracker
    --no-check-for-nan-in-loss-and-grad
)

if ! is_true "${SFT_GRADIENT_ACCUMULATION_FUSION:-0}"; then
    MISC_ARGS+=( --no-gradient-accumulation-fusion )
fi

DEBUG_ARGS=()
PEFT_ARGS=()
if [[ -n "${SFT_PEFT_ARGS:-}" ]]; then
    # shellcheck disable=SC2206  # intentional word splitting of a flat flag string
    PEFT_ARGS=( ${SFT_PEFT_ARGS} )
fi
if [[ -n "${SFT_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206  # intentional word splitting of a flat flag string
    MISC_ARGS+=( ${SFT_EXTRA_ARGS} )
fi

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
