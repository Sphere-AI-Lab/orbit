#!/usr/bin/env bash
# Kimi-K2.5 6-layer INT4 debug math OFT launcher. Self-contained launcher.
#
# Exports the small-model overrides for the debug 6-layer architecture.
# See examples/README.md for a full env-knob reference.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"
source "${ORBIT_ROOT}/scripts/lib/paths.sh"

# === Recipe identity + checkpoints ===
LAUNCHER_NAME=${LAUNCHER_NAME:-run_kimi_k25_debug6_int4_math_megatron_oft}
PRECISION_PROFILE="int4"
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${RUN_LOG:-${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log}"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
: "${HF_CKPT:?set HF_CKPT to a Hugging Face checkpoint path}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to a Megatron torch_dist checkpoint path}"
LOAD_CKPT=${LOAD_CKPT:-${MEGATRON_LOAD}}
MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-"${ORBIT_ROOT}/miles_plugins/model_args/kimi-k25-debug-6layer.sh"}
SAVE_DIR=${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/Kimi-K2.5-W4A16_math_oft}

# === Local checkpoint staging ===
LOCAL_STAGE_ROOT=${LOCAL_STAGE_ROOT:-${ORBIT_CACHE_DIR:-${HOME}/.cache/orbit}/stage}
STAGE_HF_CKPT_TO=${STAGE_HF_CKPT_TO-${LOCAL_STAGE_ROOT}/Kimi-K2.5-debug-6layer}
STAGE_MEGATRON_CKPT_TO=${STAGE_MEGATRON_CKPT_TO-${LOCAL_STAGE_ROOT}/Megatron-Bridge/checkpoints/Kimi-K2.5-debug-6layer}
FORCE_STAGE_HF_CKPT=${FORCE_STAGE_HF_CKPT:-0}
FORCE_STAGE_MEGATRON_CKPT=${FORCE_STAGE_MEGATRON_CKPT:-0}

# === Data ===
DATASET=${DATASET:-math}
: "${TRAIN_JSONL:?set TRAIN_JSONL to a training jsonl path}"
TEST_JSONL=${TEST_JSONL:-}

# === Resources + parallelism ===
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-64}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
CP_COMM_TYPE=${CP_COMM_TYPE:-}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-8}
SGLANG_EXPERT_PARALLEL_SIZE=${SGLANG_EXPERT_PARALLEL_SIZE:-8}
USE_ROLLOUT_ROUTING_REPLAY=${USE_ROLLOUT_ROUTING_REPLAY:-1}

# === Training schedule ===
# The previous NSPP=1 default zeroed group-normalized rewards, so grad_norm
# stayed at 0 and the train-vs-rollout logprob diff could not exercise
# adapter drift.
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-4}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-1024}
ROLLOUT_MAX_PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN:-1024}
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-4096}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
# The 6-layer debug checkpoint is not task-capable, so math rewards collapse
# to all-zero groups. Use random rewards by default to exercise real PEFT
# updates.
RM_TYPE=${RM_TYPE:-random}
DISABLE_EVAL=${DISABLE_EVAL:-1}
EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-1024}
EVAL_GENERATE_MAX_CONCURRENCY=${EVAL_GENERATE_MAX_CONCURRENCY:-128}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
NUM_ROLLOUT=${NUM_ROLLOUT:-200}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-4096}
LR=${LR:-1e-5}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000000}
EVAL_INTERVAL=${EVAL_INTERVAL:-10}

# === Rollout backend (SGLang) ===
# Keep the small debug run on the same train-vs-rollout parity surface as the
# full Kimi run. These are real mismatch controls, not metric suppression.
export USE_ROLLOUT_LOGPROBS=${USE_ROLLOUT_LOGPROBS:-0}
export SGLANG_OFT_MARLIN_MOE=${SGLANG_OFT_MARLIN_MOE:-1}
export SGLANG_FORCE_OFT_TRITON_MOE=${SGLANG_FORCE_OFT_TRITON_MOE:-0}
export SGLANG_OFT_EXPERT_PARITY_MODE=${SGLANG_OFT_EXPERT_PARITY_MODE:-0}
export SGLANG_ENABLE_FP32_LM_HEAD=${SGLANG_ENABLE_FP32_LM_HEAD:-1}
export SGLANG_ENABLE_DETERMINISTIC_INFERENCE=${SGLANG_ENABLE_DETERMINISTIC_INFERENCE:-1}
export SGLANG_MOE_ROUTER_FORCE_FP32=${SGLANG_MOE_ROUTER_FORCE_FP32:-1}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.50}
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-flashinfer}
SGLANG_MM_ATTENTION_BACKEND=${SGLANG_MM_ATTENTION_BACKEND:-triton_attn}
SGLANG_QUANTIZATION=${SGLANG_QUANTIZATION:-compressed-tensors}
export SGLANG_DISABLE_CUDNN_CHECK=${SGLANG_DISABLE_CUDNN_CHECK:-1}
export SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}
SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-${ROLLOUT_MAX_CONTEXT_LEN}}
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-1024}
SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-4096}
SGLANG_DISABLE_CUDA_GRAPH=${SGLANG_DISABLE_CUDA_GRAPH:-0}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-1}
ATTENTION_SOFTMAX_IN_FP32=${ATTENTION_SOFTMAX_IN_FP32:-0}

# === PEFT (OFT) + INT4 quantization ===
TARGET_MODULES=${TARGET_MODULES:-linear_q_up_proj,linear_kv_up_proj,linear_proj,linear_fc1,linear_fc2}
OFT_BLOCK_SIZE=${OFT_BLOCK_SIZE:-32}
OFT_EPS=${OFT_EPS:-6e-5}
OPEN_TRAINING_INT4_FAKE_QAT_FLAG=${OPEN_TRAINING_INT4_FAKE_QAT_FLAG:-1}
OPEN_TRAINING_INT4_GROUP_SIZE=${OPEN_TRAINING_INT4_GROUP_SIZE:-32}

# === W&B ===
ENABLE_WANDB=${ENABLE_WANDB:-auto}
WANDB_PROJECT=${WANDB_PROJECT:-orbit-release}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}

# === Checkpoint staging (Lustre → NVMe) ===
stage_hf_checkpoint_if_requested
stage_megatron_checkpoint_if_requested

# === Model args ===
source "${MODEL_ARGS_FILE}"   # provides MODEL_ARGS=(...)

# === ARGS arrays ===
COLOCATE_ARGS=( --colocate )

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${LOAD_CKPT}"
    --save "${SAVE_DIR}"
    --save-interval "${SAVE_INTERVAL}"
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
    --rm-type "${RM_TYPE}"
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    --rollout-temperature 1.0
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
    --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
    --use-rollout-routing-replay
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR}"
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
    --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
    --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}"
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --sequence-parallel
)
if [[ -n "${CP_COMM_TYPE}" ]]; then
    # shellcheck disable=SC2206
    PERF_ARGS+=( --cp-comm-type ${CP_COMM_TYPE} )
fi

EVAL_ARGS=( )

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
    --rollout-num-gpus 0
    --sglang-context-length "${SGLANG_CONTEXT_LENGTH}"
    --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
    --sglang-max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS}"
    --sglang-quantization "${SGLANG_QUANTIZATION}"
    --sglang-enable-fp32-lm-head
    --sglang-expert-parallel-size "${SGLANG_EXPERT_PARALLEL_SIZE}"
    --sglang-mm-attention-backend "${SGLANG_MM_ATTENTION_BACKEND}"
    --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}"
    --sglang-moe-router-force-fp32
    --router-disable-circuit-breaker
    --sglang-enable-deterministic-inference
)
if [[ "${SGLANG_DISABLE_CUDA_GRAPH,,}" =~ ^(1|true|yes|y|on)$ ]]; then
    SGLANG_ARGS+=( --sglang-disable-cuda-graph )
fi

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --trust-remote-code
    --offload-train
    --offload-train-async
    --offload-rollout
    --use-rollout-routing-replay
)

DEBUG_ARGS=(
    --log-passrate
)

PEFT_ARGS=(
    --peft-method oft
    --peft-variant standard
    --oft-type canonical_oft
    --oft-block-size "${OFT_BLOCK_SIZE}"
    --oft-eps "${OFT_EPS}"
    --target-modules "${TARGET_MODULES}"
)

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
