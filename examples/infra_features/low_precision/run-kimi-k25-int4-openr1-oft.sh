#!/usr/bin/env bash
# Kimi-K2.5 W4A16 INT4 + OFT on PEFT-Arena openr1-50k with CUDA 13.2 and CP=2. Self-contained launcher.
#
# CUDA 13.2 CP=2 variant of the PEFT-Arena OpenR1 launcher.
# Keeps the PEFT-Arena OpenR1 launcher's eval datasets/settings and only
# swaps the runtime environment to ${ORBIT_WORKSPACE_ROOT:-${HOME}/.cache/orbit/workspace}/orbit-workspace-cu13.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/load_cuda13_2_orbit_env.sh"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"
source "${ORBIT_ROOT}/scripts/lib/paths.sh"

# === Dataset conversion (lazy) ===
DATA_DIR=${DATA_DIR:-}
TRAIN_JSONL_DEFAULT=${TRAIN_JSONL_DEFAULT:-}
TEST_JSONL_DEFAULT=${TEST_JSONL_DEFAULT:-}
EVAL_DATA_DIR=${EVAL_DATA_DIR:-}
EVAL_ORBIT_DIR=${EVAL_ORBIT_DIR:-${EVAL_DATA_DIR}}
DISABLE_EVAL=${DISABLE_EVAL:-0}
PYTHON_BIN=${PYTHON_BIN:-${ORBIT_WORKSPACE_ROOT:-${HOME}/.cache/orbit/workspace}/orbit-workspace-cc/.venv/bin/python}

if [ ! -f "${TRAIN_JSONL_DEFAULT}" ] || [ ! -f "${TEST_JSONL_DEFAULT}" ]; then
    echo "[kimi-peft-arena-launcher] Converting PEFT-Arena openr1-50k -> ${DATA_DIR}" >&2
    "${PYTHON_BIN}" "${ORBIT_ROOT}/tools/convert_peftarena_data.py" \
        --output_dir "${DATA_DIR}"
fi

if ! is_true "${DISABLE_EVAL}"; then
    : "${EVAL_ORBIT_DIR:?set EVAL_ORBIT_DIR or EVAL_DATA_DIR to a directory for math eval JSONL files, or set DISABLE_EVAL=1}"
    if [ ! -f "${EVAL_ORBIT_DIR}/math500.jsonl" ] \
       || [ ! -f "${EVAL_ORBIT_DIR}/aime24.jsonl" ] \
       || [ ! -f "${EVAL_ORBIT_DIR}/amc23.jsonl" ]; then
        echo "[kimi-peft-arena-launcher] Converting math_eval test sets -> ${EVAL_ORBIT_DIR}" >&2
        "${PYTHON_BIN}" "${ORBIT_ROOT}/tools/convert_math_eval_to_orbit.py" \
            --mode math_alignment \
            --output_dir "${EVAL_ORBIT_DIR}" --force
    fi
fi

# Put Triton JIT cache on node-local storage to avoid Lustre stale-handle races.
if [[ -z "${TRITON_CACHE_DIR:-}" ]]; then
    _triton_user="${USER:-$(id -un 2>/dev/null || printf 'user')}"
    export TRITON_CACHE_DIR="/tmp/triton_cache_${_triton_user}"
fi
mkdir -p "${TRITON_CACHE_DIR}"

# === Recipe identity ===
LAUNCHER_NAME=${LAUNCHER_NAME:-run_kimi_k25_int4_peft_arena_openr1_oft_cp2_cuda132}
PRECISION_PROFILE="int4"
REQUIRE_MEGATRON_LOAD=${REQUIRE_MEGATRON_LOAD:-1}
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${RUN_LOG:-${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log}"

# === Offload ===
OFFLOAD_TRAIN=${OFFLOAD_TRAIN:-1}
OFFLOAD_TRAIN_ASYNC=${OFFLOAD_TRAIN_ASYNC:-0}
OFFLOAD_ROLLOUT=${OFFLOAD_ROLLOUT:-1}

# === Model spec ===
MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-"${ORBIT_ROOT}/miles_plugins/model_args/kimi-k25.sh"}

# === Data ===
DATASET=${DATASET:-peft_arena_openr1_50k}
: "${TRAIN_JSONL:?set TRAIN_JSONL to a training jsonl path}"
TEST_JSONL=${TEST_JSONL:-}

# === Checkpoints ===
: "${HF_CKPT:?set HF_CKPT to a Hugging Face checkpoint path}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to a Megatron torch_dist checkpoint path}"
LOAD_CKPT=${LOAD_CKPT:-${MEGATRON_LOAD}}
SAVE_DIR=${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/Kimi-K2.5-W4A16_${DATASET}_oft}

# === Resources + parallelism ===
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-64}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-8}
SGLANG_EXPERT_PARALLEL_SIZE=${SGLANG_EXPERT_PARALLEL_SIZE:-1}
USE_ROLLOUT_ROUTING_REPLAY=${USE_ROLLOUT_ROUTING_REPLAY:-1}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-2}
CP_COMM_TYPE=${CP_COMM_TYPE:-a2a}

# === Training schedule ===
NUM_ROLLOUT=${NUM_ROLLOUT:-200}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-8}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-4096}
ROLLOUT_MAX_PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN:-1024}
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-5120}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-256}
EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-4096}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}
BALANCE_DATA=${BALANCE_DATA:-1}
USE_ROLLOUT_LOGPROBS=${USE_ROLLOUT_LOGPROBS:-0}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-512}
SAVE_INTERVAL=${SAVE_INTERVAL:-20}
EVAL_INTERVAL=${EVAL_INTERVAL:-20}
EVAL_GENERATE_MAX_CONCURRENCY=${EVAL_GENERATE_MAX_CONCURRENCY:-128}
DISABLE_EVAL=${DISABLE_EVAL:-0}
SKIP_EVAL_BEFORE_TRAIN=${SKIP_EVAL_BEFORE_TRAIN:-1}

export TORCH_COMPILE_DISABLE=${TORCH_COMPILE_DISABLE:-1}
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}

# === Rollout backend (SGLang) ===
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.70}
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-flashinfer}
SGLANG_MM_ATTENTION_BACKEND=${SGLANG_MM_ATTENTION_BACKEND:-triton_attn}
SGLANG_QUANTIZATION=${SGLANG_QUANTIZATION:-compressed-tensors}
export SGLANG_DISABLE_CUDNN_CHECK=${SGLANG_DISABLE_CUDNN_CHECK:-1}
export SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}
SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-${ROLLOUT_MAX_CONTEXT_LEN}}
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-128}
SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-4096}
SGLANG_MAX_TOTAL_TOKENS=${SGLANG_MAX_TOTAL_TOKENS:-1280000}
SGLANG_CUDA_GRAPH_MAX_BS=${SGLANG_CUDA_GRAPH_MAX_BS:-128}
SGLANG_DISABLE_CUDA_GRAPH=${SGLANG_DISABLE_CUDA_GRAPH:-0}
export SGLANG_OFT_EXPERT_PARITY_MODE=${SGLANG_OFT_EXPERT_PARITY_MODE:-0}
export SGLANG_OFT_MLA_ROTATE_KV_CACHE=${SGLANG_OFT_MLA_ROTATE_KV_CACHE:-0}
export SGLANG_OFT_MARLIN_MOE=${SGLANG_OFT_MARLIN_MOE:-1}
export SGLANG_FORCE_OFT_TRITON_MOE=${SGLANG_FORCE_OFT_TRITON_MOE:-0}
SGLANG_ENABLE_DETERMINISTIC_INFERENCE=${SGLANG_ENABLE_DETERMINISTIC_INFERENCE:-1}
SGLANG_NUM_CONTINUOUS_DECODE_STEPS=${SGLANG_NUM_CONTINUOUS_DECODE_STEPS:-4}
SGLANG_ENABLE_TORCH_COMPILE=${SGLANG_ENABLE_TORCH_COMPILE:-0}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-1}
ATTENTION_SOFTMAX_IN_FP32=${ATTENTION_SOFTMAX_IN_FP32:-0}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-flash}

# === PEFT (OFT) + INT4 quantization ===
TARGET_MODULES=${TARGET_MODULES:-linear_q_up_proj,linear_kv_up_proj,linear_proj,linear_fc1,linear_fc2}
OFT_BLOCK_SIZE=${OFT_BLOCK_SIZE:-32}
OFT_EPS=${OFT_EPS:-6e-5}
OPEN_TRAINING_INT4_FAKE_QAT_FLAG=${OPEN_TRAINING_INT4_FAKE_QAT_FLAG:-1}
OPEN_TRAINING_INT4_GROUP_SIZE=${OPEN_TRAINING_INT4_GROUP_SIZE:-32}

# === W&B ===
ENABLE_WANDB=${ENABLE_WANDB:-1}
WANDB_PROJECT=${WANDB_PROJECT:-orbit-release}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}

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
    --rm-type custom
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    --rollout-temperature 1.0
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
    --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}"
    --balance-data
    --use-rollout-routing-replay
    --custom-rm-path miles.orbit.rewards.peft_arena_reward.peft_arena_reward
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
if is_true "${USE_ROLLOUT_LOGPROBS}"; then
    RL_ARGS+=( --use-rollout-logprobs )
fi

LOSS_ARGS=(
    --calculate-per-token-loss
    --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}"
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project "${WANDB_PROJECT}"
    --wandb-group "${WANDB_GROUP}"
    --disable-wandb-random-suffix
)

PERF_ARGS=(
    --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
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

EVAL_ARGS=(
    --eval-interval "${EVAL_INTERVAL}"
    --eval-prompt-data math500 "${EVAL_ORBIT_DIR}/math500.jsonl" \
                       aime24  "${EVAL_ORBIT_DIR}/aime24.jsonl" \
                       amc23   "${EVAL_ORBIT_DIR}/amc23.jsonl"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
    --eval-top-k 1
    --skip-eval-before-train
    --eval-generate-max-concurrency "${EVAL_GENERATE_MAX_CONCURRENCY}"
    --eval-pass-k-values 1 2 4 8 16
    --eval-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
    --rollout-num-gpus 0
    --sglang-context-length "${SGLANG_CONTEXT_LENGTH}"
    --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
    --sglang-max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS}"
    --sglang-max-total-tokens "${SGLANG_MAX_TOTAL_TOKENS}"
    --sglang-quantization "${SGLANG_QUANTIZATION}"
    --sglang-expert-parallel-size "${SGLANG_EXPERT_PARALLEL_SIZE}"
    --sglang-mm-attention-backend "${SGLANG_MM_ATTENTION_BACKEND}"
    --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}"
    --router-disable-circuit-breaker
    --sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}"
    --sglang-num-continuous-decode-steps "${SGLANG_NUM_CONTINUOUS_DECODE_STEPS}"
)
if is_true "${SGLANG_ENABLE_DETERMINISTIC_INFERENCE}"; then
    SGLANG_ARGS+=( --sglang-enable-deterministic-inference )
fi
if is_true "${SGLANG_DISABLE_CUDA_GRAPH}"; then
    SGLANG_ARGS+=( --sglang-disable-cuda-graph )
fi

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend "${ATTENTION_BACKEND}"
    --accumulate-allreduce-grads-in-fp32
    --trust-remote-code
    --offload-train
    --no-offload-train-async
    --offload-rollout
    --use-rollout-routing-replay
)

DEBUG_ARGS=(
    --log-passrate
    --log-reward-category acc
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
