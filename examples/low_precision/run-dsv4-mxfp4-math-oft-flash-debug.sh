#!/usr/bin/env bash
# DeepSeek V4 Flash debug MXFP4 + OFT on the math dataset.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

source "${ORBIT_ROOT}/scripts/lib/common.sh"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/paths.sh"

# Debug-parity overrides: canonical LM head on, larger OFT block size.
export SGLANG_DSV4_CANONICAL_LM_HEAD="${SGLANG_DSV4_CANONICAL_LM_HEAD:-1}"
export OFT_BLOCK_SIZE="${OFT_BLOCK_SIZE:-128}"

# === Recipe identity ===
LAUNCHER_NAME=${LAUNCHER_NAME:-run_dsv4_flash_debug_mxfp4_math_oft}
PRECISION_PROFILE="mxfp4"
# Fast-load by default: orbit detects the dist-checkpoint at LOAD_CKPT via
# is_distributed_checkpoint and dispatches _load_checkpoint_dist instead of
# the slow load_weights_hf_to_megatron path. To revert to HF loading set
# LOAD_CKPT=$HF_CKPT REQUIRE_MEGATRON_LOAD=0.
REQUIRE_MEGATRON_LOAD=${REQUIRE_MEGATRON_LOAD:-1}
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${RUN_LOG:-${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log}"

# === Model spec ===
MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-"${ORBIT_ROOT}/orbit_plugins/model_args/deepseek-v4-flash-debug.sh"}
MODEL_ARGS_NUM_LAYERS=${MODEL_ARGS_NUM_LAYERS:-6}
MEGATRON_PATH=${MEGATRON_PATH:-${ORBIT_WORKSPACE_ROOT:-${HOME}/.cache/orbit/workspace}/software/proj/Megatron-LM}

# === Data + checkpoints ===
DATASET=${DATASET:-math}
: "${TRAIN_JSONL:?set TRAIN_JSONL to a training jsonl path}"
RM_TYPE=${RM_TYPE:-random}

# === Eval ===
EVAL_DATA_DIR=${EVAL_DATA_DIR:-}
EVAL_ORBIT_DIR=${EVAL_ORBIT_DIR:-${EVAL_DATA_DIR}}
FAST_EVAL_DIR=${FAST_EVAL_DIR:-${ORBIT_ROOT}/eval_subsets/peft_arena}
FAST_MATH500=${FAST_MATH500:-${FAST_EVAL_DIR}/math500_100_stride5.jsonl}
PYTHON_BIN=${PYTHON_BIN:-python}
DISABLE_EVAL=${DISABLE_EVAL:-0}
EVAL_DATASET_NAME=${EVAL_DATASET_NAME:-math500_100}
TEST_JSONL=${TEST_JSONL:-}

if ! is_true "${DISABLE_EVAL}"; then
    if [[ -z "${TEST_JSONL}" ]]; then
        if [[ -z "${EVAL_ORBIT_DIR}" ]]; then
            echo "set EVAL_ORBIT_DIR to a directory containing math500.jsonl or set DISABLE_EVAL=1" >&2
            exit 2
        fi
        if [ ! -f "${EVAL_ORBIT_DIR}/math500.jsonl" ]; then
            echo "[dsv4-debug-fast-eval] Converting math_eval test sets -> ${EVAL_ORBIT_DIR}" >&2
            "${PYTHON_BIN}" "${ORBIT_ROOT}/tools/convert_math_eval_to_orbit.py" \
                --output_dir "${EVAL_ORBIT_DIR}" --force
        fi

        mkdir -p "${FAST_EVAL_DIR}"
        if [ ! -f "${FAST_MATH500}" ]; then
            awk '((NR - 1) % 5 == 0) && c < 100 { print; c++ }' \
                "${EVAL_ORBIT_DIR}/math500.jsonl" > "${FAST_MATH500}"
        fi
        TEST_JSONL=${FAST_MATH500}
    fi
fi

EVAL_GENERATE_MAX_CONCURRENCY=${EVAL_GENERATE_MAX_CONCURRENCY:-64}

# === Checkpoints ===
# Generate these defaults with:
#   bash scripts/conversion/convert_dsv4_hf_to_megatron.sh \
#       ${ORBIT_CACHE_DIR:-${HOME}/.cache/orbit}/stage/DeepSeek-V4-Flash-debug \
#       ${ORBIT_CACHE_DIR:-${HOME}/.cache/orbit}/stage/megatron_checkpoint/DeepSeek-V4-Flash-debug-torchdist
# The converter first runs DeepSeek's official mp1/FP4 staging, patches the
# staged HF config and tokenizer_config.json for Orbit, then writes Megatron
# torch_dist. HF_CKPT stays the source of truth for tokenizer + config; LOAD_CKPT
# points at the pre-converted torch_dist so orbit startup skips HF -> Megatron
# weight loading.
DEFAULT_DSV4_DEBUG_HF_CKPT=${DEFAULT_DSV4_DEBUG_HF_CKPT:-${ORBIT_CACHE_DIR:-${HOME}/.cache/orbit}/stage/DeepSeek-V4-Flash-debug-inference-mp1}
DEFAULT_DSV4_DEBUG_MEGATRON_LOAD=${DEFAULT_DSV4_DEBUG_MEGATRON_LOAD:-${ORBIT_CACHE_DIR:-${HOME}/.cache/orbit}/stage/megatron_checkpoint/DeepSeek-V4-Flash-debug-torchdist}
: "${HF_CKPT:?set HF_CKPT to a Hugging Face checkpoint path}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to a Megatron torch_dist checkpoint path}"
LOAD_CKPT=${LOAD_CKPT:-${MEGATRON_LOAD}}
SAVE_DIR=${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/DeepSeek-V4-Flash-debug_${DATASET}_oft}

# === Local checkpoint staging (Lustre -> local scratch) ===
LOCAL_STAGE_ROOT=${LOCAL_STAGE_ROOT:-${ORBIT_CACHE_DIR:-${HOME}/.cache/orbit}/stage}
STAGE_HF_CKPT_TO=${STAGE_HF_CKPT_TO-}
STAGE_MEGATRON_CKPT_TO=${STAGE_MEGATRON_CKPT_TO-}
FORCE_STAGE_HF_CKPT=${FORCE_STAGE_HF_CKPT:-0}
FORCE_STAGE_MEGATRON_CKPT=${FORCE_STAGE_MEGATRON_CKPT:-0}

# === Resources + parallelism ===
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-64}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-${GPUS_PER_NODE}}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-1}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-${GPUS_PER_NODE}}
EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}

# === Training schedule ===
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-2}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-256}
EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-256}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-4096}
SAVE_INTERVAL=${SAVE_INTERVAL:-50}
EVAL_INTERVAL=${EVAL_INTERVAL:-10}
SKIP_EVAL_BEFORE_TRAIN=${SKIP_EVAL_BEFORE_TRAIN:-1}

# === Rollout backend (SGLang) ===
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.60}
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
SGLANG_SERVER_CONCURRENCY=${SGLANG_SERVER_CONCURRENCY:-4}
SGLANG_MOE_A2A_BACKEND=${SGLANG_MOE_A2A_BACKEND:-deepep}
SGLANG_DEEPEP_MODE=${SGLANG_DEEPEP_MODE:-normal}
SGLANG_EXPERT_PARALLEL_SIZE=${SGLANG_EXPERT_PARALLEL_SIZE:-${ROLLOUT_NUM_GPUS_PER_ENGINE}}
SGLANG_DATA_PARALLEL_SIZE=${SGLANG_DATA_PARALLEL_SIZE:-${ROLLOUT_NUM_GPUS_PER_ENGINE}}
SGLANG_ENABLE_DP_ATTENTION=${SGLANG_ENABLE_DP_ATTENTION:-1}
SGLANG_DISABLE_CUDA_GRAPH=${SGLANG_DISABLE_CUDA_GRAPH:-0}
SGLANG_CUDA_GRAPH_BS=${SGLANG_CUDA_GRAPH_BS:-"1 2 4 8 16"}
SGLANG_DISABLE_RADIX_CACHE=${SGLANG_DISABLE_RADIX_CACHE:-0}

# === Offload + precision controls ===
OFFLOAD_TRAIN=${OFFLOAD_TRAIN:-0}
OFFLOAD_ROLLOUT=${OFFLOAD_ROLLOUT:-1}
ACCUMULATE_ALLREDUCE_GRADS_IN_FP32=${ACCUMULATE_ALLREDUCE_GRADS_IN_FP32:-0}
MEGATRON_ENABLE_RECOMPUTE=${MEGATRON_ENABLE_RECOMPUTE:-0}
MEGATRON_ENABLE_CUDA_GRAPH=${MEGATRON_ENABLE_CUDA_GRAPH:-1}
if is_true "${MEGATRON_ENABLE_CUDA_GRAPH}"; then
    CUDA_GRAPH_IMPL=${CUDA_GRAPH_IMPL:-local}
    CUDA_GRAPH_SCOPE=${CUDA_GRAPH_SCOPE:-full_iteration}
    USE_TE_RNG_TRACKER=${USE_TE_RNG_TRACKER:-1}
    SKIP_NAN_CHECK_IN_LOSS_AND_GRAD=${SKIP_NAN_CHECK_IN_LOSS_AND_GRAD:-1}
else
    CUDA_GRAPH_IMPL=${CUDA_GRAPH_IMPL:-}
    CUDA_GRAPH_SCOPE=${CUDA_GRAPH_SCOPE:-}
    USE_TE_RNG_TRACKER=${USE_TE_RNG_TRACKER:-0}
    SKIP_NAN_CHECK_IN_LOSS_AND_GRAD=${SKIP_NAN_CHECK_IN_LOSS_AND_GRAD:-0}
fi
DSV4_MOE_DISPATCHER=${DSV4_MOE_DISPATCHER:-deepep}
MOE_PAD_EXPERTS_FOR_CUDA_GRAPH_INFERENCE=${MOE_PAD_EXPERTS_FOR_CUDA_GRAPH_INFERENCE:-${MEGATRON_ENABLE_CUDA_GRAPH}}

# === RL ===
USE_KL_LOSS=${USE_KL_LOSS:-1}
LR=${LR:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.0}

# === W&B ===
ENABLE_WANDB=${ENABLE_WANDB:-auto}
WANDB_PROJECT=${WANDB_PROJECT:-orbit-release}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}

source "${SCRIPT_DIR}/dsv4-common.sh"

TRAIN_ROWS=${TRAIN_ROWS:-$(wc -l < "${TRAIN_JSONL}")}
NUM_ROLLOUT=${NUM_ROLLOUT:-$(( (TRAIN_ROWS * TOTAL_EPOCHS + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE ))}

# === Model ARGS ===
source "${MODEL_ARGS_FILE}"   # provides MODEL_ARGS=(...)

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
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR}"
    --lr-decay-style constant
    --weight-decay "${WEIGHT_DECAY}"
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
if is_true "${USE_KL_LOSS}"; then
    RL_ARGS+=(--use-kl-loss)
fi

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
    --tensor-model-parallel-size "${TENSOR_MODEL_PARALLEL_SIZE}"
    --pipeline-model-parallel-size "${PIPELINE_MODEL_PARALLEL_SIZE}"
    --context-parallel-size "${CONTEXT_PARALLEL_SIZE}"
    --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}"
    --expert-tensor-parallel-size "${EXPERT_TENSOR_PARALLEL_SIZE}"
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
)
if is_true "${MEGATRON_ENABLE_RECOMPUTE}"; then
    PERF_ARGS+=(
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers 1
    )
fi
PERF_ARGS+=(--sequence-parallel)

EVAL_ARGS=()
if ! is_true "${DISABLE_EVAL}"; then
    EVAL_ARGS=(
        --eval-interval "${EVAL_INTERVAL}"
        --eval-prompt-data "${EVAL_DATASET_NAME}" "${TEST_JSONL}"
        --n-samples-per-eval-prompt 1
        --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
        --eval-top-k 1
    )
    if is_true "${SKIP_EVAL_BEFORE_TRAIN}"; then
        EVAL_ARGS+=(--skip-eval-before-train)
    fi
    EVAL_ARGS+=(
        --eval-generate-max-concurrency "${EVAL_GENERATE_MAX_CONCURRENCY}"
        --eval-pass-k-values 1 2 4 8 16
        --eval-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
    )
fi

# Build SGLang CUDA graph BS array from space-separated string
read -r -a SGLANG_CUDA_GRAPH_BS_ARRAY <<< "${SGLANG_CUDA_GRAPH_BS}"

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
    --rollout-num-gpus 0
    --sglang-context-length "${SGLANG_CONTEXT_LENGTH}"
    --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
    --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
    --sglang-data-parallel-size "${SGLANG_DATA_PARALLEL_SIZE}"
    --sglang-expert-parallel-size "${SGLANG_EXPERT_PARALLEL_SIZE}"
    --sglang-moe-a2a-backend "${SGLANG_MOE_A2A_BACKEND}"
    --router-disable-circuit-breaker
    --sglang-deepep-mode "${SGLANG_DEEPEP_MODE}"
)
if is_true "${SGLANG_ENABLE_DP_ATTENTION}"; then
    SGLANG_ARGS+=(--sglang-enable-dp-attention)
fi
if is_true "${SGLANG_DISABLE_RADIX_CACHE}"; then
    SGLANG_ARGS+=(--sglang-disable-radix-cache)
fi
if is_true "${SGLANG_DISABLE_CUDA_GRAPH}"; then
    SGLANG_ARGS+=(--sglang-disable-cuda-graph)
fi
SGLANG_ARGS+=(--sglang-cuda-graph-bs "${SGLANG_CUDA_GRAPH_BS_ARRAY[@]}")

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend flash
    --attention-softmax-in-fp32
)
if is_true "${OFFLOAD_TRAIN}"; then
    MISC_ARGS+=(--offload-train)
else
    MISC_ARGS+=(--no-offload-train)
fi
MISC_ARGS+=(--no-offload-train-async)
if is_true "${OFFLOAD_ROLLOUT}"; then
    MISC_ARGS+=(--offload-rollout)
else
    MISC_ARGS+=(--no-offload-rollout)
fi
if [[ -n "${CUDA_GRAPH_IMPL}" ]]; then
    MISC_ARGS+=(--cuda-graph-impl "${CUDA_GRAPH_IMPL}")
fi
if [[ -n "${CUDA_GRAPH_SCOPE}" ]]; then
    MISC_ARGS+=(--cuda-graph-scope "${CUDA_GRAPH_SCOPE}")
fi
if is_true "${USE_TE_RNG_TRACKER}"; then
    MISC_ARGS+=(--te-rng-tracker)
fi
if is_true "${SKIP_NAN_CHECK_IN_LOSS_AND_GRAD}"; then
    MISC_ARGS+=(--no-check-for-nan-in-loss-and-grad)
fi
if [[ -n "${DSV4_MOE_DISPATCHER}" ]]; then
    MISC_ARGS+=(--dsv4-moe-dispatcher "${DSV4_MOE_DISPATCHER}")
fi
if is_true "${ACCUMULATE_ALLREDUCE_GRADS_IN_FP32}"; then
    MISC_ARGS+=(--accumulate-allreduce-grads-in-fp32)
fi
MISC_ARGS+=(--moe-permute-fusion)
if is_true "${MOE_PAD_EXPERTS_FOR_CUDA_GRAPH_INFERENCE}"; then
    MISC_ARGS+=(--moe-pad-experts-for-cuda-graph-inference)
fi

DEBUG_ARGS=(
    --log-passrate
)

PEFT_ARGS=(
    --peft-method "${PEFT_METHOD}"
    --peft-variant "${PEFT_VARIANT}"
    --oft-type canonical_oft
    --oft-block-size "${OFT_BLOCK_SIZE}"
    --oft-eps "${OFT_EPS}"
    --target-modules "${TARGET_MODULES}"
)

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
