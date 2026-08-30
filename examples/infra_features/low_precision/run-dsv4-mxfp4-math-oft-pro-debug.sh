#!/usr/bin/env bash
# DeepSeek V4 Pro debug MXFP4 + OFT on the math dataset.
# Reconstructed body (no prior body existed); validated by bash -n + dry-run argv; needs a real GPU run to confirm recipe correctness.
#
# Env-knob order and arg-array body mirror
# run-dsv4-mxfp4-openr1-oft-pro.sh; the values are the
# Pro-debug recipe's own (single-rank torch_dist -> 1 GPU / EP1 debug layout).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

source "${ORBIT_ROOT}/scripts/lib/common.sh"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/paths.sh"

# === Recipe identity ===
LAUNCHER_NAME=${LAUNCHER_NAME:-run_dsv4_pro_debug_mxfp4_math_oft}
PRECISION_PROFILE="mxfp4"
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${RUN_LOG:-${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log}"

NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
GPUS_PER_NODE=${GPUS_PER_NODE:-1}

# === Dataset & Eval ===
DATASET=${DATASET:-math}
RM_TYPE=${RM_TYPE:-random}
: "${TRAIN_JSONL:?set TRAIN_JSONL to a training jsonl path}"

# Fast-eval subset: convert the math_eval sets, then take a 100-row stride-5
# math500 subset for in-loop eval.
EVAL_DATA_DIR=${EVAL_DATA_DIR:-}
EVAL_ORBIT_DIR=${EVAL_ORBIT_DIR:-${EVAL_DATA_DIR}}
FAST_EVAL_DIR=${FAST_EVAL_DIR:-${ORBIT_ROOT}/eval_subsets/peft_arena}
FAST_MATH500=${FAST_MATH500:-${FAST_EVAL_DIR}/math500_100_stride5.jsonl}
PYTHON_BIN=${PYTHON_BIN:-python}
EVAL_DATASET_NAME=${EVAL_DATASET_NAME:-math500_100}
DISABLE_EVAL=${DISABLE_EVAL:-1}
TEST_JSONL=${TEST_JSONL:-}
EVAL_DATASETS=${EVAL_DATASETS:-}
if ! is_true "${DISABLE_EVAL}" && [[ -z "${EVAL_DATASETS}" ]]; then
    if [[ -z "${TEST_JSONL}" ]]; then
        if [[ -z "${EVAL_ORBIT_DIR}" ]]; then
            echo "set EVAL_ORBIT_DIR to a directory containing math500.jsonl or set DISABLE_EVAL=1" >&2
            exit 2
        fi
        if [ ! -f "${EVAL_ORBIT_DIR}/math500.jsonl" ]; then
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
    EVAL_DATASETS=${EVAL_DATASET_NAME}:${TEST_JSONL}
fi
EVAL_INTERVAL=${EVAL_INTERVAL:-10}
EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-256}
EVAL_GENERATE_MAX_CONCURRENCY=${EVAL_GENERATE_MAX_CONCURRENCY:-64}
SKIP_EVAL_BEFORE_TRAIN=${SKIP_EVAL_BEFORE_TRAIN:-1}
DISABLE_SAVE=${DISABLE_SAVE:-0}

# === Pro model ===
DEFAULT_DSV4_PRO_DEBUG_HF_CKPT=${DEFAULT_DSV4_PRO_DEBUG_HF_CKPT:-${ORBIT_CACHE_DIR:-${HOME}/.cache/orbit}/stage/DeepSeek-V4-Pro-debug-inference-mp1}
DEFAULT_DSV4_PRO_DEBUG_MEGATRON_LOAD=${DEFAULT_DSV4_PRO_DEBUG_MEGATRON_LOAD:-${ORBIT_CACHE_DIR:-${HOME}/.cache/orbit}/stage/megatron_checkpoint/DeepSeek-V4-Pro-debug-torchdist}
MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-"${ORBIT_ROOT}/miles_plugins/model_args/deepseek-v4-pro.sh"}
MODEL_ARGS_NUM_LAYERS=${MODEL_ARGS_NUM_LAYERS:-6}
: "${HF_CKPT:?set HF_CKPT to a Hugging Face checkpoint path}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to a Megatron torch_dist checkpoint path}"
LOAD_CKPT=${LOAD_CKPT:-${MEGATRON_LOAD}}
REQUIRE_MEGATRON_LOAD=${REQUIRE_MEGATRON_LOAD:-1}
SAVE_DIR=${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/DeepSeek-V4-Pro-debug_${DATASET}_oft}
STAGE_HF_CKPT_TO=${STAGE_HF_CKPT_TO-}
STAGE_MEGATRON_CKPT_TO=${STAGE_MEGATRON_CKPT_TO-}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-1}
DSV4_CHAT_ENCODING=${DSV4_CHAT_ENCODING:-official_chat}
DSV4_ENCODING_PATH=${DSV4_ENCODING_PATH:-${ORBIT_DATA_ROOT:-${HOME}/.cache/orbit/data}/hf_models/DeepSeek-V4-Pro/encoding/encoding_dsv4.py}

LOCAL_STAGE_ROOT=${LOCAL_STAGE_ROOT:-${ORBIT_CACHE_DIR:-${HOME}/.cache/orbit}/stage}
FORCE_STAGE_HF_CKPT=${FORCE_STAGE_HF_CKPT:-0}
FORCE_STAGE_MEGATRON_CKPT=${FORCE_STAGE_MEGATRON_CKPT:-0}

# === 1 GPU sharding: train TP=1 / PP=1 / CP=1 / EP=1 / ETP=1 ===
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-${GPUS_PER_NODE}}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}
PIPELINE_MODEL_PARALLEL_SIZE=${PIPELINE_MODEL_PARALLEL_SIZE:-1}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
EXPERT_TENSOR_PARALLEL_SIZE=${EXPERT_TENSOR_PARALLEL_SIZE:-1}
SEQUENCE_PARALLEL="${SEQUENCE_PARALLEL:-0}"
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-4096}

MEGATRON_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${MEGATRON_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-8192}
MEGATRON_DSV4_RECOMPUTE_ATTN_ROUND="${MEGATRON_DSV4_RECOMPUTE_ATTN_ROUND:-1}"

# === SGLang rollout ===
SGLANG_DATA_PARALLEL_SIZE=${SGLANG_DATA_PARALLEL_SIZE:-${ROLLOUT_NUM_GPUS_PER_ENGINE}}
SGLANG_EXPERT_PARALLEL_SIZE=${SGLANG_EXPERT_PARALLEL_SIZE:-${ROLLOUT_NUM_GPUS_PER_ENGINE}}
SGLANG_ENABLE_DP_ATTENTION=${SGLANG_ENABLE_DP_ATTENTION:-1}
SGLANG_MOE_DP_SIZE=${SGLANG_MOE_DP_SIZE:-1}
SGLANG_MOE_A2A_BACKEND=${SGLANG_MOE_A2A_BACKEND:-none}
SGLANG_DEEPEP_MODE=${SGLANG_DEEPEP_MODE:-normal}
SGLANG_DISABLE_RADIX_CACHE=${SGLANG_DISABLE_RADIX_CACHE:-0}
SGLANG_SYMM_MEM_PREALLOC_GB_SIZE="${SGLANG_SYMM_MEM_PREALLOC_GB_SIZE:-0}"
SGLANG_DISABLE_CUDA_GRAPH=${SGLANG_DISABLE_CUDA_GRAPH:-0}
SGLANG_ENFORCE_EAGER="${SGLANG_ENFORCE_EAGER:-0}"
SGLANG_ENABLE_PIECEWISE_CUDA_GRAPH=${SGLANG_ENABLE_PIECEWISE_CUDA_GRAPH:-0}
SGLANG_ENFORCE_PIECEWISE_CUDA_GRAPH=${SGLANG_ENFORCE_PIECEWISE_CUDA_GRAPH:-0}
SGLANG_CUDA_GRAPH_BS=${SGLANG_CUDA_GRAPH_BS:-"1 2 4 8 16"}
SGLANG_CUDA_GRAPH_MAX_BS=${SGLANG_CUDA_GRAPH_MAX_BS:-16}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.60}
SGLANG_MAX_TOTAL_TOKENS=${SGLANG_MAX_TOTAL_TOKENS:-270000}
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-16}
SGLANG_NUM_CONTINUOUS_DECODE_STEPS=${SGLANG_NUM_CONTINUOUS_DECODE_STEPS:-2}
SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-4096}
SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK="${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-4}"

# === Megatron full-CG + DeepEP ===
MEGATRON_ENABLE_RECOMPUTE=${MEGATRON_ENABLE_RECOMPUTE:-1}
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
MEGATRON_DEEPEP_ALLOW_HYBRID_MODE=${MEGATRON_DEEPEP_ALLOW_HYBRID_MODE:-0}
ACCUMULATE_ALLREDUCE_GRADS_IN_FP32=${ACCUMULATE_ALLREDUCE_GRADS_IN_FP32:-0}

# === Orbit ===
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-2}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-256}
ROLLOUT_MAX_PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN:-1024}
LR=${LR:-1e-5}
NUM_ROLLOUT=${NUM_ROLLOUT:-10}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000000}

USE_ROLLOUT_ROUTING_REPLAY=${USE_ROLLOUT_ROUTING_REPLAY:-0}
USE_KL_LOSS=${USE_KL_LOSS:-1}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.001}
USE_ROLLOUT_LOGPROBS=${USE_ROLLOUT_LOGPROBS:-0}
LOG_PROBS_CHUNK_SIZE=${LOG_PROBS_CHUNK_SIZE:-}

# === Offload ===
ORBIT_PEFT_OFFLOAD_PIN=${ORBIT_PEFT_OFFLOAD_PIN:-0}
OFFLOAD_TRAIN=${OFFLOAD_TRAIN:-1}
OFFLOAD_TRAIN_ASYNC=${OFFLOAD_TRAIN_ASYNC:-0}
OFFLOAD_TRAIN_GRAD_BUFFERS=${OFFLOAD_TRAIN_GRAD_BUFFERS:-1}
OFFLOAD_TRAIN_OPTIMIZER=${OFFLOAD_TRAIN_OPTIMIZER:-1}
USE_PRECISION_AWARE_OPTIMIZER=${USE_PRECISION_AWARE_OPTIMIZER:-1}
OFFLOAD_TRAIN_ADAPTER=${OFFLOAD_TRAIN_ADAPTER:-1}
OFFLOAD_TRAIN_FROZEN_BASE_MODE=${OFFLOAD_TRAIN_FROZEN_BASE_MODE:-flat}
OFFLOAD_ROLLOUT=${OFFLOAD_ROLLOUT:-1}
OFFLOAD_ROLLOUT_ADAPTER=${OFFLOAD_ROLLOUT_ADAPTER:-1}
OFFLOAD_ROLLOUT_CUDA_GRAPH=${OFFLOAD_ROLLOUT_CUDA_GRAPH:-1}
SGLANG_MEMORY_SAVER_CUDA_GRAPH=${SGLANG_MEMORY_SAVER_CUDA_GRAPH:-true}
OFT_BLOCK_SIZE="${OFT_BLOCK_SIZE:-64}"

# === Kernels ===
export SGLANG_DSV4_CANONICAL_LM_HEAD="${SGLANG_DSV4_CANONICAL_LM_HEAD:-1}"

# === W&B ===
ENABLE_WANDB=${ENABLE_WANDB:-auto}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
WANDB_PROJECT=${WANDB_PROJECT:-orbit-release}

source "${SCRIPT_DIR}/dsv4-common.sh"
source "${MODEL_ARGS_FILE}"   # provides MODEL_ARGS=(...)

COLOCATE_ARGS=( --colocate )

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${LOAD_CKPT}"
    --megatron-to-hf-mode bridge
)
if ! is_true "${DISABLE_SAVE}"; then
    CKPT_ARGS+=(
        --save "${SAVE_DIR}"
        --save-interval "${SAVE_INTERVAL}"
        --no-save-optim
        --no-save-rng
    )
fi

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
case "${DSV4_CHAT_ENCODING}" in
    legacy_jinja)
        ;;
    official_chat)
        ROLLOUT_ARGS+=( --apply-chat-template-kwargs '{"thinking_mode":"chat"}' )
        ;;
    official_thinking)
        ROLLOUT_ARGS+=( --apply-chat-template-kwargs '{"thinking_mode":"thinking"}' )
        ;;
    *)
        echo "Invalid DSV4_CHAT_ENCODING=${DSV4_CHAT_ENCODING}; expected legacy_jinja, official_chat, or official_thinking" >&2
        exit 2
        ;;
esac
if [[ -n "${ROLLOUT_MAX_PROMPT_LEN}" ]]; then
    ROLLOUT_ARGS+=( --rollout-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}" )
fi
if [[ -n "${ROLLOUT_MAX_CONTEXT_LEN}" ]]; then
    ROLLOUT_ARGS+=( --rollout-max-context-len "${ROLLOUT_MAX_CONTEXT_LEN}" )
fi

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR}"
    --lr-decay-style constant
    --weight-decay 0.0
    --adam-beta1 0.9
    --adam-beta2 0.999
)
if is_true "${USE_PRECISION_AWARE_OPTIMIZER}"; then
    OPTIMIZER_ARGS+=( --use-precision-aware-optimizer )
fi

RL_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef "${KL_LOSS_COEF}"
    --kl-loss-type low_var_kl
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.2
)
if is_true "${USE_KL_LOSS}"; then
    RL_ARGS+=( --use-kl-loss )
fi
if is_true "${USE_ROLLOUT_LOGPROBS}"; then
    RL_ARGS+=( --use-rollout-logprobs )
fi

LOSS_ARGS=(
    --calculate-per-token-loss
)
if [[ -n "${LOG_PROBS_CHUNK_SIZE}" ]]; then
    LOSS_ARGS+=( --log-probs-chunk-size "${LOG_PROBS_CHUNK_SIZE}" )
fi

WANDB_ARGS=()
if ! is_false "${ENABLE_WANDB}"; then
    WANDB_ARGS=(
        --use-wandb
        --wandb-project "${WANDB_PROJECT}"
        --wandb-group "${WANDB_GROUP}"
        --disable-wandb-random-suffix
    )
fi

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
if is_true "${SEQUENCE_PARALLEL}"; then
    PERF_ARGS+=( --sequence-parallel )
fi

EVAL_ARGS=()
if ! is_true "${DISABLE_EVAL}"; then
    EVAL_PROMPT_DATA_ARGS=()
    for eval_dataset in ${EVAL_DATASETS}; do
        EVAL_PROMPT_DATA_ARGS+=( "${eval_dataset%%:*}" "${eval_dataset#*:}" )
    done
    EVAL_ARGS=(
        --eval-interval "${EVAL_INTERVAL}"
        --eval-prompt-data "${EVAL_PROMPT_DATA_ARGS[@]}"
        --n-samples-per-eval-prompt 1
        --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
        --eval-top-k 1
        --eval-generate-max-concurrency "${EVAL_GENERATE_MAX_CONCURRENCY}"
        --eval-pass-k-values 1 2 4 8 16
        --eval-max-prompt-len "${ROLLOUT_MAX_PROMPT_LEN}"
    )
    if is_true "${SKIP_EVAL_BEFORE_TRAIN}"; then
        EVAL_ARGS+=( --skip-eval-before-train )
    fi
fi

read -r -a SGLANG_CUDA_GRAPH_BS_ARRAY <<< "${SGLANG_CUDA_GRAPH_BS}"
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
    --rollout-num-gpus 0
    --sglang-context-length "${SGLANG_CONTEXT_LENGTH}"
    --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
    --sglang-max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS}"
    --sglang-max-total-tokens "${SGLANG_MAX_TOTAL_TOKENS}"
    --sglang-data-parallel-size "${SGLANG_DATA_PARALLEL_SIZE}"
    --sglang-expert-parallel-size "${SGLANG_EXPERT_PARALLEL_SIZE}"
    --sglang-moe-data-parallel-size "${SGLANG_MOE_DP_SIZE}"
    --sglang-moe-a2a-backend "${SGLANG_MOE_A2A_BACKEND}"
    --router-disable-circuit-breaker
    --sglang-deepep-mode "${SGLANG_DEEPEP_MODE}"
    --sglang-cuda-graph-bs "${SGLANG_CUDA_GRAPH_BS_ARRAY[@]}"
    --sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}"
    --sglang-num-continuous-decode-steps "${SGLANG_NUM_CONTINUOUS_DECODE_STEPS}"
)
if is_true "${SGLANG_ENABLE_DP_ATTENTION}"; then
    SGLANG_ARGS+=( --sglang-enable-dp-attention )
fi
if is_true "${SGLANG_DISABLE_RADIX_CACHE}"; then
    SGLANG_ARGS+=( --sglang-disable-radix-cache )
fi
if is_true "${SGLANG_DISABLE_CUDA_GRAPH}"; then
    SGLANG_ARGS+=( --sglang-disable-cuda-graph )
fi
if is_true "${SGLANG_ENABLE_PIECEWISE_CUDA_GRAPH}" || is_true "${SGLANG_ENFORCE_PIECEWISE_CUDA_GRAPH}"; then
    SGLANG_ARGS+=( --sglang-enable-piecewise-cuda-graph )
fi
if is_true "${SGLANG_ENFORCE_PIECEWISE_CUDA_GRAPH}"; then
    SGLANG_ARGS+=( --sglang-enforce-piecewise-cuda-graph )
fi

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend flash
    --attention-softmax-in-fp32
)
if is_true "${TRUST_REMOTE_CODE}"; then
    MISC_ARGS+=( --trust-remote-code )
fi
if is_true "${OFFLOAD_TRAIN}"; then
    MISC_ARGS+=( --offload-train )
else
    MISC_ARGS+=( --no-offload-train )
fi
if is_true "${OFFLOAD_TRAIN_GRAD_BUFFERS}"; then
    MISC_ARGS+=( --offload-train-grad-buffers )
else
    MISC_ARGS+=( --no-offload-train-grad-buffers )
fi
if is_true "${OFFLOAD_TRAIN_OPTIMIZER}"; then
    MISC_ARGS+=( --offload-train-optimizer )
else
    MISC_ARGS+=( --no-offload-train-optimizer )
fi
if is_true "${OFFLOAD_TRAIN_ADAPTER}"; then
    MISC_ARGS+=( --offload-train-adapter )
else
    MISC_ARGS+=( --no-offload-train-adapter )
fi
if is_true "${OFFLOAD_TRAIN_ASYNC}"; then
    MISC_ARGS+=( --offload-train-async )
else
    MISC_ARGS+=( --no-offload-train-async )
fi
MISC_ARGS+=( --offload-train-frozen-base-mode "${OFFLOAD_TRAIN_FROZEN_BASE_MODE}" )
if is_true "${OFFLOAD_ROLLOUT}"; then
    MISC_ARGS+=( --offload-rollout )
else
    MISC_ARGS+=( --no-offload-rollout )
fi
if is_true "${OFFLOAD_ROLLOUT_ADAPTER}"; then
    MISC_ARGS+=( --offload-rollout-adapter )
else
    MISC_ARGS+=( --no-offload-rollout-adapter )
fi
if is_true "${USE_ROLLOUT_ROUTING_REPLAY}"; then
    MISC_ARGS+=( --use-rollout-routing-replay )
fi
if is_true "${ACCUMULATE_ALLREDUCE_GRADS_IN_FP32}"; then
    MISC_ARGS+=( --accumulate-allreduce-grads-in-fp32 )
fi
if [[ -n "${CUDA_GRAPH_IMPL}" ]]; then
    MISC_ARGS+=( --cuda-graph-impl "${CUDA_GRAPH_IMPL}" )
fi
if [[ -n "${CUDA_GRAPH_SCOPE}" ]]; then
    MISC_ARGS+=( --cuda-graph-scope "${CUDA_GRAPH_SCOPE}" )
fi
if is_true "${USE_TE_RNG_TRACKER}"; then
    MISC_ARGS+=( --te-rng-tracker )
fi
if is_true "${SKIP_NAN_CHECK_IN_LOSS_AND_GRAD}"; then
    MISC_ARGS+=( --no-check-for-nan-in-loss-and-grad )
fi
if [[ -n "${DSV4_MOE_DISPATCHER}" ]]; then
    MISC_ARGS+=( --dsv4-moe-dispatcher "${DSV4_MOE_DISPATCHER}" )
    if [[ "${DSV4_MOE_DISPATCHER}" == "deepep" ]]; then
        MISC_ARGS+=( --moe-permute-fusion )
    fi
fi
if is_true "${MOE_PAD_EXPERTS_FOR_CUDA_GRAPH_INFERENCE}"; then
    MISC_ARGS+=( --moe-pad-experts-for-cuda-graph-inference )
fi

DEBUG_ARGS=(
    --log-passrate
)
# Note: --log-reward-category is peft_arena-reward-specific (expects a dict
# reward); omitted here because rm-type=random returns a scalar int reward.

PEFT_ARGS=(
    --peft-method "${PEFT_METHOD}"
    --peft-variant "${PEFT_VARIANT}"
    --oft-type canonical_oft
    --oft-block-size "${OFT_BLOCK_SIZE}"
    --oft-eps "${OFT_EPS}"
    --target-modules "${TARGET_MODULES}"
)

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
