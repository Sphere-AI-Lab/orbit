#!/usr/bin/env bash
# Kimi-K2.5 NVFP4 (modelopt_fp4) + OFT on the math dataset. Self-contained launcher.
# NVFP4 sibling of run-kimi-k25-int4-math-oft.sh — actor/ref load the converted
# Megatron NVFP4 distributed checkpoint, rollout uses the modelopt FP4 HF
# checkpoint via SGLang (modelopt_fp4).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"
source "${ORBIT_ROOT}/scripts/lib/paths.sh"

# === Recipe identity ===
LAUNCHER_NAME=${LAUNCHER_NAME:-run_kimi_k25_nvfp4_math_megatron_oft}
PRECISION_PROFILE="nvfp4"
REQUIRE_MEGATRON_LOAD=${REQUIRE_MEGATRON_LOAD:-1}
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${RUN_LOG:-${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log}"

# === Offload ===
# Free GPU for SGLang during rollout by paging Megatron weights/optimizer to host.
OFFLOAD_TRAIN=${OFFLOAD_TRAIN:-1}
OFFLOAD_TRAIN_ASYNC=${OFFLOAD_TRAIN_ASYNC:-1}
OFFLOAD_ROLLOUT=${OFFLOAD_ROLLOUT:-1}

# === Model spec ===
MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-"${ORBIT_ROOT}/orbit_plugins/model_args/kimi-k25.sh"}

# === Data ===
DATASET=${DATASET:-math}
: "${TRAIN_JSONL:?set TRAIN_JSONL to a training jsonl path}"
TEST_JSONL=${TEST_JSONL:-}

# === Checkpoints ===
: "${HF_CKPT:?set HF_CKPT to a Hugging Face checkpoint path}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to a Megatron torch_dist checkpoint path}"
LOAD_CKPT=${LOAD_CKPT:-${MEGATRON_LOAD}}
SAVE_DIR=${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/Kimi-K2.5-NVFP4_${DATASET}_oft}

# === Local checkpoint staging (Lustre -> NVMe) ===
LOCAL_STAGE_ROOT=${LOCAL_STAGE_ROOT:-${ORBIT_CACHE_DIR:-${HOME}/.cache/orbit}/stage}
STAGE_HF_CKPT_TO=${STAGE_HF_CKPT_TO-${LOCAL_STAGE_ROOT}/Kimi-K2.5-NVFP4}
STAGE_MEGATRON_CKPT_TO=${STAGE_MEGATRON_CKPT_TO-${LOCAL_STAGE_ROOT}/Megatron-Bridge/checkpoints/Kimi-K2.5-NVFP4}
FORCE_STAGE_HF_CKPT=${FORCE_STAGE_HF_CKPT:-0}
FORCE_STAGE_MEGATRON_CKPT=${FORCE_STAGE_MEGATRON_CKPT:-0}

# === Resources + parallelism ===
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-8}
# Keep rollout topology matched to the Kimi INT4 math/OFT baseline. The
# precision-specific differences are the NVFP4 checkpoint and modelopt_fp4
# serving path, not DP/EP attention layout.
SGLANG_EXPERT_PARALLEL_SIZE=${SGLANG_EXPERT_PARALLEL_SIZE:-1}
if [[ -n "${SGLANG_DP_SIZE:-}" && -z "${SGLANG_DATA_PARALLEL_SIZE:-}" ]]; then
    SGLANG_DATA_PARALLEL_SIZE="${SGLANG_DP_SIZE}"
fi
SGLANG_DATA_PARALLEL_SIZE=${SGLANG_DATA_PARALLEL_SIZE:-1}
SGLANG_ENABLE_DP_ATTENTION=${SGLANG_ENABLE_DP_ATTENTION:-0}
SGLANG_ENABLE_DP_LM_HEAD=${SGLANG_ENABLE_DP_LM_HEAD:-0}
# Single-adapter fast path: matches the eval's MAX_OFTS_PER_BATCH=2.
SGLANG_MAX_OFTS_PER_BATCH=${SGLANG_MAX_OFTS_PER_BATCH:-2}
USE_ROLLOUT_ROUTING_REPLAY=${USE_ROLLOUT_ROUTING_REPLAY:-1}

# === Parity smoke mode ===
PARITY_CHECK=${PARITY_CHECK:-0}
PARITY_CHECK_FAST=${PARITY_CHECK_FAST:-${PARITY_CHECK}}
if [[ "${PARITY_CHECK,,}" =~ ^(1|true|yes|y|on)$ && "${PARITY_CHECK_FAST,,}" =~ ^(1|true|yes|y|on)$ ]]; then
    ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
    N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-1}
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
    ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-1024}
    EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-1024}
fi
if [[ "${PARITY_CHECK,,}" =~ ^(1|true|yes|y|on)$ ]]; then
    DISABLE_EVAL=${DISABLE_EVAL:-1}
    TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
    NUM_ROLLOUT=${NUM_ROLLOUT:-1}
    ENABLE_WANDB=${ENABLE_WANDB:-0}
    DISABLE_SAVE=${DISABLE_SAVE:-1}
    SGLANG_OFT_PARITY_MODE=${SGLANG_OFT_PARITY_MODE:-1}
    SGLANG_ENABLE_FP32_LM_HEAD=${SGLANG_ENABLE_FP32_LM_HEAD:-1}
    SGLANG_ENABLE_DETERMINISTIC_INFERENCE=${SGLANG_ENABLE_DETERMINISTIC_INFERENCE:-1}
    # Reuse the generic train-vs-rollout logprob diff logger.
fi

# === Training schedule ===
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-32}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-1}
ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-1024}
ROLLOUT_MAX_PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN:-1024}
ROLLOUT_MAX_CONTEXT_LEN=${ROLLOUT_MAX_CONTEXT_LEN:-$((ROLLOUT_MAX_PROMPT_LEN + ROLLOUT_MAX_RESPONSE_LEN))}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-32}
DISABLE_EVAL=${DISABLE_EVAL:-1}
EVAL_ONLY_BEFORE_TRAIN=${EVAL_ONLY_BEFORE_TRAIN:-1}
EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-1024}
EVAL_GENERATE_MAX_CONCURRENCY=${EVAL_GENERATE_MAX_CONCURRENCY:-128}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}

# === Rollout backend (SGLang) ===
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.6}
# Kimi-K2.5 uses SGLang's DeepSeek/MLA path. On B200, trtllm_mla selects the
# Blackwell MLA attention path and FP8 KV cache; forcing flashinfer here leaves
# NVFP4 decode on a much slower path.
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-trtllm_mla}
SGLANG_MM_ATTENTION_BACKEND=${SGLANG_MM_ATTENTION_BACKEND:-triton_attn}
# NVFP4-specific quantization. Keep the MoE runner on SGLang auto, matching
# the INT4 launcher, so the serving backend can choose the precision-specific
# kernel without changing the launcher topology.
SGLANG_QUANTIZATION=${SGLANG_QUANTIZATION:-modelopt_fp4}
SGLANG_MOE_RUNNER_BACKEND=${SGLANG_MOE_RUNNER_BACKEND:-auto}
export SGLANG_DISABLE_CUDNN_CHECK=${SGLANG_DISABLE_CUDNN_CHECK:-1}
export SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}
# FlashInfer TRT-LLM FP4 MoE autotune can hit cudaErrorIllegalAddress on the
# Kimi-K2.5 NVFP4 SM100 shape. Keep the backend on auto/TRT-LLM, but skip the
# unsafe autotune path by default until that FlashInfer kernel path is fixed.
export SGLANG_DISABLE_FLASHINFER_AUTOTUNE=${SGLANG_DISABLE_FLASHINFER_AUTOTUNE:-1}
SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-${ROLLOUT_MAX_CONTEXT_LEN}}
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-1024}
SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-4096}
SGLANG_DISABLE_CUDA_GRAPH=${SGLANG_DISABLE_CUDA_GRAPH:-0}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-1}
ATTENTION_SOFTMAX_IN_FP32=${ATTENTION_SOFTMAX_IN_FP32:-0}

# === PEFT (OFT) ===
# Naming API follow-up: audit whether this can become all-linear
TARGET_MODULES=${TARGET_MODULES:-linear_q_down_proj,linear_q_up_proj,linear_kv_down_proj,linear_kv_up_proj,linear_proj,linear_fc1,linear_fc2}
OFT_BLOCK_SIZE=${OFT_BLOCK_SIZE:-32}
OFT_EPS=${OFT_EPS:-6e-5}
MAX_LOGPROB_ABS_DIFF=${MAX_LOGPROB_ABS_DIFF:-0.10}
MIN_TOPK_AGREEMENT=${MIN_TOPK_AGREEMENT:-0.75}

# === RL ===
USE_KL_LOSS=${USE_KL_LOSS:-0}

# === W&B ===
ENABLE_WANDB=${ENABLE_WANDB:-auto}
WANDB_PROJECT=${WANDB_PROJECT:-orbit-release}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}

# === Checkpoint save interval ===
SAVE_INTERVAL=${SAVE_INTERVAL:-200}

TRAIN_ROWS=${TRAIN_ROWS:-$(wc -l < "${TRAIN_JSONL}")}
NUM_ROLLOUT=${NUM_ROLLOUT:-$(( (TRAIN_ROWS * TOTAL_EPOCHS + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE ))}

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
    --rm-type math
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
    --lr 3e-6
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
    --expert-model-parallel-size "${EXPERT_MODEL_PARALLEL_SIZE}"
    --expert-tensor-parallel-size 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU}"
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --sequence-parallel
)

EVAL_ARGS=( )

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
    --rollout-num-gpus 0
    --sglang-context-length "${SGLANG_CONTEXT_LENGTH}"
    --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
    --sglang-max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS}"
    --sglang-quantization "${SGLANG_QUANTIZATION}"
    --sglang-data-parallel-size "${SGLANG_DATA_PARALLEL_SIZE}"
    --sglang-expert-parallel-size "${SGLANG_EXPERT_PARALLEL_SIZE}"
    --sglang-mm-attention-backend "${SGLANG_MM_ATTENTION_BACKEND}"
    --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}"
    --sglang-moe-runner-backend "${SGLANG_MOE_RUNNER_BACKEND}"
    --router-disable-circuit-breaker
    --sglang-disable-flashinfer-autotune
)

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
