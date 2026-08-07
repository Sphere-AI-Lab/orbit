#!/usr/bin/env bash
# Kimi-K2.5 (full size) W4A16 INT4 + OFT on the math dataset. Self-contained launcher.
#
# Orbit-native port of verl/examples/run_kimi_k25_int4_math_megatron_oft.sh.
# See examples/README.md for a full env-knob reference.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"
source "${ORBIT_ROOT}/scripts/lib/paths.sh"

# === Recipe identity ===
LAUNCHER_NAME=${LAUNCHER_NAME:-run_kimi_k25_int4_math_megatron_oft}
PRECISION_PROFILE="int4"
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
SAVE_DIR=${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/Kimi-K2.5-W4A16_${DATASET}_oft}

# === Local checkpoint staging (Lustre → NVMe) ===
# By default, staged checkpoints live under
# ${ORBIT_CACHE_DIR:-${HOME}/.cache/orbit}/stage.
# Once staged, both checkpoints are read from local NVMe instead of Lustre/network FS,
# cutting cold init from ~25 min to ~7 min. Resolved paths on this cluster:
#   HF source (Lustre)       : ${ORBIT_DATA_ROOT:-${HOME}/.cache/orbit/data}/hf_models/Kimi-K2.5
#   HF stage (NVMe)          : ${ORBIT_CACHE_DIR:-${HOME}/.cache/orbit}/stage/Kimi-K2.5
#   Megatron source (network): ${ORBIT_DATA_ROOT:-${HOME}/.cache/orbit/data}/megatron_checkpoint/Kimi-K2.5
#   Megatron stage (NVMe)    : ${ORBIT_CACHE_DIR:-${HOME}/.cache/orbit}/stage/Megatron-Bridge/checkpoints/Kimi-K2.5
# Set FORCE_STAGE_HF_CKPT=1 / FORCE_STAGE_MEGATRON_CKPT=1 to re-rsync from the sources.
LOCAL_STAGE_ROOT=${LOCAL_STAGE_ROOT:-${ORBIT_CACHE_DIR:-${HOME}/.cache/orbit}/stage}
STAGE_HF_CKPT_TO=${STAGE_HF_CKPT_TO-${LOCAL_STAGE_ROOT}/Kimi-K2.5}
STAGE_MEGATRON_CKPT_TO=${STAGE_MEGATRON_CKPT_TO-${LOCAL_STAGE_ROOT}/Megatron-Bridge/checkpoints/Kimi-K2.5}
FORCE_STAGE_HF_CKPT=${FORCE_STAGE_HF_CKPT:-0}
FORCE_STAGE_MEGATRON_CKPT=${FORCE_STAGE_MEGATRON_CKPT:-0}

# === Resources + parallelism ===
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
RAY_NUM_CPUS=${RAY_NUM_CPUS:-64}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-8}
CONTEXT_PARALLEL_SIZE=${CONTEXT_PARALLEL_SIZE:-1}
CP_COMM_TYPE=${CP_COMM_TYPE:-}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-8}
SGLANG_EXPERT_PARALLEL_SIZE=${SGLANG_EXPERT_PARALLEL_SIZE:-1}
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
    USE_ROLLOUT_ROUTING_REPLAY=${USE_ROLLOUT_ROUTING_REPLAY:-1}
    DISABLE_SAVE=${DISABLE_SAVE:-1}
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
EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-1024}
EVAL_GENERATE_MAX_CONCURRENCY=${EVAL_GENERATE_MAX_CONCURRENCY:-128}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-8192}

# === Rollout backend (SGLang) ===
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.6}
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-flashinfer}
SGLANG_MM_ATTENTION_BACKEND=${SGLANG_MM_ATTENTION_BACKEND:-triton_attn}
SGLANG_QUANTIZATION=${SGLANG_QUANTIZATION:-compressed-tensors}
# Skip the cuDNN sanity check (slow + redundant for INT4).
export SGLANG_DISABLE_CUDNN_CHECK=${SGLANG_DISABLE_CUDNN_CHECK:-1}
# DeepGEMM JIT is broken under our CUDA combo; force off.
export SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}
SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-${ROLLOUT_MAX_CONTEXT_LEN}}
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-1024}
SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-4096}
# CUDA graphs work with this MoE/OFT layout when mem_fraction_static leaves
# headroom for graph state across offload/onload cycles. Validated 2026-05-09:
# 2.4x decode tok/s and successful 2-rollout smoke at mem_fraction_static=0.6.
SGLANG_DISABLE_CUDA_GRAPH=${SGLANG_DISABLE_CUDA_GRAPH:-0}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-1}
ATTENTION_SOFTMAX_IN_FP32=${ATTENTION_SOFTMAX_IN_FP32:-0}

# === PEFT (OFT) + INT4 quantization ===
# Keep the default OFT surface aligned with what SGLang can actually wrap for
# Kimi K2.5. The first MLA q/kv factors are fused together in SGLang serving
# and need dedicated per-branch OFT support before they can be enabled safely.
TARGET_MODULES=${TARGET_MODULES:-linear_q_up_proj,linear_kv_up_proj,linear_proj,linear_fc1,linear_fc2}
# If routed expert OFT is enabled, use the SGLang Marlin-OFT wrapper path. It
# applies streamed expert OFT R tensors before each Marlin MoE GEMM, keeping
# train-vs-rollout logprob meaningful without the slow Triton MoE fallback.
# Match both fused (linear_fc1/linear_fc2) and canonical_oft split naming
# (linear_fc1_gate/linear_fc1_up). Set SGLANG_FORCE_OFT_TRITON_MOE=1 only for
# fallback/debug.
export SGLANG_OFT_MLA_ROTATE_KV_CACHE=${SGLANG_OFT_MLA_ROTATE_KV_CACHE:-0}
case ",${TARGET_MODULES}," in
    *",linear_fc1,"*|*",linear_fc2,"*|*",linear_fc1_gate,"*|*",linear_fc1_up,"*)
        export SGLANG_OFT_MARLIN_MOE=${SGLANG_OFT_MARLIN_MOE:-1}
        export SGLANG_FORCE_OFT_TRITON_MOE=${SGLANG_FORCE_OFT_TRITON_MOE:-0}
        ;;
esac
OFT_BLOCK_SIZE=${OFT_BLOCK_SIZE:-32}
OFT_EPS=${OFT_EPS:-6e-5}
OPEN_TRAINING_INT4_FAKE_QAT_FLAG=${OPEN_TRAINING_INT4_FAKE_QAT_FLAG:-1}
OPEN_TRAINING_INT4_GROUP_SIZE=${OPEN_TRAINING_INT4_GROUP_SIZE:-32}

# === Parity gates ===
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
    --sglang-expert-parallel-size "${SGLANG_EXPERT_PARALLEL_SIZE}"
    --sglang-mm-attention-backend "${SGLANG_MM_ATTENTION_BACKEND}"
    --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}"
    --router-disable-circuit-breaker
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
