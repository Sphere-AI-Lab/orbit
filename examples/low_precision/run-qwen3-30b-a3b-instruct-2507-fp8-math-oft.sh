#!/usr/bin/env bash
# Qwen3-30B-A3B-Instruct-2507 (MoE) FP8 + OFT on the math dataset. Self-contained launcher.
# Rollout uses the FP8 HF checkpoint; actor/ref load the direct-write Megatron checkpoint.
# See examples/README.md for a full env-knob reference.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"
source "${ORBIT_ROOT}/scripts/lib/paths.sh"

# === Recipe identity ===
LAUNCHER_NAME=${LAUNCHER_NAME:-run_qwen3_30b_a3b_instruct_2507_fp8_math_megatron_oft}
PRECISION_PROFILE="fp8"
REQUIRE_MEGATRON_LOAD=${REQUIRE_MEGATRON_LOAD:-1}
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${RUN_LOG:-${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log}"

# === Offload ===
# Free GPU for SGLang during rollout by paging Megatron weights/optimizer to host.
OFFLOAD_TRAIN=${OFFLOAD_TRAIN:-1}
OFFLOAD_TRAIN_ASYNC=${OFFLOAD_TRAIN_ASYNC:-1}
OFFLOAD_ROLLOUT=${OFFLOAD_ROLLOUT:-1}

# === Model spec ===
# Qwen3-30B-A3B-Instruct-2507 uses rotary base 1e7.
MODEL_ARGS_ROTARY_BASE=${MODEL_ARGS_ROTARY_BASE:-10000000}
MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-"${ORBIT_ROOT}/orbit_plugins/model_args/qwen3-30B-A3B.sh"}
# SGLang's flashinfer_trtllm MoE backend (used for both FP8 and NVFP4) requires
# router logits in model dtype (bfloat16). Keep Megatron aligned by omitting
# --moe-router-dtype unless the caller explicitly opts back into fp32.
MODEL_ARGS_MOE_ROUTER_DTYPE=${MODEL_ARGS_MOE_ROUTER_DTYPE:-none}

# === Data ===
DATASET=${DATASET:-math}
: "${TRAIN_JSONL:?set TRAIN_JSONL to a training jsonl path}"
TEST_JSONL=${TEST_JSONL:-}

# === Checkpoints ===
: "${HF_CKPT:?set HF_CKPT to a Hugging Face checkpoint path}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to a Megatron torch_dist checkpoint path}"
LOAD_CKPT=${LOAD_CKPT:-${MEGATRON_LOAD}}
SAVE_DIR=${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/Qwen3-30B-A3B-Instruct-2507-FP8_${DATASET}_oft}

# === Local checkpoint staging (Lustre → NVMe) ===
LOCAL_STAGE_ROOT=${LOCAL_STAGE_ROOT:-${ORBIT_CACHE_DIR:-${HOME}/.cache/orbit}/stage}
STAGE_HF_CKPT_TO=${STAGE_HF_CKPT_TO:-${LOCAL_STAGE_ROOT}/hf_models/Qwen3-30B-A3B-Instruct-2507-FP8}
STAGE_MEGATRON_CKPT_TO=${STAGE_MEGATRON_CKPT_TO:-${LOCAL_STAGE_ROOT}/Megatron-Bridge/checkpoints/Qwen3-30B-A3B-Instruct-2507-FP8}
FORCE_STAGE_HF_CKPT=${FORCE_STAGE_HF_CKPT:-0}
FORCE_STAGE_MEGATRON_CKPT=${FORCE_STAGE_MEGATRON_CKPT:-0}

# === Resources ===
RAY_NUM_CPUS=${RAY_NUM_CPUS:-64}

# === Parity smoke mode ===
PARITY_CHECK=${PARITY_CHECK:-0}
PARITY_CHECK_FAST=${PARITY_CHECK_FAST:-${PARITY_CHECK}}
if [[ "${PARITY_CHECK,,}" =~ ^(1|true|yes|y|on)$ && "${PARITY_CHECK_FAST,,}" =~ ^(1|true|yes|y|on)$ ]]; then
    ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-1}
    N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-1}
    GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-1}
    ROLLOUT_MAX_RESPONSE_LEN=${ROLLOUT_MAX_RESPONSE_LEN:-64}
    EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-64}
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
    GPUS_PER_NODE=${GPUS_PER_NODE:-1}
    ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
    EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
    SGLANG_EXPERT_PARALLEL_SIZE=${SGLANG_EXPERT_PARALLEL_SIZE:-1}
    SGLANG_DP_SIZE=${SGLANG_DP_SIZE:-1}
    SGLANG_ENABLE_DP_ATTENTION=${SGLANG_ENABLE_DP_ATTENTION:-0}
    SGLANG_ENABLE_DP_LM_HEAD=${SGLANG_ENABLE_DP_LM_HEAD:-0}
fi
# Default GPU count after the parity block so PARITY_CHECK=1 can preempt to 1.
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# === Training schedule ===
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-128}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-512}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16384}

# === Eval ===
DISABLE_EVAL=${DISABLE_EVAL:-0}
SKIP_EVAL_BEFORE_TRAIN=${SKIP_EVAL_BEFORE_TRAIN:-0}
EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-1024}

# === Rollout backend (SGLang) — FP8 quantization ===
SGLANG_QUANTIZATION=${SGLANG_QUANTIZATION:-fp8}
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-flashinfer}
SGLANG_FP8_GEMM_BACKEND=${SGLANG_FP8_GEMM_BACKEND:-triton}
SGLANG_MOE_RUNNER_BACKEND=${SGLANG_MOE_RUNNER_BACKEND:-triton}
SGLANG_MOE_MEGATRON_WEIGHTED_SWIGLU=${SGLANG_MOE_MEGATRON_WEIGHTED_SWIGLU:-0}
# The HF FP8 checkpoint stores regular FP32 inverse scales. On Blackwell,
# SGLang's auto dense-FP8 path prefers DeepGEMM, whose UE8M0 scale mode has
# produced unstable Qwen3-30B rollouts for this checkpoint. Keep parity runs on
# Triton kernels unless the caller explicitly opts into another backend.
export SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}
# Keep router fp32 off by default. SGLang auto-enables its Megatron weighted
# SwiGLU compatibility mode inside deterministic inference.
SGLANG_MOE_ROUTER_FORCE_FP32=${SGLANG_MOE_ROUTER_FORCE_FP32:-0}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.50}
SGLANG_DISABLE_CUDA_GRAPH=${SGLANG_DISABLE_CUDA_GRAPH:-0}
ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
SGLANG_EXPERT_PARALLEL_SIZE=${SGLANG_EXPERT_PARALLEL_SIZE:-}

# === FP8 Megatron quantization ===
ROLLOUT_QUANTIZATION=${ROLLOUT_QUANTIZATION:-fp8}
MEGATRON_FP8_FORMAT=${MEGATRON_FP8_FORMAT:-e4m3}
MEGATRON_FP8_RECIPE=${MEGATRON_FP8_RECIPE:-blockwise}
MEGATRON_ACTIVATION_FP8=${MEGATRON_ACTIVATION_FP8:-none}
MEGATRON_OFT_FP8_ACTIVATION_QUANT=${MEGATRON_OFT_FP8_ACTIVATION_QUANT:-w8a8}
MEGATRON_KEEP_NATIVE_FP8_WEIGHTS=${MEGATRON_KEEP_NATIVE_FP8_WEIGHTS:-True}
MEGATRON_QWEN3_FP8_GEMM_BACKEND=${MEGATRON_QWEN3_FP8_GEMM_BACKEND:-sglang_native}
export MEGATRON_QWEN3_FP8_GEMM_BACKEND=${MEGATRON_QWEN3_FP8_GEMM_BACKEND}
TRAIN_ENV_VARS=${TRAIN_ENV_VARS:-"{\"MEGATRON_OFT_FP8_ACTIVATION_QUANT\":\"${MEGATRON_OFT_FP8_ACTIVATION_QUANT}\",\"MEGATRON_KEEP_NATIVE_FP8_WEIGHTS\":\"${MEGATRON_KEEP_NATIVE_FP8_WEIGHTS}\",\"MEGATRON_QWEN3_FP8_GEMM_BACKEND\":\"${MEGATRON_QWEN3_FP8_GEMM_BACKEND}\"}"}
# FP8 quant scales are stored as regular tensor keys (`weight_scale_inv`) in
# the Megatron DCP, so the per-module ModelOpt extra_state restore is not
# needed for FP8 actor/ref load. Skip it to sidestep the PyTorch 2.6+ vs
# ModelOpt pickle-protocol-4 incompatibility. NVFP4 keeps it on by default.
export ORBIT_RESTORE_MODELOPT_STATE=${ORBIT_RESTORE_MODELOPT_STATE:-0}

# === PEFT (OFT) ===
TARGET_MODULES=${TARGET_MODULES:-all-linear}
# Keep aligned with SGLang's max_oft_block_size on the low-precision parity paths.
OFT_BLOCK_SIZE=${OFT_BLOCK_SIZE:-32}
OFT_EPS=${OFT_EPS:-6e-5}
OFT_COFT=${OFT_COFT:-0}
OFT_BLOCK_SHARE=${OFT_BLOCK_SHARE:-0}

# === Parity gates ===
MAX_LOGPROB_ABS_DIFF=${MAX_LOGPROB_ABS_DIFF:-0.20}
MIN_TOPK_AGREEMENT=${MIN_TOPK_AGREEMENT:-0.50}

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
    --rollout-max-response-len 1024
    --rollout-temperature 1.0
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
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

EVAL_ARGS=(
    --eval-interval 10
    --eval-prompt-data math "${TEST_JSONL}"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
    --eval-top-k 1
    --eval-pass-k-values 1 2 4 8 16
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
    --rollout-num-gpus 0
    --sglang-max-running-requests 1024
    --sglang-quantization "${SGLANG_QUANTIZATION}"
    --sglang-fp8-gemm-backend "${SGLANG_FP8_GEMM_BACKEND}"
    --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}"
    --sglang-moe-runner-backend "${SGLANG_MOE_RUNNER_BACKEND}"
    --router-disable-circuit-breaker
)
if [[ -n "${SGLANG_EXPERT_PARALLEL_SIZE}" ]]; then
    SGLANG_ARGS+=(--sglang-expert-parallel-size "${SGLANG_EXPERT_PARALLEL_SIZE}")
fi
if is_true "${SGLANG_ENABLE_FP32_LM_HEAD:-0}"; then
    SGLANG_ARGS+=(--sglang-enable-fp32-lm-head)
fi
if is_true "${SGLANG_ENABLE_DETERMINISTIC_INFERENCE:-0}"; then
    SGLANG_ARGS+=(--sglang-enable-deterministic-inference)
fi
if is_true "${SGLANG_MOE_MEGATRON_WEIGHTED_SWIGLU:-0}"; then
    SGLANG_ARGS+=(--sglang-moe-megatron-weighted-swiglu)
fi
if is_true "${SGLANG_MOE_ROUTER_FORCE_FP32:-0}"; then
    SGLANG_ARGS+=(--sglang-moe-router-force-fp32)
fi
if is_true "${SGLANG_DISABLE_FLASHINFER_AUTOTUNE:-0}"; then
    SGLANG_ARGS+=(--sglang-disable-flashinfer-autotune)
fi
if is_true "${SGLANG_DISABLE_CUDA_GRAPH:-0}"; then
    SGLANG_ARGS+=(--sglang-disable-cuda-graph)
fi

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --offload-train
    --offload-train-async
    --offload-rollout
)
if [[ -n "${MEGATRON_FP8_FORMAT}" && "${MEGATRON_FP8_FORMAT,,}" != "none" ]]; then
    MISC_ARGS+=(--fp8-format "${MEGATRON_FP8_FORMAT}")
fi
if [[ -n "${MEGATRON_FP8_RECIPE}" && "${MEGATRON_FP8_RECIPE,,}" != "none" ]]; then
    MISC_ARGS+=(--fp8-recipe "${MEGATRON_FP8_RECIPE}")
fi
if [[ -n "${TRAIN_ENV_VARS}" ]]; then
    MISC_ARGS+=(--train-env-vars "${TRAIN_ENV_VARS}")
fi

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
