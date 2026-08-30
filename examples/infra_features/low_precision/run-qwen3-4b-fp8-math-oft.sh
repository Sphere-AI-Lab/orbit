#!/usr/bin/env bash
# Qwen3-4B-Instruct-2507 FP8 + OFT on the math dataset. Self-contained launcher.
#
# Orbit-native port of verl/examples/run_qwen3_4b_fp8_math_megatron_oft.sh.
# See examples/README.md for a full env-knob reference.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"
source "${ORBIT_ROOT}/scripts/lib/paths.sh"

# === Recipe identity ===
LAUNCHER_NAME=${LAUNCHER_NAME:-run_qwen3_4b_fp8_math_megatron_oft}
PRECISION_PROFILE="fp8"
REQUIRE_MEGATRON_LOAD=${REQUIRE_MEGATRON_LOAD:-1}
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${RUN_LOG:-${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log}"

# === Model spec ===
# Qwen3-Instruct-2507 uses rotary base 5e6.
MODEL_ARGS_ROTARY_BASE=${MODEL_ARGS_ROTARY_BASE:-5000000}
MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-"${ORBIT_ROOT}/miles_plugins/model_args/qwen3-4B.sh"}

# === Data ===
DATASET=${DATASET:-math}
: "${TRAIN_JSONL:?set TRAIN_JSONL to a training jsonl path}"
TEST_JSONL=${TEST_JSONL:-}

# === Checkpoints ===
: "${HF_CKPT:?set HF_CKPT to a Hugging Face checkpoint path}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to a Megatron torch_dist checkpoint path}"
LOAD_CKPT=${LOAD_CKPT:-${MEGATRON_LOAD}}
SAVE_DIR=${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/Qwen3-4B-Instruct-2507-FP8_${DATASET}_oft}

# === Resources + training schedule ===
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-128}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-512}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16384}

# === Rollout backend (SGLang) ===
SGLANG_CHUNKED_PREFILL_SIZE=${SGLANG_CHUNKED_PREFILL_SIZE:-4096}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.60}
# Full math eval has thousands of prompts; cap async fanout so the SGLang
# router does not open all backend circuits under the initial request burst.
EVAL_GENERATE_MAX_CONCURRENCY=${EVAL_GENERATE_MAX_CONCURRENCY:-128}
# This value is per rollout engine; with the default 8 one-GPU engines it caps
# rollout /generate fanout at 128 instead of submitting the full 512-sample
# rollout burst to the router at once.
SGLANG_SERVER_CONCURRENCY=${SGLANG_SERVER_CONCURRENCY:-16}

# === FP8 quantization ===
SGLANG_QUANTIZATION=${SGLANG_QUANTIZATION:-fp8}
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-flashinfer}
# 'auto' lets SGLang pick the fastest available FP8 GEMM kernel for this GPU.
SGLANG_FP8_GEMM_BACKEND=${SGLANG_FP8_GEMM_BACKEND:-auto}
ROLLOUT_QUANTIZATION=${ROLLOUT_QUANTIZATION:-fp8}
# Activations stay BF16; trainer Linear emits FP8 weights only.
MEGATRON_ACTIVATION_FP8=${MEGATRON_ACTIVATION_FP8:-none}
# Per-tensor W8A8 activation scales for OFT-wrapped Linears.
MEGATRON_OFT_FP8_ACTIVATION_QUANT=${MEGATRON_OFT_FP8_ACTIVATION_QUANT:-w8a8}
# Skip the BF16→FP8 requant on save: rollout already consumes native FP8.
MEGATRON_KEEP_NATIVE_FP8_WEIGHTS=${MEGATRON_KEEP_NATIVE_FP8_WEIGHTS:-True}

# === Parity gates ===
# FP8 has wider rounding; relax tolerances vs. INT4.
MAX_LOGPROB_ABS_DIFF=${MAX_LOGPROB_ABS_DIFF:-0.20}
MIN_TOPK_AGREEMENT=${MIN_TOPK_AGREEMENT:-0.50}

# === PEFT (OFT) ===
TARGET_MODULES=${TARGET_MODULES:-all-linear}
OFT_BLOCK_SIZE=${OFT_BLOCK_SIZE:-128}
OFT_EPS=${OFT_EPS:-6e-5}
OFT_COFT=${OFT_COFT:-0}
OFT_BLOCK_SHARE=${OFT_BLOCK_SHARE:-0}

# === RL ===
USE_KL_LOSS=${USE_KL_LOSS:-0}

# === W&B ===
ENABLE_WANDB=${ENABLE_WANDB:-auto}
WANDB_PROJECT=${WANDB_PROJECT:-orbit-release}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}

# === Checkpoint save interval ===
SAVE_INTERVAL=${SAVE_INTERVAL:-200}

# === Training schedule ===
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
    --expert-model-parallel-size 1
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
    --eval-max-response-len 1024
    --eval-top-k 1
    --eval-generate-max-concurrency "${EVAL_GENERATE_MAX_CONCURRENCY}"
    --eval-pass-k-values 1 2 4 8 16
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
    --rollout-num-gpus 0
    --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
    --sglang-max-running-requests 1024
    --sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}"
    --sglang-quantization "${SGLANG_QUANTIZATION}"
    --sglang-fp8-gemm-backend "${SGLANG_FP8_GEMM_BACKEND}"
    --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}"
    --router-disable-circuit-breaker
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --no-offload-train
    --no-offload-train-async
    --offload-rollout
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
