#!/usr/bin/env bash
# Qwen3-4B-Instruct-2507 NVFP4 + OFT on the math dataset. Self-contained launcher.
#
# Dense-model NVFP4 parity path: rollout loads the ModelOpt FP4 HF checkpoint;
# actor/ref load the converted Megatron distributed checkpoint.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"
source "${ORBIT_ROOT}/scripts/lib/paths.sh"

# === Recipe identity ===
LAUNCHER_NAME=${LAUNCHER_NAME:-run_qwen3_4b_nvfp4_math_megatron_oft}
PRECISION_PROFILE="nvfp4"
REQUIRE_MEGATRON_LOAD=${REQUIRE_MEGATRON_LOAD:-1}
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${RUN_LOG:-${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log}"

# === Model spec ===
MODEL_ARGS_ROTARY_BASE=${MODEL_ARGS_ROTARY_BASE:-5000000}
MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-"${ORBIT_ROOT}/miles_plugins/model_args/qwen3-4B-Instruct-2507.sh"}

# === Data ===
DATASET=${DATASET:-math}
: "${TRAIN_JSONL:?set TRAIN_JSONL to a training jsonl path}"
TEST_JSONL=${TEST_JSONL:-}

# === Checkpoints ===
: "${HF_CKPT:?set HF_CKPT to a Hugging Face checkpoint path}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to a Megatron torch_dist checkpoint path}"
LOAD_CKPT=${LOAD_CKPT:-${MEGATRON_LOAD}}
SAVE_DIR=${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/Qwen3-4B-Instruct-2507-NVFP4_${DATASET}_oft}

# === Local checkpoint staging (Lustre -> NVMe) ===
LOCAL_STAGE_ROOT=${LOCAL_STAGE_ROOT:-${ORBIT_CACHE_DIR:-${HOME}/.cache/orbit}/stage}
STAGE_HF_CKPT_TO=${STAGE_HF_CKPT_TO:-${LOCAL_STAGE_ROOT}/hf_models/Qwen3-4B-Instruct-2507-NVFP4}
STAGE_MEGATRON_CKPT_TO=${STAGE_MEGATRON_CKPT_TO:-${LOCAL_STAGE_ROOT}/Megatron-Bridge/checkpoints/Qwen3-4B-Instruct-2507-NVFP4}
FORCE_STAGE_HF_CKPT=${FORCE_STAGE_HF_CKPT:-0}
FORCE_STAGE_MEGATRON_CKPT=${FORCE_STAGE_MEGATRON_CKPT:-0}

# === Resources ===
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
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
    DISABLE_SAVE=${DISABLE_SAVE:-1}
    SGLANG_ENABLE_FP32_LM_HEAD=${SGLANG_ENABLE_FP32_LM_HEAD:-1}
    SGLANG_ENABLE_DETERMINISTIC_INFERENCE=${SGLANG_ENABLE_DETERMINISTIC_INFERENCE:-1}
    GPUS_PER_NODE=${GPUS_PER_NODE:-1}
    ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
fi

# === Training schedule ===
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-128}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-512}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16384}

# === Rollout backend (SGLang) ===
SGLANG_CHUNKED_PREFILL_SIZE=${SGLANG_CHUNKED_PREFILL_SIZE:-4096}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.60}
EVAL_GENERATE_MAX_CONCURRENCY=${EVAL_GENERATE_MAX_CONCURRENCY:-128}
SGLANG_SERVER_CONCURRENCY=${SGLANG_SERVER_CONCURRENCY:-16}

# === NVFP4 quantization ===
SGLANG_QUANTIZATION=${SGLANG_QUANTIZATION:-modelopt_fp4}
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-flashinfer}
SGLANG_DISABLE_CUDA_GRAPH=${SGLANG_DISABLE_CUDA_GRAPH:-0}
SGLANG_DISABLE_FLASHINFER_AUTOTUNE=${SGLANG_DISABLE_FLASHINFER_AUTOTUNE:-1}
export SGLANG_ENABLE_JIT_DEEPGEMM=${SGLANG_ENABLE_JIT_DEEPGEMM:-0}
TRUST_REMOTE_CODE=${TRUST_REMOTE_CODE:-1}

# === Eval ===
DISABLE_EVAL=${DISABLE_EVAL:-1}
SKIP_EVAL_BEFORE_TRAIN=${SKIP_EVAL_BEFORE_TRAIN:-1}

# === PEFT (OFT) ===
TARGET_MODULES=${TARGET_MODULES:-all-linear}
OFT_BLOCK_SIZE=${OFT_BLOCK_SIZE:-32}
OFT_EPS=${OFT_EPS:-6e-5}
OFT_COFT=${OFT_COFT:-0}
OFT_BLOCK_SHARE=${OFT_BLOCK_SHARE:-0}

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

EVAL_ARGS=( )

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
    --rollout-num-gpus 0
    --sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}"
    --sglang-max-running-requests 1024
    --sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}"
    --sglang-quantization "${SGLANG_QUANTIZATION}"
    --sglang-attention-backend "${SGLANG_ATTENTION_BACKEND}"
    --router-disable-circuit-breaker
    --sglang-disable-flashinfer-autotune
)

MISC_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --trust-remote-code
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
