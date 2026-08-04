#!/usr/bin/env bash
# Qwen3-30B-A3B (MoE) NVFP4 (modelopt_fp4) + OFT on the math dataset.
# Self-contained launcher.
# Rollout uses the modelopt FP4 HF checkpoint; actor/ref load the converted
# Megatron distributed checkpoint.
# See examples/README.md for a full env-knob reference.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"
source "${ORBIT_ROOT}/scripts/lib/paths.sh"

# === Recipe identity ===
LAUNCHER_NAME=${LAUNCHER_NAME:-run_qwen3_30b_a3b_nvfp4_math_megatron_oft}
PRECISION_PROFILE="nvfp4"
REQUIRE_MEGATRON_LOAD=${REQUIRE_MEGATRON_LOAD:-1}
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${RUN_LOG:-${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log}"

# === Offload ===
# Free GPU memory for SGLang during rollout, following the working Kimi low-precision recipe.
OFFLOAD_TRAIN=${OFFLOAD_TRAIN:-1}
OFFLOAD_TRAIN_ASYNC=${OFFLOAD_TRAIN_ASYNC:-1}
OFFLOAD_ROLLOUT=${OFFLOAD_ROLLOUT:-1}

# === Model spec ===
MODEL_ARGS_ROTARY_BASE=${MODEL_ARGS_ROTARY_BASE:-1000000}
MODEL_ARGS_FILE=${MODEL_ARGS_FILE:-"${ORBIT_ROOT}/orbit_plugins/model_args/qwen3-30B-A3B.sh"}

# === Data ===
DATASET=${DATASET:-math}
: "${TRAIN_JSONL:?set TRAIN_JSONL to a training jsonl path}"
TEST_JSONL=${TEST_JSONL:-}

# === Checkpoints ===
: "${HF_CKPT:?set HF_CKPT to a Hugging Face checkpoint path}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to a Megatron torch_dist checkpoint path}"
LOAD_CKPT=${LOAD_CKPT:-${MEGATRON_LOAD}}
SAVE_DIR=${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/Qwen3-30B-A3B-NVFP4_${DATASET}_oft}

# === Local checkpoint staging (Lustre -> NVMe) ===
LOCAL_STAGE_ROOT=${LOCAL_STAGE_ROOT:-${ORBIT_CACHE_DIR:-${HOME}/.cache/orbit}/stage}
STAGE_HF_CKPT_TO=${STAGE_HF_CKPT_TO:-${LOCAL_STAGE_ROOT}/hf_models/Qwen3-30B-A3B-NVFP4}
STAGE_MEGATRON_CKPT_TO=${STAGE_MEGATRON_CKPT_TO:-${LOCAL_STAGE_ROOT}/Megatron-Bridge/checkpoints/Qwen3-30B-A3B-NVFP4}
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
    # NVFP4 MoE is not compatible with SGLang's weighted-SwiGLU mode; leave it off.
    SGLANG_ENABLE_FP32_LM_HEAD=${SGLANG_ENABLE_FP32_LM_HEAD:-1}
    SGLANG_ENABLE_DETERMINISTIC_INFERENCE=${SGLANG_ENABLE_DETERMINISTIC_INFERENCE:-1}
    # Single-GPU defaults so parity runs on one device by default; override
    # by exporting GPUS_PER_NODE / SGLANG_* explicitly to run multi-GPU parity.
    GPUS_PER_NODE=${GPUS_PER_NODE:-1}
    ROLLOUT_NUM_GPUS_PER_ENGINE=${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}
    EXPERT_MODEL_PARALLEL_SIZE=${EXPERT_MODEL_PARALLEL_SIZE:-1}
    SGLANG_EXPERT_PARALLEL_SIZE=${SGLANG_EXPERT_PARALLEL_SIZE:-1}
    SGLANG_DP_SIZE=${SGLANG_DP_SIZE:-1}
    SGLANG_ENABLE_DP_ATTENTION=${SGLANG_ENABLE_DP_ATTENTION:-0}
    SGLANG_ENABLE_DP_LM_HEAD=${SGLANG_ENABLE_DP_LM_HEAD:-0}
    # Pin parity to GPU 7 so it doesn't collide with other devices in use on the
    # node. Honour any caller-supplied CUDA_VISIBLE_DEVICES (e.g. for multi-GPU
    # parity sweeps).
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
    # Legacy DSV4 name; this enables the generic train-vs-rollout logprob diff logger.
fi
# Default GPU count after the parity block so PARITY_CHECK=1 can preempt to 1.
GPUS_PER_NODE=${GPUS_PER_NODE:-8}

# === Training schedule ===
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-128}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-512}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
MAX_TOKENS_PER_GPU=${MAX_TOKENS_PER_GPU:-16384}

# === Rollout backend (SGLang) — NVFP4 quantization ===
SGLANG_QUANTIZATION=${SGLANG_QUANTIZATION:-modelopt_fp4}
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-flashinfer}
# SGLang's FlashInfer CUTLASS FP4 MoE path matches Megatron NVFP4 logprobs in
# the current Qwen3 MoE pipeline; the TRTLLM path produced large drift/NaNs.
SGLANG_MOE_RUNNER_BACKEND=${SGLANG_MOE_RUNNER_BACKEND:-flashinfer_cutlass}
SGLANG_MOE_MEGATRON_WEIGHTED_SWIGLU=${SGLANG_MOE_MEGATRON_WEIGHTED_SWIGLU:-0}
SGLANG_MOE_ROUTER_FORCE_FP32=${SGLANG_MOE_ROUTER_FORCE_FP32:-0}
export SGLANG_NVFP4_MOE_LOCAL_W2_INPUT_SCALE=${SGLANG_NVFP4_MOE_LOCAL_W2_INPUT_SCALE:-0}
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.50}
SGLANG_DISABLE_CUDA_GRAPH=${SGLANG_DISABLE_CUDA_GRAPH:-0}
SGLANG_DISABLE_FLASHINFER_AUTOTUNE=${SGLANG_DISABLE_FLASHINFER_AUTOTUNE:-1}

# === Eval ===
DISABLE_EVAL=${DISABLE_EVAL:-0}
SKIP_EVAL_BEFORE_TRAIN=${SKIP_EVAL_BEFORE_TRAIN:-0}
EVAL_MAX_RESPONSE_LEN=${EVAL_MAX_RESPONSE_LEN:-1024}

# === PEFT (OFT) ===
# Skip linear_qkv: fused TELayerNormColumnParallelLinear on Qwen3 GQA does not
# have a reliable OFT disable_adapter ref path. Keep MLP/projection wrappers.
TARGET_MODULES=${TARGET_MODULES:-linear_proj,linear_fc1,linear_fc2}
# Keep aligned with SGLang's max_oft_block_size on the low-precision parity paths.
OFT_BLOCK_SIZE=${OFT_BLOCK_SIZE:-32}
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
    --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
    --eval-top-k 1
    --eval-pass-k-values 1 2 4 8 16
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
    --rollout-num-gpus 0
    --sglang-max-running-requests 1024
    --sglang-quantization "${SGLANG_QUANTIZATION}"
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
    --attention-softmax-in-fp32
    --offload-train
    --offload-train-async
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
