#!/usr/bin/env bash
# Common Qwen2.5-3B Search-R1 PPO launcher. Source from a mode wrapper.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this file from a Search-R1 PPO wrapper instead of running it directly." >&2
    exit 2
fi

: "${SEARCH_R1_PEFT_MODE:?SEARCH_R1_PEFT_MODE must be full, lora, or oft}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"

case "${SEARCH_R1_PEFT_MODE}" in
    full | lora | oft) ;;
    *)
        echo "SEARCH_R1_PEFT_MODE must be full, lora, or oft; got ${SEARCH_R1_PEFT_MODE}" >&2
        exit 2
        ;;
esac

# === Model identity ===
SEARCH_R1_MODEL_TAG="${SEARCH_R1_MODEL_TAG:-qwen25_3b}"
SEARCH_R1_MODEL_DIR_NAME="${SEARCH_R1_MODEL_DIR_NAME:-Qwen2.5-3B-Instruct}"
SEARCH_R1_MODEL_ARGS_FILE="${SEARCH_R1_MODEL_ARGS_FILE:-qwen2.5-3B.sh}"

# === Recipe identity ===
LAUNCHER_NAME="run_${SEARCH_R1_MODEL_TAG}_bf16_search_r1_ppo_${SEARCH_R1_PEFT_MODE}"
WANDB_PROJECT=${WANDB_PROJECT:-orbit-search-r1}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${RUN_LOG:-${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log}"

# === Paths ===
: "${HF_CKPT:?set HF_CKPT to a Hugging Face checkpoint path}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to a Megatron torch_dist checkpoint path}"
SAVE_DIR="${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/${SEARCH_R1_MODEL_DIR_NAME}_search_r1_ppo_${SEARCH_R1_PEFT_MODE}}"
TRAIN_DATA="${TRAIN_DATA:-${TRAIN_JSONL:-}}"
: "${TRAIN_DATA:?set TRAIN_DATA or TRAIN_JSONL to a Search-R1 train parquet/jsonl path}"
TEST_DATA="${TEST_DATA:-${TEST_JSONL:-}}"

# === Resources ===
# PPO uses a separate full-model critic. Default 8-GPU layout:
# actor=2 GPUs, critic=2 GPUs, rollout=4 GPUs.
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
CRITIC_NUM_GPUS_PER_NODE="${CRITIC_NUM_GPUS_PER_NODE:-2}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"

# === Model args ===
source "${ORBIT_ROOT}/miles_plugins/model_args/${SEARCH_R1_MODEL_ARGS_FILE}"

# === Training schedule ===
NUM_ROLLOUT="${NUM_ROLLOUT:-3000}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-512}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-512}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-9216}"
PEFT_DISTRIBUTED_TRANSPORT="${PEFT_DISTRIBUTED_TRANSPORT:-nccl}"
if [[ "${PEFT_DISTRIBUTED_TRANSPORT}" == "nccl" ]]; then
    ADAPTER_DOUBLE_BUFFER="${ADAPTER_DOUBLE_BUFFER:-1}"
else
    ADAPTER_DOUBLE_BUFFER="${ADAPTER_DOUBLE_BUFFER:-0}"
fi

# === Search-R1 args ===
SEARCH_R1_BACKEND="${SEARCH_R1_BACKEND:-local}"
SEARCH_R1_LOCAL_URL="${SEARCH_R1_LOCAL_URL:-http://127.0.0.1:8000/retrieve}"
SEARCH_R1_TOPK="${SEARCH_R1_TOPK:-3}"
SEARCH_R1_MAX_TURNS="${SEARCH_R1_MAX_TURNS:-2}"
SEARCH_R1_CONCURRENCY="${SEARCH_R1_CONCURRENCY:-256}"
SEARCH_R1_TIMEOUT="${SEARCH_R1_TIMEOUT:-120}"
SEARCH_R1_PROXY="${SEARCH_R1_PROXY:-}"
SEARCH_R1_FORMAT_SCORE="${SEARCH_R1_FORMAT_SCORE:-0.2}"
SEARCH_R1_CONFIG_PATH="${SEARCH_R1_CONFIG_PATH:-${RUN_LOG%.log}.search_r1.yaml}"

mkdir -p "$(dirname "${SEARCH_R1_CONFIG_PATH}")"
cat > "${SEARCH_R1_CONFIG_PATH}" <<EOF
search_r1_backend: "${SEARCH_R1_BACKEND}"
search_r1_local_url: "${SEARCH_R1_LOCAL_URL}"
search_r1_topk: ${SEARCH_R1_TOPK}
search_r1_max_turns: ${SEARCH_R1_MAX_TURNS}
search_r1_concurrency: ${SEARCH_R1_CONCURRENCY}
search_r1_timeout: ${SEARCH_R1_TIMEOUT}
search_r1_proxy: null
search_r1_format_score: ${SEARCH_R1_FORMAT_SCORE}
EOF
if [[ -n "${SEARCH_R1_PROXY}" ]]; then
    python3 - "${SEARCH_R1_CONFIG_PATH}" "${SEARCH_R1_PROXY}" <<'PY'
import sys
from pathlib import Path

path = Path(sys.argv[1])
proxy = sys.argv[2]
text = path.read_text()
path.write_text(text.replace("search_r1_proxy: null\n", f'search_r1_proxy: "{proxy}"\n'))
PY
fi

# === ARGS arrays ===
COLOCATE_ARGS=()

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${MEGATRON_LOAD}"
    --save "${SAVE_DIR}/actor"
    --critic-load "${MEGATRON_LOAD}"
    --critic-save "${SAVE_DIR}/critic"
    --save-interval "${SAVE_INTERVAL:-200}"
    --no-save-optim
    --no-save-rng
    --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_DATA}"
    --input-key "${INPUT_KEY:-prompt}"
    --label-key "${LABEL_KEY:-reward_model}"
    --apply-chat-template
    --rollout-shuffle
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --balance-data
    --custom-generate-function-path miles_plugins.search_r1.generate_with_search.generate
    --custom-rm-path miles_plugins.search_r1.generate_with_search.reward_func
)

SEARCH_R1_DYNAMIC_SAMPLING_FILTER_PATH="${SEARCH_R1_DYNAMIC_SAMPLING_FILTER_PATH:-}"
if [[ -n "${SEARCH_R1_DYNAMIC_SAMPLING_FILTER_PATH}" ]]; then
    ROLLOUT_ARGS+=(--dynamic-sampling-filter-path "${SEARCH_R1_DYNAMIC_SAMPLING_FILTER_PATH}")
fi

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-1e-6}"
    --critic-lr "${CRITIC_LR:-1e-5}"
    --lr-decay-style constant
    --weight-decay "${WEIGHT_DECAY:-0.01}"
    --adam-beta1 "${ADAM_BETA1:-0.9}"
    --adam-beta2 "${ADAM_BETA2:-0.98}"
)

RL_ARGS=(
    --advantage-estimator ppo
    --kl-loss-coef "${KL_LOSS_COEF:-0.0}"
    --kl-loss-type k1
    --kl-coef "${KL_COEF:-0.0}"
    --entropy-coef "${ENTROPY_COEF:-0.0}"
    --eps-clip "${EPS_CLIP:-0.2}"
    --eps-clip-high "${EPS_CLIP_HIGH:-0.28}"
    --value-clip "${VALUE_CLIP:-0.2}"
    --gamma 1.0
    --lambd 1.0
    --num-critic-only-steps 1
    --normalize-advantages
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
    --tensor-model-parallel-size "${TP_SIZE:-1}"
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

if [[ -n "${TEST_DATA}" ]]; then
    EVAL_ARGS=(
        --eval-interval "${EVAL_INTERVAL:-25}"
        --eval-prompt-data search_r1 "${TEST_DATA}"
        --eval-input-key "${EVAL_INPUT_KEY:-${INPUT_KEY:-prompt}}"
        --eval-label-key "${EVAL_LABEL_KEY:-${LABEL_KEY:-reward_model}}"
        --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-1}"
        --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
        --eval-temperature "${EVAL_TEMPERATURE:-0.0}"
        --eval-top-p "${EVAL_TOP_P:-1.0}"
        --eval-top-k "${EVAL_TOP_K:-1}"
        --eval-pass-k-values 1 2 4 8
    )
else
    EVAL_ARGS=()
fi

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-2}"
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.70}"
    --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-64}"
    --sglang-force-native-ops
    --sglang-attention-backend triton
    --sglang-sampling-backend pytorch
    --router-disable-circuit-breaker
)

MISC_ARGS=(
    --custom-config-path "${SEARCH_R1_CONFIG_PATH}"
    --critic-num-gpus-per-node "${CRITIC_NUM_GPUS_PER_NODE}"
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --attention-backend flash
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --no-gradient-accumulation-fusion
    --no-offload-train
    --no-offload-train-async
    --no-offload-rollout
    --cuda-graph-impl local
    --cuda-graph-scope full_iteration
    --te-rng-tracker
    --no-check-for-nan-in-loss-and-grad
)

DEBUG_ARGS=(
    --log-passrate
)

case "${SEARCH_R1_PEFT_MODE}" in
    full)
        PEFT_ARGS=(--peft-method none)
        ;;
    lora)
        PEFT_ARGS=(
            --peft-method lora
            --peft-distributed-transport "${PEFT_DISTRIBUTED_TRANSPORT}"
            --peft-variant standard
            --lora-rank "${LORA_RANK:-64}"
            --lora-alpha "${LORA_ALPHA:-32}"
            --lora-dropout "${LORA_DROPOUT:-0.0}"
            --target-modules "${TARGET_MODULES:-all-linear}"
        )
        ;;
    oft)
        PEFT_ARGS=(
            --peft-method oft
            --peft-distributed-transport "${PEFT_DISTRIBUTED_TRANSPORT}"
            --peft-variant standard
            --oft-type canonical_oft
            --oft-block-size "${OFT_BLOCK_SIZE:-32}"
            --oft-eps "${OFT_EPS:-6e-5}"
            --target-modules "${TARGET_MODULES:-all-linear}"
        )
        ;;
esac

if [[ "${SEARCH_R1_PEFT_MODE}" != "full" && "${ADAPTER_DOUBLE_BUFFER}" == "1" ]]; then
    if [[ "${PEFT_DISTRIBUTED_TRANSPORT}" != "nccl" ]]; then
        echo "ADAPTER_DOUBLE_BUFFER=1 requires PEFT_DISTRIBUTED_TRANSPORT=nccl." >&2
        exit 2
    fi
    PEFT_ARGS+=(--adapter-double-buffer)
fi

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
