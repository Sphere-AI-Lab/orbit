#!/usr/bin/env bash
# Common Qwen3-4B-Instruct-2507 Tau-bench PPO launcher. Source from a mode wrapper.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this file from a Tau-bench PPO wrapper instead of running it directly." >&2
    exit 2
fi

: "${TAU_BENCH_PEFT_MODE:?TAU_BENCH_PEFT_MODE must be full, lora, or oft}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"

case "${TAU_BENCH_PEFT_MODE}" in
    full | lora | oft) ;;
    *)
        echo "TAU_BENCH_PEFT_MODE must be full, lora, or oft; got ${TAU_BENCH_PEFT_MODE}" >&2
        exit 2
        ;;
esac

# === Recipe identity ===
LAUNCHER_NAME="run_qwen3_4b_instruct_2507_bf16_tau_bench_ppo_${TAU_BENCH_PEFT_MODE}"
WANDB_PROJECT=${WANDB_PROJECT:-orbit-tau-bench}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${RUN_LOG:-${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log}"

# === Paths ===
: "${HF_CKPT:?set HF_CKPT to a Hugging Face checkpoint path}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to a Megatron torch_dist checkpoint path}"
SAVE_DIR="${SAVE_DIR:-${ORBIT_ROOT}/orbit_ckpts/Qwen3-4B-Instruct-2507_tau_bench_ppo_${TAU_BENCH_PEFT_MODE}}"
TRAIN_DATA="${TRAIN_DATA:-${TRAIN_JSONL:-}}"
: "${TRAIN_DATA:?set TRAIN_DATA or TRAIN_JSONL to a Tau-bench task-index jsonl path}"
TEST_DATA="${TEST_DATA:-${TEST_JSONL:-}}"

# === Resources ===
# PPO default 8-GPU layout: actor=2 GPUs, critic=2 GPUs, rollout=4 GPUs.
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
CRITIC_NUM_GPUS_PER_NODE="${CRITIC_NUM_GPUS_PER_NODE:-2}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"

# === Model args ===
MODEL_ARGS_FILE="${MODEL_ARGS_FILE:-${ORBIT_ROOT}/miles_plugins/model_args/qwen3-4B-Instruct-2507.sh}"
source "${MODEL_ARGS_FILE}"

# === Training schedule ===
NUM_ROLLOUT="${NUM_ROLLOUT:-500}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-32}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-1024}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-1024}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-9216}"
PEFT_DISTRIBUTED_TRANSPORT="${PEFT_DISTRIBUTED_TRANSPORT:-nccl}"
if [[ "${PEFT_DISTRIBUTED_TRANSPORT}" == "nccl" ]]; then
    ADAPTER_DOUBLE_BUFFER="${ADAPTER_DOUBLE_BUFFER:-1}"
else
    ADAPTER_DOUBLE_BUFFER="${ADAPTER_DOUBLE_BUFFER:-0}"
fi

# === Tau-bench args ===
TAU_BENCH_ENV="${TAU_BENCH_ENV:-retail}"
TAU_BENCH_TASK_SPLIT="${TAU_BENCH_TASK_SPLIT:-train}"
TAU_BENCH_EVAL_NAME="${TAU_BENCH_EVAL_NAME:-retail-dev}"
TAU_BENCH_USER_STRATEGY="${TAU_BENCH_USER_STRATEGY:-llm}"
TAU_BENCH_USER_MODEL_PROVIDER="${TAU_BENCH_USER_MODEL_PROVIDER:-${TAU_USER_MODEL_PROVIDER:-gemini}}"
TAU_BENCH_USER_MODEL="${TAU_BENCH_USER_MODEL:-${TAU_USER_MODEL:-gemini-2.5-flash-lite}}"
TAU_BENCH_AGENT_MAX_STEPS="${TAU_BENCH_AGENT_MAX_STEPS:-30}"
TAU_BENCH_TOOL_PARSER="${TAU_BENCH_TOOL_PARSER:-qwen25}"
TAU_BENCH_DYNAMIC_SAMPLING_FILTER_PATH="${TAU_BENCH_DYNAMIC_SAMPLING_FILTER_PATH:-miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std}"
TAU_BENCH_CONFIG_PATH="${TAU_BENCH_CONFIG_PATH:-${RUN_LOG%.log}.tau_bench.yaml}"

# Keep the provider selection visible to child processes that inherit env vars.
export TAU_USER_MODEL_PROVIDER="${TAU_BENCH_USER_MODEL_PROVIDER}"
export TAU_USER_MODEL="${TAU_BENCH_USER_MODEL}"

mkdir -p "$(dirname "${TAU_BENCH_CONFIG_PATH}")"
cat > "${TAU_BENCH_CONFIG_PATH}" <<EOF
tau_bench_env: "${TAU_BENCH_ENV}"
tau_bench_task_split: "${TAU_BENCH_TASK_SPLIT}"
tau_bench_user_strategy: "${TAU_BENCH_USER_STRATEGY}"
tau_bench_user_model_provider: "${TAU_BENCH_USER_MODEL_PROVIDER}"
tau_bench_user_model: "${TAU_BENCH_USER_MODEL}"
tau_bench_agent_max_steps: ${TAU_BENCH_AGENT_MAX_STEPS}
tau_bench_tool_parser: "${TAU_BENCH_TOOL_PARSER}"
EOF

# === ARGS arrays ===
COLOCATE_ARGS=()

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${MEGATRON_LOAD}"
    --save "${SAVE_DIR}/actor"
    --critic-load "${MEGATRON_LOAD}"
    --critic-save "${SAVE_DIR}/critic"
    --save-interval "${SAVE_INTERVAL:-100}"
    --no-save-optim
    --no-save-rng
    --megatron-to-hf-mode bridge
)

ROLLOUT_ARGS=(
    --prompt-data "${TRAIN_DATA}"
    --input-key "${INPUT_KEY:-index}"
    --rollout-shuffle
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --balance-data
    --custom-generate-function-path miles_plugins.tau_bench.generate_with_tau.generate
)

if [[ -n "${TAU_BENCH_DYNAMIC_SAMPLING_FILTER_PATH}" && "${TAU_BENCH_DYNAMIC_SAMPLING_FILTER_PATH}" != "none" ]]; then
    ROLLOUT_ARGS+=(--dynamic-sampling-filter-path "${TAU_BENCH_DYNAMIC_SAMPLING_FILTER_PATH}")
fi

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-1e-6}"
    --critic-lr "${CRITIC_LR:-1e-5}"
    --lr-decay-style constant
    --weight-decay "${WEIGHT_DECAY:-0.1}"
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
    --tensor-model-parallel-size "${TP_SIZE:-2}"
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
        --eval-interval "${EVAL_INTERVAL:-5}"
        --eval-prompt-data "${TAU_BENCH_EVAL_NAME}" "${TEST_DATA}"
        --eval-input-key "${EVAL_INPUT_KEY:-${INPUT_KEY:-index}}"
        --n-samples-per-eval-prompt "${N_SAMPLES_PER_EVAL_PROMPT:-1}"
        --eval-max-response-len "${EVAL_MAX_RESPONSE_LEN}"
        --eval-temperature "${EVAL_TEMPERATURE:-0.0}"
        --eval-top-p "${EVAL_TOP_P:-1.0}"
        --eval-top-k "${EVAL_TOP_K:-1}"
    )
else
    EVAL_ARGS=()
fi

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine "${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.70}"
    --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-64}"
    --sglang-force-native-ops
    --sglang-attention-backend triton
    --sglang-sampling-backend pytorch
    --router-disable-circuit-breaker
)

if [[ -n "${SGLANG_SERVER_CONCURRENCY:-}" ]]; then
    SGLANG_ARGS+=(--sglang-server-concurrency "${SGLANG_SERVER_CONCURRENCY}")
fi

MISC_ARGS=(
    --custom-config-path "${TAU_BENCH_CONFIG_PATH}"
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

case "${TAU_BENCH_PEFT_MODE}" in
    full)
        PEFT_ARGS=(--peft-method none)
        ;;
    lora)
        PEFT_ARGS=(
            --peft-method lora
            --peft-distributed-transport "${PEFT_DISTRIBUTED_TRANSPORT}"
            --peft-variant standard
            --lora-rank "${LORA_RANK:-32}"
            --lora-alpha "${LORA_ALPHA:-64}"
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

if [[ "${TAU_BENCH_PEFT_MODE}" != "full" && "${ADAPTER_DOUBLE_BUFFER}" == "1" ]]; then
    if [[ "${PEFT_DISTRIBUTED_TRANSPORT}" != "nccl" ]]; then
        echo "ADAPTER_DOUBLE_BUFFER=1 requires PEFT_DISTRIBUTED_TRANSPORT=nccl." >&2
        exit 2
    fi
    PEFT_ARGS+=(--adapter-double-buffer)
fi

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
