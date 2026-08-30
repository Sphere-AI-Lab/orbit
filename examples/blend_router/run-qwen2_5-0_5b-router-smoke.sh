#!/usr/bin/env bash
# Qwen2.5-0.5B-Instruct BF16 GRPO on a HETEROGENEOUS Ultra blend via the
# reward router: each group routes by metadata.agent to its grader —
# llm_judge equivalence (math/equivalence agents), genrm_judge (genrm
# agents), sandbox code_rm (code_gen agent); unmapped agents zero-reward
# loudly. Requires a judge server for judge/genrm rows (JUDGE_BASE_URL).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"

# === Recipe identity ===
LAUNCHER_NAME=smoke_qwen25_05b_router
WANDB_PROJECT=${WANDB_PROJECT:-orbit-release}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log"

# === Paths ===
: "${HF_CKPT:?set HF_CKPT to a Hugging Face checkpoint path}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to a Megatron torch_dist checkpoint path}"
# Judge server is only needed for judge/genrm-routed rows; rule-based-only
# blends (tool_call/mcqa/structured/ifbench/code) can run without one.
JUDGE_BASE_URL="${JUDGE_BASE_URL:-}"
SAVE_DIR="${ORBIT_ROOT}/orbit_ckpts/Qwen2.5-0.5B-Instruct_router_smoke"
: "${TRAIN_JSONL:?set TRAIN_JSONL to a training jsonl path}"
TEST_JSONL=${TEST_JSONL:-}

# === Resources ===
# actor=2 GPUs, rollout=2 GPUs.
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-2}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"

# === Model args ===
source "${ORBIT_ROOT}/miles_plugins/model_args/${MODEL_ARGS_FILE:-qwen2.5-0.5B}.sh"   # provides MODEL_ARGS=(...)

# === Training schedule ===
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-4}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-512}"
TRAIN_ROWS=${TRAIN_ROWS:-$(wc -l < "${TRAIN_JSONL}")}
NUM_ROLLOUT=${NUM_ROLLOUT:-$(( (TRAIN_ROWS * TOTAL_EPOCHS + ROLLOUT_BATCH_SIZE - 1) / ROLLOUT_BATCH_SIZE ))}

# === ARGS arrays ===
COLOCATE_ARGS=()

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${MEGATRON_LOAD}"
    --save "${SAVE_DIR}/actor"
    --save-interval 200
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
    --custom-rm-path "${CUSTOM_RM_OVERRIDE:-miles.orbit.rewards.reward_router.reward_func}"
    --group-rm
    --code-rm-timeout-secs "${CODE_RM_TIMEOUT_SECS:-6}"
    --code-rm-max-tests "${CODE_RM_MAX_TESTS:-8}"
)
if [ -n "${JUDGE_BASE_URL}" ]; then
    ROLLOUT_ARGS+=( --judge-base-url "${JUDGE_BASE_URL}" )
fi
if [ -n "${TOOL_KEY:-}" ]; then
    ROLLOUT_ARGS+=( --tool-key "${TOOL_KEY}" )
fi

OPTIMIZER_ARGS=(
    --optimizer "${OPTIMIZER:-adam}"
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.999
)
# Inject extra optimizer flags (e.g. the muon-kimi preset in
# examples/optimizers/muon-kimi.env) without editing this launcher.
if [ -n "${EXTRA_OPTIMIZER_ARGS:-}" ]; then
    OPTIMIZER_ARGS+=( ${EXTRA_OPTIMIZER_ARGS} )
fi

RL_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.0
    --kl-loss-type k1
    --kl-coef 0.0
    --entropy-coef 0.0
    --eps-clip 4e-4
    --eps-clip-high 4e-4
    --gamma 1.0
    --lambd 1.0
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
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --sequence-parallel
)

# Per-domain eval: judge-equivalence accuracy on reasoning val + executor
# pass-rate on held-out code rows, both scored through the router (eval
# groups are singletons: judge/code routes stay meaningful, genrm would not).
if [ -n "${REASONING_VAL:-}" ] && [ -n "${CODE_VAL:-}" ]; then
    EVAL_ARGS=(
        --eval-interval "${EVAL_INTERVAL:-10}"
        --eval-prompt-data reasoning "${REASONING_VAL}" code "${CODE_VAL}"
        --n-samples-per-eval-prompt 1
        --eval-max-response-len 1024
        --eval-top-k 1
    )
else
    EVAL_ARGS=()
fi

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.60}"
    --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-1024}"
    --sglang-force-native-ops
    --sglang-attention-backend triton
    --sglang-sampling-backend pytorch
    --router-disable-circuit-breaker
)

MISC_ARGS=(
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

PEFT_ARGS=()

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
