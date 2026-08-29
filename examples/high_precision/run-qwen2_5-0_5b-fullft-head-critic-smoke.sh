#!/usr/bin/env bash
# Qwen2.5-3B-Instruct BF16 FULL finetuning with GRPO on the OpenR1 math set.
# Companion to the PPO critic-comparison benchmark: identical data, schedule,
# sampling, clipping, optimizer, seed, and evaluation matrix — only the two
# factor under study changes: full FT (no PEFT adapter) with the benchmark's
# exact PPO + separate full-critic recipe. Topology: 1 actor + 1 critic +
# 2 rollout GPUs (the full-critic-controlled layout).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"
source "${ORBIT_ROOT}/scripts/lib/common.sh"

# === Recipe identity ===
SEED="${SEED:-1234}"
LAUNCHER_NAME=run_qwen25_05b_fullft_head_critic_smoke_seed${SEED}
WANDB_PROJECT=${WANDB_PROJECT:-orbit-ppo-critic-compare}
WANDB_GROUP=${WANDB_GROUP:-${LAUNCHER_NAME}}
PRECISION_PROFILE=bf16
ORBIT_ENTRYPOINT="${ORBIT_ENTRYPOINT:-${ORBIT_ROOT}/train.py}"
RUN_LOG="${ORBIT_ROOT}/logs/${LAUNCHER_NAME}_$(date +%Y%m%d_%H%M%S).log"

# === Paths ===
: "${HF_CKPT:?set HF_CKPT to the Qwen2.5-3B-Instruct Hugging Face checkpoint path}"
: "${MEGATRON_LOAD:?set MEGATRON_LOAD to the Megatron torch_dist checkpoint path}"
: "${TRAIN_JSONL:?set TRAIN_JSONL to an OpenR1-style math JSONL path}"
SAVE_ROOT="${SAVE_ROOT:-${ORBIT_ROOT}/orbit_ckpts/fullft_ppo}"
SAVE_DIR="${SAVE_DIR:-${SAVE_ROOT}/Qwen2.5-0.5B_fullft_head_critic_smoke_seed${SEED}}"

# Match the critic benchmark's reward-verification budget. The scorer default
# is 10s; under the eval burst's CPU-parallel grading that deflates Math500
# pass@1 by ~16 points via verification timeouts (identical generations,
# stricter grading). The benchmark recipe exports 60 and manifests it.
export ORBIT_PEFT_ARENA_REWARD_TIMEOUT_S="${ORBIT_PEFT_ARENA_REWARD_TIMEOUT_S:-60}"

# === Resources: 1 actor + 3 rollout (no critic) ===
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-2}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
export PYTHONHASHSEED="${SEED}"

# === Model args ===
source "${ORBIT_ROOT}/orbit_plugins/model_args/qwen2.5-0.5B.sh"   # provides MODEL_ARGS=(...)

# === Training schedule (matches ppo_critic_compare_common.sh benchmark mode) ===
NUM_ROLLOUT="${NUM_ROLLOUT:-2}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-2}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
ROLLOUT_MAX_RESPONSE_LEN="${ROLLOUT_MAX_RESPONSE_LEN:-128}"
EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-1024}"
SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
EVAL_INTERVAL="${EVAL_INTERVAL:-25}"

COLOCATE_ARGS=()

CKPT_ARGS=(
    --hf-checkpoint "${HF_CKPT}"
    --load "${MEGATRON_LOAD}"
    --ckpt-format torch_dist
    --save "${SAVE_DIR}/actor"
    --critic-save "${SAVE_DIR}/critic"
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
    --rollout-seed "${SEED}"
    --rm-type custom
    --num-rollout "${NUM_ROLLOUT}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
    --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
    --rollout-max-response-len "${ROLLOUT_MAX_RESPONSE_LEN}"
    --rollout-temperature 1.0
    --rollout-top-p 1.0
    --rollout-top-k -1
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --custom-rm-path orbit.peft.rewards.peft_arena_reward.peft_arena_reward
    --reward-key score
    --eval-reward-key score
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --critic-lr 1e-5
    --lr-decay-style constant
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.999
)

RL_ARGS=(
    --advantage-estimator ppo
    --critic-mode head
    --kl-loss-coef 0.0
    --kl-loss-type k1
    --kl-coef 0.0
    --entropy-coef 0.0
    --eps-clip 0.2
    --eps-clip-high 0.28
    --value-clip 0.2
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

EVAL_ARGS=()

SGLANG_ARGS=(
    --num-gpus-per-node 4
    --rollout-num-gpus-per-engine 1
    --rollout-num-gpus "${ROLLOUT_NUM_GPUS}"
    --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC:-0.60}"
    --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS:-1024}"
    --sglang-enable-deterministic-inference
    --sglang-force-native-ops
    --sglang-attention-backend triton
    --sglang-sampling-backend pytorch
    --router-disable-circuit-breaker
    --sglang-router-policy round_robin
    # Same rationale as the critic benchmark: sglang v0.5.16's prefill CUDA
    # graph is disabled for parity with the benchmark engine config.
    --sglang-cuda-graph-backend-prefill disabled
)

MISC_ARGS=(
    --seed "${SEED}"
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

# Full finetuning: no PEFT adapter (empty array satisfies the launcher contract).
PEFT_ARGS=()

source "${ORBIT_ROOT}/scripts/lib/launcher.sh"
