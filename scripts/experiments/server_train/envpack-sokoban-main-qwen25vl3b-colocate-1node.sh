#!/bin/bash
#
# Envpack Sokoban-main GRPO on Qwen2.5-VL-3B-Instruct, 1 node x 8 GPUs
# colocated, TP=4 + sequence-parallel.
#
# The training/eval JSONL rows carry metadata.envpack only. Orbit still owns
# tokenization, SGLang, logprobs, and Sample assembly; envpack owns reset,
# step, finalize, parser, rubric, and image observations.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ORBIT_REPO=${ORBIT_REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
RECIPE_NAME=${RECIPE_NAME:-$(basename "${BASH_SOURCE[0]}" .sh)}
ENVPACK_ADAPTER_DIR=${ENVPACK_ADAPTER_DIR:-"$ORBIT_REPO/orbit_plugins/envpack_adapter"}
# shellcheck disable=SC1091
source "$ENVPACK_ADAPTER_DIR/recipes/common.sh"
envpack_resolve_repo

# ---------------------------------------------------------------------------
# Resource metadata
# ---------------------------------------------------------------------------
EXPERIMENT_NODES=${EXPERIMENT_NODES:-1}
EXPERIMENT_TIME=${EXPERIMENT_TIME:-48:00:00}

# ---------------------------------------------------------------------------
# Asset metadata
# ---------------------------------------------------------------------------
HF_CACHE_DIR=${HF_CACHE_DIR:-/data/shared/hf_cache}

HF_MODEL_REPO=${ORBIT_SCRIPT_HF_MODEL_REPO:-"Qwen/Qwen2.5-VL-3B-Instruct"}
HF_DATASETS=()
HF_MODEL_DIR=${ORBIT_SCRIPT_HF_MODEL_DIR:-"$HF_CACHE_DIR/models/Qwen2.5-VL-3B-Instruct"}

ENVPACK_SAVE_DIR=${ENVPACK_SAVE_DIR:-"${RUN_DIR:-/tmp/envpack-mvp/${RECIPE_NAME}}/traces"}

# Dataset: build with scripts/experiments/server_train/build-envpack-main.sh.
ENVPACK_DATASET_NAME=${ENVPACK_DATASET_NAME:-envpack-sokoban-main}
ENVPACK_DATA_ROOT=${ENVPACK_DATA_ROOT:-"$ORBIT_REPO/data/$ENVPACK_DATASET_NAME"}
ENVPACK_TRAIN_DATA=${ENVPACK_TRAIN_DATA:-"$ENVPACK_DATA_ROOT/train/samples.jsonl"}
ENVPACK_EVAL_DATA=${ENVPACK_EVAL_DATA:-"$ENVPACK_DATA_ROOT/eval/samples.jsonl"}
ENVPACK_EVAL_NAME=${ENVPACK_EVAL_NAME:-envpack_sokoban_val}

_BUILD_HINT="Build it with: scripts/experiments/server_train/build-envpack-main.sh sokoban"
envpack_require_dataset "$ENVPACK_TRAIN_DATA" "$ENVPACK_EVAL_DATA" "$_BUILD_HINT"

# ---------------------------------------------------------------------------
# train.py args
# ---------------------------------------------------------------------------
MODEL_RECIPE=${ORBIT_SCRIPT_MODEL_RECIPE:-qwen2.5-3B.sh}
# shellcheck disable=SC1090
source "$ORBIT_REPO/scripts/models/$MODEL_RECIPE"
MODEL_ARGS+=( --megatron-to-hf-mode bridge )

BASE_RUN_NAME=${SLURM_JOB_NAME:-$RECIPE_NAME}
if [[ "$BASE_RUN_NAME" == http-* ]]; then
    RUN_NAME="$BASE_RUN_NAME"
else
    RUN_NAME="http-$BASE_RUN_NAME"
fi

# ---------------------------------------------------------------------------
# Human-facing experiment config
# ---------------------------------------------------------------------------
# Edit this block for normal training changes. The wiring below maps these
# readable values into Orbit args and envpack adapter YAML.
TRAINING_SCHEDULE_ARGS=(
    --num-rollout             400
    --rollout-batch-size       32
    --n-samples-per-prompt      8
    --global-batch-size      auto
)

# Per-sample multi-turn interaction budget.
INTERACTION_BUDGET_ARGS=(
    --max-env-turns-per-sample    5
    --max-model-tokens-per-turn 512
    --rollout-max-context-len 10000
    --rollout-max-response-len 4096
)

# Optional DAPO/rejection-sampling controls.
ENABLE_DAPO=0
DAPO_ARGS=(
    --eps-clip-high 0.28
    --dynamic-sampling-filter-path orbit.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
    --over-sampling-batch-size auto
)

# Checkpoint and save cadence.
CKPT_ARGS=(
    --hf-checkpoint  "$HF_MODEL_DIR"
    --load           "$HF_MODEL_DIR"
    --save           "$ENVPACK_SAVE_DIR"
    --save-interval  1000000
)

# Multimodal processor wiring.
MULTIMODAL_ARGS=(
    --multimodal-keys '{"image": "images"}'
)

# Model parallelism and memory/performance settings.
PERF_ARGS=(
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --qkv-format bshd
    --micro-batch-size 1
)

# RL objective.
GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --kl-coef 0.00
    --entropy-coef 0.00
    --eps-clip 0.2
)

# Optimizer.
OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.999
)

# SGLang rollout engine.
SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static  0.6
)

# Miscellaneous trainer/runtime flags.
MISC_ARGS=(
    --colocate
    --attention-dropout 0.0
    --hidden-dropout    0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

# Tracking.
WANDB_ARGS=(
    --use-wandb
    --wandb-team    M3TRL
    --wandb-project vagen
    --wandb-group   "$RUN_NAME"
    --disable-wandb-random-suffix
)

# Evaluation.
EVAL_ARGS=(
    --eval-prompt-data         "$ENVPACK_EVAL_NAME" "$ENVPACK_EVAL_DATA"
    --eval-interval            20
    --n-samples-per-eval-prompt 1
    --eval-max-response-len    4096
)

# Orbit fault tolerance.
FT_ARGS=(
    --use-fault-tolerance
    --rollout-health-check-interval 30
    --rollout-health-check-timeout  30
    --rollout-health-check-first-wait 60
)

# Node/GPU ownership for Orbit. Remote envpack server nodes are added by the
# thin 2-node wrapper and excluded from Ray by the Slurm launcher.
LAYOUT_ARGS=(
    --actor-num-nodes        1
    --actor-num-gpus-per-node 8
    --rollout-num-gpus       8
)

# ---------------------------------------------------------------------------
# Derived wiring from the human-facing config above
# ---------------------------------------------------------------------------
NUM_ROLLOUT=$(envpack_recipe_arg_value --num-rollout 400 "${TRAINING_SCHEDULE_ARGS[@]}")
ROLLOUT_BATCH_SIZE=$(envpack_recipe_arg_value --rollout-batch-size 32 "${TRAINING_SCHEDULE_ARGS[@]}")
N_SAMPLES_PER_PROMPT=$(envpack_recipe_arg_value --n-samples-per-prompt 8 "${TRAINING_SCHEDULE_ARGS[@]}")
GLOBAL_BATCH_SIZE=$(envpack_recipe_arg_value --global-batch-size auto "${TRAINING_SCHEDULE_ARGS[@]}")
if [[ "$GLOBAL_BATCH_SIZE" == "auto" ]]; then
    GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
fi

MAX_ENV_TURNS_PER_SAMPLE=$(envpack_recipe_arg_value --max-env-turns-per-sample 5 "${INTERACTION_BUDGET_ARGS[@]}")
MAX_MODEL_TOKENS_PER_TURN=$(envpack_recipe_arg_value --max-model-tokens-per-turn 512 "${INTERACTION_BUDGET_ARGS[@]}")
ROLLOUT_MAX_CONTEXT_LEN=$(envpack_recipe_arg_value --rollout-max-context-len 10000 "${INTERACTION_BUDGET_ARGS[@]}")
ROLLOUT_MAX_RESPONSE_LEN=$(envpack_recipe_arg_value --rollout-max-response-len 4096 "${INTERACTION_BUDGET_ARGS[@]}")

EFFECTIVE_ROLLOUT_BATCH_SIZE=$ROLLOUT_BATCH_SIZE
if [[ "$ENABLE_DAPO" == "1" ]]; then
    EFFECTIVE_ROLLOUT_BATCH_SIZE=$(envpack_recipe_arg_value --over-sampling-batch-size auto "${DAPO_ARGS[@]}")
    if [[ "$EFFECTIVE_ROLLOUT_BATCH_SIZE" == "auto" ]]; then
        EFFECTIVE_ROLLOUT_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * 2))
    fi
    GRPO_ARGS+=(
        --eps-clip-high "$(envpack_recipe_arg_value --eps-clip-high 0.28 "${DAPO_ARGS[@]}")"
        --dynamic-sampling-filter-path "$(envpack_recipe_arg_value --dynamic-sampling-filter-path orbit.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "${DAPO_ARGS[@]}")"
        --over-sampling-batch-size "$EFFECTIVE_ROLLOUT_BATCH_SIZE"
    )
fi

envpack_prepare_adapter_config sokoban vision_free_think_local sokoban-vision 5 512
envpack_set_rollout_args

ORBIT_ARGS=(
    "${LAYOUT_ARGS[@]}"
    "${MODEL_ARGS[@]}"
    "${CKPT_ARGS[@]}"
    "${MULTIMODAL_ARGS[@]}"
    "${ROLLOUT_ARGS[@]}"
    "${OPTIMIZER_ARGS[@]}"
    "${GRPO_ARGS[@]}"
    "${PERF_ARGS[@]}"
    "${SGLANG_ARGS[@]}"
    "${MISC_ARGS[@]}"
    "${EVAL_ARGS[@]}"
    "${WANDB_ARGS[@]}"
    "${FT_ARGS[@]}"
)
