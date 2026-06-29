#!/bin/bash
#
# Shared Qwen3-VL-8B Sokoban envpack recipe.
#
# Source this from a user-facing recipe after setting RECIPE_NAME and any
# dataset/node/VIT overrides. Human-facing config is grouped below; derived
# wiring is kept at the end.

set -euo pipefail

COMMON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_REPO=${MILES_REPO:-$(cd "$COMMON_DIR/../../.." && pwd)}
RECIPE_NAME=${RECIPE_NAME:-$(basename "${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}" .sh)}
ENVPACK_ADAPTER_DIR=${ENVPACK_ADAPTER_DIR:-"$MILES_REPO/miles_plugins/envpack_adapter"}
# shellcheck disable=SC1091
source "$ENVPACK_ADAPTER_DIR/recipes/common.sh"
envpack_resolve_repo

# ---------------------------------------------------------------------------
# Resource metadata
# ---------------------------------------------------------------------------
EXPERIMENT_NODES=${EXPERIMENT_NODES:-1}
EXPERIMENT_TIME=${EXPERIMENT_TIME:-72:00:00}

# ---------------------------------------------------------------------------
# Assets and dataset
# ---------------------------------------------------------------------------
HF_CACHE_DIR=${HF_CACHE_DIR:-/data/shared/hf_cache}

HF_MODEL_REPO=${MILES_SCRIPT_HF_MODEL_REPO:-"Qwen/Qwen3-VL-8B-Instruct"}
HF_DATASETS=()
HF_MODEL_DIR=${MILES_SCRIPT_HF_MODEL_DIR:-"$HF_CACHE_DIR/models/Qwen3-VL-8B-Instruct"}
MODEL_RECIPE=${MILES_SCRIPT_MODEL_RECIPE:-qwen3-8B.sh}
export VAGEN_THINK_TAG=${VAGEN_THINK_TAG:-thinking}
export MODEL_ARGS_ROTARY_BASE=${MODEL_ARGS_ROTARY_BASE:-5000000}

ENVPACK_DATASET_NAME=${ENVPACK_DATASET_NAME:-envpack-sokoban-main}
ENVPACK_BUILD_TARGET=${ENVPACK_BUILD_TARGET:-sokoban}
ENVPACK_SAVE_DIR=${ENVPACK_SAVE_DIR:-"${RUN_DIR:-/tmp/envpack-mvp/${RECIPE_NAME}}/traces"}
ENVPACK_DATA_ROOT=${ENVPACK_DATA_ROOT:-"$MILES_REPO/data/$ENVPACK_DATASET_NAME"}
ENVPACK_TRAIN_DATA=${ENVPACK_TRAIN_DATA:-"$ENVPACK_DATA_ROOT/train/samples.jsonl"}
ENVPACK_EVAL_DATA=${ENVPACK_EVAL_DATA:-"$ENVPACK_DATA_ROOT/eval/samples.jsonl"}
ENVPACK_EVAL_NAME=${ENVPACK_EVAL_NAME:-envpack_sokoban_val}
SAVE_INTERVAL=${SAVE_INTERVAL:-100}

_BUILD_HINT="Build it with: scripts/experiments/server_train/build-envpack-main.sh $ENVPACK_BUILD_TARGET"
envpack_require_dataset "$ENVPACK_TRAIN_DATA" "$ENVPACK_EVAL_DATA" "$_BUILD_HINT"

# ---------------------------------------------------------------------------
# Human-facing experiment config
# ---------------------------------------------------------------------------
SOKOBAN_ENV_ARGS=(
    render_style sprite      # sprite | tiny | raw_planes
    tiny_scale 16            # used only when render_style=tiny
    raw_plane_scale 16       # used only when render_style=raw_planes
)
SOKOBAN_RENDER_STYLE=${SOKOBAN_RENDER_STYLE:-$(envpack_recipe_arg_value render_style sprite "${SOKOBAN_ENV_ARGS[@]}")}
SOKOBAN_TINY_SCALE=${SOKOBAN_TINY_SCALE:-$(envpack_recipe_arg_value tiny_scale 16 "${SOKOBAN_ENV_ARGS[@]}")}
SOKOBAN_RAW_PLANE_SCALE=${SOKOBAN_RAW_PLANE_SCALE:-$(envpack_recipe_arg_value raw_plane_scale 16 "${SOKOBAN_ENV_ARGS[@]}")}
SOKOBAN_PROFILE=${SOKOBAN_PROFILE:-vision_free_think_local}
SOKOBAN_POOL_ID=${SOKOBAN_POOL_ID:-sokoban-vision}

TRAINING_SCHEDULE_ARGS=(
    --num-rollout             400
    --rollout-batch-size       32
    --n-samples-per-prompt      8
    --global-batch-size      auto
)

INTERACTION_BUDGET_ARGS=(
    --max-env-turns-per-sample    15
    --max-model-tokens-per-turn 512
    --rollout-max-context-len 16000
    --rollout-max-response-len 8192
)

ENABLE_DAPO=${ENABLE_DAPO:-0}
DAPO_ARGS=(
    --eps-clip-high 0.28
    --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
    --over-sampling-batch-size auto
)

CKPT_ARGS=(
    --hf-checkpoint  "$HF_MODEL_DIR"
    --load           "$HF_MODEL_DIR"
    --save           "$ENVPACK_SAVE_DIR"
    --save-interval  "$SAVE_INTERVAL"
)

MULTIMODAL_ARGS=(
    --multimodal-keys '{"image": "images"}'
)

PERF_ARGS=(
    --tensor-model-parallel-size 8
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

GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --kl-coef 0.00
    --entropy-coef 0.00
    --use-rollout-entropy
    --eps-clip 0.2
)

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.01
    --adam-beta1 0.9
    --adam-beta2 0.999
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.5
)

MISC_ARGS=(
    --colocate
    --attention-dropout 0.0
    --hidden-dropout    0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

BASE_RUN_NAME=${SLURM_JOB_NAME:-$RECIPE_NAME}
WANDB_RUN_PREFIX=${WANDB_RUN_PREFIX-new-http}
if [[ -z "$WANDB_RUN_PREFIX" ]]; then
    RUN_NAME="$BASE_RUN_NAME"
elif [[ "$BASE_RUN_NAME" == "$WANDB_RUN_PREFIX"-* ]]; then
    RUN_NAME="$BASE_RUN_NAME"
else
    RUN_NAME="$WANDB_RUN_PREFIX-$BASE_RUN_NAME"
fi

WANDB_ARGS=(
    --use-wandb
    --wandb-team    M3TRL
    --wandb-project vagen
    --wandb-group   "$RUN_NAME"
    --disable-wandb-random-suffix
)

EVAL_ARGS=(
    --eval-prompt-data         "$ENVPACK_EVAL_NAME" "$ENVPACK_EVAL_DATA"
    --eval-interval            20
    --n-samples-per-eval-prompt 1
    --eval-max-response-len    4096
)

FT_ARGS=(
    --use-fault-tolerance
    --rollout-health-check-interval 30
    --rollout-health-check-timeout  30
    --rollout-health-check-first-wait 60
)

LAYOUT_ARGS=(
    --actor-num-nodes        1
    --actor-num-gpus-per-node 8
    --rollout-num-gpus       8
)

# ---------------------------------------------------------------------------
# Derived wiring from the human-facing config above
# ---------------------------------------------------------------------------
# shellcheck disable=SC1090
source "$MILES_REPO/scripts/models/$MODEL_RECIPE"
MODEL_ARGS+=( --megatron-to-hf-mode bridge )

NUM_ROLLOUT=$(envpack_recipe_arg_value --num-rollout 400 "${TRAINING_SCHEDULE_ARGS[@]}")
ROLLOUT_BATCH_SIZE=$(envpack_recipe_arg_value --rollout-batch-size 32 "${TRAINING_SCHEDULE_ARGS[@]}")
N_SAMPLES_PER_PROMPT=$(envpack_recipe_arg_value --n-samples-per-prompt 8 "${TRAINING_SCHEDULE_ARGS[@]}")
GLOBAL_BATCH_SIZE=$(envpack_recipe_arg_value --global-batch-size auto "${TRAINING_SCHEDULE_ARGS[@]}")
if [[ "$GLOBAL_BATCH_SIZE" == "auto" ]]; then
    GLOBAL_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))
fi

MAX_ENV_TURNS_PER_SAMPLE=$(envpack_recipe_arg_value --max-env-turns-per-sample 15 "${INTERACTION_BUDGET_ARGS[@]}")
MAX_MODEL_TOKENS_PER_TURN=$(envpack_recipe_arg_value --max-model-tokens-per-turn 512 "${INTERACTION_BUDGET_ARGS[@]}")
ROLLOUT_MAX_CONTEXT_LEN=$(envpack_recipe_arg_value --rollout-max-context-len 16000 "${INTERACTION_BUDGET_ARGS[@]}")
ROLLOUT_MAX_RESPONSE_LEN=$(envpack_recipe_arg_value --rollout-max-response-len 8192 "${INTERACTION_BUDGET_ARGS[@]}")
N_SAMPLES_PER_EVAL_PROMPT=$(envpack_recipe_arg_value --n-samples-per-eval-prompt 1 "${EVAL_ARGS[@]}")

EFFECTIVE_ROLLOUT_BATCH_SIZE=$ROLLOUT_BATCH_SIZE
if [[ "$ENABLE_DAPO" == "1" ]]; then
    EFFECTIVE_ROLLOUT_BATCH_SIZE=$(envpack_recipe_arg_value --over-sampling-batch-size auto "${DAPO_ARGS[@]}")
    if [[ "$EFFECTIVE_ROLLOUT_BATCH_SIZE" == "auto" ]]; then
        EFFECTIVE_ROLLOUT_BATCH_SIZE=$((ROLLOUT_BATCH_SIZE * 2))
    fi
    GRPO_ARGS+=(
        --eps-clip-high "$(envpack_recipe_arg_value --eps-clip-high 0.28 "${DAPO_ARGS[@]}")"
        --dynamic-sampling-filter-path "$(envpack_recipe_arg_value --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "${DAPO_ARGS[@]}")"
        --over-sampling-batch-size "$EFFECTIVE_ROLLOUT_BATCH_SIZE"
    )
fi

export ENVPACK_SOKOBAN_RENDER_STYLE="$SOKOBAN_RENDER_STYLE"
export ENVPACK_SOKOBAN_TINY_SCALE="$SOKOBAN_TINY_SCALE"
export ENVPACK_SOKOBAN_RAW_PLANE_SCALE="$SOKOBAN_RAW_PLANE_SCALE"

envpack_prepare_adapter_config sokoban "$SOKOBAN_PROFILE" "$SOKOBAN_POOL_ID" "$MAX_ENV_TURNS_PER_SAMPLE" "$MAX_MODEL_TOKENS_PER_TURN"
envpack_set_rollout_args

MILES_ARGS=(
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
