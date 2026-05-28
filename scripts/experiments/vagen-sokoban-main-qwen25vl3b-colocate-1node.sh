#!/bin/bash
#
# Sokoban-main GRPO on Qwen2.5-VL-3B-Instruct, 1 node × 8 GPUs colocated,
# TP=4 + sequence-parallel.
#
# Reads precomputed train/eval jsonl built by
# examples/vagen/scripts/sokoban-main.sh. The Sokoban yaml deliberately
# overrides VAGEN-main's `wm` prompt-format to `free_think` — see
# examples/vagen/configs/sokoban_train_env.yaml.
#
# See examples/vagen/docs/launch_recipe.md for the VAGEN→miles knob mapping,
# eval cadence, perf-args rationale, deliberate deviations (no save_freq,
# no filter unless MILES_VAGEN_DAPO=1), and scaling knobs.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_REPO=${MILES_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}
RECIPE_NAME=$(basename "${BASH_SOURCE[0]}" .sh)

# ---------------------------------------------------------------------------
# Resource metadata
# ---------------------------------------------------------------------------
EXPERIMENT_NODES=1
EXPERIMENT_TIME=48:00:00

# ---------------------------------------------------------------------------
# Asset metadata
# ---------------------------------------------------------------------------
HF_CACHE_DIR=${HF_CACHE_DIR:-/data/shared/hf_cache}

HF_MODEL_REPO="Qwen/Qwen2.5-VL-3B-Instruct"
HF_DATASETS=()    # VAGEN data is a precomputed jsonl on disk.
HF_MODEL_DIR="$HF_CACHE_DIR/models/Qwen2.5-VL-3B-Instruct"

# Traces under slurm: $RUN_DIR/traces. Standalone: /tmp/vagen-mvp/$RECIPE_NAME/traces.
VAGEN_SAVE_DIR=${VAGEN_SAVE_DIR:-"${RUN_DIR:-/tmp/vagen-mvp/${RECIPE_NAME}}/traces"}

# Dataset: precomputed jsonl per split, built by
# examples/vagen/scripts/sokoban-main.sh. See examples/vagen/docs/dataset.md.
VAGEN_DATASET_NAME=${VAGEN_DATASET_NAME:-sokoban-main}
VAGEN_DATA_ROOT=${VAGEN_DATA_ROOT:-"$MILES_REPO/data/$VAGEN_DATASET_NAME"}
VAGEN_TRAIN_DATA=${VAGEN_TRAIN_DATA:-"$VAGEN_DATA_ROOT/train/samples.jsonl"}
VAGEN_EVAL_DATA=${VAGEN_EVAL_DATA:-"$VAGEN_DATA_ROOT/eval/samples.jsonl"}
VAGEN_EVAL_NAME=${VAGEN_EVAL_NAME:-sokoban_val}

_BUILD_HINT="Build it with: examples/vagen/scripts/${VAGEN_DATASET_NAME}.sh"
if [[ ! -s "$VAGEN_TRAIN_DATA" ]]; then
    echo "error: missing train data: $VAGEN_TRAIN_DATA" >&2
    echo "       $_BUILD_HINT" >&2
    exit 1
fi
if [[ ! -s "$VAGEN_EVAL_DATA" ]]; then
    echo "error: missing eval data: $VAGEN_EVAL_DATA" >&2
    echo "       $_BUILD_HINT" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# train.py args
# ---------------------------------------------------------------------------
# shellcheck disable=SC1090
source "$MILES_REPO/scripts/models/qwen2.5-3B.sh"
MODEL_ARGS+=( --megatron-to-hf-mode bridge )

RUN_NAME=${SLURM_JOB_NAME:-$RECIPE_NAME}

NUM_ROLLOUT=${MILES_SCRIPT_NUM_ROLLOUT:-400}
ROLLOUT_BATCH_SIZE=${MILES_SCRIPT_ROLLOUT_BATCH_SIZE:-32}
N_SAMPLES_PER_PROMPT=${MILES_SCRIPT_N_SAMPLES_PER_PROMPT:-8}
GLOBAL_BATCH_SIZE=${MILES_SCRIPT_GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}

CKPT_ARGS=(
    --hf-checkpoint  "$HF_MODEL_DIR"
    --load           "$HF_MODEL_DIR"
    --save           "$VAGEN_SAVE_DIR"
    --save-interval  1000000
)

MULTIMODAL_ARGS=(
    --multimodal-keys '{"image": "images"}'
)

# --seed 0 = label-level alignment with VAGEN main (see docs/launch_recipe.md).
ROLLOUT_ARGS=(
    --data-source-path examples.vagen.data_source.VagenEnvSpecDataSource
    --prompt-data       "$VAGEN_TRAIN_DATA"
    --custom-generate-function-path examples.vagen.rollout.generate
    --rollout-all-samples-process-path examples.vagen.debug_dump.dump_samples
    --rollout-shuffle
    --seed                    0
    --num-rollout            "$NUM_ROLLOUT"
    --rollout-batch-size      "$ROLLOUT_BATCH_SIZE"
    --n-samples-per-prompt    "$N_SAMPLES_PER_PROMPT"
    --rollout-max-context-len 10000
    --rollout-max-response-len 4096
    --global-batch-size       "$GLOBAL_BATCH_SIZE"
)

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

GRPO_ARGS=(
    --advantage-estimator grpo
    --kl-loss-coef 0.00
    --kl-loss-type low_var_kl
    --kl-coef 0.00
    --entropy-coef 0.00
    --eps-clip 0.2
)

if [[ "${MILES_VAGEN_DAPO:-0}" == "1" ]]; then
    OVER_SAMPLING_BATCH_SIZE=${MILES_SCRIPT_OVER_SAMPLING_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * 2))}
    GRPO_ARGS+=(
        --eps-clip-high 0.28
        --dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
        --over-sampling-batch-size "$OVER_SAMPLING_BATCH_SIZE"
    )
fi

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
    --sglang-mem-fraction-static  0.6
)

MISC_ARGS=(
    --colocate
    --attention-dropout 0.0
    --hidden-dropout    0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project miles-imp
    --wandb-group   "$RUN_NAME"
    --disable-wandb-random-suffix
)

# Eval cadence mirrors VAGEN main (see docs/launch_recipe.md).
# NOTE: deliberately no --eval-max-prompt-len; prompt is rebuilt from
# env.reset inside our generate(), so filter on the placeholder is moot.
EVAL_ARGS=(
    --eval-prompt-data         "$VAGEN_EVAL_NAME" "$VAGEN_EVAL_DATA"
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
