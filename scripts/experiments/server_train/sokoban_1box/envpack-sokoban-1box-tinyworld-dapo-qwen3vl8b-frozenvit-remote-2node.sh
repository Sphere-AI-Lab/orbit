#!/bin/bash
#
# Qwen3-VL-8B Sokoban 1box, TinyWorld render, DAPO dynamic sampling,
# 2 nodes, ViT FROZEN.
#
# Submit:
#   JOB_NAME=rv2-8b-vitoff-tiny-dapo-1box WANDB_RUN_PREFIX=new-http-tinyworld-dapo TIME=72:00:00 \
#   WANDB_INIT_TIMEOUT=300 NODES=2 ENVPACK_SERVER_NODE_COUNT=1 MILES_ENV_NAME=miles_imp \
#   bash scripts/slurm/submit.sh \
#     server_train/sokoban_1box/envpack-sokoban-1box-tinyworld-dapo-qwen3vl8b-frozenvit-remote-2node

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RECIPE_NAME=$(basename "${BASH_SOURCE[0]}" .sh)

EXPERIMENT_NODES=${EXPERIMENT_NODES:-2}
export ENVPACK_SERVER_NODE_COUNT=${ENVPACK_SERVER_NODE_COUNT:-1}
export ENVPACK_SERVER_LOCAL=0
export ENVPACK_DATASET_NAME=${ENVPACK_DATASET_NAME:-envpack-sokoban-full110}
export ENVPACK_BUILD_TARGET=${ENVPACK_BUILD_TARGET:-sokoban_full110}
export ENVPACK_EVAL_NAME=${ENVPACK_EVAL_NAME:-envpack_sokoban_full110_val}
export SOKOBAN_RENDER_STYLE=${SOKOBAN_RENDER_STYLE:-tiny}
export SOKOBAN_CURRICULUM_ENABLED=${SOKOBAN_CURRICULUM_ENABLED:-1}
export ENABLE_DAPO=1
export DAPO_OVER_SAMPLING_BATCH_SIZE=${DAPO_OVER_SAMPLING_BATCH_SIZE:-32}
export SGLANG_SERVER_CONCURRENCY=${SGLANG_SERVER_CONCURRENCY:-512}
WANDB_RUN_PREFIX=${WANDB_RUN_PREFIX:-new-http-tinyworld-dapo}

# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_qwen3vl8b_common.sh"

MODEL_ARGS+=( --freeze-vision-model )
MILES_ARGS+=( --freeze-vision-model )
