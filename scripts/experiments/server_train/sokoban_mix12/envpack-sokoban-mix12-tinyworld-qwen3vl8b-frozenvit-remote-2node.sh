#!/bin/bash
#
# Qwen3-VL-8B Sokoban mix12, TinyWorld render, 2 nodes, ViT FROZEN.
#
# mix12 = concatenated 6x6 1box + 6x6 2box balanced datasets. This recipe
# reuses those puzzle rows and changes only the runtime visual render style from
# sprite to TinyWorld.
#
# Submit:
#   JOB_NAME=rv2-8b-vitoff-tiny-mix12 WANDB_RUN_PREFIX=new-http-tinyworld TIME=72:00:00 \
#   WANDB_INIT_TIMEOUT=300 NODES=2 ENVPACK_SERVER_NODE_COUNT=1 ORBIT_ENV_NAME=orbit \
#   bash scripts/slurm/submit.sh \
#     server_train/sokoban_mix12/envpack-sokoban-mix12-tinyworld-qwen3vl8b-frozenvit-remote-2node

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RECIPE_NAME=$(basename "${BASH_SOURCE[0]}" .sh)

EXPERIMENT_NODES=${EXPERIMENT_NODES:-2}
export ENVPACK_SERVER_NODE_COUNT=${ENVPACK_SERVER_NODE_COUNT:-1}
export ENVPACK_SERVER_LOCAL=0
export ENVPACK_DATASET_NAME=${ENVPACK_DATASET_NAME:-envpack-sokoban-mix12}
export ENVPACK_BUILD_TARGET=${ENVPACK_BUILD_TARGET:-sokoban_mix12}
export ENVPACK_EVAL_NAME=${ENVPACK_EVAL_NAME:-envpack_sokoban_mix12_val}
export SOKOBAN_RENDER_STYLE=${SOKOBAN_RENDER_STYLE:-tiny}
WANDB_RUN_PREFIX=${WANDB_RUN_PREFIX:-new-http-tinyworld}

# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_qwen3vl8b_common.sh"

MODEL_ARGS+=( --freeze-vision-model )
ORBIT_ARGS+=( --freeze-vision-model )
