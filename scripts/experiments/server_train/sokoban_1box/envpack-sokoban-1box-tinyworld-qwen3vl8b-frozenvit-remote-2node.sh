#!/bin/bash
#
# Qwen3-VL-8B Sokoban 1box, TinyWorld render, 2 nodes, ViT FROZEN.
#
# Reuses the 1box Sokoban puzzle dataset and changes only the runtime visual
# render style from sprite to TinyWorld.
#
# Submit:
#   JOB_NAME=rv2-8b-vitoff-tiny-1box WANDB_RUN_PREFIX=new-http-tinyworld TIME=72:00:00 \
#   WANDB_INIT_TIMEOUT=300 NODES=2 ENVPACK_SERVER_NODE_COUNT=1 ORBIT_ENV_NAME=orbit \
#   bash scripts/slurm/submit.sh \
#     server_train/sokoban_1box/envpack-sokoban-1box-tinyworld-qwen3vl8b-frozenvit-remote-2node

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RECIPE_NAME=$(basename "${BASH_SOURCE[0]}" .sh)

EXPERIMENT_NODES=${EXPERIMENT_NODES:-2}
export ENVPACK_SERVER_NODE_COUNT=${ENVPACK_SERVER_NODE_COUNT:-1}
export ENVPACK_SERVER_LOCAL=0
export SOKOBAN_RENDER_STYLE=${SOKOBAN_RENDER_STYLE:-tiny}
WANDB_RUN_PREFIX=${WANDB_RUN_PREFIX:-new-http-tinyworld}

# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_qwen3vl8b_common.sh"

MODEL_ARGS+=( --freeze-vision-model )
ORBIT_ARGS+=( --freeze-vision-model )
