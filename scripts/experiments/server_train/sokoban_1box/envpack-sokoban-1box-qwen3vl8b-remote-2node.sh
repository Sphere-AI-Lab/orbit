#!/bin/bash
#
# Qwen3-VL-8B Sokoban 1box, 2 nodes, ViT ON.
#
# Submit:
#   JOB_NAME=rv2-8b-viton-1box WANDB_RUN_PREFIX= TIME=72:00:00 \
#   WANDB_INIT_TIMEOUT=300 NODES=2 ENVPACK_SERVER_NODE_COUNT=1 ORBIT_ENV_NAME=orbit \
#   bash scripts/slurm/submit.sh \
#     server_train/sokoban_1box/envpack-sokoban-1box-qwen3vl8b-remote-2node

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RECIPE_NAME=$(basename "${BASH_SOURCE[0]}" .sh)

EXPERIMENT_NODES=${EXPERIMENT_NODES:-2}
export ENVPACK_SERVER_NODE_COUNT=${ENVPACK_SERVER_NODE_COUNT:-1}
export ENVPACK_SERVER_LOCAL=0

# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_qwen3vl8b_common.sh"

# ViT ON: deliberately no --freeze-vision-model.
