#!/bin/bash
#
# Qwen3-VL-8B Sokoban 1box, 2 nodes, ViT FROZEN.
#
# Submit:
#   JOB_NAME=rv2-8b-vitoff-1box WANDB_RUN_PREFIX= TIME=72:00:00 \
#   WANDB_INIT_TIMEOUT=300 NODES=2 ENVPACK_SERVER_NODE_COUNT=1 MILES_ENV_NAME=miles_imp \
#   bash scripts/slurm/submit.sh \
#     server_train/sokoban_1box/envpack-sokoban-1box-qwen3vl8b-frozenvit-remote-2node

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RECIPE_NAME=$(basename "${BASH_SOURCE[0]}" .sh)

EXPERIMENT_NODES=${EXPERIMENT_NODES:-2}
export ENVPACK_SERVER_NODE_COUNT=${ENVPACK_SERVER_NODE_COUNT:-1}
export ENVPACK_SERVER_LOCAL=0

# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_qwen3vl8b_common.sh"

MODEL_ARGS+=( --freeze-vision-model )
MILES_ARGS+=( --freeze-vision-model )
