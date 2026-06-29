#!/bin/bash
#
# Qwen3-VL-8B Sokoban mix12, 2 nodes, ViT ON.
#
# mix12 = concatenated 6x6 1box + 6x6 2box balanced datasets.
#
# Submit:
#   JOB_NAME=rv2-8b-viton-mix12 WANDB_RUN_PREFIX= TIME=72:00:00 \
#   WANDB_INIT_TIMEOUT=300 NODES=2 ENVPACK_SERVER_NODE_COUNT=1 MILES_ENV_NAME=miles_imp \
#   bash scripts/slurm/submit.sh \
#     server_train/sokoban_mix12/envpack-sokoban-mix12-qwen3vl8b-remote-2node

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RECIPE_NAME=$(basename "${BASH_SOURCE[0]}" .sh)

EXPERIMENT_NODES=${EXPERIMENT_NODES:-2}
export ENVPACK_SERVER_NODE_COUNT=${ENVPACK_SERVER_NODE_COUNT:-1}
export ENVPACK_SERVER_LOCAL=0
export ENVPACK_DATASET_NAME=${ENVPACK_DATASET_NAME:-envpack-sokoban-mix12}
export ENVPACK_BUILD_TARGET=${ENVPACK_BUILD_TARGET:-sokoban_mix12}
export ENVPACK_EVAL_NAME=${ENVPACK_EVAL_NAME:-envpack_sokoban_mix12_val}

# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_qwen3vl8b_common.sh"

# ViT ON: deliberately no --freeze-vision-model.
