#!/bin/bash
#
# Qwen3-VL-8B Sokoban 1box, 1 node same-node HTTP, ViT FROZEN.
#
# This is the local HTTP smoke/ablation recipe. The envpack server runs on the
# same node as Orbit; the two-node recipes move the server to a reserved env node.
#
# Submit:
#   JOB_NAME=rv2-8b-vitoff-1box-1node WANDB_RUN_PREFIX= TIME=72:00:00 \
#   WANDB_INIT_TIMEOUT=300 NODES=1 ORBIT_ENV_NAME=orbit \
#   bash scripts/slurm/submit.sh \
#     server_train/sokoban_1box/envpack-sokoban-1box-qwen3vl8b-frozenvit-colocate-1node

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RECIPE_NAME=$(basename "${BASH_SOURCE[0]}" .sh)

EXPERIMENT_NODES=${EXPERIMENT_NODES:-1}
export ENVPACK_SERVER_NODE_COUNT=${ENVPACK_SERVER_NODE_COUNT:-0}
export ENVPACK_SERVER_LOCAL=${ENVPACK_SERVER_LOCAL:-1}

# shellcheck disable=SC1091
source "$SCRIPT_DIR/../_qwen3vl8b_common.sh"

MODEL_ARGS+=( --freeze-vision-model )
ORBIT_ARGS+=( --freeze-vision-model )
