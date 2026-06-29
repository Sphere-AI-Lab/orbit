#!/bin/bash
#
# Envpack FrozenLake-main GRPO with one Miles/SGLang node and one envpack
# server node in the same Slurm job.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
RECIPE_NAME=$(basename "${BASH_SOURCE[0]}" .sh)

EXPERIMENT_NODES=${EXPERIMENT_NODES:-2}
export ENVPACK_SERVER_NODE_COUNT=${ENVPACK_SERVER_NODE_COUNT:-1}
export ENVPACK_SERVER_LOCAL=0

# The base recipe keeps Miles actor/rollout layout at one 8-GPU node. The
# launcher reserves the final allocated node for envpack and excludes it from
# Ray, then fills ENVPACK_SERVER_URL before sourcing this recipe inside Slurm.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/envpack-frozenlake-main-qwen3vl2b-colocate-1node.sh"
