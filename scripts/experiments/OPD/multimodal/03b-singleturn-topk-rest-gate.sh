#!/bin/bash
# Milestone 03b: matched 20-step single-turn multimodal Top-K + Rest gate.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-20}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-03b-top2-rest-gate}

# Keep every model, data, objective coefficient, and parallel-layout argument
# identical to 03a; only the run length and W&B identity change.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/03a-singleturn-topk-rest-smoke.sh"
