#!/bin/bash
# Milestone 08d: matched 20-step fully-async Geo3K multi-turn OPD + task-RL gate.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-20}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-08d-geo3k-mt-opd-rl-async-pf2-gate}

# Keep every 08c model, data, objective, scoring, parallel-layout and async
# argument fixed. Launch only after both 08a and 08c five-step smokes pass.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/08c-geo3k-multiturn-opd-rl-fully-async-smoke.sh"
