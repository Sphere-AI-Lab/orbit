#!/bin/bash
# Milestone 08b: matched 20-step synchronous Geo3K multi-turn OPD + task-RL reference.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-20}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-08b-geo3k-mt-opd-rl-sync-reference}

# Keep model, data, objective coefficients, scoring, optimizer and layout
# identical to 08a. Launch only after the five-step composition smoke passes.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/08a-geo3k-multiturn-opd-rl-sync-smoke.sh"
