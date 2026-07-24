#!/bin/bash
# Milestone 06b: matched 20-step synchronous Geo3K multi-turn hybrid OPD gate.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-20}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-06b-geo3k-mt-hybrid-rkld1-dagger0p5-gate}

# Keep every model, data, multi-turn, objective, scoring, optimizer, and
# parallel-layout argument identical to 06a. Launch only after 06a passes.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/06a-geo3k-multiturn-hybrid-smoke.sh"
