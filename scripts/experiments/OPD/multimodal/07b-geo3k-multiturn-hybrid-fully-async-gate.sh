#!/bin/bash
# Milestone 07b: matched 20-step Geo3K multi-turn hybrid OPD fully-async gate.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-20}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-07b-geo3k-mt-hybrid-fully-async-gate}

# Keep every model, data, objective, scoring, parallel-layout and async-window
# argument identical to 07a. Launch only after the five-step smoke passes.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/07a-geo3k-multiturn-hybrid-fully-async-smoke.sh"
