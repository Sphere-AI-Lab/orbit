#!/bin/bash
# Milestone 04b: matched 20-step synchronous Geo3K multi-turn sampled-RKLD gate.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-20}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-04b-geo3k-mt-rkld-gate}

# Keep every model, data, multi-turn, objective, scoring, and parallel-layout
# argument identical to 04a; only the run length and W&B identity change.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/04a-geo3k-multiturn-rkld-smoke.sh"
