#!/bin/bash
# Milestone 09b: matched 200-step same-size teacher/student gate.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-200}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-09b-geo3k-hybrid-async-teacher8b-200step}

# Change only the optimizer-step count and run identity relative to 09a.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/09a-geo3k-multiturn-hybrid-fully-async-same-size-smoke.sh"
