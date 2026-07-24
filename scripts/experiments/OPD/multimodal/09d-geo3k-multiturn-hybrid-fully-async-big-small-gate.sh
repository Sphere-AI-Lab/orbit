#!/bin/bash
# Milestone 09d: matched 200-step big-teacher/small-student gate.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-200}
export OPD_STUDENT_MEM_FRACTION=${OPD_STUDENT_MEM_FRACTION:-0.80}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-09d-geo3k-hybrid-async-teacher30b-200step}

# Pin the long-window student headroom proven by the successful 09d rerun.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/09c-geo3k-multiturn-hybrid-fully-async-big-small-smoke.sh"
