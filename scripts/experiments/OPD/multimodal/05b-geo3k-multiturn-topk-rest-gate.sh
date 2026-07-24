#!/bin/bash
# Milestone 05b: matched 20-step synchronous Geo3K multi-turn Top-K + Rest gate.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-20}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-05b-geo3k-mt-dagger-top2-rest-gate}

# Keep every model, data, multi-turn, objective, scoring, optimizer, and
# parallel-layout argument identical to 05a. Launch only after 05a passes.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/05a-geo3k-multiturn-topk-rest-smoke.sh"
