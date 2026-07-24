#!/bin/bash
# Milestone 02b: matched 20-step single-turn multimodal sampled-RKLD gate.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-20}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-02b-rkld-gate}

# Keep every model, data, objective and parallel-layout argument identical to
# 02a; only the run length and W&B identity change.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/02a-singleturn-rkld-smoke.sh"
