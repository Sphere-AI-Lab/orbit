#!/bin/bash
# Milestone 04a: 5-step synchronous Geo3K multi-turn sampled-RKLD smoke.
#
# This changes only the rollout sequence contract relative to 02a. The model
# pair, three-node ownership, exact multimodal suffix scoring, sampled-RKLD
# objective, Megatron layout, optimizer, and task-reward telemetry stay fixed.
# Geo3K follow-up observations are pure text, so the final Sample can be scored
# in one teacher request: assistant spans are active and tool-feedback spans are
# retained positionally with loss_mask=0.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_KL_COEF=1
export OPD_DAGGER_TOP_K=0
export OPD_DAGGER_COEF=0
export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-5}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-04a-geo3k-mt-rkld-smoke}

# Start from the validated 02a model, scoring, trainer, and hardware contract.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/02a-singleturn-rkld-smoke.sh"

# Replace only the single-turn data/rollout arrays, then rebuild MILES_ARGS.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/geo3k-multiturn-overlay.sh"
