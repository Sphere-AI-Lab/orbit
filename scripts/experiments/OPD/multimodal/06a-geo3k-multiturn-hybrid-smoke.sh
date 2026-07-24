#!/bin/bash
# Milestone 06a: 5-step synchronous Geo3K multi-turn hybrid OPD smoke.
#
# This combines the two independently validated objectives on the unchanged 05
# sequence and infrastructure contract. A single teacher Top-2 response supplies
# both the sampled-action log-probs used by RKLD-PG and the sparse [T,2] targets
# used by trainer-direct Top-K + Rest DAgger. Task reward remains telemetry only.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Keep the canonical arm immutable under stray environment overrides. The
# conservative DAgger weight follows the earlier hybrid evidence: preserve the
# sampled-RKLD anchor while adding a directly differentiable sparse objective.
export OPD_KL_COEF=1
export OPD_DAGGER_TOP_K=2
export OPD_DAGGER_COEF=0.5
export OPD_DAGGER_LOSS=cross_entropy
export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-5}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-06a-geo3k-mt-hybrid-rkld1-dagger0p5-smoke}

# Reuse the validated model, exact-suffix scoring, optimizer, and Megatron
# contract from 02, then the same objective-free Geo3K multi-turn overlay as 04
# and 05. Neither shared layer owns hybrid loss values.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/02a-singleturn-rkld-smoke.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/geo3k-multiturn-overlay.sh"
