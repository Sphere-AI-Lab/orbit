#!/bin/bash
# Milestone 05a: 5-step synchronous Geo3K multi-turn teacher Top-K + Rest smoke.
#
# This holds the validated 04 sequence, model pair, exact-suffix scoring path,
# optimizer, and Megatron layout fixed. Only the trainer objective changes:
# sampled RKLD-PG is disabled and trainer-direct teacher Top-2 + Rest DAgger is
# enabled. Task reward remains telemetry only.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_KL_COEF=0
export OPD_DAGGER_TOP_K=2
export OPD_DAGGER_COEF=1
export OPD_DAGGER_LOSS=cross_entropy
export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-5}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-05a-geo3k-mt-dagger-top2-rest-smoke}

# Reuse the exact 02 model, scoring, optimizer, and parallel contract, then the
# same data/rollout overlay as 04. Neither shared layer owns objective values.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/02a-singleturn-rkld-smoke.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/geo3k-multiturn-overlay.sh"
