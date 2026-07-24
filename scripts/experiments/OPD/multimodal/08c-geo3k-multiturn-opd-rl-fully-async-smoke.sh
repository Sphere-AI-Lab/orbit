#!/bin/bash
# Milestone 08c: 5-step fully-async Geo3K multi-turn OPD + task-RL smoke.
#
# This retains the complete 07 async ownership/staleness contract, widens the
# active producer window from one to two rollout batches, and enables the same
# task-reward objective validated synchronously by 08a.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-5}
export OPD_TASK_REWARD_COEF=1
export FULLY_ASYNC_PREFETCH_BATCHES=2
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-08c-geo3k-mt-opd-rl-async-pf2-smoke}

# shellcheck disable=SC1091
source "$SCRIPT_DIR/07a-geo3k-multiturn-hybrid-fully-async-smoke.sh"

MILES_ARGS+=(
   --opd-optimize-task-reward
   --opd-task-reward-coef "$OPD_TASK_REWARD_COEF"
)
