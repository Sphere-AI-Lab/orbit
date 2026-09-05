#!/bin/bash
# Milestone 08a: 5-step synchronous Geo3K multi-turn OPD + task-RL smoke.
#
# Keep the complete 06 hybrid objective and infrastructure fixed. The only
# algorithmic change is that the already-observed math reward now enters the
# GRPO base advantage before sampled RKLD is applied; Top-K + Rest remains a
# separate trainer-direct loss.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Pin the synchronous control even if the submit environment previously set an
# async entry point.
export ORBIT_TRAIN_ENTRY=train.py
export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-5}
export OPD_TASK_REWARD_COEF=1
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-08a-geo3k-mt-opd-rl-sync-smoke}

# shellcheck disable=SC1091
source "$SCRIPT_DIR/06a-geo3k-multiturn-hybrid-smoke.sh"

ORBIT_ARGS+=(
   --opd-optimize-task-reward
   --opd-task-reward-coef "$OPD_TASK_REWARD_COEF"
)
