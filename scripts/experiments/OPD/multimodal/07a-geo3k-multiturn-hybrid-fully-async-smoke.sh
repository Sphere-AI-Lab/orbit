#!/bin/bash
# Milestone 07a: 5-step Geo3K multi-turn hybrid OPD fully-async smoke.
#
# This keeps the complete 06 hybrid algorithm and three-node ownership contract
# fixed. It changes scheduling only: train_async.py consumes completed groups
# while the existing fully-async worker continuously performs Student SGLang
# generation followed by the single OPD teacher-scoring request.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# ray_lifecycle.sh launches this entry point instead of the synchronous driver.
export MILES_TRAIN_ENTRY=train_async.py

# Pin the canonical smoke identity before sourcing 06a. Objective coefficients
# remain owned by 06a and cannot be changed by this scheduling wrapper.
export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-5}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-07a-geo3k-mt-hybrid-fully-async-smoke}

# shellcheck disable=SC1091
source "$SCRIPT_DIR/06a-geo3k-multiturn-hybrid-smoke.sh"

# Conservative first gate: keep one rollout batch actively generating, allow
# at most two completed batches to wait in CPU memory, and recycle work more
# than two rollout-engine weight versions old. No TIS/importance-ratio option
# is enabled; sampled RKLD and trainer-direct DAgger retain the 06 semantics.
FULLY_ASYNC_ARGS=(
   --rollout-function-path examples.fully_async.fully_async_rollout.generate_rollout_fully_async
   --fully-async-prefetch-batches 1
   --fully-async-max-completed-queue-groups 32
   --max-weight-staleness 2
   --update-weights-interval 1
)

MILES_ARGS+=("${FULLY_ASYNC_ARGS[@]}")
