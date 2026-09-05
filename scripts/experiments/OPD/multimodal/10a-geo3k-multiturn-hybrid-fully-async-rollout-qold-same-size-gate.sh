#!/bin/bash
# Milestone 10a: 200-step same-size teacher/student rollout-q_old gate.
#
# This is the paired treatment for 09b. Model pair, pure-hybrid objective,
# multimodal multi-turn data, prefetch-two scheduler, memory layout, and
# no-checkpoint contract are inherited unchanged. The only training-semantic
# switch is --use-rollout-logprobs.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-200}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-10a-geo3k-hybrid-async-rollout-qold-teacher8b-200step}

# 09b supplies the validated Qwen3-VL-8B-Thinking -> 8B-Instruct contract.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/09b-geo3k-multiturn-hybrid-fully-async-same-size-gate.sh"

# Keep this as a clean old-policy-source experiment. TIS is incompatible with
# rollout logprobs, while mismatch-metrics mode would restore the trainer
# pre-update forward and invalidate the systems-cost comparison.
for arg in "${ORBIT_ARGS[@]}"; do
   case "$arg" in
      --use-rollout-logprobs | --use-tis | --get-mismatch-metrics)
         echo "FATAL: milestone 10 owns the rollout-q_old switch (found inherited $arg)" >&2
         return 1 2>/dev/null || exit 1
         ;;
   esac
done

ORBIT_ARGS+=(--use-rollout-logprobs)
