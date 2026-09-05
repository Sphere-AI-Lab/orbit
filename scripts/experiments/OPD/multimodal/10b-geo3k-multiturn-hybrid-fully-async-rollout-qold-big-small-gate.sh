#!/bin/bash
# Milestone 10b: 200-step big-teacher/small-student rollout-q_old gate.
#
# This is the paired treatment for 09d. It preserves the successful 0.80
# Student SGLang memory fraction as well as every other 09d setting, and changes
# only the source of q_old used by RKLD-PG and sampled-token PPO.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-200}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-10b-geo3k-hybrid-async-rollout-qold-teacher30b-200step}

# 09d supplies the validated Qwen3-VL-30B-A3B-Thinking -> 8B-Instruct contract.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/09d-geo3k-multiturn-hybrid-fully-async-big-small-gate.sh"

for arg in "${ORBIT_ARGS[@]}"; do
   case "$arg" in
      --use-rollout-logprobs | --use-tis | --get-mismatch-metrics)
         echo "FATAL: milestone 10 owns the rollout-q_old switch (found inherited $arg)" >&2
         return 1 2>/dev/null || exit 1
         ;;
   esac
done

ORBIT_ARGS+=(--use-rollout-logprobs)
