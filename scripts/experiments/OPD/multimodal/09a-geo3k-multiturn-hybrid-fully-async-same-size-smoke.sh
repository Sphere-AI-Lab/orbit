#!/bin/bash
# Milestone 09a: 5-step fully-async same-size teacher/student smoke.
#
# Teacher: Qwen3-VL-8B-Thinking. Student: Qwen3-VL-8B-Instruct.
# Keep the completed 07 pure-hybrid objective, multimodal multi-turn rollout,
# and bounded-staleness scheduler. Task reward is observed only and never enters
# the training advantage.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export HF_CACHE_DIR=${HF_CACHE_DIR:-/data/shared/hf_cache}
export OPD_TEACHER_MODEL_DIR="$HF_CACHE_DIR/models/Qwen3-VL-8B-Thinking"
if [[ ! -f "$OPD_TEACHER_MODEL_DIR/config.json" ]]; then
   echo "FATAL: same-size OPD teacher not found at $OPD_TEACHER_MODEL_DIR" >&2
   echo "  hf download Qwen/Qwen3-VL-8B-Thinking --local-dir $OPD_TEACHER_MODEL_DIR" >&2
   return 1 2>/dev/null || exit 1
fi

# Hold the validated head-node TP=8 serving layout fixed across both 09 arms so
# the first comparison changes teacher weights/capacity, not server topology.
export OPD_TEACHER_TP=8
export OPD_TEACHER_GPUS=0,1,2,3,4,5,6,7
export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-5}
export FULLY_ASYNC_PREFETCH_BATCHES=2
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-09a-geo3k-hybrid-async-teacher8b-smoke}

# 07a owns the pure-hybrid objective and fully-async contract. In particular,
# it inherits --opd-log-task-reward but not --opd-optimize-task-reward.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/07a-geo3k-multiturn-hybrid-fully-async-smoke.sh"

# These are training-dynamics runs, not checkpoint-producing jobs. Fail closed
# if a future shared recipe introduces any checkpoint-save option.
for arg in "${ORBIT_ARGS[@]}"; do
   case "$arg" in
      --save | --save-* | --async-save)
         echo "FATAL: milestone 09 must not save checkpoints (found $arg)" >&2
         return 1 2>/dev/null || exit 1
         ;;
   esac
done
