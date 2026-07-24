#!/bin/bash
# Milestone 09c: 5-step fully-async big-teacher/small-student smoke.
#
# Teacher: Qwen3-VL-30B-A3B-Thinking. Student: Qwen3-VL-8B-Instruct.
# This reruns the validated 07 model pair with prefetch two so it is matched to
# the same-size 09 arm. Task reward remains detached telemetry only.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export HF_CACHE_DIR=${HF_CACHE_DIR:-/data/shared/hf_cache}
export OPD_TEACHER_MODEL_DIR="$HF_CACHE_DIR/models/Qwen3-VL-30B-A3B-Thinking"
export OPD_TEACHER_TP=8
export OPD_TEACHER_GPUS=0,1,2,3,4,5,6,7
export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-5}
export FULLY_ASYNC_PREFETCH_BATCHES=2
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-09c-geo3k-hybrid-async-teacher30b-smoke}

# 07a supplies multimodal multi-turn generation, sampled RKLD-PG, trainer-direct
# Top-K + Rest, task-reward logging, and the bounded fully-async scheduler.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/07a-geo3k-multiturn-hybrid-fully-async-smoke.sh"

# These are training-dynamics runs, not checkpoint-producing jobs. Fail closed
# if a future shared recipe introduces any checkpoint-save option.
for arg in "${MILES_ARGS[@]}"; do
   case "$arg" in
      --save | --save-* | --async-save)
         echo "FATAL: milestone 09 must not save checkpoints (found $arg)" >&2
         return 1 2>/dev/null || exit 1
         ;;
   esac
done
