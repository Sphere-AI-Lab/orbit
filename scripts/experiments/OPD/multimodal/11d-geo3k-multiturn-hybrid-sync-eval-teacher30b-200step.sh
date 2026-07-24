#!/bin/bash
# Milestone 11d: matched 200-step synchronous hybrid OPD with fixed eval.
# Teacher: Qwen3-VL-30B-A3B-Thinking. Student: Qwen3-VL-8B-Instruct.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export HF_CACHE_DIR=${HF_CACHE_DIR:-/data/shared/hf_cache}
export OPD_TEACHER_MODEL_DIR="$HF_CACHE_DIR/models/Qwen3-VL-30B-A3B-Thinking"
export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-200}
export OPD_EVAL_NUM_PROMPTS=${OPD_EVAL_NUM_PROMPTS:-30}
export OPD_EVAL_INTERVAL=${OPD_EVAL_INTERVAL:-5}
export OPD_EVAL_MAX_CONTEXT_LEN=${OPD_EVAL_MAX_CONTEXT_LEN:-12000}
export OPD_STUDENT_MEM_FRACTION=${OPD_STUDENT_MEM_FRACTION:-0.80}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-11d-sync-hybrid-teacher30b-eval${OPD_EVAL_INTERVAL}-n${OPD_EVAL_NUM_PROMPTS}-ctx${OPD_EVAL_MAX_CONTEXT_LEN}-${OPD_NUM_ROLLOUT}step}

# 06 owns the validated synchronous hybrid objective and three-node topology.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/06a-geo3k-multiturn-hybrid-smoke.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/11-fixed-eval-overlay.sh"
