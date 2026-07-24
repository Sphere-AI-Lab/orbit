#!/bin/bash
# Milestone 11c: 200-step synchronous hybrid OPD with fixed eval every 5 steps.
# Teacher: Qwen3-VL-8B-Thinking. Student: Qwen3-VL-8B-Instruct.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export HF_CACHE_DIR=${HF_CACHE_DIR:-/data/shared/hf_cache}
export OPD_TEACHER_MODEL_DIR="$HF_CACHE_DIR/models/Qwen3-VL-8B-Thinking"
export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-200}
export OPD_EVAL_NUM_PROMPTS=${OPD_EVAL_NUM_PROMPTS:-30}
export OPD_EVAL_INTERVAL=${OPD_EVAL_INTERVAL:-5}
export OPD_EVAL_MAX_CONTEXT_LEN=${OPD_EVAL_MAX_CONTEXT_LEN:-12000}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-11c-sync-hybrid-teacher8b-eval${OPD_EVAL_INTERVAL}-n${OPD_EVAL_NUM_PROMPTS}-ctx${OPD_EVAL_MAX_CONTEXT_LEN}-${OPD_NUM_ROLLOUT}step}

# 06 owns the validated synchronous hybrid objective and three-node topology.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/06a-geo3k-multiturn-hybrid-smoke.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/11-fixed-eval-overlay.sh"
