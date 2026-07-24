#!/bin/bash
# Milestone 11b: one fixed-set Geo3K evaluation of Qwen3-VL-30B-A3B-Thinking.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_EVAL_MODEL_NAME=Qwen3-VL-30B-A3B-Thinking
export OPD_EVAL_MODEL_ARGS_FILE=qwen3-30B-A3B.sh
export OPD_EVAL_NUM_PROMPTS=${OPD_EVAL_NUM_PROMPTS:-30}
export OPD_EVAL_MAX_CONTEXT_LEN=${OPD_EVAL_MAX_CONTEXT_LEN:-12000}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-11b-teacher30b-fixed-eval-n${OPD_EVAL_NUM_PROMPTS}-ctx${OPD_EVAL_MAX_CONTEXT_LEN}}

# shellcheck disable=SC1091
source "$SCRIPT_DIR/11-teacher-reference-common.sh"
