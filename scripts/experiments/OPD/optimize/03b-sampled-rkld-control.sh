#!/bin/bash
# Milestone 03b: matched 50-step sampled RKLD-PG control.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_TOP_K=0
export OPD_KL_COEF=${OPD_KL_COEF:-1}
export OPD_DAGGER_TOP_K=0
export OPD_DAGGER_COEF=0
export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-50}
export TRAIN_TP_SIZE=2
export TRAIN_PP_SIZE=1
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-03b-rkld${OPD_KL_COEF}-control}

# shellcheck disable=SC1091
source "$SCRIPT_DIR/../math_qwen3_32b_8b_3nodes/qwen3-8B.sh"
