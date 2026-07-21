#!/bin/bash
# Milestone 03d: matched 50-step sampled RKLD-PG + Top-K + Rest gate.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_TOP_K=0
export OPD_KL_COEF=${OPD_KL_COEF:-1}
export OPD_DAGGER_TOP_K=${OPD_DAGGER_TOP_K:-2}
export OPD_DAGGER_COEF=${OPD_DAGGER_COEF:-1}
export OPD_DAGGER_LOSS=cross_entropy
export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-50}
export TRAIN_TP_SIZE=2
export TRAIN_PP_SIZE=1
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-03d-rkld${OPD_KL_COEF}-top${OPD_DAGGER_TOP_K}-rest${OPD_DAGGER_COEF}-gate}

# shellcheck disable=SC1091
source "$SCRIPT_DIR/../math_qwen3_32b_8b_3nodes/qwen3-8B.sh"
