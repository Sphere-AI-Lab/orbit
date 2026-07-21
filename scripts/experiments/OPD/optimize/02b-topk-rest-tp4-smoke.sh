#!/bin/bash
# Milestone 02b: 5-step TP-size generalization smoke (TP=4, DP=2).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_TOP_K=0
export OPD_KL_COEF=0
export OPD_DAGGER_TOP_K=${OPD_DAGGER_TOP_K:-2}
export OPD_DAGGER_COEF=${OPD_DAGGER_COEF:-1}
export OPD_DAGGER_LOSS=cross_entropy
export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-5}
export TRAIN_TP_SIZE=4
export TRAIN_PP_SIZE=1
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-02b-top${OPD_DAGGER_TOP_K}-rest-c${OPD_DAGGER_COEF}-tp4-smoke}

# shellcheck disable=SC1091
source "$SCRIPT_DIR/../math_3nodes/qwen3-8B.sh"
