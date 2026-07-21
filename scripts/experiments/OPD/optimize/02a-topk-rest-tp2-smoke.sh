#!/bin/bash
# Milestone 02a: canonical 5-step Stable TP Top-K + Rest smoke (TP=2, DP=4).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_TOP_K=0
export OPD_KL_COEF=0
export OPD_DAGGER_TOP_K=${OPD_DAGGER_TOP_K:-2}
export OPD_DAGGER_COEF=${OPD_DAGGER_COEF:-1}
export OPD_DAGGER_LOSS=cross_entropy
export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-5}
export TRAIN_TP_SIZE=2
export TRAIN_PP_SIZE=1
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-02a-top${OPD_DAGGER_TOP_K}-rest-c${OPD_DAGGER_COEF}-tp2-smoke}

# shellcheck disable=SC1091
source "$SCRIPT_DIR/../math_3nodes/qwen3-8B.sh"
