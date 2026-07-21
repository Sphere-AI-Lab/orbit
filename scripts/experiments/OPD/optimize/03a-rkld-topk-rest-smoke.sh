#!/bin/bash
# Milestone 03a: 5-step sampled RKLD-PG + Stable-TP Top-K + Rest smoke.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_TOP_K=0
export OPD_KL_COEF=${OPD_KL_COEF:-1}
export OPD_DAGGER_TOP_K=${OPD_DAGGER_TOP_K:-2}
export OPD_DAGGER_COEF=${OPD_DAGGER_COEF:-1}
export OPD_DAGGER_LOSS=cross_entropy
export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-5}
export TRAIN_TP_SIZE=2
export TRAIN_PP_SIZE=1
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-03a-rkld${OPD_KL_COEF}-top${OPD_DAGGER_TOP_K}-rest${OPD_DAGGER_COEF}-smoke}

# shellcheck disable=SC1091
source "$SCRIPT_DIR/../math_3nodes/qwen3-8B.sh"
