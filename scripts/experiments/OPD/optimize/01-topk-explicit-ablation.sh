#!/bin/bash
# Milestone 01: native teacher Top-K targets with trainer-side explicit CE.
#
# One teacher request returns sampled log-probs plus [T,K] sparse targets.
# The trainer evaluates current student log-probs at those IDs; there is no
# response-wide union, Student SGLang rescore, Rest bucket, or sampled RKLD
# contribution in this isolated 50-step validation run.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_TOP_K=0
export OPD_KL_COEF=0
export OPD_DAGGER_TOP_K=${OPD_DAGGER_TOP_K:-2}
export OPD_DAGGER_COEF=${OPD_DAGGER_COEF:-1}
export OPD_DAGGER_LOSS=explicit_cross_entropy
export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-50}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-01-teacher-top${OPD_DAGGER_TOP_K}-ce}

# shellcheck disable=SC1091
source "$SCRIPT_DIR/../math_3nodes/qwen3-8B.sh"
