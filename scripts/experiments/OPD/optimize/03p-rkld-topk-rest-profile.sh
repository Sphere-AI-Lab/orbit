#!/bin/bash
# Milestone 03p: dedicated Stable-TP operator profile on the hybrid workload.
# This trace arm is intentionally separate from the 03b-03d decision runs.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_TOP_K=0
export OPD_KL_COEF=${OPD_KL_COEF:-1}
export OPD_DAGGER_TOP_K=${OPD_DAGGER_TOP_K:-2}
export OPD_DAGGER_COEF=${OPD_DAGGER_COEF:-1}
export OPD_DAGGER_LOSS=cross_entropy
# Keep profiling bounded: one wait step, one profiler warmup step, then three
# active post-warmup steps by default. Remaining steps verify clean trace exit.
export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-8}
export TRAIN_TP_SIZE=2
export TRAIN_PP_SIZE=1
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-03p-rkld${OPD_KL_COEF}-top${OPD_DAGGER_TOP_K}-rest${OPD_DAGGER_COEF}-profile}
export ORBIT_PROFILE_OPD_DAGGER=1

# shellcheck disable=SC1091
source "$SCRIPT_DIR/../math_3nodes/qwen3-8B.sh"

PROFILE_STEP_START=${OPD_PROFILE_STEP_START:-2}
PROFILE_STEP_END=${OPD_PROFILE_STEP_END:-5}
if (( PROFILE_STEP_START < 1 || PROFILE_STEP_END <= PROFILE_STEP_START )); then
   echo "OPD profile window requires 1 <= start < end" >&2
   exit 2
fi
if (( PROFILE_STEP_END > OPD_NUM_ROLLOUT )); then
   echo "OPD profile end ($PROFILE_STEP_END) exceeds rollout count ($OPD_NUM_ROLLOUT)" >&2
   exit 2
fi

OPD_PROFILE_DIR=${OPD_PROFILE_DIR:-${HF_CACHE_DIR}/opd_profiles/${WANDB_RUN_NAME}}
ORBIT_ARGS+=(
   --use-pytorch-profiler
   --profile-target train_overall
   --profile-step-start "$PROFILE_STEP_START"
   --profile-step-end "$PROFILE_STEP_END"
   --tensorboard-dir "$OPD_PROFILE_DIR"
)
