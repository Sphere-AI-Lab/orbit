#!/usr/bin/env bash
# Shared, explicit runtime and output contract for the clean E4 env2 rerun.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ORBIT_ICLR_ROOT="$(cd -- "${HERE}/../../.." && pwd -P)"

ORBIT_ENV2_ROOT="${ORBIT_ENV2_ROOT:-/fast/zqiu/orbit-iclr/orbit_env_v2}"
ORBIT_ENV2_ACTIVATE="${ORBIT_ENV2_ACTIVATE:-${ORBIT_ENV2_ROOT}/bin/activate}"
if [[ ! -f "${ORBIT_ENV2_ACTIVATE}" ]]; then
    echo "env2 activation script not found: ${ORBIT_ENV2_ACTIVATE}" >&2
    exit 2
fi
# shellcheck disable=SC1090
source "${ORBIT_ENV2_ACTIVATE}"
if [[ "${VIRTUAL_ENV:-}" != "${ORBIT_ENV2_ROOT}" ]]; then
    echo "expected VIRTUAL_ENV=${ORBIT_ENV2_ROOT}, got ${VIRTUAL_ENV:-<unset>}" >&2
    exit 2
fi

E4_ENV2_RUN_ROOT="${E4_ENV2_RUN_ROOT:-/lustre/fast/fast/zqiu/orbit-iclr/experiment-runs/env2-rerun-20260821}"
E4_ENV2_RESULTS_DIR="${E4_ENV2_RUN_ROOT}/results"
LORA_REGRET_LOG_DIR="${E4_ENV2_RUN_ROOT}/logs/lora_regret"
WANDB_DIR="${E4_ENV2_RUN_ROOT}/wandb"
LORA_REGRET_CKPT_DIR="${E4_ENV2_RUN_ROOT}/orbit_ckpts/lora_regret"
E4_ENV2_SCHEDULER_DIR="${E4_ENV2_RUN_ROOT}/scheduler"

mkdir -p \
    "${E4_ENV2_RESULTS_DIR}" \
    "${LORA_REGRET_LOG_DIR}" \
    "${WANDB_DIR}" \
    "${LORA_REGRET_CKPT_DIR}" \
    "${E4_ENV2_SCHEDULER_DIR}"

export ORBIT_ICLR_ROOT ORBIT_ENV2_ROOT ORBIT_ENV2_ACTIVATE
export E4_ENV2_RUN_ROOT E4_ENV2_RESULTS_DIR E4_ENV2_SCHEDULER_DIR
export LORA_REGRET_LOG_DIR LORA_REGRET_CKPT_DIR WANDB_DIR
export WANDB_MODE=offline
# Keep this rerun's offline files isolated until its dedicated sync wrapper is used.
export WANDB_AUTOSYNC=0
export CUDA_HOME="${CUDA_HOME:-/is/software/nvidia/cuda-13.2}"
export PYTHONPATH="${ORBIT_ICLR_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export RL_EXTRA_ARGS="${RL_EXTRA_ARGS:---disable-grpo-std-normalization} --sglang-cuda-graph-backend-prefill disabled"
