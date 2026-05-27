#!/usr/bin/env bash
# Slim Orbit launcher orchestrator. Caller defines all argument arrays;
# this file validates the contract, runs preflight, starts private Ray, and
# invokes the training driver.
#
# Required by caller (validated below):
#   ORBIT_ROOT, LAUNCHER_NAME, PRECISION_PROFILE, HF_CKPT, MEGATRON_LOAD,
#   SAVE_DIR, RUN_LOG, GPUS_PER_NODE
#   ORBIT_ENTRYPOINT  (default: ${ORBIT_ROOT}/train.py)
#   MODEL_ARGS, CKPT_ARGS, ROLLOUT_ARGS, EVAL_ARGS, PERF_ARGS, RL_ARGS,
#   OPTIMIZER_ARGS, SGLANG_ARGS, PEFT_ARGS, WANDB_ARGS, MISC_ARGS, LOSS_ARGS,
#   DEBUG_ARGS, COLOCATE_ARGS  (arrays; any may be empty)
#
# Cross-cutting env knobs (orthogonal to recipe content):
#   ORBIT_DRY_RUN_ARGV     Print python argv and exit 0 before Ray.
#   ORBIT_DEBUG_MODE       rollout|train -> --debug-rollout-only / --debug-train-only
#   ORBIT_LOG_FILTER       1 (default) filters Ray-actor noise from RUN_LOG.
#   ORBIT_LAUNCHER_XTRACE  1 enables set -x.
#   PARITY_CHECK           1 enables parity-check summary capture.
#   STAGE_HF_CKPT_TO       Local path to rsync HF_CKPT into before training.
#   STAGE_MEGATRON_CKPT_TO Same for MEGATRON_LOAD.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this from a launcher; do not run directly." >&2
    exit 2
fi

# === Contract validation ===
: "${ORBIT_ROOT:?ORBIT_ROOT must be set by the launcher}"
: "${LAUNCHER_NAME:?LAUNCHER_NAME must be set by the launcher}"
: "${PRECISION_PROFILE:?PRECISION_PROFILE must be set by the launcher}"
: "${HF_CKPT:?HF_CKPT must be set by the launcher}"
: "${MEGATRON_LOAD:?MEGATRON_LOAD must be set by the launcher}"
: "${SAVE_DIR:?SAVE_DIR must be set by the launcher}"
: "${RUN_LOG:?RUN_LOG must be set by the launcher}"
: "${GPUS_PER_NODE:?GPUS_PER_NODE must be set by the launcher}"
ORBIT_ENTRYPOINT=${ORBIT_ENTRYPOINT:-"${ORBIT_ROOT}/train.py"}

for _name in MODEL_ARGS CKPT_ARGS ROLLOUT_ARGS EVAL_ARGS PERF_ARGS \
             RL_ARGS OPTIMIZER_ARGS SGLANG_ARGS PEFT_ARGS WANDB_ARGS \
             MISC_ARGS LOSS_ARGS DEBUG_ARGS COLOCATE_ARGS; do
    if ! declare -p "${_name}" >/dev/null 2>&1; then
        echo "Launcher contract: ${_name} array must be defined" >&2
        exit 2
    fi
done
unset _name

# === Internal helpers ===
_LIB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${_LIB_DIR}/common.sh"
source "${_LIB_DIR}/preflight.sh"
source "${_LIB_DIR}/wandb.sh"
source "${_LIB_DIR}/ray.sh"
source "${_LIB_DIR}/driver.sh"

# === Logging ===
mkdir -p "$(dirname "${RUN_LOG}")"
export ORBIT_RAY_LOG_TO_DRIVER=${ORBIT_RAY_LOG_TO_DRIVER:-1}
export ORBIT_LOG_FILTER=${ORBIT_LOG_FILTER:-1}
ORBIT_LOG_FILTER_PATTERN=${ORBIT_LOG_FILTER_PATTERN:-'Decode batch|Prefill batch|event=oft_request_adapter_active|INFO: +[0-9].* HTTP/1\.[01]|GET /(health|server_info|get_server_info)|POST /generate|Reloaded tiktoken model|\[server-poll\]|\[Gloo\] Rank [0-9]+ is connected to|Rank [0-9]+ workspace\[[0-9]+\] 0x[0-9a-fA-F]+|Cache flushed successfully!|Calling super\(\)\.encode|\[INT4 transform\]|OFT on fused TELayerNormColumnParallelLinear|OFT wrapped Megatron modules|Loading a model with trust_remote_code|low_precision_bootstrap\.py:181|tokenization_kimi\.py:[0-9]+|bridge_peft_helpers\.py|Registered INT4 buffers|auto_bridge\.py:[0-9]+|LANG_RL Log directory|repeated [0-9]+x across cluster|wandb: |checkpoint\.load_state_dict|sgl-router|smg::core::'}
if ! is_true "${ORBIT_DRY_RUN_ARGV:-0}"; then
    if is_true "${ORBIT_LOG_FILTER}"; then
        exec > >(grep --line-buffered -E -v "${ORBIT_LOG_FILTER_PATTERN}" | tee -a "${RUN_LOG}") 2>&1
    else
        exec > >(tee -a "${RUN_LOG}") 2>&1
    fi
    echo "Logging to ${RUN_LOG}"
fi

configure_process_env
validate_eval_args

# === Ray resource defaults ===
apply_ray_defaults

# === Preflight (low-precision only) ===
if [[ "${PRECISION_PROFILE}" != "bf16" ]]; then
    if ! is_true "${ORBIT_DRY_RUN_ARGV:-0}"; then
        preflight_low_precision
    fi
fi

# === W&B ===
load_wandb_key
configure_wandb_args

# === Checkpoint staging (opt-in via STAGE_*_TO) ===
if ! is_true "${ORBIT_DRY_RUN_ARGV:-0}"; then
    if declare -F stage_hf_checkpoint_if_requested >/dev/null 2>&1; then
        stage_hf_checkpoint_if_requested
    fi
    if declare -F stage_megatron_checkpoint_if_requested >/dev/null 2>&1; then
        stage_megatron_checkpoint_if_requested
    fi
fi

# === Parity check log (PARITY_CHECK=1) ===
prepare_parity_check_log

# === Dry-run early-exit ===
# Dry-run early-exit: run_training_driver prints argv and returns 0 immediately.
if is_true "${ORBIT_DRY_RUN_ARGV:-0}"; then
    run_training_driver
    exit 0
fi

# === Ray + driver ===
prepare_ray_lifecycle
prepare_ray_env
start_ray_head
set +e
run_training_driver
_RC=$?
set -e
if is_true "${PARITY_CHECK:-0}"; then
    print_parity_check_summary "${PARITY_CHECK_LOG}"
fi
exit "${_RC}"
