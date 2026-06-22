#!/usr/bin/env bash
# Python driver shim for launching Orbit inside the private Ray cluster.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this file from a launcher instead of running it directly." >&2
    exit 2
fi

prepare_parity_check_log() {
    if ! is_true "${PARITY_CHECK:-0}"; then
        return
    fi

    PARITY_CHECK_LOG=${PARITY_CHECK_LOG:-/tmp/orbit-parity-${LAUNCHER_NAME}-$$.log}
    : >"${PARITY_CHECK_LOG}"
    echo "PARITY_CHECK=1 enabled. Capturing launcher output for summary."
    echo "Capturing launcher output to: ${PARITY_CHECK_LOG}"
}

print_parity_check_summary() {
    local log="$1"
    if [[ ! -f "${log}" ]]; then
        echo "PARITY_CHECK summary: log file ${log} not found." >&2
        return
    fi

    echo
    echo "==================================================="
    echo "===== PARITY CHECK SUMMARY ========================"
    echo "==================================================="
    echo "Log file: ${log}"

    # log_utils emits Python dicts like:
    #   step 0: {'train/loss': ..., 'train/train_rollout_logprob_abs_diff': ...}
    local keys="train/tis_abs train/tis train/tis_clipfrac train/ppo_kl train/train_rollout_logprob_abs_diff train/kl_loss train/pg_loss train/loss train/entropy_loss"
    local found=0
    local key
    for key in ${keys}; do
        local matches
        matches=$(grep -oE "'${key}': [-+0-9.eE]+" "${log}" 2>/dev/null | tail -3 || true)
        if [[ -n "${matches}" ]]; then
            found=1
            while IFS= read -r line; do
                printf '  %s\n' "${line}"
            done <<<"${matches}"
        fi
    done
    if [[ "${found}" -eq 0 ]]; then
        echo "  (no train/* metrics found - did the run reach the first training step?)"
    fi
    echo "==================================================="
}

run_training_driver() {
    # Keep the argv list below in sync with the python3 invocation further down.
    if is_true "${ORBIT_DRY_RUN_ARGV:-0}"; then
        printf '%s\n' \
            "${ORBIT_ENTRYPOINT}" \
            --actor-num-nodes 1 \
            --actor-num-gpus-per-node "${GPUS_PER_NODE}" \
            "${COLOCATE_ARGS[@]}" \
            "${MODEL_ARGS[@]}" \
            "${CKPT_ARGS[@]}" \
            "${ROLLOUT_ARGS[@]}" \
            "${OPTIMIZER_ARGS[@]}" \
            "${RL_ARGS[@]}" \
            "${LOSS_ARGS[@]}" \
            "${WANDB_ARGS[@]}" \
            "${PERF_ARGS[@]}" \
            "${EVAL_ARGS[@]}" \
            "${SGLANG_ARGS[@]}" \
            "${MISC_ARGS[@]}" \
            "${DEBUG_ARGS[@]}" \
            "${PEFT_ARGS[@]}"
        return 0
    fi
    set +x
    _PARITY_TEE_TARGET=/dev/null
    if is_true "${PARITY_CHECK:-0}"; then
        : "${PARITY_CHECK_LOG:?PARITY_CHECK_LOG must be set when PARITY_CHECK=1}"
        _PARITY_TEE_TARGET="${PARITY_CHECK_LOG}"
    fi
    python3 - "${ORBIT_ENTRYPOINT}" \
       --actor-num-nodes 1 \
       --actor-num-gpus-per-node "${GPUS_PER_NODE}" \
       "${COLOCATE_ARGS[@]}" \
       "${MODEL_ARGS[@]}" \
       "${CKPT_ARGS[@]}" \
       "${ROLLOUT_ARGS[@]}" \
       "${OPTIMIZER_ARGS[@]}" \
       "${RL_ARGS[@]}" \
       "${LOSS_ARGS[@]}" \
       "${WANDB_ARGS[@]}" \
       "${PERF_ARGS[@]}" \
       "${EVAL_ARGS[@]}" \
       "${SGLANG_ARGS[@]}" \
       "${MISC_ARGS[@]}" \
       "${DEBUG_ARGS[@]}" \
       "${PEFT_ARGS[@]}" <<'PY' 2>&1 | tee "${_PARITY_TEE_TARGET}"
import os
import re
import runpy
import sys
import time
from pathlib import Path

import ray
import ray._private.services as services

train_path = sys.argv[1]
sys.argv = [train_path, *sys.argv[2:]]

_proxy_env = {
    k: os.environ[k]
    for k in ("no_proxy", "NO_PROXY", "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY")
    if k in os.environ
}

_original_find_node_ids = services.find_node_ids


def _private_ray_node_ids():
    temp_dir = os.environ.get("RAY_TEMP_DIR")
    if not temp_dir:
        return _original_find_node_ids()

    gcs_log = Path(temp_dir) / "session_latest" / "logs" / "gcs_server.out"
    node_id_pattern = re.compile(r"node_id=([0-9a-f]+)")
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            match = node_id_pattern.search(gcs_log.read_text(errors="ignore"))
            if match:
                return {match.group(1)}
        except OSError:
            pass
        time.sleep(0.1)
    return _original_find_node_ids()


services.find_node_ids = _private_ray_node_ids

_log_to_driver = os.environ.get("ORBIT_RAY_LOG_TO_DRIVER", "0").lower() in ("1", "true", "yes", "y", "on")
_ray_init_kwargs = {
    "address": os.environ["ORBIT_RAY_ADDRESS"],
    "log_to_driver": _log_to_driver,
    "runtime_env": {"env_vars": _proxy_env} if _proxy_env else None,
}
_driver_debug = os.environ.get("ORBIT_DRIVER_DEBUG", "0").lower() in ("1", "true", "yes", "y", "on")

_max_ray_init_attempts = int(os.environ.get("ORBIT_RAY_INIT_MAX_ATTEMPTS", "30"))
for _ray_init_attempt in range(1, _max_ray_init_attempts + 1):
    try:
        if _driver_debug:
            print(f"[orbit-driver-debug] ray.init attempt {_ray_init_attempt}", flush=True)
        ray.init(**_ray_init_kwargs)
        if _driver_debug:
            print("[orbit-driver-debug] ray.init returned", flush=True)
        break
    except ConnectionError as exc:
        if _ray_init_attempt == _max_ray_init_attempts:
            raise
        print(
            f"ray.init failed on attempt {_ray_init_attempt}/{_max_ray_init_attempts}; retrying: {exc!r}",
            file=sys.stderr,
        )
        time.sleep(1)

try:
    if _driver_debug:
        print(f"[orbit-driver-debug] runpy.run_path start {train_path}", flush=True)
    runpy.run_path(train_path, run_name="__main__")
finally:
    if _driver_debug:
        print("[orbit-driver-debug] ray.shutdown start", flush=True)
    ray.shutdown()
    if _driver_debug:
        print("[orbit-driver-debug] ray.shutdown returned", flush=True)
PY
    local driver_rc="${PIPESTATUS[0]:-$?}"
    echo "Training driver exited with code ${driver_rc}"
    return "${driver_rc}"
}
