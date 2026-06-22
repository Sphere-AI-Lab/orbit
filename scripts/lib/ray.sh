#!/usr/bin/env bash
# Ray lifecycle, private port reservation, and launcher Ray environment.
#
# Sourced by:
#   * scripts/lib/launcher.sh — for the training launcher entrypoints
#     (apply_ray_defaults, prepare_ray_lifecycle, prepare_ray_env,
#     start_ray_head, cleanup_private_ray).
#   * standalone Ray-job scripts (e.g. parity_check, conversion,
#     validation gates) — via start_private_ray_job_head /
#     cleanup_private_ray (also used by training launcher).
# A single process picks exactly one of these audiences.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this file from a launcher instead of running it directly." >&2
    exit 2
fi

ORBIT_PORT_LOCK_DIR=${ORBIT_PORT_LOCK_DIR:-/tmp/orbit-port-locks-${USER}}
PORT_LOCK_FDS=()

_ray_array_value_after() {
    local array_name="$1"
    local flag="$2"
    local -n args_ref="${array_name}"
    local idx

    for ((idx = 0; idx < ${#args_ref[@]}; idx++)); do
        if [[ "${args_ref[idx]}" == "${flag}" && $((idx + 1)) -lt ${#args_ref[@]} ]]; then
            printf '%s\n' "${args_ref[idx + 1]}"
            return 0
        fi
    done
    return 1
}

_ray_array_value_after_any() {
    local flag="$1"
    shift
    local array_name

    for array_name in "$@"; do
        if declare -p "${array_name}" >/dev/null 2>&1; then
            _ray_array_value_after "${array_name}" "${flag}" && return 0
        fi
    done
    return 1
}

_ray_advantage_estimator() {
    if [[ -n "${ADVANTAGE_ESTIMATOR:-}" ]]; then
        printf '%s\n' "${ADVANTAGE_ESTIMATOR}"
        return 0
    fi
    if declare -p RL_ARGS >/dev/null 2>&1; then
        _ray_array_value_after RL_ARGS --advantage-estimator && return 0
    fi
    printf 'grpo\n'
}

_ray_critic_num_gpus() {
    local actor_gpus="$1"
    local estimator
    estimator="$(_ray_advantage_estimator)"
    if [[ "${estimator}" != "ppo" ]]; then
        printf '0\n'
        return 0
    fi

    local critic_gpus_per_node="${CRITIC_NUM_GPUS_PER_NODE:-}"
    local critic_num_nodes="${CRITIC_NUM_NODES:-}"
    if [[ -z "${critic_gpus_per_node}" ]]; then
        critic_gpus_per_node="$(_ray_array_value_after_any --critic-num-gpus-per-node CKPT_ARGS ROLLOUT_ARGS OPTIMIZER_ARGS RL_ARGS LOSS_ARGS WANDB_ARGS PERF_ARGS EVAL_ARGS SGLANG_ARGS MISC_ARGS DEBUG_ARGS PEFT_ARGS COLOCATE_ARGS || true)"
    fi
    if [[ -z "${critic_num_nodes}" ]]; then
        critic_num_nodes="$(_ray_array_value_after_any --critic-num-nodes CKPT_ARGS ROLLOUT_ARGS OPTIMIZER_ARGS RL_ARGS LOSS_ARGS WANDB_ARGS PERF_ARGS EVAL_ARGS SGLANG_ARGS MISC_ARGS DEBUG_ARGS PEFT_ARGS COLOCATE_ARGS || true)"
    fi

    critic_gpus_per_node="${critic_gpus_per_node:-${actor_gpus}}"
    critic_num_nodes="${critic_num_nodes:-1}"
    printf '%s\n' "$((critic_gpus_per_node * critic_num_nodes))"
}

_ray_launcher_is_colocated() {
    case "${ORBIT_COLOCATE:-}" in
        0|false|False|FALSE|no|No|NO|off|Off|OFF)
            return 1
            ;;
        1|true|True|TRUE|yes|Yes|YES|on|On|ON)
            return 0
            ;;
    esac

    if declare -p COLOCATE_ARGS >/dev/null 2>&1; then
        local arg
        for arg in "${COLOCATE_ARGS[@]}"; do
            if [[ "${arg}" == "--colocate" ]]; then
                return 0
            fi
        done
        return 1
    fi

    return 0
}

_ray_rollout_num_gpus() {
    if [[ -n "${ROLLOUT_NUM_GPUS:-}" ]]; then
        printf '%s\n' "${ROLLOUT_NUM_GPUS}"
        return 0
    fi
    if declare -p SGLANG_ARGS >/dev/null 2>&1; then
        _ray_array_value_after SGLANG_ARGS --rollout-num-gpus && return 0
    fi
    printf '0\n'
}

_ray_valid_megatron_path() {
    local path="$1"
    [[ -f "${path}/megatron/training/training.py" \
        && -f "${path}/megatron/post_training/checkpointing.py" ]]
}

_infer_megatron_path_from_editable() {
    local python_bin="${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin/python}"
    if [[ -z "${python_bin}" || ! -x "${python_bin}" ]]; then
        python_bin="$(command -v python3 || command -v python || true)"
    fi
    [[ -n "${python_bin}" ]] || return 1

    VIRTUAL_ENV="${VIRTUAL_ENV}" "${python_bin}" - <<'PY' 2>/dev/null
import ast
import os
from pathlib import Path

venv = os.environ.get("VIRTUAL_ENV")
if not venv:
    raise SystemExit(1)

for site_packages in Path(venv).glob("lib/python*/site-packages"):
    for finder in site_packages.glob("__editable___megatron_core*_finder.py"):
        try:
            tree = ast.parse(finder.read_text())
        except OSError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.AnnAssign):
                continue
            if not isinstance(node.target, ast.Name) or node.target.id != "MAPPING":
                continue
            mapping = ast.literal_eval(node.value)
            core_path = mapping.get("megatron.core")
            if core_path:
                print(Path(core_path).parents[1])
                raise SystemExit(0)
raise SystemExit(1)
PY
}

_prepare_megatron_pythonpath() {
    local candidate
    if [[ -z "${MEGATRON_PATH:-}" ]]; then
        local candidates=()
        if [[ -n "${MEGATRON_BRIDGE_ROOT:-}" ]]; then
            candidates+=("${MEGATRON_BRIDGE_ROOT}/3rdparty/Megatron-LM")
        fi
        candidate="$(_infer_megatron_path_from_editable || true)"
        if [[ -n "${candidate}" ]]; then
            candidates+=("${candidate}")
        fi
        candidates+=("${ORBIT_ROOT}/../env_for_cc/Megatron-Bridge/3rdparty/Megatron-LM")

        for candidate in "${candidates[@]}"; do
            if [[ -n "${candidate}" ]] && _ray_valid_megatron_path "${candidate}"; then
                MEGATRON_PATH="${candidate}"
                break
            fi
        done
    fi

    if [[ -n "${MEGATRON_PATH:-}" ]]; then
        export MEGATRON_PATH
        case ":${PYTHONPATH:-}:" in
            *":${MEGATRON_PATH}:"*) ;;
            *) export PYTHONPATH="${MEGATRON_PATH}${PYTHONPATH:+:${PYTHONPATH}}" ;;
        esac
    fi
}

apply_ray_defaults() {
    RAY_NUM_CPUS=${RAY_NUM_CPUS:-64}
    if [[ -z "${RAY_NUM_GPUS:-}" ]]; then
        local actor_gpus="${GPUS_PER_NODE:-0}"
        local critic_gpus
        critic_gpus="$(_ray_critic_num_gpus "${actor_gpus}")"
        if _ray_launcher_is_colocated; then
            RAY_NUM_GPUS=$((actor_gpus + critic_gpus))
        else
            local rollout_gpus
            rollout_gpus="$(_ray_rollout_num_gpus)"
            RAY_NUM_GPUS=$((actor_gpus + critic_gpus + rollout_gpus))
        fi
    fi
}

apply_colocate_defaults() {
    : "${ORBIT_COLOCATE:=1}"
    : "${ROLLOUT_NUM_GPUS:=0}"
    if [[ "${ORBIT_COLOCATE}" -eq 1 ]]; then
        COLOCATE_ARGS=("--colocate")
        : "${RAY_NUM_GPUS:=${GPUS_PER_NODE}}"
    else
        COLOCATE_ARGS=()
        : "${RAY_NUM_GPUS:=$((GPUS_PER_NODE + ROLLOUT_NUM_GPUS))}"
    fi
}

prepare_ray_lifecycle() {
    ORBIT_RAY_LIFECYCLE=${ORBIT_RAY_LIFECYCLE:-private}
    if [[ "${ORBIT_RAY_LIFECYCLE}" == "legacy" ]]; then
        pkill -9 -u "$(id -un)" sglang 2>/dev/null || true
        sleep 2
        ray stop --force 2>/dev/null || true
        pkill -9 -u "$(id -un)" -f "ray::|raylet|gcs_server|SGLangEngine|MegatronTrainRayActor" 2>/dev/null || true
        sleep 2
    else
        echo "Using private Ray lifecycle; skipping global Ray/SGLang cleanup."
    fi
}

prepare_ray_env() {
    export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
    ORBIT_ENTRYPOINT=${ORBIT_ENTRYPOINT:-"${ORBIT_ROOT}/train.py"}
    RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-}
    RAY_DASHBOARD_AGENT_LISTEN_PORT=${RAY_DASHBOARD_AGENT_LISTEN_PORT:-}
    RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/ray-${USER}-$$}
    mkdir -p "${RAY_TEMP_DIR}"
    if [[ -z "${RAY_NUM_GPUS:-}" ]]; then
        apply_ray_defaults
    fi
    export PYTHONPATH="${ORBIT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
    _prepare_megatron_pythonpath
    export ORBIT_RAY_LIFECYCLE GPUS_PER_NODE RAY_NUM_GPUS RAY_NUM_CPUS RAY_TEMP_DIR
}

reserve_launcher_port() {
    local result_var="$1"
    local min_port="$2"
    local max_port="$3"
    local port lock_fd lock_file
    mkdir -p "${ORBIT_PORT_LOCK_DIR}"

    for _ in $(seq 1 200); do
        port=$(python3 - "${min_port}" "${max_port}" <<'PY'
import random
import sys

lower, upper = map(int, sys.argv[1:3])
print(random.SystemRandom().randint(lower, upper))
PY
)
        lock_file="${ORBIT_PORT_LOCK_DIR}/${port}.lock"
        if ! exec {lock_fd}>"${lock_file}"; then
            continue
        fi
        if flock -n "${lock_fd}" && python3 - "${port}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", port))
except OSError:
    sys.exit(1)
PY
        then
            PORT_LOCK_FDS+=("${lock_fd}")
            printf -v "${result_var}" '%s' "${port}"
            return 0
        fi
        eval "exec ${lock_fd}>&-"
    done

    echo "Could not reserve a free launcher port in ${min_port}-${max_port}" >&2
    return 1
}

reserve_launcher_port_range() {
    local result_var="$1"
    local min_port="$2"
    local max_port="$3"
    local range_size="$4"
    local start_port end_start port lock_fd lock_file ok
    local acquired_fds=()
    mkdir -p "${ORBIT_PORT_LOCK_DIR}"

    end_start=$((max_port - range_size + 1))
    if (( end_start < min_port )); then
        echo "Invalid launcher port range ${min_port}-${max_port} for size ${range_size}" >&2
        return 1
    fi

    for _ in $(seq 1 200); do
        start_port=$(python3 - "${min_port}" "${end_start}" <<'PY'
import random
import sys

lower, upper = map(int, sys.argv[1:3])
print(random.SystemRandom().randint(lower, upper))
PY
)
        acquired_fds=()
        ok=1
        for port in $(seq "${start_port}" "$((start_port + range_size - 1))"); do
            lock_file="${ORBIT_PORT_LOCK_DIR}/${port}.lock"
            if ! exec {lock_fd}>"${lock_file}"; then
                ok=0
                break
            fi
            if flock -n "${lock_fd}"; then
                acquired_fds+=("${lock_fd}")
            else
                eval "exec ${lock_fd}>&-"
                ok=0
                break
            fi
        done

        if (( ok )) && python3 - "${start_port}" "${range_size}" <<'PY'
import socket
import sys

start_port = int(sys.argv[1])
range_size = int(sys.argv[2])
try:
    for port in range(start_port, start_port + range_size):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("0.0.0.0", port))
except OSError:
    sys.exit(1)
PY
        then
            PORT_LOCK_FDS+=("${acquired_fds[@]}")
            printf -v "${result_var}" '%s' "${start_port}"
            return 0
        fi
        for lock_fd in "${acquired_fds[@]}"; do
            eval "exec ${lock_fd}>&-"
        done
    done

    echo "Could not reserve a free launcher port range of ${range_size} in ${min_port}-${max_port}" >&2
    return 1
}

RAY_START_PID=""

_ray_status_ready() {
    timeout "${RAY_READY_CHECK_TIMEOUT:-5s}" ray status --address "${ORBIT_RAY_ADDRESS}" >/dev/null 2>&1
}

_ray_job_server_ready() {
    timeout "${RAY_READY_CHECK_TIMEOUT:-5s}" ray job list --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" >/dev/null 2>&1
}

cleanup_private_ray() {
    local exit_code=$?
    trap - EXIT INT TERM

    if [[ "${ORBIT_RAY_LIFECYCLE}" == "private" && -n "${RAY_START_PID:-}" ]]; then
        if kill -0 "${RAY_START_PID}" 2>/dev/null; then
            kill "${RAY_START_PID}" 2>/dev/null || true
            wait "${RAY_START_PID}" 2>/dev/null || true
        fi

        if [[ -n "${RAY_TEMP_DIR:-}" ]]; then
            mapfile -t ray_pids < <(pgrep -u "$(id -u)" -f "${RAY_TEMP_DIR}" 2>/dev/null || true)
            if (( ${#ray_pids[@]} > 0 )); then
                kill "${ray_pids[@]}" 2>/dev/null || true
                sleep 2
                mapfile -t ray_pids < <(pgrep -u "$(id -u)" -f "${RAY_TEMP_DIR}" 2>/dev/null || true)
                if (( ${#ray_pids[@]} > 0 )); then
                    kill -9 "${ray_pids[@]}" 2>/dev/null || true
                fi
            fi
        fi
    fi

    for _fd in "${PORT_LOCK_FDS[@]}"; do
        eval "exec ${_fd}>&-" 2>/dev/null || true
    done
    PORT_LOCK_FDS=()

    exit "${exit_code}"
}

start_ray_head() {
    if [[ "${ORBIT_RAY_LIFECYCLE}" == "legacy" ]]; then
        RAY_HEAD_PORT=${RAY_HEAD_PORT:-6379}
        RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-22265}
        RAY_DASHBOARD_AGENT_LISTEN_PORT=${RAY_DASHBOARD_AGENT_LISTEN_PORT:-52465}
        export ORBIT_RAY_ADDRESS="${MASTER_ADDR}:${RAY_HEAD_PORT}"
        ray start --head --node-ip-address "${MASTER_ADDR}" --port "${RAY_HEAD_PORT}" \
            --num-gpus "${RAY_NUM_GPUS}" \
            --num-cpus "${RAY_NUM_CPUS}" \
            --temp-dir "${RAY_TEMP_DIR}" \
            --disable-usage-stats \
            --dashboard-host=127.0.0.1 --dashboard-port="${RAY_DASHBOARD_PORT}" \
            --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_LISTEN_PORT}"
    else
        RAY_WORKER_PORT_RANGE_SIZE=${RAY_WORKER_PORT_RANGE_SIZE:-512}
        local min_worker_port_range_size=$((RAY_NUM_CPUS + 16))
        if (( RAY_WORKER_PORT_RANGE_SIZE < min_worker_port_range_size )); then
            echo "Increasing RAY_WORKER_PORT_RANGE_SIZE from ${RAY_WORKER_PORT_RANGE_SIZE} to ${min_worker_port_range_size} to cover Ray's prestarted workers."
            RAY_WORKER_PORT_RANGE_SIZE="${min_worker_port_range_size}"
        fi
        [[ -n "${RAY_HEAD_PORT:-}" ]] || reserve_launcher_port RAY_HEAD_PORT 24000 29999
        [[ -n "${RAY_NODE_MANAGER_PORT:-}" ]] || reserve_launcher_port RAY_NODE_MANAGER_PORT 30000 34999
        [[ -n "${RAY_OBJECT_MANAGER_PORT:-}" ]] || reserve_launcher_port RAY_OBJECT_MANAGER_PORT 35000 39999
        [[ -n "${RAY_DASHBOARD_PORT:-}" ]] || reserve_launcher_port RAY_DASHBOARD_PORT 40000 44999
        [[ -n "${RAY_DASHBOARD_AGENT_LISTEN_PORT:-}" ]] || reserve_launcher_port RAY_DASHBOARD_AGENT_LISTEN_PORT 45000 49999
        [[ -n "${RAY_DASHBOARD_AGENT_GRPC_PORT:-}" ]] || reserve_launcher_port RAY_DASHBOARD_AGENT_GRPC_PORT 50000 54999
        [[ -n "${RAY_RUNTIME_ENV_AGENT_PORT:-}" ]] || reserve_launcher_port RAY_RUNTIME_ENV_AGENT_PORT 55000 59999
        [[ -n "${RAY_METRICS_EXPORT_PORT:-}" ]] || reserve_launcher_port RAY_METRICS_EXPORT_PORT 60000 64999
        [[ -n "${RAY_MIN_WORKER_PORT:-}" ]] || reserve_launcher_port_range RAY_MIN_WORKER_PORT 20000 23999 "${RAY_WORKER_PORT_RANGE_SIZE}"
        RAY_MAX_WORKER_PORT=${RAY_MAX_WORKER_PORT:-$((RAY_MIN_WORKER_PORT + RAY_WORKER_PORT_RANGE_SIZE - 1))}
        echo "Private Ray ports: head=${RAY_HEAD_PORT}, dashboard=${RAY_DASHBOARD_PORT}, workers=${RAY_MIN_WORKER_PORT}-${RAY_MAX_WORKER_PORT}"
        export ORBIT_RAY_ADDRESS="${MASTER_ADDR}:${RAY_HEAD_PORT}"
        RAY_START_LOG="${RAY_TEMP_DIR}/ray-start.log"
        trap cleanup_private_ray EXIT INT TERM
        ray start --head --node-ip-address "${MASTER_ADDR}" --port "${RAY_HEAD_PORT}" \
            --node-manager-port "${RAY_NODE_MANAGER_PORT}" \
            --object-manager-port "${RAY_OBJECT_MANAGER_PORT}" \
            --num-gpus "${RAY_NUM_GPUS}" \
            --num-cpus "${RAY_NUM_CPUS}" \
            --min-worker-port "${RAY_MIN_WORKER_PORT}" \
            --max-worker-port "${RAY_MAX_WORKER_PORT}" \
            --temp-dir "${RAY_TEMP_DIR}" \
            --disable-usage-stats \
            --include-dashboard=false \
            --dashboard-host=127.0.0.1 --dashboard-port="${RAY_DASHBOARD_PORT}" \
            --dashboard-agent-listen-port "${RAY_DASHBOARD_AGENT_LISTEN_PORT}" \
            --dashboard-agent-grpc-port "${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
            --runtime-env-agent-port "${RAY_RUNTIME_ENV_AGENT_PORT}" \
            --metrics-export-port "${RAY_METRICS_EXPORT_PORT}" \
            --block >"${RAY_START_LOG}" 2>&1 &
        RAY_START_PID=$!

        for _ in $(seq 1 120); do
            if ! kill -0 "${RAY_START_PID}" 2>/dev/null; then
                echo "Private Ray head exited before becoming ready; log follows:" >&2
                cat "${RAY_START_LOG}" >&2 || true
                exit 1
            fi
            if _ray_status_ready; then
                break
            fi
            sleep 1
        done
        set +e
        _ray_status_ready
        local ray_ready_rc=$?
        set -e
        if (( ray_ready_rc != 0 )); then
            echo "Timed out waiting for private Ray head at ${ORBIT_RAY_ADDRESS}; log follows:" >&2
            cat "${RAY_START_LOG}" >&2 || true
            exit 1
        fi
    fi
}

start_private_ray_job_head() {
    ORBIT_RAY_LIFECYCLE=${ORBIT_RAY_LIFECYCLE:-private}
    RAY_NUM_CPUS=${RAY_NUM_CPUS:-64}
    RAY_NUM_GPUS=${RAY_NUM_GPUS:-${GPUS_PER_NODE:-0}}
    MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    RAY_TEMP_DIR=${RAY_TEMP_DIR:-/tmp/ray-${USER}-$$}
    mkdir -p "${RAY_TEMP_DIR}"
    export MASTER_ADDR RAY_NUM_CPUS RAY_TEMP_DIR ORBIT_RAY_LIFECYCLE

    if [[ "${ORBIT_RAY_LIFECYCLE}" == "legacy" ]]; then
        RAY_HEAD_PORT=${RAY_HEAD_PORT:-6379}
        RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-22265}
        RAY_DASHBOARD_AGENT_LISTEN_PORT=${RAY_DASHBOARD_AGENT_LISTEN_PORT:-52465}
        export ORBIT_RAY_ADDRESS="${MASTER_ADDR}:${RAY_HEAD_PORT}"

        pkill -9 -u "$(id -un)" sglang 2>/dev/null || true
        sleep 2
        ray stop --force 2>/dev/null || true
        pkill -9 -u "$(id -un)" -f "ray::|raylet|gcs_server|SGLangEngine|MegatronTrainRayActor" 2>/dev/null || true
        sleep 2

        ray start --head --node-ip-address "${MASTER_ADDR}" --port "${RAY_HEAD_PORT}" \
            --num-gpus "${RAY_NUM_GPUS}" \
            --num-cpus "${RAY_NUM_CPUS}" \
            --temp-dir "${RAY_TEMP_DIR}" \
            --disable-usage-stats \
            --dashboard-host=127.0.0.1 --dashboard-port="${RAY_DASHBOARD_PORT}" \
            --dashboard-agent-listen-port="${RAY_DASHBOARD_AGENT_LISTEN_PORT}"
        return 0
    fi

    echo "Using private Ray lifecycle for ray job; skipping global Ray/SGLang cleanup."

    RAY_WORKER_PORT_RANGE_SIZE=${RAY_WORKER_PORT_RANGE_SIZE:-512}
    local min_worker_port_range_size=$((RAY_NUM_CPUS + 16))
    if (( RAY_WORKER_PORT_RANGE_SIZE < min_worker_port_range_size )); then
        echo "Increasing RAY_WORKER_PORT_RANGE_SIZE from ${RAY_WORKER_PORT_RANGE_SIZE} to ${min_worker_port_range_size} to cover Ray's prestarted workers."
        RAY_WORKER_PORT_RANGE_SIZE="${min_worker_port_range_size}"
    fi
    [[ -n "${RAY_HEAD_PORT:-}" ]] || reserve_launcher_port RAY_HEAD_PORT 24000 29999
    [[ -n "${RAY_NODE_MANAGER_PORT:-}" ]] || reserve_launcher_port RAY_NODE_MANAGER_PORT 30000 34999
    [[ -n "${RAY_OBJECT_MANAGER_PORT:-}" ]] || reserve_launcher_port RAY_OBJECT_MANAGER_PORT 35000 39999
    [[ -n "${RAY_DASHBOARD_PORT:-}" ]] || reserve_launcher_port RAY_DASHBOARD_PORT 40000 44999
    [[ -n "${RAY_DASHBOARD_AGENT_LISTEN_PORT:-}" ]] || reserve_launcher_port RAY_DASHBOARD_AGENT_LISTEN_PORT 45000 49999
    [[ -n "${RAY_DASHBOARD_AGENT_GRPC_PORT:-}" ]] || reserve_launcher_port RAY_DASHBOARD_AGENT_GRPC_PORT 50000 54999
    [[ -n "${RAY_RUNTIME_ENV_AGENT_PORT:-}" ]] || reserve_launcher_port RAY_RUNTIME_ENV_AGENT_PORT 55000 59999
    [[ -n "${RAY_METRICS_EXPORT_PORT:-}" ]] || reserve_launcher_port RAY_METRICS_EXPORT_PORT 60000 64999
    [[ -n "${RAY_MIN_WORKER_PORT:-}" ]] || reserve_launcher_port_range RAY_MIN_WORKER_PORT 20000 23999 "${RAY_WORKER_PORT_RANGE_SIZE}"
    RAY_MAX_WORKER_PORT=${RAY_MAX_WORKER_PORT:-$((RAY_MIN_WORKER_PORT + RAY_WORKER_PORT_RANGE_SIZE - 1))}

    export RAY_DASHBOARD_PORT RAY_DASHBOARD_AGENT_LISTEN_PORT
    export ORBIT_RAY_ADDRESS="${MASTER_ADDR}:${RAY_HEAD_PORT}"
    RAY_START_LOG="${RAY_TEMP_DIR}/ray-start.log"
    trap cleanup_private_ray EXIT INT TERM

    ray start --head --node-ip-address "${MASTER_ADDR}" --port "${RAY_HEAD_PORT}" \
        --node-manager-port "${RAY_NODE_MANAGER_PORT}" \
        --object-manager-port "${RAY_OBJECT_MANAGER_PORT}" \
        --num-gpus "${RAY_NUM_GPUS}" \
        --num-cpus "${RAY_NUM_CPUS}" \
        --min-worker-port "${RAY_MIN_WORKER_PORT}" \
        --max-worker-port "${RAY_MAX_WORKER_PORT}" \
        --temp-dir "${RAY_TEMP_DIR}" \
        --disable-usage-stats \
        --dashboard-host=127.0.0.1 --dashboard-port="${RAY_DASHBOARD_PORT}" \
        --dashboard-agent-listen-port "${RAY_DASHBOARD_AGENT_LISTEN_PORT}" \
        --dashboard-agent-grpc-port "${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
        --runtime-env-agent-port "${RAY_RUNTIME_ENV_AGENT_PORT}" \
        --metrics-export-port "${RAY_METRICS_EXPORT_PORT}" \
        --block >"${RAY_START_LOG}" 2>&1 &
    RAY_START_PID=$!

    for _ in $(seq 1 120); do
        if ! kill -0 "${RAY_START_PID}" 2>/dev/null; then
            echo "Private Ray head exited before becoming ready; log follows:" >&2
            cat "${RAY_START_LOG}" >&2 || true
            exit 1
        fi
        if _ray_status_ready && _ray_job_server_ready; then
            return 0
        fi
        sleep 1
    done

    echo "Timed out waiting for private Ray job server at http://127.0.0.1:${RAY_DASHBOARD_PORT}; log follows:" >&2
    cat "${RAY_START_LOG}" >&2 || true
    exit 1
}
