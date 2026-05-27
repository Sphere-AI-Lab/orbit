#!/usr/bin/env bash
# Shell utility predicates and process-env setup. Sourced by every
# scripts/lib helper that needs is_true / is_false.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "Source this file from a launcher instead of running it directly." >&2
    exit 2
fi

is_true() {
    case "${1,,}" in
        1 | true | yes | y | on) return 0 ;;
        *) return 1 ;;
    esac
}

is_false() {
    case "${1,,}" in
        0 | false | no | n | off) return 0 ;;
        *) return 1 ;;
    esac
}

validate_eval_args() {
    if is_true "${DISABLE_EVAL:-0}"; then
        EVAL_ARGS=()
        return 0
    fi
    if [[ ${#EVAL_ARGS[@]} -eq 0 ]]; then
        return 0
    fi

    local i dataset path
    for ((i = 0; i < ${#EVAL_ARGS[@]}; i++)); do
        if [[ "${EVAL_ARGS[$i]}" != "--eval-prompt-data" ]]; then
            continue
        fi

        i=$((i + 1))
        while ((i < ${#EVAL_ARGS[@]})); do
            if [[ "${EVAL_ARGS[$i]}" == --* ]]; then
                break
            fi

            dataset="${EVAL_ARGS[$i]}"
            path="${EVAL_ARGS[$((i + 1))]:-}"
            if [[ -z "${path}" || "${path}" == --* ]]; then
                echo "eval dataset ${dataset} has an empty path; set TEST_JSONL/EVAL_ORBIT_DIR or set DISABLE_EVAL=1" >&2
                exit 2
            fi
            if [[ "${path}" == "/math500.jsonl" || "${path}" == "/aime24.jsonl" || "${path}" == "/amc23.jsonl" ]]; then
                echo "set EVAL_ORBIT_DIR or EVAL_DATA_DIR to a directory containing converted math eval JSONL files, or set DISABLE_EVAL=1" >&2
                exit 2
            fi
            i=$((i + 2))
        done
    done
}

configure_process_env() {
    # CUDA env, no_proxy, NCCL_NVLS_ENABLE, and CUDA_DEVICE_MAX_CONNECTIONS
    # are owned by tool_env.sh, which leaf launchers source first.
    set -e
    export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

    ORBIT_LAUNCHER_XTRACE=${ORBIT_LAUNCHER_XTRACE:-0}
    if is_true "${ORBIT_LAUNCHER_XTRACE}"; then
        set -x
    fi
}
