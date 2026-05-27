#!/bin/bash

: "${ORBIT_CONVERSION_PROGRESS:=1}"
: "${ORBIT_CONVERSION_PROGRESS_INTERVAL:=10}"

orbit_progress_truthy() {
    case "${1:-}" in
        "" | 0 | false | False | FALSE | no | No | NO | off | Off | OFF)
            return 1
            ;;
        *)
            return 0
            ;;
    esac
}

orbit_format_elapsed() {
    local elapsed_seconds="${1:-0}"
    printf '%02d:%02d:%02d' \
        "$((elapsed_seconds / 3600))" \
        "$(((elapsed_seconds % 3600) / 60))" \
        "$((elapsed_seconds % 60))"
}

orbit_format_num_bytes() {
    local bytes="${1:-0}"
    local -a units=(B KiB MiB GiB TiB PiB)
    local unit_index=0
    local whole="${bytes}"
    local frac=0

    while (( whole >= 1024 && unit_index < ${#units[@]} - 1 )); do
        frac=$(( (whole % 1024) * 10 / 1024 ))
        whole=$(( whole / 1024 ))
        unit_index=$(( unit_index + 1 ))
    done

    printf '%s.%s %s' "${whole}" "${frac}" "${units[$unit_index]}"
}

orbit_snapshot_output_dir() {
    local output_path="$1"

    if [[ ! -e "${output_path}" ]]; then
        echo "0 0"
        return 0
    fi

    local snapshot
    snapshot="$(find "${output_path}" -type f -printf '%s\n' 2>/dev/null | awk '
        BEGIN { count = 0; sum = 0 }
        { count += 1; sum += $1 }
        END { printf "%d %d", count, sum }
    ')"

    if [[ -z "${snapshot}" ]]; then
        echo "0 0"
    else
        echo "${snapshot}"
    fi
}

start_conversion_progress_monitor() {
    local output_path="$1"
    local interval="${ORBIT_CONVERSION_PROGRESS_INTERVAL}"

    if ! orbit_progress_truthy "${ORBIT_CONVERSION_PROGRESS}"; then
        return 0
    fi

    if [[ ! "${interval}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        interval=10
        ORBIT_CONVERSION_PROGRESS_INTERVAL="${interval}"
    fi

    ORBIT_CONVERSION_PROGRESS_START_TS="$(date +%s)"
    ORBIT_CONVERSION_PROGRESS_OUTPUT_PATH="${output_path}"

    echo "Save progress monitor: ${ORBIT_CONVERSION_PROGRESS} (interval=${interval}s)"

    (
        last_snapshot=""
        while true; do
            sleep "${interval}"
            snapshot="$(orbit_snapshot_output_dir "${output_path}")"
            if [[ "${snapshot}" != "${last_snapshot}" ]]; then
                read -r file_count total_bytes <<< "${snapshot}"
                now="$(date +%s)"
                elapsed="$((now - ORBIT_CONVERSION_PROGRESS_START_TS))"
                printf '[save progress] elapsed %s | files %s | written %s\n' \
                    "$(orbit_format_elapsed "${elapsed}")" \
                    "${file_count}" \
                    "$(orbit_format_num_bytes "${total_bytes}")"
                last_snapshot="${snapshot}"
            fi
        done
    ) &
    ORBIT_CONVERSION_PROGRESS_MONITOR_PID=$!
}

stop_conversion_progress_monitor() {
    local output_path="${1:-${ORBIT_CONVERSION_PROGRESS_OUTPUT_PATH:-}}"
    local now
    local elapsed
    local start_ts
    local file_count
    local total_bytes

    if ! orbit_progress_truthy "${ORBIT_CONVERSION_PROGRESS}"; then
        return 0
    fi

    if [[ -n "${ORBIT_CONVERSION_PROGRESS_MONITOR_PID:-}" ]]; then
        kill "${ORBIT_CONVERSION_PROGRESS_MONITOR_PID}" 2>/dev/null || true
        wait "${ORBIT_CONVERSION_PROGRESS_MONITOR_PID}" 2>/dev/null || true
        unset ORBIT_CONVERSION_PROGRESS_MONITOR_PID
    fi

    if [[ -n "${output_path}" ]]; then
        read -r file_count total_bytes <<< "$(orbit_snapshot_output_dir "${output_path}")"
        now="$(date +%s)"
        start_ts="${ORBIT_CONVERSION_PROGRESS_START_TS:-${now}}"
        elapsed="$((now - start_ts))"
        printf '[final save progress] elapsed %s | files %s | written %s\n' \
            "$(orbit_format_elapsed "${elapsed}")" \
            "${file_count}" \
            "$(orbit_format_num_bytes "${total_bytes}")"
    fi

    unset ORBIT_CONVERSION_PROGRESS_OUTPUT_PATH
    unset ORBIT_CONVERSION_PROGRESS_START_TS
}
