#!/usr/bin/env bash
# Evaluate all existing Orbit iter_*/adapter checkpoints in a save directory
# once, write per-iter PEFT-Arena metrics, and emit a curve-friendly summary.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WRAPPER="${EVAL_WRAPPER:-${ORBIT_ROOT}/examples/peft_arena/eval/eval-math-peft-arena.sh}"
SUMMARY_SCRIPT="${ORBIT_ROOT}/tools/summarize_eval_results.py"

: "${SAVE_DIR:?SAVE_DIR (orbit_ckpts/<run>) is required}"
SAVE_DIR="$(cd -- "${SAVE_DIR}" && pwd)"

RUN_NAME="$(basename "${SAVE_DIR}")"
EVAL_RESULTS_ROOT="${EVAL_RESULTS_ROOT:-${ORBIT_ROOT}/eval_results}"
EVAL_RESULTS="${EVAL_RESULTS_ROOT}/${RUN_NAME}"
LOG_DIR="${LOG_DIR:-${ORBIT_ROOT}/logs}"
DATA_NAMES="${DATA_NAMES:-math500,aime24,amc23}"
N_SAMPLING="${N_SAMPLING:-16}"
NUM_GPUS="${NUM_GPUS:-1}"
TEMPERATURE="${TEMPERATURE:-0.6}"
MAX_TOKENS_PER_CALL="${MAX_TOKENS_PER_CALL:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-9216}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"

mkdir -p "${EVAL_RESULTS}" "${LOG_DIR}"

dataset_metrics_exists() {
    local iter_name="$1"
    local dataset="$2"
    find "${EVAL_RESULTS}/${iter_name}" -path "*/${dataset}/*metrics.json" -print -quit 2>/dev/null | grep -q .
}

iter_has_complete_metrics() {
    local iter_name="$1"
    local dataset
    IFS=',' read -r -a _datasets <<< "${DATA_NAMES}"
    for dataset in "${_datasets[@]}"; do
        if ! dataset_metrics_exists "${iter_name}" "${dataset}"; then
            return 1
        fi
    done
    return 0
}

format_duration() {
    local seconds="$1"
    local hours=$((seconds / 3600))
    local minutes=$(((seconds % 3600) / 60))
    local secs=$((seconds % 60))
    if [ "${hours}" -gt 0 ]; then
        printf '%dh%02dm%02ds' "${hours}" "${minutes}" "${secs}"
    else
        printf '%dm%02ds' "${minutes}" "${secs}"
    fi
}

format_eta() {
    local done_count="$1"
    local total_count="$2"
    local started_at="$3"
    local now="$4"
    if [ "${done_count}" -le 0 ]; then
        printf 'unknown'
        return
    fi
    local elapsed=$((now - started_at))
    local remaining=$((total_count - done_count))
    local eta_seconds=$((elapsed * remaining / done_count))
    format_duration "${eta_seconds}"
}

declare -a adapter_dirs=()
for adapter_dir in "${SAVE_DIR}"/iter_*/adapter; do
    [ -d "${adapter_dir}" ] || continue
    [ -f "${adapter_dir}/adapter_model.safetensors" ] || continue
    [ -f "${adapter_dir}/adapter_config.json" ] || continue
    adapter_dirs+=("${adapter_dir}")
done

total_count="${#adapter_dirs[@]}"
echo "[eval-once] found ${total_count} adapter checkpoints under ${SAVE_DIR}"
echo "[eval-once] results under ${EVAL_RESULTS}"
echo "[eval-once] DATA_NAMES=${DATA_NAMES} N_SAMPLING=${N_SAMPLING} NUM_GPUS=${NUM_GPUS}"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && echo "[eval-once] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

if [ "${total_count}" -eq 0 ]; then
    "${PYTHON_BIN:-python}" "${SUMMARY_SCRIPT}" --run-dir "${EVAL_RESULTS}" >/dev/null
    exit 0
fi

started_at="$(date +%s)"
done_count=0
failed_count=0

for adapter_dir in "${adapter_dirs[@]}"; do
    iter_name="$(basename "$(dirname "${adapter_dir}")")"
    done_count=$((done_count + 1))

    if iter_has_complete_metrics "${iter_name}"; then
        now="$(date +%s)"
        eta="$(format_eta "${done_count}" "${total_count}" "${started_at}" "${now}")"
        echo "[eval-once] $(date +%H:%M:%S) ${iter_name}: complete metrics already exist, skipping (progress ${done_count}/${total_count}, ETA ${eta})"
        continue
    fi

    echo "[eval-once] $(date +%H:%M:%S) evaluating ${iter_name} (progress ${done_count}/${total_count})"
    iter_started_at="$(date +%s)"
    log="${LOG_DIR}/eval_once_${RUN_NAME}_${iter_name}.log"
    iter_output_dir="${EVAL_RESULTS}/${iter_name}/math"
    if ITER_DIR="${adapter_dir}" \
       OUTPUT_DIR="${iter_output_dir}" \
       DATA_NAMES="${DATA_NAMES}" \
       N_SAMPLING="${N_SAMPLING}" \
       NUM_GPUS="${NUM_GPUS}" \
       TEMPERATURE="${TEMPERATURE}" \
       MAX_TOKENS_PER_CALL="${MAX_TOKENS_PER_CALL}" \
       GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
       MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
       NUM_SAMPLES="${NUM_SAMPLES}" \
       bash "${WRAPPER}" > "${log}" 2>&1; then
        status="done"
    else
        status="FAIL"
        failed_count=$((failed_count + 1))
    fi

    now="$(date +%s)"
    duration="$((now - iter_started_at))"
    eta="$(format_eta "${done_count}" "${total_count}" "${started_at}" "${now}")"
    echo "[eval-once] $(date +%H:%M:%S) ${status} ${iter_name} duration=$(format_duration "${duration}") (progress ${done_count}/${total_count}, ETA ${eta}, log: ${log})"
done

summary_path="$("${PYTHON_BIN:-python}" "${SUMMARY_SCRIPT}" --run-dir "${EVAL_RESULTS}")"
echo "[eval-once] summary: ${summary_path}"

if [ "${failed_count}" -gt 0 ]; then
    echo "[eval-once] ${failed_count} checkpoint eval(s) failed" >&2
    exit 1
fi
