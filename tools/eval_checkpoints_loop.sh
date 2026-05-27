#!/usr/bin/env bash
# Watch an Orbit save dir for new iter_*/adapter checkpoints and run the
# math eval wrapper against each one as it appears. Designed for a separate
# GPU (or host) from the training job, so eval doesn't contend with training.
#
# Required env:
#   SAVE_DIR        Path to a save dir, e.g.
#                   orbit_ckpts/Qwen3-4B-Instruct-2507-BF16_peft_arena_openr1_50k_oft
# Common overrides:
#   POLL_INTERVAL   Seconds between scans (default 60)
#   CUDA_VISIBLE_DEVICES   Pin the eval to specific GPU(s). Honored by
#                          the vendored eval_math.sh's auto-derivation.
#   N_SAMPLING      Forward to wrapper (default 16 = pass@16)
#   DATA_NAMES      Forward to wrapper (default math500,aime24,amc23)
#   MAX_ITERS       Stop after this many evals (default unbounded)
#
# Example:
#   CUDA_VISIBLE_DEVICES=4 \
#   SAVE_DIR=$(pwd)/orbit_ckpts/Qwen3-4B-Instruct-2507-BF16_peft_arena_openr1_50k_oft \
#       bash tools/eval_checkpoints_loop.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WRAPPER="${ORBIT_ROOT}/examples/peft_arena/eval/eval-math-peft-arena.sh"
SUMMARY_SCRIPT="${ORBIT_ROOT}/tools/summarize_eval_results.py"

: "${SAVE_DIR:?SAVE_DIR (orbit_ckpts/<run>) is required}"
SAVE_DIR="$(cd -- "${SAVE_DIR}" && pwd)"
POLL_INTERVAL=${POLL_INTERVAL:-60}
MAX_ITERS=${MAX_ITERS:-}
DATA_NAMES="${DATA_NAMES:-math500,aime24,amc23}"
N_SAMPLING="${N_SAMPLING:-16}"
NUM_GPUS="${NUM_GPUS:-1}"
TEMPERATURE="${TEMPERATURE:-0.6}"
MAX_TOKENS_PER_CALL="${MAX_TOKENS_PER_CALL:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-9216}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"

RUN_NAME="$(basename "${SAVE_DIR}")"
EVAL_RESULTS_ROOT="${EVAL_RESULTS_ROOT:-${ORBIT_ROOT}/eval_results}"
EVAL_RESULTS="${EVAL_RESULTS_ROOT}/${RUN_NAME}"
mkdir -p "${EVAL_RESULTS}"

declare -A done_iters
done_count=0

iter_has_metrics() {
    local iter_name="$1"
    local dataset
    IFS=',' read -r -a _datasets <<< "${DATA_NAMES}"
    for dataset in "${_datasets[@]}"; do
        if ! find "${EVAL_RESULTS}/${iter_name}" -path "*/${dataset}/*metrics.json" -print -quit 2>/dev/null | grep -q .; then
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
    local started_at="$2"
    local now="$3"
    if [ "${done_count}" -le 0 ]; then
        printf 'unknown'
        return
    fi
    local elapsed=$((now - started_at))
    local eta_seconds=$((elapsed / done_count))
    format_duration "${eta_seconds}"
}

echo "[eval-loop] watching ${SAVE_DIR}"
echo "[eval-loop] poll interval ${POLL_INTERVAL}s, results under ${EVAL_RESULTS}"
echo "[eval-loop] DATA_NAMES=${DATA_NAMES} N_SAMPLING=${N_SAMPLING} NUM_GPUS=${NUM_GPUS}"
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && echo "[eval-loop] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
started_at="$(date +%s)"

while true; do
    for adapter_dir in "${SAVE_DIR}"/iter_*/adapter; do
        [ -d "${adapter_dir}" ] || continue
        # Atomicity: only act once both files exist (training writes config last)
        [ -f "${adapter_dir}/adapter_model.safetensors" ] || continue
        [ -f "${adapter_dir}/adapter_config.json" ]      || continue

        iter_name="$(basename "$(dirname "${adapter_dir}")")"
        [ -n "${done_iters[${iter_name}]:-}" ] && continue

        if iter_has_metrics "${iter_name}"; then
            echo "[eval-loop] $(date +%H:%M:%S) ${iter_name}: complete metrics already exist, skipping"
            done_iters[${iter_name}]=1
            continue
        fi

        echo "[eval-loop] $(date +%H:%M:%S) evaluating ${iter_name}"
        iter_started_at="$(date +%s)"
        log="${ORBIT_ROOT}/logs/eval_loop_${RUN_NAME}_${iter_name}.log"
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
        fi
        done_iters[${iter_name}]=1
        done_count=$((done_count + 1))
        now="$(date +%s)"
        duration="$((now - iter_started_at))"
        eta="$(format_eta "${done_count}" "${started_at}" "${now}")"
        echo "[eval-loop] $(date +%H:%M:%S) ${status} ${iter_name} duration=$(format_duration "${duration}") avg=$(format_eta "${done_count}" "${started_at}" "${now}") (evaluated ${done_count}, next-checkpoint ETA ${eta}, log: ${log})"
        "${PYTHON_BIN:-python}" "${SUMMARY_SCRIPT}" --run-dir "${EVAL_RESULTS}" >/dev/null

        if [ -n "${MAX_ITERS}" ] && [ "${done_count}" -ge "${MAX_ITERS}" ]; then
            echo "[eval-loop] reached MAX_ITERS=${MAX_ITERS}, exiting"
            exit 0
        fi
    done
    sleep "${POLL_INTERVAL}"
done
