#!/usr/bin/env bash
# Run PEFT-Arena's math benchmark suite (math500, aime24, amc23) against an
# Orbit-trained PEFT adapter checkpoint, using SGLang as the inference backend
# (because Orbit's venv has SGLang but not vLLM).
#
# Required env vars:
#   ITER_DIR          Path to an iter_NNNNNNN/adapter directory under orbit_ckpts/
#   PEFT_ARENA_ROOT   Optional override for the PEFT-Arena repo path.
#                     Defaults to ${ORBIT_ROOT}/examples/peft_arena/backend, which
#                     is a trimmed vendored snapshot.
#
# Common overrides (defaults shown):
#   OUTPUT_DIR=${ORBIT_ROOT}/eval_results/<run>/<iter>/math
#   NUM_GPUS=1
#   DATA_NAMES=math500,aime24,amc23
#   N_SAMPLING=16
#   TEMPERATURE=0.6
#   MAX_TOKENS_PER_CALL=8192
#   GPU_MEMORY_UTILIZATION=0.7
#   MAX_MODEL_LEN=9216
#   PYTHON_BIN=${ORBIT_WORKSPACE_ROOT:-${HOME}/.cache/orbit/workspace}/orbit-workspace/.venv/bin/python
#
# Example:
#   ITER_DIR=/.../orbit_ckpts/Qwen3-4B-Instruct-2507-BF16_math_oft/iter_0000199/adapter \
#       bash examples/peft_arena/eval/eval-math-peft-arena.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ORBIT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

# SGLang's deep_gemm import requires CUDA_HOME; the standard launcher sources
# tool_env.sh to set it (and PATH/LD_LIBRARY_PATH). Reuse the same setup so
# this wrapper works in the same shells the training launchers do.
# shellcheck source=../../scripts/lib/tool_env.sh
source "${ORBIT_ROOT}/scripts/lib/tool_env.sh"

: "${ITER_DIR:?ITER_DIR (path to iter_*/adapter) is required}"
PEFT_ARENA_ROOT="${PEFT_ARENA_ROOT:-${ORBIT_ROOT}/examples/peft_arena/backend}"
if [ ! -f "${PEFT_ARENA_ROOT}/eval/eval_math.sh" ]; then
    echo "[eval-math-peft-arena] ERROR: PEFT_ARENA_ROOT='${PEFT_ARENA_ROOT}' does not contain eval/eval_math.sh." >&2
    echo "  The vendored copy is expected at \${ORBIT_ROOT}/examples/peft_arena/backend. Did the vendor task complete?" >&2
    exit 1
fi

ITER_DIR="$(cd -- "${ITER_DIR}" && pwd)"

# --- Pre-flight: directory shape -------------------------------------------
if [ ! -f "${ITER_DIR}/adapter_model.safetensors" ]; then
    echo "[eval-math-peft-arena] ERROR: missing adapter_model.safetensors under ${ITER_DIR}" >&2
    echo "  This wrapper only supports adapters written under the post-fix Orbit save path." >&2
    echo "  Rerun training with the current orbit/backends/megatron_utils/peft_utils.py." >&2
    exit 1
fi
if [ ! -f "${ITER_DIR}/adapter_config.json" ]; then
    echo "[eval-math-peft-arena] ERROR: missing adapter_config.json under ${ITER_DIR}" >&2
    exit 1
fi

# --- Pre-flight: peft loadability ------------------------------------------
PYTHON_BIN="${PYTHON_BIN:-${ORBIT_WORKSPACE_ROOT:-${HOME}/.cache/orbit/workspace}/orbit-workspace/.venv/bin/python}"

BASE_PATH="$(${PYTHON_BIN} - <<EOF
from peft import PeftConfig
import sys
cfg = PeftConfig.from_pretrained("${ITER_DIR}")
base = cfg.base_model_name_or_path or ""
if not base:
    sys.exit("adapter_config.json missing base_model_name_or_path")
print(base)
EOF
)"

if [ ! -d "${BASE_PATH}" ]; then
    echo "[eval-math-peft-arena] ERROR: base_model_name_or_path '${BASE_PATH}' does not exist on disk" >&2
    exit 1
fi
echo "[eval-math-peft-arena] Base model: ${BASE_PATH}"

# --- Output dir derivation -------------------------------------------------
ITER_NAME="$(basename "$(dirname "${ITER_DIR}")")"           # iter_NNNNNNN
RUN_NAME="$(basename "$(dirname "$(dirname "${ITER_DIR}")")")" # the run dir under orbit_ckpts/
DEFAULT_OUTPUT_DIR="${ORBIT_ROOT}/eval_results/${RUN_NAME}/${ITER_NAME}/math"
OUTPUT_DIR="${OUTPUT_DIR:-${DEFAULT_OUTPUT_DIR}}"
mkdir -p "${OUTPUT_DIR}"
echo "[eval-math-peft-arena] Output dir: ${OUTPUT_DIR}"

# --- Knobs (forwarded to PEFT-Arena's eval_math.sh) ------------------------
NUM_GPUS="${NUM_GPUS:-1}"
DATA_NAMES="${DATA_NAMES:-math500,aime24,amc23}"
N_SAMPLING="${N_SAMPLING:-16}"
TEMPERATURE="${TEMPERATURE:-0.6}"
MAX_TOKENS_PER_CALL="${MAX_TOKENS_PER_CALL:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-9216}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"

# Sanity: max_tokens_per_call + a tokenized prompt must fit inside the
# model's max_model_len. Reject any combination that obviously cannot work
# rather than wait for SGLang to fail on the first request.
if [ "${MAX_TOKENS_PER_CALL}" -ge "${MAX_MODEL_LEN}" ]; then
    echo "[eval-math-peft-arena] ERROR: MAX_TOKENS_PER_CALL (${MAX_TOKENS_PER_CALL}) >= MAX_MODEL_LEN (${MAX_MODEL_LEN}); leave room for the prompt." >&2
    exit 1
fi

# --- Hand off to PEFT-Arena ------------------------------------------------
exec bash "${PEFT_ARENA_ROOT}/eval/eval_math.sh" \
    --checkpoint_path "${ITER_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_gpus "${NUM_GPUS}" \
    --data_names "${DATA_NAMES}" \
    --n_sampling "${N_SAMPLING}" \
    --temperature "${TEMPERATURE}" \
    --num_samples "${NUM_SAMPLES}" \
    --max_tokens_per_call "${MAX_TOKENS_PER_CALL}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --max_model_len "${MAX_MODEL_LEN}" \
    --use_sglang
