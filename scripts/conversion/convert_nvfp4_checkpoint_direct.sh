#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/tool_env.sh"

usage() {
    echo "Usage: bash scripts/conversion/convert_nvfp4_checkpoint_direct.sh <hf_model> [megatron_path]"
    echo "       DEFAULT_HF_MODEL=/path/to/model bash scripts/conversion/convert_nvfp4_checkpoint_direct.sh"
    echo "Environment: DEBUG_LAYER_RANGE=START:END optionally writes a partial debug checkpoint"
}

if [[ $# -gt 2 ]]; then
    usage
    exit 1
fi

if [[ $# -eq 0 && -z "${DEFAULT_HF_MODEL:-}" ]]; then
    usage
    exit 1
fi

HF_MODEL="${1:-${DEFAULT_HF_MODEL:-}}"
if [[ $# -ge 2 ]]; then
    MEGATRON_PATH="$2"
else
    MEGATRON_PATH="$(default_output_path "${HF_MODEL}")"
fi

CACHE_ROOT="${MEGATRON_BRIDGE_NVFP4_CACHE_ROOT:-${ORBIT_CACHE_ROOT:-/tmp/orbit_low_precision}}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg_cache}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
export MEGATRON_BRIDGE_DIRECT_SPILL_DIR="${MEGATRON_BRIDGE_DIRECT_SPILL_DIR:-${CACHE_ROOT}/nvfp4_spill}"
export MEGATRON_BRIDGE_DIRECT_STORAGE_WRITERS_PER_RANK="${MEGATRON_BRIDGE_DIRECT_STORAGE_WRITERS_PER_RANK:-16}"

mkdir -p \
    "${HF_HOME}" \
    "${HF_HUB_CACHE}" \
    "${HF_DATASETS_CACHE}" \
    "${XDG_CACHE_HOME}" \
    "${TORCH_HOME}" \
    "${TMPDIR}" \
    "${MEGATRON_BRIDGE_DIRECT_SPILL_DIR}" \
    "$(dirname "${MEGATRON_PATH}")"

PY_ARGS=(--hf-model-path "${HF_MODEL}" --megatron-path "${MEGATRON_PATH}")
if [[ -n "${DEBUG_LAYER_RANGE:-}" ]]; then
    PY_ARGS+=(--debug-layer-range "${DEBUG_LAYER_RANGE}")
fi

echo "HF NVFP4 -> Megatron NVFP4 (direct-write)"
echo "Source:      ${HF_MODEL}"
echo "Destination: ${MEGATRON_PATH}"
echo "Cache root:  ${CACHE_ROOT}"
echo "Spill dir:   ${MEGATRON_BRIDGE_DIRECT_SPILL_DIR}"
if [[ -n "${DEBUG_LAYER_RANGE:-}" ]]; then
    echo "Debug range: ${DEBUG_LAYER_RANGE}"
fi

python3 tools/convert_nvfp4_checkpoint_direct.py "${PY_ARGS[@]}"
