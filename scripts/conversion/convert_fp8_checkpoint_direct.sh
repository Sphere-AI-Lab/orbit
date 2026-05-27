#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/tool_env.sh"

usage() {
    echo "Usage: bash scripts/conversion/convert_fp8_checkpoint_direct.sh <hf_model> [megatron_path]"
    echo "       DEFAULT_HF_MODEL=/path/to/model bash scripts/conversion/convert_fp8_checkpoint_direct.sh"
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
    MEGATRON_PATH="$(default_output_path "${HF_MODEL}" "-FP8-direct")"
fi

CACHE_ROOT="${MEGATRON_BRIDGE_FP8_CACHE_ROOT:-${ORBIT_CACHE_ROOT:-/tmp}}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${CACHE_ROOT}}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg_cache}"
export TMPDIR="${TMPDIR:-${CACHE_ROOT}/tmp}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"

mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${XDG_CACHE_HOME}" "${TMPDIR}"
mkdir -p "$(dirname "${MEGATRON_PATH}")"

echo "HF FP8 -> Megatron FP8 (direct-write)"
echo "Source:      ${HF_MODEL}"
echo "Destination: ${MEGATRON_PATH}"
echo "Cache root:  ${CACHE_ROOT}"

python3 tools/convert_fp8_checkpoint_direct.py \
    --hf-model-path "${HF_MODEL}" \
    --megatron-path "${MEGATRON_PATH}"
