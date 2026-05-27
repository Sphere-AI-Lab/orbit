#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/tool_env.sh"

usage() {
    echo "Usage: bash scripts/conversion/convert_int4_checkpoint_direct.sh <hf_model> [megatron_path]"
    echo "       DEFAULT_HF_MODEL=/path/to/model bash scripts/conversion/convert_int4_checkpoint_direct.sh"
    echo "Environment: GROUP_SIZE=<int> optionally overrides inferred INT4 group size"
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
    MEGATRON_PATH="$(default_output_path "${HF_MODEL}" "-INT4-direct")"
fi

mkdir -p "$(dirname "${MEGATRON_PATH}")"

PY_ARGS=(--hf-model-path "${HF_MODEL}" --megatron-path "${MEGATRON_PATH}")
if [[ -n "${GROUP_SIZE:-}" ]]; then
    PY_ARGS+=(--group-size "${GROUP_SIZE}")
fi

echo "HF INT4 -> Megatron INT4 (direct-write)"
echo "Source:      ${HF_MODEL}"
echo "Destination: ${MEGATRON_PATH}"
if [[ -n "${GROUP_SIZE:-}" ]]; then
    echo "Group size:  ${GROUP_SIZE}"
fi

python3 tools/convert_int4_checkpoint_direct.py "${PY_ARGS[@]}"
