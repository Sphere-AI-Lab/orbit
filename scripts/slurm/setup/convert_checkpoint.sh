#!/bin/bash
#
# convert_checkpoint.sh — HF checkpoint -> Megatron `torch_dist` format.
# Per docs/getting-started/quick-start.md §3. Single GPU is enough for ≤9B
# dense models; for larger, wrap in `torchrun --nproc-per-node 8`.
#
# Usage (salloc 1 GPU; ~5 min for Qwen3-4B):
#   salloc --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=30 --pty bash
#   bash scripts/slurm/setup/convert_checkpoint.sh
#
# Optional env vars:
#   HF_CACHE_DIR    root of HF downloads          [/data/shared/hf_cache]
#   MODEL_FAMILY    scripts/models/*.sh stem      [qwen3-4B]
#   HF_DIR          source HF snapshot            [$HF_CACHE_DIR/models/Qwen3-4B]
#   SAVE_DIR        target torch_dist             [$HF_CACHE_DIR/models/Qwen3-4B_torch_dist]
#   MILES_REPO      this repo                     [$PWD]
#   MILES_ENV_NAME  conda env                     [miles]
#   THIRDPARTY_DIR  repo's thirdparty/            [$MILES_REPO/thirdparty]

set -euo pipefail

HF_CACHE_DIR=${HF_CACHE_DIR:-/data/shared/hf_cache}
MODEL_FAMILY=${MODEL_FAMILY:-qwen3-4B}
HF_DIR=${HF_DIR:-$HF_CACHE_DIR/models/Qwen3-4B}
SAVE_DIR=${SAVE_DIR:-$HF_CACHE_DIR/models/Qwen3-4B_torch_dist}
MILES_REPO=${MILES_REPO:-$PWD}
MILES_ENV_NAME=${MILES_ENV_NAME:-miles}
THIRDPARTY_DIR=${THIRDPARTY_DIR:-$MILES_REPO/thirdparty}
CONDA_ROOT=${CONDA_ROOT:-/data/shared/conda/miniconda3}

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$MILES_ENV_NAME"

# shellcheck disable=SC1090
source "$MILES_REPO/scripts/models/${MODEL_FAMILY}.sh"

if [ -f "$SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
    echo "[skip] $SAVE_DIR already converted"
    exit 0
fi
mkdir -p "$SAVE_DIR"

export PYTHONPATH="$THIRDPARTY_DIR/Megatron-LM${PYTHONPATH:+:$PYTHONPATH}"

echo "[convert] $HF_DIR -> $SAVE_DIR  (family=$MODEL_FAMILY)"
python "$MILES_REPO/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_DIR" \
    --save "$SAVE_DIR"

echo "[done] $SAVE_DIR"
