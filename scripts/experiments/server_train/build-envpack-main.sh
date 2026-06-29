#!/bin/bash
#
# Build envpack Sokoban/FrozenLake JSONL datasets for the server_train
# experiment recipes.
#
# Usage:
#   scripts/experiments/server_train/build-envpack-main.sh sokoban
#   scripts/experiments/server_train/build-envpack-main.sh frozenlake
#
# Requires thirdparty/envpack to be installed in the active Miles environment.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_REPO=${MILES_REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}
ENVPACK_REPO=${ENVPACK_REPO:-"$MILES_REPO/thirdparty/envpack"}
if [[ ! -f "$ENVPACK_REPO/pyproject.toml" ]]; then
    echo "error: envpack repo not found at $ENVPACK_REPO" >&2
    echo "       initialize thirdparty/envpack or set ENVPACK_REPO explicitly" >&2
    exit 1
fi
export PYTHONPATH="$ENVPACK_REPO${PYTHONPATH:+:$PYTHONPATH}"

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 <sokoban|frozenlake>" >&2
    exit 64
fi

ENV_NAME=$1
case "$ENV_NAME" in
    sokoban)
        DATASET_NAME=${ENVPACK_DATASET_NAME:-envpack-sokoban-main}
        TRAIN_YAML="$SCRIPT_DIR/configs/sokoban_train_env.yaml"
        EVAL_YAML="$SCRIPT_DIR/configs/sokoban_val_env.yaml"
        ;;
    frozenlake)
        DATASET_NAME=${ENVPACK_DATASET_NAME:-envpack-frozenlake-main}
        TRAIN_YAML="$SCRIPT_DIR/configs/frozenlake_train_env.yaml"
        EVAL_YAML="$SCRIPT_DIR/configs/frozenlake_val_env.yaml"
        ;;
    *)
        echo "error: unknown env '$ENV_NAME'; expected sokoban or frozenlake" >&2
        exit 64
        ;;
esac

DATA_ROOT=${ENVPACK_DATA_ROOT:-"$MILES_REPO/data/$DATASET_NAME"}

FORCE_FLAG=()
if [[ "${FORCE:-0}" == "1" ]]; then
    FORCE_FLAG=(--force)
fi

cd "$MILES_REPO"

echo "[envpack_data] train -> $DATA_ROOT/train"
python3 -m miles_plugins.envpack_adapter.build_env_dataset \
    --yaml       "$TRAIN_YAML" \
    --output-dir "$DATA_ROOT/train" \
    --split      train \
    --base-seed  0 \
    "${FORCE_FLAG[@]}"

echo "[envpack_data] eval  -> $DATA_ROOT/eval"
python3 -m miles_plugins.envpack_adapter.build_env_dataset \
    --yaml         "$EVAL_YAML" \
    --output-dir   "$DATA_ROOT/eval" \
    --split        eval \
    --base-seed    0 \
    --exclude-data "$DATA_ROOT/train/samples.jsonl" \
    --dedup-within \
    --target-kept  256 \
    "${FORCE_FLAG[@]}"

echo "[envpack_data] done. artifacts:"
ls -lh "$DATA_ROOT"/{train,eval}/samples.jsonl 2>/dev/null || true
