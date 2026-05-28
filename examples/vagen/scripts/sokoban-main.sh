#!/bin/bash
#
# Build the offline sokoban-main dataset for
# vagen-sokoban-main-qwen25vl3b-colocate-1node.
#
# Produces two splits under $MILES_REPO/data/sokoban-main/:
#   train/samples.jsonl  — 10k seeds [1,10000], from sokoban_train_env.yaml
#   eval/samples.jsonl   — 256 map-disjoint-from-train seeds, drawn from
#                          sokoban_val_env.yaml (4096 candidates) via
#                          build_env_dataset --exclude-data
#
# Heldout (map-disjoint-from-train) is the default eval. To skip heldout
# filtering and just expand the val yaml directly, drop the --exclude-data /
# --dedup-within / --target-kept flags from the eval step below.
#
# Idempotent: build_env_dataset stamps yaml_md5 + base_seed + exclude_data_md5
# into dataset_meta.json and short-circuits when nothing changed. Pass
# `--force` (env var FORCE=1) to rebuild from scratch.
#
# Usage:
#   env -u LD_LIBRARY_PATH conda run -n miles \
#       examples/vagen/scripts/sokoban-main.sh
#
# --base-seed 0 matches VAGEN's default (config.get("base_seed", 0)) and the
# launcher passes --seed 0 to miles so train's seed expansion matches.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_REPO=${MILES_REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}

DATASET_NAME=${VAGEN_DATASET_NAME:-sokoban-main}
DATA_ROOT="$MILES_REPO/data/$DATASET_NAME"

TRAIN_YAML="$MILES_REPO/examples/vagen/configs/sokoban_train_env.yaml"
EVAL_YAML="$MILES_REPO/examples/vagen/configs/sokoban_val_env.yaml"

FORCE_FLAG=()
if [[ "${FORCE:-0}" == "1" ]]; then
    FORCE_FLAG=(--force)
fi

cd "$MILES_REPO"

echo "[build_data] train -> $DATA_ROOT/train"
python3 -m examples.vagen.build_env_dataset \
    --yaml       "$TRAIN_YAML" \
    --output-dir "$DATA_ROOT/train" \
    --split      train \
    --base-seed  0 \
    "${FORCE_FLAG[@]}"

# Eval: draw 256 unique maps from the 4096-candidate pool that are NOT in
# train. --dedup-within rejects candidates whose env_uuid duplicates one
# already kept this build, so --target-kept 256 means 256 *unique* maps (vs.
# 256 rows with intra-set duplicates from Sokoban's many-to-one orbit).
# --target-kept fails the build if the pool was too small / overlap was too
# high to assemble a clean heldout split.
echo "[build_data] eval  -> $DATA_ROOT/eval"
python3 -m examples.vagen.build_env_dataset \
    --yaml         "$EVAL_YAML" \
    --output-dir   "$DATA_ROOT/eval" \
    --split        eval \
    --base-seed    0 \
    --exclude-data "$DATA_ROOT/train/samples.jsonl" \
    --dedup-within \
    --target-kept  256 \
    "${FORCE_FLAG[@]}"

echo "[build_data] done. artifacts:"
ls -lh "$DATA_ROOT"/{train,eval}/samples.jsonl 2>/dev/null || true
