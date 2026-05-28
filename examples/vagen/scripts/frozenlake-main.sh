#!/bin/bash
#
# Build the offline frozenlake-main dataset for
# vagen-frozenlake-main-colocate-1node (and reusable by the filter variant).
#
# Produces two splits under $MILES_REPO/data/frozenlake-main/:
#   train/samples.jsonl  — 10k seeds [1,10000], from frozenlake_train_env.yaml
#   eval/samples.jsonl   — 256 map-disjoint-from-train seeds, drawn from
#                          frozenlake_val_env.yaml (4096 candidates) via
#                          build_env_dataset --exclude-data
#
# Heldout (map-disjoint-from-train) is the default eval. To skip heldout
# filtering and just expand the val yaml directly, drop the --exclude-data /
# --dedup-within / --target-kept flags from the eval step below.
#
# FrozenLake's generate_random_map (vagen/envs/frozenlake/utils/utils.py)
# uses `np.random.default_rng(seed)` and a BFS-reachability acceptance test.
# The retry is internal to the same RNG object (no next-seed fallback), so
# no patch like Sokoban's _stable_next_seed is needed for determinism. But
# the acceptance test still makes seed -> final_map many-to-one, so eval
# vs train seed-disjointness does NOT guarantee map-disjointness — hence
# the heldout filter is on by default.
#
# Idempotent: build_env_dataset stamps yaml_md5 + base_seed + exclude_data_md5
# + target_kept + dedup_within into dataset_meta.json. FORCE=1 to rebuild.
#
# Usage:
#   env -u LD_LIBRARY_PATH conda run -n miles \
#       examples/vagen/scripts/frozenlake-main.sh
#
# --base-seed 0 matches VAGEN's default (config.get("base_seed", 0)) and the
# launcher passes --seed 0 to miles for VAGEN-main alignment.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_REPO=${MILES_REPO:-$(cd "$SCRIPT_DIR/../../.." && pwd)}

DATASET_NAME=${VAGEN_DATASET_NAME:-frozenlake-main}
DATA_ROOT="$MILES_REPO/data/$DATASET_NAME"

TRAIN_YAML="$MILES_REPO/examples/vagen/configs/frozenlake_train_env.yaml"
EVAL_YAML="$MILES_REPO/examples/vagen/configs/frozenlake_val_env.yaml"

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
# train. --dedup-within makes --target-kept N mean N unique maps; --target-kept
# also fails the build if the pool was too small / overlap was too high.
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
