#!/bin/bash
# Shared Geo3K multi-turn data/rollout overlay for the 04-06 objectives.
#
# The caller must source 02a first so the named argument arrays and
# build_opd_multimodal_miles_args() exist. This file owns no objective,
# optimizer, model, scoring, or parallel-layout values.

if ! declare -F build_opd_multimodal_miles_args >/dev/null; then
   echo "FATAL: source 02a-singleturn-rkld-smoke.sh before the Geo3K multi-turn overlay" >&2
   return 1 2>/dev/null || exit 1
fi

HF_DATASETS=(
   "VeraIsHere/geo3k_imgurl_processed"
)
HF_TRAIN_DATA="$HF_CACHE_DIR/data/geo3k_imgurl_processed/train.parquet"

ROLLOUT_ARGS=(
   --prompt-data "$HF_TRAIN_DATA"
   --input-key problem
   --label-key answer
   --apply-chat-template
   --custom-generate-function-path examples.geo3k_vlm.multi_turn.rollout.generate
   --custom-config-path examples/geo3k_vlm/multi_turn/geo3k_vlm_multi_turn_config.yaml
   --rollout-shuffle
   --num-rollout "$OPD_NUM_ROLLOUT"
   --rollout-batch-size 16
   --n-samples-per-prompt 4
   --rollout-max-response-len 4096
   --rollout-temperature 1.0
   --global-batch-size 64
   --balance-data
)

MONITOR_ARGS+=(--log-multi-turn)

# 02a assembled MILES_ARGS before this overlay replaced the rollout array.
build_opd_multimodal_miles_args
