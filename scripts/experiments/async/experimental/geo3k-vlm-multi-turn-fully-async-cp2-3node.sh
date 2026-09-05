#!/bin/bash
#
# geo3k-vlm-multi-turn-fully-async-cp2-3node — CP=2 variant of
# geo3k-vlm-multi-turn-fully-async-3node.
#
# IDENTICAL to geo3k-vlm-multi-turn-fully-async-3node.sh EXCEPT the trainer
# context-parallel path:
#   base : --context-parallel-size 1   (TP4 x CP1 x PP1 -> DP2 on the 8-GPU trainer)
#   THIS : --context-parallel-size 2 + --allgather-cp
#          (TP4 x CP2 x PP1 -> DP1 on the 8-GPU trainer)
#
# Purpose: exercise the stable context-parallel path on fully-async multi-turn
# VLM. CP splits the sequence dim across 2 ranks, so one replica spans all
# 8 trainer GPUs (DP=1). --allgather-cp uses the production CP layout: global
# concat -> contiguous CP chunks -> gather/redistribute logprobs. This avoids
# the default THD zigzag path's per-sample chunk-boundary alignment issues.
#
# Carries the same fixes as the base recipe:
#   - C1: examples/geo3k_vlm/multi_turn/rollout.py now records weight_version per turn
#     (restores the staleness filter).
#   - health-check: 300s timeout + 3 consecutive failures before killing an engine
#     (the base recipe died CLUSTER_DEAD from a single 30s false-positive timeout).
#
# Submit:
#   JOB_NAME=geo3k-async-mt-8b-cp2 TIME=72:00:00 NODES=3 ORBIT_ENV_NAME=orbit \
#   SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true \
#   bash scripts/slurm/submit.sh async/experimental/geo3k-vlm-multi-turn-fully-async-cp2-3node

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ORBIT_REPO=${ORBIT_REPO:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}
RECIPE_NAME=$(basename "${BASH_SOURCE[0]}" .sh)

# Opt into the async driver. ray_lifecycle.sh runs `python3 ${ORBIT_TRAIN_ENTRY:-train.py}`.
export ORBIT_TRAIN_ENTRY=train_async.py

EXPERIMENT_NODES=3
EXPERIMENT_TIME=72:00:00

HF_CACHE_DIR=${HF_CACHE_DIR:-/data/shared/hf_cache}

HF_MODEL_REPO="Qwen/Qwen3-VL-8B-Instruct"
HF_DATASETS=(
    "VeraIsHere/geo3k_imgurl_processed"
)
HF_MODEL_DIR="$HF_CACHE_DIR/models/Qwen3-VL-8B-Instruct"
HF_TRAIN_DATA="$HF_CACHE_DIR/data/geo3k_imgurl_processed/train.parquet"

# ---------------------------------------------------------------------------
# train_async.py args
# ---------------------------------------------------------------------------
MODEL_ARGS_ROTARY_BASE=5000000
# shellcheck disable=SC1090
source "$ORBIT_REPO/scripts/models/qwen3-8B.sh"
MODEL_ARGS+=( --megatron-to-hf-mode bridge )

RUN_NAME=${SLURM_JOB_NAME:-$RECIPE_NAME}

CKPT_ARGS=(
   --hf-checkpoint  "$HF_MODEL_DIR"
   --load           "$HF_MODEL_DIR"
)

MULTIMODAL_ARGS=(
   --multimodal-keys '{"image": "images"}'
)

ASYNC_ROLLOUT_ARGS=(
   --fully-async
   --custom-generate-function-path examples.geo3k_vlm.multi_turn.rollout.generate
   --custom-config-path           examples/geo3k_vlm/multi_turn/geo3k_vlm_multi_turn_config.yaml
   --max-weight-staleness         2
   # pre-sync worker semantics: aborted/stale groups go back to the data buffer
   # for regeneration (the class-based rollout's default is drop)
   --async-unused-samples-handler retry
   --update-weights-interval      1
)

ROLLOUT_ARGS=(
   --prompt-data   "$HF_TRAIN_DATA"
   --input-key     problem
   --label-key     answer
   --apply-chat-template
   --rollout-shuffle
   --rm-type       math
   --num-rollout   3000
   --rollout-batch-size       64
   --n-samples-per-prompt     8
   --rollout-max-response-len 4096
   --rollout-temperature      1.0
   --global-batch-size        512
   --balance-data
)

# Difference vs the base recipe: --context-parallel-size 2 + --allgather-cp.
# TP4 x CP2 x PP1 = 8 -> DP1 on the 8-GPU trainer node.
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --allgather-cp
   # Qwen3-VL hard-requires per-token loss under CP>1: Megatron-Bridge asserts
   # `calculate_per_token_loss` at modelling_qwen3_vl/model.py:203, else every
   # trainer actor dies at init -> CLUSTER_DEAD (this killed j21102). This flag
   # changes loss normalization (per-token vs the default per-sample), so this
   # recipe is positioned as a CP-PATH correctness/perf test (allgather-cp
   # alignment, train_rollout_logprob_abs_diff, throughput/memory) — NOT a
   # loss-curve comparison against the CP1 prefetch recipes, which stay per-sample.
   --calculate-per-token-loss
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

MONITOR_ARGS=(
   --use-rollout-entropy
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static  0.85
   --sglang-server-concurrency   64
   --sglang-cuda-graph-bs        1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256
)

WANDB_ARGS=(
   --use-wandb
   --wandb-team    M3TRL
   --wandb-project async_envpack
   --wandb-group   "$RUN_NAME"
   --disable-wandb-random-suffix
)

FT_ARGS=(
   --use-fault-tolerance
   --rollout-health-check-interval 30
   --rollout-health-check-timeout  300
   --rollout-health-check-first-wait 60
   --rollout-health-check-max-consecutive-failures 3
)

# 1 trainer node (8 GPUs) + 2 sampler nodes (16 GPUs). NO --colocate.
LAYOUT_ARGS=(
   --actor-num-nodes        1
   --actor-num-gpus-per-node 8
   --rollout-num-gpus       16
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout    0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

ORBIT_ARGS=(
   "${LAYOUT_ARGS[@]}"
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${MULTIMODAL_ARGS[@]}"
   "${ASYNC_ROLLOUT_ARGS[@]}"
   "${ROLLOUT_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${GRPO_ARGS[@]}"
   "${MONITOR_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${WANDB_ARGS[@]}"
   "${FT_ARGS[@]}"
   "${MISC_ARGS[@]}"
)
