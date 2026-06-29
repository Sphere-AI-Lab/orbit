#!/bin/bash
#
# geo3k-vlm-8b-disagg-mt-2node — SYNC disagg, multi-turn geo3k.
#
# This is the "everything the fully-async run does EXCEPT the async worker"
# control. It runs the SAME multi-turn custom generate, the SAME processed
# dataset, the SAME temperature (1.0) and the SAME thd/dynamic trainer scoring
# path as geo3k-vlm-multi-turn-fully-async-3node.sh, but on the SYNC driver
# (train.py, default sync rollout function) — NO fully-async background worker,
# NO weight staleness.
#
# Comparisons it enables for train_rollout_logprob_abs_diff:
#   - vs the live fully-async run (20862): the only remaining differences are the
#     async driver + fully-async rollout worker (+ sampler count, a throughput
#     knob) -> isolates the ASYNC MECHANISM.
#   - vs geo3k-vlm-8b-disagg-thd-2node: differs by turns + dataset + temperature
#     -> isolates the multi-turn rollout assembly / data / sampling bundle.
#
# No eval: the processed dataset ships train-only, and the multi-turn generate
# has no evaluation path wired here (matches the fully-async recipe + the
# colocate multi-turn recipe, both eval-less).
#
# Submit:
#   JOB_NAME=geo3k-vlm-8b-disagg-mt NODES=2 SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true \
#   bash scripts/slurm/submit.sh geo3k-vlm-8b-disagg-mt-2node

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_REPO=${MILES_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}
RECIPE_NAME=$(basename "${BASH_SOURCE[0]}" .sh)

EXPERIMENT_NODES=2
EXPERIMENT_TIME=24:00:00

HF_CACHE_DIR=${HF_CACHE_DIR:-/data/shared/hf_cache}
HF_MODEL_REPO="Qwen/Qwen3-VL-8B-Instruct"
# Same dataset as the fully-async run (processed = multi-turn-ready, train-only).
HF_DATASETS=(
    "VeraIsHere/geo3k_imgurl_processed"
)
HF_MODEL_DIR="$HF_CACHE_DIR/models/Qwen3-VL-8B-Instruct"
HF_TRAIN_DATA="$HF_CACHE_DIR/data/geo3k_imgurl_processed/train.parquet"

# ---------------------------------------------------------------------------
# train.py args (Qwen3-VL-8B language tower from qwen3-8B.sh, bridge export)
# ---------------------------------------------------------------------------
MODEL_ARGS_ROTARY_BASE=5000000
# shellcheck disable=SC1090
source "$MILES_REPO/scripts/models/qwen3-8B.sh"
MODEL_ARGS+=( --megatron-to-hf-mode bridge )

RUN_NAME=${SLURM_JOB_NAME:-$RECIPE_NAME}

CKPT_ARGS=(
   --hf-checkpoint  "$HF_MODEL_DIR"
   --load           "$HF_MODEL_DIR"
)
if [[ "${WITH_REF_LOAD:-0}" == "1" ]]; then
   CKPT_ARGS+=( --ref-load "$HF_MODEL_DIR" )
fi

MULTIMODAL_ARGS=(
   --multimodal-keys '{"image": "images"}'
)

# Multi-turn per-sample generation (same custom fn + config as the fully-async
# recipe), temp 1.0 to match it. No --rollout-function-path => default sync
# rollout function (NOT the fully-async worker).
ROLLOUT_ARGS=(
   --prompt-data   "$HF_TRAIN_DATA"
   --input-key     problem
   --label-key     answer
   --apply-chat-template
   --custom-generate-function-path examples.geo3k_vlm_multi_turn.rollout.generate
   --custom-config-path            examples/geo3k_vlm_multi_turn/geo3k_vlm_multi_turn_config.yaml
   --rollout-shuffle
   --rm-type       math
   --num-rollout   3000
   --rollout-batch-size      64
   --n-samples-per-prompt    8
   --rollout-max-response-len 4096
   --rollout-temperature     1.0
   --global-batch-size       512
   --balance-data
)

# thd + dynamic batch (matches the fully-async recipe). max-tokens-per-gpu 16384.
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
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
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout    0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
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
   --rollout-health-check-timeout  30
   --rollout-health-check-first-wait 60
)

LAYOUT_ARGS=(
   --actor-num-nodes        1
   --actor-num-gpus-per-node 8
   --rollout-num-gpus       8
)

MILES_ARGS=(
   "${LAYOUT_ARGS[@]}"
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${MULTIMODAL_ARGS[@]}"
   "${ROLLOUT_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${GRPO_ARGS[@]}"
   "${MONITOR_ARGS[@]}"
   "${WANDB_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${MISC_ARGS[@]}"
   "${FT_ARGS[@]}"
)
