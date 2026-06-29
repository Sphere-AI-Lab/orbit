#!/bin/bash
#
# geo3k-vlm-8b-disagg-thd-2node — 8B-VLM, 2-node SYNC disaggregated
# single-turn Geo3K diagnostic.
#
# This is the one retained single-turn disagg comparison recipe. It uses the
# same bridge weight-update path as the multi-turn disagg recipe, but keeps the
# original single-turn Geo3K dataset and evaluation path. The trainer uses THD
# dynamic batching so rollout/trainer logprob diagnostics are comparable with
# the multi-turn and fully-async recipes.
#
# Submit (no eval omission needed — single-turn keeps eval):
#   JOB_NAME=geo3k-vlm-8b-disagg-thd NODES=2 SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true \
#   bash scripts/slurm/submit.sh geo3k-vlm-8b-disagg-thd-2node

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_REPO=${MILES_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}
RECIPE_NAME=$(basename "${BASH_SOURCE[0]}" .sh)

EXPERIMENT_NODES=2
EXPERIMENT_TIME=24:00:00

HF_CACHE_DIR=${HF_CACHE_DIR:-/data/shared/hf_cache}
HF_MODEL_REPO="Qwen/Qwen3-VL-8B-Instruct"
HF_DATASETS=(
    "chenhegu/geo3k_imgurl"
)
HF_MODEL_DIR="$HF_CACHE_DIR/models/Qwen3-VL-8B-Instruct"
HF_TRAIN_DATA="$HF_CACHE_DIR/data/geo3k_imgurl/train.parquet"
HF_EVAL_DATA="$HF_CACHE_DIR/data/geo3k_imgurl/test.parquet"

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

ROLLOUT_ARGS=(
   --prompt-data   "$HF_TRAIN_DATA"
   --input-key     problem
   --label-key     answer
   --apply-chat-template
   --rollout-shuffle
   --rm-type       math
   --num-rollout   3000
   --rollout-batch-size      64
   --n-samples-per-prompt    8
   --rollout-max-response-len 4096
   --rollout-temperature     0.8
   --global-batch-size       512
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data geo3k_imgurl "$HF_EVAL_DATA"
   --n-samples-per-eval-prompt 1
   --eval-max-response-len     4096
)

# THD + dynamic batch matches the fully-async recipes. max-tokens-per-gpu 16384
# matches geo3k-vlm-multi-turn-fully-async-3node.sh.
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
   "${EVAL_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${MISC_ARGS[@]}"
   "${FT_ARGS[@]}"
)
