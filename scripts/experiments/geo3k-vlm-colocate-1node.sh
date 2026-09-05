#!/bin/bash
#
# geo3k-vlm-colocate-1node — Qwen3-VL-2B-Instruct GRPO on geo3k, 1-node colocated.
#
# 1-node analogue of geo3k-vlm-colocate-2node, sized for the 2B model.
# Mirrors upstream examples/geo3k_vlm/run_geo3k_vlm.sh when invoked as
# `ORBIT_SCRIPT_MODEL_NAME=Qwen3-VL-2B-Instruct ORBIT_SCRIPT_NUM_GPUS=8`:
#   - megatron backend, TP=4, DP=2 across 8 GPUs
#   - colocate (rollout shares the same 8 GPUs as training)
#   - VL-2B MODEL_ARGS come from scripts/models/qwen3-1.7B.sh
#     (upstream maps Qwen3-VL-2B → qwen3-1.7B for the megatron arg block)
#
# Colocate (not disagg) for the same reason as the 2-node recipe:
# orbit' megatron→HF converter only ships LLM mappings; disagg's
# per-update HF round-trip dies on vision_model.* params.
#
# VLM-specific (same as 2-node recipe):
#   - MODEL_ARGS_ROTARY_BASE=5000000 before sourcing qwen3-1.7B.sh.
#   - --megatron-to-hf-mode bridge in MODEL_ARGS.
#   - --multimodal-keys '{"image": "images"}'.
#   - --load $HF_MODEL_DIR; HF_TORCHDIST_DIR unset so launcher skips convert.
#   - No --ref-load, no --use-kl-loss (matches upstream).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ORBIT_REPO=${ORBIT_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}
RECIPE_NAME=$(basename "${BASH_SOURCE[0]}" .sh)

# ---------------------------------------------------------------------------
# Resource metadata — read by orchestrator wrappers for sbatch/k8s/etc.
# ---------------------------------------------------------------------------
EXPERIMENT_NODES=1
EXPERIMENT_TIME=48:00:00

# ---------------------------------------------------------------------------
# Asset metadata — read by orchestrator wrappers for `hf download` step
# ---------------------------------------------------------------------------
HF_CACHE_DIR=${HF_CACHE_DIR:-/data/shared/hf_cache}

HF_MODEL_REPO="Qwen/Qwen3-VL-2B-Instruct"
HF_DATASETS=(
    "chenhegu/geo3k_imgurl"
)
HF_MODEL_DIR="$HF_CACHE_DIR/models/Qwen3-VL-2B-Instruct"
# HF_TORCHDIST_DIR intentionally unset — VLM loads HF directly via bridge.
HF_TRAIN_DATA="$HF_CACHE_DIR/data/geo3k_imgurl/train.parquet"
HF_EVAL_DATA="$HF_CACHE_DIR/data/geo3k_imgurl/test.parquet"

# ---------------------------------------------------------------------------
# train.py args
# ---------------------------------------------------------------------------
MODEL_ARGS_ROTARY_BASE=5000000
# shellcheck disable=SC1090
source "$ORBIT_REPO/scripts/models/qwen3-1.7B.sh"
MODEL_ARGS+=( --megatron-to-hf-mode bridge )

RUN_NAME=${SLURM_JOB_NAME:-$RECIPE_NAME}

CKPT_ARGS=(
   --hf-checkpoint  "$HF_MODEL_DIR"
   --load           "$HF_MODEL_DIR"
)

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
   --qkv-format bshd
   --micro-batch-size 1
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
   --sglang-mem-fraction-static  0.6
)

MISC_ARGS=(
   --colocate
   --attention-dropout 0.0
   --hidden-dropout    0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

WANDB_ARGS=(
   --use-wandb
   --wandb-project orbit
   --wandb-group   "$RUN_NAME"
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

ORBIT_ARGS=(
   "${LAYOUT_ARGS[@]}"
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${MULTIMODAL_ARGS[@]}"
   "${ROLLOUT_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${GRPO_ARGS[@]}"
   "${WANDB_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${EVAL_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${MISC_ARGS[@]}"
   "${FT_ARGS[@]}"
)
