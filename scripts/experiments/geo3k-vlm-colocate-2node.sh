#!/bin/bash
#
# geo3k-vlm-colocate-2node — Qwen3-VL-8B-Instruct GRPO on geo3k, 2-node colocated.
#
# We are forced into colocate mode for VLM because orbit' megatron→HF weight
# converter only ships LLM mappings: dispatch in
# orbit/backends/megatron_utils/megatron_to_hf/__init__.py matches "qwen3"
# and routes to convert_qwen2_to_hf, which has no vision_model.* entries.
# Disagg requires that converter to round-trip the full ViT+LLM state at
# every update_weights — which is exactly where geo3k-vlm-disagg-2node
# died (j11465: "Unknown parameter name module.module.vision_model.
# patch_embed.proj.weight"). Colocate trains and serves on the same GPUs
# so the broadcast-via-HF step is bypassed; this matches the upstream
# examples/geo3k_vlm/run_geo3k_vlm.sh shape.
#
# Layout (2 nodes × 8 GPUs, all 16 GPUs shared by train + rollout):
#   - training: actor-num-nodes=2, actor-num-gpus-per-node=8 (TP=4, DP=4)
#   - rollout:  rollout-num-gpus=16, per-engine=1 (16 SGLang engines)
#
# VLM-specific (same as the disagg recipe, kept for parity):
#   - MODEL_ARGS_ROTARY_BASE=5000000 before sourcing qwen3-8B.sh.
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
EXPERIMENT_NODES=2
EXPERIMENT_TIME=48:00:00

# ---------------------------------------------------------------------------
# Asset metadata — read by orchestrator wrappers for `hf download` step
# ---------------------------------------------------------------------------
HF_CACHE_DIR=${HF_CACHE_DIR:-/data/shared/hf_cache}

HF_MODEL_REPO="Qwen/Qwen3-VL-8B-Instruct"
HF_DATASETS=(
    "chenhegu/geo3k_imgurl"
)
HF_MODEL_DIR="$HF_CACHE_DIR/models/Qwen3-VL-8B-Instruct"
# HF_TORCHDIST_DIR intentionally unset — VLM loads HF directly via bridge.
HF_TRAIN_DATA="$HF_CACHE_DIR/data/geo3k_imgurl/train.parquet"
HF_EVAL_DATA="$HF_CACHE_DIR/data/geo3k_imgurl/test.parquet"

# ---------------------------------------------------------------------------
# train.py args
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
   --wandb-team    M3TRL
   --wandb-project async_envpack
   --wandb-group   "$RUN_NAME"
   --disable-wandb-random-suffix
)

FT_ARGS=(
   --use-fault-tolerance
   --rollout-health-check-interval 30
   --rollout-health-check-first-wait 60
   # timeout (300s) + max-consecutive-failures (3) now come from the safe argparse
   # defaults (raised from 30s/1). Colocate engines share GPUs and legitimately stall
   # during big rollouts / weight updates; a 1-strike 30s check false-killed them and
   # cascaded the whole job (j21091/21092: 10 engine kills, 0 steps, no real OOM/crash).
)

LAYOUT_ARGS=(
   --actor-num-nodes        2
   --actor-num-gpus-per-node 8
   --rollout-num-gpus       16
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
