#!/bin/bash
#
# geo3k-vlm-multi-turn-colocate-1node — Qwen3-VL-2B-Instruct GRPO on
# geo3k multi-turn, 1-node colocated.
#
# Based on geo3k-vlm-colocate-1node, with the multi-turn custom rollout from
# examples/geo3k_vlm/multi_turn/run_geo3k_vlm_multi_turn.py.
#
# Defaults are full-training sized. For smoke/debug runs, override the rollout
# sizing from the launcher environment:
#   MILES_SCRIPT_NUM_ROLLOUT=1
#   MILES_SCRIPT_ROLLOUT_BATCH_SIZE=8
#   MILES_SCRIPT_N_SAMPLES_PER_PROMPT=2
#
# VLM-specific:
#   - MODEL_ARGS_ROTARY_BASE=5000000 before sourcing qwen3-1.7B.sh.
#   - --megatron-to-hf-mode bridge in MODEL_ARGS.
#   - --multimodal-keys '{"image": "images"}'.
#   - --load $HF_MODEL_DIR; HF_TORCHDIST_DIR unset so launcher skips convert.
#   - Colocate because disagg's megatron->HF round-trip has no VL converter.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_REPO=${MILES_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}
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
   "VeraIsHere/geo3k_imgurl_processed"
)
HF_MODEL_DIR="$HF_CACHE_DIR/models/Qwen3-VL-2B-Instruct"
# HF_TORCHDIST_DIR intentionally unset — VLM loads HF directly via bridge.
HF_TRAIN_DATA="$HF_CACHE_DIR/data/geo3k_imgurl_processed/train.parquet"

# ---------------------------------------------------------------------------
# train.py args
# ---------------------------------------------------------------------------
MODEL_ARGS_ROTARY_BASE=5000000
# shellcheck disable=SC1090
source "$MILES_REPO/scripts/models/qwen3-1.7B.sh"
MODEL_ARGS+=( --megatron-to-hf-mode bridge )

RUN_NAME=${SLURM_JOB_NAME:-$RECIPE_NAME}

NUM_ROLLOUT=${MILES_SCRIPT_NUM_ROLLOUT:-3000}
ROLLOUT_BATCH_SIZE=${MILES_SCRIPT_ROLLOUT_BATCH_SIZE:-64}
N_SAMPLES_PER_PROMPT=${MILES_SCRIPT_N_SAMPLES_PER_PROMPT:-8}
GLOBAL_BATCH_SIZE=${MILES_SCRIPT_GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}

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
   --custom-generate-function-path examples.geo3k_vlm.multi_turn.rollout.generate
   --custom-config-path examples/geo3k_vlm/multi_turn/geo3k_vlm_multi_turn_config.yaml
   --rollout-shuffle
   --rm-type       math
   --num-rollout   "$NUM_ROLLOUT"
   --rollout-batch-size      "$ROLLOUT_BATCH_SIZE"
   --n-samples-per-prompt    "$N_SAMPLES_PER_PROMPT"
   --rollout-max-response-len 4096
   --rollout-temperature     1
   --global-batch-size       "$GLOBAL_BATCH_SIZE"
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
   --wandb-project miles-imp
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

MILES_ARGS=(
   "${LAYOUT_ARGS[@]}"
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${MULTIMODAL_ARGS[@]}"
   "${ROLLOUT_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${GRPO_ARGS[@]}"
   "${WANDB_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${MISC_ARGS[@]}"
   "${FT_ARGS[@]}"
)
