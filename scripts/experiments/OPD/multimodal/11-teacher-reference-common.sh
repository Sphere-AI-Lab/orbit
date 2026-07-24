#!/bin/bash
# Shared two-node eval-only recipe for the Milestone 11 teacher references.
# Source through 11a or 11b; do not submit this file directly.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_REPO=${MILES_REPO:-$(cd "$SCRIPT_DIR/../../../.." && pwd)}

: "${OPD_EVAL_MODEL_NAME:?11a/11b must set OPD_EVAL_MODEL_NAME}"
: "${OPD_EVAL_MODEL_ARGS_FILE:?11a/11b must set OPD_EVAL_MODEL_ARGS_FILE}"

EXPERIMENT_NODES=2
EXPERIMENT_TIME=24:00:00

HF_CACHE_DIR=${HF_CACHE_DIR:-/data/shared/hf_cache}
HF_MODEL_REPO="Qwen/$OPD_EVAL_MODEL_NAME"
HF_DATASETS=(
   "VeraIsHere/geo3k_imgurl_processed"
)
HF_MODEL_DIR="$HF_CACHE_DIR/models/$OPD_EVAL_MODEL_NAME"
HF_TRAIN_DATA="$HF_CACHE_DIR/data/geo3k_imgurl_processed/train.parquet"

MODEL_ARGS_ROTARY_BASE=5000000
# shellcheck disable=SC1090
source "$MILES_REPO/scripts/models/$OPD_EVAL_MODEL_ARGS_FILE"
MODEL_ARGS+=(--megatron-to-hf-mode bridge)

OPD_EVAL_NUM_PROMPTS=${OPD_EVAL_NUM_PROMPTS:-30}
OPD_EVAL_SEED=${OPD_EVAL_SEED:-20260720}
OPD_EVAL_INTERVAL=1
OPD_EVAL_MAX_CONTEXT_LEN=${OPD_EVAL_MAX_CONTEXT_LEN:-12000}
OPD_FIXED_EVAL_MANIFEST=${OPD_FIXED_EVAL_MANIFEST:-"$HF_CACHE_DIR/data/geo3k_imgurl_processed/opd_eval_seed${OPD_EVAL_SEED}_n${OPD_EVAL_NUM_PROMPTS}.parquet"}
OPD_EVAL_TP_SIZE=${OPD_EVAL_TP_SIZE:-8}
OPD_EVAL_SGLANG_MEM_FRACTION=${OPD_EVAL_SGLANG_MEM_FRACTION:-0.80}
RUN_NAME=${WANDB_RUN_NAME:-"opd-mm-11-teacher-reference-${OPD_EVAL_MODEL_NAME,,}-n${OPD_EVAL_NUM_PROMPTS}"}

LAYOUT_ARGS=(
   --actor-num-nodes 1
   --actor-num-gpus-per-node 8
   --rollout-num-gpus 8
)

CKPT_ARGS=(
   --hf-checkpoint "$HF_MODEL_DIR"
   --load "$HF_MODEL_DIR"
)

MULTIMODAL_ARGS=(
   --multimodal-keys '{"image": "images"}'
)

ROLLOUT_ARGS=(
   --prompt-data "$OPD_FIXED_EVAL_MANIFEST"
   --input-key problem
   --label-key answer
   --apply-chat-template
   --custom-generate-function-path examples.geo3k_vlm_multi_turn.rollout.generate
   --custom-config-path examples/geo3k_vlm_multi_turn/geo3k_vlm_multi_turn_config.yaml
   --num-rollout 0
   --rollout-batch-size 1
   --n-samples-per-prompt 1
   --rollout-max-response-len "$OPD_EVAL_MAX_CONTEXT_LEN"
   --rollout-max-context-len "$OPD_EVAL_MAX_CONTEXT_LEN"
   --rollout-temperature 0
   --global-batch-size 1
)

RM_ARGS=(
   --rm-type math
)

ALGO_ARGS=(
   --advantage-estimator grpo
   --kl-coef 0
   --kl-loss-coef 0
   --entropy-coef 0
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
   # --num-rollout 0 makes miles derive train_iters=0, and Megatron's
   # OptimizerParamScheduler asserts lr_decay_steps > 0 during actor init even
   # though no optimizer step ever runs. Pin a positive, inert horizon.
   --lr-decay-iters 1
)

PERF_ARGS=(
   --tensor-model-parallel-size "$OPD_EVAL_TP_SIZE"
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

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 8
   --sglang-mem-fraction-static "$OPD_EVAL_SGLANG_MEM_FRACTION"
)

MONITOR_ARGS=(
   --log-multi-turn
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

WANDB_ARGS=(
   --use-wandb
   --wandb-team M3TRL
   --wandb-project OPD
   --wandb-group "$RUN_NAME"
   --disable-wandb-random-suffix
)

FT_ARGS=(
   --use-fault-tolerance
   --rollout-health-check-interval 30
   --rollout-health-check-timeout 30
   --rollout-health-check-first-wait 60
)

MILES_ARGS=(
   "${LAYOUT_ARGS[@]}"
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${MULTIMODAL_ARGS[@]}"
   "${ROLLOUT_ARGS[@]}"
   "${RM_ARGS[@]}"
   "${ALGO_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${MONITOR_ARGS[@]}"
   "${WANDB_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${MISC_ARGS[@]}"
   "${FT_ARGS[@]}"
)

# shellcheck disable=SC1091
source "$SCRIPT_DIR/11-fixed-eval-overlay.sh"
