#!/bin/bash
#
# geo3k-vlm-disagg-2node — first VLM smoke test on the slinky slurm cluster.
#
# Recipe is orchestrator-agnostic (pure config, no side effects). Mirrors
# the layout of qwen3-4B-disagg-2node.sh:
#   - node 0: 8 GPUs Megatron training  (--actor-num-nodes 1 --actor-num-gpus-per-node 8)
#   - node 1: 8 GPUs SGLang rollout     (--rollout-num-gpus 8, --rollout-num-gpus-per-engine 2)
#
# VLM-specific deltas vs. text-only disagg recipe (cross-check against
# examples/geo3k_vlm/run_geo3k_vlm.sh, the upstream colocated reference):
#   - MODEL_ARGS_ROTARY_BASE=5000000 before sourcing qwen3-8B.sh (VL needs
#     a higher rotary base than text-only Qwen3-8B).
#   - --megatron-to-hf-mode bridge appended into MODEL_ARGS — train.py loads
#     HF weights directly via the bridge, no torch_dist conversion needed.
#   - HF_TORCHDIST_DIR intentionally not set, so launch_miles.sbatch skips
#     its HF->torch_dist auto-convert step (the offline converter doesn't
#     know about ViT/merger weights). --load points at $HF_MODEL_DIR.
#   - --multimodal-keys '{"image": "images"}'.
#   - No --ref-load and no --use-kl-loss — needs a Megatron-format ref
#     policy that we'd have to convert; skip it for the smoke test (the
#     upstream VLM recipe also runs without a KL term).
#   - rm-type=math; geo3k uses shorter responses (4 k) than dapo-math (8 k).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_REPO=${MILES_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}
RECIPE_NAME=$(basename "${BASH_SOURCE[0]}" .sh)

# ---------------------------------------------------------------------------
# Resource metadata — read by orchestrator wrappers for sbatch/k8s/etc.
# ---------------------------------------------------------------------------
EXPERIMENT_NODES=2
EXPERIMENT_TIME=24:00:00

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
# VL needs rotary_base=5000000 (upstream run_geo3k_vlm.sh:190). Must be set
# BEFORE sourcing the model file because it's read at source-time.
MODEL_ARGS_ROTARY_BASE=5000000
# shellcheck disable=SC1090
source "$MILES_REPO/scripts/models/qwen3-8B.sh"
# `bridge` is required to load Qwen3-VL HF weights into the Megatron model
# (upstream run_geo3k_vlm.sh:183). Add it to MODEL_ARGS so it propagates to
# both convert_hf_to_torch_dist.py and train.py.
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
   --use-dynamic-batch-size
   --max-tokens-per-gpu 4096
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
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static  0.6
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
   --wandb-project miles-imp
   --wandb-group   "$RUN_NAME"
   # WANDB_API_KEY comes from the env (exported by submit.sh / launch_miles.sbatch);
   # we don't pass it on the CLI because it would leak into run.log and args.json.
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
   "${EVAL_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${MISC_ARGS[@]}"
   "${FT_ARGS[@]}"
)
