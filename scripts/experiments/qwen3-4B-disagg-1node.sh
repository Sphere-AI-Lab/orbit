#!/bin/bash
#
# qwen3-4B-disagg-1node — single-node disaggregated variant.
#
# Functionally identical to qwen3-4B-disagg-2node.sh except for layout and
# resource metadata. See that file's header for the orchestrator contract.
#
# Layout (1 node × 8 GPUs, disaggregated 4+4):
#   - GPUs 0-3: Megatron training (--actor-num-nodes 1 --actor-num-gpus-per-node 4)
#   - GPUs 4-7: SGLang rollout    (--rollout-num-gpus 4)
# For colocated 8+8 time-sliced on the same GPUs, see upstream
# scripts/run-qwen3-4B.sh (which uses --colocate).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ORBIT_REPO=${ORBIT_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}
RECIPE_NAME=$(basename "${BASH_SOURCE[0]}" .sh)

EXPERIMENT_NODES=1
EXPERIMENT_TIME=48:00:00

HF_CACHE_DIR=${HF_CACHE_DIR:-/data/shared/hf_cache}

HF_MODEL_REPO="Qwen/Qwen3-4B"
HF_DATASETS=(
    "zhuzilin/dapo-math-17k"
    "zhuzilin/aime-2024"
)
HF_MODEL_DIR="$HF_CACHE_DIR/models/Qwen3-4B"
HF_TORCHDIST_DIR="$HF_CACHE_DIR/models/Qwen3-4B_torch_dist"
HF_TRAIN_DATA="$HF_CACHE_DIR/data/dapo-math-17k/dapo-math-17k.jsonl"
HF_EVAL_DATA="$HF_CACHE_DIR/data/aime-2024/aime-2024.jsonl"

# shellcheck disable=SC1090
source "$ORBIT_REPO/scripts/models/qwen3-4B.sh"

RUN_NAME=${SLURM_JOB_NAME:-$RECIPE_NAME}

CKPT_ARGS=(
   --hf-checkpoint  "$HF_MODEL_DIR"
   --ref-load       "$HF_TORCHDIST_DIR"
   --load           "$ORBIT_REPO/checkpoints/$RUN_NAME"
   --save           "$ORBIT_REPO/checkpoints/$RUN_NAME"
   --save-interval  20
)

ROLLOUT_ARGS=(
   --prompt-data   "$HF_TRAIN_DATA"
   --input-key     prompt
   --label-key     label
   --apply-chat-template
   --rollout-shuffle
   --rm-type       deepscaler
   --num-rollout   3000
   --rollout-batch-size      64
   --n-samples-per-prompt    8
   --rollout-max-response-len 8192
   --rollout-temperature     1
   --global-batch-size       512
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime "$HF_EVAL_DATA"
   --n-samples-per-eval-prompt 16
   --eval-max-response-len     16384
   --eval-top-p                1
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 9216
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
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
   --wandb-project orbit
   --wandb-group   "$RUN_NAME"
   # WANDB_API_KEY comes from the env (exported by submit.sh / launch_orbit.sbatch);
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
   --actor-num-gpus-per-node 4
   --rollout-num-gpus       4
)

ORBIT_ARGS=(
   "${LAYOUT_ARGS[@]}"
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
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
