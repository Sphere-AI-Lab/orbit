#!/bin/bash
#
# OPD/archive/megatron_teacher_baseline/qwen3-8B — archived on-policy
# distillation recipe with a Megatron-side
# teacher. Port of examples/on_policy_distillation/run-qwen3-8B-opd-megatron.sh
# to the submit.sh recipe contract (pure config: no `ray start`, no `sbatch`,
# no downloads — the orchestrator does the I/O).
#
# Layout (1 node x 8 GPUs, same split as the upstream example):
#   - actor:   2 GPUs Megatron (--actor-num-nodes 1 --actor-num-gpus-per-node 2)
#   - rollout: 4 GPUs SGLang   (--rollout-num-gpus 4)
#   GPUs 6-7 stay idle — kept identical to the example on purpose.
#
# Teacher: --opd-teacher-load takes a Megatron torch_dist checkpoint with the
# SAME architecture as the student. Default is the student's own torch_dist
# (self-distillation — the upstream example's smoke config; the launcher
# auto-converts it if missing). Point OPD_TEACHER_LOAD at a stronger
# converted model for a real run, e.g.:
#   OPD_TEACHER_LOAD=/data/shared/models/<teacher>_torch_dist \
#   bash scripts/slurm/submit.sh OPD/archive/megatron_teacher_baseline/qwen3-8B
#
# QUICK-CHECK config: no --save/--save-interval (nothing is checkpointed;
# weights come from --ref-load via the --load fallback).
#
# Submit (HF_CACHE_DIR=/data/shared on slinky — the default /data/shared/hf_cache
# is read-only there):
#   HF_CACHE_DIR=/data/shared bash scripts/slurm/submit.sh OPD/archive/megatron_teacher_baseline/qwen3-8B

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ORBIT_REPO=${ORBIT_REPO:-$(cd "$SCRIPT_DIR/../../../../.." && pwd)}
RECIPE_NAME=$(basename "${BASH_SOURCE[0]}" .sh)

# ---------------------------------------------------------------------------
# Resource metadata — read by orchestrator wrappers for sbatch/k8s/etc.
# ---------------------------------------------------------------------------
EXPERIMENT_NODES=1
EXPERIMENT_TIME=24:00:00

# ---------------------------------------------------------------------------
# Asset metadata — read by orchestrator wrappers for `hf download` step
# ---------------------------------------------------------------------------
HF_CACHE_DIR=${HF_CACHE_DIR:-/data/shared/hf_cache}

HF_MODEL_REPO="Qwen/Qwen3-8B"
HF_DATASETS=(
    "zhuzilin/dapo-math-17k"
)
HF_MODEL_DIR="$HF_CACHE_DIR/models/Qwen3-8B"
HF_TORCHDIST_DIR="$HF_CACHE_DIR/models/Qwen3-8B_torch_dist"
HF_TRAIN_DATA="$HF_CACHE_DIR/data/dapo-math-17k/dapo-math-17k.jsonl"

OPD_TEACHER_LOAD=${OPD_TEACHER_LOAD:-"$HF_TORCHDIST_DIR"}

# ---------------------------------------------------------------------------
# train.py args
# ---------------------------------------------------------------------------
# shellcheck disable=SC1090
source "$ORBIT_REPO/scripts/models/qwen3-8B.sh"

# Official-baseline naming: baseline-opd-<student>-<teacher mode>-t<teacher size>.
# Default teacher is the 8B itself (self-distillation); override together with
# OPD_TEACHER_LOAD when swapping in a real teacher.
RUN_NAME=${WANDB_RUN_NAME:-baseline-opd-qwen3-8B-megatron-t8B}

CKPT_ARGS=(
   --hf-checkpoint  "$HF_MODEL_DIR"
   --ref-load       "$HF_TORCHDIST_DIR"
)

ROLLOUT_ARGS=(
   --prompt-data   "$HF_TRAIN_DATA"
   --input-key     prompt
   --apply-chat-template
   --rollout-shuffle
   --num-rollout   300
   --rollout-batch-size      16
   --n-samples-per-prompt    4
   --rollout-max-response-len 16384
   --rollout-temperature     1
   --global-batch-size       64
   --balance-data
)

RM_ARGS=(
   --rm-type math
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
   --max-tokens-per-gpu 16384
)

GRPO_ARGS=(
   --advantage-estimator grpo

   --use-opd
   --opd-type megatron
   --opd-kl-coef 1.0
   --opd-teacher-load "$OPD_TEACHER_LOAD"

   # This archived recipe keeps --rm-type math, so GRPO task advantage and OPD
   # are both active. Reward KL, loss KL, entropy, and advantage normalization
   # remain disabled.
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
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static  0.4
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
   --wandb-project OPD
   --wandb-group   "$RUN_NAME"
   --disable-wandb-random-suffix
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
   --actor-num-gpus-per-node 2
   --rollout-num-gpus       4
)

ORBIT_ARGS=(
   "${LAYOUT_ARGS[@]}"
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${ROLLOUT_ARGS[@]}"
   "${RM_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${GRPO_ARGS[@]}"
   "${WANDB_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${MISC_ARGS[@]}"
   "${FT_ARGS[@]}"
)
