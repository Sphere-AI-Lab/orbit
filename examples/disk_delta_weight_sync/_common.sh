#!/bin/bash
#
# Shared recipe body for the disk-delta weight-sync examples.
#
# Source this through a numbered wrapper; the wrapper sets only the axis it
# varies (transfer mode, layout, encoding) and leaves everything else alone.
# The wrappers satisfy the scripts/slurm/submit.sh recipe contract
# (MILES_ARGS + HF_MODEL_REPO + EXPERIMENT_NODES), so they can be submitted
# through a thin shim under scripts/experiments/ — see README.md.
#
# Workload is Qwen3-4B on dapo-math-17k, deliberately identical across every
# wrapper: the only thing that moves is how weights reach the rollout engines.

set -euo pipefail

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
   echo "FATAL: source _common.sh through a wrapper, do not run it directly" >&2
   exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_REPO=${MILES_REPO:-$(cd "$SCRIPT_DIR/../.." && pwd)}

if [[ -z "${DD_RUN_NAME:-}" ]]; then
   echo "FATAL: wrapper must set DD_RUN_NAME" >&2
   return 1
fi

# ---------------------------------------------------------------- axes ----

# broadcast is the control arm: same workload, weights pushed over NCCL.
DD_TRANSFER_MODE=${DD_TRANSFER_MODE:-disk-delta}
DD_TRAINER_NODES=${DD_TRAINER_NODES:-1}
DD_TRAINER_GPUS_PER_NODE=${DD_TRAINER_GPUS_PER_NODE:-4}
DD_ROLLOUT_GPUS=${DD_ROLLOUT_GPUS:-4}
DD_NUM_ROLLOUT=${DD_NUM_ROLLOUT:-3000}
DD_CHECK_EQUAL=${DD_CHECK_EQUAL:-0}

# xor is smaller and faster but must land exactly once on the right base;
# overwrite is idempotent. Both are byte-level, so both are dtype-blind.
DD_ENCODING=${DD_ENCODING:-xor}
DD_CHECKSUM=${DD_CHECKSUM:-xxh3-128}

case "$DD_TRANSFER_MODE" in
   disk-delta | broadcast) ;;
   *)
      echo "FATAL: DD_TRANSFER_MODE must be disk-delta or broadcast, got '$DD_TRANSFER_MODE'" >&2
      return 1
      ;;
esac

EXPERIMENT_NODES=${EXPERIMENT_NODES:-$DD_TRAINER_NODES}
EXPERIMENT_TIME=${EXPERIMENT_TIME:-04:00:00}

RUN_NAME=${SLURM_JOB_NAME:-$DD_RUN_NAME}

# --------------------------------------------------------------- assets ---

# Defaults are Qwen3-4B (dense). A wrapper overrides the model block to test a different
# architecture; the dataset and the RL hyperparameters stay fixed so the transport arms of
# any one model remain comparable to each other.
HF_CACHE_DIR=${HF_CACHE_DIR:-/data/shared/hf_cache}
DD_MODEL_CONFIG=${DD_MODEL_CONFIG:-qwen3-4B}
HF_MODEL_REPO=${DD_HF_MODEL_REPO:-Qwen/Qwen3-4B}
HF_DATASETS=(
   "zhuzilin/dapo-math-17k"
)
# submit.sh downloads HF_MODEL_REPO only when HF_MODEL_DIR/config.json is absent, so pointing
# these at an already-present checkpoint outside HF_CACHE_DIR skips the download entirely.
HF_MODEL_DIR=${DD_HF_MODEL_DIR:-$HF_CACHE_DIR/models/Qwen3-4B}
HF_TORCHDIST_DIR=${DD_HF_TORCHDIST_DIR:-$HF_CACHE_DIR/models/Qwen3-4B_torch_dist}
HF_TRAIN_DATA="$HF_CACHE_DIR/data/dapo-math-17k/dapo-math-17k.jsonl"

# shellcheck disable=SC1090
source "$MILES_REPO/scripts/models/$DD_MODEL_CONFIG.sh"

# ----------------------------------------------------------- delta dirs ---

# Published deltas: must be visible to the trainer and to every rollout host.
# WARNING: the trainer rmtree()s this directory when it captures the baseline —
# a stale version stream would apply against the wrong base. Point it at a
# per-run scratch path, never at anything you want to keep.
DD_DISK_DIR=${DD_DISK_DIR:-$MILES_REPO/checkpoints/$RUN_NAME/delta-updates}

# Host-local full HF checkpoint each engine patches in place and reloads from.
# Local disk on purpose (NVMe if you have it) — the whole point is that only
# the changed bytes cross the shared filesystem. Sized for a full checkpoint:
# Qwen3-4B bf16 is ~8 GB per rollout host, so /tmp needs the room.
DD_LOCAL_CKPT_DIR=${DD_LOCAL_CKPT_DIR:-/tmp/miles-delta-local-ckpt/$RUN_NAME}

# ----------------------------------------------------------------- args ---

CKPT_ARGS=(
   --hf-checkpoint "$HF_MODEL_DIR"
   --ref-load      "$HF_TORCHDIST_DIR"
   --load          "$MILES_REPO/checkpoints/$RUN_NAME"
   --save          "$MILES_REPO/checkpoints/$RUN_NAME"
   --save-interval 20
)

# Concurrency is rollout-batch-size x n-samples-per-prompt requests in flight. That has to track
# the serving capacity: job 39267 kept these at the 4-node values while running on a quarter of
# the rollout GPUs, and the queue ran away -- 362 requests in flight, /health timing out, the
# router's circuit breaker opening, and generation dying after 60 retries against a server that
# was saturated rather than dead.
ROLLOUT_ARGS=(
   --prompt-data   "$HF_TRAIN_DATA"
   --input-key     prompt
   --label-key     label
   --apply-chat-template
   --rollout-shuffle
   --rm-type       deepscaler
   --num-rollout   "$DD_NUM_ROLLOUT"
   --rollout-batch-size       "${DD_ROLLOUT_BATCH_SIZE:-64}"
   --n-samples-per-prompt     "${DD_SAMPLES_PER_PROMPT:-8}"
   --rollout-max-response-len 8192
   --rollout-temperature      1
   --global-batch-size        "${DD_GLOBAL_BATCH_SIZE:-512}"
   --balance-data
)

PERF_ARGS=(
   --tensor-model-parallel-size "${DD_TP:-2}"
   --sequence-parallel
   --pipeline-model-parallel-size "${DD_PP:-1}"
   --context-parallel-size "${DD_CP:-1}"
   --expert-model-parallel-size "${DD_EP:-1}"
   --expert-tensor-parallel-size "${DD_ETP:-1}"
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "${DD_MAX_TOKENS_PER_GPU:-9216}"
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
# A large MoE needs the optimizer state off the GPU; a dense 4B does not.
if [[ -n "${DD_EXTRA_OPTIMIZER_ARGS+x}" ]]; then
   OPTIMIZER_ARGS+=("${DD_EXTRA_OPTIMIZER_ARGS[@]}")
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "${DD_GPUS_PER_ENGINE:-2}"
   --sglang-mem-fraction-static  "${DD_MEM_FRACTION:-0.85}"
)
# MoE serving wants expert parallelism and DP attention on the engine side too.
if [[ -n "${DD_EXTRA_SGLANG_ARGS+x}" ]]; then
   SGLANG_ARGS+=("${DD_EXTRA_SGLANG_ARGS[@]}")
fi

# Disaggregated by construction. disk-delta asserts `not colocate`: under
# colocate the weights cross via CUDA IPC (only a handle moves), so snapshot +
# diff + encode would be pure overhead.
LAYOUT_ARGS=(
   --actor-num-nodes         "$DD_TRAINER_NODES"
   --actor-num-gpus-per-node "$DD_TRAINER_GPUS_PER_NODE"
   --rollout-num-gpus        "$DD_ROLLOUT_GPUS"
)

WEIGHT_SYNC_ARGS=(--update-weight-transfer-mode "$DD_TRANSFER_MODE")
if [[ "$DD_TRANSFER_MODE" == "disk-delta" ]]; then
   WEIGHT_SYNC_ARGS+=(
      --update-weight-disk-dir             "$DD_DISK_DIR"
      --update-weight-local-checkpoint-dir "$DD_LOCAL_CKPT_DIR"
      --update-weight-delta-encoding       "$DD_ENCODING"
      --update-weight-delta-checksum       "$DD_CHECKSUM"
   )
fi

# Compares engine tensors against the trainer's after a sync. Cheap enough for
# a smoke run and the only direct proof the applied delta reconstructed the
# right bytes; too slow to leave on for a throughput measurement.
if [[ "$DD_CHECK_EQUAL" == "1" ]]; then
   WEIGHT_SYNC_ARGS+=(--check-weight-update-equal)
fi

WANDB_ARGS=(
   --use-wandb
   --wandb-project miles-imp
   --wandb-group   "$RUN_NAME"
)

FT_ARGS=(
   --use-fault-tolerance
   --rollout-health-check-interval   30
   --rollout-health-check-timeout    30
   --rollout-health-check-first-wait 60
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout    0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

MILES_ARGS=(
   "${LAYOUT_ARGS[@]}"
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${ROLLOUT_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${GRPO_ARGS[@]}"
   "${WANDB_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${WEIGHT_SYNC_ARGS[@]}"
   "${MISC_ARGS[@]}"
   "${FT_ARGS[@]}"
)

echo "[disk-delta example] run=$RUN_NAME mode=$DD_TRANSFER_MODE nodes=$EXPERIMENT_NODES"
if [[ "$DD_TRANSFER_MODE" == "disk-delta" ]]; then
   echo "[disk-delta example]   shared publish dir : $DD_DISK_DIR  (wiped at baseline)"
   echo "[disk-delta example]   host-local ckpt dir: $DD_LOCAL_CKPT_DIR  (~${DD_CKPT_SIZE_HINT:-8 GB} per rollout host)"
   echo "[disk-delta example]   encoding=$DD_ENCODING checksum=$DD_CHECKSUM"
fi
