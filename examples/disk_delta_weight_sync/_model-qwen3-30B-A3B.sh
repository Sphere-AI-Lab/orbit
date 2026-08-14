#!/bin/bash
#
# Model block: Qwen3-30B-A3B (MoE, 48 layers, 128 experts, 8 active per token, bf16).
#
# Source this before _common.sh from a wrapper that varies only the transport.
#
# Why a MoE arm exists at all. Two reasons, and the second is the one that matters:
#
#   1. It is where the mechanism should pay. 57 GB of weights cross the fabric on every
#      broadcast sync, against ~3B active parameters per token. Delta size tracks what the
#      optimizer touched, not what the model weighs.
#   2. It covers code the dense arm cannot reach. _for_each_hf_bucket runs a TP pass and
#      then a separate EP pass, and _drop_duplicate_names exists because expert tensors can
#      be gathered by more than one rank. With --expert-model-parallel-size 1 the dense
#      Qwen3-4B run never entered either. This is the first exercise of the expert path.
#
# Parallelism mirrors examples/p2p_weight_transfer/run-qwen3-30B-A3B-4node-profile.sh, which
# is the validated shape for this model on 4 nodes: 2 training + 2 rollout.

# The HF checkpoint is already on the cluster, outside HF_CACHE_DIR — submit.sh skips the
# download because HF_MODEL_DIR/config.json is present.
DD_MODEL_CONFIG=qwen3-30B-A3B
DD_HF_MODEL_REPO=Qwen/Qwen3-30B-A3B
DD_HF_MODEL_DIR=${DD_HF_MODEL_DIR:-/data/shared/models/Qwen3-30B-A3B}

# No shared torch_dist conversion of this model exists (and the shared HF cache is read-only),
# so there is no default that works for everyone: point this at your own conversion. Convert
# once with scripts/slurm/setup/convert_checkpoint.sh — pre-convert rather than relying on the
# launcher's auto-convert; a 30B conversion is not a 5-minute head-node job.
if [[ -z "${DD_HF_TORCHDIST_DIR:-}" ]]; then
   echo "FATAL: set DD_HF_TORCHDIST_DIR to a Megatron torch_dist conversion of Qwen3-30B-A3B" >&2
   echo "       (scripts/slurm/setup/convert_checkpoint.sh converts one from $DD_HF_MODEL_DIR)" >&2
   return 1
fi

# 4 nodes: 16 training GPUs + 16 rollout GPUs.
DD_TRAINER_NODES=2
DD_TRAINER_GPUS_PER_NODE=8
DD_ROLLOUT_GPUS=16
EXPERIMENT_NODES=${EXPERIMENT_NODES:-4}

DD_TP=4
DD_PP=1
DD_CP=1
DD_EP=8
DD_ETP=1
DD_MAX_TOKENS_PER_GPU=2048

# Two engines of 8 GPUs, expert-parallel with DP attention.
DD_GPUS_PER_ENGINE=8
DD_MEM_FRACTION=0.8
DD_EXTRA_SGLANG_ARGS=(
   --sglang-ep-size 8
   --sglang-enable-dp-attention
   --sglang-enable-dp-lm-head
   --sglang-model-loader-extra-config '{"enable_multithread_load":true,"num_threads":8}'
)

# 30B of Adam state does not fit alongside the weights.
DD_EXTRA_OPTIMIZER_ARGS=(
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

# Each rollout host patches its own full copy; /tmp on slinky has room (6.5 TB free).
DD_CKPT_SIZE_HINT="57 GB"

# Measured per-sync publish size for this model (~0.5 GB/sync at EP8, rounded up);
# feeds the disk-budget guard in _common.sh.
DD_DELTA_MB_HINT=${DD_DELTA_MB_HINT:-600}
