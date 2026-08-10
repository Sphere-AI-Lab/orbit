#!/bin/bash
# MoE smoke on a single node: 4 training + 4 rollout GPUs.
#
# Same expert coverage as 04 at a quarter of the allocation. The H200s have 141 GB each, so
# 57 GB of weights shard to ~14 GB per GPU on both sides and the Adam state (30B x 12 B = 360 GB)
# lives in the node's 1.8 TB of host RAM via --optimizer-cpu-offload.
#
# The one forced change from 04: EP8 does not fit across 4 training GPUs, so experts split 4 ways
# instead of 8 (32 per rank rather than 16). The EP gather and the duplicate-name drop still run,
# which is what the MoE arm is for.
#
# Single-node also removes the fabric from the picture entirely — the "shared" publish dir and the
# host-local checkpoint are on the same box. Use 04/05 when the cross-host path is the question.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

DD_RUN_NAME=dd-07-qwen3-30B-A3B-1node-delta-smoke
DD_TRANSFER_MODE=disk-delta
DD_NUM_ROLLOUT=5
DD_CHECK_EQUAL=1
EXPERIMENT_TIME=02:00:00

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_model-qwen3-30B-A3B.sh"

# Override the 4-node layout from the model block: one node, disaggregated 4+4.
DD_TRAINER_NODES=1
DD_TRAINER_GPUS_PER_NODE=4
DD_ROLLOUT_GPUS=4
EXPERIMENT_NODES=1

DD_EP=4
DD_GPUS_PER_ENGINE=4
DD_EXTRA_SGLANG_ARGS=(
   --sglang-ep-size 4
   --sglang-enable-dp-attention
   --sglang-enable-dp-lm-head
   --sglang-model-loader-extra-config '{"enable_multithread_load":true,"num_threads":8}'
)

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"
