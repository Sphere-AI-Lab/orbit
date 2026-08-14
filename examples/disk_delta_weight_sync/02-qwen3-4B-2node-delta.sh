#!/bin/bash
# Measurement arm: disk-delta across a real host boundary.
#
# 2 nodes, 8 trainer GPUs on node 0 and 8 rollout GPUs on node 1, so deltas
# actually cross the shared filesystem and each rollout host patches its own
# local checkpoint. Pair with 03-qwen3-4B-2node-broadcast.sh, which is the same
# workload with weights pushed over NCCL instead.
#
# Compare perf/update_weights_density and perf/update_weights_wire_bytes
# against the broadcast arm's sync time.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

DD_RUN_NAME=dd-02-qwen3-4B-2node-delta
DD_TRANSFER_MODE=disk-delta
DD_TRAINER_NODES=1
DD_TRAINER_GPUS_PER_NODE=8
DD_ROLLOUT_GPUS=8
EXPERIMENT_NODES=2

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"
