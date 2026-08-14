#!/bin/bash
# Control arm for 02: identical workload and layout, weights pushed over NCCL.
#
# Everything except --update-weight-transfer-mode is held fixed by _common.sh,
# so the difference in weight-sync time between this run and 02 is the delta
# pipeline's effect and nothing else.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

DD_RUN_NAME=dd-03-qwen3-4B-2node-broadcast
DD_TRANSFER_MODE=broadcast
DD_TRAINER_NODES=1
DD_TRAINER_GPUS_PER_NODE=8
DD_ROLLOUT_GPUS=8
EXPERIMENT_NODES=2

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"
