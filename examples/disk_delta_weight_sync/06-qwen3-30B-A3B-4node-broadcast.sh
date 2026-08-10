#!/bin/bash
# Control arm for 05: identical model, layout, and workload, weights pushed over NCCL.
#
# _model-qwen3-30B-A3B.sh and _common.sh hold everything else fixed, so the difference in
# weight-sync time between this run and 05 is the transport and nothing else.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

DD_RUN_NAME=dd-06-qwen3-30B-A3B-broadcast
DD_TRANSFER_MODE=broadcast

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_model-qwen3-30B-A3B.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"
