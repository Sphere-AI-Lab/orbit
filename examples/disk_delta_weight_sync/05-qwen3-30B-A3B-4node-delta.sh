#!/bin/bash
# MoE measurement arm: disk-delta on Qwen3-30B-A3B across 4 nodes.
#
# Pair with 06-qwen3-30B-A3B-4node-broadcast.sh. This is the comparison the mechanism exists
# for: broadcast moves all 57 GB every sync regardless of how little the step changed.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

DD_RUN_NAME=dd-05-qwen3-30B-A3B-delta
DD_TRANSFER_MODE=disk-delta

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_model-qwen3-30B-A3B.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"
