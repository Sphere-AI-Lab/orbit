#!/bin/bash
# MoE smoke: does disk-delta reconstruct expert weights correctly?
#
# The dense arm (01) never touched the EP gather pass or the duplicate-name drop, so this is
# where those first run. --check-weight-update-equal is on, so a delta that rebuilt an expert
# tensor wrongly fails the run rather than quietly serving bad weights.
#
# 4 nodes, short. Run this before 05/06.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

DD_RUN_NAME=dd-04-qwen3-30B-A3B-delta-smoke
DD_TRANSFER_MODE=disk-delta
DD_NUM_ROLLOUT=5
DD_CHECK_EQUAL=1
EXPERIMENT_TIME=02:00:00

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_model-qwen3-30B-A3B.sh"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"
