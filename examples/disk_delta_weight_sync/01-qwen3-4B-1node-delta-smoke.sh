#!/bin/bash
# Smoke: does disk-delta sync produce byte-correct engine weights at all?
#
# 1 node, disaggregated 4 train + 4 rollout. Both the publish dir and the
# "host-local" checkpoint are ordinary local paths here — with one host the
# distinction is moot, but the full pipeline still runs: baseline seed ->
# pull_weights(0) -> per-sync publish -> /pull_weights apply -> reload.
#
# --check-weight-update-equal is on, so the run fails loudly if an applied
# delta reconstructs the wrong bytes. Short by design; this arm is about
# correctness, not throughput.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

DD_RUN_NAME=dd-01-qwen3-4B-1node-delta-smoke
DD_TRANSFER_MODE=disk-delta
DD_TRAINER_NODES=1
DD_TRAINER_GPUS_PER_NODE=4
DD_ROLLOUT_GPUS=4
DD_NUM_ROLLOUT=5
DD_CHECK_EQUAL=1
EXPERIMENT_TIME=01:00:00

# shellcheck disable=SC1091
source "$SCRIPT_DIR/_common.sh"
