#!/bin/bash
# Persistent-HTTP treatment for the already-run origin-topk0-response-http baseline.
#
# This keeps the response-window/T+1 scoring behavior from b88d7cf and layers
# the persistent transport, its bounded stale-connection recovery, and transport
# telemetry on top. Model, data, objective, and 3-node resources still come
# from the canonical sampled-token recipe. Set OPD_NUM_ROLLOUT=5 for a smoke.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_TOP_K=0
export OPD_SCORING_PERSISTENT_SESSION=1
# http2 = rerun on the hardened transport (4c869b37); the original -http name
# stays on the pre-fix attempts (24681/24686) so the curves never mix.
export WANDB_RUN_NAME=persistent-topk0-response-http2

# shellcheck disable=SC1091
source "$SCRIPT_DIR/qwen3-8B.sh"
