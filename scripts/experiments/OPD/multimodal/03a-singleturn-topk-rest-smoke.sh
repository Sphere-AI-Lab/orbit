#!/bin/bash
# Milestone 03a: 5-step single-turn multimodal Top-K + Rest DAgger smoke.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Isolate trainer-direct teacher Top-K distillation. The same teacher request
# still returns sampled-token logprobs for diagnostics, but sampled RKLD-PG is
# disabled and task reward remains telemetry only.
export OPD_KL_COEF=0
export OPD_DAGGER_TOP_K=2
export OPD_DAGGER_COEF=1
export OPD_DAGGER_LOSS=cross_entropy
export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-5}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-03a-top2-rest-smoke}

# Reuse the exact 02 model, data, scoring, and TP=4/DP=2/SP/PP=1/CP=1 layout.
# shellcheck disable=SC1091
source "$SCRIPT_DIR/02a-singleturn-rkld-smoke.sh"
