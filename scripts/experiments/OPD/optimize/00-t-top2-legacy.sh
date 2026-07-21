#!/bin/bash
# Historical characterization wrapper for the teacher-top-k rebuild.
#
# This intentionally preserves the current Miles only-teacher implementation:
# teacher [T,K] -> response-wide ID union U -> student SGLang [N,U] rescore ->
# per-position [T,K] gather -> detached [T] scalar -> sampled-token PPO/GRPO.
# Job 24749 showed that the student-rescore leg crashes decode-concurrent
# SGLang v0.5.13 engines at step 0. Keep this only as a bounded reproduction;
# it produced no measured before-distribution and is not Top-K DAgger.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export OPD_TOP_K=${OPD_TOP_K:-2}
export OPD_NUM_ROLLOUT=${OPD_NUM_ROLLOUT:-5}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-t-top2-legacy}

# shellcheck disable=SC1091
source "$SCRIPT_DIR/../math_qwen3_32b_8b_3nodes_legacy_teacher/qwen3-8B.sh"
