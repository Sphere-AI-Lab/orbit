#!/bin/bash
#
# Milestone 12a: scaled fully-async rollout-q_old arm.
#
# q_adv = q_den = q_rollout. Student SGLang supplies the behavior-policy
# log-probabilities, and the trainer skips its separate no-grad old-logprob
# forward. The experiment is frozen at 200 optimizer steps.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export M12_OLD_POLICY_SOURCE=rollout
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-12a-scale2t4s-rollout-qold-200step}

# shellcheck disable=SC1091
source "$SCRIPT_DIR/12-old-policy-scale-common.sh"
