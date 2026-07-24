#!/bin/bash
#
# Milestone 12b: scaled fully-async trainer-recomputed q_old arm.
#
# q_adv = q_den = q_trainer-preupdate. The trainer runs the separate no-grad
# old-logprob forward, making the sampled-token PPO ratio one at the update.
# The experiment is frozen at 200 optimizer steps.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export M12_OLD_POLICY_SOURCE=trainer
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-opd-mm-12b-scale2t4s-recompute-qold-200step}

# shellcheck disable=SC1091
source "$SCRIPT_DIR/12-old-policy-scale-common.sh"
