#!/usr/bin/env bash
# Fixed four-B200 budget: the adapter critic frees one extra rollout GPU.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PPO_CRITIC_MODE=adapter
PPO_COMPARISON_PANEL=budget
GPUS_PER_NODE=1
CRITIC_NUM_GPUS_PER_NODE=0
ROLLOUT_NUM_GPUS=3
source "${SCRIPT_DIR}/ppo_critic_compare_common.sh"
