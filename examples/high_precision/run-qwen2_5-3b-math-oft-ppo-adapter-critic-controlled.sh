#!/usr/bin/env bash
# Controlled learning panel: matched rollout capacity with an adapter critic.
# One of four B200s is deliberately idle in this panel.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PPO_CRITIC_MODE=adapter
PPO_COMPARISON_PANEL=controlled
GPUS_PER_NODE=1
CRITIC_NUM_GPUS_PER_NODE=0
ROLLOUT_NUM_GPUS=2
source "${SCRIPT_DIR}/ppo_critic_compare_common.sh"
