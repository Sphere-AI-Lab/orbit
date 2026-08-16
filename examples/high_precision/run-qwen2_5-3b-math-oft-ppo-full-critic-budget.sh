#!/usr/bin/env bash
# Fixed four-B200 budget: Qwen2.5 math PPO with a separate full critic.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PPO_CRITIC_MODE=full
PPO_COMPARISON_PANEL=budget
GPUS_PER_NODE=1
CRITIC_NUM_GPUS_PER_NODE=1
ROLLOUT_NUM_GPUS=2
source "${SCRIPT_DIR}/ppo_critic_compare_common.sh"
