#!/bin/bash
#
# Overlay on vagen-sokoban-main-qwen25vl3b-colocate-1node with
# --global-batch-size 32 (8 PPO updates per rollout instead of 1).
# VAGEN-main's ppo_mini_batch_size=32 is in sample units after the rollout.n
# repeat, giving 256/32 = 8 updates per rollout — match that here.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export ORBIT_SCRIPT_GLOBAL_BATCH_SIZE=32

# shellcheck disable=SC1091
source "$SCRIPT_DIR/vagen-sokoban-main-qwen25vl3b-colocate-1node.sh"
