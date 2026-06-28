#!/usr/bin/env bash
# Qwen2.5-3B-Instruct BF16 LoRA PPO for Search-R1.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEARCH_R1_PEFT_MODE=lora
source "${SCRIPT_DIR}/qwen2_5_3b_search_r1_ppo_common.sh"

