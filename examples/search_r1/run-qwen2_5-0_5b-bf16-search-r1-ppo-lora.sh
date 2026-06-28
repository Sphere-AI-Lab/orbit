#!/usr/bin/env bash
# Qwen2.5-0.5B-Instruct BF16 LoRA PPO for Search-R1.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SEARCH_R1_PEFT_MODE=lora
SEARCH_R1_MODEL_TAG=qwen25_05b
SEARCH_R1_MODEL_DIR_NAME=Qwen2.5-0.5B-Instruct
SEARCH_R1_MODEL_ARGS_FILE=qwen2.5-0.5B.sh
source "${SCRIPT_DIR}/qwen2_5_3b_search_r1_ppo_common.sh"

