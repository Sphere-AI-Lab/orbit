#!/usr/bin/env bash
# Qwen3-4B-Instruct-2507 BF16 OFT PPO for Tau-bench.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TAU_BENCH_PEFT_MODE=oft
source "${SCRIPT_DIR}/qwen3_4b_tau_bench_ppo_common.sh"

