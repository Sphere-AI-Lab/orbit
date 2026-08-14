#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
exec bash "${SCRIPT_DIR}/_run_qwen3_30b_a3b_gsm8k_2node_sharding.sh" 2 1 1 8 1
