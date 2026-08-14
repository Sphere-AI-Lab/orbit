#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
exec bash "${SCRIPT_DIR}/_run_qwen2_5_0_5b_gsm8k.sh" muon
