#!/usr/bin/env bash
#
# Focused E4 Math OFT BS128 lower-learning-rate sweep:
# 1e-7, 3e-7, 1e-6, 3e-6, 1e-5. Book a whole 8-GPU node.
#
#   source scripts/lora_regret/env_v0516.sh
#   cd /fast/zqiu/orbit-iclr/orbit
#   bash scripts/lora_regret/run_e4_math_oft_b128_low_lr_8gpu.sh
#
# Resumable: successful rows in the dedicated ledger are skipped. Use one
# writer for this RESULTS file.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/e4_protocol.sh"

exec env \
    MATRIX=e4oftb128low \
    METHOD_RE='^oftlow-b128-all-math-lr' \
    RESULTS=results/e4_math_oft_b128_low_lr.jsonl \
    EXPECT_ARMS=5 \
    ALLOW_OFT=1 \
    PREFLIGHT_STAGE=e4oftb128low \
    bash "${HERE}/campaign.sh" "$@"
