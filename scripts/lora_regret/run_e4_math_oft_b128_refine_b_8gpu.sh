#!/usr/bin/env bash
# Math OFT BS128 refinement B: 8e-6, 9e-6, 2e-5 on one 8-GPU node.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/e4_protocol.sh"

exec env \
    MATRIX=e4oftb128refine \
    METHOD_RE='^oftrefine-b128-all-math-lr(8e-06|9e-06|2e-05)-s0$' \
    RESULTS=results/e4_math_oft_b128_refine_b.jsonl \
    EXPECT_ARMS=3 \
    ALLOW_OFT=1 \
    PREFLIGHT_STAGE=e4oftb128refine \
    bash "${HERE}/campaign.sh" "$@"
