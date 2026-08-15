#!/usr/bin/env bash
# Math OFT BS128 refinement A: 5e-6, 6e-6, 7e-6 on one 8-GPU node.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/e4_protocol.sh"

exec env \
    MATRIX=e4oftb128refine \
    METHOD_RE='^oftrefine-b128-all-math-lr(5e-06|6e-06|7e-06)-s0$' \
    RESULTS=results/e4_math_oft_b128_refine_a.jsonl \
    EXPECT_ARMS=3 \
    ALLOW_OFT=1 \
    PREFLIGHT_STAGE=e4oftb128refine \
    bash "${HERE}/campaign.sh" "$@"
