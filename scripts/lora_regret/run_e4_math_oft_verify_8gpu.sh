#!/usr/bin/env bash
#
# E4 math, OFT reproducibility check: the b8/b128/b1024 ladder, all at 7e-06.
# Book a WHOLE node.
#
#   source scripts/lora_regret/env_v0516.sh
#   cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
#   bash scripts/lora_regret/run_e4_math_oft_verify_8gpu.sh
#
# 3 arms, one per rung, the LR pinned so the block size is the only difference
# between them. What each one is measured against:
#
#   block   prior math row                              endpoint
#   b8      oftscout-b8-all-math-lr3e-05-s0             0.2130   (different LR)
#   b128    oftrefine-b128-all-math-lr7e-06-s0          0.2742   (same LR)
#   b1024   none                                        --
#
# Only b128 is a like-for-like reproduction. b8 moves off the single LR it was
# ever measured at, and b1024 has no successful math row at all -- its `e4`
# scout arms failed and its one gsm8k row carries a null accuracy. Those two are
# new measurements on a known-healthy LR, not checks, and the report should say
# so rather than quietly present three reproductions.
#
# 7e-06 is b128's measured argmax and the only OFT point on math backed by a
# curve rather than a single row; see `E4_MATH_OFT_VERIFY_LR` in arms.py.
#
# Its own matrix rather than another `--only` over `e4`: 7e-06 is not in the
# scout grid (2e-06, 5e-06, 1e-05, 3e-05, 7e-05, 2e-04, 4e-04), so no regex over
# `e4` can select this selection. Arms are named `oftverify-*` so they cannot be
# mistaken for the `oftscout`/`oftlow`/`oftrefine` rows they are compared to.
#
# ALLOW_OFT=1 -- required, and this is a dedicated OFT ledger, so no FullFT or
# LoRA row can enter it. b1024 is exactly OFT_MAX_BLOCK_SGLANG; it needs a
# package carrying both rotation-kernel commits (893f329a2 and 166041d28).
#
# **Resumable.** Re-run the same script; finished arms are skipped. One writer
# per RESULTS file.
#
# **Every RL arm is an 8-GPU arm.**
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/e4_protocol.sh"

exec env \
    MATRIX=e4oftverify \
    METHOD_RE='^oftverify-b(8|128|1024)-all-math-lr7e\-06-s0$' \
    RESULTS=results/e4_math_oft_verify.jsonl \
    EXPECT_ARMS=3 \
    ALLOW_OFT=1 \
    PREFLIGHT_STAGE=e4oftverify \
    bash "${HERE}/campaign.sh" "$@"
