#!/usr/bin/env bash
#
# E4 math, LoRA reproducibility check: r1/r16/r256, each at its OWN measured
# best learning rate.  Book a WHOLE node.
#
#   source scripts/lora_regret/env_v0516.sh
#   cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
#   bash scripts/lora_regret/run_e4_math_lora_verify_8gpu.sh
#
# 3 arms, and NOT a column: the `run_e4_math_lr*_8gpu.sh` scripts hold the LR
# fixed and vary the rank, which is what a sweep needs. This holds the ARGMIN
# fixed per rank -- the point each curve actually peaked at -- because the
# question here is whether the panel's endpoints come back, not where they are.
#
#   rank    LR       2026-08-10 endpoint (math_test @ rollout 149)
#   r1      1e-05    0.2536
#   r16     3e-05    0.2758   <- the panel's best LoRA arm, above FullFT's 0.2660
#   r256    7e-05    0.2378
#
# The three LRs differ because math's LoRA optimum MOVES with rank (the gsm8k
# panel's did not; all three peaked at 3e-05 there). Running all three at one
# shared LR would re-measure a different thing.
#
# Its own ledger, deliberately. The sweep skips arms a ledger already records as
# ok, so pointing this at `results/e4_math_lr*.jsonl` would skip all three and
# do nothing -- a fresh RESULTS file is what makes this a re-run rather than a
# resume. Compare afterwards against the originals, which stay untouched:
#
#   python -m tools.lora_regret.analyze --ledgers 'results/e4_math_lr*.jsonl'
#   python -m tools.lora_regret.analyze --ledgers 'results/e4_math_lora_verify.jsonl'
#
# ALLOW_OFT is left at its default refusal: this ledger is a LoRA comparable
# set, and the OFT ladder runs separately in
# `run_e4_math_oft_verify_8gpu.sh`.
#
# **Resumable.** Re-run the same script; finished arms are skipped. One writer
# per RESULTS file.
#
# **Every RL arm is an 8-GPU arm.** The LoRA arms were measured at eight.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/e4_protocol.sh"

exec env \
    MATRIX=e4 \
    METHOD_RE='^lora-(r1-all-math-lr1e\-05|r16-all-math-lr3e\-05|r256-all-math-lr7e\-05)-s0$' \
    RESULTS=results/e4_math_lora_verify.jsonl \
    EXPECT_ARMS=3 \
    PREFLIGHT_STAGE=e4 \
    bash "${HERE}/campaign.sh" "$@"
