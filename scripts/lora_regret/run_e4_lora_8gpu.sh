#!/usr/bin/env bash
#
# E4, LoRA: the rank ladder that decides C5.  Book a WHOLE node.
#
#   bash scripts/lora_regret/run_e4_lora_8gpu.sh
#
# 12 arms, ~17 h -- r1, r16 and r256 on all four projections, four learning
# rates each, centred a decade above FullFT (1e-5 against 1e-6) as C2's rule
# carried over as a prior. If the RL argmins disagree with that decade, it is a
# finding rather than a grid error.
#
# **Rank 1 is the claim's whole point** -- "LoRA matches FullFT under policy
# gradient even at rank 1" -- so it is the last arm to drop under budget
# pressure, not the first.
#
# The single largest cell in the campaign. It is also the one that can be split
# across two nodes if you have them: the ranks are independent, so
# `--only '^lora-r1-'` and `--only '^lora-r(16|256)-'` with SEPARATE --results
# files run concurrently without contending for a ledger.
exec env MATRIX=e4 METHOD_RE='^lora-' RESULTS=results/e4_lora.jsonl EXPECT_ARMS=12 \
    bash "$(dirname "${BASH_SOURCE[0]}")/campaign.sh" "$@"
