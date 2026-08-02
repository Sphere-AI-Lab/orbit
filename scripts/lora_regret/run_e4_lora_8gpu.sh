#!/usr/bin/env bash
#
# E4, LoRA: the rank ladder that decides C5.  Book a WHOLE node.
#
#   bash scripts/lora_regret/run_e4_lora_8gpu.sh
#
# 21 arms -- r1, r16 and r256 on all four projections, seven learning rates each,
# on the SAME 1e-06 .. 1e-03 grid the FullFT cell runs. Not offset a decade: the
# post says 10x in prose, 2-4x in its own RL figure and 6.4x in the SVG's
# x-positions, so any offset would decide that contradiction in the grid instead
# of measuring it. The ratio is two argmins on one lattice.
#
# ~2.8 h per arm per 100 rollouts at the measured LoRA pace (89 s/rollout, plus
# eval), so ~59 h at NUM_ROLLOUT=100 and ~89 h at 150.
#
# **Rank 1 is the claim's whole point** -- "LoRA matches FullFT under policy
# gradient even at rank 1" -- so it is the last arm to drop under budget
# pressure, not the first.
#
# The single largest cell in the campaign. It is also the one that can be split
# across two nodes if you have them: the ranks are independent, so
# `--only '^lora-r1-'` and `--only '^lora-r(16|256)-'` with SEPARATE --results
# files run concurrently without contending for a ledger.
exec env MATRIX=e4 METHOD_RE='^lora-' RESULTS=results/e4_lora.jsonl EXPECT_ARMS=21 \
    bash "$(dirname "${BASH_SOURCE[0]}")/campaign.sh" "$@"
