#!/usr/bin/env bash
#
# E4, learning-rate column 2 of 7: FullFT at 1e-06 and LoRA r1/r16/r256 at 1e-05.
# Book a WHOLE node.
#
#   bash scripts/lora_regret/run_e4_lr2_8gpu.sh
#
# 4 arms -- one point on each of C5's four curves. The seven `run_e4_lr*_8gpu.sh`
# scripts partition e4's FullFT and LoRA cells exactly: 7 x 4 = 28 arms, no
# overlap, no gaps. Run them on seven nodes at once, or in sequence and stop
# when the picture is clear. What a column CANNOT give you on its own is an
# argmin -- every claim in C5 is about the shape across columns, so a partial
# run is a partial curve, not a partial answer.
#
# FullFT and LoRA sit on separate grids an order of magnitude apart (runbook
# section 23.4), so column 2 pairs the 2th point of each: 1e-06 against 1e-05.
#
# Its own ledger, so the seven can run concurrently without appending to one
# file. `analyze` takes globs, and the four curves reassemble across the seven:
#
#   python -m tools.lora_regret.analyze --ledgers 'results/e4_lr*.jsonl' ...
#
# **Every RL arm is an 8-GPU arm.** FullFT has no choice -- TP=4/DP=2 is the
# only configuration it fits in (section 22.2) -- and the LoRA arms were
# measured at eight. Roughly 2.3 h (FullFT) and 2.8 h (LoRA) per 100 rollouts,
# so a column is ~11 h at NUM_ROLLOUT=100 if the four run in sequence.
exec env MATRIX=e4 METHOD_RE='^(full-na-na-lr1e\-06|lora-r(1|16|256)-all-lr1e\-05)-s' RESULTS=results/e4_lr2.jsonl EXPECT_ARMS=4 \
    bash "$(dirname "${BASH_SOURCE[0]}")/campaign.sh" "$@"
