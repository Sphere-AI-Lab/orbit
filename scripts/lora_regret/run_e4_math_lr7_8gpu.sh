#!/usr/bin/env bash
#
# E4, math panel, learning-rate column 7 of 7: FullFT at 0.0001 and LoRA
# r1/r16/r256 at 0.001.  Book a WHOLE node.
#
#   bash scripts/lora_regret/run_e4_math_lr7_8gpu.sh
#
# 4 arms -- one point on each of C5's four curves, for one of Figure 6's two
# panels. The fourteen `run_e4_<dataset>_lr*_8gpu.sh` scripts partition e4's
# FullFT and LoRA cells exactly: 2 datasets x 7 columns x 4 arms = 56, no
# overlap, no gaps. A column on its own cannot give an argmin -- every claim in
# C5 is about the shape ACROSS columns, so a partial run is a partial curve
# rather than a partial answer.
#
# FullFT and LoRA sit on separate grids an order of magnitude apart (runbook
# section 23.4), so column 7 pairs the 7th point of each: 0.0001 against 0.001.
#
# Trains on math_train.jsonl and is scored on math_test.jsonl alone. Not both:
# `parse_final_accuracy` means across whatever datasets were evaluated, so
# scoring against MATH as well would make every point of this panel the average
# of two datasets.
#
# Its own ledger, so the fourteen can run concurrently without appending to one
# file. `analyze` takes globs, and each panel reassembles across its seven:
#
#   python -m tools.lora_regret.analyze --ledgers 'results/e4_math_lr*.jsonl' ...
#
# **Every RL arm is an 8-GPU arm.** FullFT has no choice -- TP=4/DP=2 is the
# only configuration it fits in (section 22.2) -- and the LoRA arms were
# measured at eight.
exec env MATRIX=e4 METHOD_RE='^(full-na-na-math-lr0\.0001|lora-r(1|16|256)-all-math-lr0\.001)-s' RESULTS=results/e4_math_lr7.jsonl EXPECT_ARMS=4 \
    bash "$(dirname "${BASH_SOURCE[0]}")/campaign.sh" "$@"
