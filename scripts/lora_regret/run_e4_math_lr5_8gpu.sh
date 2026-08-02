#!/usr/bin/env bash
#
# E4, math panel, learning-rate column 5 of 7: FullFT at 2e-05 and LoRA
# r1/r16/r256 at 0.0002.  Book a WHOLE node.
#
#   bash scripts/lora_regret/run_e4_math_lr5_8gpu.sh
#
# 4 arms -- one point on each of C5's four curves, for one of Figure 6's two
# panels. The fourteen `run_e4_<dataset>_lr*_8gpu.sh` scripts partition e4's
# FullFT and LoRA cells exactly: 2 datasets x 7 columns x 4 arms = 56, no
# overlap, no gaps.
#
# Split by dataset as well as by column so a panel is schedulable on its own:
# gsm8k has six times the dynamic range of math (0.06 -> 0.75 against
# 0.035 -> 0.29) and math's baseline is partly guessing -- 56% of its correct
# answers at rollout 0 have a single-character label, against 23% on gsm8k -- so
# running gsm8k first and deciding on math afterwards is a real option. A column
# on its own still cannot give an argmin: every claim in C5 is about the shape
# ACROSS columns.
#
# FullFT and LoRA sit on separate grids an order of magnitude apart (runbook
# section 23.4), so column 5 pairs the 5th point of each: 2e-05 against 0.0002.
#
# Trains on math_train.jsonl and is scored on math_test.jsonl alone. Not both:
# `parse_final_accuracy` means across whatever datasets were evaluated, so
# scoring against the other one too would make every point of this panel an
# average of two datasets. `arm_env` sets EVAL_DATASETS per arm.
#
# **Resumable.** The ledger gets `status: "ok"` per finished arm and the sweep
# skips those next time, so an interrupted node picks up where it stopped: just
# re-run the same script. Do not point two nodes at one RESULTS file.
#
# `analyze` takes globs, and each panel reassembles across its seven columns:
#
#   python -m tools.lora_regret.analyze --ledgers 'results/e4_math_lr*.jsonl' ...
#
# **Every RL arm is an 8-GPU arm.** FullFT has no choice -- TP=4/DP=2 is the
# only configuration it fits in (section 22.2) -- and the LoRA arms were
# measured at eight.
exec env MATRIX=e4 METHOD_RE='^(full-na-na-math-lr2e\-05|lora-r(1|16|256)-all-math-lr0\.0002)-s' RESULTS=results/e4_math_lr5.jsonl EXPECT_ARMS=4 \
    bash "$(dirname "${BASH_SOURCE[0]}")/campaign.sh" "$@"
