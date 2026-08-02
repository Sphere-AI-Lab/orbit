#!/usr/bin/env bash
#
# E4, learning-rate column 7 of 7, BOTH panels: FullFT at 0.0001 and LoRA
# r1/r16/r256 at 0.001, on gsm8k and on math.  Book a WHOLE node.
#
#   bash scripts/lora_regret/run_e4_lr7_8gpu.sh
#
# 8 arms = 4 curves x 2 datasets -- one point of each of Figure 6's two panels.
# The seven `run_e4_lr*_8gpu.sh` scripts partition e4's FullFT and LoRA cells
# exactly: 7 x 8 = 56, no overlap, no gaps. A column on its own cannot give an
# argmin: every claim in C5 is about the shape ACROSS columns, so a partial run
# is a partial curve rather than a partial answer.
#
# FullFT and LoRA sit on separate grids an order of magnitude apart (runbook
# section 23.4), so column 7 pairs the 7th point of each: 0.0001 against 0.001.
#
# Each arm trains on its own dataset and is scored on that dataset alone --
# `parse_final_accuracy` means across whatever was evaluated, so scoring a
# gsm8k arm on math too would make every point of the gsm8k panel an average
# of two datasets. `arm_env` sets EVAL_DATASETS per arm; nothing to pass here.
#
# **Resumable.** The ledger gets `status: "ok"` per finished arm and the sweep
# skips those next time, so an interrupted node picks up where it stopped: just
# re-run the same script. Do not point two nodes at one RESULTS file.
#
# `analyze` takes globs, and each panel reassembles across the seven columns:
#
#   python -m tools.lora_regret.analyze --ledgers 'results/e4_lr*.jsonl' ...
#
# **Every RL arm is an 8-GPU arm.** FullFT has no choice -- TP=4/DP=2 is the
# only configuration it fits in (section 22.2) -- and the LoRA arms were
# measured at eight.
exec env MATRIX=e4 METHOD_RE='^(full-na-na-(gsm8k|math)-lr0\.0001|lora-r(1|16|256)-all-(gsm8k|math)-lr0\.001)-s' RESULTS=results/e4_lr7.jsonl EXPECT_ARMS=8 \
    bash "$(dirname "${BASH_SOURCE[0]}")/campaign.sh" "$@"
