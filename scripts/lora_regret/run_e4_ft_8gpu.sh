#!/usr/bin/env bash
#
# E4, FullFT: the reference line C5 is read against.  Book a WHOLE node.
#
#   bash scripts/lora_regret/run_e4_ft_8gpu.sh
#
# 4 arms, ~5 h -- one per learning rate on the half-decade grid centred at 1e-6.
# Half-decade rather than E1's 0.3 because C5's second half is about the WIDTH
# of the performant band, which needs coverage more than resolution.
#
# Eight GPUs is not a preference here. At TP=1 the standing cost is
# (2+4)*P/TP + 12*P/N = 60 GB per card against a step that wants ~20 more, and
# the arm dies in the fp32 cross-entropy logits 694 MiB short. The launcher
# derives TP=4/DP=2 from GPUS_PER_NODE=8 and prints it at launch (runbook §22.2).
#
# Run this before or alongside run_e4_lora_8gpu.sh -- they are independent, and
# C5 is the difference between them.
exec env MATRIX=e4 METHOD_RE='^full-' RESULTS=results/e4_full.jsonl EXPECT_ARMS=4 \
    bash "$(dirname "${BASH_SOURCE[0]}")/campaign.sh" "$@"
