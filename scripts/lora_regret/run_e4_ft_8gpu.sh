#!/usr/bin/env bash
#
# E4, FullFT: the reference line C5 is read against.  Book a WHOLE node.
#
#   bash scripts/lora_regret/run_e4_ft_8gpu.sh
#
# 7 arms -- one per learning rate on the shared half-decade grid, 1e-06 .. 1e-03.
# ~2.3 h per arm per 100 rollouts at the measured FullFT pace (59 s/rollout,
# plus eval), so ~16 h at NUM_ROLLOUT=100 and ~24 h at 150.
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
exec env MATRIX=e4 METHOD_RE='^full-' RESULTS=results/e4_full.jsonl EXPECT_ARMS=7 \
    bash "$(dirname "${BASH_SOURCE[0]}")/campaign.sh" "$@"
