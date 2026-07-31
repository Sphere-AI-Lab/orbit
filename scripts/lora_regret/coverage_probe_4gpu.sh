#!/usr/bin/env bash
#
# Coverage probe: the 4-GPU subset.  Book FOUR cards.
#
#   bash scripts/lora_regret/coverage_probe_4gpu.sh
#
# 2 runs -- SFT full fine-tuning on Tulu3 and on OpenThoughts3. Three rollouts
# each, sequential.
#
# Four is not a preference: per-GPU optimizer state is 4*P + 12*P/N GB, so at
# 8.03B a FullFT arm is 56 GB/GPU at N=4 and 80 GB at N=2 with nothing left for
# activations. tools/lora_regret/models.py solves it per model and the launcher
# refuses below the floor. Measuring these on 8 cards would halve the per-GPU
# state and estimate an arm the sweep will not run.
#
# The shortest of the three scripts -- two runs -- so it is cheap to slot into
# any half-node window.
#
# Preflight runs at stage `e1-full` (needs >= 4 GPUs).
#
# Writes into the same results/probe ledger directory as its 1- and 8-GPU
# siblings, so the final report fills in as each finishes, in any order.

exec env ONLY_GPUS=4 bash "$(dirname "${BASH_SOURCE[0]}")/coverage_probe.sh" "$@"
