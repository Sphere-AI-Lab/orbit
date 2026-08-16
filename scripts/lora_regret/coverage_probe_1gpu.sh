#!/usr/bin/env bash
#
# Coverage probe: the 1-GPU subset.  Book ONE card.
#
#   bash scripts/lora_regret/coverage_probe_1gpu.sh
#
# 8 runs -- every SFT LoRA and OFT code path, across both datasets and all three
# placements (all / attention-only / MLP-only). Three rollouts each, sequential.
#
# These are the arms that run on one card in the real sweep, so one card is what
# they are measured on: a per-rollout time from a 4- or 8-GPU node would estimate
# a machine that will never run them.
#
# The cheapest of the three scripts by a wide margin, and the one to run first if
# you are validating the environment rather than the RL stack -- it needs no
# multi-GPU reservation at all.
#
# Preflight runs at stage `e1-lora` (needs >= 1 GPU), so it passes on a one-card
# box instead of failing an audit for eight cards this script never uses.
#
# Writes into the same results/probe ledger directory as its 4- and 8-GPU
# siblings, so the final report fills in as each finishes, in any order.

exec env ONLY_GPUS=1 bash "$(dirname "${BASH_SOURCE[0]}")/coverage_probe.sh" "$@"
