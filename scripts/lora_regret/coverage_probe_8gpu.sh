#!/usr/bin/env bash
#
# Coverage probe: the 8-GPU subset.  Book a WHOLE node.
#
#   bash scripts/lora_regret/coverage_probe_8gpu.sh
#
# 7 runs -- every RL code path: FullFT, plus LoRA and OFT at each of the three
# placements (all / attention-only / MLP-only). Three rollouts each, sequential.
#
# RUN THIS ONE FIRST if you can only afford one. Every path in it has never
# executed: the RL launcher has never produced a real accuracy line, OFT under
# policy gradient has never run in any form, and e4place's MLP placement was not
# even in the earlier per-method plan. The SFT paths the other two scripts cover
# already have a passing smoke behind them.
#
# Eight is what the real RL arms get -- the policy and the rollout engine share
# the node -- so eight is what they are measured on.
#
# The most expensive of the three, and the longest: RL rollouts include
# generation, so these dominate the campaign's wall clock. That is precisely why
# their per-rollout time is the number most worth measuring before committing.
#
# Preflight runs at stage `e4` (needs 8 GPUs).
#
# Writes into the same results/probe ledger directory as its 1- and 4-GPU
# siblings, so the final report fills in as each finishes, in any order.

# RL FullFT is skipped, deliberately, and its failure is already characterised:
# with `--no-offload-train` the 8B policy weights plus distributed-optimizer
# state stay resident on all eight cards, so colocated SGLang cannot `resume`
# the KV-cache arena it paused -- torch_memory_saver reports cudaError 2 at
# `func=resume`, ~7 minutes into every attempt, right after update_weights.
# That is a missing train-offload path, not a configuration to retry, and each
# retry costs the node 7 minutes plus a slow Ray teardown before the OFT arms
# (the ones the tiled kernel just unblocked) get to run at all.
#
# DELETE THIS LINE to cover it again once RL FullFT has an offload path.
SKIP=full
exec env ONLY_GPUS=8 SKIP_METHODS="${SKIP:-}" bash "$(dirname "${BASH_SOURCE[0]}")/coverage_probe.sh" "$@"
