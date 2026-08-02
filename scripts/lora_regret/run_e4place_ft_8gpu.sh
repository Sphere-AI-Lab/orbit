#!/usr/bin/env bash
#
# E4-place, FullFT: the reference line for the placement dashboard.  OPTIONAL.
#
#   bash scripts/lora_regret/run_e4place_ft_8gpu.sh
#
# 7 arms. **Drop this one first under budget pressure.** These arms answer
# no placement question -- there is no adapter to place -- and they duplicate
# what E4 already measures on the same four learning rates. They are tagged
# `place` in their names for exactly that reason: untagged they would be
# byte-identical to E4's FullFT arms and collide the moment both ledgers are
# globbed into `analyze`.
#
# What they buy is a reference line inside `math-gsm8k-rl-placement`'s own wandb
# dashboard, so the placement cells can be read without cross-referencing
# another project. If you skip them, read FullFT from results/e4_full.jsonl.
exec env MATRIX=e4place METHOD_RE='^full-' RESULTS=results/e4place_full.jsonl EXPECT_ARMS=7 \
    bash "$(dirname "${BASH_SOURCE[0]}")/campaign.sh" "$@"
