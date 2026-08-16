#!/usr/bin/env bash
#
# E4-place, LoRA: attention vs MLP at matched parameters.  Book a WHOLE node.
#
#   bash scripts/lora_regret/run_e4place_lora_8gpu.sh
#
# 14 arms -- attention-only r256 against MLP-only r92, seven learning rates
# each on E4's own grid so the placement result and the rank result are
# comparable arm for arm.
#
# **r92, not the post's r128.** Orbit fuses qkv and gate+up, so the post's pair
# is not matched in this layout; r92 is the count solved for Orbit's shapes by
# `orbit.utils.peft_param_match.matched_mlp_rank`. An unmatched pair would
# compare placement and capacity at once, which is the confound this matrix
# exists to avoid.
#
# There is no all-modules cell: E4 already runs LoRA r256 all-modules on this
# exact grid, so read it from results/e4_lora.jsonl and glob both files into
# `analyze`. Including it here would produce four byte-identical arm names.
#
# The post studies placement for SFT only. This is one of the two cells that go
# beyond it.
exec env MATRIX=e4place METHOD_RE='^lora-' RESULTS=results/e4place_lora.jsonl EXPECT_ARMS=14 \
    bash "$(dirname "${BASH_SOURCE[0]}")/campaign.sh" "$@"
