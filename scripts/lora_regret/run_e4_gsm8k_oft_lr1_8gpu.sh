#!/usr/bin/env bash
#
# E4 OFT, gsm8k panel, learning-rate column 1 of 6:
# b8/b128/b1024 at 5e-06. Book a WHOLE 8-GPU node.
#
#   source scripts/lora_regret/env_v0516.sh
#   cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
#   bash scripts/lora_regret/run_e4_gsm8k_oft_lr1_8gpu.sh
#
# The fourteen OFT wrappers partition 42 arms: two datasets x seven learning
# rates x three capacities. e4_protocol.sh supplies the same training and
# evaluation protocol as the completed FullFT/LoRA sweep.
#
# Resumable: rerunning this script skips arms already recorded with status
# "ok". Use only one writer per RESULTS file.
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/e4_protocol.sh"

exec env MATRIX=e4 METHOD_RE='^oftscout-b(8|128|1024)-all-gsm8k-lr5e\-06-s' RESULTS=results/e4_gsm8k_oft_lr1.jsonl EXPECT_ARMS=3 ALLOW_OFT=1 \
    bash "${HERE}/campaign.sh" "$@"
