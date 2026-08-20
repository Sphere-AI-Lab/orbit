#!/usr/bin/env bash
#
# E4 LR0 extension, Math panel: LoRA r1/r16/r256 at 2e-06.
# Book a whole 8-GPU node. Finished arms recorded in the ledger are skipped.
#
#   source scripts/lora_regret/env_v0516.sh
#   cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
#   bash scripts/lora_regret/run_e4_math_lr0_8gpu.sh
set -uo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${HERE}/e4_protocol.sh"

exec env MATRIX=e4lr0 METHOD_RE='^lora-r(1|16|256)-all-math-lr2e\-06-s' \
    RESULTS=results/e4_math_lr0.jsonl EXPECT_ARMS=3 \
    bash "${HERE}/campaign.sh" "$@"
