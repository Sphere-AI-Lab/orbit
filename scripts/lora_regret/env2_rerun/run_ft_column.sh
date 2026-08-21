#!/usr/bin/env bash
# Run one clean env2 FullFT column: the single FullFT arm at lrN.
#
# Same arm as the FullFT half of run_column.sh, but into its own ledger so a
# FullFT-only node and a LoRA-only node can run the same column at once.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ "$#" -lt 2 ]]; then
    echo "usage: $0 {math|gsm8k} {1..7} [campaign args...]" >&2
    exit 2
fi
dataset=$1
column=$2
shift 2

# shellcheck disable=SC1091
source "${HERE}/columns.sh"
check_dataset "${dataset}"
check_column "${column}"

# shellcheck disable=SC1091
source "${HERE}/env.sh"

results="${E4_ENV2_RESULTS_DIR}/e4_${dataset}_ft_lr${column}.jsonl"
campaign="${ORBIT_ICLR_ROOT}/scripts/lora_regret/campaign.sh"

printf '\n=== env2 FullFT rerun: %s lr%s ===\n' "${dataset}" "${column}"
printf 'FullFT lr=%s\n' "${FULLFT_LR[${column}]}"
printf 'results=%s\nlogs=%s\nwandb=%s\ncheckpoints=%s\n' \
    "${results}" "${LORA_REGRET_LOG_DIR}" "${WANDB_DIR}" "${LORA_REGRET_CKPT_DIR}"

MATRIX=e4 \
METHOD_RE="^full-na-na-${dataset}-lr${FULLFT_LR_RE[${column}]}-s" \
RESULTS="${results}" \
EXPECT_ARMS=1 \
bash "${campaign}" "$@"
