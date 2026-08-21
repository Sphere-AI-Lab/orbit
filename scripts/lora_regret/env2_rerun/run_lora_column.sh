#!/usr/bin/env bash
# Run one clean env2 LoRA column for one rank: the single LoRA arm at lrN.
#
# Same arm as one third of the LoRA half of run_column.sh, but into a ledger
# keyed by rank, so the three ranks can run the same column on three nodes.

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ "$#" -lt 3 ]]; then
    echo "usage: $0 {math|gsm8k} {1|16|256} {1..7} [campaign args...]" >&2
    exit 2
fi
dataset=$1
rank=$2
column=$3
shift 3

# shellcheck disable=SC1091
source "${HERE}/columns.sh"
check_dataset "${dataset}"
set_dataset_rollouts "${dataset}"
check_lora_rank "${rank}"
check_column "${column}"

# shellcheck disable=SC1091
source "${HERE}/env.sh"

results="${E4_ENV2_RESULTS_DIR}/e4_${dataset}_lora_r${rank}_lr${column}.jsonl"
campaign="${ORBIT_ICLR_ROOT}/scripts/lora_regret/campaign.sh"

printf '\n=== env2 LoRA rerun: %s r%s lr%s ===\n' "${dataset}" "${rank}" "${column}"
printf 'LoRA r%s modules=all lr=%s (matrix %s)\n' \
    "${rank}" "${LORA_LR[${column}]}" "${LORA_MATRIX[${column}]}"
printf 'rollouts=%s\n' "${NUM_ROLLOUT}"
printf 'results=%s\nlogs=%s\nwandb=%s\ncheckpoints=%s\n' \
    "${results}" "${LORA_REGRET_LOG_DIR}" "${WANDB_DIR}" "${LORA_REGRET_CKPT_DIR}"

MATRIX="${LORA_MATRIX[${column}]}" \
METHOD_RE="^lora-r${rank}-all-${dataset}-lr${LORA_LR_RE[${column}]}-s" \
RESULTS="${results}" \
EXPECT_ARMS=1 \
bash "${campaign}" "$@"
