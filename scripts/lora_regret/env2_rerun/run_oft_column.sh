#!/usr/bin/env bash
# Run one clean env2 OFT column: block 128 on all target modules.

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

results="${E4_ENV2_RESULTS_DIR}/e4_${dataset}_oft_lr${column}.jsonl"
campaign="${ORBIT_ICLR_ROOT}/scripts/lora_regret/campaign.sh"

printf '\n=== env2 OFT rerun: %s lr%s ===\n' "${dataset}" "${column}"
printf 'OFT block=128 modules=all lr=%s\n' "${OFT_LR[${column}]}"
printf 'results=%s\nlogs=%s\nwandb=%s\ncheckpoints=%s\n' \
    "${results}" "${LORA_REGRET_LOG_DIR}" "${WANDB_DIR}" "${LORA_REGRET_CKPT_DIR}"

MATRIX=e4oftenv2 \
METHOD_RE="^oftenv2-b128-all-${dataset}-lr${OFT_LR_RE[${column}]}-s" \
RESULTS="${results}" \
EXPECT_ARMS=1 \
ALLOW_OFT=1 \
PREFLIGHT_STAGE=e4oftenv2 \
bash "${campaign}" "$@"
