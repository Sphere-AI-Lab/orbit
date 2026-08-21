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

case "${dataset}" in
    math|gsm8k) ;;
    *) echo "unsupported dataset: ${dataset}" >&2; exit 2 ;;
esac
if [[ ! "${column}" =~ ^[1-7]$ ]]; then
    echo "column must be an integer from 1 through 7, got: ${column}" >&2
    exit 2
fi

# shellcheck disable=SC1091
source "${HERE}/env.sh"
set_env2_rollout_budget "${dataset}"

OFT_LR=(unused 5e-07 1e-06 3e-06 7e-06 2e-05 4e-05 0.0001)
OFT_LR_RE=(unused '5e\-07' '1e\-06' '3e\-06' '7e\-06' '2e\-05' '4e\-05' '0\.0001')

results="${E4_ENV2_RESULTS_DIR}/e4_${dataset}_oft_lr${column}.jsonl"
campaign="${ORBIT_ICLR_ROOT}/scripts/lora_regret/campaign.sh"

printf '\n=== env2 OFT rerun: %s lr%s ===\n' "${dataset}" "${column}"
printf 'OFT block=128 modules=all lr=%s rollouts=%s\n' \
    "${OFT_LR[${column}]}" "${NUM_ROLLOUT}"
printf 'results=%s\nlogs=%s\nwandb=%s\ncheckpoints=%s\n' \
    "${results}" "${LORA_REGRET_LOG_DIR}" "${WANDB_DIR}" "${LORA_REGRET_CKPT_DIR}"

MATRIX=e4oftenv2 \
METHOD_RE="^oftenv2-b128-all-${dataset}-lr${OFT_LR_RE[${column}]}-s" \
RESULTS="${results}" \
EXPECT_ARMS=1 \
ALLOW_OFT=1 \
PREFLIGHT_STAGE=e4oftenv2 \
bash "${campaign}" "$@"
