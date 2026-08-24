#!/usr/bin/env bash
# Run one clean env2 E4 column: FullFT lrN plus LoRA's previous-grid lr(N-1).

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

FULLFT_LR=(unused 5e-08 1e-07 3e-07 7e-07 2e-06 4e-06 1e-05)
FULLFT_LR_RE=(unused '5e\-08' '1e\-07' '3e\-07' '7e\-07' '2e\-06' '4e\-06' '1e\-05')
LORA_LR=(unused 2e-06 5e-06 1e-05 3e-05 7e-05 0.0002 0.0004)
LORA_LR_RE=(unused '2e\-06' '5e\-06' '1e\-05' '3e\-05' '7e\-05' '0\.0002' '0\.0004')
LORA_MATRIX=(unused e4lr0 e4 e4 e4 e4 e4 e4)

results="${E4_ENV2_RESULTS_DIR}/e4_${dataset}_lr${column}.jsonl"
campaign="${ORBIT_ICLR_ROOT}/scripts/lora_regret/campaign.sh"

printf '\n=== env2 rerun: %s lr%s ===\n' "${dataset}" "${column}"
printf 'FullFT lr=%s; LoRA r1/r16/r256 lr=%s; rollouts=%s\n' \
    "${FULLFT_LR[${column}]}" "${LORA_LR[${column}]}" "${NUM_ROLLOUT}"
printf 'results=%s\nlogs=%s\nwandb=%s\ncheckpoints=%s\n' \
    "${results}" "${LORA_REGRET_LOG_DIR}" "${WANDB_DIR}" "${LORA_REGRET_CKPT_DIR}"

MATRIX=e4 \
METHOD_RE="^full-na-na-${dataset}-lr${FULLFT_LR_RE[${column}]}-s" \
RESULTS="${results}" \
EXPECT_ARMS=1 \
bash "${campaign}" "$@"

MATRIX="${LORA_MATRIX[${column}]}" \
METHOD_RE="^lora-r(1|16|256)-all-${dataset}-lr${LORA_LR_RE[${column}]}-s" \
RESULTS="${results}" \
EXPECT_ARMS=3 \
bash "${campaign}" "$@"
