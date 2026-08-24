#!/usr/bin/env bash
# Qwen2.5-3B rung, OFT arms: oft/sync and oft/async_db (constraint 8: OFT has no
# single-slot async arm; the engine rejects it with "distributed non-double-
# buffer OFT adapter sync ... not supported"). Two sequential 4-GPU jobs.
# Launch-verified on the cu130 env, 4xB200, 2026-08-23 with NUM_ROLLOUT=1:
# sync ok (251 s, payload 426 MB), async_db ok (161 s, payload 106.5 MB).
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
adapter_first_select_model q25_3b
run_harness "${CAMPAIGN:-phase1-q25-3b-oft-$(date +%Y%m%d_%H%M%S)}" \
    --profile main --models qwen25_3b --pefts oft --modes sync,async_db \
    --num-rollout "${NUM_ROLLOUT:-4}" --no-eval
