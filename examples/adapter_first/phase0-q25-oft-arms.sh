#!/usr/bin/env bash
# Phase-0 q25 four-arm qualification, OFT arms: oft/sync, oft/async_db and the
# full-FT async control (OFT has no single-slot NCCL arm — design constraint 8).
# Three sequential 4-GPU jobs on Qwen2.5-0.5B, 4 rollouts each, eval off.
# Verified on the cu130 env, 4xB200, 2026-08-23: 3/3 ok (454 s cold, 143 s, 180 s).
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
adapter_first_select_model q25_05b
run_harness "${CAMPAIGN:-phase0-q25-oft-$(date +%Y%m%d_%H%M%S)}" \
    --profile q25 --pefts oft --modes sync,async_db,async_fullft \
    --num-rollout "${NUM_ROLLOUT:-4}" --no-eval
