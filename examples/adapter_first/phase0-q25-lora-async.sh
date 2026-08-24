#!/usr/bin/env bash
# Phase-0 q25 four-arm qualification, LoRA single-slot async arm (the NCCL arm
# OFT cannot provide). One 4-GPU job on Qwen2.5-0.5B, 4 rollouts, eval off.
# Verified on the cu130 env, 4xB200, 2026-08-23: ok (457 s, warm sync 0.06 s).
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
adapter_first_select_model q25_05b
run_harness "${CAMPAIGN:-phase0-q25-lora-$(date +%Y%m%d_%H%M%S)}" \
    --profile q25 --pefts lora --modes async \
    --num-rollout "${NUM_ROLLOUT:-4}" --no-eval
