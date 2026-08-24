#!/usr/bin/env bash
# Qwen3-30B-A3B-Instruct-2507 bf16 rung, constraint-8 arm assignment. Every
# 30B case in the harness is an 8-GPU job (gpu_total=8), so this needs an
# 8-GPU allocation, e.g. HTCondor: request_gpus = 8,
# requirements = (CUDADeviceName == "NVIDIA B200").
# NOT launch-verified: the 2026-08-23 sweep ran in a 4-GPU slot. Checkpoints
# (HF + torch_dist) exist at the env.sh defaults.
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
if [ "$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)" -lt 8 ]; then
    echo "FATAL: the qwen3_30b harness cases need 8 GPUs in this allocation" >&2
    exit 1
fi
adapter_first_select_model q3_30b
# env.sh's 30B rung is Qwen3-30B-A3B-Instruct-2507 (rope_theta 10000000); the
# shared model-args file defaults --rotary-base to the base model's 1000000 and
# hf_validate_args rejects the mismatch (30B probe, 2026-08-23).
export MODEL_ARGS_ROTARY_BASE="${MODEL_ARGS_ROTARY_BASE:-10000000}"
rc=0
run_harness "${CAMPAIGN:-phase1-q3-30b-$(date +%Y%m%d_%H%M%S)}-oft" \
    --profile q3_30b --pefts oft --modes sync,async_db \
    --num-rollout "${NUM_ROLLOUT:-4}" --no-eval || rc=$?
run_harness "${CAMPAIGN:-phase1-q3-30b-$(date +%Y%m%d_%H%M%S)}-lora" \
    --profile q3_30b --pefts lora --modes sync,async,async_db \
    --num-rollout "${NUM_ROLLOUT:-4}" --no-eval || rc=$?
exit "$rc"
