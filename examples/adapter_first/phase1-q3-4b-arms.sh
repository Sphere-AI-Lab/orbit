#!/usr/bin/env bash
# Qwen3-4B-Instruct-2507 bf16 rung, all four arms under the constraint-8
# assignment: OFT sync + OFT double-buffer async, LoRA sync/async/async_db.
# Five sequential 4-GPU jobs. Requires the 4B torch_dist at Q3_4B_TORCH_DIST
# (default: the group hf_models path); produce it with
#   python tools/convert_hf_to_torch_dist.py --hf-checkpoint $Q3_4B_HF --save $Q3_4B_TORCH_DIST
# (33 s on one B200 to node-local disk).
# Launch-verified on the cu130 env, 4xB200, 2026-08-23 with NUM_ROLLOUT=1: 5/5 ok.
set -uo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/env.sh"
adapter_first_select_model q3_4b
rc=0
run_harness "${CAMPAIGN:-phase1-q3-4b-$(date +%Y%m%d_%H%M%S)}-oft" \
    --profile q3_4b --precisions bf16 --pefts oft --modes sync,async_db \
    --num-rollout "${NUM_ROLLOUT:-4}" --no-eval || rc=$?
run_harness "${CAMPAIGN:-phase1-q3-4b-$(date +%Y%m%d_%H%M%S)}-lora" \
    --profile q3_4b --precisions bf16 --pefts lora --modes sync,async,async_db \
    --num-rollout "${NUM_ROLLOUT:-4}" --no-eval || rc=$?
exit "$rc"
