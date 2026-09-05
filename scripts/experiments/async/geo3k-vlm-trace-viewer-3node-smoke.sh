#!/usr/bin/env bash
# Geo3K VLM multi-turn, 3 nodes — trace-viewer wiring smoke.
#
# PURPOSE: prove that the trace viewer attaches purely through
# --custom-rollout-log-function-path, on the real 3-node topology, and actually
# writes trace steps. This is a plumbing check, not a training run: 3 rollout
# steps, small batches, short responses. Do not read anything into the rewards.
#
# It derives from the production prefetch2 recipe so the rollout path under test
# is the real one (fully-async driver + geo3k multi-turn custom generate), then
# shrinks the workload and appends the trace flags. argparse is last-wins, so
# every override below beats the base value.
#
# What it exercises that a unit test cannot:
#   - the hook resolving by dotted path inside the ray worker process
#   - traces written from the actual accepted-rollout boundary, multi-node
#   - real PIL images surviving into per-turn trace records
#   - default rollout metrics still logging (hook returns False)
#
# Submit:
#   JOB_NAME=geo3k-trace-smoke TIME=01:00:00 NODES=3 ORBIT_ENV_NAME=orbit_zeju \
#   bash scripts/slurm/submit.sh async/geo3k-vlm-trace-viewer-3node-smoke
#
# Then verify (on the login node, against the printed run dir):
#   python3 examples/model_response_trace_viewer/verify_trace_run.py <run-dir>

set -euo pipefail

VARIANT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck disable=SC1091
source "$VARIANT_DIR/geo3k-vlm-multi-turn-fully-async-prefetch2-3node.sh"

# Smoke, not a 72h production run.
EXPERIMENT_TIME=01:00:00

# RUN_DIR is exported by submit.sh and set before launch_orbit.sbatch sources
# this recipe, so traces land beside run.log rather than in a stale directory.
TRACE_DIR="${RUN_DIR:-$ORBIT_REPO/runs/$RUN_NAME}/traces"

ORBIT_ARGS+=(
   # --- the thing under test -------------------------------------------------
   # The viewer attaches here and nowhere else: no call site in rollout_manager,
   # no writer module in orbit/. The hook returns False so Orbit still emits its
   # own rollout metrics on top.
   --custom-rollout-log-function-path examples.model_response_trace_viewer.hook.log_rollout_data
   --save-model-response-trace-dir    "$TRACE_DIR"
   # This smoke accepts exactly 4 prompts * 4 samples, so the bounded export is
   # still a direct check against the full accepted batch.
   --model-response-trace-max-samples-per-step 16

   # --- shrink to a smoke ----------------------------------------------------
   --num-rollout              3
   --rollout-batch-size       4
   --n-samples-per-prompt     4
   # 4 prompts * 4 samples = 16 accepted samples per step; the trainer must
   # consume exactly that, so global-batch-size has to match or training stalls
   # waiting for samples that never arrive.
   --global-batch-size        16
   --rollout-max-response-len 1024
   # Prefetch is pointless over 3 steps and only widens the staleness window.
   --fully-async-prefetch-batches 1

   # --- keep the smoke out of the real dashboards ---------------------------
   --wandb-project trace_viewer
)
