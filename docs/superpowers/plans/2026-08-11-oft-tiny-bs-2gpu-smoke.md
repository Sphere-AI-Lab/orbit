# OFT Tiny-Block Two-GPU Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reusable two-GPU training smoke that runs the Llama-3.1-8B OFT production path sequentially at block sizes 4, 8, and 16 and leaves durable, mechanically verifiable evidence for every arm.

**Architecture:** A Bash campaign wrapper owns topology validation, per-arm environment isolation, launcher execution, and atomic status files. Pytest drives the real wrapper against a temporary fake training launcher, so it verifies observable commands, files, exit codes, and continuation behavior without importing Orbit or requiring GPUs.

**Tech Stack:** Bash, GNU `timeout` on the Linux compute node, Python `subprocess`, stdlib `unittest` (pytest-compatible).

## Global Constraints

- Work only in `/Users/zqiu/Documents/GitHub/orbit-iclr/orbit/.worktrees/oft-bs4` on `codex/oft-bs4`; preserve all pre-existing transport changes.
- Use exactly the already-visible two GPUs; validate `CUDA_VISIBLE_DEVICES` but never rewrite it.
- Set both trainer tensor parallelism and SGLang rollout-engine tensor parallelism to 2.
- Run block sizes in the literal order `4 8 16`, with three rollouts and evaluation interval 2.
- Set `SAVE_INTERVAL` to the empty string so the launcher does not force a final checkpoint.
- Require an absolute, existing `RUN_ROOT`; refuse before launching if any campaign-owned arm/status path already exists.
- Put every console stream, Orbit log, timing extract, environment record, and atomic completion status under `RUN_ROOT`.
- Use a private Ray lifecycle and a distinct Ray temporary directory for every arm; clear inherited Ray address/port variables inside each arm.
- Continue through BS16 after an ordinary launcher or verification failure; fail the campaign overall if any arm fails.
- The script does not submit Condor jobs, activate environments, select devices, delete processes, or remove output.

---

### Task 1: Two-GPU OFT tiny-block training smoke

**Files:**
- Create: `scripts/lora_regret/smoke_oft_tiny_bs_2gpu.sh`
- Create: `tests/fast/utils/test_oft_tiny_bs_2gpu_smoke.py`

**Interfaces:**
- Consumes: `RUN_ROOT` (required absolute existing directory), `CUDA_VISIBLE_DEVICES` (must name exactly two devices), optional `OFT_TINY_SMOKE_LAUNCHER`, `OFT_TINY_SMOKE_ARM_TIMEOUT` (default `90m`, empty disables), and `DRY_RUN`.
- Produces: `RUN_ROOT/bs{4,8,16}/{console.log,orbit.log,environment.txt,timings.txt,completion.status}` and `RUN_ROOT/completion.status`.
- Invokes: `examples/high_precision/run-llama3_1-8b-bf16-rl-math-gsm8k.sh` by default.

- [ ] **Step 1: Write the failing behavioral tests**

Create a fake launcher that appends its selected environment to a call ledger and writes these real completion markers to `RUN_LOG`:

```python
FAKE_SUCCESS = r'''#!/usr/bin/env bash
set -eu
printf '%s\t%s\t%s\t%s\t%s\t<%s>\n' \
  "${OFT_BLOCK_SIZE}" "${GPUS_PER_NODE}" \
  "${TENSOR_MODEL_PARALLEL_SIZE}" "${ROLLOUT_NUM_GPUS_PER_ENGINE}" \
  "${NUM_ROLLOUT}" "${SAVE_INTERVAL}" >> "${CALL_LEDGER}"
printf '%s\n' \
  'weight_sync stage=update_weights_complete rank=0' \
  'progress rollout=2/2 completed=3/3 remaining=0 elapsed=00:00:03' \
  'Training driver exited with code 0' >> "${RUN_LOG}"
'''
```

Run the real wrapper with `CUDA_VISIBLE_DEVICES=GPU-a,GPU-b`, an existing temporary `RUN_ROOT`, the fake launcher, and an empty timeout. Assert:

1. Success invokes exactly `4, 8, 16` and every ledger row records `2` trainer GPUs, TP=2, rollout TP=2, three rollouts, and `< >` with no character between the angle brackets for `SAVE_INTERVAL`.
2. Every arm and campaign status has `final_exit_code=0`; every arm has console, Orbit, environment, and timing files.
3. One visible GPU exits 2 before the fake launcher runs and creates no arm directory.
4. An existing `RUN_ROOT/bs4` exits 2 before the fake launcher runs and does not modify that directory.
5. A fake launcher that exits 7 for BS8 still receives BS16; BS8 records launcher/final code 7 and the campaign exits nonzero.
6. A zero-exit fake that omits the rollout marker records `launcher_exit_code=0`, `verification_exit_code=1`, and fails the campaign.

- [ ] **Step 2: Run the tests and witness RED**

Run:

```bash
/Users/zqiu/Documents/GitHub/orbit-iclr/orbit/.venv/bin/python -m unittest tests.fast.utils.test_oft_tiny_bs_2gpu_smoke
```

Expected: fail because `scripts/lora_regret/smoke_oft_tiny_bs_2gpu.sh` does not exist.

- [ ] **Step 3: Implement the minimal campaign wrapper**

Implement these behaviors:

```bash
BLOCK_SIZES=(4 8 16)
GPUS_PER_NODE=2
RAY_NUM_GPUS=2
ROLLOUT_NUM_GPUS_PER_ENGINE=2
TENSOR_MODEL_PARALLEL_SIZE=2
PIPELINE_MODEL_PARALLEL_SIZE=1
PEFT_METHOD=oft
NUM_ROLLOUT=3
EVAL_INTERVAL=2
SAVE_INTERVAL=""
WANDB_MODE=offline
ORBIT_RAY_LIFECYCLE=private
ORBIT_LOG_WEIGHT_SYNC=1
```

For each arm, record the exported values in `environment.txt`, assign arm-local `RUN_LOG`, `SAVE_DIR`, `WANDB_DIR`, `WANDB_RUN_NAME`, `LAUNCHER_NAME`, and `RAY_TEMP_DIR`, then run the launcher through `timeout --signal=TERM --kill-after=120s` unless the timeout is empty. Tee the combined stream to `console.log`.

Verification requires all three literal markers in `orbit.log`:

```text
Training driver exited with code 0
progress rollout=2/2 completed=3/3 remaining=0
stage=update_weights_complete
```

Extract phase timing lines containing `done elapsed=` into `timings.txt`. Write every `completion.status` through a temporary sibling followed by `mv`; include block size where applicable, launcher exit code, verification exit code, final exit code, duration seconds, and UTC completion time. Print a concise per-arm and campaign verdict and return the campaign final code.

- [ ] **Step 4: Run the focused suite and verify GREEN**

Run:

```bash
/Users/zqiu/Documents/GitHub/orbit-iclr/orbit/.venv/bin/python -m unittest tests.fast.utils.test_oft_tiny_bs_2gpu_smoke
bash -n scripts/lora_regret/smoke_oft_tiny_bs_2gpu.sh
git diff --check
```

Expected: all pass.

- [ ] **Step 5: Review the production invocation without launching training**

Run with `DRY_RUN=1`, a fresh temporary absolute `RUN_ROOT`, and no GPU requirement. Verify the printed plan names BS4, BS8, and BS16 in order, the production launcher path, TP=2, three rollouts, empty checkpoint interval, and the exact output paths. Remove only the temporary directory created specifically for this dry run.

- [ ] **Step 6: Commit only the smoke deliverable**

```bash
git add \
  docs/superpowers/plans/2026-08-11-oft-tiny-bs-2gpu-smoke.md \
  scripts/lora_regret/smoke_oft_tiny_bs_2gpu.sh \
  tests/fast/utils/test_oft_tiny_bs_2gpu_smoke.py
git commit -m "test(oft): add two-gpu tiny-block training smoke"
```

Do not stage the pre-existing `ipc.py`, `sglang_engine.py`, or `tests/test_peft_ipc_transport.py` changes.
