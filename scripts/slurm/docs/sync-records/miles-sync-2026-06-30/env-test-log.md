# Runnable-env test log — sync-upstream-20260630

**Goal:** validate that the 2026-06-30 upstream sync (branch `sync-upstream-20260630`, sglang/pins deferred) actually runs on the **existing `miles` conda env**, by launching a real 3-node run of `scripts/experiments/async/geo3k-vlm-multi-turn-fully-async-prefetch2-3node.sh` and watching RL metrics for regression.

**Date:** 2026-06-30 (two sessions: ~09:35–11:40 UTC, then ~18:25–21:01 UTC).
**Recipe:** Qwen3-VL-8B, geo3k multi-turn, fully-async, 3 nodes (1 trainer + 2 samplers, 24 GPUs), `--megatron-to-hf-mode bridge`, no eval/ckpt.
**Submit knobs used:** `JOB_NAME=geo3k-async-mt-pf2-8b NODES=3 TIME=72:00:00 MILES_ENV_NAME=miles SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true` + `SBATCH_EXTRA="--qos=viga ..."`. wandb: `M3TRL/async_envpack`, group `geo3k-async-mt-pf2-8b`.

---

## TL;DR

- **One real sync regression found and fixed:** PR **#1367** switched the rollout router/session-server launch `fork → spawn`; the spawned child cold-imports `sglang_router.launch_router` → torch+sglang (**measured 236s** on this env), blowing the unchanged **30s** `wait_for_server_ready` gate → `RolloutManager` dies. **Fix (uncommitted working-tree edit):** `miles/ray/rollout/router_manager.py` two `timeout=30` → `600`.
- **End-to-end run never completed** — the cluster had a bad night: pathologically slow shared FS + wedged GPUs + NCCL flakiness + Ray-head startup failures. Five distinct **infra** failure modes, none the sync.
- **Status:** router fix not yet validated end-to-end (runs keep dying at infra stages *before* the router). Metrics regression check still **pending**. Recommend pausing and resuming when the cluster recovers.

---

## Run attempts (chronological)

| # | job | run stamp | nodes | new config | result | cause |
|---|-----|-----------|-------|-----------|--------|-------|
| 1 | 21511 | 093540 | 45,12,22 | qos=viga (queued ~50 min first) | FAIL @ healthcheck tier2 | slinky-45 GPU wedged (`gpu_probe` Killed) |
| 2 | 21519 | 103253 | 12,13,16 | excl 45 | FAIL @ tier2 | slinky-13,16 wedged |
| 3 | 21520 | 103717 | 0,20,23 | excl 45,13,16 | FAIL @ tier2 | slinky-0,23 wedged |
| — | *probe* | — | 9 cand. | batch `gpu_probe` | all gpu=0 | bad GPUs are a fixed subset, not cluster-wide |
| 4 | 21522 | 105304 | 12,24,34 | pin from probe | FAIL @ healthcheck **tier-ib** | slinky-24,34 GPU-ok but **IB rail** bad (`ib_probe` exit 1) |
| — | *probe* | — | 10 cand. | batch **gpu+ib** | good=12,10,19,20,31,35,36; gpu-bad=37,38,39 | need both tiers to pick nodes |
| 5 | **21528** | 111126 | 10,19,20 | gpu+ib good | **healthcheck PASS** (incl tier-nccl) → **FAIL @ sglang router** | `Server …:3976 not ready after 30s` → RolloutManager ActorDiedError. **← the sync regression** |
| — | *investigation* | — | — | timed cold import | `sglang_router.launch_router` = **236.5s** (pulls torch+sglang) | #1367 fork→spawn + 30s timeout |
| — | *fix* | — | — | `router_manager.py` timeout 30→600 | applied (uncommitted) | — |
| — | *(≈7 h gap, resumed evening; FS now very slow)* | | | | | |
| 6 | 21564 | 182549 | 10,19,20 | + router fix | FAIL @ tier2 (**false positive**) | slinky-19,10 "bad" — actually slow `import torch` > 60s |
| — | *diagnostic* | — | slinky-22 | split timing | torch import **84.8s**, set_device **2.8s** | node healthy; 60s `gpu_probe` falsely fails it |
| — | *fix* | — | — | `HEALTHCHECK_TIMEOUT=300` | applied (submit env) | — |
| 7 | 21569 | 183550 | 19,22,50 | + HEALTHCHECK_TIMEOUT=300 | tier2/ib PASS (fix worked) → **FAIL @ tier-nccl** (both attempts) | cross-node all-reduce on triplet 50,19,22 fails (TCPStore recv 0 bytes); IB up but collective bad |
| 8 | 21571 | 184419 | 10,19,20 | earlier-nccl-good triplet | **healthcheck PASS** incl tier-nccl 263 GB/s → **FAIL @ Ray assembly** | `[ray] did not assemble in 300s`; head (19) "node timed out during startup / GCS overloaded", srun exit 1; workers "No node info in GCS" |
| — | *fix* | — | — | `RAY_BRINGUP_TIMEOUT=600` | applied (submit env) | — |
| 9 | 21581 | — | 10,19,20 | — | PENDING → cancelled | slinky-19 became `alloc` (busy) |
| 10 | **21582** | 204242 | 2,10,20 | + RAY_BRINGUP_TIMEOUT=600 | **healthcheck PASS** incl tier-nccl 261 GB/s → **FAIL @ Ray assembly (hung)** | head (2) GCS slow (74s to ready); `ray_client_server` proxy timed out on node-info → exit 1 → `--block` killed head; workers "No node info in GCS". **Same root cause as #8 (slow GCS vs Ray's hardcoded 30s wait); NOT a port conflict** — see F4 |

---

## Findings

### F1 — sglang router regression (THE sync issue; fixed)
- `start_router` (`miles/ray/rollout/router_manager.py`) launches the sglang router as a **spawned** subprocess (PR #1367, to avoid a wandb-fork deadlock), then `wait_for_server_ready(host, port, process, timeout=30)`.
- `run_router` (`http_utils.py`) does `from sglang_router.launch_router import launch_router`, which in **sglang-router 0.3.2** transitively imports **torch + sglang**. Cold import measured **236.5s** on this env. With `spawn`, every launch pays that cost; with `fork` (pre-sync) the child inherited the parent's modules and started instantly.
- `wait_for_server_ready` proved the child was *alive* the full 30s (it raises a distinct "process died" otherwise), so it's slow startup, not a crash. → confirmed timing regression.
- **Fix:** bump both `timeout=30` → `600` in `router_manager.py` (router + session-server). Uncommitted. Proper fix = sglang-sync (newer router imports lean) or upstream raising the timeout for the spawn path.

### F2 — slow shared FS breaks tight launcher timeouts (env, evening of 2026-06-30)
Cold imports were 3–5× normal (torch ~85s, sglang ~236s). That tripped, in sequence:
- **gpu_probe healthcheck** (`HEALTHCHECK_TIMEOUT`, default 60s): `import torch` > 60s → healthy nodes false-failed `gpu=124`. → mitigate with `HEALTHCHECK_TIMEOUT=300`.
- **Ray bring-up** (`RAY_BRINGUP_TIMEOUT`, default 300s): workers slow to register. → raised to 600 (but see F4).

### F3 — cross-node NCCL flakiness (env)
tier-nccl all-reduce failed on triplet (50,19,22) both attempts despite IB rails LinkUp (TCPStore recv 0 bytes / ranks abort). Passed (≈260 GB/s) on triplet (10,19,20) and (2,10,20). ⇒ pairwise/triplet-specific; swap a node.

### F4 — Ray head fails to assemble: slow GCS/raylet registration (env; root-caused 2026-07-01, supersedes the earlier "systemic / port-conflict" guess)
Both evening assembly failures are the **same** root cause: the head's **GCS/raylet was too slow to register the node**, and Ray's wait for that is a **hardcoded 30s** (`raylet_start_wait_time_s`, `ray/_private/node.py:411`, not env-tunable) — so **no launcher knob (incl. `RAY_BRINGUP_TIMEOUT`) can extend it**. Both surfaced before training submit; neither is the sync (morning #5 on the same branch + same triplet 10,19,20 assembled in 37s; evening it took 74s+ and missed the window).
- **#8 (head=19):** the head's own node-info loop expired → `Failed to get node info … Deadline Exceeded` → "node timed out during startup". Workers died first with "No node info in GCS".
- **#10 (head=2):** GCS came up *slowly* (`Failed to connect to GCS within 5s`; "Ray runtime started" only at +74s). The `ray_client_server` proxy timed out waiting for node info and exited 1 *one second after* the success marker → Ray's `--block` monitor killed the head. Confirmed from the surviving `ray_client_server.err` on slinky-2: it **bound 10001 fine** then `RuntimeError: Timed out waiting for node info`. **This instance was not a port conflict** — the bind succeeded; the failure was the node-info timeout (no `Failed to bind` line in 21582's `.err`). Note a 10001 collision *can* produce the identical `ray_client_server [exit code=1]` outer symptom: this env's **grpc 1.80.0 raises** `RuntimeError: Failed to bind to address …:10001` on a failed bind (`grpc/_common.py:179 validate_port_binding_result`) → unhandled → exit 1, independently reproduced. The two are told apart by the `RuntimeError` string in `ray_client_server.err`: `Failed to bind…` = port conflict, `Timed out waiting for node info…` = this case (slow GCS). (Correction: an earlier draft here claimed a 10001 collision "wouldn't exit non-zero" — that was wrong for grpc 1.80, which does raise; the source trace stopped one frame short of the bind-result validation.)
- **Launcher fix shipped** (this branch, `launch_miles.sbatch` + `docs/{launcher,watchdogs}.md`, uncommitted working tree): the bring-up poll now also watches the head/worker `ray start … --block` srun PIDs and **fails fast** on a dead PID (no more idling out the timeout); on failure it **snapshots** the node-local Ray component logs to `ray-debug-attempt<N>/` (gcs_server/raylet/ray_client_server/*_agent — they vanished with the node before, which is why #10 was nearly undiagnosable) and **retries** on a fresh `ray start` up to `RAY_BRINGUP_ATTEMPTS=3`; exhaustion writes `MANIFEST FAILED failure_reason=ray_bringup` (fixes the old "stuck at RUNNING"). The retry — not a longer timeout — is the only available lever against the hardcoded 30s wait; the real cure for a persistently-slow node is to re-submit on healthy nodes.

---

## Node-health map observed (drifts over time — re-probe each session!)
- **GPU-wedged** (`set_device` hangs; admin reboot only fix): slinky-0, 13, 16, 23, 37, 38, 39, 45, 52, 55
- **IB-rail bad** (`ib_probe` fail): slinky-24, 34
- **NCCL-bad as part of a triplet**: 50 (with 19,22)
- **Seen fully-good at some point**: slinky-2, 10, 12, 19, 20, 22, 31, 35, 36 (but 10/19 later wedged, then recovered — drift)
- ⚠️ Don't trust a 60s probe when the FS is slow — it false-fails good nodes. Split torch-import vs `set_device` timing to tell a real wedge from slow import.

---

## Validated vs pending
- ✅ Merge + 3 conflict resolutions compile; launcher healthcheck (#16) works (correctly catches wedged GPUs / bad IB / bad NCCL).
- ✅ Router regression root-caused + fix written.
- ✅ Healthcheck false-positive root-caused + worked around.
- ✅ Ray-assembly failures (#8/#10) root-caused (slow GCS vs Ray's hardcoded 30s wait; port-conflict guess refuted) + launcher robustness fix shipped (fail-fast + retry + log-snapshot + FAILED manifest).
- ❌ **End-to-end run / RL metrics: NOT reached** — every attempt died at an infra stage before training. Router fix + launcher bring-up fix unvalidated end-to-end. No reward/entropy/loss data yet.

## How to resume (when cluster FS is healthy + Ray startable)
1. `git -C ~/workspace/miles-imp checkout sync-upstream-20260630` (router fix is an uncommitted edit to `router_manager.py` — keep it).
2. Find 3 currently-good nodes via the batch **gpu+ib** probe (see `reference_tgt_cluster_node_health` memory); confirm Ray can `ray start --head` on a candidate (the F4 blocker).
3. Submit: `… HEALTHCHECK_TIMEOUT=300 RAY_BRINGUP_TIMEOUT=600 SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true SBATCH_EXTRA="--qos=viga --nodelist=<3 good>" bash scripts/slurm/submit.sh async/geo3k-vlm-multi-turn-fully-async-prefetch2-3node`.
4. Expect bring-up ~10–15 min (slow imports). Milestones: healthcheck OK → tier-nccl PASS → ray assembled → **Router launched at** (the 600s fix's real test) → sglang engines → `rollout 0` → first train step → watch wandb metrics vs a known-good baseline.

## Open threads
- Commit the router fix to the branch (currently uncommitted, per instruction "don't commit yet").
- Commit the launcher Ray-bring-up robustness fix (`launch_miles.sbatch` + `docs/{launcher,watchdogs}.md`, uncommitted) — decide whether it rides the sync branch or a separate launcher-hygiene branch/PR (it's independent of the sync).
- Push `sync-upstream-20260630` + open the PR.
- Do the deferred sglang-sync (proper fix for the router import cost).
- Flag #1367 (spawn + 30s timeout) upstream as a follow-up.
