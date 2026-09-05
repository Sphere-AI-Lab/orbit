# orbit watchdogs & job-killers — what can kill a run, why, and at which fully-async stage

Goal: one place to reason about every mechanism that can **fail the job**, **kill an
engine/actor**, or **discard in-flight work**, so we can tell "infra blip" from "real
bug" and tune thresholds deliberately.

## Fully-async lifecycle stages (referenced below)

```
S0 submit/alloc → S1 node healthcheck → S2 ray assembly → S3 model init (bridge build,
sglang init, first update_weights) → S4 warmup (cuda-graph capture, first rollout) →
S5 STEADY STATE: background worker generates continuously  ‖  trainer trains  →
   S5a per-step weight broadcast (trainer→engines) → loop
(+ outer, always-on: ray-status watchdog, slurm walltime/preempt/OOM)
```

In fully-async, S5 rollout and training **overlap continuously** — the background worker
keeps generating while the trainer consumes finished groups. That overlap is why the
engine health-monitor and the weight-broadcast cascade matter most here.

---

## TL;DR table

| # | Mechanism | Stage | Trigger | Default | Outcome |
|---|-----------|-------|---------|---------|---------|
| 1 | Node healthcheck T1 (nvidia-smi) | S1 | nvidia-smi fails/timeouts on a node | 60s/node | bad node → **requeue** (≤3) |
| 2 | Node healthcheck T2 (gpu_probe) | S1 | `torch.cuda.set_device`+alloc fails | 60s/node | bad node → **requeue** (≤3) |
| 3 | Cleanup gate POISONED | S1 | D-state proc OR GPU mem >500 MB | — | **requeue** (exit 75) |
| 4 | Healthcheck restart cap | S1 | still-bad after 3 requeues | 3 | **FAILED** (no requeue) |
| 5 | Ray bring-up (fail-fast + retry) | S2 | a `ray start` PID dies, or cluster ≠ EXPECTED_GPUS in time | 300s/attempt ×3 | retry in place; exhausted → **FAILED** (`ray_bringup`) |
| 6 | Init asserts | S3 | bad config (see §6) | — | **FAILED** at init |
| 7 | Rollout engine health-monitor | S5/S5a | `health_generate` timeout ×N consecutive | 300s timeout, 3 strikes, 30s interval | **kill engine → cascades to whole job** |
| 8 | Weight-broadcast cascade | S5a | a dead/missing engine in the broadcast | (dist timeout 10min) | **FAILED/hang** |
| 9 | torch.distributed timeout | S3/S5a/S7 | NCCL/Gloo collective hangs | 10 min | **FAILED** |
| 10 | sglang server watchdog | S4/S5 | a forward batch hangs in the engine | 300s | engine self-kills → cascade |
| 11 | ray-status watchdog → CLUSTER_DEAD | outer | `ray job status` unreadable ×24 (~6min) **and no fresh heartbeat** | 15s×24 | **CLUSTER_DEAD** |
| 12 | slurm walltime → DEADLINE | outer | reach `SLURM_JOB_END_TIME−120s` | — | **DEADLINE**, requeue if `--requeue` |
| 13 | slurm preemption (SIGTERM) | outer | scheduler sends `SIGTERM@120` | 120s grace | **INTERRUPTED**, graceful |
| 14 | OOM detection | outer | oom_kill / exit −9 / OUT_OF_MEMORY | — | **FAILED** (rc 137) |
| 15 | max_weight_staleness recycle | S5 | group older than current weight by >N | None (off) | **discard work** (not job) |
| 16 | finished-group buffer capacity | S5 | buffer ≥ `floor(F × rollout_batch_size)` | F = 2.0 | **block the producer** (backpressure) |

Stages: S1 startup, S2 assembly, S3 init, S4 warmup, S5 steady state, S5a weight broadcast,
S7 training compute, "outer" = any time after submit.

---

## A. Outer / always-on watchdogs (fire at any stage)

### 11. ray-status watchdog → CLUSTER_DEAD  (`scripts/slurm/lib/ray_lifecycle.sh`)
- **What**: the controller polls `timeout 10 ray job status` every 15s. `RUNNING`/`PENDING`
  → healthy; `SUCCEEDED/FAILED/STOPPED` → terminal; **unreadable** (timeout / unparseable)
  → strike. After `RAY_STATUS_FAIL_GRACE=24` consecutive unreadable probes (~6 min) it
  declares `CLUSTER_DEAD` (job_rc=3).
- **Why**: detect a wedged/dead cluster the Ray Jobs API can no longer describe. Probes run
  in the controller shell (not recurring `srun`) so a failure means the **Ray control plane
  (GCS/dashboard)** is unreachable, not a slurm step-launch glitch.
- **Stage**: any. In practice it mis-fired in S5 on 2026-06-28 (jobs 21100/21101) — the Ray
  *status API* hung ~6 min while the *training actors were alive* (rollout samples still
  logged). That is the canonical **false-kill**: it kills on a proxy (API readability), not
  on actual training liveness.
- **FIX (committed)**: a **heartbeat sentinel**. The driver writes
  `train_status.json` (`running` at start + every step, terminal at exit) to **node-local**
  disk (`ORBIT_TRAIN_STATUS_FILE`, set by the launcher under `${TMPDIR:-/tmp}`). When the Ray API is unreadable
  but the heartbeat's `updated_at` is < `HEARTBEAT_MAX_AGE_S=600s` old, the watchdog prints
  `ALIVE`, **resets the grace counter, and keeps waiting** instead of killing. A stale
  heartbeat (driver not progressing) still falls through to CLUSTER_DEAD. Before the fix,
  `train_async.py` wrote **no sentinel at all** → async jobs had no fallback liveness signal.
  At terminal exit the launcher copies the sentinel to `$RUN_DIR/train_status.final.json`
  and any nonempty unreadable-probe diagnostics to `$RUN_DIR/probe.log` for postmortem use.
- **Known gap (confirmed on j21354)**: the per-step heartbeat only refreshes once the training
  loop starts; during the long S3/S4 warmup (bridge model-load + cuda-graph capture, ~20–40 min)
  the sentinel goes stale, so the veto is **inert during warmup** — a Ray-API outage *while
  warming up* can still false-kill. j21354 survived its 60-min warmup only because ray-status
  happened to recover intermittently (veto fired 0×). **CLOSED**: a **background heartbeat
  thread** (`start_heartbeat`, every 60s) now keeps `updated_at` fresh through warmup/save/eval;
  the per-step write still bumps `step`, so liveness and progress cross-validate. Trade-off §E.

### 12. slurm walltime → DEADLINE
- Poll loop exits at `SLURM_JOB_END_TIME − 120s` → `DEADLINE` (rc=124). The 120s lets it
  shut down cleanly before slurm SIGKILLs. Requeues if `#SBATCH --requeue` + a checkpoint.

### 13. slurm preemption (SIGTERM)
- `#SBATCH --signal=B:SIGTERM@120` → a trap writes `INTERRUPTED`, tears down, exits 0
  (resumes from checkpoint on requeue). Graceful, not a failure.

### 14. OOM detection
- After exit, the launcher greps `ray_head.log` for `oom_kill|exit code=-9|Out Of Memory`
  and checks `sacct State==OUT_OF_MEMORY`, overriding the state to OOM (rc=137). Detection
  is post-hoc; the actual kill is the Linux OOM-killer / CUDA OOM exception during S5/S7.

### 9. torch.distributed (NCCL/Gloo) collective timeout
- `init_process_group(timeout=distributed-timeout-minutes)`, **default 10 min**. Any
  collective that hangs longer than this (all-reduce in S7, **broadcast in S5a
  update_weights**) raises and crashes the rank → job FAILED. This is the hard backstop
  behind the weight-broadcast cascade (#8).

---

## B. Stage-ordered gates

### S1 — node healthcheck (before any orbit code runs)  (`launch_orbit.sbatch`, `lib/gpu_probe.py`)
- **T1 nvidia-smi** (`--query-gpu=count`) and **T2 gpu_probe** (`torch.cuda.set_device(i)` +
  tiny alloc per GPU), 60s/node. A failing node is recorded and the job **requeues excluding
  it** (`scontrol update ExcNodeList`, with an SBATCH_EXTRA `--exclude` HINT printed),
  capped at 3 restarts → else FAILED ("healthcheck_exhausted").
- **Cleanup gate POISONED**: after killing leftover procs, if a **D-state process persists**
  (5 re-samples) or **GPU mem used > 500 MB**, the node is POISONED → requeue (exit 75).
- **Tier weka** (`lib/weka_probe.sh`): a bounded ~64 MiB **O_DIRECT** read from the shared FS
  (`/data`, WekaFS), per-node, same requeue-excluding path as T1/T2. Catches a **wedged Weka
  client** — reads hang in D-state and otherwise stall engine/weight bring-up with idle GPUs
  and no error (2026-07-01, j21623). Complements the cleanup gate: that flags only *leftover*
  D-state procs, this actively reads to catch a wedged mount on an otherwise-clean node. Opt
  out `ORBIT_HEALTHCHECK_WEKA=0`.
- **Why**: never start training on a node with dead/orphaned GPU procs (leaked vLLM/sglang
  workers) — they'd OOM or hang the run. **This is what killed 21351/21352/21353** today:
  gpu_probe BAD on slinky-11/47/29/34/51 (leftover GPU procs). Note: the in-job
  `scontrol ExcNodeList` update can fail ("requeue may land on same bad nodes") → then you
  must resubmit with `SBATCH_EXTRA=--exclude=...` yourself. **Not a training/code problem.**

### S2 — ray cluster assembly  (`launch_orbit.sbatch` ~"did not assemble")
- The poll watches both `ray status` (must report `EXPECTED_GPUS` = workers×8 +
  `ORBIT_RAY_HEAD_NUM_GPUS` (default 8; head-sidecar recipes export 0) within
  `RAY_BRINGUP_TIMEOUT=300s`) **and** the head/worker `ray start … --block` srun PIDs.
  A dead PID = that node's Ray exited (cluster collapsed) → the poll stops at once
  instead of idling out the timeout. **Why**: a worker that never joins (network / a
  slow node) would otherwise hang init forever. Historical slinky-24/34 "ray wedge"
  lived here.
- On failure the launcher snapshots the node-local Ray component logs to
  `ray-debug-attempt<N>/` and retries on a fresh `ray start`, up to
  `RAY_BRINGUP_ATTEMPTS=3`. The dominant cause is the head's GCS/raylet being slow to
  register the node, which trips **Ray's hardcoded 30s `raylet_start_wait_time_s`**
  (`ray/_private/node.py`) — no launcher env var extends it, so the in-place retry (not
  a longer timeout) is the lever. Exhausting all attempts is terminal:
  `MANIFEST state=FAILED failure_reason=ray_bringup`, exit 1 (no requeue — there is no
  bad-node exclusion for bring-up, so a reroll would just hit the same slow nodes).

### S3 — model init (placement, bridge model build, sglang init, first update_weights)
- **§6 fatal asserts** (AssertionError → FAILED at init):
  - `train_async.py:14` — `assert not args.colocate` ("Colocation is not supported for async
    training"). Async needs disagg.
  - **Qwen3-VL + CP>1 requires `calculate_per_token_loss`** — asserted in the bridge submodule
    `thirdparty/Megatron-Bridge/.../qwen_vl/modelling_qwen3_vl/model.py:203`. **This killed
    cp2 (21102/21105).** Subtlety: the orbit arg parses fine, but the bridge builds its
    `TransformerConfig` from the HF `config.json` and drops training-time args → fix is to
    set `provider.calculate_per_token_loss` after `to_megatron_provider()` (see
    `scripts/experiments/async/experimental/cp2-calculate-per-token-loss.md`).
  - bridge weight-name mismatches (the old disagg VLM failure) also surface here.
- **torch.distributed timeout** (#9) and the **ray-status watchdog** (#11) are already armed.

### S4 — warmup (sglang cuda-graph capture across the bs list, first rollout)
- Slow but normal (the 4-node bridge model-load can be ~20–30 min and silent). The
  ray-status watchdog's heartbeat now keeps this from being mistaken for a hang. **sglang's
  own watchdog** (#10, `watchdog_timeout≈300s`) will self-kill an engine whose forward batch
  hangs; that engine death then cascades (#8).

### S5 — steady state (continuous rollout ‖ training)
- **7. Rollout engine health-monitor** (`orbit/utils/health_monitor.py`): a background thread
  calls `engine.health_generate(timeout=300s)` every 30s. `< max_consecutive_failures(3)`
  → log + retry; `== 3` consecutive → **`stop_engines()` (ray.kill the actor)**.
  - **Why it cascades**: once an engine actor is killed, the next per-step **weight broadcast
    (#8, S5a)** has a missing peer → the collective hangs until the dist timeout (#9) →
    whole job dies. So an engine kill ≈ a job kill.
  - **FIX (committed a679d8d)**: timeout **30s→300s**, max-consecutive **1→3**. Rationale: a
    busy engine serving a big rollout, or one **paused mid weight-update**, legitimately
    exceeds 30s answering a probe; a 1-strike kill false-killed healthy engines and cascaded
    (runs 20862 async, 21091/21092 colocate). The monitor `pause()`s during offload/weight
    windows and re-arms with a `first_wait` grace after `resume()`, and clears the failure
    counter across a pause so a pause doesn't count as a strike.
- **15. max_weight_staleness recycle** (`fully_async_rollout.py`): a completed group whose
  oldest rollout weight version trails the current engine version by > N is **reset and
  returned to the data buffer** (re-sampled later), not trained on. **Discards work, never
  kills the job.** Default `None` (off); our recipes set 2.
- **16. finished-group buffer capacity** (`--async-data-buffer-capacity-factor`, default 2.0):
  bounds the finished-group buffer at `floor(factor * rollout_batch_size)` groups; when it is
  full the producer **blocks on put** until training consumes. Backpressure / memory guard,
  not a kill. (`--fully-async-max-completed-queue-groups` used to do this by pausing launches;
  the 2026-08-18 class-based rewrite made it inert and argument validation now warns on it.)

### S5a — per-step weight broadcast (trainer → engines)
- **8. The cascade hub.** `update_weights` broadcasts new weights to all engines via a
  collective. If any engine is dead/missing (killed by #7 or #10, or a node dropped), the
  broadcast hangs → dist timeout (#9, 10 min) → FAILED. Most "whole-job deaths" route
  through here. This is also the window where engines look briefly unresponsive to the
  health-monitor (hence the 300s/3 fix).

### S7 — training compute
- Standard megatron fwd/bwd; failures are CUDA OOM (→ #14) or NCCL collective hangs (→ #9).

---

## C. Detection-only (does NOT kill)

- `scripts/slurm/check_run.sh` greps the last 200 log lines for
  `crash-debug|Traceback|FATAL|OOM|...|terminal state: (FAILED|STOPPED|CLUSTER_DEAD|DEADLINE|OOM)`
  to surface trouble to the operator/monitor loop. It only reports; it never kills.

---

## D. How to read a death (decision guide)

1. `terminal state: CLUSTER_DEAD` + engines were 200 OK / rollout samples still logging
   near the end → **ray-status false-kill** (#11). Infra (Ray control plane), not your code.
   The heartbeat fix should now prevent it.
2. `[healthcheck] BAD ... gpu_probe` / `POISONED` at the very start, step=none → **bad node**
   (#1–3). Resubmit with `--exclude`. Not your code.
3. `Health check failed ... threshold reached. Killing actor` then a hang/CLUSTER_DEAD →
   **engine health-monitor kill → broadcast cascade** (#7→#8). With 300s/3 this should only
   happen for a genuinely dead engine.
4. `AssertionError` at init, step=none → **config assert** (#6).
5. Death exactly at `SLURM_JOB_END_TIME` → **walltime** (#12); at a `SIGTERM` → preemption (#13).
6. `OUT_OF_MEMORY` / exit −9 → **OOM** (#14).

## E. The two thresholds we changed (and why)
- **Engine health-monitor** (#7): `30s→300s` timeout, `1→3` strikes (commit a679d8d). Stops
  false-killing busy/paused-but-alive engines in S5/S5a.
- **ray-status watchdog** (#11): added the **heartbeat-veto** so a transient Ray-control-plane
  outage in S5 no longer kills a live job (committed; validated reaching step on j21354).
  Covers steady-state S5; the S3/S4 **warmup window is still open** (§A.11 "Known gap") until a
  background heartbeat thread lands. Both fixes target the same failure shape: *killing healthy
  training because a secondary signal went dark.*

### Implemented: background heartbeat thread + step cross-validation (closes the warmup gap)
`start_heartbeat()` runs a driver daemon writing the sentinel every 60s, keeping `updated_at`
fresh through warmup/save/eval — not just the training loop. The per-step write additionally
bumps `step`, so ONE sentinel carries BOTH signals and they cross-validate: `updated_at`
freshness = *process alive* (drives the veto); `step` = *progress*. The watchdog logs
`ALIVE step=N` on each veto, so "fresh heartbeat but frozen step" (= wedged) is visible across
vetoes. Trade-off: a heartbeat proves *liveness*, not *progress*, so a genuinely-hung
warmup/step now survives to walltime instead of CLUSTER_DEAD at ~6 min (still caught by
walltime / dist-timeout / OOM); the `step` field is what surfaces that case for a future
stall-detector.
