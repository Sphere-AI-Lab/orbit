# Launcher design notes

Why the slurm launcher is shaped the way it is. The main script is
`scripts/slurm/launch_miles.sbatch`; mechanical bits live in
`scripts/slurm/lib/`. If you just want to launch a run, read
`.claude/skills/slurm-launch/SKILL.md` — this file is for "why does
the script do X" questions.

## Running the v0.5.12 sync: torch-2.9.1 env + `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK`

The 2026-06 upstream sync through `radixark/miles@1e1679706` (commit date
2026-06-01) bumped the sglang **source** to v0.5.12 (submodule
`thirdparty/sglang`), but we keep running it on the **existing torch-2.9.1
`miles` conda env** — we do *not* rebuild to torch 2.11. Upstream v0.5.12 pairs
with torch 2.11, whose `sgl-kernel` / `deep_gemm` are published **cu13-only**;
this host is CUDA 12.8, so those kernels can't load. The torch-2.9.1 env (cu12
`sgl-kernel` 0.4.1) runs the v0.5.12 source fine — validated end-to-end on geo3k
and VAGEN (FrozenLake / Sokoban), smoke through multi-step training (Sokoban eval
0.48 → 0.70 over 20 rollouts).

**Required knob — pass `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true` when you
submit.** v0.5.12's engine startup asserts `sglang-kernel >= 0.4.2.post2` (and,
for the flashinfer attention backend, `flashinfer >= 0.6.11.post1`) in
`sglang/srt/entrypoints/engine.py:_set_envs_and_config`. The env has 0.4.1 /
0.6.7.post2 — below those floors but functionally fine: the assert is a *version
guard*, not a real ABI gate (cuda-graph capture, rollout, and training all pass).
Without the skip, every `SGLangEngine` subprocess dies at launch with
`sglang-kernel is installed with version 0.4.1, which is less than the minimum
required version 0.4.2.post2`.

`--export=ALL` carries the var `submit.sh` → `sbatch` → ray-start `srun` →
`ray start` → the engine subprocess, so setting it on the submit line is enough:

```bash
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true \
  bash scripts/slurm/submit.sh <experiment-name>
```

Caveats:
- **`verify_env.py --imports-only` does not catch this** — the assert fires at
  engine *launch*, not import, so it passes (24/24) while the engine would still
  die. The runtime smoke is the real gate.
- **Do not** `pip install -U sglang-kernel` / `deep_gemm` / `flashinfer` to
  satisfy the assert — that pulls the cu13 / torch-2.11 build, which won't load
  on this host.
- **The pin-model intentionally rejects this combo — don't regenerate or
  fresh-install.** `setup/{pins.env,extract_pins.py,install_env.sh}` are left
  exactly as `main` (the #12 pin bundle), so `pins.env` still pins the working
  torch-2.9.1 / `cu129-x86_64` wheels — which is what the `miles` env actually
  runs. But the submodule is now v0.5.12 (pyproject `torch==2.11`), *ahead* of
  that wheel bundle, so `extract_pins.py --check` and `install_env.sh` **fail
  closed with an ABI mismatch** (`wheels torch 2.9.1 != submodule torch 2.11`).
  That is expected: the source is deliberately ahead of the wheels. **Do not**
  `extract_pins.py --write` and **do not** fresh-install for this sync — use the
  prebuilt `miles` env. (These guards are dev tooling, not CI-gated, so they do
  not fail the PR.) A clean-room installer for "torch-2.9.1 binaries + v0.5.12
  source" needs the parked torch-constraint work
  (`docs/debug-notes/miles-sync-2026-06-02/setup-scripts-vs-main.patch`) and is
  follow-up.

## Step memory cap (`--mem=0` on every srun)

`#SBATCH --mem=0` only sizes the job allocation. `srun --overlap` child
steps do **not** inherit it and fall back to slurm's `DefMemPerCPU ×
--cpus-per-task` default (≈ 64 GiB on this cluster).

The ray-head step in 1-node mode hosts the Megatron train actor plus
multiple SGLang engines on the same cgroup, so this silent 64 GiB cap
caused our first OOM (j11332, MaxRSS ≈ 67 GiB on slinky-54). Adding
`--mem=0` to every `srun --overlap` site makes each step inherit the
job's full node memory. Drop this and runs will OOM unpredictably.

Verify with `sacct -j <jobid> --format=AllocTRES%50` — every step should
show `mem=<full node mem>`, not `mem=64G`.

## Healthcheck

Four **per-node** tiers (run from the head node via `srun --overlap
--mem=0`, one srun per node) plus one **cross-node** tier, all before Ray
bring-up so a bad node/fabric never wastes a ~12 min model load.

1. **Tier 1 — `nvidia-smi --query-gpu=count`** (~2 s). Catches nodes
   where the driver is missing or `nvidia-smi` itself hangs.
2. **Tier 2 — `python lib/gpu_probe.py`** (~10 s). Imports torch and
   runs `torch.cuda.set_device(i)` + a tiny allocation on every visible
   GPU. Catches the failure mode where `nvidia-smi` happily reports 8
   GPUs but a specific GPU is stuck at the driver level — symptom:
   `cudaErrorDevicesUnavailable` (slinky-35 GPU 0+1 on 2026-05-20).
   Per-node output goes to `<run-dir>/gpu_probe.<node>.log` when the
   probe fails (cleaned up on success).
3. **Tier ib — `bash lib/ib_probe.sh`** (~1 s, pure `ibstat`, no torch).
   Every InfiniBand-layer rail must be `Physical state: LinkUp`. Catches
   a rail stuck in `Polling` (e.g. `mlx5_8` at Rate 10 instead of 400G
   LinkUp on slinky-20/23, 2026-05-26): NCCL auto-enumerates all 8 IB
   rails, so one Polling rail breaks the whole multi-node ring on the
   *first* collective with `NET/IB ... error 12, vendor err 129 (Recv)`
   → 120 s watchdog → SIGABRT. RoCE/Ethernet rails (`mlx5_0/1`, 100 G)
   are ignored — NCCL doesn't auto-select them. Skips (non-blocking) if
   `ibstat` is missing or reports no IB rails. Opt out with
   `MILES_HEALTHCHECK_IB=0`.
4. **Tier weka — `bash lib/weka_probe.sh`** (~2–10 s, pure bash + `dd`, no
   torch). Reads ~64 MiB **O_DIRECT** from a large file on the shared FS (the
   conda env under `/data`, WekaFS). Catches a **wedged WekaFS client**: a
   healthy read returns in seconds, but a wedge makes `/data` reads hang in
   uninterruptible D-state, which otherwise stalls sglang engine/weight bring-up
   indefinitely with idle GPUs and no error (see
   `docs/debug-notes/miles-sync-2026-06-30/wekafs-wedge-2026-07-01.md`). O_DIRECT
   bypasses the page cache so a probe file left warm by a prior job can't mask
   the wedge; it targets a *wedge* (reads that never return), not slowness
   (64 MiB completes well within `HEALTHCHECK_TIMEOUT`). Skips (non-blocking) if
   no sizable file is found. Opt out with `MILES_HEALTHCHECK_WEKA=0`.

Tiers 1–4 are per-node and **localizable** — a failure appends the node to
`BAD_NODES` and rides the shared requeue machinery:

- `<run-dir>/.bad_nodes` accumulates the bad-node list across requeues
  of this job (dedup'd before use).
- `scontrol update JobId=$JOBID ExcNodeList=<cumulative-bad>` extends
  the job's excluded-node constraint, so the next slurm allocation will
  skip the bad nodes.
- `exit 75` (`EX_TEMPFAIL`) → slurm `--requeue` rerolls the allocation.
- `<run-dir>/.health_restarts` counts attempts. After
  `HEALTHCHECK_MAX_RESTARTS` (= 3), the launcher hard-fails (exit 1,
  `MANIFEST state=FAILED` with `failure_reason=healthcheck_exhausted`)
  instead of looping forever.

If `scontrol update` fails (older slurm, permissions), the launcher
still exits 75 with a manual `[healthcheck] HINT: SBATCH_EXTRA='--exclude=...'`
line — the requeue may land on the same bad nodes, so the cap protects
against an infinite loop.

5. **Tier nccl — `lib/nccl_probe.py`** (cross-node, ~1 min; runs after
   `NODE_PREAMBLE`, before HF convert / Ray bring-up; skipped when fewer
   than 2 Ray/training nodes). Forms a real process group across the
   Ray/training nodes (`RAY_NODES` — i.e. excluding any envpack-server
   nodes, which run an HTTP env server over TCP and never join training
   NCCL) and runs a `torch.distributed` all-reduce size sweep — the exact
   transport training uses. Catches what per-node `ibstat` can't: a rail
   that is LinkUp but errors under load, or a topology NCCL can't ring.
   `NCCL_HEALTHCHECK_MIN_BUSBW_GB` (default 50) floors bus bandwidth — a
   preflight pass/fail gate, not a throttle — so it also catches a *silent*
   IB→socket fallback that completes the all-reduce at ~socket speed, which a
   completion-only check misses. The 50 GB/s floor is ~8× below healthy IB
   (~470 GB/s here) so it does not false-fail a good run; set 0 to disable.
   It runs with the `NODE_PREAMBLE` (conda + the `ulimit -Sl`
   memlock fix), so it mirrors training; `MASTER_ADDR` is injected as the
   first Ray node (the Ray head) so rendezvous never targets an excluded
   or envpack-only node. Being an
   **aggregate** check it can't attribute a failure to one node, so —
   unlike tiers 1–4 — it does **not** requeue: after 2 in-place attempts
   it writes `MANIFEST state=FAILED failure_reason=nccl_probe` and exits
   1, leaving a clean repro at `<run-dir>/nccl_probe.log` (`NCCL_DEBUG=INFO`)
   for infra. Rationale: the deterministic causes (Polling rail, memlock)
   are already auto-healed by tier-ib + the preamble, so a tier-nccl
   failure is a genuine fabric problem worth human eyes, not blind
   requeue-thrash. Opt out / tune with `MILES_HEALTHCHECK_NCCL=0`,
   `NCCL_HEALTHCHECK_MAX_BYTES`, `NCCL_HEALTHCHECK_WALL`,
   `NCCL_HEALTHCHECK_MIN_BUSBW_GB` (default 50; 0 disables the bandwidth floor).

Ad-hoc fabric check outside a run (e.g. to re-test a repaired node) — grab the
nodes and run the same `lib/nccl_probe.py` with the miles env active:

    salloc -N2 --gres=gpu:8 --exclusive --nodelist=slinky-20,slinky-23
    srun --ntasks-per-node=8 --gpus-per-task=1 --cpus-per-task=8 bash -c '
      source /data/shared/conda/miniconda3/etc/profile.d/conda.sh && conda activate miles
      ulimit -Sl "$(ulimit -Hl)"          # RDMA memlock, as in NODE_PREAMBLE
      export LD_LIBRARY_PATH="$(python -c "import site;print(site.getsitepackages()[0])")/nvidia/cudnn/lib:$LD_LIBRARY_PATH"
      python scripts/slurm/lib/nccl_probe.py'

`NCCL_IB_HCA=mlx5_2 srun …` pins one rail; `NCCL_HEALTHCHECK_MAX_BYTES=8589934592`
runs the full 8 GiB sweep.

All `MILES_HEALTHCHECK_*` / `NCCL_HEALTHCHECK_*` toggles are read at launcher
start, *before* the recipe is sourced — pass them as submit-time env vars
(`MILES_HEALTHCHECK_NCCL=0 bash scripts/slurm/submit.sh <exp>`), not as recipe
knobs, or they will not take effect.

Implementation: `launch_miles.sbatch` healthcheck sections + `lib/gpu_probe.py`,
`lib/ib_probe.sh`, `lib/nccl_probe.py`.

## Two-phase cleanup + D-state gate

`[cleanup]` step: SIGTERM → 5 s grace → SIGKILL → check for `D`-state
processes and leftover GPU memory.

Single-phase SIGKILL is unsafe because SGLang can be mid-CUDA-syscall;
killing it cold leaves D-state survivors that eat GPU memory and break
the next launch. The grace lets it finish in-flight kernels. If the
post-cleanup gate sees `D`-state processes (other than ours) or > 500 MiB
leftover GPU memory, the node is poisoned — we exit 75 so slurm
`--requeue` picks a different node instead of retrying onto the bad one.

Borrowed from M3TRL `cli/cluster/cleanup.py`.

## Ray submission: `--no-wait` + status poll

`ray job submit` (blocking mode) follows the job's stdout via SSE. When
raylet dies mid-run — typically an OOM-kill cascade — the SSE stream
drops, the CLI prints `Status: RUNNING / Job is currently running.`,
then exits 0. The exit code reflects the log-tail's fate, not the
underlying job's terminal state. That mismatch made one OOM look like
a clean train-and-exit.

The launcher now:
1. `ray job submit --no-wait` (via `srun --overlap` once, to capture
   submission id) to get a submission id immediately.
2. Background `ray job logs --follow` runs locally in the controller
   shell against `$RAY_ADDRESS` and streams stdout into `run.log`. If the
   log stream drops unexpectedly mid-run, the wrapper reconnects after
   2 s and keeps tailing.
3. Foreground `ray job status` also runs locally, every
   `RAY_STATUS_POLL_INTERVAL=15s`, with each probe wrapped in
   `timeout $RAY_STATUS_PROBE_TIMEOUT=10s`. Terminal states: `SUCCEEDED`
   (rc 0), `FAILED` (rc 1), `STOPPED` (rc 2). A readable `RUNNING`/`PENDING`
   reply resets the grace counter. If the dashboard returns **unreadable**
   output for `RAY_STATUS_FAIL_GRACE=24` consecutive probes (= 6 min by
   default), we consult the node-local `$MILES_TRAIN_STATUS_FILE` (set to
   `${TMPDIR:-/tmp}/miles-$JOBID.train_status.json`), which `train.py` writes
   from inside the Ray job: a fresh `ALIVE` heartbeat means the
   actors are healthy and the dashboard is merely wedged, so the grace
   counter resets and polling continues; a completed sentinel resolves the
   run as `SUCCEEDED`, a failed one as `FAILED`; no fresh heartbeat at all
   falls through to `CLUSTER_DEAD` (rc 3). Terminal log-tail markers are
   treated as hints and reconfirmed with an authoritative `ray job status`
   before the loop trusts them. If the SLURM wall deadline is reached
   without a terminal state, `DEADLINE` (rc 124). Per-probe diagnostics are
   written to node-local scratch under `${TMPDIR:-/tmp}`; `run.log` prints a
   one-line warning and the local diagnostics path. On terminal exit, a
   nonempty probe log is copied to `$RUN_DIR/probe.log`, and the final driver
   sentinel is copied to `$RUN_DIR/train_status.final.json`. This avoids
   synchronously opening files in `$RUN_DIR` on every poll while still
   giving the launcher an independent completion signal when the Ray Jobs
   API becomes unreadable.
4. The bg log tail watches a node-local marker the fg poll touches when
   finished, so teardown is fast and the active `ray job logs` child is
   killed with the wrapper.

Why the bg/fg probes use the local Ray CLI instead of recurring `srun
--overlap`: the old shape created a fresh Slurm step every 15 s, so
`CLUSTER_DEAD` conflated Ray Jobs API/dashboard unavailability with Slurm
step launch or transport failures. Local CLI polling narrows that signal.
It does not make the run resilient to real node or storage loss; e.g. a
Weka mount outage on a trainer or sampler can still fail the Ray job and
needs requeue/resume or longer-term Ray-level fault tolerance.

Borrowed from M3TRL `cli/dispatch/submit_driver.py:_submit_ray_job`.

## OOM crash debug

After the poll loop returns a state, two sources are checked:

1. `$RUN_DIR/ray_head.log` — slurmstepd / ray-cli stamp
   `oom_kill event`, `exit code=-9`, or `Out Of Memory` immediately.
2. `sacct -j $JOBID` — `State=OUT_OF_MEMORY` for any step shows up
   within a few seconds of the kill (we `sleep 3` to cover lag).

Either match forces `STATE=OOM` and `JOB_RC=137`. The override exists
so a misclassified `SUCCEEDED` (which still happens occasionally when
ray manages to mark the job before tearing down) can't hide a memory
leak in a recipe.

Implementation: `lib/ray_lifecycle.sh:crash_debug_check`. This is the
minimum floor — a richer RL debug skill is planned that will layer
NaN-loss / grad-explosion / sglang-weight-transfer signals on top.

## Ray bring-up

After the healthcheck passes, the launcher boots Ray — `ray start --head … --block`
on the head plus `ray start --address … --block` on each worker (backgrounded
`srun`s) — then polls until the cluster is ready. The poll watches **two** signals:

- **`ray status`** — the GPUs registered with the head's GCS reach `EXPECTED_GPUS`
  (= workers × 8 + `MILES_RAY_HEAD_NUM_GPUS`, default 8 — recipes hosting a
  whole-node GPU sidecar on the head, e.g. an OPD teacher, export 0).
- **the head/worker srun PIDs** — each `ray start … --block` blocks forever on
  success, so a dead PID means that node's Ray exited and the cluster has collapsed.
  The poll detects this and stops *immediately* rather than idling out the whole
  `RAY_BRINGUP_TIMEOUT` (default 300s/attempt).

The dominant bootstrap failure is the head's GCS/raylet being slow to register the
node — `Failed to get node info … Deadline Exceeded` / "node timed out during
startup", or a `ray_client_server [exit code=1]` whose own node-info wait expired,
which then trips Ray's `--block` monitor into killing the head. The window for this
is **Ray's hardcoded 30s `raylet_start_wait_time_s`** (`ray/_private/node.py`), which
no launcher env var can extend — so raising `RAY_BRINGUP_TIMEOUT` does **not** help
it. These failures are usually transient (the same nodes assemble fine moments
later), so on a failed attempt the launcher:

1. snapshots the bootstrap logs to `ray-debug-attempt<N>/` — the per-node srun logs
   plus Ray's node-local component logs (`gcs_server.*`, `raylet.*`,
   `ray_client_server.*`, `*_agent.log`) pulled from `/tmp/ray/session_latest/logs`,
   which otherwise vanish with the node and are the *only* record of the real cause;
2. `ray stop --force` on every node and retries on a fresh `ray start`.

Up to `RAY_BRINGUP_ATTEMPTS` (default 3). Exhausting all attempts is terminal —
`MANIFEST state=FAILED failure_reason=ray_bringup` (with `attempts`), exit 1. It does
**not** requeue: there is no bad-node exclusion for bring-up, so a reroll would just
land on the same slow nodes (and, uncapped, loop to walltime). Re-submit with
known-good nodes instead.

## `MANIFEST.json` schema

One per run, at `$RUN_DIR/MANIFEST.json`. Fields:

| Field | Type | Source |
|---|---|---|
| `state` | str | `RUNNING` → terminal (`SUCCEEDED` / `FAILED` / `STOPPED` / `CLUSTER_DEAD` / `DEADLINE` / `OOM` / `INTERRUPTED`) |
| `started_at` / `updated_at` | ISO-8601 str | First write / latest write |
| `ended_at` | ISO-8601 str | Set only at terminal write |
| `job_id` | str | `SLURM_JOB_ID` |
| `head_node` | str | First good node from healthcheck |
| `run_dir` | str | The run dir (absolute) |
| `recipe` | str | Path to the experiment recipe |
| `restarts` | int | `SLURM_RESTART_COUNT` |
| `job_rc` | int | Final exit code (terminal write only) |
| `failure_reason` | str | Set on launcher-side fails: `healthcheck_exhausted`, `head_ip_unresolved`, `nccl_probe` (with `bad_nodes` / `probe_nodes`), `ray_bringup` (with `attempts`) |

`submit.sh` reads the most recent 3 manifests for the same job name and
warns about any non-`SUCCEEDED` state. That's a heads-up to the operator,
never a block — bad configs deserve retries too.

Implementation: `lib/manifest.sh:write_manifest`, `read_recent_manifests`.

## Exit code conventions

| RC | Meaning | Caused by |
|---|---|---|
| 0 | `SUCCEEDED` | Normal training completion |
| 1 | `FAILED` | Ray reported job failure |
| 2 | `STOPPED` | Ray reported job stopped (admin-issued) |
| 3 | `CLUSTER_DEAD` | Ray dashboard unresponsive past grace window |
| 75 | `EX_TEMPFAIL` | Healthcheck / cleanup tripped — slurm `--requeue` retries (healthcheck path also adds bad nodes to `ExcNodeList`, capped at `HEALTHCHECK_MAX_RESTARTS=3`) |
| 78 | `EX_CONFIG` | Recipe missing required fields / submission id unparseable |
| 124 | `DEADLINE` | SLURM wall hit before terminal state |
| 137 | `OOM` | Postmortem upgrade (overrides anything above) |

Exit 75 is special — paired with `#SBATCH --requeue`, slurm requeues
the job onto a different node automatically. Use it for transient
node-level failures, never for code bugs.

## Run-dir file layout

```
runs/<job-name>/<YYMMDD_HHMMSS>/
├── run.log             # slurm --output, contains everything from launch_miles
├── args.json           # MILES_ARGS array parsed into JSON for diffing across runs
├── ray_head.log        # stdout/stderr of `ray start --head` (last attempt)
├── ray_worker_<N>.log  # stdout/stderr of each `ray start --address` worker
├── ray-debug-attempt<N>/  # only on a failed bring-up attempt: that attempt's per-node
│                          #   srun logs + Ray component logs (gcs_server/raylet/
│                          #   ray_client_server/*_agent) pulled from the node's /tmp/ray
└── MANIFEST.json       # state/timing/job_id audit record (see above)
```

The stamp is `YYMMDD_HHMMSS` (not `MMDD-HHMMSS`) so the year is visible
and the underscore disambiguates from the legacy format. Created by
`submit.sh` before sbatch, so slurm can `--output` directly into it
(no symlinks or tee).

## Signals + traps

- `SIGTERM` / `SIGINT` → `write_manifest INTERRUPTED` → `teardown` →
  `exit 0`. Slurm sends `SIGTERM` ahead of the wall clock via
  `#SBATCH --signal=B:SIGTERM@120` (= 120 s warning).
- `EXIT` (any path) → `teardown` runs `ray stop --force` on every
  allocated node and `kill`s the ray bring-up srun PIDs.

The teardown does *not* update MANIFEST itself — `state=OOM/SUCCEEDED/…`
is the more specific value, written by the explicit `write_manifest`
call just before `exit`.

## What lives where

| File | Owns |
|---|---|
| [`launch_miles.sbatch`](../launch_miles.sbatch) | High-level orchestration: SBATCH headers, env defaults, run dir, healthcheck, cleanup, recipe sourcing, ray bring-up, teardown trap, calls into lib/. |
| [`lib/manifest.sh`](../lib/manifest.sh) | `write_manifest` (used by launcher) + `read_recent_manifests` (used by submit.sh). |
| [`lib/gpu_probe.py`](../lib/gpu_probe.py) | Tier-2 healthcheck — `torch.cuda.set_device(i)` probe per GPU. |
| [`lib/ib_probe.sh`](../lib/ib_probe.sh) | Tier-ib healthcheck — every InfiniBand rail must be LinkUp (catches a Polling rail). |
| [`lib/nccl_probe.py`](../lib/nccl_probe.py) | Tier-nccl healthcheck — cross-node all-reduce smoke test over the fabric. |
| [`lib/ray_lifecycle.sh`](../lib/ray_lifecycle.sh) | `ray_submit_and_wait` + `crash_debug_check`. |
| [`submit.sh`](../submit.sh) | Login-node wrapper: argv parsing, asset download (`hf download`), per-run dir creation, prior-run warning, `sbatch` call. |
| [`check_run.sh`](../check_run.sh) | Snapshot script — concise health report (MANIFEST + sacct + last rollout/train/eval). Called by [`rl-monitor-loop` SKILL.md](../../../.claude/skills/rl-monitor-loop/SKILL.md). |
| [`setup/install_env.sh`](../setup/install_env.sh) | One-time conda env build. Source of truth for what's installed. |

If you're adding behavior, prefer extending an existing file over adding
a new module — we keep the launcher boundary thin on purpose.

## See also

- [`architecture.md`](architecture.md) — 30-second submit → sbatch → ray flow
  overview (the "what" if this file is the "why").
- [`../setup/README.md`](../setup/README.md) — one-time install of the `miles`
  conda env, plus the env-var knob table.
- [`slurm-launch` SKILL.md](../../../.claude/skills/slurm-launch/SKILL.md) — how
  to actually launch a run (filesystem layout, recipe pattern, dispatch flow).
- [`rl-monitor-loop` SKILL.md](../../../.claude/skills/rl-monitor-loop/SKILL.md)
  — adaptive-cadence Claude-driven monitor that wraps `check_run.sh`.
- [`sync/`](sync/) — upstream-sync trail (`prs.md`, `pr-body.md`,
  `divergence.patch` / `.stat`) for the v0.5.10 → v0.5.12 sglang sync.
- Upstream miles docs: [`docs/getting-started/installation.md`](../../../docs/getting-started/installation.md)
  + [`docker/Dockerfile`](../../../docker/Dockerfile) — the canonical install
  reference that `setup/install_env.sh` mirrors.
