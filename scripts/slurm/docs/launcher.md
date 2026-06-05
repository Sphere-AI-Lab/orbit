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

`[healthcheck]` step: two tiers per allocated node, run from the head
node via `srun --overlap --mem=0`.

1. **Tier 1 — `nvidia-smi --query-gpu=count`** (~2 s). Catches nodes
   where the driver is missing or `nvidia-smi` itself hangs.
2. **Tier 2 — `python lib/gpu_probe.py`** (~10 s). Imports torch and
   runs `torch.cuda.set_device(i)` + a tiny allocation on every visible
   GPU. Catches the failure mode where `nvidia-smi` happily reports 8
   GPUs but a specific GPU is stuck at the driver level — symptom:
   `cudaErrorDevicesUnavailable` (slinky-35 GPU 0+1 on 2026-05-20).
   Per-node output goes to `<run-dir>/gpu_probe.<node>.log` when the
   probe fails (cleaned up on success).

On bad nodes:

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

Implementation: `launch_miles.sbatch` healthcheck section + `lib/gpu_probe.py`.

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
   (rc 0), `FAILED` (rc 1), `STOPPED` (rc 2). If the dashboard returns
   unreadable output for `RAY_STATUS_FAIL_GRACE=24` consecutive probes
   (= 6 min by default), we declare `CLUSTER_DEAD` (rc 3). If the SLURM
   wall deadline is reached without a terminal state, `DEADLINE` (rc
   124). Per-probe diagnostics are written to node-local scratch under
   `${TMPDIR:-/tmp}`; `run.log` prints a one-line warning and the local
   diagnostics path. This avoids synchronously opening files in
   `$RUN_DIR` while deciding whether the Ray Jobs API is still readable.
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
├── ray_head.log        # stdout/stderr of `ray start --head`
├── ray_worker_<N>.log  # stdout/stderr of each `ray start --address` worker
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
