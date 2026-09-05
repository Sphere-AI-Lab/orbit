# Architecture (30-second overview)

The slurm launcher boots a Ray cluster across allocated nodes, runs
`train.py` against it, and tears down on signal or terminal job state.

```
operator                login node            slurm                 compute nodes (--exclusive, 8× H100/H200)
   │                       │                    │                          │
   │  $ bash submit.sh <exp>                    │                          │
   │ ─────────────────────►│                    │                          │
   │                       │ source recipe      │                          │
   │                       │ hf download (model + dataset, idempotent)     │
   │                       │ mkdir run dir (runs/<exp>/<YYMMDD_HHMMSS>/)   │
   │                       │ read_recent_manifests → warn on prior fails   │
   │                       │ sbatch launch_orbit.sbatch                    │
   │                       │ ─────────────────► │                          │
   │                       │                    │ allocate N nodes         │
   │                       │                    │ exec launcher on head    │
   │                       │                    │ ───────────────────────► │
   │                       │                    │                          │ write MANIFEST.json
   │                       │                    │                          │ healthcheck: per-node nvidia-smi + cuda set_device + IB rails, then cross-node NCCL all-reduce
   │                       │                    │                          │ cleanup (two-phase SIGTERM→SIGKILL + gate)
   │                       │                    │                          │ source recipe + dump args.json
   │                       │                    │                          │ auto-convert HF→torch_dist (if missing)
   │                       │                    │                          │ ray start --head on $HEAD
   │                       │                    │                          │ ray start --address on each worker
   │                       │                    │                          │ poll until N×8 GPUs visible
   │                       │                    │                          │ ray job submit --no-wait → JOB_ID
   │                       │                    │                          │ bg: ray job logs --follow → run.log
   │                       │                    │                          │ fg: ray job status loop (15 s)
   │                       │                    │                          │      train.py iterates here
   │                       │                    │                          │      until SUCCEEDED / FAILED / STOPPED / dead
   │                       │                    │                          │ crash debug: scan ray_head.log + sacct for OOM
   │                       │                    │                          │ write_manifest <terminal-state>
   │                       │                    │                          │ teardown trap: ray stop --force on every node
   │                       │                    │ ◄─────────────────────── │ exit $JOB_RC
   │                       │                    │
```

## Reading the log

- Adaptive polling (from Claude): the [`rl-monitor-loop`](../../../.claude/skills/rl-monitor-loop/SKILL.md)
  skill — invoked via `/loop /rl-monitor-loop <job-name>`. Stays close
  during startup, reports every ~30 min steady state, instant push on
  failure. Low context cost.
- One-shot snapshot (terminal or Claude): `bash scripts/slurm/check_run.sh <job-name>`
  for the MANIFEST + sacct + last-3-rollouts/train/evals summary.
- Raw stream (terminal): `tail -F runs/<job>/<stamp>/run.log` directly,
  if you want everything (warning: very noisy — ~480 lines per eval).
- Historical: `cat runs/<job>/<stamp>/run.log` for the raw stream;
  `cat runs/<job>/<stamp>/MANIFEST.json` for the audit record;
  `sacct -j $(jq -r .job_id runs/<job>/<stamp>/MANIFEST.json) --format=JobID,State,MaxRSS,AllocTRES%50`
  for slurm-side step accounting.

## Where to look when something breaks

| Symptom | First place to check |
|---|---|
| Slurm rejects the submission | `submit.sh` stdout; recipe metadata (`EXPERIMENT_NODES`, etc.) |
| `[healthcheck] BAD <node> (tier1\|tier2\|tier-ib)` | Bad GPU or IB rail — launcher auto-adds to `ExcNodeList` and requeues (cap `HEALTHCHECK_MAX_RESTARTS=3`); see `docs/launcher.md` "Healthcheck" |
| `[healthcheck] FATAL: cross-node NCCL probe failed` | Fabric/link fault, not node-localizable — fails loud after 2 tries with a repro at `nccl_probe.log`; see `docs/launcher.md` "Healthcheck" |
| `[cleanup] POISONED <node>` | Stale processes from a prior crash; slurm requeues automatically. If it loops, drain the node manually. |
| `[crash-debug] ... oom_kill` | Memory budget — see `docs/launcher.md` "Step memory cap" |
| `terminal state: CLUSTER_DEAD` | Ray dashboard unresponsive; usually downstream of an OOM or worker crash. Look in `ray_head.log` / `ray_worker_*.log`. |
| `job_rc=124` (DEADLINE) | Wall hit before training finished — bump `EXPERIMENT_TIME` or the `TIME=` env override. |
| Silence in monitor | Filter mismatch — `rl-monitor-loop` skill body documents the expected markers. |

## Files involved

```
scripts/slurm/
├── submit.sh              # login-node entry; only file you run by hand
├── launch_orbit.sbatch    # sbatch entry, runs on the slurm-allocated head node
├── check_run.sh           # snapshot script (called by rl-monitor-loop skill)
├── lib/
│   ├── manifest.sh        # MANIFEST.json read/write
│   ├── gpu_probe.py       # torch.cuda.set_device probe (tier-2 healthcheck)
│   ├── ib_probe.sh        # InfiniBand rail check (tier-ib healthcheck)
│   ├── nccl_probe.py      # cross-node NCCL all-reduce (tier-nccl healthcheck)
│   └── ray_lifecycle.sh   # ray submit + wait + crash debug
├── docs/
│   ├── architecture.md    # (this file)
│   └── launcher.md        # design rationale
└── setup/
    ├── install_env.sh     # one-time conda env build
    ├── verify_env.py      # env smoke test
    ├── pins.env           # version pins (auto-generated)
    ├── extract_pins.py    # regenerate pins.env from Dockerfile
    ├── track_submodules.py
    ├── convert_checkpoint.sh
    └── README.md
```

[`.claude/skills/slurm-launch/SKILL.md`](../../../.claude/skills/slurm-launch/SKILL.md)
and [`.claude/skills/rl-monitor-loop/SKILL.md`](../../../.claude/skills/rl-monitor-loop/SKILL.md)
package the launcher and monitor for Claude Code.

## See also

- [`launcher.md`](launcher.md) — design rationale (the "why" behind every
  knob and grep in the launcher).
- [`../setup/README.md`](../setup/README.md) — one-time conda env install +
  knob table.
- [`../submit.sh`](../submit.sh) + [`../launch_orbit.sbatch`](../launch_orbit.sbatch)
  — the two scripts diagrammed above.
- [`slurm-launch` SKILL.md](../../../.claude/skills/slurm-launch/SKILL.md) —
  operator-facing launch guide (filesystem layout, recipe pattern).
- [`rl-monitor-loop` SKILL.md](../../../.claude/skills/rl-monitor-loop/SKILL.md)
  — adaptive-cadence Claude-driven monitor (the recommended way to keep
  an eye on a long run without flooding the session).
- [`../check_run.sh`](../check_run.sh) — the snapshot script the loop calls
  each wake-up. Usable standalone too.
