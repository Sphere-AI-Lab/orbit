---
name: slurm-launch
description: Launch a miles RL training run on the slinky slurm cluster — conda env (no docker) + sbatch + multi-node Ray, with two-tier bad-node healthcheck (nvidia-smi + torch cuda set_device), auto-requeue-with-ExcNodeList on bad nodes, and SGLang health-check fault tolerance. One experiment file per recipe handles asset download + sbatch submission. Scoped to "kick off and survive node death"; result curation is out of scope for v1.
---

# slurm-launch — slurm dispatch for miles-imp

End-to-end launcher for one miles training experiment on the slinky slurm cluster. Boots a Ray cluster inside the `miles` conda env, healthchecks every allocated node (nvidia-smi count + torch cuda set_device probe), submits `train.py` against the cluster, tails logs into `runs/<name>/<YYMMDD_HHMMSS>/run.log`, and tears down on signal. On bad-node detection the launcher updates `ExcNodeList` and requeues automatically (capped at `HEALTHCHECK_MAX_RESTARTS=3`). On runtime node failure or preempt, slurm requeues and miles auto-resumes from `--load` via `latest_checkpointed_iteration.txt`.

For deeper "why does it work this way" answers — `--mem=0` on every srun, ray submit `--no-wait` + status poll, OOM crash debug, MANIFEST.json schema, exit-code table — see [`scripts/slurm/docs/launcher.md`](../../../scripts/slurm/docs/launcher.md). For a 30-second flow overview see [`scripts/slurm/docs/architecture.md`](../../../scripts/slurm/docs/architecture.md). This skill stays focused on "how to launch"; the docs cover "why."

## 0. Filesystem layout

`/data/home/$USER/` and `/data/shared/` are both on the same Weka mount (visible from every node) — the only difference is private vs group-writable. `/data/shared/` is for things meant to be **shared across projects** (HF cache); everything else stays in the repo.

| Path | What |
|---|---|
| `/data/home/$USER/workspace/miles-imp/` | This repo |
| `.../miles-imp/thirdparty/Megatron-LM/` | `radixark/Megatron-LM @ miles-main` — git submodule, pinned in `.gitmodules` |
| `.../miles-imp/thirdparty/sglang/` | `sgl-project/sglang @ sglang-miles` — git submodule, pinned in `.gitmodules` |
| `.../miles-imp/runs/<name>/<YYMMDD_HHMMSS>/` | per-run dir: run.log, args.json, ray_*.log, MANIFEST.json (gitignored) |
| `.../miles-imp/checkpoints/<name>/` | `--save` / `--load` target (gitignored) |
| `/data/shared/conda/miniconda3/envs/miles/` | Conda env |
| `/data/shared/hf_cache/` | `HF_HOME` — **general** cross-project HF cache |
| `/data/shared/hf_cache/hub/`, `xet/` | HF's automatic cache (created by `hf` CLI) |
| `/data/shared/hf_cache/models/<repo>/` | `--local-dir` model snapshots |
| `/data/shared/hf_cache/data/<repo>/` | `--local-dir` dataset snapshots |

## 1. One-time setup

See [`scripts/slurm/setup/README.md`](../../../scripts/slurm/setup/README.md) for the install walkthrough (conda env build via `install_env.sh`, source-installs of `thirdparty/Megatron-LM` + `thirdparty/sglang`, version pins, the manual `convert_checkpoint.sh` path for multi-node converts).

Two operational facts from that setup that are relevant during dispatch:

- **Auto-convert**: `launch_miles.sbatch` checks for `$HF_TORCHDIST_DIR/latest_checkpointed_iteration.txt` after sourcing the recipe and, if missing, runs `tools/convert_hf_to_torch_dist.py` on the head node before ray bring-up (~5 min for Qwen3-4B). Idempotent. While it runs, worker GPUs sit idle — for large multi-node converts pre-stage manually per the setup README.
- **Auto-download**: `scripts/slurm/submit.sh` runs `hf download` for any missing model + dataset declared in the experiment file before submitting.

## 2. The experiment file pattern

Two-layer split:

- `scripts/experiments/<name>.sh` — **pure config**, orchestrator-agnostic. Declares asset metadata (`HF_MODEL_REPO`, `HF_DATASETS`, `HF_*_DIR` paths), resource metadata (`EXPERIMENT_NODES`, `EXPERIMENT_TIME`), and the `MILES_ARGS` bash array. **No side effects when sourced** — no `ray start`, no `sbatch`, no `hf download`. Could be sourced by a local runner, a k8s wrapper, anything.
- `scripts/slurm/submit.sh` — the slurm wrapper. Takes an experiment name, sources the recipe, downloads missing assets, calls `sbatch ... launch_miles.sbatch`. The only file you run by hand.

## 3. Dispatch flow

```bash
cd /data/home/$USER/workspace/miles-imp
bash scripts/slurm/submit.sh qwen3-4B-disagg-2node
```

`submit.sh` (login node):
1. Sources `~/.config/secrets.env` (HF_TOKEN, WANDB_API_KEY).
2. Activates the `miles` conda env.
3. Sources `scripts/experiments/qwen3-4B-disagg-2node.sh` to pick up `MILES_ARGS`, `HF_*` metadata, and `EXPERIMENT_NODES/TIME`.
4. Idempotently `hf download`s `Qwen/Qwen3-4B`, `DAPO-Math-17K`, `aime-2024` into `/data/shared/hf_cache/{models,data}/...` if missing.
5. `exec sbatch ... --export=ALL,RECIPE=<abs path to scripts/experiments/<name>.sh> scripts/slurm/launch_miles.sbatch`. (The torch_dist artifact is NOT checked here — `launch_miles.sbatch` auto-converts on the head node if missing.)

`launch_miles.sbatch` then:
1. **Activates** the `miles` conda env on the controller process.
2. **Healthcheck** — two tiers per allocated node via `srun --overlap --mem=0` (`HEALTHCHECK_TIMEOUT=60` per tier). Tier 1 = `nvidia-smi --query-gpu=count`. Tier 2 = `python scripts/slurm/lib/gpu_probe.py` — `torch.cuda.set_device(i)` + tiny alloc on every visible GPU (catches `cudaErrorDevicesUnavailable` that tier 1 misses). On bad node: bad node added to `<run-dir>/.bad_nodes`, `scontrol update JobId=X ExcNodeList=<cumulative-bad>` extends the constraint, exit `EX_TEMPFAIL=75` → slurm requeues to fresh nodes. Capped at `HEALTHCHECK_MAX_RESTARTS=3`; hard fail after.
3. **Stale-process cleanup** — parallel `pkill sglang/ray/python` on every allocated node (safe under `--exclusive`).
4. **Source recipe early** — sources `$RECIPE` to bring `MILES_ARGS`, `HF_TORCHDIST_DIR`, `MODEL_ARGS` into the controller's scope. Dumps args to `<run_dir>/args.json`.
5. **Auto-convert if needed** — if `$HF_TORCHDIST_DIR/latest_checkpointed_iteration.txt` is missing, runs `tools/convert_hf_to_torch_dist.py` on the head node only (~5 min for Qwen3-4B; idle workers for that duration).
6. **Ray bring-up** — `srun -N1 -w <node>` per node; each one re-sources the conda env + sets `PYTHONPATH=thirdparty/Megatron-LM`, then `ray start --head` on first good node, `ray start --address=…` on the rest.
7. **Cluster wait** — polls `ray status` until `N×8` GPUs visible (5 min timeout).
8. **Submit** — runs `ray job submit … -- python3 train.py "${MILES_ARGS[@]}"` from the head node (the recipe was already sourced in step 4).
9. **Teardown** — `trap` on `EXIT/SIGTERM/SIGINT` issues `ray stop --force` on every node + kills child srun handles.

To override sbatch knobs from the command line:
```bash
NODES=4 TIME=24:00:00 JOB_NAME=qwen3-4B-prod SBATCH_EXTRA="--exclude=slinky-15" \
    bash scripts/slurm/submit.sh qwen3-4B-disagg-2node
```

### After dispatch — emit a monitor command for the user

`/loop` puts Claude into self-paced ScheduleWakeup mode, but the user has
to type it themselves — Claude cannot enter `/loop` autonomously. So
**after a successful `submit.sh` invocation, end your response with a
copy-pasteable monitor command** like:

```
Submitted batch job 12345 → runs/qwen3-4B-disagg-1node/260520_134522/
To monitor (adaptive cadence, low context):
    /loop /rl-monitor-loop qwen3-4B-disagg-1node
```

Use the experiment name (not the run-dir path) — `rl-monitor-loop`
auto-picks the latest stamp, so a re-submit on the same recipe doesn't
need a new monitor command.

## 4. Resume semantics

- Experiment file uses `$MILES_REPO/checkpoints/$RUN_NAME` for both `--load` and `--save`.
- `--save-interval 20` writes a Megatron `torch_dist` checkpoint every 20 rollouts + updates `latest_checkpointed_iteration.txt`.
- On slurm requeue (SIGTERM caught by `--signal=B:SIGTERM@120`, node death, preempt), the launcher re-runs from scratch — boots a fresh cluster, resubmits with the same args. Miles reads `latest_checkpointed_iteration.txt` from `--load` and continues from that step.
- To force a fresh run: `rm -rf checkpoints/<run-name>/` before resubmitting.

## 5. Bad-node policy

| Layer | Detection | Action |
|---|---|---|
| **Healthcheck** (launcher) | `nvidia-smi` count or `torch.cuda.set_device(i)` probe fails | Bad node persisted to `<run-dir>/.bad_nodes`, `scontrol update ExcNodeList`, exit `EX_TEMPFAIL` → slurm requeues to different nodes. Cap `HEALTHCHECK_MAX_RESTARTS=3`. |
| **Slurm** | Node unreachable during run | Requeues (because `--requeue` is set). |
| **SGLang health** (miles) | `/health_generate` fails | `ray.kill` the engine; rollout continues with remaining engines. Triggered by `--use-fault-tolerance`. |

The launcher does NOT silently shrink the topology — if 1 of 2 allocated nodes is bad, it fails (the recipe's `--rollout-num-gpus` is sized for the full allocation). To pin an exclude list across requeues:

```bash
SBATCH_EXTRA="--exclude=slinky-15,slinky-26" \
    bash scripts/slurm/submit.sh qwen3-4B-disagg-2node
```

## 6. Monitoring

```bash
JOBID=<from squeue>
NAME=<job-name>
RUN_DIR=/data/home/$USER/workspace/miles-imp/runs/$NAME/j$JOBID

tail -F "$RUN_DIR/run.log" \
   "$RUN_DIR/ray_head.log" \
   "$RUN_DIR"/ray_worker_*.log 2>/dev/null \
   | grep -E --line-buffered \
      "Traceback|RuntimeError|AssertionError|OOM|RayActorError|Killed|FATAL|EXIT=[1-9]|\[healthcheck\]|\[ray\]|\[submit\]|\[teardown\]|\[signal\]|iter [0-9]+/|loss=|reward=|rollout=|train="
```

Markers to watch in `run.log`:

| Marker | Means |
|---|---|
| `[healthcheck] OK <node>` | Node passed both tiers (nvidia-smi + gpu_probe) |
| `[healthcheck] BAD <node> (tier1\|tier2 …)` | Node failed a tier; persisted to `<run-dir>/.bad_nodes`, auto-excluded, requeued |
| `[healthcheck] FATAL: max restarts hit` | `HEALTHCHECK_MAX_RESTARTS` exceeded — launcher hard-fails with `MANIFEST state=FAILED reason=healthcheck_exhausted` |
| `[ray] cluster ready: N/N GPU` | Ray cluster assembled |
| `[submit] train.py exit code: 0` | Clean termination |
| `[signal] received` | SIGTERM caught (time limit / preempt); teardown started |
| `iter K/N | loss= reward=` | Training progress from `train.py` itself |

## 7. Common failure modes + recovery

### 7a. install_env.sh fails on transformer_engine

**Symptom**: `pip ... transformer_engine[pytorch]==2.10.0` fails with `nvcc` errors.

**Cause**: TE compiles against the CUDA toolchain torch was built against. Driver 570 reports max CUDA 12.8; `cu128` wheels are the right match.

**Recovery**: re-run `install_env.sh`. If still failing, conda-install matching CUDA toolkit: `conda install -n miles -c nvidia cuda-toolkit=12.8` (heavy, ~3GB).

### 7b. Healthcheck FATAL on a node

**Symptom**: `[healthcheck] BAD <node>` in `run.log`, followed by auto-requeue. After `HEALTHCHECK_MAX_RESTARTS=3` failed attempts, `[healthcheck] FATAL: max restarts hit` and `MANIFEST state=FAILED reason=healthcheck_exhausted`.

**Recovery**: usually nothing — the launcher auto-adds the bad node to `ExcNodeList` and slurm re-allocates. If you hit the cap, `cat <run-dir>/.bad_nodes` for the list of nodes the launcher gave up on, then resubmit with `SBATCH_EXTRA="--exclude=$(paste -sd, <run-dir>/.bad_nodes | sort -u)"`. Persistent failures on the same node = report to cluster admins (likely needs a node reboot — e.g. slinky-35 GPU 0+1 on 2026-05-20 was a stuck-driver state).

### 7c. Cluster never assembles

**Symptom**: `[ray] FATAL: cluster did not assemble in 300s`.

**Cause**: worker `conda activate` failed (check `<run_dir>/ray_worker_*.log` — usually `conda: command not found` or `ImportError`), or NCCL/network can't reach the head IP.

**Recovery**: fix the env. On port collision: `SBATCH_EXTRA="--export=ALL,RAY_PORT=6500,RAY_DASHBOARD_PORT=8275" bash scripts/slurm/submit.sh <name>`.

### 7d. Auto-convert fails inside the job

**Symptom**: `[convert] FATAL: conversion finished but sentinel file missing`, or `tools/convert_hf_to_torch_dist.py` traceback in `run.log` between `[recipe]` and `[ray]` markers.

**Cause**: `--hf-checkpoint` is corrupted, `MODEL_ARGS` from `scripts/models/<arch>.sh` don't match the HF config, or the head node hit OOM during checkpoint shard write.

**Recovery**: convert manually on a dedicated salloc to see the full output, then resubmit:
```bash
salloc --gres=gpu:1 --time=30 --pty bash
bash scripts/slurm/setup/convert_checkpoint.sh
```

### 7e. HF_TOKEN / WANDB_API_KEY not set

**Symptom**: launcher exits immediately with `HF_TOKEN not exported`.

**Cause**: sbatch starts a non-interactive shell; the launcher sources `~/.config/secrets.env` explicitly. If you move secrets, update both the bashrc and the launcher.

### 7f. Requeue starts from step 0

**Cause**: `--load` directory was empty (no save fired yet before SIGTERM).

**Recovery**: lower `--save-interval` for short-time-limit runs.

## 8. File map

| Path | Purpose |
|---|---|
| `scripts/experiments/<name>.sh` | One per experiment — **pure config**, orchestrator-agnostic. Defines `MILES_ARGS` + asset & resource metadata. |
| `scripts/experiments/qwen3-4B-disagg-1node.sh` | Template: 1-node 4+4 disaggregated Qwen3-4B |
| `scripts/experiments/qwen3-4B-disagg-2node.sh` | Template: 2-node node-pinned disaggregated Qwen3-4B |
| `scripts/slurm/submit.sh` | **The only slurm script run by hand** — takes experiment name, downloads assets, calls sbatch. |
| `scripts/slurm/launch_miles.sbatch` | Sbatch entry — healthcheck + ray + submit + teardown (invoked by `submit.sh`). |
| `scripts/slurm/setup/README.md` | First-time install walkthrough (read once). |
| `scripts/slurm/setup/install_env.sh` | One-time: build the `miles` conda env via uv. |
| `scripts/slurm/setup/convert_checkpoint.sh` | Optional: manual HF → Megatron `torch_dist` (used for large multi-node converts). |
| `miles/utils/health_monitor.py` | `RolloutHealthMonitor` — activated by `--use-fault-tolerance`. |
| `miles/ray/placement_group.py` | PACK + (node_ip, gpu_id) sort that pins train-on-head, rollout-on-worker. |

## 9. Out of scope (v1)

- Multi-experiment parallel dispatch (one sbatch per experiment)
- Result curation / SUMMARY.md / metrics CSV
- Per-experiment audit gates — `train.py` exit code is the only signal
- Watcher daemon — `tail -F | grep` is enough
- Apex / flash-attn-3 / mamba-ssm / INT4 — add to install_env.sh when a recipe needs them
