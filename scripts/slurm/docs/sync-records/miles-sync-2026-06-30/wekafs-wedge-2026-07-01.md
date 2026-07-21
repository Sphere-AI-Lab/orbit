# WekaFS wedge — 2026-07-01 (blocked geo3k validation, job 21623)

## Summary
The shared `/data` filesystem (**WekaFS**) client wedged on **slinky-2 and slinky-50**
(confirmed, job 21623): real data reads hang in **uninterruptible D-state** in the kernel Weka
client, so any process reading `/data` (Python imports, model-weight load) stalls indefinitely.
This froze sglang engine bring-up for job 21623 (nodes slinky-2,19,50): the engines reached
`Load weight begin` and then stalled reading weight shards from `/data` — weight load did not
complete before the job was killed. It is **not** a GPU/IB/NCCL/slurm fault (all passed
healthcheck) and **not** application code — it is a storage-layer hang.

**Scope:** confirmed only on **slinky-2/50** during job 21623. Whether it is broader or transient
across the cluster is an **unproven hypothesis** — a later slinky-19 observation was inconclusive,
and the run that prompted it (21656) actually failed for an unrelated reason: a post-sync SGLang
`/begin_weight_update` 404 (see the 2026-07-01 note at the end + its own write-up).

## Evidence
Run sequence (job 21623, `run.log`): healthcheck OK on all 3 nodes → tier-nccl PASS
(`ranks=24 max_busbw=254.9GB/s`) → Ray bring-up attempt 1 succeeded, cluster ready →
router launched, listening at `10.245.188.13:3976` → SGLang engines reached
`Load weight begin`, then **stalled during weight-shard loading** (one shard reached ~25 %)
until the job was cancelled. So Ray / NCCL / GPU / router were all fine; the stall is at
reading weights/env files from `/data`.

Live read probe (`timeout 15 cat <torch .so> >/dev/null; echo rc=$?`), confirmed post-hoc
(`/data` is `wekafs` on all three):
- `slinky-2`  → **rc=124 (hung)**
- `slinky-50` → **rc=124 (hung)**
- `slinky-19` → **rc=0 (healthy)**

On the two hung nodes, engine spawn/worker Python procs sat in **`D` (uninterruptible)**
state for minutes, `wchan = wekafs_dentry_revalidate` / `commit_blocking_request`.
`nvidia-smi` showed GPUs near-0 MB — consistent with the stall being in the **host-side
weight-file read, before any GPU transfer** (not GPU-busy). `stat`/`ls` on `/data` still
return (cached metadata); only real **data reads** hang.

## What it is NOT (ruled out by evidence)
- **Not throughput-slow.** wandb streamed to the cloud fine throughout (`200 OK`, ~130 ms,
  every 15 s). The router 600 s fix worked (router listening on :3976); `wandb.init()`
  completed. Ray assembled on attempt 1; tier-nccl passed at 255 GB/s.
- **Not a code deadlock.** The block is in the *kernel* Weka client (D-state, `wekafs_*`),
  not a Python lock (that would be `S`/futex with a Python stack).
- **Healthcheck now probes it (at launch).** A `tier-weka` liveness read was added (this
  branch) to catch a node already wedged at launch (like the 21623 case) and exclude it. A
  preflight probe only sees launch-time state; storage stability is the real fix.

## Easy repro / detector
Run on a suspect node (or via `srun --overlap --jobid=<J> -w <node> bash -c '…'`):

1. **Detect wedged processes.** Sample a few times (D-state can be transient) and widen
   `wchan` — the default column truncates `commit_blocking_request` to `commit`, dropping
   the `wekafs` hint, so match both:
   ```bash
   for i in 1 2 3; do
     ps -eo stat,pid,etime,comm,wchan:32 | awk '$1 ~ /^D/ && $NF ~ /wekafs|commit_blocking_request/'
     sleep 2
   done
   ```
   The same row across samples = a process stuck uninterruptibly in the Weka client → mount
   is wedged. (A healthy Weka op completes in µs–ms.)

2. **Active read probe (bypasses metadata cache — read real bytes):**
   ```bash
   timeout 15 cat /data/shared/conda/miniconda3/envs/miles/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so > /dev/null
   echo "rc=$?"     # 0 = healthy, 124 = HUNG (wedged)
   ```
   Use a large, not-recently-read file (a torch `.so` is ~1 GB). `stat`/`ls` alone will
   *not* detect it — they hit cached metadata.

## Impact & mitigation
- Any job reading `/data` (imports, weight load) hangs. `scancel` leaves the job in
  `COMPLETING` because D-state procs can't be killed until the Weka op returns.
- **Raising app/launcher timeouts does not help** — a wedged read may never return (this is
  why bumping `RAY_STATUS_FAIL_GRACE`, engine readiness, etc. would not have saved the run).
- **Fix is storage-side** (Weka client/agent restart on the affected nodes, or backend
  recovery) — for cluster ops.
- **Launcher guard (added, this branch):** a `tier-weka` healthcheck tier (`lib/weka_probe.sh`,
  a timed O_DIRECT `/data` read) excludes a node whose `/data` is wedged at launch, like
  bad-IB/GPU nodes. Caveat: a preflight probe only catches launch-time state.

## Context
Discovered validating the 2026-06-30 upstream sync (`sync-upstream-20260630`). The sync
fixes all worked as far as the wedge allowed (ray bring-up, tier-nccl, router 600 s
timeout); only the WekaFS wedge blocked reaching training / RL-metrics.

## Note — 2026-07-01, job 21656: NOT this issue (separate SGLang 404)

A follow-up geo3k run (21656, on pre-probed-healthy slinky-19,20,36) was initially read as a
recurrence of this wedge, but the run artifacts do **not** support that:
- The run **loaded weights and all engines came up healthy** (`/health` 200 continuously,
  02:16–02:30) — it did **not** stall in init. (The init was *slow* — GPUs stayed idle a long
  time — which an in-flight snapshot mistook for a stall.)
- It **failed on `POST /begin_weight_update` → 404 Not Found** at the first weight update
  (`sglang_engine.py:560`) — a **post-sync SGLang API mismatch**, not storage. See
  `sglang-begin-weight-update-404-2026-07-01.md`.
- **No Weka evidence in the 21656 run dir** (`grep -c 'wekafs|commit_blocking|O_DIRECT' run.log`
  = 0; no `weka_probe.*.log`). The slinky-19 "~12 s / 32 MiB" reading was a live operator
  observation, not an artifact-confirmed cause — and that node is not what failed the run.

So 21656 is **removed** from this doc's evidence. Corollary: the unreadable `ray job status`
probes seen in both runs are a Ray Jobs-API observability quirk, **not** direct storage-wedge
evidence.
