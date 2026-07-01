---
name: cluster-health
description: Scan and supervise the slinky slurm cluster's health — GPU (driver + set_device + leftover-occupancy), InfiniBand rails, cross-node NCCL fabric, and WekaFS responsiveness — decoupled from any training launch. Three modes: (1) FULL cluster scan (all probes incl. Weka, ~daily), (2) REACTIVE root-cause of the nodes a failed run/job died on, (3) QUICK cheap low-intrusion pass safe anytime. Drives the launcher's shared probes (scripts/slurm/lib/{gpu_probe,ib_probe,nccl_probe}) plus three skill-local probes (gpu_occupancy_probe.sh, fs_probe.sh, gpu_deep_probe.sh in the skill dir). Hard rules: GPU-grabbing tiers only on idle nodes; fully-allocated nodes are REPORT-ONLY (a standalone srun is a NEW job — on this cluster it can preempt-requeue the incumbent), and every srun carries --account=default --qos=normal --immediate (a non-preempting account+QOS pair) so a raced probe skips instead of preempting. Ledger lives on node-local /tmp (never block a scan on Weka). Diagnoses and hands infra a confirmed nodelist+log; it does not reboot nodes.
---

# cluster-health — standalone slinky cluster health scanning

The launcher (`launch_miles.sbatch`) healthchecks nodes **before a run** and rides
the requeue machinery. This skill is the **decoupled** version: scan and supervise
the cluster at any time — *without* launching training. It does **not** reimplement
the probes; it drives the launcher's shared `scripts/slurm/lib/*` probes plus three skill-local ones and adds the
decision logic (what each failure means + what to do) and a persisted ledger. Deep
"why" per tier: [`scripts/slurm/docs/launcher.md`](../../../scripts/slurm/docs/launcher.md) "Healthcheck".

## The probes

| Probe | Script | ~Time | Catches | Intrusion / where |
|---|---|---|---|---|
| tier1 | `nvidia-smi --query-gpu=count` | 2 s | driver gone / nvidia-smi hung | **zero** — idle/mix (rule 2) |
| ib | `lib/ib_probe.sh` | 1 s | an IB rail stuck `Polling` (breaks the NCCL ring on 1st collective) | **zero** — idle/mix (rule 2) |
| **occ** | `$SKILL_DIR/gpu_occupancy_probe.sh` | 2 s | **leftover GPU mem + orphaned procs** on an idle node — dead vLLM/sglang holding GPU mem AND lingering ray/sglang/vllm/train procs (the "POISONED" / dead-job-residue class) that tier2 passes | **zero** — meaningful on **idle** (busy occupancy is the job's) |
| **fs** | `$SKILL_DIR/fs_probe.sh` | ~1–45 s | **WekaFS DATA-path hang/slow** — a **64 MiB O_DIRECT read** that hangs/crawls while `stat`/`ls` stay fine (this wedges multi-node weight-load + slows shells); **+ persistent D-state procs** wedged in `weka`/`commit` — all invisible to GPU/IB probes | **zero** — idle/mix via srun; also runnable **login-local** (same shared mount) |
| **deep** | `$SKILL_DIR/gpu_deep_probe.sh` | 2 s | **GPU degrading under the radar** — uncorrectable ECC, thermal/power/HW throttle, PCIe-below-max, inactive NVLink: a GPU that passes tier1/tier2 but fails or crawls under real training load | **zero** — most meaningful on **idle** (throttle on busy = the job's load) |
| tier2 | `lib/gpu_probe.py` | 10 s | `cudaErrorDevicesUnavailable` — a GPU stuck at driver level tier1 misses | **GPU alloc** — idle only |
| nccl | `lib/nccl_probe.py` | ~1 min | LinkUp-but-errors-under-load rail, un-ringable topology, **silent IB→socket fallback** (busbw floor), **memlock-not-unlimited** (the RDMA→socket root cause); comms are verified by the all_reduce sweep completing | **grabs 8 GPUs/node** — ≥2 idle nodes only |

**Zero-intrusion set** = tier1 + ib + occ + fs + deep — read-only *payload*, no GPU
alloc. **Heavy set** = tier2 + nccl (real GPUs). Payload intrusion is only half the
story: **how you reach the node matters too** — see hard rule 2 (a standalone srun is
a new slurm job and can preempt).

## ⛔ Two hard rules (node selection & scheduling safety)

1. **The heavy set (tier2 + nccl) runs ONLY on idle nodes** — it grabs real GPUs and
   would perturb a live run.
2. **Never point a standalone srun at a fully-allocated (alloc) node.** From a login
   shell, `--overlap` does NOT join someone else's allocation — that requires
   `--jobid` on a job *you own* (what the launcher does). A standalone srun is a
   **NEW competing job**; under this cluster's `preempt/qos` + `CR_CORE_MEMORY`
   (`--mem=0` = whole-node memory), a high-priority-QOS srun REQUEUE-**preempts the
   running job** to start "now" (verified live: `srun --test-only` → `Preempts: <jobid>`).
   So: **alloc nodes are report-only**; **mix** nodes (free cores exist) may take
   light probes **only with the SAFE flags** below; every snippet in this skill
   carries the SAFE flags so a raced/mis-targeted srun **skips instead of preempting**.

```bash
cd "${MILES_REPO:-$HOME/miles-imp}"   # your miles-imp checkout (repo root); override MILES_REPO if elsewhere
SKILL_DIR=.claude/skills/cluster-health   # skill-local probes (occ, fs, deep) live here; shared ones in scripts/slurm/lib

# Node state is authoritative from `sinfo` (NOT `squeue -t RUNNING -o %N`, which
# misses sallocs and some job types). ONE atomic snapshot parsed into all four
# lists: separate `sinfo -t ...` calls race each other (states flip between calls),
# and on this slurm `-t alloc` ALSO matches mix nodes (verified live: 11 mix nodes
# double-classified), which would wrongly ledger every mix node as report-only too.
SNAP=$(sinfo -h -N -o '%n %t' | sort -u)
IDLE=$(echo  "$SNAP" | awk '$2=="idle"  {print $1}' | paste -sd, -)   # full stack incl. heavy set
MIX=$(echo   "$SNAP" | awk '$2=="mix"   {print $1}' | paste -sd, -)   # partially allocated: light probes WITH SAFE flags only
ALLOC=$(echo "$SNAP" | awk '$2=="alloc" {print $1}' | paste -sd, -)   # fully allocated: REPORT-ONLY — never srun these
DOWN=$(echo  "$SNAP" | awk '$2~/^(drain|drng|down|fail|maint)/{print $1}' | paste -sd, -)  # slurm already flags these; report, don't probe
# Everything else — suffixed states (idle* = slurmd not responding, idle~ = powered
# down) and transitional ones (comp/resv/boot/plnd) — must not silently vanish from
# a supervision scan: bucket them as report-only (never srun; a raced srun would
# SKIP anyway, but surface them so every node is accounted for).
OTHER=$(echo "$SNAP" | awk '$2!="idle" && $2!="mix" && $2!="alloc" && $2!~/^(drain|drng|down|fail|maint)/ {print $1"("$2")"}' | paste -sd' ' -)
echo "idle=$IDLE"; echo "mix=$MIX"; echo "alloc=$ALLOC"; echo "down/drain=$DOWN"
[[ -n "$OTHER" ]] && echo "other/report-only: $OTHER"
```

## Standalone preamble (must match training, or the GPU/NCCL probes false-fail)

```bash
PREAMBLE='
  source /data/shared/conda/miniconda3/etc/profile.d/conda.sh && conda activate "${MILES_ENV_NAME:-miles}"
  ulimit -Sl "$(ulimit -Hl)" 2>/dev/null || true   # RDMA memlock, as in NODE_PREAMBLE
  export LD_LIBRARY_PATH="$(python -c "import site;print(site.getsitepackages()[0])")/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
'
```

The zero-intrusion probes (tier1/ib/occ/fs/deep) do **not** need the preamble; only
tier2 and nccl do.

## Reusable probe helpers

**Every srun below carries `"${SAFE[@]}"` = `--account=default --qos=normal --immediate=10`.**
`--qos=normal` cannot preempt anything (unlike the high-priority training QOS) and
`--immediate=10` bails instead of queueing — so a probe that races a node going
busy **skips** (`srun: error: … Unable to allocate resources`) rather than
requeue-killing a teammate's job. Note `--test-only` returns a *plan*, not a lock:
the non-preempting QOS is what actually closes the race.

**The `--account` pairing is load-bearing.** `--qos` sets the QOS but not the
account, and `normal` only exists under the `default` account — under the usual
`viga` default account, `--qos=normal` alone dies with `Invalid qos specification`
and the whole scan silently no-ops (every node ledgered `skipped`, nothing probed).
If you must change the pair: pick a QOS whose **Preempt list is empty** —
`sacctmgr show qos format=name,priority,preempt` — **not** just any QOS your
account has; a preempting QOS like `viga` will REQUEUE-kill the incumbent job.
Pair it with the account that grants it (`--account=default`).

Every helper is **also wrapped in `timeout "$PROBE_T"`**: `--immediate` only bounds
the *allocation* wait — a probe that hangs at **runtime** (an nvidia-smi wedge is
exactly tier1's failure mode) would otherwise stall the whole scan forever. Same
invariant as the launcher's `timeout "$HEALTHCHECK_TIMEOUT" srun …`. And each helper
is **ONE srun per node** (probes batched inside a single step): fewer scheduler
round-trips, faster scans, less queue noise. Probe paths inside the steps are
repo-root-relative — srun inherits the submission cwd.

```bash
SAFE=(--account=default --qos=normal --immediate=10)
PROBE_T=${PROBE_T:-90}    # light-step runtime cap; fs_probe worst case ~45s (2x 20s dd + sleep)
TIER2_T=${TIER2_T:-300}   # tier2 cap: its preamble does conda activate + torch import off
                          # Weka — ~20s solo/warm but MINUTES under cold cache or concurrent
                          # fan-out (a whole-idle-set fan-out at 90s hit rc=124 on 10/11 nodes)

# Classify the srun outcome into a distinct trailing marker so summaries can be
# grepped mechanically: SKIP (raced busy — not a fault), HANG (outer timeout killed
# the step — the hang IS the diagnosis), FAIL (a probe inside exited nonzero and
# printed its own FAIL:/RISK: line).
probe_marker() { local tag=$1 n=$2 rc=$3 out=$4
  if   (( rc == 124 ));                                  then echo "$tag HANG $n (rc=124 — probe wedged on-node)"
  elif grep -q 'Unable to allocate resources' <<<"$out"; then echo "$tag SKIP $n (raced busy — not a fault)"
  elif (( rc != 0 ));                                    then echo "$tag FAIL $n (rc=$rc)"
  fi
}

# zero-intrusion payload, single node — MIX or IDLE, never alloc (rule 2)
light_node() { local n=$1 out rc
  echo "== $n (light) =="
  out=$(timeout "$PROBE_T" srun "${SAFE[@]}" --mem=256M -c1 -N1 -n1 -w "$n" bash -c '
    r=0
    nvidia-smi --query-gpu=count --format=csv,noheader || { echo "tier1 FAIL: $(hostname) nvidia-smi count"; r=1; }
    bash scripts/slurm/lib/ib_probe.sh                 || r=1
    bash .claude/skills/cluster-health/fs_probe.sh     || r=1
    exit $r' 2>&1); rc=$?
  printf '%s\n' "$out"; probe_marker light "$n" "$rc" "$out"
}
# full zero-intrusion set, single node — IDLE only (occ/deep are only faults there)
idle_node() { local n=$1 out rc
  echo "== $n (idle, full zero-intrusion) =="
  out=$(timeout "$PROBE_T" srun "${SAFE[@]}" --mem=256M -c1 -N1 -n1 -w "$n" bash -c '
    r=0
    nvidia-smi --query-gpu=count --format=csv,noheader || { echo "tier1 FAIL: $(hostname) nvidia-smi count"; r=1; }
    bash scripts/slurm/lib/ib_probe.sh                        || r=1
    bash .claude/skills/cluster-health/fs_probe.sh            || r=1
    bash .claude/skills/cluster-health/gpu_occupancy_probe.sh || r=1
    bash .claude/skills/cluster-health/gpu_deep_probe.sh      || r=1
    exit $r' 2>&1); rc=$?
  printf '%s\n' "$out"; probe_marker idle-set "$n" "$rc" "$out"
}
# heavy per-node GPU probe (idle only; needs preamble). --gres=gpu:8 matches the
# launcher: the gres plugin then sets CUDA_VISIBLE_DEVICES=0-7 itself, so an
# inherited CVD (e.g. probing from inside an salloc, or CVD="") can't leak in
# via --export=ALL and silently under-test. Uses TIER2_T, not PROBE_T (see above);
# fan out at most ~4 tier2 probes at a time — N concurrent conda activations are a
# Weka metadata storm that pushes even healthy nodes past any timeout.
tier2_node() { local n=$1 out rc
  out=$(timeout "$TIER2_T" srun "${SAFE[@]}" --mem=0 --gres=gpu:8 -N1 -n1 -w "$n" bash -c "$PREAMBLE python scripts/slurm/lib/gpu_probe.py" 2>&1); rc=$?
  printf '%s\n' "$out"; probe_marker tier2 "$n" "$rc" "$out"
}
```

**FAIL vs SKIP vs HANG:** the trailing marker now says which it was. `FAIL` — a
probe payload exited nonzero and printed its own `FAIL:`/`RISK:` diagnosis line
above. `SKIP` — the srun couldn't start (raced a node going busy, or a mix node
with no free memory); ledger it as `skipped`, never `bad`. `HANG` — the outer
`timeout` killed the step at runtime: the probe **HUNG on-node** and that hang *is*
the diagnosis (nvidia-smi wedge / FS wedge deeper than fs_probe's internal guards);
treat as FAIL, retry once to confirm.

---

# The three modes

## Mode 1 — FULL cluster scan (everything, whole cluster, ~1×/day)

The complete picture. Idle nodes get the full stack; mix nodes get light probes
(SAFE flags — start-or-skip); alloc nodes are **report-only**; then one cross-node
fabric probe over a pair of idle nodes. Slowest mode (nccl is a ~1-min 16-GPU
grab) — run ~daily, **on-demand** (call it when you think of it; no self-running loop).

```bash
# Alloc nodes: REPORT-ONLY (rule 2) — record who/what/how long for the ledger.
# NB %N is a COMPRESSED nodelist (slinky-[1-5]); expand per node for the ledger
# with `scontrol show hostnames <compressed>`.
squeue -h -o '%N %i %u %j %M %T' | sort
# Down/drain nodes: copy slurm's own reason into the ledger (report, don't probe):
sinfo -R -h -o '%n %t %E'
# Cluster-wide Weka data-path view WITHOUT touching busy nodes: run fs_probe right
# here (login/controller node — same shared mount):
bash "$SKILL_DIR/fs_probe.sh"

for n in ${IDLE//,/ }; do idle_node "$n"; tier2_node "$n"; done   # idle: full per-node (2 sruns)
for n in ${MIX//,/ }; do light_node "$n"; done                    # mix: light only (start-or-skip, never preempts)
```

Then the cross-node fabric probe (needs ≥2 idle nodes):

```bash
# Do NOT pre-compute/inject MASTER_ADDR from $NODES: slurm orders the allocation by
# ITS node index, not your -w order (verified live: -w slinky-12,slinky-13 allocated
# as slinky-[13,12] with rank 0 on slinky-13) — an injected head IP then points at a
# node where rank 0 is not listening and every rank dies with a 120 s connect
# timeout: a false fabric FAIL on healthy IB. nccl_probe.py derives the master from
# SLURM_STEP_NODELIST, whose order matches the actual rank placement.
NODES=slinky-20,slinky-23   # ≥2 idle nodes from $IDLE; K = their count
K=2
mkdir -p /tmp/cluster-health
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET NCCL_HEALTHCHECK_MIN_BUSBW_GB=50 timeout 420 \
  srun "${SAFE[@]}" --mem=0 -N"$K" --ntasks-per-node=8 --gpus-per-task=1 --cpus-per-task=8 \
       -w "$NODES" \
       bash -c "$PREAMBLE python scripts/slurm/lib/nccl_probe.py" \
  2>&1 | tee /tmp/cluster-health/nccl_$(date +%y%m%d_%H%M%S).log
```

(`timeout 420`, not 300: 2× cold conda/torch off Weka + the 120 s rendezvous window
already ate ~300 s once — a healthy pair PASSes well inside 420.)

`--cpus-per-task=8` is mandatory. `MIN_BUSBW_GB=50` catches a silent socket fallback
(healthy IB here ~470 GB/s; 50 is ~8× below, so no false-fail). Look for the final
`PASS ... max_busbw=NNN` or `FAIL ... < floor`. Update the ledger + best-effort flush
(below), report full table + flips.

## Mode 2 — REACTIVE root-cause (a run/job just died)

Given the nodes a failed run died on, run the **full applicable stack on exactly
those nodes** to find + confirm the cause before escalating to infra.

```bash
RUN_DIR=runs/<name>/<YYMMDD_HHMMSS>              # launcher stamps run dirs by datetime, NOT jobid
# latest attempt of a recipe: RUN_DIR=$(ls -dt runs/<name>/*/ | head -1)
grep -o '"probe_nodes":"[^"]*"' "$RUN_DIR/MANIFEST.json"   # only written when the run died on the
                                                           # aggregate NCCL probe (failure_reason=nccl_probe)
cat "$RUN_DIR/.bad_nodes" 2>/dev/null                      # per-node faults the launcher excluded
```

Set `NODES`/`K` to that set (or a user-supplied list), then:
1. `idle_node` on each suspect that is now **idle**, `light_node` on each that is **mix** (occ catches leftover-procs; fs catches a Weka stall that looked like a hang). A suspect that is already running someone else's job is **report-only** (rule 2) — note it in the ledger and probe it when it frees up.
2. If a suspect node is **idle**, add `tier2_node`.
3. If ≥2 suspect nodes are idle, run the **Mode-1 cross-node nccl probe** on exactly them to reproduce a fabric fault. The launcher's own `$RUN_DIR/nccl_probe.log` is a clean `NCCL_DEBUG=INFO` repro — your job is to confirm it **still** reproduces (node may have self-healed) and hand infra the confirmed nodelist + fresh log.

## Mode 3 — QUICK scan (cheap, low-intrusion, safe anytime)

The zero-intrusion set across idle + mix nodes — no GPU grab, no nccl, and with the
SAFE flags it never touches a running job. The "is anything obviously wrong right
now" pass; run often.

```bash
bash "$SKILL_DIR/fs_probe.sh"                                   # Weka view from right here (login node, same mount)
for n in ${IDLE//,/ }; do idle_node "$n"; done                  # idle: full zero-intrusion set (1 srun/node)
for n in ${MIX//,/ }; do light_node "$n"; done                  # mix: light only (start-or-skip)
# Alloc nodes: report-only — squeue state goes in the ledger; their turn comes when idle.
```

No tier2, no nccl. Update ledger + flip alerts as usual.

### Parallel fan-out (large clusters)

The serial loops above are fine for a handful of nodes; at tens of nodes a degraded
FS makes them crawl (fs_probe worst case ~45 s/node). Fan out with per-node logs so
output doesn't interleave — each probe step is 1 core / 256 MB, so even dozens of
concurrent sruns are a trivial scheduler load (batch ~16 at a time if that worries
you):

```bash
mkdir -p /tmp/cluster-health
for n in ${IDLE//,/ }; do idle_node  "$n" > "/tmp/cluster-health/scan.$n.log" 2>&1 & done
for n in ${MIX//,/ };  do light_node "$n" > "/tmp/cluster-health/scan.$n.log" 2>&1 & done
wait
grep -hE 'FAIL|RISK|SKIP|HANG|NOTE' /tmp/cluster-health/scan.*.log || echo "all scanned nodes clean"
```

tier2 is the exception to free fan-out: its conda-off-Weka preamble means ~4
concurrent tier2 probes max (see `TIER2_T` note above).

---

## Ledger — node-local /tmp, best-effort flush to Weka

**Never block a scan on WekaFS** (the very thing `fs` probes, and what hangs git on
this cluster). So the live ledger is node-local; Weka is a best-effort archive only.

- **Live (authoritative during a scan):** `/tmp/cluster-health/ledger.json` — fast,
  local, never hangs. One entry per node:
  `{ "slinky-20": {"last_scan":"<ISO8601>","tier1":"ok","ib":"ok|mlx5_8:Polling","occ":"ok|occupied:gpu3=140000MiB","fs":"ok|slow","tier2":"ok","busbw_gb":471.0,"verdict":"ok|bad","note":"..."} }`
- **Archive (best-effort, timeout-guarded):** at the END of a scan, flush a copy to
  Weka so history survives if this controller node changes — but a slow Weka must
  never wedge the scan:
  ```bash
  mkdir -p /tmp/cluster-health
  if timeout 10 mkdir -p runs/cluster-health 2>/dev/null \
     && timeout 10 cp /tmp/cluster-health/ledger.json runs/cluster-health/ledger.json 2>/dev/null \
     && timeout 10 cp /tmp/cluster-health/nccl_*.log  runs/cluster-health/ 2>/dev/null; then
    echo "[ledger] flushed to runs/cluster-health (weka)"
  else
    echo "[ledger] weka flush SKIPPED (FS slow/unresponsive) — live ledger on /tmp is authoritative"
  fi
  ```
  `runs/cluster-health/` is gitignored. On a fresh session, seed `/tmp` from the Weka
  archive if present (best-effort read), else start empty and rebuild by scanning.

**The ledger is agent-maintained — nothing here is automated.** After each scan,
you (the agent running this skill) parse the probes' stdout into the JSON fields,
diff against the previous ledger, and **surface only flips** (OK↔BAD). An srun-level
skip (`SKIP` marker / `Unable to allocate resources`) is `skipped`, never `bad`. The
helpers' trailing `FAIL`/`SKIP`/`HANG` marker lines keep outcomes visible in the
transcript but write nothing. Real timestamps from `date -Is` — never invent them.

### On-demand — no self-running loop
All three modes are **called manually** when you want them (a self-running `/loop` patrol
just burns tokens). The `/tmp` ledger persists across your manual calls, so each scan still
diffs against the last and **surfaces only flips** (OK↔BAD). Because the Weka/IB degradations
are often transient, one scan can miss them — the value is running Mode 3 whenever you think
of it and watching the ledger trend across calls.

## Interpretation playbook (failure → cause → action)

| Signal | Cause | Action |
|---|---|---|
| **tier1 FAIL** (no count / hangs) | node/driver dead | mark bad; report to infra; `SBATCH_EXTRA='--exclude=<n>'`. |
| **occ FAIL/RISK** `GPU occupied` / `orphaned heavy procs` on an **idle** node | leftover GPU mem or lingering ray/sglang/vllm/train procs from a dead job — slurm shows idle but a launch onto it OOMs/wedges | mark bad; exclude from launches; report PIDs to infra (they can kill/reset). This is the j21351-3 / slinky-39 class. |
| **fs FAIL/RISK** `64MiB O_DIRECT read HUNG/slow` (or `stat/write hung`) | **WekaFS data path degraded** — real reads hang/crawl while metadata (`stat`/`ls`) looks fine; wedges multi-node weight-load + slows shells. Often **TRANSIENT** | report to infra with the `dd … iflag=direct` repro + `findmnt -T /data`; it may NOT be a fixed node list — lean on the ledger trend across your manual scans, not one scan. Don't trust that node's data reads. |
| **fs RISK** `persistent D-state procs wedged in the Weka client` | a thread wedged in the Weka client on FS/IB I/O; survives SIGKILL and wedges a fresh job | node likely needs an admin reset; exclude + report pid/comm/wchan to infra. |
| **fs NOTE** `persistent D-state, non-weka wchan` | on mix/login, legit heavy I/O holds D across samples; on an **idle** node it's suspicious | weak signal on mix/login (don't mark bad); on idle, watch the trend and escalate if it persists across scans. |
| **deep FAIL** `uncorrectable ECC errors` | failing GPU memory | mark bad with the GPU index; report to infra (reboot/RMA). |
| **deep RISK** `throttling / PCIe-below-max / NVLink inactive` | GPU runs slow under load (thermal/power) or a degraded interconnect | on an **idle** node it's suspect → report; throttle on a busy node may just be the running job's load. |
| **nccl WARN** `RLIMIT_MEMLOCK soft=… (<1 TiB)` | a real memlock cap: RDMA can't register memory → NCCL falls back to TCP sockets (→ the busbw-floor FAIL). NB "unlimited" here is a <1 TiB threshold, not the RLIM_INFINITY sentinel — this cluster's healthy cap is a literal 2^63 | **config, not hardware** — raise `ulimit -l unlimited` in the launch preamble/NODE_PREAMBLE, then re-run nccl to confirm busbw recovers. |
| **tier2 FAIL** `cudaErrorDevicesUnavailable: GPU i` | a GPU stuck at driver level (tier1 still sees 8) | mark bad with the **GPU index**; only admin reboot fixes — report to infra. |
| **ib FAIL** `mlx5_X(state=Polling,…)` | one IB rail down; NCCL enumerates all 8 → 1st collective `vendor err 129` → 120 s watchdog → SIGABRT | name the **rail** to infra; silently kills any multi-node run until repaired. |
| **ib WARN** (`ibstat` missing) | tooling absent, not a fault | note it; don't mark bad. |
| helper `SKIP` marker (`Unable to allocate resources`, immediate bail) | probe raced a node going busy, or a mix node has free cores but **no free memory** for even a 256M step (common under CR_CORE_MEMORY) | **SKIP, not a fault** — ledger `skipped`; retry when the node is idle. |
| `srun: error: … Invalid qos specification` (every node!) | the SAFE `(--account, --qos)` pair doesn't exist for your association — the scan is **silently no-oping**, nothing was probed | fix `SAFE` (see the account-pairing note above); do NOT ledger these nodes as scanned, and do NOT swap in a preempting QOS like `viga`. |
| `srun: … STEP CANCELLED / PREEMPTED / Force Terminated` mid-probe | our probe job (normal QOS) got preempted by a training job arriving mid-run — most likely during the ~1-min nccl probe | **SKIP, not a fault** — retry later; do NOT read a preempted nccl probe as a fabric FAIL. |
| helper `HANG` marker (rc=124, outer `timeout` killed the step) | the probe **HUNG on-node**: nvidia-smi wedge, or an FS wedge deeper than fs_probe's internal guards. For **tier2**, first suspect the cap itself: conda+torch off cold/contended Weka takes minutes (hence `TIER2_T`, max ~4 concurrent) | treat as **FAIL** — the hang is the diagnosis; retry once (solo, generous cap) to confirm, then exclude + report. |
| **nccl FAIL** `< floor … socket fallback` | NCCL silently fell to sockets — completes at socket speed | hand infra the `nccl_*.log` (`NCCL_DEBUG=INFO`). Aggregate — if per-node ib passed, suspect link/fabric, not a Polling rail. |
| **nccl FAIL** `Watchdog / NET/IB … error 12` | a rail LinkUp but erroring under load | confirm + escalate with the log; re-run pinning one rail (`NCCL_IB_HCA=mlx5_2`) to localize. |
| **all PASS** | healthy | update ledger; no action. |

## 30-second smoke self-test (after editing this skill)

Runs the three skill-local probes right here (login node, no srun) — validates
syntax + output format, not remote node health:

```bash
bash -n "$SKILL_DIR"/*.sh && echo "syntax OK"
bash "$SKILL_DIR/gpu_occupancy_probe.sh"; echo "occ rc=$?"   # login node: nvidia-smi WARN expected
bash "$SKILL_DIR/gpu_deep_probe.sh";      echo "deep rc=$?"  # login node: WARN expected
bash "$SKILL_DIR/fs_probe.sh";            echo "fs rc=$?"    # real Weka data-path check — meaningful even here
```

## Scope / non-goals

- **Diagnose, don't repair.** Never reboots nodes or files tickets — produces a
  confirmed nodelist + log for infra. Reboot is the only fix for tier2/stuck-rail and
  is an admin action.
- **Idle-only for the heavy set.** tier2/nccl never touch a busy node.
- **Not the control plane.** This skill covers GPU/IB/NCCL/FS **fabric** faults, NOT
  Ray-dashboard/`ray job status` outages that trigger `CLUSTER_DEAD` — that's the
  launcher's train-status **heartbeat** (see `scripts/slurm/docs/watchdogs.md`).
- **Not a launch path.** To start a run (with its own in-launch healthcheck +
  requeue) use the `slurm-launch` skill.

## Backlog / genuinely-hard-to-scan (be honest about the limits)

- **Load-dependent faults** — a GPU that only Xids under sustained load, an IB rail that
  only errors at high bandwidth, a Weka path that stalls only under a real weight-load
  access pattern. A short probe can miss these; the nccl busbw sweep + the 64 MiB read
  approximate load but don't reproduce a full run. **Mode 2** (probe the *actual* failed
  node set) is the closest catch.
- **Transient degradations** — Weka data-path slowness and flapping IB rails come and go
  (infra confirmed on slinky-2/50). One scan proves nothing; run Mode 3 repeatedly and read
  the ledger trend. There is **no fixed bad-node list**.
- **Kernel Xid events** — `dmesg` Xid lines need root (`dmesg_restrict`); `gpu_deep_probe`
  covers uncorrectable ECC + throttle instead. A privileged dmesg scan is a possible add.
- **Busy-node coverage** — an alloc node is report-only (rule 2: a standalone srun
  would preempt the incumbent), so its stuck-GPU / IB / Weka state is invisible until
  it goes idle. A non-slurm read-only channel (ssh/pdsh if PAM permits, or DCGM /
  node-exporter telemetry) would restore that visibility with zero scheduler
  involvement — a possible add.
- All `NCCL_HEALTHCHECK_*` knobs are documented in `nccl_probe.py`'s header.
