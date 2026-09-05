---
name: rl-monitor-loop
description: Self-paced low-context monitor for a orbit slurm training run — stays close during startup, reports every ~30 min in steady state, instant alert on failure. Use whenever the user wants to monitor a launched run without flooding the main session with event notifications. Built on `/loop` + `ScheduleWakeup` + `scripts/slurm/check_run.sh`. Replaces the older `rl-monitor` skill (deleted).
---

# rl-monitor-loop — adaptive-cadence monitoring

## When to use this skill

The user has launched a orbit training run (typically via the `slurm-launch`
skill or `bash scripts/slurm/submit.sh`) and wants ongoing visibility
**without** the per-event noise of streaming Monitor (~50KB/hr → ~2.4MB/48h).

This skill polls on a self-paced schedule instead of streaming.

## Invocation

The user invokes this **through `/loop`** so the `ScheduleWakeup` tool
becomes available:

```
/loop /rl-monitor-loop <run-dir-or-job-name>
```

Examples:
- `/loop /rl-monitor-loop qwen3-4B-disagg-1node` (auto-pick latest stamp)
- `/loop /rl-monitor-loop runs/qwen3-4B-disagg-1node/260520_064601`

If the user types `/rl-monitor-loop <args>` without the leading `/loop`,
respond with the corrected form rather than running once-and-stopping.

## What to do at each wake-up

1. **Resolve the run dir** from the argument (a path → use directly; a
   bare name → `runs/<name>/<latest-stamp>`).
2. **Run the snapshot**:
   ```
   bash scripts/slurm/check_run.sh <run-dir>
   ```
   It returns ~20 lines: MANIFEST, slurm state, log freshness, last 3
   rollouts, last 3 train steps, eval history, failures-in-last-200-lines.
3. **First wake-up only — arm the alerts pipeline**. One Monitor, one
   pipeline, two outputs: every failure-class signal is appended to
   `<run-dir>/alerts.log` (no conversation noise), and only the launcher's
   final-verdict line pushes a Monitor notification (instant alert,
   fires 0–2 times in a whole run).

   ```
   Monitor({
     description: "rl-monitor-loop alerts (file + terminal-state push) for <run-dir>",
     persistent: true,
     timeout_ms: 3600000,
     command: "tail -F <run-dir>/run.log 2>/dev/null | grep -E --line-buffered 'crash-debug|Traceback|FATAL|OOM|cudaError|FileNotFoundError|ActorDiedError|CUDA out of memory|srun: error|POISONED|terminal state: (FAILED|STOPPED|CLUSTER_DEAD|DEADLINE|OOM)|job_rc=[1-9]' | stdbuf -oL tee -a <run-dir>/alerts.log | grep -E --line-buffered 'terminal state: (FAILED|STOPPED|CLUSTER_DEAD|DEADLINE|OOM)|job_rc=[1-9]'"
   })
   ```

   Why `stdbuf -oL tee`: `tee` is block-buffered when piping to another
   process, so without `stdbuf -oL` the file writes get held in memory
   until the buffer fills — a multi-minute lag on slow alert streams.
   `--line-buffered` does the same job for grep.

   The `check_run.sh` snapshot reads `alerts.log` automatically; you
   don't have to tail it manually unless the snapshot says there are
   entries.
4. **Report briefly** to the user. 4-8 lines max, the form depends on
   phase (see below).
5. **Choose next wake-up interval** (see cadence rules below) and call
   `ScheduleWakeup(delaySeconds, reason, prompt)` with the same `/loop`
   prompt verbatim so the next firing repeats this skill.

## Cadence rules

Pick `delaySeconds` based on what the snapshot shows:

| Run phase | Detected by | delaySeconds | Why |
|---|---|---|---|
| Pre-submit / bad arg | `check_run.sh` returns error | (stop, don't reschedule) | report and exit loop |
| Slurm queued | `SLURM ... PENDING` | 120 | should start soon |
| Ray bring-up | `MANIFEST state=RUNNING` but no `rollout 0` yet | 90 | most failures land here |
| First rollout but no first eval | `ROLLOUTS` non-empty, `EVAL (history)` empty | 300 | warming up |
| Steady state | `EVAL (history)` has at least one row | **1800** | settled — every 30 min |
| Terminal (SUCCEEDED/FAILED/STOPPED/OOM/etc.) | MANIFEST state is terminal | (stop) | final report, don't reschedule |

If `LOG ... ⚠ STALE (>10m no writes)` appears, treat as suspicious — drop
to 120s next wake-up and explicitly call it out.

## Failure handling

**Two paths, depending on how the failure surfaces:**

**Path A — alerts.log accumulating but Monitor silent.** Discovered at
the next regular wake-up via `check_run.sh`'s `ALERTS` section. Mid-run
non-terminal events (a recoverable Traceback, a transient OOM that
slurm requeues, an `srun: error` on a step that retries) fall in this
bucket — they're failure-class but the launcher hasn't reached a final
verdict. Action:
1. Read `alerts.log` for the new entries and the surrounding context
   from `run.log`.
2. Surface what happened in your wake-up report.
3. Continue scheduling — the run isn't over.

**Path B — Monitor fired (instant push).** The pipeline's inner grep
matched a terminal-state line — `terminal state: (FAILED|STOPPED|...)` or
`job_rc=[1-9]`. The run is definitively over. Action:
1. Send a `PushNotification` with a one-line summary (e.g., "j11339
   ended OOM on slinky-54 — see alerts.log").
2. **Also append the same summary to `<run-dir>/alerts.log`** with an
   ISO timestamp prefix, so the file is the single source of truth for
   "everything the monitor system did + said":
   ```bash
   printf '[push:%s] %s\n' "$(date -Is)" "<push body>" >> <run-dir>/alerts.log
   ```
3. Call `bash scripts/slurm/check_run.sh <run-dir>` for the final state.
4. Stop the loop (terminal state reached) — see Stop conditions below.
5. `TaskStop` the alerts Monitor (single task; the file pipeline lives
   inside the same Monitor's stdout pipeline so it stops with it).

The `[push:<iso>]` prefix makes skill-emitted entries trivially
distinguishable from raw `tee`'d log lines when reading `alerts.log`
later. If you fire `PushNotification` from any other path in this skill
(e.g., escalating after several stale wake-ups), use the same prefix.

## Output format per wake-up

Keep it short. Default template:

```
[rl-monitor-loop  T+<elapsed since first wake-up>  <YYYY-MM-DD HH:MM:SS local>  run=<job-name>/<stamp>]
<phase>: <single-sentence status>
<3-line metric snapshot from check_run.sh — most recent rollout/train/eval>
next wake-up in <duration>
```

Get the wall-clock with `date '+%Y-%m-%d %H:%M:%S'` so wake-ups can be
correlated with run.log / sacct timestamps after the fact.

Don't dump the full `check_run.sh` output — only on the user's first
wake-up of a session, or on failure, or when the user explicitly asks.

## Stop conditions

Do not call `ScheduleWakeup` (i.e., let the loop end) if any of:
- MANIFEST state is terminal (`SUCCEEDED`, `FAILED`, `STOPPED`, `OOM`,
  `CLUSTER_DEAD`, `DEADLINE`, `INTERRUPTED`).
- User explicitly says "stop monitoring".
- `check_run.sh` returned a non-zero exit code (probably the run dir
  vanished or the arg was wrong) — print the error and exit cleanly.

When the loop stops, also `TaskStop` the alerts Monitor from the first
wake-up, so it doesn't dangle.

## Context cost

- ~1 wake-up per 30 min once steady → ~96 wake-ups per 48h run
- ~1 message in + 1 message out per wake-up, ~2KB total
- ~200KB across a 48h run vs ~2.4MB for streaming Monitor
- 10× cheaper

## See also

- [`scripts/slurm/check_run.sh`](../../../scripts/slurm/check_run.sh) —
  the underlying snapshot script. Read-only, no side effects. Also
  usable standalone from a terminal.
- [`scripts/slurm/docs/launcher.md`](../../../scripts/slurm/docs/launcher.md)
  — design notes for the launcher whose stdout markers
  (`[crash-debug]`, `terminal state ...`, `job_rc=`) this skill
  reacts to.
- [`slurm-launch` SKILL.md](../slurm-launch/SKILL.md) — the sister skill
  that launches the runs this skill watches.
