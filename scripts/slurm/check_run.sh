#!/bin/bash
#
# check_run.sh — concise health snapshot of a miles slurm run.
#
# Read-only. Designed to be called every wake-up by the rl-monitor-loop skill;
# no state, no side effects. Reads:
#   $RUN_DIR/MANIFEST.json   for declared state
#   $RUN_DIR/run.log         for wandb-aligned step boundaries
#   sacct / squeue           for slurm-side accounting
#
# Output: ~20-25 line summary (manifest, slurm state, log freshness, last 3
# rollouts/train steps, eval history, failure markers in last 200 lines).
#
# Usage:
#   bash scripts/slurm/check_run.sh <run-dir>
#   bash scripts/slurm/check_run.sh <job-name>     # auto-pick latest stamp

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <run-dir|job-name>" >&2
    exit 64
fi
ARG="$1"

# Resolve ARG to a concrete run dir.
if [[ -d "$ARG" && -f "$ARG/run.log" ]]; then
    RUN_DIR="$(cd "$ARG" && pwd)"
elif [[ -d "$MILES_REPO/runs/$ARG" ]]; then
    LATEST=$(ls -1 "$MILES_REPO/runs/$ARG" 2>/dev/null | sort | tail -1)
    [[ -z "$LATEST" ]] && { echo "no runs found under $MILES_REPO/runs/$ARG/" >&2; exit 66; }
    RUN_DIR="$MILES_REPO/runs/$ARG/$LATEST"
else
    echo "could not resolve '$ARG' as run dir or job-name under $MILES_REPO/runs/" >&2
    exit 66
fi

RUN_LOG="$RUN_DIR/run.log"
MANIFEST="$RUN_DIR/MANIFEST.json"
[[ -f "$RUN_LOG" ]] || { echo "missing $RUN_LOG" >&2; exit 66; }

echo "=== check_run ${RUN_DIR#$MILES_REPO/} ==="

# --- MANIFEST.json ---
if [[ -f "$MANIFEST" ]]; then
    python3 - "$MANIFEST" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
fields = ['state', 'job_id', 'head_node', 'started_at', 'updated_at']
parts = [f"{k}={d.get(k,'?')}" for k in fields if d.get(k) not in (None, '')]
if d.get('job_rc') is not None: parts.append(f"job_rc={d['job_rc']}")
print("MANIFEST    " + "  ".join(parts))
PY
else
    echo "MANIFEST    (missing)"
fi

# --- slurm side: squeue if alive, sacct if ended ---
JOB_ID=$(python3 -c "import json; print(json.load(open('$MANIFEST')).get('job_id',''))" 2>/dev/null || echo "")
if [[ -n "$JOB_ID" ]]; then
    SQ=$(squeue -j "$JOB_ID" -h -o "%i %T TIME=%M NODELIST=%R" 2>/dev/null || true)
    if [[ -n "$SQ" ]]; then
        echo "SLURM       $SQ"
    else
        SA=$(sacct -j "$JOB_ID" --noheader -P --format=JobID,State,ExitCode,Elapsed,NodeList 2>/dev/null \
            | awk -F'|' '$1 !~ /\./ {print}' | head -1)
        [[ -n "$SA" ]] && echo "SLURM       (ended) $SA" || echo "SLURM       (no sacct record yet)"
    fi
fi

# --- run.log size + freshness ---
LOG_LINES=$(wc -l < "$RUN_LOG")
LOG_SIZE=$(du -h "$RUN_LOG" | awk '{print $1}')
LOG_MTIME=$(stat -c %Y "$RUN_LOG")
NOW=$(date +%s)
AGE=$((NOW - LOG_MTIME))
if (( AGE < 60 )); then AGE_STR="${AGE}s"; elif (( AGE < 3600 )); then AGE_STR="$((AGE/60))m"; else AGE_STR="$((AGE/3600))h$(((AGE%3600)/60))m"; fi
WARN=""
(( AGE > 600 )) && WARN="  ⚠ STALE (>10m no writes)"
echo "LOG         ${LOG_LINES} lines, ${LOG_SIZE}, last write ${AGE_STR} ago${WARN}"

# --- all dict-form summaries parsed in one python pass (avoids pipefail SIGPIPE risk) ---
python3 - "$RUN_LOG" <<'PY'
import re, sys, ast
path = sys.argv[1]
# Match on message format only — file:line in the logger format is brittle
# (any unrelated insertion upstream shifts the line and silently blinds us).
ROLLOUT_RE = re.compile(r' - rollout (\d+): (\{[^}]*\})')
TRAIN_RE   = re.compile(r' - step (\d+): (\{[^}]*\})')
EVAL_RE    = re.compile(r' - eval (\d+): (\{[^}]*\})')

rollouts, trains, evals = [], [], []
with open(path, 'r', errors='replace') as f:
    for line in f:
        if (m := ROLLOUT_RE.search(line)):
            try: rollouts.append((m.group(1), ast.literal_eval(m.group(2))))
            except Exception: pass
        elif (m := TRAIN_RE.search(line)):
            try: trains.append((m.group(1), ast.literal_eval(m.group(2))))
            except Exception: pass
        elif (m := EVAL_RE.search(line)):
            try: evals.append((m.group(1), ast.literal_eval(m.group(2))))
            except Exception: pass

print()
print("ROLLOUTS (last 3)")
for rid, d in rollouts[-3:]:
    print(f"  {rid:>3}  rewards={d.get('rollout/rewards',float('nan')):.2e}  "
          f"raw_reward={d.get('rollout/raw_reward',float('nan')):.3f}  "
          f"truncated={d.get('rollout/truncated',float('nan')):.2f}  "
          f"resp_len={d.get('rollout/response_lengths',float('nan')):.0f}")
if not rollouts: print("  (none yet)")

print()
print("TRAIN (last 3)")
for sid, d in trains[-3:]:
    print(f"  {sid:>3}  loss={d.get('train/loss',float('nan')):.2e}  "
          f"pg_loss={d.get('train/pg_loss',float('nan')):.2e}  "
          f"grad_norm={d.get('train/grad_norm',float('nan')):.3f}  "
          f"lr={d.get('train/lr-pg_0',float('nan')):.0e}")
if not trains: print("  (none yet)")

print()
print("EVAL (history)")
if not evals:
    print("  (none yet)")
else:
    for eid, d in evals:
        # top-level eval/<dataset> key (count==1 slash)
        acc_keys = [k for k in d if k.startswith('eval/') and k.count('/') == 1]
        if not acc_keys: continue
        acc_key = acc_keys[0]
        rl_key = next((k for k in d if k.endswith('/response_len/mean')), None)
        rl_val = d.get(rl_key, float('nan')) if rl_key else float('nan')
        print(f"  {eid:>3}  {acc_key}={d.get(acc_key, float('nan')):.3f}  resp_len/mean={rl_val:.0f}")
PY

# --- alerts: prefer $RUN_DIR/alerts.log if armed by rl-monitor-loop,
#     else fall back to scanning last 200 lines of run.log -----------
echo
ALERTS_FILE="$RUN_DIR/alerts.log"
if [[ -f "$ALERTS_FILE" ]]; then
    COUNT=$(wc -l < "$ALERTS_FILE")
    if (( COUNT == 0 )); then
        echo "ALERTS      alerts.log: 0 entries (clean)"
    else
        echo "ALERTS      alerts.log: $COUNT entries"
        tail -5 "$ALERTS_FILE" | sed 's/^/  /'
        if (( COUNT > 5 )); then
            echo "  (showing last 5; older entries above)"
        fi
    fi
else
    FAILS=$(tail -200 "$RUN_LOG" 2>/dev/null | grep -E "crash-debug|Traceback|FATAL|OOM|cudaError|FileNotFoundError|ActorDiedError|CUDA out of memory|srun: error|POISONED|terminal state: (FAILED|STOPPED|CLUSTER_DEAD|DEADLINE|OOM)" || true)
    if [[ -z "$FAILS" ]]; then
        echo "ALERTS      (no alerts.log; last 200 run.log lines clean)"
    else
        echo "ALERTS      (no alerts.log; scanning last 200 run.log lines)"
        echo "$FAILS" | head -8 | sed 's/^/  /'
        REST=$(echo "$FAILS" | wc -l)
        if (( REST > 8 )); then
            echo "  ... and $((REST - 8)) more"
        fi
    fi
fi
