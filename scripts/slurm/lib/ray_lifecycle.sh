#!/bin/bash
#
# lib/ray_lifecycle.sh — submit train.py to ray + poll for terminal state.
#
# Sourced by launch_miles.sbatch after the ray cluster is up. Provides two
# functions; both write their outputs to caller-visible env vars (bash has
# no clean return-multi-value).
#
# See docs/launcher.md "ray status poll" + "OOM crash debug" for rationale.

# ray_submit_and_wait
#
# Inputs (env, all required):
#   JOBID, HEAD_NODE, HEAD_IP, RAY_DASHBOARD_PORT,
#   RUN_DIR, MILES_REPO, MEGATRON_SRC, NODE_PREAMBLE,
#   MILES_ARGS (bash array)
#
# Inputs (env, optional with defaults):
#   RAY_STATUS_POLL_INTERVAL   seconds between probes      [15]
#   RAY_STATUS_PROBE_TIMEOUT   per-probe timeout (s)       [10]
#   RAY_STATUS_FAIL_GRACE      unreadable probes -> dead   [24]
#                              (default 24 × 15s = 6 min)
#
# Outputs (env, set by this function):
#   STATE  — SUCCEEDED | FAILED | STOPPED | CLUSTER_DEAD | DEADLINE | UNKNOWN
#   JOB_RC — 0 on SUCCEEDED, 1 on FAILED, 2 on STOPPED, 3 on CLUSTER_DEAD,
#            124 on DEADLINE
#   RAY_ADDRESS — http://HEAD_IP:RAY_DASHBOARD_PORT (also exported)
#
# Side effects: streams Ray job logs to stdout, which sbatch --output
# captures in $RUN_DIR/run.log. Per-probe diagnostics are kept on
# node-local scratch while the job is running, so status polling does not
# synchronously touch the shared run dir. May leave the ray cluster running
# on terminal exit (teardown trap in caller).
read_train_status_sentinel() {
    local path="${MILES_TRAIN_STATUS_FILE:-$RUN_DIR/train_status.json}"
    [[ -f "$path" ]] || return 1
    python3 - "$path" <<'PY'
import json
import sys
from datetime import datetime, timezone

# A "running" heartbeat older than this is treated as NOT alive (the driver
# stopped making progress). Generous vs the ~per-step heartbeat cadence.
HEARTBEAT_MAX_AGE_S = 600

path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
except Exception:
    sys.exit(1)

state = str(payload.get("state", "")).lower()
try:
    rc = int(payload.get("rc", 1))
except (TypeError, ValueError):
    rc = 1

if state == "completed" and rc == 0:
    print("SUCCEEDED 0")
    sys.exit(0)
if state == "failed":
    print(f"FAILED {rc if rc != 0 else 1}")
    sys.exit(0)

# Heartbeat: a "running" job whose sentinel was updated recently is alive even
# if the Ray status API is unreadable. A stale heartbeat means the driver is no
# longer progressing, so fall through to "unresolved".
if state == "running":
    updated_at = payload.get("updated_at")
    if updated_at:
        try:
            ts = datetime.fromisoformat(updated_at)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age = (datetime.now(timezone.utc) - ts).total_seconds()
            if 0 <= age <= HEARTBEAT_MAX_AGE_S:
                print(f"ALIVE step={payload.get('step')}")
                sys.exit(0)
        except Exception:
            pass
sys.exit(1)
PY
}

ray_submit_and_wait() {
    RAY_ADDRESS="http://${HEAD_IP}:${RAY_DASHBOARD_PORT}"
    # Entry script is overridable so recipes can opt into the async driver
    # (train_async.py) instead of the default synchronous train.py. Set via
    # `export MILES_TRAIN_ENTRY=train_async.py` in the recipe.
    local TRAIN_ENTRY="${MILES_TRAIN_ENTRY:-train.py}"
    echo "[submit] $(date -Is)  ray job submit --no-wait -> $TRAIN_ENTRY"
    # Liveness/terminal sentinel on NODE-LOCAL disk (not the shared run dir). The
    # training driver (Ray head) and this watchdog (controller shell) are the same
    # node in the normal case (HEAD_NODE = GOOD_NODES[0] = batch host), so a
    # node-local path is shared writer<->reader and is immune to shared-FS stalls.
    # If they ever differ, the file is simply absent and the watchdog behaves as
    # before (no fresh heartbeat -> falls through to its prior logic).
    export MILES_TRAIN_STATUS_FILE="${TMPDIR:-/tmp}/miles-${JOBID}.train_status.json"
    rm -f "$MILES_TRAIN_STATUS_FILE" "$MILES_TRAIN_STATUS_FILE".tmp.* 2>/dev/null || true
    rm -f "$RUN_DIR/train_status.json" "$RUN_DIR"/train_status.json.tmp.* 2>/dev/null || true
    local submit_out submit_rc
    submit_out=$(srun --jobid="$JOBID" --overlap --mem=0 -N1 -n1 -w "$HEAD_NODE" \
         --export=ALL,RAY_ADDRESS="$RAY_ADDRESS",MILES_TRAIN_ENTRY="$TRAIN_ENTRY",MILES_TRAIN_STATUS_FILE="$MILES_TRAIN_STATUS_FILE",MILES_ARGS_STR="$(printf '%q ' "${MILES_ARGS[@]}")" \
         bash -c "$NODE_PREAMBLE"'
            cd '"$MILES_REPO"'
            RUNTIME_ENV_JSON=$(python - <<EOF
import json, os
print(json.dumps({"env_vars": {
    "PYTHONPATH": "'"$MEGATRON_SRC"'",
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "1",
    "HF_HOME": os.environ["HF_HOME"],
    "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
    "MILES_RUN_DIR": os.environ.get("RUN_DIR", ""),
    "MILES_TRAIN_STATUS_FILE": os.environ.get("MILES_TRAIN_STATUS_FILE", ""),
}}))
EOF
)
            eval "set -- $MILES_ARGS_STR"
            ray job submit \
                --no-wait \
                --address "$RAY_ADDRESS" \
                --runtime-env-json "$RUNTIME_ENV_JSON" \
                -- python3 "${MILES_TRAIN_ENTRY:-train.py}" "$@"
         ' 2>&1)
    submit_rc=$?
    echo "$submit_out"
    if (( submit_rc != 0 )); then
        echo "[submit] FATAL: ray job submit returned $submit_rc"
        exit "$submit_rc"
    fi
    local job_id
    job_id=$(echo "$submit_out" | grep -oE 'raysubmit_[A-Za-z0-9]+' | head -1)
    if [[ -z "$job_id" ]]; then
        echo "[submit] FATAL: could not parse submission id from ray job submit output"
        exit 78
    fi
    echo "[submit] job_id=$job_id"

    # Probe + log-follow run from the controller shell directly (conda env
    # already active), NOT via recurring `srun --overlap` steps. This narrows
    # CLUSTER_DEAD to Ray Jobs API/dashboard unavailability instead of Slurm
    # step-launch transport failures. See docs/launcher.md "ray status poll".
    local poll_interval=${RAY_STATUS_POLL_INTERVAL:-15}
    local probe_timeout=${RAY_STATUS_PROBE_TIMEOUT:-10}
    local fail_grace=${RAY_STATUS_FAIL_GRACE:-24}
    local probe_dir="${TMPDIR:-/tmp}/miles-${JOBID}-${job_id}"
    mkdir -p "$probe_dir"
    local probe_log="$probe_dir/probe.log"
    local probe_err="$probe_dir/probe.err"
    local poll_done_marker="$probe_dir/poll_done"
    rm -f "$poll_done_marker" "$probe_err"
    : > "$probe_log"
    echo "[submit] local probe diagnostics: $probe_log"

    # bg log tail — local `ray job logs --follow`, reconnect if it drops
    # unexpectedly (e.g. transient dashboard glitch). The wrapper tracks the
    # active child so teardown can stop both the subshell and the Ray CLI.
    (
        log_child=""
        cleanup_log_child() {
            if [[ -n "${log_child:-}" ]]; then
                kill "$log_child" 2>/dev/null || true
                wait "$log_child" 2>/dev/null || true
            fi
        }
        trap cleanup_log_child TERM INT EXIT

        while [[ ! -f "$poll_done_marker" ]]; do
            ray job logs --follow --address "$RAY_ADDRESS" "$job_id" 2>&1 &
            log_child=$!
            wait "$log_child"
            rc=$?
            log_child=""
            [[ -f "$poll_done_marker" ]] && break
            echo "[submit] WARN: ray job logs --follow exited rc=$rc, reconnecting in 2s"
            sleep 2 &
            log_child=$!
            wait "$log_child" || true
            log_child=""
        done
    ) &
    local log_tail_pid=$!

    # fg status poll. timeout=$probe_timeout per probe so a stuck dashboard
    # can't hang the script. $fail_grace × $poll_interval = grace window
    # (default 24 × 15s = 6 min) before declaring CLUSTER_DEAD.
    JOB_RC=1
    STATE=UNKNOWN
    local status_fail_count=0
    local deadline
    deadline=$(( ${SLURM_JOB_END_TIME:-$(( $(date +%s) + 86400 ))} - 120 ))
    local status_out status_rc err_summary sentinel_out
    while (( $(date +%s) < deadline )); do
        sleep "$poll_interval"
        # `if`-wrap the probe so `set -e` does NOT exit the script when the
        # inner timeout/ray job status returns non-zero — that's the failure
        # we're explicitly trying to count, not abort on.
        if status_out=$(timeout "$probe_timeout" \
                            ray job status --address "$RAY_ADDRESS" "$job_id" \
                            2>"$probe_err"); then
            status_rc=0
        else
            status_rc=$?
        fi
        case "$status_out" in
            *SUCCEEDED*) STATE=SUCCEEDED; JOB_RC=0; break;;
            *FAILED*)    STATE=FAILED;    JOB_RC=1; break;;
            *STOPPED*)   STATE=STOPPED;   JOB_RC=2; break;;
            *RUNNING*|*PENDING*)
                status_fail_count=0
                ;;
            *)
                status_fail_count=$((status_fail_count + 1))
                err_summary=$(tr '\n' ' ' < "$probe_err" 2>/dev/null | cut -c1-300 || true)
                {
                    echo "[$(date -Is)] unreadable status probe ${status_fail_count}/${fail_grace} rc=$status_rc"
                    echo "stdout: ${status_out:-<empty>}"
                    echo "stderr: ${err_summary:-<empty>}"
                } >> "$probe_log"
                echo "[submit] WARN: unreadable probe ${status_fail_count}/${fail_grace} (rc=$status_rc, stderr=${err_summary:-<empty>})"
                if (( status_fail_count >= fail_grace )); then
                    if sentinel_out=$(read_train_status_sentinel); then
                        if [[ "$sentinel_out" == ALIVE* ]]; then
                            # Ray status API is unreadable, but the training driver's
                            # heartbeat is fresh — the job is alive, not dead. Reset the
                            # grace counter and keep waiting instead of false-killing.
                            echo "[submit] Ray status unreadable ~$((fail_grace * poll_interval))s but train_status heartbeat is fresh ($sentinel_out) — job alive, resetting grace"
                            status_fail_count=0
                            continue
                        fi
                        STATE=${sentinel_out%% *}
                        JOB_RC=${sentinel_out##* }
                        echo "[submit] train_status.json resolved unreadable Ray status as $STATE job_rc=$JOB_RC"
                    else
                        echo "[submit] $fail_grace consecutive unreadable status probes (~$((fail_grace * poll_interval))s) and no fresh heartbeat — declaring cluster dead"
                        STATE=CLUSTER_DEAD
                        JOB_RC=3
                    fi
                    break
                fi
                ;;
        esac
    done

    if [[ "$STATE" == "UNKNOWN" ]]; then
        echo "[submit] WARN: status poll reached SLURM deadline without terminal state"
        STATE=DEADLINE
        JOB_RC=124
    fi

    # Signal bg log-follow to exit (no reconnect), then kill+wait.
    touch "$poll_done_marker" 2>/dev/null || true
    kill "$log_tail_pid" 2>/dev/null || true
    wait "$log_tail_pid" 2>/dev/null || true
    rm -f "$poll_done_marker" "$probe_err"

    echo "[submit] $(date -Is)  ${MILES_TRAIN_ENTRY:-train.py} terminal state: $STATE  job_rc=$JOB_RC"
}

# crash_debug_check
#
# Inputs (env): JOBID, RUN_DIR, plus STATE and JOB_RC (read+write).
#
# Checks ray_head.log and sacct for OOM markers; on a hit, overrides
# STATE=OOM and JOB_RC=137 and emits [crash-debug] lines.
crash_debug_check() {
    local crash_debug_hits=0
    if [[ -f "$RUN_DIR/ray_head.log" ]] && \
       grep -qE "oom_kill event|exit code=-9|Out Of Memory" "$RUN_DIR/ray_head.log"; then
        echo "[crash-debug] ray_head.log contains oom_kill / -9 markers:"
        grep -nE "oom_kill event|exit code=-9|Out Of Memory" "$RUN_DIR/ray_head.log" \
            | sed 's/^/[crash-debug]   /' | head -10
        crash_debug_hits=$((crash_debug_hits + 1))
    fi

    # sacct sometimes lags 1-3s on step state transitions
    sleep 3
    local sacct_oom
    sacct_oom=$(sacct -j "$JOBID" --noheader -P \
                    --format=JobID,State,MaxRSS,AllocTRES 2>/dev/null \
                | awk -F'|' '$2 == "OUT_OF_MEMORY" {print}')
    if [[ -n "$sacct_oom" ]]; then
        echo "[crash-debug] sacct reports OUT_OF_MEMORY steps:"
        echo "$sacct_oom" | sed 's/^/[crash-debug]   /'
        crash_debug_hits=$((crash_debug_hits + 1))
    fi

    if (( crash_debug_hits > 0 )); then
        echo "[crash-debug] forcing JOB_RC=137 (was $JOB_RC, state=$STATE)"
        JOB_RC=137
        STATE=OOM
    fi
}
