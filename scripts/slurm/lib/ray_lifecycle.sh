#!/bin/bash
#
# lib/ray_lifecycle.sh — submit train.py to ray + poll for terminal state.
#
# Sourced by launch_orbit.sbatch after the ray cluster is up. Provides two
# functions; both write their outputs to caller-visible env vars (bash has
# no clean return-multi-value).
#
# See docs/launcher.md "ray status poll" + "OOM crash debug" for rationale.

# ray_submit_and_wait
#
# Inputs (env, all required):
#   JOBID, HEAD_NODE, HEAD_IP, RAY_DASHBOARD_PORT,
#   RUN_DIR, ORBIT_REPO, MEGATRON_SRC, NODE_PREAMBLE,
#   ORBIT_ARGS (bash array)
#
# Inputs (env, optional with defaults):
#   RAY_STATUS_POLL_INTERVAL   seconds between probes      [15]
#   RAY_STATUS_PROBE_TIMEOUT   per-probe timeout (s)       [10]
#   RAY_STATUS_FAIL_GRACE      unreadable probes -> dead   [24]
#
# After fail_grace consecutive unreadable probes the train_status.json
# heartbeat is consulted before declaring CLUSTER_DEAD: a fresh ALIVE
# heartbeat resets the grace counter (job is alive, dashboard just wedged);
# a terminal sentinel resolves the run — see docs/launcher.md "ray status poll".
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
    local path="${ORBIT_TRAIN_STATUS_FILE:-$RUN_DIR/train_status.json}"
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
    # First line is the machine-readable verdict; subsequent lines carry the
    # driver's recorded exception so the launcher can surface it in run.log
    # (the 23771 post-mortem needed a follow-up job just to read these fields).
    print(f"FAILED {rc if rc != 0 else 1}")
    if payload.get("error_type"):
        print(f"error_type: {payload['error_type']}")
    for line in str(payload.get("error") or "").splitlines():
        print(f"error: {line}")
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

# ray_job_state_token JOB_ID OUTPUT
#
# Normalize `ray job status` CLI output to one token: RUNNING, PENDING,
# SUCCEEDED, FAILED, STOPPED, or UNREADABLE. Ray's CLI prints non-terminal
# states as "Status for job 'X': RUNNING" but terminal states only as a
# lowercase banner ("Job 'X' failed") — matching uppercase enums alone turns
# every terminal state into a full fail_grace of "unreadable" probes (that
# masked the 23771/23779 crashes for ~8 minutes each). Only the definitive
# line for the exact job id is consulted, so enum words inside the user-log
# excerpt that terminal status output embeds cannot spoof the verdict; when
# no such line exists (other CLI formats / connection errors), fall back to
# scanning the raw output.
ray_job_state_token() {
    local job_id=$1 out=$2 line
    line=$(printf '%s\n' "$out" \
        | grep -aiE "Job '${job_id}' (succeeded|failed|stopped)|Status for job '${job_id}'" \
        | head -1)
    [[ -z "$line" ]] && line=$out
    line=${line,,}
    case "$line" in
        *succeeded*) echo SUCCEEDED ;;
        *failed*)    echo FAILED ;;
        *stopped*)   echo STOPPED ;;
        *running*)   echo RUNNING ;;
        *pending*)   echo PENDING ;;
        *)           echo UNREADABLE ;;
    esac
}

ray_submit_and_wait() {
    RAY_ADDRESS="http://${HEAD_IP}:${RAY_DASHBOARD_PORT}"
    # Entry script is overridable so recipes can opt into the async driver
    # (train_async.py) instead of the default synchronous train.py. Set via
    # `export ORBIT_TRAIN_ENTRY=train_async.py` in the recipe.
    local TRAIN_ENTRY="${ORBIT_TRAIN_ENTRY:-train.py}"
    echo "[submit] $(date -Is)  ray job submit --no-wait -> $TRAIN_ENTRY"
    # Liveness/terminal sentinel on NODE-LOCAL disk (not the shared run dir). The
    # training driver (Ray head) and this watchdog (controller shell) are the same
    # node in the normal case (HEAD_NODE = GOOD_NODES[0] = batch host), so a
    # node-local path is shared writer<->reader and is immune to shared-FS stalls.
    # If they ever differ, the file is simply absent and the watchdog behaves as
    # before (no fresh heartbeat -> falls through to its prior logic).
    export ORBIT_TRAIN_STATUS_FILE="${TMPDIR:-/tmp}/orbit-${JOBID}.train_status.json"
    rm -f "$ORBIT_TRAIN_STATUS_FILE" "$ORBIT_TRAIN_STATUS_FILE".tmp.* 2>/dev/null || true
    rm -f "$RUN_DIR/train_status.json" "$RUN_DIR"/train_status.json.tmp.* 2>/dev/null || true
    local submit_out submit_rc
    # srun's --export NAME=VALUE list splits on commas with no escape syntax,
    # so any serialized arg value containing a comma (e.g. the MoE
    # --moe-layer-freq "[1,1,...]" list) silently truncates ORBIT_ARGS_STR at
    # its first comma — job 27805 lost everything from --rollout-batch-size on.
    # Ship it through the process environment instead; --export=ALL carries it
    # verbatim.
    export ORBIT_ARGS_STR
    ORBIT_ARGS_STR="$(printf '%q ' "${ORBIT_ARGS[@]}")"
    submit_out=$(srun --jobid="$JOBID" --overlap --mem=0 -N1 -n1 -w "$HEAD_NODE" \
         --export=ALL,RAY_ADDRESS="$RAY_ADDRESS",ORBIT_TRAIN_ENTRY="$TRAIN_ENTRY",ORBIT_TRAIN_STATUS_FILE="$ORBIT_TRAIN_STATUS_FILE" \
         bash -c "$NODE_PREAMBLE"'
            cd '"$ORBIT_REPO"'
            RUNTIME_ENV_JSON=$(python - <<EOF
import json, os
print(json.dumps({"env_vars": {
    # ray buffers python stdout/stderr without this; the launcher tails logs live
    "PYTHONUNBUFFERED": "1",
    "PYTHONPATH": "'"$MEGATRON_SRC"'",
    "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "1",
    "HF_HOME": os.environ["HF_HOME"],
    "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
    "WANDB_API_KEY": os.environ.get("WANDB_API_KEY", ""),
    "ORBIT_RUN_DIR": os.environ.get("RUN_DIR", ""),
    "ORBIT_TRAIN_STATUS_FILE": os.environ.get("ORBIT_TRAIN_STATUS_FILE", ""),
}}))
EOF
)
            eval "set -- $ORBIT_ARGS_STR"
            ray job submit \
                --no-wait \
                --address "$RAY_ADDRESS" \
                --runtime-env-json "$RUNTIME_ENV_JSON" \
                -- python3 "${ORBIT_TRAIN_ENTRY:-train.py}" "$@"
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
    local probe_dir="${TMPDIR:-/tmp}/orbit-${JOBID}-${job_id}"
    mkdir -p "$probe_dir"
    local probe_log="$probe_dir/probe.log"
    local probe_err="$probe_dir/probe.err"
    local poll_done_marker="$probe_dir/poll_done"
    local log_terminal_marker="$probe_dir/log_terminal_state"
    rm -f "$poll_done_marker" "$probe_err" "$log_terminal_marker"
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
            (
                ray job logs --follow --address "$RAY_ADDRESS" "$job_id" 2>&1 \
                    | awk -v marker="$log_terminal_marker" -v job="$job_id" '
                        {
                            print
                            fflush()
                            line = tolower($0)
                            if (index($0, job) && index(line, "succeeded")) {
                                print "SUCCEEDED" > marker
                                close(marker)
                            } else if (index($0, job) && index(line, "failed")) {
                                print "FAILED" > marker
                                close(marker)
                            } else if (index($0, job) && index(line, "stopped")) {
                                print "STOPPED" > marker
                                close(marker)
                            }
                        }
                    '
            ) &
            log_child=$!
            wait "$log_child"
            rc=$?
            log_child=""
            [[ -f "$poll_done_marker" ]] && break
            if [[ -s "$log_terminal_marker" ]]; then
                echo "[submit] ray job logs observed terminal state $(cat "$log_terminal_marker"); not reconnecting"
                break
            fi
            if (( rc == 0 )); then
                # Ray closes the log websocket mid-run (normal close 1000) on
                # dashboard hiccups, so rc=0 does NOT mean the job ended — that
                # assumption blinded run.log for the whole 23771 crash window.
                # Only stop when the job is confirmed terminal.
                follow_state=$(ray_job_state_token "$job_id" \
                    "$(timeout 10 ray job status --address "$RAY_ADDRESS" "$job_id" 2>/dev/null || true)")
                if [[ "$follow_state" == SUCCEEDED || "$follow_state" == FAILED || "$follow_state" == STOPPED ]]; then
                    echo "[submit] ray job logs --follow exited rc=0 and job is $follow_state; not reconnecting"
                    break
                fi
                echo "[submit] WARN: ray job logs --follow exited rc=0 but job is $follow_state — reconnecting in 2s"
            else
                echo "[submit] WARN: ray job logs --follow exited rc=$rc, reconnecting in 2s"
            fi
            sleep 2 &
            log_child=$!
            wait "$log_child" || true
            log_child=""
        done
    ) &
    local log_tail_pid=$!

    # fg status poll. timeout per probe so a stuck dashboard can't hang us;
    # unreadable probes accrue toward fail_grace — see docs/launcher.md.
    JOB_RC=1
    STATE=UNKNOWN
    local status_fail_count=0
    local deadline
    deadline=$(( ${SLURM_JOB_END_TIME:-$(( $(date +%s) + 86400 ))} - 120 ))
    local status_out status_rc err_summary log_state confirm_out sentinel_out sentinel_head
    while (( $(date +%s) < deadline )); do
        sleep "$poll_interval"
        if [[ -s "$log_terminal_marker" ]]; then
            # The log-tail awk match is loose (fires on any line with the job id
            # + succeeded/failed/stopped, incl user logs), so treat it as a hint:
            # confirm via `ray job status`, clear false positives, keep polling.
            log_state=$(cat "$log_terminal_marker" 2>/dev/null || true)
            if confirm_out=$(timeout "$probe_timeout" \
                                 ray job status --address "$RAY_ADDRESS" "$job_id" 2>/dev/null); then
                case "$(ray_job_state_token "$job_id" "$confirm_out")" in
                    SUCCEEDED) STATE=SUCCEEDED; JOB_RC=0; break;;
                    FAILED)    STATE=FAILED;    JOB_RC=1; break;;
                    STOPPED)   STATE=STOPPED;   JOB_RC=2; break;;
                    *) echo "[submit] INFO: log marker '$log_state' not confirmed by status — false positive, clearing"
                       : > "$log_terminal_marker";;
                esac
            fi
        fi
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
        case "$(ray_job_state_token "$job_id" "$status_out")" in
            SUCCEEDED) STATE=SUCCEEDED; JOB_RC=0; break;;
            FAILED)    STATE=FAILED;    JOB_RC=1; break;;
            STOPPED)   STATE=STOPPED;   JOB_RC=2; break;;
            RUNNING|PENDING)
                status_fail_count=0
                ;;
            *)
                # Unreadable probe: count toward fail_grace, then consult the
                # train_status.json heartbeat before declaring the cluster dead
                # — see docs/launcher.md "ray status poll".
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
                        # First line is "STATE RC"; any further lines carry the
                        # driver's recorded exception (surfaced below).
                        sentinel_head=${sentinel_out%%$'\n'*}
                        if [[ "$sentinel_head" == ALIVE* ]]; then
                            # Ray status API is unreadable, but the training driver's
                            # heartbeat is fresh — the job is alive, not dead. Reset the
                            # grace counter and keep waiting instead of false-killing.
                            echo "[submit] Ray status unreadable ~$((fail_grace * poll_interval))s but train_status heartbeat is fresh ($sentinel_head) — job alive, resetting grace"
                            status_fail_count=0
                            continue
                        fi
                        STATE=${sentinel_head%% *}
                        JOB_RC=${sentinel_head##* }
                        echo "[submit] train_status.json resolved unreadable Ray status as $STATE job_rc=$JOB_RC"
                        if [[ "$sentinel_out" == *$'\n'* ]]; then
                            printf '%s\n' "${sentinel_out#*$'\n'}" | sed 's/^/[submit] train_status /'
                        fi
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
    rm -f "$poll_done_marker" "$probe_err" "$log_terminal_marker"

    # Persist post-mortem evidence into the shared run dir: probe diagnostics
    # and the driver's final status sentinel live on node-local scratch and
    # vanish with the allocation (reading 23771's took a dedicated slurm job).
    if [[ -s "$probe_log" ]]; then
        cp -f "$probe_log" "$RUN_DIR/probe.log" 2>/dev/null || true
    fi
    if [[ -f "$ORBIT_TRAIN_STATUS_FILE" ]]; then
        cp -f "$ORBIT_TRAIN_STATUS_FILE" "$RUN_DIR/train_status.final.json" 2>/dev/null || true
    fi

    echo "[submit] $(date -Is)  ${ORBIT_TRAIN_ENTRY:-train.py} terminal state: $STATE  job_rc=$JOB_RC"
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
