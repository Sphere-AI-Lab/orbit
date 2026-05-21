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
# Outputs (env, set by this function):
#   STATE  — SUCCEEDED | FAILED | STOPPED | CLUSTER_DEAD | DEADLINE | UNKNOWN
#   JOB_RC — 0 on SUCCEEDED, 1 on FAILED, 2 on STOPPED, 3 on CLUSTER_DEAD,
#            124 on DEADLINE
#   RAY_ADDRESS — http://HEAD_IP:RAY_DASHBOARD_PORT (also exported)
#
# Side effects: writes $RUN_DIR/ray_head.log via the bg log tail; may
# leave the ray cluster running on terminal exit (teardown trap in caller).
ray_submit_and_wait() {
    RAY_ADDRESS="http://${HEAD_IP}:${RAY_DASHBOARD_PORT}"
    echo "[submit] $(date -Is)  ray job submit --no-wait -> train.py"
    local submit_out submit_rc
    submit_out=$(srun --jobid="$JOBID" --overlap --mem=0 -N1 -n1 -w "$HEAD_NODE" \
         --export=ALL,RAY_ADDRESS="$RAY_ADDRESS",MILES_ARGS_STR="$(printf '%q ' "${MILES_ARGS[@]}")" \
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
}}))
EOF
)
            eval "set -- $MILES_ARGS_STR"
            ray job submit \
                --no-wait \
                --address "$RAY_ADDRESS" \
                --runtime-env-json "$RUNTIME_ENV_JSON" \
                -- python3 train.py "$@"
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

    # bg log tail — drives run.log so the monitor sees train.py output.
    srun --jobid="$JOBID" --overlap --mem=0 -N1 -n1 -w "$HEAD_NODE" \
         --export=ALL,RAY_ADDRESS="$RAY_ADDRESS",JOB_ID="$job_id" \
         bash -c "$NODE_PREAMBLE"' ray job logs --follow --address "$RAY_ADDRESS" "$JOB_ID" 2>&1 || true' &
    local log_tail_pid=$!

    # fg status poll. timeout 10 per probe so a stuck dashboard can't hang
    # the script. STATUS_FAIL_GRACE=6 ≈ 90s without a readable status → dead.
    JOB_RC=1
    STATE=UNKNOWN
    local status_fail_count=0
    local status_fail_grace=6
    local deadline
    deadline=$(( ${SLURM_JOB_END_TIME:-$(( $(date +%s) + 86400 ))} - 120 ))
    local status_out
    while (( $(date +%s) < deadline )); do
        sleep 15
        status_out=$(timeout 10 srun --jobid="$JOBID" --overlap --mem=0 -N1 -n1 -w "$HEAD_NODE" \
                         --export=ALL,RAY_ADDRESS="$RAY_ADDRESS",JOB_ID="$job_id" \
                         bash -c "$NODE_PREAMBLE"' ray job status --address "$RAY_ADDRESS" "$JOB_ID" 2>&1 || true' \
                     2>&1 || true)
        case "$status_out" in
            *SUCCEEDED*) STATE=SUCCEEDED; JOB_RC=0; break;;
            *FAILED*)    STATE=FAILED;    JOB_RC=1; break;;
            *STOPPED*)   STATE=STOPPED;   JOB_RC=2; break;;
            *RUNNING*|*PENDING*)
                status_fail_count=0
                ;;
            *)
                status_fail_count=$((status_fail_count + 1))
                if (( status_fail_count >= status_fail_grace )); then
                    echo "[submit] $status_fail_grace consecutive unreadable status probes — declaring cluster dead"
                    STATE=CLUSTER_DEAD
                    JOB_RC=3
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

    kill "$log_tail_pid" 2>/dev/null || true
    wait "$log_tail_pid" 2>/dev/null || true

    echo "[submit] $(date -Is)  train.py terminal state: $STATE  job_rc=$JOB_RC"
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
