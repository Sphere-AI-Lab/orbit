#!/bin/bash
#
# launch_miles.sbatch — slurm launcher for one miles RL training run.
#
# Boots a Ray cluster across N allocated nodes inside the `miles` conda env,
# healthchecks each node (nvidia-smi count + torch.cuda probe + InfiniBand
# rail check) and the fabric (a cross-node NCCL all-reduce smoke test),
# submits `train.py` via `ray job submit`, tears down on signal. On node
# failure or preempt, slurm requeues and miles auto-resumes from --load via
# `latest_checkpointed_iteration.txt`. Bad nodes detected by healthcheck
# are auto-excluded across requeues (see docs/launcher.md "Healthcheck").
#
# Usage:
#   You don't normally invoke this directly — run the slurm submit wrapper:
#       bash scripts/slurm/submit.sh <experiment-name>
#   which downloads any missing assets and sbatches this script with the
#   right --export=RECIPE=<absolute path to scripts/experiments/<name>.sh>.
#
# Required env:
#   RECIPE       — absolute path to a recipe file in scripts/experiments/
#                  that defines the MILES_ARGS bash array
#
# Optional env (with defaults):
#   MILES_REPO          this repo                  [/data/home/$USER/workspace/miles-imp]
#   MILES_ENV_NAME      conda env name             [miles]
#   CONDA_ROOT          conda install root         [/data/shared/conda/miniconda3]
#   THIRDPARTY_DIR      Megatron-LM + sglang src   [$MILES_REPO/thirdparty]
#   HF_HOME             general HF cache           [/data/shared/hf_cache]
#   RAY_PORT                                       [6379]
#   RAY_DASHBOARD_PORT                             [8265]
#   HEALTHCHECK_TIMEOUT   per-node probe timeout (s)         [300]
#   HEALTHCHECK_MAX_RESTARTS  auto-requeue cap for bad nodes [3]
#   MILES_HEALTHCHECK_IB      per-node IB-rail probe; 0/off/false disables  [1]
#   MILES_HEALTHCHECK_WEKA    per-node shared-FS (WekaFS) read probe; 0/off/false disables [1]
#   MILES_HEALTHCHECK_NCCL    cross-node NCCL all-reduce; 0/off/false disables [1]
#   NCCL_HEALTHCHECK_WALL     NCCL probe wall timeout (s)    [300]
#   NCCL_HEALTHCHECK_MIN_BUSBW_GB  tier-nccl busbw floor (GB/s); 0=off [50]
#   RAY_BRINGUP_TIMEOUT   per-attempt cluster-ready timeout (s)  [300]
#   RAY_BRINGUP_ATTEMPTS  ray bring-up tries before FAILED   [3]
#   RAY_STATUS_POLL_INTERVAL  ray job status poll period (s) [15]
#   RAY_STATUS_PROBE_TIMEOUT  per-probe timeout (s)          [10]
#   RAY_STATUS_FAIL_GRACE     unreadable-probe count -> dead [24]
#                             (default 24 × 15s = 6 min; a fresh train_status.json
#                             heartbeat resets it — see docs/launcher.md)
#   MILES_RAY_HEAD_NUM_GPUS   GPUs Ray registers on the head node [8]
#                             (recipes hosting a GPU sidecar on the head — e.g.
#                             a whole-node OPD teacher — export 0; honored both
#                             from the submit env and from the recipe itself)
#   ENVPACK_SERVER_NODE_COUNT  nodes reserved for one envpack server [0]
#   ENVPACK_SERVER_ENV_NAME    conda env for envpack server [$MILES_ENV_NAME]
#   ENVPACK_SERVER_WAIT_TIMEOUT envpack server readiness timeout (s) [60]
#   ENVPACK_SERVER_WATCHDOG_INTERVAL envpack server health poll interval (s) [15]
#   ENVPACK_SERVER_WATCHDOG_FAILURES consecutive health failures before requeue [3]

#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120

set -uo pipefail

MILES_REPO=${MILES_REPO:-/data/home/$USER/workspace/miles-imp}
MILES_ENV_NAME=${MILES_ENV_NAME:-miles}
CONDA_ROOT=${CONDA_ROOT:-/data/shared/conda/miniconda3}
THIRDPARTY_DIR=${THIRDPARTY_DIR:-$MILES_REPO/thirdparty}
HF_HOME=${HF_HOME:-/data/shared/hf_cache}
RAY_PORT=${RAY_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
# 300s, not 60: tier2's `import torch` cold-reads ~1GB+ of .so from /data and
# takes 85s+ on a busy FS / cold page cache — 60s false-fails healthy nodes
# (bit jobs 21571-era and 22261).
HEALTHCHECK_TIMEOUT=${HEALTHCHECK_TIMEOUT:-${PREFLIGHT_TIMEOUT:-300}}
HEALTHCHECK_MAX_RESTARTS=${HEALTHCHECK_MAX_RESTARTS:-3}
# Normalize boolean env to the bare 0/1 the (( )) guards below need — a
# non-numeric value in (( )) aborts under `set -u`. Unrecognized -> 1.
_bool() { case "${1,,}" in 0|off|false|no) echo 0 ;; *) echo 1 ;; esac; }
HEALTHCHECK_IB=$(_bool "${MILES_HEALTHCHECK_IB:-1}")
HEALTHCHECK_WEKA=$(_bool "${MILES_HEALTHCHECK_WEKA:-1}")
HEALTHCHECK_NCCL=$(_bool "${MILES_HEALTHCHECK_NCCL:-1}")
NCCL_HEALTHCHECK_WALL=${NCCL_HEALTHCHECK_WALL:-300}
NCCL_HEALTHCHECK_MIN_BUSBW_GB=${NCCL_HEALTHCHECK_MIN_BUSBW_GB:-50}
RAY_BRINGUP_TIMEOUT=${RAY_BRINGUP_TIMEOUT:-300}
RAY_BRINGUP_ATTEMPTS=${RAY_BRINGUP_ATTEMPTS:-3}
# GPUs Ray registers on the head node; whole-node head-sidecar recipes export
# 0. Validated wherever it can change (submit env here, recipe re-source
# below): under `set -uo pipefail` without -e, garbage in the EXPECTED_GPUS
# arithmetic would fail the assignment and silently leave a stale value.
MILES_RAY_HEAD_NUM_GPUS=${MILES_RAY_HEAD_NUM_GPUS:-8}
_validate_head_num_gpus() {
    if ! [[ "$MILES_RAY_HEAD_NUM_GPUS" =~ ^[0-8]$ ]]; then
        echo "FATAL: MILES_RAY_HEAD_NUM_GPUS must be 0-8, got '$MILES_RAY_HEAD_NUM_GPUS'" >&2
        # First call predates manifest.sh sourcing; skip the manifest then
        # (nothing was written yet), finalize it on the post-recipe call.
        declare -F write_manifest >/dev/null && write_manifest FAILED
        exit 78
    fi
}
_validate_head_num_gpus
ENVPACK_SERVER_NODE_COUNT=${ENVPACK_SERVER_NODE_COUNT:-0}
ENVPACK_SERVER_ENV_NAME=${ENVPACK_SERVER_ENV_NAME:-$MILES_ENV_NAME}
ENVPACK_SERVER_WAIT_TIMEOUT=${ENVPACK_SERVER_WAIT_TIMEOUT:-60}
ENVPACK_SERVER_WATCHDOG_INTERVAL=${ENVPACK_SERVER_WATCHDOG_INTERVAL:-15}
ENVPACK_SERVER_WATCHDOG_FAILURES=${ENVPACK_SERVER_WATCHDOG_FAILURES:-3}
LAUNCHER_PID=$$

: "${RECIPE:?RECIPE must be the absolute path to a recipe file (scripts/experiments/*.sh)}"

# ---------- credentials ---------------------------------------------------

# sbatch starts a non-interactive shell that does NOT source ~/.bashrc.
[ -r "$HOME/.config/secrets.env" ] && . "$HOME/.config/secrets.env"
: "${HF_TOKEN:?HF_TOKEN not exported; check ~/.config/secrets.env}"
: "${WANDB_API_KEY:?WANDB_API_KEY not exported; check ~/.config/secrets.env}"
export HF_TOKEN WANDB_API_KEY HF_HOME

# ---------- conda env (controller) ----------------------------------------

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$MILES_ENV_NAME"

MEGATRON_SRC="$THIRDPARTY_DIR/Megatron-LM"
if [ ! -d "$MEGATRON_SRC" ]; then
    echo "FATAL: $MEGATRON_SRC missing — run scripts/slurm/setup/install_env.sh first" >&2
    exit 1
fi

# ---------- run dir + MANIFEST ------------------------------------------

JOBID=${SLURM_JOB_ID:-noslurm}
RUN_NAME=${SLURM_JOB_NAME:-miles}
: "${RUN_DIR:?RUN_DIR not exported — submit.sh must set this}"
[[ -d "$RUN_DIR" ]] || { echo "FATAL: RUN_DIR=$RUN_DIR does not exist" >&2; exit 78; }

echo "[$(date -Is)] launch_miles.sbatch start"
echo "[meta] jobid=$JOBID  run_name=$RUN_NAME  run_dir=$RUN_DIR"
echo "[meta] repo=$MILES_REPO  thirdparty=$THIRDPARTY_DIR  env=$MILES_ENV_NAME  recipe=$RECIPE"
echo "[meta] nodelist=$SLURM_JOB_NODELIST  restarts=${SLURM_RESTART_COUNT:-0}"

# shellcheck disable=SC1091
source "$MILES_REPO/scripts/slurm/lib/manifest.sh"
write_manifest RUNNING

# ---------- healthcheck --------------------------------------------------
# Per-node probes (this loop) + one cross-node probe (after cleanup). See
# docs/launcher.md "Healthcheck".
#   tier 1  : `nvidia-smi --query-gpu=count` (fast, ~2s)
#   tier 2  : `python gpu_probe.py` — torch.cuda.set_device(i) on every GPU
#   tier ib : `bash ib_probe.sh` — every InfiniBand rail must be LinkUp
#   tier weka: `bash weka_probe.sh` — bounded read from the shared FS; catches a
#             wedged WekaFS client (reads hang in D-state, stalling weight/engine load)
# Per-node failures persist in $RUN_DIR/.bad_nodes across requeues and are
# added to the job's ExcNodeList so the next allocation skips them.
#   tier nccl: cross-node all-reduce smoke test (lib/nccl_probe.py), run
#             below once GOOD_NODES is known — see "healthcheck tier nccl".

ALL_NODES=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
echo "[healthcheck] nodes=${ALL_NODES[*]}"

GOOD_NODES=()
BAD_NODES=()
for n in "${ALL_NODES[@]}"; do
    if ! timeout "$HEALTHCHECK_TIMEOUT" srun --jobid="$JOBID" --overlap --mem=0 \
            -N1 -n1 -w "$n" --gres=gpu:8 \
            nvidia-smi --query-gpu=count --format=csv,noheader \
            >/dev/null 2>&1; then
        BAD_NODES+=("$n")
        echo "[healthcheck] BAD $n (tier1 nvidia-smi)"
        continue
    fi
    if ! timeout "$HEALTHCHECK_TIMEOUT" srun --jobid="$JOBID" --overlap --mem=0 \
            -N1 -n1 -w "$n" --gres=gpu:8 \
            python "$MILES_REPO/scripts/slurm/lib/gpu_probe.py" \
            > "$RUN_DIR/gpu_probe.$n.log" 2>&1; then
        BAD_NODES+=("$n")
        echo "[healthcheck] BAD $n (tier2 gpu_probe; see $RUN_DIR/gpu_probe.$n.log)"
        continue
    fi
    rm -f "$RUN_DIR/gpu_probe.$n.log"
    if (( HEALTHCHECK_IB )) && ! timeout "$HEALTHCHECK_TIMEOUT" srun --jobid="$JOBID" --overlap --mem=0 \
            -N1 -n1 -w "$n" --gres=gpu:8 \
            bash "$MILES_REPO/scripts/slurm/lib/ib_probe.sh" \
            > "$RUN_DIR/ib_probe.$n.log" 2>&1; then
        BAD_NODES+=("$n")
        echo "[healthcheck] BAD $n (tier-ib ib_probe; $(tail -n1 "$RUN_DIR/ib_probe.$n.log" 2>/dev/null))"
        continue
    fi
    rm -f "$RUN_DIR/ib_probe.$n.log"
    if (( HEALTHCHECK_WEKA )) && ! timeout "$HEALTHCHECK_TIMEOUT" srun --jobid="$JOBID" --overlap --mem=0 \
            -N1 -n1 -w "$n" --gres=gpu:8 \
            bash "$MILES_REPO/scripts/slurm/lib/weka_probe.sh" \
            > "$RUN_DIR/weka_probe.$n.log" 2>&1; then
        BAD_NODES+=("$n")
        echo "[healthcheck] BAD $n (tier-weka weka_probe; $(tail -n1 "$RUN_DIR/weka_probe.$n.log" 2>/dev/null))"
        continue
    fi
    rm -f "$RUN_DIR/weka_probe.$n.log"
    GOOD_NODES+=("$n")
    echo "[healthcheck] OK  $n"
done

if (( ${#BAD_NODES[@]} > 0 )); then
    BAD_LIST_FILE="$RUN_DIR/.bad_nodes"
    HEALTH_RESTART_FILE="$RUN_DIR/.health_restarts"
    HEALTH_RESTARTS=$( [[ -f "$HEALTH_RESTART_FILE" ]] && cat "$HEALTH_RESTART_FILE" || echo 0 )
    printf '%s\n' "${BAD_NODES[@]}" >> "$BAD_LIST_FILE"
    CUMULATIVE_BAD=$(sort -u "$BAD_LIST_FILE" | paste -sd, -)

    echo "[healthcheck] FATAL: ${#BAD_NODES[@]} of ${#ALL_NODES[@]} nodes failed probe"
    echo "[healthcheck] FATAL: bad this attempt: ${BAD_NODES[*]}"
    echo "[healthcheck] cumulative bad: $CUMULATIVE_BAD"
    echo "[healthcheck] restart attempt: $((HEALTH_RESTARTS+1)) of $HEALTHCHECK_MAX_RESTARTS"

    if (( HEALTH_RESTARTS >= HEALTHCHECK_MAX_RESTARTS )); then
        echo "[healthcheck] FATAL: max restarts hit — giving up"
        MANIFEST_EXTRA_JSON=$(printf '{"job_rc":1,"failure_reason":"healthcheck_exhausted","bad_nodes":"%s","ended_at":"%s"}' \
                              "$CUMULATIVE_BAD" "$(date -Is)") \
            write_manifest FAILED
        exit 1
    fi

    echo "$((HEALTH_RESTARTS+1))" > "$HEALTH_RESTART_FILE"
    if scontrol update JobId="$JOBID" ExcNodeList="$CUMULATIVE_BAD" 2>&1; then
        echo "[healthcheck] updated JobId=$JOBID ExcNodeList=$CUMULATIVE_BAD; exiting EX_TEMPFAIL for requeue"
    else
        echo "[healthcheck] WARN: scontrol update ExcNodeList failed; requeue may land on same bad nodes"
        echo "[healthcheck] HINT: resubmit with SBATCH_EXTRA='--exclude=$CUMULATIVE_BAD'"
    fi
    exit 75   # EX_TEMPFAIL → slurm requeue per --requeue
fi

if ! [[ "$ENVPACK_SERVER_NODE_COUNT" =~ ^[0-9]+$ ]]; then
    echo "[topology] FATAL: ENVPACK_SERVER_NODE_COUNT must be an integer, got '$ENVPACK_SERVER_NODE_COUNT'" >&2
    exit 78
fi
if (( ENVPACK_SERVER_NODE_COUNT > 1 )); then
    echo "[topology] FATAL: ENVPACK_SERVER_NODE_COUNT=$ENVPACK_SERVER_NODE_COUNT is not supported yet; use 0 or 1" >&2
    exit 78
fi
if (( ENVPACK_SERVER_NODE_COUNT >= ${#GOOD_NODES[@]} )); then
    echo "[topology] FATAL: ENVPACK_SERVER_NODE_COUNT=$ENVPACK_SERVER_NODE_COUNT leaves no nodes for Miles/Ray" >&2
    exit 78
fi

MILES_NODE_COUNT=$(( ${#GOOD_NODES[@]} - ENVPACK_SERVER_NODE_COUNT ))
RAY_NODES=( "${GOOD_NODES[@]:0:$MILES_NODE_COUNT}" )
ENVPACK_NODES=( "${GOOD_NODES[@]:$MILES_NODE_COUNT}" )

HEAD_NODE="${RAY_NODES[0]}"
# getent can exit 0 with empty output (IPv6-only / mis-resolving node) or
# nonzero (NXDOMAIN); `|| true` coerces both to empty so the guard below fires.
# Fail fast — an empty HEAD_IP otherwise surfaces as a ~5 min ray bring-up timeout.
HEAD_IP="$(getent ahostsv4 "$HEAD_NODE" | awk 'NR==1{print $1}' || true)"
if [[ -z "$HEAD_IP" ]]; then
    echo "[topology] FATAL: could not resolve $HEAD_NODE to an IPv4 address (getent ahostsv4 empty)"
    MANIFEST_EXTRA_JSON=$(printf '{"job_rc":78,"failure_reason":"head_ip_unresolved","head_node":"%s","ended_at":"%s"}' \
                          "$HEAD_NODE" "$(date -Is)") write_manifest FAILED
    exit 78
fi
WORKER_NODES=( "${RAY_NODES[@]:1}" )
# The head contributes MILES_RAY_HEAD_NUM_GPUS (default 8) to Ray; recipes
# hosting a whole-node GPU sidecar on the head export 0. Recomputed after the
# recipe is sourced below in case the export comes from the recipe itself.
EXPECTED_GPUS=$(( (${#RAY_NODES[@]} - 1) * 8 + ${MILES_RAY_HEAD_NUM_GPUS:-8} ))

if (( ENVPACK_SERVER_NODE_COUNT > 0 )); then
    ENVPACK_SERVER_NODE="${ENVPACK_NODES[0]}"
    ENVPACK_SERVER_IP=$(getent ahostsv4 "$ENVPACK_SERVER_NODE" | awk 'NR==1{print $1}')
    ENVPACK_SERVER_PORT=${ENVPACK_SERVER_PORT:-18081}
    export ENVPACK_SERVER_NODE ENVPACK_SERVER_IP ENVPACK_SERVER_PORT
    export ENVPACK_SERVER_URL="http://${ENVPACK_SERVER_IP}:${ENVPACK_SERVER_PORT}"
fi

echo "[topology] miles_head=$HEAD_NODE ($HEAD_IP)  miles_workers=${WORKER_NODES[*]:-<none>}  miles_gpus=$EXPECTED_GPUS"
if (( ENVPACK_SERVER_NODE_COUNT > 0 )); then
    echo "[topology] envpack_nodes=${ENVPACK_NODES[*]}  envpack_url=${ENVPACK_SERVER_URL}"
fi

# ---------- stale-process cleanup ----------------------------------------
# Two-phase SIGTERM → SIGKILL + D-state / leftover-GPU-mem gate.
# See docs/launcher.md "Cleanup gate" for rationale.

echo "[cleanup] two-phase kill (SIGTERM 5s -> SIGKILL) + D-state / GPU-mem gate on all nodes"
CLEANUP_PIDS=()
for n in "${GOOD_NODES[@]}"; do
    srun --jobid="$JOBID" --overlap --mem=0 -N1 -n1 -w "$n" \
         --export=ALL,NODE="$n" \
         bash -c '
            {
                pkill -TERM -u "$USER" sglang || true
                pkill -TERM -u "$USER" python || true
                ray stop --force      || true
                sleep 5
                pkill -9    -u "$USER" sglang || true
                pkill -9    -u "$USER" python || true
                pkill -9    -u "$USER" ray    || true
                sleep 1
            } >/dev/null 2>&1
            # D-state must PERSIST to count as poison. On WekaFS, normal I/O
            # (including the ps/awk pipeline here) can flicker a process into
            # uninterruptible sleep on ~20% of single samples, so a one-shot
            # check false-poisons clean nodes. A genuinely stuck process stays D
            # across every sample; transient noise does not. Re-sample up to 5x,
            # keeping the offending lines for the detail dump, and clear on the
            # first 0.
            DSTATE=0
            DSTATE_LINES=""
            for _ in 1 2 3 4 5; do
                DSTATE_LINES=$(ps -eo pid,state,stat,user:32,comm,wchan:32,args --no-headers 2>/dev/null \
                       | awk -v u="$USER" "\$2==\"D\" && \$4==u")
                d=$(printf "%s\n" "$DSTATE_LINES" | awk "NF {n++} END {print n+0}")
                if (( d == 0 )); then DSTATE=0; break; fi
                DSTATE=$d
                sleep 1
            done
            GPUMEM_LINES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null || true)
            GPUMEM=$(printf "%s\n" "$GPUMEM_LINES" | awk -F, "{gsub(/ /, \"\", \$2); s+=\$2} END {print s+0}")
            if (( DSTATE > 0 )) || (( GPUMEM > 500 )); then
                echo "[cleanup] POISONED  node=$NODE  dstate=$DSTATE  gpu_used_mb=$GPUMEM"
                if (( DSTATE > 0 )); then
                    echo "[cleanup] D-state detail node=$NODE  columns=pid,state,stat,user,comm,wchan,args"
                    printf "%s\n" "$DSTATE_LINES" | sed "s/^/[cleanup] D-state process node=$NODE  /"
                fi
                if (( GPUMEM > 500 )); then
                    echo "[cleanup] GPU memory detail node=$NODE  columns=index,used_mb"
                    printf "%s\n" "$GPUMEM_LINES" | sed "s/^/[cleanup] GPU memory node=$NODE  /"
                fi
                exit 75
            fi
            exit 0
         ' &
    CLEANUP_PIDS+=( $! )
done
POISONED_COUNT=0
for pid in "${CLEANUP_PIDS[@]}"; do
    if ! wait "$pid"; then POISONED_COUNT=$((POISONED_COUNT + 1)); fi
done
if (( POISONED_COUNT > 0 )); then
    echo "[cleanup] FATAL: $POISONED_COUNT/$((${#CLEANUP_PIDS[@]})) node(s) poisoned — requeueing"
    exit 75
fi
echo "[cleanup] all nodes clean"

# ---------- source recipe (early — needed for auto-convert step below) --

cd "$MILES_REPO"
echo "[recipe] sourcing $RECIPE"
# shellcheck disable=SC1090
source "$RECIPE"
if [[ -z "${MILES_ARGS+x}" ]]; then
    echo "[recipe] FATAL: $RECIPE did not define MILES_ARGS"
    exit 78   # EX_CONFIG
fi
# Recipes may export MILES_RAY_HEAD_NUM_GPUS (whole-node head sidecar) —
# recompute the bring-up expectation now that the recipe has run.
_validate_head_num_gpus
EXPECTED_GPUS=$(( (${#RAY_NODES[@]} - 1) * 8 + ${MILES_RAY_HEAD_NUM_GPUS:-8} ))
if (( EXPECTED_GPUS == 0 )); then
    echo "[recipe] FATAL: MILES_RAY_HEAD_NUM_GPUS=0 with no worker nodes leaves Ray no GPUs" >&2
    write_manifest FAILED
    exit 78
fi
if (( MILES_RAY_HEAD_NUM_GPUS != 8 )); then
    # Surface the reduced head in run.log: if a recipe that DOES need all
    # nodes' GPUs inherits this from a stale shell export, bring-up would pass
    # at the smaller total and the job would then pend forever on placement —
    # this line is the tell.
    echo "[recipe] MILES_RAY_HEAD_NUM_GPUS=$MILES_RAY_HEAD_NUM_GPUS — head contributes $MILES_RAY_HEAD_NUM_GPUS GPU(s) to Ray, expected total $EXPECTED_GPUS"
fi
WANDB_RUN_ID_FILE="$RUN_DIR/wandb_run_id"
if [[ -f "$WANDB_RUN_ID_FILE" ]]; then
    WANDB_RUN_ID=$(tr -d '[:space:]' < "$WANDB_RUN_ID_FILE")
fi
if [[ -z "${WANDB_RUN_ID:-}" ]]; then
    WANDB_RUN_ID="${SLURM_JOB_ID:-$JOBID}"
fi
if [[ ! -f "$WANDB_RUN_ID_FILE" || ! -s "$WANDB_RUN_ID_FILE" ]]; then
    printf '%s\n' "$WANDB_RUN_ID" > "$WANDB_RUN_ID_FILE"
fi
if [[ -n "$WANDB_RUN_ID" ]]; then
    MILES_ARGS+=( --wandb-run-id "$WANDB_RUN_ID" )
    echo "[wandb] run_id=$WANDB_RUN_ID persisted at $WANDB_RUN_ID_FILE"
else
    echo "[wandb] WARN: empty wandb run id; W&B resume across requeue will not be stable"
fi
echo "[recipe] ${#MILES_ARGS[@]} flags loaded — full args dumped to $RUN_DIR/args.json"
python3 - "${MILES_ARGS[@]}" >"$RUN_DIR/args.json" <<'PY'
import json, sys
def coerce(v):
    for t in (int, float):
        try: return t(v)
        except ValueError: pass
    return v
out, i, argv = {}, 1, sys.argv
while i < len(argv):
    tok = argv[i]
    if not tok.startswith("--"):
        i += 1; continue
    key, j = tok[2:], i + 1
    vals = []
    while j < len(argv) and not argv[j].startswith("--"):
        vals.append(coerce(argv[j])); j += 1
    out[key] = True if not vals else (vals[0] if len(vals) == 1 else vals)
    i = j
json.dump(out, sys.stdout, indent=2, sort_keys=True)
sys.stdout.write("\n")
PY

# Validate required recipe fields up front so a misconfigured recipe fails in
# seconds — before the multi-minute tier-nccl probe and the convert step.
: "${HF_MODEL_DIR:?recipe must define HF_MODEL_DIR}"

# ---------- shared per-node preamble -------------------------------------

NODE_PREAMBLE='
    set -euo pipefail
    # Raise soft memlock to hard cap so RDMA can register the pinned buffers
    # NCCL needs for multi-node IB (slurm GRES seeds soft=8MB). Use $(ulimit -Hl)
    # not literal `unlimited`: nodes with finite-int64 hard reject `-l unlimited`
    # (soft=RLIM_INFINITY > hard -> EPERM); reading hard back works for both.
    ulimit -Sl "$(ulimit -Hl)" 2>/dev/null || true
    source '"$CONDA_ROOT"'/etc/profile.d/conda.sh
    conda activate '"$MILES_ENV_NAME"'
    PY_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
    export LD_LIBRARY_PATH="$PY_SITE/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH='"$MEGATRON_SRC"'${PYTHONPATH:+:$PYTHONPATH}
    export HF_HOME='"$HF_HOME"'
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NCCL_NVLS_ENABLE=1
    : "${NCCL_DEBUG:=WARN}"
    export NCCL_DEBUG
    # Forward NCCL transport / debug overrides set by the submitter (e.g.
    # NCCL_IB_HCA to pin one rail when default enumeration exhausts RDMA limits).
    [[ -n "${NCCL_IB_HCA:-}" ]] && export NCCL_IB_HCA
    [[ -n "${NCCL_DEBUG_SUBSYS:-}" ]] && export NCCL_DEBUG_SUBSYS
    [[ -n "${NCCL_NET:-}" ]] && export NCCL_NET
    [[ -n "${NCCL_SOCKET_IFNAME:-}" ]] && export NCCL_SOCKET_IFNAME
    export PYTHONUNBUFFERED=1
'

# ---------- healthcheck tier nccl: cross-node all-reduce -----------------
# Cross-node all-reduce over IB (lib/nccl_probe.py) — catches a fabric/topology
# fault per-node ibstat can't. Aggregate check: on failure FAIL LOUD with a
# repro at $RUN_DIR/nccl_probe.log rather than requeue. Probes the Ray/training
# nodes only (RAY_NODES, not envpack-server nodes); skipped for <2 such nodes.
# See docs/launcher.md "Healthcheck".
if (( HEALTHCHECK_NCCL )) && (( ${#RAY_NODES[@]} >= 2 )); then
    RAY_CSV=$(IFS=,; echo "${RAY_NODES[*]}")
    NCCL_PROBE_LOG="$RUN_DIR/nccl_probe.log"
    NCCL_OK=0
    # --cpus-per-task=8: the batch alloc exports SLURM_CPUS_PER_TASK=128, which
    # the 8 tasks/node here would each inherit (8*128 > node CPUs -> srun
    # "Requested node configuration is not available"); pin a small per-task value.
    # NCCL_HEALTHCHECK_MIN_BUSBW_GB floors busbw to fail a silent IB->socket
    # fallback — see docs/launcher.md "Healthcheck".
    for attempt in 1 2; do
        echo "[healthcheck] tier-nccl all-reduce probe (attempt $attempt/2) across ${#RAY_NODES[@]} nodes: $RAY_CSV"
        if NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET \
                NCCL_HEALTHCHECK_MIN_BUSBW_GB="$NCCL_HEALTHCHECK_MIN_BUSBW_GB" \
                timeout "$NCCL_HEALTHCHECK_WALL" \
                srun --jobid="$JOBID" --overlap --mem=0 \
                     -N"${#RAY_NODES[@]}" --ntasks-per-node=8 --gpus-per-task=1 --cpus-per-task=8 -w "$RAY_CSV" \
                     --export=ALL,MASTER_ADDR="$HEAD_IP" \
                     bash -c "$NODE_PREAMBLE"' python "'"$MILES_REPO"'/scripts/slurm/lib/nccl_probe.py" ' \
                > "$NCCL_PROBE_LOG" 2>&1; then
            echo "[healthcheck] tier-nccl OK: $(grep -m1 'PASS' "$NCCL_PROBE_LOG" || echo passed)"
            NCCL_OK=1
            break
        fi
        echo "[healthcheck] tier-nccl attempt $attempt FAILED (see $NCCL_PROBE_LOG)"
        grep -m4 -E 'vendor err|NET/IB|nccl-probe. FAIL|Watchdog|timed out' "$NCCL_PROBE_LOG" \
            | sed 's/^/[healthcheck]   /' || true
    done
    if (( ! NCCL_OK )); then
        echo "[healthcheck] FATAL: cross-node NCCL probe failed on $RAY_CSV after 2 attempts"
        echo "[healthcheck] per-node IB rails were LinkUp (tier-ib passed) — suspect fabric/link, not a Polling rail"
        echo "[healthcheck] hand $NCCL_PROBE_LOG to infra — it is the clean NCCL_DEBUG=INFO repro (docs/launcher.md shows an ad-hoc rerun)"
        MANIFEST_EXTRA_JSON=$(printf '{"job_rc":1,"failure_reason":"nccl_probe","probe_nodes":"%s","ended_at":"%s"}' \
                              "$RAY_CSV" "$(date -Is)") \
            write_manifest FAILED
        exit 1
    fi
fi

ENVPACK_NODE_PREAMBLE='
    set -euo pipefail
    source '"$CONDA_ROOT"'/etc/profile.d/conda.sh
    conda activate '"$ENVPACK_SERVER_ENV_NAME"'
    export HF_HOME='"$HF_HOME"'
    export PYTHONUNBUFFERED=1
'

envpack_healthcheck() {
    local health_url="$1"
    if [[ -n "${ENVPACK_AUTH_TOKEN:-}" ]]; then
        curl -sf -H "Authorization: Bearer ${ENVPACK_AUTH_TOKEN}" "$health_url" >/dev/null 2>&1
    else
        curl -sf "$health_url" >/dev/null 2>&1
    fi
}

validate_positive_integer() {
    local name="$1"
    local value="${!name}"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "[envpack-server] FATAL: $name must be a positive integer, got '$value'" >&2
        exit 78
    fi
}

validate_envpack_server_timing() {
    validate_positive_integer ENVPACK_SERVER_WAIT_TIMEOUT
    validate_positive_integer ENVPACK_SERVER_WATCHDOG_INTERVAL
    validate_positive_integer ENVPACK_SERVER_WATCHDOG_FAILURES
}

# ---------- auto-convert HF -> torch_dist if missing ----------------------
# Runs on the head node only. For large multi-node converts, pre-stage via
# scripts/slurm/setup/convert_checkpoint.sh — see docs/launcher.md.
#
# Recipes opt out by leaving HF_TORCHDIST_DIR unset/empty (e.g. VLM
# recipes that load HF format directly via --megatron-to-hf-mode bridge,
# bypassing Megatron's torch_dist checkpoint flow).

if [[ -z "${HF_TORCHDIST_DIR:-}" ]]; then
    echo "[convert] HF_TORCHDIST_DIR empty — recipe opted out of torch_dist convert (loading HF directly)"
elif [[ ! -f "$HF_TORCHDIST_DIR/latest_checkpointed_iteration.txt" ]]; then
    echo "[convert] $HF_TORCHDIST_DIR missing — converting on $HEAD_NODE"
    mkdir -p "$HF_TORCHDIST_DIR"
    # Pre-export instead of listing in --export: srun's NAME=VALUE list splits
    # on commas, which corrupts serialized args for MoE models (see the
    # matching fix in lib/ray_lifecycle.sh).
    export MODEL_ARGS_STR HF_MODEL_DIR HF_TORCHDIST_DIR
    MODEL_ARGS_STR=$(printf '%q ' "${MODEL_ARGS[@]}")
    srun --jobid="$JOBID" --overlap --mem=0 -N1 -n1 -w "$HEAD_NODE" \
         --export=ALL \
         bash -c "$NODE_PREAMBLE"'
            cd "'"$MILES_REPO"'"
            eval "set -- $MODEL_ARGS_STR"
            python tools/convert_hf_to_torch_dist.py \
                "$@" \
                --hf-checkpoint "$HF_MODEL_DIR" \
                --save          "$HF_TORCHDIST_DIR"
         '
    if [[ ! -f "$HF_TORCHDIST_DIR/latest_checkpointed_iteration.txt" ]]; then
        echo "[convert] FATAL: conversion finished but sentinel file missing"
        exit 78
    fi
    echo "[convert] done -> $HF_TORCHDIST_DIR"
else
    echo "[convert] torch_dist present: $HF_TORCHDIST_DIR"
fi

# ---------- optional: envpack session server -----------------------------
# envpack HTTP session recipes that want a same-node server export
# ENVPACK_LOCAL_SERVER_CMD (and ENVPACK_LOCAL_SERVER_HEALTH). This batch script
# body runs on the head node, so a background server here listens on the head
# node's localhost — reachable by the colocated rollout actors. 1-node only:
# with worker nodes, actors off the head node cannot reach 127.0.0.1. Torn down
# with the rest of the job via teardown().
#
# Two-node recipes reserve one node with ENVPACK_SERVER_NODE_COUNT=1 and export
# ENVPACK_REMOTE_SERVER_CMD. The launcher runs that command via srun on the
# reserved envpack node and points Miles at ENVPACK_SERVER_URL.
ENVPACK_SERVER_PID=""
ENVPACK_SERVER_WATCHDOG_PID=""
if [[ -n "${ENVPACK_LOCAL_SERVER_CMD:-}" && -n "${ENVPACK_REMOTE_SERVER_CMD:-}" ]]; then
    echo "[envpack-server] FATAL: set only one of ENVPACK_LOCAL_SERVER_CMD or ENVPACK_REMOTE_SERVER_CMD" >&2
    exit 78
fi
if [[ -n "${ENVPACK_LOCAL_SERVER_CMD:-}" || -n "${ENVPACK_REMOTE_SERVER_CMD:-}" ]]; then
    validate_envpack_server_timing
fi
if [[ -n "${ENVPACK_LOCAL_SERVER_CMD:-}" ]]; then
    echo "[envpack-server] $HEAD_NODE: $ENVPACK_LOCAL_SERVER_CMD"
    bash -c "$ENVPACK_LOCAL_SERVER_CMD" >"$RUN_DIR/envpack_server.log" 2>&1 &
    ENVPACK_SERVER_PID=$!
    ENVPACK_SERVER_HEALTH_URL=${ENVPACK_LOCAL_SERVER_HEALTH:-http://127.0.0.1:18081/v1/health}
    ENVPACK_SERVER_READY=0
    for _ in $(seq 1 "$ENVPACK_SERVER_WAIT_TIMEOUT"); do
        if envpack_healthcheck "$ENVPACK_SERVER_HEALTH_URL"; then
            ENVPACK_SERVER_READY=1
            echo "[envpack-server] healthy ($ENVPACK_SERVER_HEALTH_URL)"
            break
        fi
        if ! kill -0 "$ENVPACK_SERVER_PID" 2>/dev/null; then
            echo "[envpack-server] FATAL: server exited during startup; tail $RUN_DIR/envpack_server.log:"
            tail -n 20 "$RUN_DIR/envpack_server.log" 2>/dev/null || true
            write_manifest FAILED
            exit 78
        fi
        sleep 1
    done
    if (( ENVPACK_SERVER_READY == 0 )); then
        echo "[envpack-server] FATAL: not healthy after ${ENVPACK_SERVER_WAIT_TIMEOUT}s; tail $RUN_DIR/envpack_server.log:"
        tail -n 20 "$RUN_DIR/envpack_server.log" 2>/dev/null || true
        kill "$ENVPACK_SERVER_PID" 2>/dev/null || true
        write_manifest FAILED
        exit 78
    fi
fi
if [[ -n "${ENVPACK_REMOTE_SERVER_CMD:-}" ]]; then
    : "${ENVPACK_SERVER_NODE:?ENVPACK_SERVER_NODE_COUNT must reserve an envpack node}"
    : "${ENVPACK_SERVER_URL:?ENVPACK_SERVER_URL must be set for remote envpack server}"
    echo "[envpack-server] $ENVPACK_SERVER_NODE: $ENVPACK_REMOTE_SERVER_CMD"
    # Pre-export: a comma inside the command would be split by srun's --export
    # list (see the matching fix in lib/ray_lifecycle.sh).
    export ENVPACK_REMOTE_SERVER_CMD
    srun --jobid="$JOBID" --overlap --mem=0 -N1 -n1 -w "$ENVPACK_SERVER_NODE" \
         --export=ALL \
         bash -c "$ENVPACK_NODE_PREAMBLE"'
            cd "'"$MILES_REPO"'"
            eval "$ENVPACK_REMOTE_SERVER_CMD"
         ' >"$RUN_DIR/envpack_server_${ENVPACK_SERVER_NODE}.log" 2>&1 &
    ENVPACK_SERVER_PID=$!
    ENVPACK_SERVER_HEALTH_URL=${ENVPACK_REMOTE_SERVER_HEALTH:-${ENVPACK_SERVER_URL%/}/v1/health}
    ENVPACK_SERVER_READY=0
    for _ in $(seq 1 "$ENVPACK_SERVER_WAIT_TIMEOUT"); do
        if envpack_healthcheck "$ENVPACK_SERVER_HEALTH_URL"; then
            ENVPACK_SERVER_READY=1
            echo "[envpack-server] healthy ($ENVPACK_SERVER_HEALTH_URL)"
            break
        fi
        if ! kill -0 "$ENVPACK_SERVER_PID" 2>/dev/null; then
            echo "[envpack-server] FATAL: remote server exited during startup; tail $RUN_DIR/envpack_server_${ENVPACK_SERVER_NODE}.log:"
            tail -n 20 "$RUN_DIR/envpack_server_${ENVPACK_SERVER_NODE}.log" 2>/dev/null || true
            write_manifest FAILED
            exit 78
        fi
        sleep 1
    done
    if (( ENVPACK_SERVER_READY == 0 )); then
        echo "[envpack-server] FATAL: not healthy after ${ENVPACK_SERVER_WAIT_TIMEOUT}s; tail $RUN_DIR/envpack_server_${ENVPACK_SERVER_NODE}.log:"
        tail -n 20 "$RUN_DIR/envpack_server_${ENVPACK_SERVER_NODE}.log" 2>/dev/null || true
        kill "$ENVPACK_SERVER_PID" 2>/dev/null || true
        write_manifest FAILED
        exit 78
    fi
fi

start_envpack_server_watchdog() {
    local health_url="$1"
    if [[ -z "$health_url" ]]; then
        return 0
    fi
    validate_envpack_server_timing
    trap 'echo "[envpack-server] watchdog failure - clean shutdown then requeue"; MANIFEST_EXTRA_JSON=$(printf '\''{"job_rc":75,"failure_reason":"envpack_server_unhealthy","ended_at":"%s"}'\'' "$(date -Is)") write_manifest FAILED; teardown; exit 75' USR1
    (
        failures=0
        while true; do
            sleep "$ENVPACK_SERVER_WATCHDOG_INTERVAL"
            if envpack_healthcheck "$health_url"; then
                failures=0
                continue
            fi
            failures=$((failures + 1))
            echo "[envpack-server] watchdog health failure $failures/$ENVPACK_SERVER_WATCHDOG_FAILURES ($health_url)" >&2
            if (( failures >= ENVPACK_SERVER_WATCHDOG_FAILURES )); then
                echo "[envpack-server] FATAL: watchdog marked envpack server unhealthy; requesting requeue" >&2
                kill -USR1 "$LAUNCHER_PID" 2>/dev/null || true
                exit 75
            fi
        done
    ) &
    ENVPACK_SERVER_WATCHDOG_PID=$!
    echo "[envpack-server] watchdog started pid=$ENVPACK_SERVER_WATCHDOG_PID url=$health_url interval=${ENVPACK_SERVER_WATCHDOG_INTERVAL}s failures=$ENVPACK_SERVER_WATCHDOG_FAILURES"
}

# ---------- ray bring-up (fail-fast detection + retry) --------------------
# Boot ray head + workers, then poll BOTH `ray status` (GPUs registered) and the
# head/worker srun PIDs. Each `ray start … --block` blocks forever on success, so
# a dead PID means that node's ray exited — i.e. the cluster collapsed (commonly
# slow GCS/raylet registration tripping Ray's hardcoded 30s node-info wait, which
# no launcher timeout can extend; see docs/launcher.md "Ray bring-up"). On that
# signal we stop polling at once instead of idling out the whole
# RAY_BRINGUP_TIMEOUT, snapshot the node-local Ray component logs (they vanish
# with the node otherwise), and retry on a fresh `ray start` — these failures are
# usually transient.

# Launch head + workers as backgrounded srun; (re)populates the RAY_PIDS array.
start_ray_cluster() {
    RAY_PIDS=()
    echo "[ray] starting head on $HEAD_NODE"
    srun --jobid="$JOBID" --overlap --mem=0 -N1 -n1 -w "$HEAD_NODE" \
         --export=ALL,MASTER_ADDR="$HEAD_IP",RAY_PORT="$RAY_PORT",RAY_DASHBOARD_PORT="$RAY_DASHBOARD_PORT" \
         bash -c "$NODE_PREAMBLE"'
            ray stop --force >/dev/null 2>&1 || true
            # MILES_RAY_HEAD_NUM_GPUS: recipes that host a GPU sidecar on the
            # head node (e.g. a whole-node OPD teacher via the local-server
            # hook) export 0 so Ray never schedules actor/rollout bundles
            # onto the GPUs the sidecar owns.
            ray start --head \
                --node-ip-address "$MASTER_ADDR" \
                --port "$RAY_PORT" \
                --num-gpus "${MILES_RAY_HEAD_NUM_GPUS:-8}" \
                --dashboard-host 0.0.0.0 \
                --dashboard-port "$RAY_DASHBOARD_PORT" \
                --disable-usage-stats --block
         ' >"$RUN_DIR/ray_head.log" 2>&1 &
    RAY_PIDS=( $! )

    for w in "${WORKER_NODES[@]}"; do
        echo "[ray] starting worker on $w"
        srun --jobid="$JOBID" --overlap --mem=0 -N1 -n1 -w "$w" \
             --export=ALL,MASTER_ADDR="$HEAD_IP",RAY_PORT="$RAY_PORT" \
             bash -c "$NODE_PREAMBLE"'
                ray stop --force >/dev/null 2>&1 || true
                ray start --address "${MASTER_ADDR}:${RAY_PORT}" \
                    --num-gpus 8 --disable-usage-stats --block
             ' >"$RUN_DIR/ray_worker_${w}.log" 2>&1 &
        RAY_PIDS+=( $! )
    done
}

# Snapshot a failed attempt's logs to $RUN_DIR/ray-debug-attempt<N>/: the
# per-node top-level srun logs plus Ray's node-local component logs
# (gcs_server / raylet / ray_client_server / *_agent), the only place the real
# bootstrap cause is recorded and which otherwise disappear with the node.
collect_ray_logs() {
    local attempt="$1"
    local dst="$RUN_DIR/ray-debug-attempt${attempt}"
    local node role
    mkdir -p "$dst"
    cp "$RUN_DIR"/ray_head.log "$RUN_DIR"/ray_worker_*.log "$dst"/ 2>/dev/null || true
    for node in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
        [[ "$node" == "$HEAD_NODE" ]] && role=head || role=worker
        srun --jobid="$JOBID" --overlap --mem=0 -N1 -n1 -w "$node" \
             bash -c '
                d="$1/$2-$3"; mkdir -p "$d"; s=/tmp/ray/session_latest/logs
                for f in gcs_server.out gcs_server.err raylet.out raylet.err \
                         ray_client_server.out ray_client_server.err \
                         dashboard_agent.log dashboard.log runtime_env_agent.log \
                         ray_process_exit.log monitor.out monitor.err log_monitor.log; do
                    [ -e "$s/$f" ] && cp "$s/$f" "$d/" 2>/dev/null
                done
             ' _ "$dst" "$role" "$node" >/dev/null 2>&1 || true
    done
    echo "[ray] failed-attempt ${attempt} logs saved to $dst"
}

# Stop ray on every allocated node + reap the bring-up srun PIDs (between retries).
stop_all_ray() {
    local n
    for n in "$HEAD_NODE" "${WORKER_NODES[@]}"; do
        srun --jobid="$JOBID" --overlap --mem=0 -N1 -n1 -w "$n" \
             bash -c "$NODE_PREAMBLE"' ray stop --force ' >/dev/null 2>&1 || true
    done
    kill "${RAY_PIDS[@]}" 2>/dev/null || true
}

RAY_PIDS=()

teardown() {
    echo "[teardown] $(date -Is)  stopping ray + killing srun children"
    [[ -n "${ENVPACK_SERVER_WATCHDOG_PID:-}" ]] && kill "$ENVPACK_SERVER_WATCHDOG_PID" 2>/dev/null || true
    [[ -n "${ENVPACK_SERVER_PID:-}" ]] && kill "$ENVPACK_SERVER_PID" 2>/dev/null || true
    for n in "${GOOD_NODES[@]}"; do
        srun --jobid="$JOBID" --overlap --mem=0 -N1 -n1 -w "$n" \
             bash -c "$NODE_PREAMBLE"' ray stop --force ' >/dev/null 2>&1 || true
    done
    kill "${RAY_PIDS[@]}" 2>/dev/null || true
}
trap 'echo "[signal] received — clean shutdown then exit"; write_manifest INTERRUPTED; teardown; exit 0' SIGTERM SIGINT
trap teardown EXIT

if [[ -n "${ENVPACK_SERVER_HEALTH_URL:-}" ]]; then
    start_envpack_server_watchdog "$ENVPACK_SERVER_HEALTH_URL"
fi

# ---------- wait for cluster ready (fail-fast + retry) -------------------

RAY_READY=0
for (( attempt=1; attempt <= RAY_BRINGUP_ATTEMPTS; attempt++ )); do
    echo "[ray] bring-up attempt ${attempt}/${RAY_BRINGUP_ATTEMPTS}"
    start_ray_cluster
    echo "[ray] waiting up to ${RAY_BRINGUP_TIMEOUT}s for $EXPECTED_GPUS GPUs"
    ELAPSED=0
    ATTEMPT_FAIL=""
    while (( ELAPSED < RAY_BRINGUP_TIMEOUT )); do
        sleep 10
        ELAPSED=$((ELAPSED + 10))

        # fail-fast: a backgrounded `ray start … --block` PID dies only if that
        # node's ray exited — the cluster is gone, don't wait out the timeout.
        for pid in "${RAY_PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null || { ATTEMPT_FAIL="ray start pid $pid exited"; break; }
        done
        [[ -n "$ATTEMPT_FAIL" ]] && break

        OUT=$(srun --jobid="$JOBID" --overlap --mem=0 -N1 -n1 -w "$HEAD_NODE" \
                  bash -c "$NODE_PREAMBLE"' ray status --address "'"$HEAD_IP"':'"$RAY_PORT"'" ' 2>/dev/null || true)
        GPU_LINE=$(echo "$OUT" | grep -E "^[[:space:]]*[0-9.]+/[0-9.]+ GPU" | head -1 || true)
        TOTAL=$(echo "$GPU_LINE" | sed -E 's|.*/([0-9]+)(\.[0-9]+)? GPU.*|\1|')
        if [[ -n "${TOTAL:-}" && "$TOTAL" -ge "$EXPECTED_GPUS" ]]; then
            echo "[ray] cluster ready: $GPU_LINE"
            RAY_READY=1
            break
        fi
    done

    (( RAY_READY )) && break

    if [[ -n "$ATTEMPT_FAIL" ]]; then
        echo "[ray] attempt ${attempt} collapsed fast ($ATTEMPT_FAIL) — see ray_head.log / ray_worker_*.log"
    else
        echo "[ray] attempt ${attempt} did not assemble in ${RAY_BRINGUP_TIMEOUT}s"
    fi
    collect_ray_logs "$attempt"
    stop_all_ray
    if (( attempt < RAY_BRINGUP_ATTEMPTS )); then
        echo "[ray] retrying bring-up after 15s…"
        sleep 15
    fi
done

if (( ! RAY_READY )); then
    # Terminal, not requeue: the in-place retries above already cover transient
    # bring-up flakiness, and there is no bad-node exclusion for ray bring-up, so a
    # blind --requeue would just reroll onto the same slow nodes (and, uncapped, loop
    # to walltime — the old "stuck at RUNNING" failure). Record FAILED and stop.
    echo "[ray] FATAL: cluster did not assemble after ${RAY_BRINGUP_ATTEMPTS} attempt(s); see ray-debug-attempt*/"
    MANIFEST_EXTRA_JSON=$(printf '{"job_rc":1,"failure_reason":"ray_bringup","attempts":%d,"head_node":"%s","ended_at":"%s"}' \
        "$RAY_BRINGUP_ATTEMPTS" "$HEAD_NODE" "$(date -Is)") \
        write_manifest FAILED
    exit 1
fi

# ---------- submit train.py + wait + crash debug -------------------------
# Implementation moved to lib/ray_lifecycle.sh; see docs/launcher.md for
# the why (--no-wait + status poll + OOM crash debug).

# shellcheck disable=SC1091
source "$MILES_REPO/scripts/slurm/lib/ray_lifecycle.sh"
ray_submit_and_wait
crash_debug_check

MANIFEST_EXTRA_JSON=$(printf '{"job_rc": %d, "ended_at": "%s"}' "$JOB_RC" "$(date -Is)") \
    write_manifest "$STATE"
exit "$JOB_RC"
