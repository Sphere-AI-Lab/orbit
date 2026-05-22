#!/bin/bash
#
# launch_miles.sbatch — slurm launcher for one miles RL training run.
#
# Boots a Ray cluster across N allocated nodes inside the `miles` conda env,
# healthchecks each node (nvidia-smi count + torch.cuda.set_device probe),
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
#   HEALTHCHECK_TIMEOUT   per-node probe timeout (s)         [60]
#   HEALTHCHECK_MAX_RESTARTS  auto-requeue cap for bad nodes [3]
#   RAY_BRINGUP_TIMEOUT   cluster-ready timeout (s)          [300]
#   RAY_STATUS_POLL_INTERVAL  ray job status poll period (s) [15]
#   RAY_STATUS_PROBE_TIMEOUT  per-probe timeout (s)          [10]
#   RAY_STATUS_FAIL_GRACE     unreadable-probe count -> dead [24]
#                             (default 24 × 15s = 6 min)

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
HEALTHCHECK_TIMEOUT=${HEALTHCHECK_TIMEOUT:-${PREFLIGHT_TIMEOUT:-60}}
HEALTHCHECK_MAX_RESTARTS=${HEALTHCHECK_MAX_RESTARTS:-3}
RAY_BRINGUP_TIMEOUT=${RAY_BRINGUP_TIMEOUT:-300}

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
# Two tiers per node, see docs/launcher.md "Healthcheck".
#   tier 1: `nvidia-smi --query-gpu=count` (fast, ~2s)
#   tier 2: `python gpu_probe.py` — torch.cuda.set_device(i) on every GPU
#           (catches cudaErrorDevicesUnavailable that tier 1 misses)
# Bad nodes are persisted in $RUN_DIR/.bad_nodes across requeues, and added
# to the job's ExcNodeList so the next allocation skips them.
# TODO(xiuyul): tier 3 — multi-node NCCL all-reduce probe (catches dead IB links / bad topology before ray bring-up).

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

HEAD_NODE="${GOOD_NODES[0]}"
HEAD_IP=$(getent ahostsv4 "$HEAD_NODE" | awk 'NR==1{print $1}')
WORKER_NODES=( "${GOOD_NODES[@]:1}" )
EXPECTED_GPUS=$(( ${#GOOD_NODES[@]} * 8 ))
echo "[topology] head=$HEAD_NODE ($HEAD_IP)  workers=${WORKER_NODES[*]:-<none>}  gpus=$EXPECTED_GPUS"

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
            DSTATE=$(ps -eo state,user --no-headers 2>/dev/null \
                       | awk -v u="$USER" "\$1==\"D\" && \$2==u" | wc -l)
            GPUMEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
                       | awk "{s+=\$1} END {print s+0}")
            if (( DSTATE > 0 )) || (( GPUMEM > 500 )); then
                echo "[cleanup] POISONED  node=$NODE  dstate=$DSTATE  gpu_used_mb=$GPUMEM"
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

# ---------- shared per-node preamble -------------------------------------

NODE_PREAMBLE='
    set -euo pipefail
    source '"$CONDA_ROOT"'/etc/profile.d/conda.sh
    conda activate '"$MILES_ENV_NAME"'
    PY_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
    export LD_LIBRARY_PATH="$PY_SITE/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH='"$MEGATRON_SRC"'${PYTHONPATH:+:$PYTHONPATH}
    export HF_HOME='"$HF_HOME"'
    export CUDA_DEVICE_MAX_CONNECTIONS=1
    export NCCL_NVLS_ENABLE=1
    export NCCL_DEBUG=WARN
    export PYTHONUNBUFFERED=1
'

# ---------- auto-convert HF -> torch_dist if missing ----------------------
# Runs on the head node only. For large multi-node converts, pre-stage via
# scripts/slurm/setup/convert_checkpoint.sh — see docs/launcher.md.
#
# Recipes opt out by leaving HF_TORCHDIST_DIR unset/empty (e.g. VLM
# recipes that load HF format directly via --megatron-to-hf-mode bridge,
# bypassing Megatron's torch_dist checkpoint flow).

: "${HF_MODEL_DIR:?recipe must define HF_MODEL_DIR}"

if [[ -z "${HF_TORCHDIST_DIR:-}" ]]; then
    echo "[convert] HF_TORCHDIST_DIR empty — recipe opted out of torch_dist convert (loading HF directly)"
elif [[ ! -f "$HF_TORCHDIST_DIR/latest_checkpointed_iteration.txt" ]]; then
    echo "[convert] $HF_TORCHDIST_DIR missing — converting on $HEAD_NODE"
    mkdir -p "$HF_TORCHDIST_DIR"
    MODEL_ARGS_STR=$(printf '%q ' "${MODEL_ARGS[@]}")
    srun --jobid="$JOBID" --overlap --mem=0 -N1 -n1 -w "$HEAD_NODE" \
         --export=ALL,MODEL_ARGS_STR="$MODEL_ARGS_STR",HF_MODEL_DIR="$HF_MODEL_DIR",HF_TORCHDIST_DIR="$HF_TORCHDIST_DIR" \
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

# ---------- ray bring-up --------------------------------------------------

echo "[ray] starting head on $HEAD_NODE"
srun --jobid="$JOBID" --overlap --mem=0 -N1 -n1 -w "$HEAD_NODE" \
     --export=ALL,MASTER_ADDR="$HEAD_IP",RAY_PORT="$RAY_PORT",RAY_DASHBOARD_PORT="$RAY_DASHBOARD_PORT" \
     bash -c "$NODE_PREAMBLE"'
        ray stop --force >/dev/null 2>&1 || true
        ray start --head \
            --node-ip-address "$MASTER_ADDR" \
            --port "$RAY_PORT" \
            --num-gpus 8 \
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

teardown() {
    echo "[teardown] $(date -Is)  stopping ray + killing srun children"
    for n in "${GOOD_NODES[@]}"; do
        srun --jobid="$JOBID" --overlap --mem=0 -N1 -n1 -w "$n" \
             bash -c "$NODE_PREAMBLE"' ray stop --force ' >/dev/null 2>&1 || true
    done
    kill "${RAY_PIDS[@]}" 2>/dev/null || true
}
trap 'echo "[signal] received — clean shutdown then exit"; write_manifest INTERRUPTED; teardown; exit 0' SIGTERM SIGINT
trap teardown EXIT

# ---------- wait for cluster ready ---------------------------------------

echo "[ray] waiting up to ${RAY_BRINGUP_TIMEOUT}s for $EXPECTED_GPUS GPUs"
ELAPSED=0
while (( ELAPSED < RAY_BRINGUP_TIMEOUT )); do
    sleep 10
    ELAPSED=$((ELAPSED + 10))
    OUT=$(srun --jobid="$JOBID" --overlap --mem=0 -N1 -n1 -w "$HEAD_NODE" \
              bash -c "$NODE_PREAMBLE"' ray status --address "'"$HEAD_IP"':'"$RAY_PORT"'" ' 2>/dev/null || true)
    GPU_LINE=$(echo "$OUT" | grep -E "^[[:space:]]*[0-9.]+/[0-9.]+ GPU" | head -1 || true)
    TOTAL=$(echo "$GPU_LINE" | sed -E 's|.*/([0-9]+)(\.[0-9]+)? GPU.*|\1|')
    if [[ -n "${TOTAL:-}" && "$TOTAL" -ge "$EXPECTED_GPUS" ]]; then
        echo "[ray] cluster ready: $GPU_LINE"
        break
    fi
done
if (( ELAPSED >= RAY_BRINGUP_TIMEOUT )); then
    echo "[ray] FATAL: cluster did not assemble in ${RAY_BRINGUP_TIMEOUT}s"
    exit 75
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
