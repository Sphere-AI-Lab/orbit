#!/bin/bash
#
# submit.sh — slurm-side entry: pick an experiment, download its assets,
# call sbatch.
#
# This is the only slurm script meant to be run by hand. Everything below
# (launch_orbit.sbatch, the per-node srun calls) is invoked by slurm itself.
#
# Usage:
#   bash scripts/slurm/submit.sh <experiment-name>
#
#   <experiment-name> matches a file in scripts/experiments/<name>.sh
#
# Example:
#   bash scripts/slurm/submit.sh qwen3-4B-disagg-2node
#
# Env-var overrides (all optional):
#   NODES        # overrides EXPERIMENT_NODES from the recipe
#   TIME         # overrides EXPERIMENT_TIME
#   JOB_NAME     # overrides the slurm job name (defaults to experiment name)
#   SBATCH_EXTRA # extra args spliced into sbatch, e.g. "--exclude=slinky-15"
#
# Passed straight through to the run via sbatch --export=ALL (not parsed here):
#   SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=true
#                # REQUIRED for the v0.5.12 sync on the torch-2.9.1 env — the
#                # sglang engine asserts sglang-kernel>=0.4.2.post2 at launch,
#                # which the (working) 0.4.1 kernel fails. See docs/launcher.md
#                # "Running the v0.5.12 sync".

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ORBIT_REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Scrub any inherited SLURM_* env. If submit.sh is run from inside an
# salloc/srun, `sbatch --export=ALL` below would otherwise leak the caller's
# allocation (SLURM_JOB_ID / SLURM_NODELIST / SLURM_*_GPUS / ...) into the new
# batch job. That poisons launch_orbit.sbatch's healthcheck `srun --jobid=...
# --overlap` step, which then fails with "Requested node configuration is not
# available" and false-marks every allocated node BAD at startup.
unset $(compgen -v | grep '^SLURM_') 2>/dev/null || true

if [[ $# -ne 1 ]]; then
    echo "Usage: bash $0 <experiment-name>" >&2
    echo "Available experiments:" >&2
    ls "$ORBIT_REPO/scripts/experiments/" | sed 's/\.sh$//' | sed 's/^/  /' >&2
    exit 64   # EX_USAGE
fi
EXP_NAME=$1
RECIPE="$ORBIT_REPO/scripts/experiments/$EXP_NAME.sh"
if [[ ! -f "$RECIPE" ]]; then
    echo "FATAL: experiment '$EXP_NAME' not found at $RECIPE" >&2
    exit 66   # EX_NOINPUT
fi

# ---------- credentials + conda (for the `hf` CLI + asset paths) ----------

[ -r "$HOME/.config/secrets.env" ] && . "$HOME/.config/secrets.env"
: "${HF_TOKEN:?HF_TOKEN not set — check ~/.config/secrets.env}"
: "${WANDB_API_KEY:?WANDB_API_KEY not set — check ~/.config/secrets.env}"
export HF_TOKEN WANDB_API_KEY

CONDA_ROOT=${CONDA_ROOT:-/data/shared/conda/miniconda3}
ORBIT_ENV_NAME=${ORBIT_ENV_NAME:-orbit}
# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ORBIT_ENV_NAME"

# ---------- source the recipe to read metadata + asset list ---------------

# shellcheck disable=SC1090
source "$RECIPE"

[[ -n "${ORBIT_ARGS+x}" ]]       || { echo "FATAL: $RECIPE did not define ORBIT_ARGS" >&2; exit 78; }
[[ -n "${HF_MODEL_REPO+x}" ]]    || { echo "FATAL: $RECIPE did not define HF_MODEL_REPO" >&2; exit 78; }
[[ -n "${EXPERIMENT_NODES+x}" ]] || { echo "FATAL: $RECIPE did not define EXPERIMENT_NODES" >&2; exit 78; }

# ---------- download assets if missing (idempotent) -----------------------

mkdir -p "$HF_CACHE_DIR/models" "$HF_CACHE_DIR/data"

if [ ! -f "$HF_MODEL_DIR/config.json" ]; then
    echo "[assets] downloading $HF_MODEL_REPO -> $HF_MODEL_DIR"
    hf download "$HF_MODEL_REPO" --local-dir "$HF_MODEL_DIR"
else
    echo "[assets] model present: $HF_MODEL_DIR"
fi

for repo in "${HF_DATASETS[@]}"; do
    name=$(basename "${repo,,}")
    dest="$HF_CACHE_DIR/data/$name"
    if [ ! -d "$dest" ] || [ -z "$(ls -A "$dest" 2>/dev/null)" ]; then
        echo "[assets] downloading dataset $repo -> $dest"
        hf download --repo-type dataset "$repo" --local-dir "$dest"
    else
        echo "[assets] dataset present: $dest"
    fi
done

# If the Megatron torch_dist artifact is missing, launch_orbit.sbatch will
# auto-convert on the head node before training starts (~5 min for Qwen3-4B).
# For large multi-node models, pre-convert via scripts/slurm/setup/convert_checkpoint.sh.

# ---------- sbatch --------------------------------------------------------

JOB_NAME=${JOB_NAME:-$EXP_NAME}
NODES=${NODES:-$EXPERIMENT_NODES}
TIME=${TIME:-$EXPERIMENT_TIME}
SBATCH_EXTRA=${SBATCH_EXTRA:-}

# Per-launch dir, named by wall-clock submit time. Created up-front so slurm
# can write --output directly into it (no symlink, no tee).
RUN_STAMP=$(date +%y%m%d_%H%M%S)
RUN_DIR="$ORBIT_REPO/runs/$JOB_NAME/$RUN_STAMP"
mkdir -p "$RUN_DIR"

# Warn if any of the last 3 runs for this job_name ended in a non-success state.
# shellcheck disable=SC1091
source "$ORBIT_REPO/scripts/slurm/lib/manifest.sh"
read_recent_manifests "$ORBIT_REPO/runs/$JOB_NAME" 3 "$RUN_STAMP"

echo "[sbatch] $EXP_NAME  nodes=$NODES  time=$TIME  job-name=$JOB_NAME  run_dir=$RUN_DIR"
exec sbatch \
    --nodes="$NODES" --time="$TIME" \
    --job-name="$JOB_NAME" \
    --output="$RUN_DIR/run.log" \
    --export=ALL,ORBIT_REPO="$ORBIT_REPO",RECIPE="$RECIPE",RUN_DIR="$RUN_DIR" \
    $SBATCH_EXTRA \
    "$ORBIT_REPO/scripts/slurm/launch_orbit.sbatch"
