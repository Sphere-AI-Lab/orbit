# baseline wrapper — 2026-08-18 sync gate, OPD arm.
# Workload is single-sourced from the standing OPD baseline recipe; this
# wrapper only pins the baseline naming and wandb destination (M3TRL/baseline).
export WANDB_PROJECT=${WANDB_PROJECT:-baseline}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-sync20260818-opd-geo3k-mm-mt-fullyasync-200step}
export JOB_NAME=${JOB_NAME:-sync20260818-opd-baseline}
source "$ORBIT_REPO/scripts/experiments/OPD/multimodal/baseline/baseline-geo3k-multimodal-multiturn-fully-async-200step.sh"
