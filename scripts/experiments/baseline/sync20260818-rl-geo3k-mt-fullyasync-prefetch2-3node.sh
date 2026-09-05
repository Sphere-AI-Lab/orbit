# baseline wrapper — 2026-08-18 sync gate, plain-RL arm (fully-async, prefetch=2).
# Exercises the class-based FullyAsyncRolloutFn port end to end (the sync's
# biggest structural change): --fully-async selection, prefetch derivation,
# fail-closed staleness buffer, weight-version passthrough.
#
# Naming chain: this recipe derives its wandb group from SLURM_JOB_NAME, so the
# full baseline name goes through JOB_NAME (submit.sh) rather than WANDB_RUN_NAME.
export WANDB_PROJECT=${WANDB_PROJECT:-baseline}
export JOB_NAME=${JOB_NAME:-sync20260818-rl-geo3k-mt-fullyasync-prefetch2-3node}
source "$ORBIT_REPO/scripts/experiments/async/geo3k-vlm-multi-turn-fully-async-prefetch2-3node.sh"
