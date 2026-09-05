#!/bin/bash
#
# geo3k-vlm-multi-turn-fully-async-prefetch2-3node-torchdist-dummy — the full
# fast-init config: IDENTICAL to -torchdist (trainer loads the pre-converted
# torch_dist artifact) PLUS engines boot with --sglang-load-format dummy.
#
# Why dummy engines: all 16 engines otherwise each read the full 17.5 GB HF
# checkpoint over the shared FS (132-374 s, FS-weather dependent). With dummy,
# engines build the model structure with random weights in seconds; the initial
# update_weights() broadcast in train_async.py — which ALWAYS runs before the
# first rollout — becomes their real first-time load (~4 s for 16 GB to 16
# engines over NCCL/IB). Even without dummy those disk-loaded weights are
# overwritten by the same broadcast before rollout 0. Measured: 21m44s ->
# 14m29s to step 0 vs -torchdist (44m14s for the base recipe).
#
# The equality check runs in AFTER-UPDATE mode: boot mode's reference is the
# just-booted (here: random) weights, so it cannot work under dummy. This mode
# snapshots the first broadcast, poisons, re-broadcasts, and requires a bitwise
# restore (~17 s) — proving coverage/determinism every run. It cannot catch
# content bugs (wrong artifact / export transform); rerun -torchdist (boot
# mode, disk engines) whenever the artifact, model, or bridge/sglang stack
# changes. Bonus: without boot mode's blocking snapshot, the trainer leg
# overlaps engine bring-up.
#
# Known FT limitation (the base recipe enables --use-fault-tolerance): a
# restarted engine re-enters routing with random weights and serves requests
# until the next update_weights broadcast reaches it (previously: stale-but-
# real weights). Trajectories sampled in that window train with ordinary
# group-normalized advantages (negative when groupmates score higher, not
# zero), i.e. a bounded perturbation, not a no-op. Gating recovered engines
# out of routing until their first sync lands is future work.
#
# Submit (artifact must exist — see -torchdist header for the one-off convert):
#   JOB_NAME=geo3k-async-mt-pf2-8b-td-dummy TIME=24:00:00 NODES=3 ORBIT_ENV_NAME=orbit_0809 \
#   WANDB_PROJECT=baseline bash scripts/slurm/submit.sh async/geo3k-vlm-multi-turn-fully-async-prefetch2-3node-torchdist-dummy

set -euo pipefail

VARIANT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck disable=SC1091
source "$VARIANT_DIR/geo3k-vlm-multi-turn-fully-async-prefetch2-3node-torchdist.sh"

# Engines skip the 16x HF disk read (the pre-rollout broadcast is their load);
# the inherited --check-weight-update-equal anchors on the first broadcast
# instead of the (random) boot weights — see header.
ORBIT_ARGS+=(
   --sglang-load-format dummy
   --check-weight-update-equal-mode after-update
)
