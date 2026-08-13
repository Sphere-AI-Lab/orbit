#!/bin/bash
#
# geo3k-vlm-multi-turn-fully-async-prefetch2-3node-torchdist — IDENTICAL to the
# base prefetch2-3node recipe EXCEPT the trainer loads a pre-converted Megatron
# torch_dist checkpoint instead of reading HF safetensors through the bridge.
#
# Why: the HF->Megatron bridge load is ~24% of time-to-step-0 on this recipe
# (every trainer rank reads the full 17.5 GB safetensors set; the torch_dist
# path reads only per-rank shards). Measured: 44m14s -> 21m44s to step 0.
#
# The artifact is converted ONCE, in BRIDGE mode (raw mode — the stock
# convert_checkpoint.sh default — would build a text-only model with no vision
# tower):
#   srun --gres=gpu:1 --cpus-per-task=16 --mem=128G --time=75 \
#     env MILES_ENV_NAME=miles_0809 MODEL_FAMILY=qwen3-8B \
#         MODEL_ARGS_ROTARY_BASE=5000000 \
#         HF_DIR=/data/shared/hf_cache/models/Qwen3-VL-8B-Instruct \
#         SAVE_DIR=/data/shared/hf_cache/models/Qwen3-VL-8B-Instruct_torch_dist \
#         CONVERT_EXTRA_ARGS="--megatron-to-hf-mode bridge" \
#     bash scripts/slurm/setup/convert_checkpoint.sh
# If the artifact is missing at job start, launch_miles.sbatch auto-converts on
# the head node — flag-correct for this recipe (it serializes MODEL_ARGS, which
# include --megatron-to-hf-mode bridge) — but the other 23 GPUs idle meanwhile,
# so pre-converting is preferred.
#
# --check-weight-update-equal stays ON (inherited from the base MISC_ARGS) and
# doubles as the artifact validation: engines still disk-load the HF weights,
# so the boot snapshot is ground truth, and the post-sync compare bitwise
# verifies trainer torch_dist load -> bridge export -> broadcast against it
# (the Megatron dist loader silently skips missing keys, so this end-to-end
# check is the real gate). Rerun this variant per new artifact/model/bridge
# stack; -torchdist-dummy's after-update check proves broadcast coverage
# only, not artifact content.
#
# Submit:
#   JOB_NAME=geo3k-async-mt-pf2-8b-td TIME=24:00:00 NODES=3 MILES_ENV_NAME=miles_0809 \
#   WANDB_PROJECT=baseline bash scripts/slurm/submit.sh async/geo3k-vlm-multi-turn-fully-async-prefetch2-3node-torchdist

set -euo pipefail

VARIANT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# shellcheck disable=SC1091
source "$VARIANT_DIR/geo3k-vlm-multi-turn-fully-async-prefetch2-3node.sh"

# Pre-converted trainer checkpoint (bridge-built Qwen3-VL, "release" tracker).
# Setting HF_TORCHDIST_DIR arms the launcher's present-check / auto-convert hook.
HF_TORCHDIST_DIR="$HF_CACHE_DIR/models/Qwen3-VL-8B-Instruct_torch_dist"

# argparse last-wins: overrides the base recipe's --load (HF dir). The "release"
# tracker makes Megatron load iteration 0 and skip optimizer/RNG on its own;
# the base --hf-checkpoint stays (engines, tokenizer/processor, bridge provider,
# HF export — config-only reads, no weight tensors).
MILES_ARGS+=( --load "$HF_TORCHDIST_DIR" )
