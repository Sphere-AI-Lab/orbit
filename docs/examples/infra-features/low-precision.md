---
title: "Low-Precision Launchers"
description: "RL on quantized checkpoints (INT4/NVFP4/FP8/MXFP4) with OFT."
# Generated from examples/infra_features/low_precision/README.md by scripts/tools/sync_example_docs.py. Edit that README, not this file.
---
These launchers run int4, fp8, and nvfp4 Orbit training recipes. They keep
precision-specific checkpoint and parity defaults in the entrypoint and share
common Ray, eval, W&B, and argument assembly through `scripts/lib/`.

Set `SKIP_INT4_CHECKPOINT_PREFLIGHT=1`, `SKIP_FP8_CHECKPOINT_PREFLIGHT=1`, or
`SKIP_NVFP4_CHECKPOINT_PREFLIGHT=1` only for command-path smoke checks.
