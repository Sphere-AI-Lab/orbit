# Low-Precision Launchers

These launchers run int4, fp8, and nvfp4 Orbit training recipes. They keep
precision-specific checkpoint and parity defaults in the entrypoint and share
common Ray, eval, W&B, and argument assembly through `scripts/lib/`.

Set `SKIP_INT4_CHECKPOINT_PREFLIGHT=1`, `SKIP_FP8_CHECKPOINT_PREFLIGHT=1`, or
`SKIP_NVFP4_CHECKPOINT_PREFLIGHT=1` only for command-path smoke checks.
