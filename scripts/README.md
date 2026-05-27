# Orbit Scripts

The `scripts/` tree contains active Orbit entrypoints.

- `lib/`: shared shell helpers used by launchers.
- `conversion/`: checkpoint conversion wrappers.

Training launchers themselves live under `examples/high_precision/` and
`examples/low_precision/`.

All launchers support environment overrides for data paths, checkpoints, Ray
resources, W&B, eval, and smoke-test batch sizes.

`scripts/lib/launcher.sh` is the shared training launcher orchestrator.
Focused helpers keep the mechanics separate:

- `common.sh`: shell utility predicates and process-env setup.
- `tool_env.sh`: CUDA modules, PATH/LD_LIBRARY_PATH, PYTHONPATH, no_proxy. Sourced by leaf training launchers and standalone tool/parity/conversion scripts.
- `paths.sh`: path validation, checkpoint resolution, and staging.
- `wandb.sh`: W&B key loading and W&B CLI args.
- `preflight.sh`: precision-specific parity preflights.
- `megatron.sh`, `sglang.sh`, `peft.sh`, `rollout.sh`, `rl.sh`, `train.sh`: per-domain defaults and argument arrays.
- `ray.sh`: Ray lifecycle, private port reservation, and the standalone Ray-job entrypoint.
- `driver.sh`: Python driver shim.

Low-precision parity checks are retained as Python CLIs under `tools/`:
`tools/check_fp8_checkpoint_parity.py`,
`tools/check_int4_checkpoint_parity.py`,
`tools/check_nvfp4_checkpoint_parity.py`,
`tools/check_fp8_runtime_parity.py`,
`tools/check_int4_runtime_parity.py`, and
`tools/check_nvfp4_runtime_parity.py`. `scripts/lib/preflight.sh` calls these
tools directly for precision-specific preflight checks.
