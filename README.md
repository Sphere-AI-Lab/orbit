<div align="center">
  <img src="assets/orbit_logo.png" alt="Orbit" width="500"/>
</div>

# Orbit

A lightweight, ultra-scale RL infrastructure framework for post-training on trillions of parameters. 

## Installation

Orbit's release environment targets Python 3.12 and CUDA 13.2. The public
launchers and helper scripts support only this CUDA 13.2 runtime path. See
[docs/CUDA-13-install.md](docs/CUDA-13-install.md) for the supported install
flow. See [docs/troubleshooting.md](docs/troubleshooting.md) for common
resolver, import, and launcher smoke failures. The interim local release expects
sibling backend checkouts next to Orbit:

```bash
<workspace>/orbit
<workspace>/Megatron-Bridge
<workspace>/Megatron-Bridge/3rdparty/Megatron-LM
<workspace>/sglang
```

Keep those checkouts at the refs recorded in `pyproject.toml` under
`tool.orbit.release.backend-pins`. Orbit's `tool.uv.sources` uses local path
sources for those repos now; when the repos are public, replace the paths with
public Git URLs and the same `rev` values.

Release maintainers can verify a public clean-room install with
`scripts/release/clean_room_gate.sh` after setting `PUBLIC_ORBIT_URL`. This gate
is for the future public Git-ref release after backend repositories are uploaded;
it is not expected to pass against the interim local-path backend sources.

Install the CUDA-specific Torch layer and compiled extensions from the CUDA 13.2
guide into the Orbit environment, then sync Orbit and the backend sources:

```bash
cd orbit
uv python pin 3.12
uv venv
source .venv/bin/activate
# install the CUDA/Torch layer from docs/CUDA-13-install.md
uv sync --inexact
```

The manifest intentionally prevents transitive dependencies from selecting an
untested Torch build during `uv sync`; `uv sync --inexact` also avoids pruning
the CUDA/Torch packages installed by the CUDA guide.

## Active Entry Points

- `train.py`: synchronous training driver.
- `train_async.py`: asynchronous training driver. For async PEFT adapter
  double buffering, see `examples/README.md`.
- `examples/high_precision/`: BF16 and high-precision training launchers.
- `examples/low_precision/`: int4, fp8, and nvfp4 training launchers.
- `scripts/conversion/`: checkpoint conversion entrypoints.
- `tools/check_*_parity.py`: checkpoint and runtime parity checks.

## Launcher Contract

Launchers are independent bash entrypoints. They own model-specific defaults and
source shared helpers from `scripts/lib/` for CUDA setup, private Ray lifecycle,
W&B handling, eval toggles, checkpoint preflight, and common argument assembly.

PEFT KL launchers compute reference log-probs with the loaded model while
adapters are disabled. They pass `--load` and do not require `--ref-load` or
separate reference workers.

## Useful Smoke Settings

Use these environment variables to shrink a launcher for command-path smoke
tests:

```bash
NUM_ROLLOUT=1 TOTAL_EPOCHS=1 TRAIN_ROWS=1 \
ROLLOUT_BATCH_SIZE=1 N_SAMPLES_PER_PROMPT=1 GLOBAL_BATCH_SIZE=1 \
DISABLE_EVAL=1 ENABLE_WANDB=0 \
bash examples/high_precision/run-qwen3-4b-instruct-2507-bf16-math-oft.sh
```

## Acknowledgements

Orbit is built upon the excellent work of the following projects:

- [verl](https://github.com/volcengine/verl)
- [slime](https://github.com/THUDM/slime)
- [MILES](https://github.com/radixark/miles)
- [SGLang](https://github.com/sgl-project/sglang)
- [Megatron-Bridge](https://github.com/NVIDIA/Megatron-Bridge)
