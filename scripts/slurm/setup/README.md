# `scripts/slurm/setup/` — one-time install of the `orbit` conda env

Two scripts here, run once per account on a GPU-visible compute node.
If you are starting from a login node, use an interactive 1-GPU `salloc`.

| Script | When to run |
|---|---|
| `install_env.sh` | Always — builds the `orbit` conda env from source |
| `convert_checkpoint.sh` | Optional — pre-convert a model to skip the auto-convert at launch time (Qwen3-4B-class converts happen automatically inside the launcher) |

## Build

```bash
# Optional when already on a GPU-visible compute node (`--pty` is an srun option, not salloc's):
salloc --gres=gpu:1 --cpus-per-task=16 --mem=128G --time=4:00:00
srun --pty bash
cd /data/home/$USER/workspace/orbit
# CONDA_ENVS_DIRS (a conda setting) places the env somewhere other than $CONDA_ROOT/envs:
CONDA_ENVS_DIRS=/data/home/$USER/workspace/envs bash scripts/slurm/setup/install_env.sh
```

Safe to re-run; `uv` / pip reuse installed artifacts where they can.
Source of truth for everything the env contains: the script itself
(`install_env.sh`) and the official `docker/Dockerfile` it mirrors —
this README intentionally does **not** duplicate the install list.

## Knobs (env vars, all optional)

| Var | Default | What |
|---|---|---|
| `ORBIT_ENV_NAME` | `orbit` | conda env name |
| `ORBIT_PY_VERSION` | `3.12` | python version |
| `ORBIT_REPO` | `$PWD` | this repo |
| `THIRDPARTY_DIR` | `$ORBIT_REPO/thirdparty` | submodule dir |
| `PULL_REMOTE` | `0` | set to `1` to `git submodule update --remote` after init |
| `CUDA_HOME` | auto (`/usr/local/cuda-13.0` / `/usr/local/cuda-13` / `/usr/local/cuda`) | CUDA-13 toolkit for torch_memory_saver's extension build and flashinfer JIT |
| `TORCH_VERSION` | `pins.env` (extracted from `thirdparty/sglang`) | must equal the submodule's `torch==` pin |
| `TORCH_INDEX_URL` | derived from `MILES_WHEELS_TAG` (`cu130` → `.../whl/cu130`) | pytorch wheel index |
| `TE_VERSION` | `pins.env` (Dockerfile pin, `2.17.0`) | TE version; the whole triplet comes prebuilt from the wheels bundle |
| `MBRIDGE_COMMIT` / `TMS_COMMIT` | (Dockerfile pins) | git commits |
| `FLASHINFER_INDEX_URL` | derived from `MILES_WHEELS_TAG` (`https://flashinfer.ai/whl/cu130`) | extra index for flashinfer |
| `INSTALL_FLASH_ATTN` / `INSTALL_FLASH_ATTN_3` / `INSTALL_APEX` | `1` | toggle each prebuilt wheel |
| `MILES_WHEELS_REPO` / `MILES_WHEELS_TAG` | `yueming-yuan/miles-wheels` / `cu130-torch213-x86_64` | prebuilt-wheel source; must be a CUDA-13 (`cu130*`) bundle built for the submodule's torch |
| `WHEELS_DIR` | `$THIRDPARTY_DIR/wheels` | local wheel cache (gitignored) |
| `CUDNN_VERSION` | `pins.env` `CUDNN_CU13_VERSION` (Dockerfile pin) | `nvidia-cudnn-cu13` wheel version |
| `ALLOW_CUDA_MINOR_FORWARD_COMPAT` | `1` | set to `0` to hard-fail if the driver's CUDA minor < wheel CUDA minor |

## Convert HF → Megatron `torch_dist` (optional)

`launch_orbit.sbatch` auto-converts on the head node before training
for any model where the torch_dist artifact is missing. Skip this
section unless you want to pre-stage a large model:

```bash
# Optional when already on a GPU-visible compute node:
salloc --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=30 --pty bash
bash scripts/slurm/setup/convert_checkpoint.sh                # defaults: qwen3-4B
# different family:
MODEL_FAMILY=deepseek-v3 HF_DIR=... SAVE_DIR=... \
    bash scripts/slurm/setup/convert_checkpoint.sh
# multi-node convert (large MoE) — wrap the python call in torchrun;
# see docs/getting-started/quick-start.md step 3 for the pattern.
```

Idempotent (checks `latest_checkpointed_iteration.txt`).

## See also

- [`install_env.sh`](install_env.sh) — authoritative list of what gets installed,
  with inline rationale next to each pip command. Read this if the
  knob table above doesn't answer your question.
- [`docs/getting-started/installation.md`](../../../docs/getting-started/installation.md)
  + [`docker/Dockerfile`](../../../docker/Dockerfile) in the repo root — the
  upstream orbit install reference. `install_env.sh` mirrors the Dockerfile's
  `ENABLE_CUDA_13=1` path only (`cu130*` wheels tag: prebuilt TE triplet +
  `docker/patch/cu13`, `nvidia-cudnn-cu13`); the CUDA-12 path was retired when
  `thirdparty/sglang` moved to torch 2.13.
- [`../docs/launcher.md`](../docs/launcher.md) — design notes for the slurm
  launcher itself (separate from the install).
- [`verify_env.py`](verify_env.py) — re-runs the install smoke test against
  the current env (`python scripts/slurm/setup/verify_env.py`).
- [`extract_pins.py`](extract_pins.py) + [`pins.env`](pins.env) — version pins
  extracted from the Dockerfile, sourced by `install_env.sh`. Regenerate via
  `python scripts/slurm/setup/extract_pins.py --write`.
- [`track_submodules.py`](track_submodules.py) — show pinned vs
  `origin/<branch>` commit deltas for `thirdparty/{Megatron-LM,sglang,Megatron-Bridge}`.
