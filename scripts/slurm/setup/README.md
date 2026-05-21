# `scripts/slurm/setup/` — one-time install of the `miles` conda env

Two scripts here, run once per account on a GPU-visible compute node.
If you are starting from a login node, use an interactive 1-GPU `salloc`.

| Script | When to run |
|---|---|
| `install_env.sh` | Always — builds the `miles` conda env from source |
| `convert_checkpoint.sh` | Optional — pre-convert a model to skip the auto-convert at launch time (Qwen3-4B-class converts happen automatically inside the launcher) |

## Build

```bash
# Optional when already on a GPU-visible compute node:
salloc --gres=gpu:1 --cpus-per-task=16 --mem=128G --time=2:00:00 --pty bash
cd /data/home/$USER/workspace/miles-imp
bash scripts/slurm/setup/install_env.sh
```

Safe to re-run; `uv` / pip reuse installed artifacts where they can.
Source of truth for everything the env contains: the script itself
(`install_env.sh`) and the official `docker/Dockerfile` it mirrors —
this README intentionally does **not** duplicate the install list.

## Knobs (env vars, all optional)

| Var | Default | What |
|---|---|---|
| `MILES_ENV_NAME` | `miles` | conda env name |
| `MILES_PY_VERSION` | `3.12` | python version |
| `MILES_REPO` | `$PWD` | this repo |
| `THIRDPARTY_DIR` | `$MILES_REPO/thirdparty` | submodule dir |
| `PULL_REMOTE` | `0` | set to `1` to `git submodule update --remote` after init |
| `CUDA_HOME` | auto (`/usr/local/cuda-12.{8,9}` / `/usr/local/cuda`) | override the CUDA toolkit path used for source builds |
| `TORCH_VERSION` | `2.9.1` | matches `thirdparty/sglang`'s pin |
| `TORCH_INDEX_URL` | `https://download.pytorch.org/whl/cu129` | pytorch wheel index |
| `TE_VERSION` | `2.10.0` | Dockerfile pin |
| `MBRIDGE_COMMIT` / `TMS_COMMIT` | (Dockerfile pins) | git commits |
| `FLASHINFER_INDEX_URL` | `https://flashinfer.ai/whl/cu129` | extra index for flashinfer |
| `INSTALL_FLASH_ATTN` / `INSTALL_FLASH_ATTN_3` / `INSTALL_APEX` | `1` | toggle each prebuilt wheel |
| `MILES_WHEELS_REPO` / `MILES_WHEELS_TAG` | `yueming-yuan/miles-wheels` / `cu129-x86_64` | prebuilt-wheel source |
| `WHEELS_DIR` | `$THIRDPARTY_DIR/wheels` | local wheel cache (gitignored) |
| `CUDNN_CU12_VERSION` | `9.16.0.29` | cudnn pin (pytorch/pytorch#168167 workaround) |
| `ALLOW_CUDA_MINOR_FORWARD_COMPAT` | `1` | set to `0` to hard-fail if the driver's CUDA minor < wheel CUDA minor |

## Convert HF → Megatron `torch_dist` (optional)

`launch_miles.sbatch` auto-converts on the head node before training
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
  upstream miles install reference. `install_env.sh` mirrors the Dockerfile's
  CUDA-12 / H100 path and intentionally rejects the CUDA-13 / Blackwell
  variant (cu13, TE 2.12, cudnn-cu13) — use the Dockerfile directly for that.
- [`../docs/launcher.md`](../docs/launcher.md) — design notes for the slurm
  launcher itself (separate from the install).
- [`verify_env.py`](verify_env.py) — re-runs the install smoke test against
  the current env (`python scripts/slurm/setup/verify_env.py`).
- [`extract_pins.py`](extract_pins.py) + [`pins.env`](pins.env) — version pins
  extracted from the Dockerfile, sourced by `install_env.sh`. Regenerate via
  `python scripts/slurm/setup/extract_pins.py --write`.
- [`track_submodules.py`](track_submodules.py) — show pinned vs
  `origin/<branch>` commit deltas for `thirdparty/{Megatron-LM,sglang,Megatron-Bridge}`.
