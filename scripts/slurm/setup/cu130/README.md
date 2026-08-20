# Orbit from the RadixArk Miles CUDA 13 image

This installation path treats the published Miles image as a reusable binary
environment and installs only the project-specific Python source overlays.
It is intended for H200 Slurm nodes where Enroot is available.

The expected layering is:

```text
radixark/miles:latest
  CUDA 13 + Python + PyTorch + Triton
  SGLang 0.5.16 native environment
  sglang-kernel + FlashInfer + FlashAttention + Transformer Engine
               |
               +-- external venv with system-site-packages
                     Sphere-Lab SGLang Python/Triton source
                     Sphere-Lab Megatron Python source
                     Orbit source
```

The important rule is: **do not rebuild or replace `sglang-kernel`**. The
Sphere-Lab SGLang fork is based on SGLang 0.5.16 and changes Python and Triton
files, so it should use the native kernel already present in the image. Triton
kernels can JIT-compile on first use and should write to a persistent,
experiment-specific cache.

## Why use this path

A cold Conda installation builds hundreds of CUDA/C++ targets for SGLang,
FlashAttention, Transformer Engine, and related extensions. The Miles image has
already paid that cost. An editable Python overlay should take minutes rather
than hours.

Use a source build instead when the fork:

- changes files under a native C++ or CUDA extension;
- imports a kernel symbol absent from the image's `sglang-kernel`;
- requires a different PyTorch, Python, CUDA, or Triton ABI; or
- fails the image-overlay smoke test below.

## Current base image

The upstream Miles build currently uses these CUDA 13 defaults:

- image: `radixark/miles:latest`
- SGLang base: `lmsysorg/sglang:v0.5.16`
- Miles wheel family: CUDA 13.0, x86-64

`latest` is mutable. It is convenient for the first compatibility experiment,
but production jobs should replace it with the tested image digest:

```text
docker://radixark/miles@sha256:<tested-digest>
```

References:

- <https://github.com/radixark/miles/blob/main/docker/Dockerfile>
- <https://github.com/radixark/miles/blob/main/docker/build.py>
- <https://hub.docker.com/r/radixark/miles/tags>

## 1. Allocate an H200 node

Create a descriptive tmux session on the Slurm login node and request one GPU.
Adjust the partition, memory, and time limit for the cluster if needed.

```bash
tmux new-session -s orbit-miles-cu13-overlay

srun \
  --partition=all \
  --job-name=orbit-miles-cu13-overlay \
  --nodes=1 \
  --ntasks=1 \
  --gres=gpu:1 \
  --cpus-per-task=16 \
  --mem=128G \
  --time=04:00:00 \
  --pty bash -l
```

Confirm that the allocation is on an H200 node before downloading the image:

```bash
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
```

The host driver must be new enough to run the CUDA 13 image. The definitive
check is the in-container PyTorch test in section 6.

## 2. Create isolated image and runtime caches

Use a new run root for every independent experiment. This prevents the test
from silently reusing the Conda installation's uv, Triton, Torch extension, or
Enroot caches.

```bash
export RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_ROOT="/data/home/$USER/.local/state/orbit-miles-cu13/$RUN_ID"

export ENROOT_CACHE_PATH="$RUN_ROOT/cache/enroot"
export ENROOT_DATA_PATH="$RUN_ROOT/enroot-data"
export ENROOT_RUNTIME_PATH="$RUN_ROOT/enroot-runtime"
export ENROOT_TEMP_PATH="$RUN_ROOT/tmp/enroot"

export XDG_CACHE_HOME="$RUN_ROOT/cache"
export PIP_CACHE_DIR="$RUN_ROOT/cache/pip"
export UV_CACHE_DIR="$RUN_ROOT/cache/uv"
export TRITON_CACHE_DIR="$RUN_ROOT/cache/triton"
export TORCH_EXTENSIONS_DIR="$RUN_ROOT/cache/torch-extensions"
export CUDA_CACHE_PATH="$RUN_ROOT/cache/cuda"
export HF_HOME="$RUN_ROOT/cache/huggingface"
export TMPDIR="$RUN_ROOT/tmp"

mkdir -p \
  "$ENROOT_CACHE_PATH" \
  "$ENROOT_DATA_PATH" \
  "$ENROOT_RUNTIME_PATH" \
  "$ENROOT_TEMP_PATH" \
  "$PIP_CACHE_DIR" \
  "$UV_CACHE_DIR" \
  "$TRITON_CACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" \
  "$CUDA_CACHE_PATH" \
  "$HF_HOME" \
  "$TMPDIR" \
  "$RUN_ROOT/sources"
```

Keep the value of `RUN_ROOT`. Re-entering the session with a different
`RUN_ID` creates a different image store and environment.

## 3. Import and create the Miles image

```bash
export MILES_IMAGE="docker://radixark/miles:latest"
export IMAGE_FILE="$RUN_ROOT/radixark-miles-cu13.sqsh"
export CONTAINER_NAME="orbit-miles-cu13-${SLURM_JOB_ID}"

enroot import --output "$IMAGE_FILE" "$MILES_IMAGE"
enroot create --name "$CONTAINER_NAME" "$IMAGE_FILE"
```

Record the resolved image digest from the import output in the run metadata.
Use that digest instead of `latest` after the image passes verification.

## 4. Prepare the source overlays

The following commits are the initial Orbit CUDA installation pins. Update
them only through the project's pinning workflow.

```bash
export SGLANG_REPO="https://github.com/Sphere-AI-Lab/sglang.git"
export SGLANG_COMMIT="51845dc4acca94507ab184b007c8fcfd656b191f"
export SGLANG_SRC="$RUN_ROOT/sources/sglang"

export MEGATRON_REPO="https://github.com/Sphere-AI-Lab/Megatron-LM.git"
export MEGATRON_COMMIT="00eb75b0c803b0fc8e5413d736529d9d3b82b6bd"
export MEGATRON_SRC="$RUN_ROOT/sources/Megatron-LM"

git clone "$SGLANG_REPO" "$SGLANG_SRC"
git -C "$SGLANG_SRC" checkout --detach "$SGLANG_COMMIT"

git clone "$MEGATRON_REPO" "$MEGATRON_SRC"
git -C "$MEGATRON_SRC" checkout --detach "$MEGATRON_COMMIT"

git -C "$SGLANG_SRC" rev-parse HEAD
git -C "$MEGATRON_SRC" rev-parse HEAD
```

Point `ORBIT_SRC` at the Orbit checkout or worktree that should be tested:

```bash
export ORBIT_SRC="/data/home/$USER/miles-orbit/orbit"
test -f "$ORBIT_SRC/pyproject.toml"
```

Do not modify the image's `/sgl-workspace/sglang` checkout. The external
editable install will take precedence over it while leaving the image's native
kernel package intact.

## 5. Create the lightweight overlay environment

Start the image with the run root and Orbit checkout mounted. The venv uses
the image's site packages, so it can see all prebuilt CUDA dependencies.

```bash
export OVERLAY_VENV="$RUN_ROOT/overlay-venv"

enroot start --root \
  --mount "$RUN_ROOT:$RUN_ROOT" \
  --mount "$ORBIT_SRC:$ORBIT_SRC" \
  --env "XDG_CACHE_HOME=$XDG_CACHE_HOME" \
  --env "PIP_CACHE_DIR=$PIP_CACHE_DIR" \
  --env "TRITON_CACHE_DIR=$TRITON_CACHE_DIR" \
  --env "TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR" \
  --env "CUDA_CACHE_PATH=$CUDA_CACHE_PATH" \
  --env "HF_HOME=$HF_HOME" \
  --env "SGLANG_SRC=$SGLANG_SRC" \
  --env "MEGATRON_SRC=$MEGATRON_SRC" \
  --env "ORBIT_SRC=$ORBIT_SRC" \
  --env "OVERLAY_VENV=$OVERLAY_VENV" \
  "$CONTAINER_NAME" bash
```

Inside the container, create the venv and install only editable package
metadata and Python source links:

```bash
python3 -m venv --system-site-packages "$OVERLAY_VENV"

"$OVERLAY_VENV/bin/python" -m pip install \
  --no-deps --no-build-isolation -e "$SGLANG_SRC/python"

"$OVERLAY_VENV/bin/python" -m pip install \
  --no-deps --no-build-isolation -e "$MEGATRON_SRC"

"$OVERLAY_VENV/bin/python" -m pip install \
  --no-deps --no-build-isolation -e "$ORBIT_SRC"
```

Do not install `sglang-kernel`, PyTorch, Triton, FlashInfer, FlashAttention, or
Transformer Engine in this venv. Do not run a dependency-resolving install
without `--no-deps`; doing so can replace the coherent packages supplied by
the image.

If uv is used instead of pip, disable the ambient Orbit project configuration:

```bash
export UV_NO_CONFIG=1
uv pip install --python "$OVERLAY_VENV/bin/python" \
  --no-deps --no-build-isolation -e "$SGLANG_SRC/python"
```

## 6. Verify the binary/source boundary

Run this inside the image with the overlay venv. It verifies that the SGLang
Python package comes from Sphere-Lab while `sgl_kernel` still comes from the
Miles image.

```bash
export PATH="$OVERLAY_VENV/bin:$PATH"

python - <<'PY'
from importlib.metadata import version
from pathlib import Path
import os

import torch
import triton
import sglang
import sgl_kernel
from sglang.srt.server_args import ServerArgs

sglang_root = Path(os.environ["SGLANG_SRC"]).resolve()
sglang_file = Path(sglang.__file__).resolve()
kernel_file = Path(sgl_kernel.__file__).resolve()

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("triton:", triton.__version__)
print("sglang distribution:", version("sglang"))
print("sglang-kernel distribution:", version("sglang-kernel"))
print("sglang source:", sglang_file)
print("sgl_kernel source:", kernel_file)
print("ServerArgs source:", ServerArgs.__module__)

assert sglang_file.is_relative_to(sglang_root), (
    "Sphere-Lab SGLang is not taking precedence",
    sglang_file,
)
assert not kernel_file.is_relative_to(sglang_root), (
    "The source overlay unexpectedly replaced the image native kernel",
    kernel_file,
)
assert torch.cuda.is_available(), "CUDA is not visible inside the image"

print("GPU:", torch.cuda.get_device_name(0))
print("compute capability:", torch.cuda.get_device_capability(0))
x = torch.arange(1024, device="cuda", dtype=torch.float32)
print("CUDA tensor sum:", x.sum().item())
print("image binary + Sphere-Lab source overlay: PASS")
PY
```

An H200 should report compute capability `(9, 0)`. Importing Python modules and
running the simple CUDA tensor test must not start a 510-target native build.

Next, run a small Orbit/SGLang model smoke test that exercises one of the
modified Triton paths. The first run may spend time in Triton JIT compilation;
subsequent runs should reuse `TRITON_CACHE_DIR`.

## 7. Reuse the validated environment

For later jobs, reuse all of the following as one tested unit:

- the digest-pinned Miles SquashFS image;
- the overlay venv;
- the three exact source commits;
- the Triton cache, when the source and runtime pins are unchanged; and
- the activation and mount arguments used during verification.

Do not copy the overlay venv onto a different image digest. A system-site-
packages venv intentionally depends on the image's Python and binary packages.

The fast update loop is then:

```text
start digest-pinned Miles image
  -> mount the existing overlay venv and source checkouts
  -> update only approved Python/Triton commits
  -> refresh editable metadata if needed
  -> rerun the boundary and model smoke tests
```

## Troubleshooting

### `torch.cuda.is_available()` is false

The image is not receiving the allocated GPU, or the host driver cannot run
the image's CUDA runtime. Confirm that the command runs inside the Slurm GPU
allocation and that Enroot's NVIDIA hooks are enabled.

### `sglang` imports from `/sgl-workspace/sglang`

The overlay venv is not active, or the editable SGLang install failed. Check
`which python`, `python -m pip show sglang`, and `sglang.__file__`.

### `sgl_kernel` imports from the Sphere-Lab checkout

Stop. The overlay crossed the intended binary boundary. Remove any native
kernel build or install from the overlay venv and restore the image package.

### Missing symbol in `sgl_kernel`

The Python fork expects a newer native API than the image provides. Either pin
a compatible Python commit or build a new image/binary layer with the required
kernel. Do not patch around a native ABI mismatch with additional pip installs.

### A long CUDA build starts

Cancel that installation command and inspect what requested the build. The
expected editable installs are Python-only and use `--no-deps`; only Triton JIT
compilation during runtime is expected.
