# Reusable CUDA 12.8 / H200 binary layer

> **Status:** Experimental until a cold build is archived, unpacked into a second
> clean prefix, and passes `verify_env.py --full-h200` there.

The normal CUDA 12.8 installer builds every native dependency from its pinned
source. That is the source-of-truth qualification path, but repeating it for every
Orbit checkout wastes hours on CUDA extensions that have not changed. This guide
describes a two-layer workflow similar to starting Miles-IMP from a built image:

```text
pins.env + exact native source commits
                  |
                  v
       cold Conda/uv installation
                  |
                  v
      relocatable Conda binary archive
                  |
                  v
  cheap SGLang/Megatron/Orbit source overlay
                  |
                  v
       verify_env.py --full-h200
```

The archive is a derived artifact. `pins.env`, `install_env.sh`, and the pinned
source repositories remain authoritative.

## Reuse boundary

The binary layer may reuse components whose Python, Torch, CUDA, and C++ ABI
identity is unchanged:

| Layer | Contents |
|---|---|
| Binary base | Python 3.12, Torch 2.11+cu128, Triton, CUDA runtime wheels, cuDNN, NCCL, FlashInfer, Transformer Engine, FlashAttention, DeepEP, Apex, and other compiled dependencies |
| Source overlay | Sphere-Lab SGLang Python code, Sphere-Lab Megatron-LM, Megatron-Bridge, and Orbit |
| Conditional | `sglang-kernel` is reusable only when its source tree and Torch/CUDA ABI are unchanged; rebuild it when the kernel tree changes |

Megatron-LM, Megatron-Bridge, and most of SGLang are Python-level code, so changing
those commits normally requires only a new source overlay. Do not infer native
compatibility from a package version alone. Key an archive by at least:

- Python major/minor version.
- CUDA profile and target architecture.
- Exact Torch build and Triton version.
- Exact commits or source-tree hashes for every compiled package.
- cuDNN and NCCL versions.

## Independent cold build

Use new environment, source, and cache paths. The builder must not read another
Orbit environment or its package caches; otherwise it does not prove that the
artifact is reproducible.

```bash
export BUILD_ID=<unique-build-id>
export ENV_PREFIX=/data/home/zeju/miles-orbit/envs/orbit_cu128_binary_${BUILD_ID}
export SOURCE_ROOT=/data/home/zeju/miles-orbit/sources/orbit_cu128_binary_${BUILD_ID}
export CACHE_ROOT=$HOME/.cache/orbit-cu128-binary/${BUILD_ID}

mkdir -p \
  "$CACHE_ROOT/conda-pkgs" \
  "$CACHE_ROOT/pip" \
  "$CACHE_ROOT/uv" \
  "$CACHE_ROOT/ccache" \
  "$CACHE_ROOT/tmp"

export CONDA_PKGS_DIRS="$CACHE_ROOT/conda-pkgs"
export PIP_CACHE_DIR="$CACHE_ROOT/pip"
export UV_CACHE_DIR="$CACHE_ROOT/uv"
export CCACHE_DIR="$CACHE_ROOT/ccache"
export TMPDIR="$CACHE_ROOT/tmp"
export CUDA_HOME=/usr/local/cuda-12.8
export SGLANG_BUILD_RUST_EXTS=none

scripts/slurm/setup/cu128/install_env.sh \
  --env-prefix "$ENV_PREFIX" \
  --source-root "$SOURCE_ROOT" \
  --jobs 32
```

`SGLANG_BUILD_RUST_EXTS=none` prevents the SGLang Python overlay from rebuilding
the Rust router. The CUDA 12.8 profile installs its separately pinned router wheel.

If `ccache` is available, also set:

```bash
export CMAKE_C_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
```

An empty compiler cache does not make the first build faster. It makes later native
rebuilds faster. The relocatable archive is what avoids compilation entirely when
the binary identity has not changed.

## Archive and hydrate

After the installer passes full H200 verification, record the resolved packages,
source commits, input checksums, and archive checksum. A typical archive step is:

```bash
python -m pip install 'conda-pack==0.8.1'
conda-pack \
  --prefix "$ENV_PREFIX" \
  --output orbit-cu128-h200-binary-layer.tar.gz \
  --format tar.gz \
  --compress-level 1 \
  --ignore-editable-packages
sha256sum orbit-cu128-h200-binary-layer.tar.gz \
  > orbit-cu128-h200-binary-layer.tar.gz.sha256
```

Hydrate into a new prefix, then reapply the cheap source overlay because editable
package metadata from the build prefix is not portable:

```bash
export TARGET_PREFIX=/data/home/zeju/miles-orbit/envs/orbit_cu128_from_binary_<id>
mkdir -p "$TARGET_PREFIX"
tar -xzf orbit-cu128-h200-binary-layer.tar.gz -C "$TARGET_PREFIX"
"$TARGET_PREFIX/bin/conda-unpack"

"$TARGET_PREFIX/bin/uv" pip install \
  --python "$TARGET_PREFIX/bin/python" \
  --no-deps \
  --editable "$SOURCE_ROOT/sglang/python" \
  --editable "$SOURCE_ROOT/Megatron-LM" \
  --editable "$SOURCE_ROOT/Megatron-Bridge" \
  --editable "$PWD"

"$TARGET_PREFIX/bin/python" scripts/slurm/setup/cu128/verify_env.py \
  --source-root "$SOURCE_ROOT" \
  --full-h200
```

Do not publish or reuse the archive until this second-prefix verification passes.

## FlashInfer `nvep` and CUDA Python

FlashInfer 0.6.14 declares the following optional dependency in its upstream
[`pyproject.toml`](https://github.com/flashinfer-ai/flashinfer/blob/v0.6.14/pyproject.toml):

```toml
[project.optional-dependencies]
nvep = [
    "cuda-python>=13.0",
]
```

This requirement is active only for `flashinfer-python[nvep]`. The CUDA 12.8
profile installs plain `flashinfer-python==0.6.14` and must not enable the `nvep`
extra. The selected CUTLASS CUDA 12 packages require `cuda-python>=12.8`, which is
satisfied by the profile pin `cuda-python==12.9.2`.

Orbit's root `pyproject.toml` contains a CUDA 13.2 project-level override:

```toml
[tool.uv]
override-dependencies = [
    "cuda-python==13.2.0",
]
```

`uv pip install` discovers that configuration from the current working directory.
Without isolation, running the CUDA 12.8 installer from the Orbit worktree therefore
combines the project override with the profile's explicit `cuda-python==12.9.2` and
fails resolution. The CUDA 12.8 installer exports `UV_NO_CONFIG=1`; its generated
pins and explicit override file are authoritative for every uv invocation. A manual
resolver command launched outside the worktree can hide this problem because no
Orbit project configuration is discovered there.

When diagnosing a similar report, preserve the pins and inspect the actual graph:

```bash
UV_NO_CONFIG=1 uv pip install --dry-run -vv <the exact original arguments>
```

Confirm whether an extra such as `[nvep]` is active and inspect the wheel's
`*.dist-info/METADATA` before changing CUDA Python, Torch, or CUDA runtime pins.

## Qualification evidence

A reusable layer is qualified only when its provenance records:

- The exact Orbit revision and `pins.env` checksum.
- Every external source commit used by the build.
- The complete resolved Python package list.
- The archive SHA-256 checksum.
- The cold-build H200 verification result.
- The second-prefix archive hydration and H200 verification result.

Keep raw logs and archives in the durable run store or artifact storage, not in Git.
