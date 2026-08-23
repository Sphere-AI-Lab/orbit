# Orbit native CUDA 13 / H100 or B200 environment

This profile follows the Miles-IMP installation model. The RadixArk Miles
Dockerfile is a pinned dependency recipe, not a runtime container. Its prebuilt
CUDA 13 wheels are installed into a persistent Conda environment, then Orbit's
exact Sphere-Lab sources are installed as editable overlays.

## Source-of-truth flow

~~~text
radixark/miles docker/Dockerfile at a pinned commit
                 +
orbit/pyproject.toml
                 |
                 v
        extract_pins.py --write
                 |
                 v
             pins.env
                 |
                 v
         install_env.sh
                 |
                 v
          verify_env.py
~~~

No Docker, Enroot, or Apptainer runtime is involved.

## What is prebuilt

The primary path does not compile CUDA extensions. It installs:

- PyTorch 2.11 and Triton 3.6 for CUDA 13
- official sglang-kernel 0.4.5+cu130
- official sgl-deep-gemm
- FlashInfer CUDA 13 components
- FlashAttention 2 and 3 from the Miles wheel release
- Transformer Engine and Apex from the Miles wheel release
- additional Miles release wheels when available

A required missing wheel is an error. The installer does not silently fall back
to a multi-hour CUDA build.

## What is editable

These repositories are checked out at exact commits under the source root:

- Sphere-Lab SGLang
- Sphere-Lab Megatron-LM
- Sphere-Lab Megatron-Bridge

They and the current Orbit checkout are installed editable with --no-deps. That
keeps Python and Triton changes live while preserving the prebuilt binary layer.

## Paths

~~~text
Environment:
/fast/zqiu/orbit-iclr/orbit/envs/orbit-cu130-v1

Sources:
/fast/zqiu/orbit-iclr/orbit/sources/orbit-cu130-v1

Cache:
/fast/zqiu/orbit-iclr/orbit/cache/orbit-cu130-v1
~~~

These directories must be excluded from Git. The installer never resets an
existing source checkout or overwrites an unknown environment.

The uv cache defaults to cluster home (see "MPI cache placement"); for the
fastest extraction point `UV_CACHE_DIR` at node-local disk (for example under
`/tmp`). Either way the installer symlinks site-packages into the cache while it
runs and then copies everything into the prefix (`materialize_env.py`, parallel),
so the finished environment depends on neither the cache nor the node.

## Refresh or audit pins

From the Orbit repository root:

~~~bash
python scripts/slurm/setup/cu130/extract_pins.py --write
python scripts/slurm/setup/cu130/extract_pins.py --check
~~~

The RadixArk commit is embedded in extract_pins.py, so regeneration does not
silently follow a moving main branch.

## Inspect the installation plan

Dry-run mode performs no network access and creates no files:

~~~bash
scripts/slurm/setup/cu130/install_env.sh --dry-run
~~~

## Install inside an H100 or B200 allocation

~~~bash
scripts/slurm/setup/cu130/install_env.sh
~~~

The default command creates or resumes the paths above. Separate paths can be
provided with --env-prefix, --source-root, and --cache-dir.

## Activate

~~~bash
source /home/zqiu/anaconda3/etc/profile.d/conda.sh
conda activate /fast/zqiu/orbit-iclr/orbit/envs/orbit-cu130-v1
~~~

The editable source links are part of the environment. No extra PYTHONPATH is
normally needed. Before running launchers, load Orbit's runtime loader with the
prefix as `ORBIT_VENV`; it adds the `z3/lib` path that `megatron.bridge`
(via nvidia-modelopt) needs and the cuDNN/FlashInfer runtime settings:

~~~bash
ORBIT_VENV=/fast/zqiu/orbit-iclr/orbit/envs/orbit-cu130-v1 \
  source examples/load_cuda13_2_orbit_env.sh
~~~

Without a Rust toolchain on `PATH`, the Sphere-Lab SGLang editable install skips
its `setuptools-rust` extensions (`SGLANG_BUILD_RUST_EXTS=none`); Orbit uses the
separate `sglang-router` wheel instead.

## Why cuda-python 13 appears

The cuda-python requirement comes from SGLang's Python metadata. This profile
satisfies it with Orbit's exact CUDA Python pin, then installs Sphere-Lab SGLang
with --no-deps so dependency resolution cannot replace the controlled stack.

## Re-run verification

Inside an H100 or B200 allocation (FlashAttention 3 is sm_90a-only and is not exercised on B200):

~~~bash
/fast/zqiu/orbit-iclr/orbit/envs/orbit-cu130-v1/bin/python \
  scripts/slurm/setup/cu130/verify_env.py \
  --source-root /fast/zqiu/orbit-iclr/orbit/sources/orbit-cu130-v1 \
  --full-h100
~~~

## MPI cache placement

The large, immutable Miles wheel cache remains under `CACHE_ROOT` (normally
`/fast/.../cache/orbit-cu130-v1`). The mutable `uv` distribution cache requires
file locking, but the MPI `/fast` filesystem can return `Function not implemented
(os error 38)` for that operation. The installer therefore defaults
`UV_CACHE_DIR` to `${HOME}/.cache/orbit-cu130-v1/uv`, which resolves to a
Lustre-backed home directory on MPI. Override it explicitly when needed:

```bash
UV_CACHE_DIR=/lustre/home/$USER/.cache/orbit-cu130-v1/uv \
  scripts/slurm/setup/cu130/install_env.sh
```

This changes only the lock-sensitive `uv` cache. Prebuilt CUDA wheels and other
large reusable artifacts remain in `CACHE_ROOT`.

## Prebuilt router and editable overlays

The CUDA 13 workflow installs `sglang_router` from the manylinux_2_28 wheel that
`orbit/pyproject.toml` pins under `[tool.uv.sources]` (`SGLANG_ROUTER_WHEEL_URL`
in `pins.env`), not from the RadixArk Miles wheel set: the Miles build is tagged
manylinux_2_39 and fails to load on glibc 2.35 nodes (Ubuntu 22.04) with
`GLIBC_2.38 not found`. The editable Sphere-Lab SGLang checkout is a Python and
Triton source overlay, so rebuilding its Rust router would duplicate that
component and require an unnecessary Rust toolchain. For the SGLang editable
install only, `install_env.sh` sets `SGLANG_BUILD_RUST_EXTS=none`.

Megatron-LM is installed with `--no-cache --link-mode copy --force-reinstall --editable` so an existing non-editable
`megatron-core` distribution cannot cause `uv` to skip the editable link.

## TileLang Z3 runtime loader

The editable Sphere-Lab Megatron path imports TileLang's bundled TVM. That native
library depends on `libz3.so.4.15`, which is supplied by the installed
`z3-solver` wheel but is outside the default dynamic-loader search path. The
installer exports the wheel's `z3/lib` directory for verification and writes an
idempotent Conda activation hook at
`$ENV_PREFIX/etc/conda/activate.d/orbit-cu130-z3.sh` so later `conda activate`
commands receive the same runtime path.
The runtime pins use NumPy 2.3.5 and align `flashinfer-python`,
`flashinfer-cubin`, and `flashinfer-jit-cache` at 0.6.15.post1; the JIT-cache
wheel carries the expected `+cu130` local version suffix.

## Clean-room verification

To verify the complete CUDA 13 workflow without reusing an existing environment,
source checkout, wheel cache, or uv cache, run this command from the Orbit
repository inside an H100 allocation:

```bash
ENV_PREFIX=/fast/zqiu/orbit-iclr/orbit/envs/orbit-cu130-v2-clean \
SOURCE_ROOT=/fast/zqiu/orbit-iclr/orbit/sources/orbit-cu130-v2-clean \
CACHE_DIR=/fast/zqiu/orbit-iclr/orbit/cache/orbit-cu130-v2-clean \
UV_CACHE_DIR=/lustre/home/$USER/.cache/orbit-cu130-v2-clean/uv \
  scripts/slurm/setup/cu130/install_env.sh
```

These paths are independent of `orbit-cu130-v1`. Do not delete or overwrite the
validated v1 environment. A successful installation ends with
`[summary] 38/38 passed`.

For repeated clean-room checks, choose a new shared suffix for all four paths so
that the environment, sources, wheel cache, and uv cache are all unused.
