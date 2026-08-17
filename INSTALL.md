# Building the orbit environment

One `uv sync` builds the whole CUDA-13.2 stack from source. This document is the recipe plus
the three cluster-specific decisions that make the result **survive a node change** — the
default settings do not, and getting them wrong produces an environment that looks fine and
imports nothing.

For the prebuilt-wheel path installed layer by layer, see [CUDA-13-install.md](CUDA-13-install.md).
This document supersedes it for the from-scratch build.

| | |
|---|---|
| Python | 3.12 (`requires-python = ">=3.12,<3.13"`) |
| CUDA | 13.2 (`/is/software/nvidia/cuda-13.2`, module `cuda/13.2`) |
| torch | 2.11.0 + torchvision 0.26.0 + torchaudio 2.11.0 |
| cuDNN / NCCL | 9.22.0.52 / 2.30.4 (from the venv, not the system) |
| transformers | 5.12.1 (hard-pinned; sglang requires it) |
| Built from source | TransformerEngine, flash-attn 2.8.3, sgl-kernel, DeepEP, DeepGEMM, mamba-ssm, causal-conv1d, torch-memory-saver, fast-hadamard-transform |
| Target GPUs | H100 (sm_90) **and** B200 (sm_100), one fat binary |
| Build time | ~1–2 h, dominated by sgl-kernel's CUTLASS templates |

## Quick start

```bash
cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
export UV_PROJECT_ENVIRONMENT=/fast/zqiu/orbit-iclr/orbit_env
export UV_LINK_MODE=copy
export CUDA_HOME=/is/software/nvidia/cuda-13.2
source env.sh
uv sync --extra allinone
```

`env.sh` defaults the cache to `$HOME/.cache/uv_cu13_orbit` and the arch list to `9.0 10.0`, so
neither needs to be passed. Everything below explains *why* those are the defaults, and how to
check the build actually did what it claims.

**The 1–2 hour figure is for a cold cache.** `$HOME/.cache/uv_cu13_orbit` already holds built
wheels for every expensive package at the exact revisions `pyproject.toml` pins — flash-attn
2.8.3, sglang-kernel 0.4.5, TransformerEngine 2.14.0, DeepEP, DeepGEMM, mamba-ssm,
causal-conv1d — so a sync against it skips every source build. Keeping that cache is worth
roughly two hours per rebuild; this is the second reason not to park it somewhere volatile.

Note the kernel package renamed with the v0.5.16 move: the `sgl-kernel/` subdirectory publishes
**`sglang-kernel` 0.4.5** (it was `sgl-kernel` 0.3.21 on the v0.5.9 line), so a cache warmed
before that move has no entry for it and will rebuild it once. The import name stays `sgl_kernel`.

Measured 2026-07-29 with a warm cache: resolution 617 ms, orbit's own build 7.5 s, and
**68 minutes wall clock** for the whole sync — 330 packages, 12 GB. Nearly all of that hour is
`UV_LINK_MODE=copy` writing to Lustre, not compilation; the last ~17 packages alone are the
large CUDA ones (flash-attn 934 MB, sgl-kernel `flash_ops` 852 MB, `libtransformer_engine`
535 MB). Symlink mode would cut this to a few minutes at the cost of permanently coupling the
venv to the cache. Budget the hour; it buys an environment that survives losing the cache.

## The three decisions that matter

### 1. The uv cache must be flock-capable *and* persistent

uv takes `flock` on its cache during builds. **Lustre (`/lustre/fast`) returns `ENOSYS` on
`flock`**, so the cache cannot live next to the code. That leaves two candidates, and only one
is correct:

- `/tmp` — flock-capable, but **node-local and cleared when you leave the node.** This was the
  old default and it destroyed the environment on 2026-07-29.
- `$HOME/.cache/uv_cu13_orbit` — cluster-home is NFS: flock-capable, persistent, and visible
  from every node. **This is the default now.**

The failure mode when the cache disappears is the reason to care. uv's default install mode is
**symlink**, so site-packages holds links into the cache rather than copies. Lose the cache and
all ~95,000 links dangle. Python then treats each package directory — present, but with no
loadable `__init__.py` — as a **namespace package**, so `import torch` *succeeds* and you get:

```
AttributeError: module 'torch' has no attribute '__version__'
```

No `ImportError`, no warning. That silence is the whole hazard: the env looks importable, a job
launches, a GPU slot burns, and the failure surfaces somewhere unrelated.

### 2. `UV_LINK_MODE=copy` decouples the venv from the cache

With `copy`, site-packages holds real files and the environment is a self-contained artifact —
deleting or relocating the cache afterwards cannot break it. The cost is roughly 25 GB of extra
space on `/fast` (which has hundreds of TB) and a slower sync, since ~95k small files get
written to Lustre.

Symlink mode against the home cache would also be node-portable, and saves that space. It stays
coupled to the cache forever, which is exactly the coupling that just failed. Prefer `copy`
unless disk pressure forces otherwise. Either way: **never run `uv cache clean`** — under
symlink mode it guts every environment pointing into that cache.

### 3. Build for both architectures, never auto-detect

`env.sh` used to read the arch off `nvidia-smi`. Built on an H100 node that pins **sm_90 only**,
and the kernels then fail to load on B200 — after the two hours are already spent. The default
is now an explicit fat-binary list, and it needs to be spelled four times because these builds
do not share a convention:

```
TORCH_CUDA_ARCH_LIST="9.0 10.0"        # torch cpp_extension builds (mamba, causal-conv1d, ...)
NVTE_CUDA_ARCHS="90;100"               # TransformerEngine
FLASH_ATTN_CUDA_ARCHS="90;100"         # flash-attn
CMAKE_CUDA_ARCHITECTURES="90a;100a"    # sgl-kernel (cmake); the `a` variants expose
                                       #   wgmma (sm_90a) and tcgen05 (sm_100a)
```

Setting only `TORCH_CUDA_ARCH_LIST` is the trap — TE, flash-attn and sgl-kernel each ignore it.
Verify the result rather than assuming it (see below); a single-arch build is not detectable
until you run on the other machine.

Measured on the cached wheels with `cuobjdump --list-elf` (2026-07-29), which is what the
current environment installs:

| Binary | Arches present |
|---|---|
| `flash_attn_2_cuda...so` | sm_80, **sm_90**, **sm_100**, sm_120 |
| `libtransformer_engine.so` | sm_75, sm_80, sm_89, **sm_90/90a**, **sm_100/100a**, sm_103a, sm_120 |
| `sgl_kernel/sm90/common_ops` and `sgl_kernel/sm100/common_ops` | sm_80, sm_89, **sm_90/90a**, **sm_100a**, sm_103a, sm_120a |

`sgl_kernel/flash_ops.abi3.so` carries only sm_80/86/**90a** — that is upstream's design, not a
misconfiguration: those are the FlashAttention-3 kernels, which are Hopper-only. Blackwell goes
through the `sm100/` ops directory instead, which is why sgl-kernel ships `sm90/` and `sm100/`
as separate subpackages rather than one fat module.

## Full procedure

### 0. Prerequisites

```bash
command -v uv                       # 0.10.11 at /home/zqiu/.local/bin/uv
ls -d /is/software/nvidia/cuda-13.2  # or: module load cuda/13.2
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

`module load` is a no-op in non-interactive shells, so set `CUDA_HOME` explicitly when scripting.
Budget ~25 GB on home for the cache and ~25 GB on `/fast` for the venv.

### 1. Remove the old environment

`uv sync` reconciles an existing venv rather than rebuilding it, and it cannot repair one whose
symlinks are dangling. Start clean:

```bash
rm -rf /fast/zqiu/orbit-iclr/orbit_env
```

### 2. Build

```bash
cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
export UV_PROJECT_ENVIRONMENT=/fast/zqiu/orbit-iclr/orbit_env
export UV_LINK_MODE=copy
export CUDA_HOME=/is/software/nvidia/cuda-13.2
source env.sh
uv sync --extra allinone 2>&1 | tee "$HOME/log/orbit_env_build.log"
```

What `env.sh` sets that `pyproject.toml` cannot: `CUDA_HOME` and the CUDA `PATH`/`LD_LIBRARY_PATH`;
`CPATH`/`LIBRARY_PATH` pointing at the venv's own NCCL, cuDNN and NVSHMEM headers (the CUDA
module ships none of them, and TransformerEngine's source build needs `nccl.h`);
`CMAKE_PREFIX_PATH` at torch's cmake config (without it sgl-kernel fails with "kineto not
found"); `UV_CONCURRENT_BUILDS=1` (nine concurrent CUDA packages each running `ninja -j32`
exhausts file descriptors); and `CMAKE_BUILD_PARALLEL_LEVEL` scaled to RAM, because sgl-kernel's
CUTLASS template units take 10–30 GB *each* under nvcc and `nproc`-wide parallelism OOMs the
node.

### 3. Verify

Imports and versions. **Source `env.sh` too, not just the activate script** — `deep_ep` and
`deep_gemm` call `find_cuda_home()` at import time and assert if it returns `None`, and
`megatron.core` imports `deep_ep` transitively, so without `CUDA_HOME` all three fail with a
bare `AssertionError` and no message:

```bash
source /fast/zqiu/orbit-iclr/orbit_env/bin/activate
cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
export CUDA_HOME=/is/software/nvidia/cuda-13.2   # see note below
source env.sh
python - <<'PY'
import importlib
for m in ["torch","transformers","sglang","megatron.core","deep_ep","deep_gemm",
          "transformer_engine","sgl_kernel","flash_attn","orbit"]:
    mod = importlib.import_module(m)
    assert getattr(mod, "__file__", None), f"{m} is a NAMESPACE PACKAGE — env is broken"
    print(f"  {m:20s} {getattr(mod,'__version__','ok')}")
import torch; print("  cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0))
PY
```

The `__file__` assertion is the real check: a dangling-symlink env imports every one of these
*successfully* as an empty namespace package, so `import` alone proves nothing.

`CUDA_HOME` must be set explicitly in any non-interactive shell. `env.sh` tries `module load
cuda/13.2` first, but `module` is a no-op when not interactive, and its fallback list
(`/usr/local/cuda-13.2`, `/usr/local/cuda`, `/opt/cuda-13.2`, `/opt/cuda`) does not include this
cluster's `/is/software/nvidia/cuda-13.2`. It warns rather than failing silently:
`env.sh: WARNING — CUDA 13.2 toolkit not found.`

Verified on 2026-08-17 (`i106`, H100 80GB), against the v0.5.16 sglang line:

```
  torch                2.11.0+cu130        megatron.core   0.18.0rc0
  transformers         5.12.1              deep_ep         2.0.0
  sglang               0.0.0.dev15479+g05cd76b4d           deep_gemm       0.1.4.post1
  transformer_engine   2.14.0+71bbefbf     sgl_kernel      0.4.5
  flash_attn           2.8.3               orbit           ok
  cuda True NVIDIA H100 80GB HBM3
```

No broken links, which is the check that would have caught the 2026-07-29 failure at build time:

```bash
find /fast/zqiu/orbit-iclr/orbit_env -xtype l | wc -l    # must be 0
```

**Both architectures present** in the source-built kernels:

```bash
SP=/fast/zqiu/orbit-iclr/orbit_env/lib/python3.12/site-packages
for so in $(find $SP -name "*.so" -path "*flash_attn*" -o -name "*.so" -path "*transformer_engine*" \
            -o -name "*.so" -path "*sgl_kernel*" | head); do
  echo "$so: $(cuobjdump --list-elf $so 2>/dev/null | grep -o 'sm_[0-9]*' | sort -u | tr '\n' ' ')"
done
```

Each should list **both** `sm_90` and `sm_100`. Anything showing one arch was built from a
variable the package ignored — fix that variable and rebuild just that package with
`uv sync --extra allinone --reinstall-package <name>`.

CPU test suite. Use `tests`, not `tests/fast` — the 18 top-level files under `tests/` are the
CUDA-touching ones, and they are exactly the tests that distinguish a real environment from a
version-matched CPU stand-in:

```bash
cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
python -m pytest tests -q -p no:cacheprovider
```

**391 passed, 0 failed, 0 collection errors in 110 s** on 2026-07-29. For comparison, the
CPU-only proxy venv used while this env was unavailable gave 373 passed with **5 collection
errors** — those 5 modules import cleanly here, which is where the extra 18 tests come from. A
run that reports collection errors is a signal the CUDA layer is missing, not a pre-existing
condition to wave through.

## Daily use

```bash
source /fast/zqiu/orbit-iclr/orbit_env/bin/activate
cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
export CUDA_HOME=/is/software/nvidia/cuda-13.2
source env.sh                                  # CUDA_HOME, LD_LIBRARY_PATH, z3 soname path
source examples/load_cuda13_2_orbit_env.sh     # cuDNN / flashinfer runtime — launchers only
```

`env.sh` is not optional even for "just running the tests": `megatron.core` needs it, via the
`deep_ep` import chain described above.

**Order matters — activate first, then `env.sh`.** `env.sh` resolves the target venv as
`ORBIT_VENV` → `UV_PROJECT_ENVIRONMENT` → `VIRTUAL_ENV` → `./.venv`, so activating first lets
`$VIRTUAL_ENV` point it at the right site-packages. Source it the other way round and it falls
through to a `./.venv` that does not exist here, `SITE_PACKAGES` points at nothing, and
`deep_ep` fails with `No libnccl.so found in .../.venv/lib/python3.12/site-packages/nvidia/...`
— a path that is the tell, since the real env is `orbit_env`, not `.venv`.

`env.sh` prepends the venv's cuDNN to `LD_LIBRARY_PATH` *after* the CUDA module has added the
system one, so the venv's 9.22 wins. Load them in the other order and TransformerEngine fails at
import with an undefined symbol in `libcudnn_graph.so.9`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `AttributeError: module 'torch' has no attribute '__version__'` | uv cache gone; every package is a dangling symlink resolving as a namespace package | Full rebuild. Not repairable by re-linking — the payload is gone |
| `No module named pytest.__main__` | Same cause | Same |
| `Disk quota exceeded` mid-build | Cluster-home enforces a **per-user quota**; `df` shows the shared pool, not your cap | Free space on home, or move `UV_CACHE_DIR` to another persistent flock-capable path |
| `os error 38` / `ENOSYS` on a lock file | Cache placed on Lustre (`/lustre/fast`) | Move `UV_CACHE_DIR` to home |
| sgl-kernel build OOMs the node | CUTLASS units take 10–30 GB each under nvcc | Lower `CMAKE_BUILD_PARALLEL_LEVEL`, or raise `ORBIT_SGL_KERNEL_JOB_GB` |
| "Too many open files" during build | Concurrent package builds each spawning `ninja -j32` | `UV_CONCURRENT_BUILDS=1` (already set by `env.sh`) |
| sgl-kernel: "kineto not found" | `CMAKE_PREFIX_PATH` missing torch's cmake config | `source env.sh` before `uv sync` |
| TE import: undefined symbol in `libcudnn_graph.so.9` | System cuDNN from the CUDA module shadows the venv's 9.22 | Source `env.sh` after loading CUDA; do not prepend system cuDNN afterwards |
| Kernels load on H100 but not B200 (or vice versa) | Single-arch build from an ignored arch variable | Re-check all four arch variables, rebuild the affected package |
| Bare `AssertionError` with no message from `deep_ep`, `deep_gemm` or `megatron.core` | `CUDA_HOME` unset; their `find_cuda_home()` asserts | `export CUDA_HOME=...` and `source env.sh` |
| `No libnccl.so found in .../.venv/...` | `env.sh` sourced before the venv was activated, so it fell back to a non-existent `./.venv` | Activate first, then `source env.sh` |
| `env.sh: WARNING — CUDA 13.2 toolkit not found` | Non-interactive shell: `module` is a no-op and the fallback list lacks `/is/software/nvidia/` | Set `CUDA_HOME` explicitly |
