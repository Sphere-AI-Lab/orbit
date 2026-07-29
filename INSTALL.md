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
| transformers | 4.57.1 (hard-pinned; sglang requires it) |
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
2.8.3, sgl-kernel 0.3.21, TransformerEngine 2.14.0, DeepEP, DeepGEMM, mamba-ssm,
causal-conv1d — so a sync against it skips every source build and finishes in minutes. Keeping
that cache is worth roughly two hours per rebuild; this is the second reason not to park it
somewhere volatile.

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

Imports and versions:

```bash
source /fast/zqiu/orbit-iclr/orbit_env/bin/activate
python - <<'PY'
import torch, transformers, sglang, megatron.core, orbit
print("torch     ", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__)
print("cuda avail", torch.cuda.is_available(), torch.cuda.device_count())
PY
```

`torch.__version__` printing at all is the real assertion here — it is precisely what a
dangling-symlink env cannot do.

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

CPU test suite (always pass explicit paths — `norecursedirs` matches `tools` and `scripts` at
any depth, so a bare `pytest tests/fast` silently skips whole directories):

```bash
cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
python -m pytest tests/fast -q -p no:cacheprovider
```

## Daily use

```bash
source /fast/zqiu/orbit-iclr/orbit_env/bin/activate     # tests, linting, imports
```

Anything that touches CUDA at runtime — a launcher, a GPU smoke test — needs the runtime paths
too:

```bash
source /fast/zqiu/orbit-iclr/orbit_env/bin/activate
cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
source env.sh                                  # CUDA_HOME, LD_LIBRARY_PATH, z3 soname path
source examples/load_cuda13_2_orbit_env.sh     # cuDNN / flashinfer runtime
```

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
