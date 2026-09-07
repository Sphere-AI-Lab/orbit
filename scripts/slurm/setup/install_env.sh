#!/bin/bash
#
# install_env.sh — from-source conda env build for orbit.
#
# Follows docs/getting-started/installation.md Method 2, extended with the
# transitive deps that orbit recipes actually need at runtime — the bare
# `pip install -r requirements.txt && pip install -e . --no-deps` from the
# docs is NOT sufficient outside the radixark/miles docker base image.
#
# What this adds beyond the docs:
#   - SGLang and Megatron-LM are both installed editable from `thirdparty/`
#     (NOT from PyPI). For SGLang we drop the Dockerfile's `--no-deps` because
#     we don't start from `lmsysorg/sglang:v0.5.10` — pip needs to resolve the
#     runtime tree itself (fastapi/uvicorn/orjson/flashinfer/sglang-kernel/...)
#     against the patched fork's own pyproject.toml.
#   - mbridge          — tools/convert_hf_to_torch_dist.py imports it
#   - torch_memory_saver — orbit/backends/megatron_utils/actor.py imports it
#   - transformer_engine + source-built torch extension — Megatron training requires it
#   - flash-attn 2/3   — `--attention-backend flash` and FA3-only Megatron paths
#
# Versions for the non-thirdparty deps (TE, flash-attn, mbridge, tms) match
# docker/Dockerfile pins. SGLang + Megatron versions come from the submodule
# pointers in .gitmodules.
#
# Target: H200/Hopper on CUDA 13. Mirrors docker/Dockerfile's ENABLE_CUDA_13=1 path only:
# torch from the cu130 index, the wheels bundle's prebuilt transformer_engine{,_cu13,_torch}
# triplet + docker/patch/cu13, nvidia-cudnn-cu13. The CUDA-12 path (cu129 index, TE torch
# extension built from source) was retired once thirdparty/sglang moved to torch 2.13 —
# miles-wheels publishes torch-2.13 bundles for CUDA 13 only, so a cu12 bundle can no
# longer match the submodule and the installer refuses cu12 tags outright.
#
# Usage:
#   salloc --gres=gpu:1 --cpus-per-task=16 --mem=128G --time=2:00:00 --pty bash
#   bash scripts/slurm/setup/install_env.sh
#
# Knobs (env vars, all optional):
#   ORBIT_ENV_NAME    conda env name                       [orbit]
#   ORBIT_PY_VERSION  python version                       [3.12]
#   ORBIT_REPO        this repo                            [$PWD]
#   THIRDPARTY_DIR    submodule dir                        [$ORBIT_REPO/thirdparty]
#   PULL_REMOTE       `git submodule update --remote`?     [0]  set 1 to bump to branch HEAD
#   CUDA_HOME         override system CUDA toolkit path    [auto-detected]
#   CUDNN_VERSION     override the nvidia-cudnn-cu13 pin    [pins.env CUDNN_CU13_VERSION]
#   KERNELS_SPEC      transformers hub-kernels compat cap  [kernels>=0.12,<0.15]
#   SGLANG_SRC        external sglang checkout/worktree    [$THIRDPARTY_DIR/sglang]
#   SGL_WHL_INDEX_URL extra index for +cu130 kernel wheels [pins.env, docs.sglang.ai/whl/cu130]
#                     (sglang-kernel / sgl-deep-gemm; PyPI defaults are cu13 too)
#   SGLANG_EXTRA_CONSTRAINT  extra uv constraint file for the sglang step [unset]

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# --- Knobs (paths, names, feature toggles) -------------------------------

ORBIT_ENV_NAME=${ORBIT_ENV_NAME:-orbit}
ORBIT_PY_VERSION=${ORBIT_PY_VERSION:-3.12}
ORBIT_REPO=${ORBIT_REPO:-$PWD}
THIRDPARTY_DIR=${THIRDPARTY_DIR:-$ORBIT_REPO/thirdparty}
WHEELS_DIR=${WHEELS_DIR:-$THIRDPARTY_DIR/wheels}
PULL_REMOTE=${PULL_REMOTE:-0}
CONDA_ROOT=${CONDA_ROOT:-/data/shared/conda/miniconda3}

# Toggle heavy optional wheels (default ON since we target Hopper/H200 on the
# slinky cluster). FA3 is Hopper-only; apex enables Megatron fused optimizer
# and layernorm paths.
INSTALL_FLASH_ATTN=${INSTALL_FLASH_ATTN:-1}     # FA2 (Megatron --attention-backend flash)
INSTALL_FLASH_ATTN_3=${INSTALL_FLASH_ATTN_3:-1} # FA3 (Hopper TMA path in Megatron attention.py)
INSTALL_APEX=${INSTALL_APEX:-1}                  # FusedAdam, FastLayerNorm, multi_tensor_applier
export INSTALL_FLASH_ATTN INSTALL_FLASH_ATTN_3 INSTALL_APEX

# --- Pinned versions / commits (sourced from pins.env) -------------------
# pins.env is auto-generated from docker/Dockerfile + sglang upstream by
# extract_pins.py. Each value can still be overridden via env vars.

# shellcheck disable=SC1091
source "$SCRIPT_DIR/pins.env"

# Re-derive the sglang-stack fields from the EFFECTIVE MILES_WHEELS_TAG. pins.env
# bakes MILES_WHEELS_TORCH_VERSION / MILES_WHEELS_SGLANG_VERSION / SGLANG_ROUTER_VERSION
# as a static snapshot, but MILES_WHEELS_TAG is independently overridable at runtime
# (`MILES_WHEELS_TAG=... bash install_env.sh`). Without re-deriving, an override of the
# tag would leave the torch/sglang fields stale and the ABI guard below would pass while
# _fetch_orbit_wheel pulls a mismatched-ABI wheel set. WHEELS_STACK in extract_pins.py is
# the single source of truth; resolve against it so the tag is always authoritative.
_resolved=$(python3 "$SCRIPT_DIR/extract_pins.py" --resolve "$MILES_WHEELS_TAG") || {
    echo "FATAL: could not resolve MILES_WHEELS_TAG=$MILES_WHEELS_TAG via extract_pins.py --resolve" >&2
    echo "       (unknown wheels tag, or python3 unavailable). Add a WHEELS_STACK row, or use a known tag." >&2
    exit 1
}
eval "$_resolved"
unset _resolved

# sglang-stack torch-ABI guard (fail closed). The prebuilt flash-attn /
# flash-attn-3 / apex wheels in $MILES_WHEELS_TAG are compiled against a specific
# torch C++ ABI; MILES_WHEELS_TORCH_VERSION (derived from the tag) MUST equal the
# torch we install (TORCH_VERSION, from the sglang submodule's pyproject). Mismatch
# = torch-X wheels into a torch-Y env = ImportError/segfault. Placed before the GPU
# preflight so a pins/override check fails on the real cause, not a missing GPU.
if [[ -n "${MILES_WHEELS_TORCH_VERSION:-}" && "$MILES_WHEELS_TORCH_VERSION" != "$TORCH_VERSION" ]]; then
    echo "FATAL: MILES_WHEELS_TAG=$MILES_WHEELS_TAG ships torch-$MILES_WHEELS_TORCH_VERSION wheels," >&2
    echo "       but TORCH_VERSION=$TORCH_VERSION. flash-attn/apex are torch-ABI-bound — this would" >&2
    echo "       build an ImportError/segfault env. Run sglang-sync to realign pins + submodule," >&2
    echo "       or set MILES_WHEELS_TAG to the release built for torch $TORCH_VERSION." >&2
    exit 1
fi

# Non-fatal drift check — warn if pins.env hasn't been regenerated since the
# Dockerfile bumped. CI / `--check` makes it hard. Skipped if sources missing
# (e.g. fresh clone without submodules — install_env.sh inits them later).
if command -v python3 >/dev/null \
    && [[ -f "$ORBIT_REPO/docker/Dockerfile" ]] \
    && [[ -f "$THIRDPARTY_DIR/sglang/docker/Dockerfile" ]]; then
    python3 "$SCRIPT_DIR/extract_pins.py" --check >/dev/null 2>&1 \
        || echo "[pins] WARN: pins.env is stale vs upstream sources — regenerate with:
                python3 scripts/slurm/setup/extract_pins.py --write" >&2
fi

# ---------- preflight ----------------------------------------------------

command -v "$CONDA_ROOT/bin/conda" >/dev/null \
    || { echo "FATAL: conda not at $CONDA_ROOT/bin/conda — set CONDA_ROOT" >&2; exit 1; }
command -v uv >/dev/null \
    || { echo "FATAL: uv not on PATH" >&2; exit 1; }
nvidia-smi >/dev/null 2>&1 \
    || { echo "FATAL: nvidia-smi not working — are you on a salloc with a GPU?" >&2; exit 1; }

# Torch/flashinfer/prebuilt wheels are all CUDA-13 (cu130) builds. Keep those CUDA
# build tags aligned, then make sure the driver advertises a compatible CUDA
# runtime before spending time on the install.
_cuda_tag_from() {
    local value=${1:-}
    if [[ "$value" =~ cu([0-9][0-9][0-9]) ]]; then
        printf '%s' "${BASH_REMATCH[1]}"
    fi
    return 0
}

torch_cu_tag=$(_cuda_tag_from "$TORCH_INDEX_URL")
flashinfer_cu_tag=$(_cuda_tag_from "$FLASHINFER_INDEX_URL")
wheels_cu_tag=$(_cuda_tag_from "$MILES_WHEELS_TAG")
if [[ "${torch_cu_tag:0:2}" != "13" ]]; then
    echo "FATAL: TORCH_INDEX_URL is cu${torch_cu_tag:-?} — this installer mirrors the Dockerfile's CUDA-13 (ENABLE_CUDA_13=1) path only." >&2
    echo "       The CUDA-12 path was retired: no torch-2.13 wheels bundle exists for CUDA 12. Use a cu130 MILES_WHEELS_TAG." >&2
    exit 1
fi
echo "[preflight] CUDA build: cu$torch_cu_tag (wheels $MILES_WHEELS_TAG)"
for tagged_source in \
    "FLASHINFER_INDEX_URL:$flashinfer_cu_tag" \
    "MILES_WHEELS_TAG:$wheels_cu_tag"; do
    tag_name=${tagged_source%%:*}
    tag_value=${tagged_source#*:}
    if [[ -n "$torch_cu_tag" && -n "$tag_value" && "$tag_value" != "$torch_cu_tag" ]]; then
        echo "FATAL: CUDA build tag mismatch: TORCH_INDEX_URL=cu$torch_cu_tag but $tag_name=cu$tag_value" >&2
        echo "       Set TORCH_INDEX_URL, FLASHINFER_INDEX_URL, and MILES_WHEELS_TAG to the same CUDA build." >&2
        exit 1
    fi
done

if [[ -n "$torch_cu_tag" ]]; then
    required_cuda="${torch_cu_tag:0:2}.${torch_cu_tag:2:1}"
    driver_cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)
    if [[ -n "$driver_cuda" ]]; then
        driver_major=${driver_cuda%%.*}
        driver_minor=${driver_cuda##*.}
        need_major=${required_cuda%%.*}
        need_minor=${required_cuda##*.}
        if (( driver_major > need_major )) || \
           (( driver_major == need_major && driver_minor >= need_minor )); then
            echo "[preflight] driver CUDA capability: $driver_cuda (required >= $required_cuda)"
        elif (( driver_major == need_major )); then
            # CUDA Minor Version Forward Compatibility: a 12.y driver can run
            # 12.x apps that avoid 12.x-only features. Most training kernels
            # are fine; some bleeding-edge FA3/flashinfer paths could miss
            # symbols at runtime. The smoke-test import catches this.
            if [[ "${ALLOW_CUDA_MINOR_FORWARD_COMPAT:-1}" != "1" ]]; then
                echo "FATAL: $TORCH_INDEX_URL needs a driver advertising CUDA >= $required_cuda; nvidia-smi shows $driver_cuda" >&2
                echo "       Use a newer driver, switch to a matching CUDA build, or set ALLOW_CUDA_MINOR_FORWARD_COMPAT=1 to opt into minor-version compat." >&2
                exit 1
            fi
            echo "[preflight] WARNING: driver CUDA $driver_cuda < wheel CUDA $required_cuda — relying on CUDA $need_major.x minor-version forward compatibility."
            echo "[preflight] If smoke-test imports fail with CUDA symbol errors, install a driver supporting CUDA >= $required_cuda or switch to cu${driver_major}${driver_minor} wheels."
        else
            echo "FATAL: $TORCH_INDEX_URL needs a driver advertising CUDA $required_cuda; nvidia-smi shows $driver_cuda (different major — no forward compat)." >&2
            exit 1
        fi
    fi
fi

# FA3 is a Hopper kernel family. Fail early if it is explicitly enabled on a
# non-Hopper GPU instead of discovering the problem after the environment build.
if [[ "$INSTALL_FLASH_ATTN_3" == "1" ]]; then
    gpu_cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ' || true)
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
    if [[ -n "$gpu_cc" ]]; then
        gpu_cc_major=${gpu_cc%%.*}
        if [[ "$gpu_cc_major" != "9" ]]; then
            echo "FATAL: INSTALL_FLASH_ATTN_3=1 requires a Hopper GPU (SM90); detected $gpu_name compute capability $gpu_cc" >&2
            echo "       Disable FA3 with INSTALL_FLASH_ATTN_3=0, or run on H100/H200/H800." >&2
            exit 1
        fi
    elif [[ ! "$gpu_name" =~ H100|H200|H800 ]]; then
        echo "FATAL: INSTALL_FLASH_ATTN_3=1 requires Hopper (H100/H200/H800); detected '$gpu_name'" >&2
        exit 1
    fi
    echo "[preflight] FA3 target GPU: ${gpu_name:-unknown} (compute capability ${gpu_cc:-unknown})"
fi

# nvcc (CUDA 13) is still needed: torch_memory_saver builds a CUDA extension below and
# flashinfer JIT-compiles kernels at runtime (transformer_engine itself comes prebuilt).
# Prefer the system CUDA toolkit; if absent, instruct the user (conda-install is
# heavy; we don't auto-install ~3 GB without consent).
if [[ -z "${CUDA_HOME:-}" ]]; then
    for cuda_dir in /usr/local/cuda-13.0 /usr/local/cuda-13 /usr/local/cuda; do
        if [[ -x "$cuda_dir/bin/nvcc" ]]; then
            export CUDA_HOME="$cuda_dir"
            break
        fi
    done
fi
if [[ -z "${CUDA_HOME:-}" ]] || [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "FATAL: nvcc not found (needed for torch_memory_saver's CUDA extension and flashinfer JIT)." >&2
    echo "       Either:" >&2
    echo "         1. Set CUDA_HOME=/path/to/cuda-13 before re-running, or" >&2
    echo "         2. conda install -n $ORBIT_ENV_NAME -c nvidia/label/cuda-13.0.0 cuda-nvcc cuda-cudart-dev cuda-libraries-dev" >&2
    exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
echo "[preflight] nvcc: $(nvcc --version | tail -1)"
echo "[preflight] CUDA_HOME=$CUDA_HOME"
nvcc_major=$(nvcc --version | sed -n 's/.*release \([0-9]*\)\..*/\1/p' | head -1)
if [[ -n "$nvcc_major" && "$nvcc_major" != "13" ]]; then
    echo "[preflight] WARNING: nvcc is CUDA $nvcc_major but the wheels bundle is CUDA 13 — torch_memory_saver's" >&2
    echo "[preflight]          CUDA extension would compile against the wrong toolkit; set CUDA_HOME to a CUDA-13 toolkit." >&2
fi

mkdir -p "$THIRDPARTY_DIR"

# ---------- submodules: init + fail-closed validation BEFORE env mutation -
# Init the source tree and validate the sglang/torch line HERE — before the
# torch install below mutates a (possibly reused) conda env. A wrong submodule
# vs MILES_WHEELS_TAG, or a TORCH_VERSION override that disagrees with the
# submodule's pyproject, must abort before we touch torch. Needs only git +
# pins.env values; no conda/torch dependency, so it is safe this early.
echo "[src] initialising submodules under $THIRDPARTY_DIR"
git -C "$ORBIT_REPO" submodule update --init --recursive \
    thirdparty/Megatron-LM thirdparty/sglang thirdparty/Megatron-Bridge

if [[ "$PULL_REMOTE" == "1" ]]; then
    echo "[src] PULL_REMOTE=1 — bumping submodules to branch HEAD"
    git -C "$ORBIT_REPO" submodule update --remote --recursive \
        thirdparty/Megatron-LM thirdparty/sglang thirdparty/Megatron-Bridge
fi

MEGATRON_SRC="$THIRDPARTY_DIR/Megatron-LM"
# SGLANG_SRC may point at an external sglang checkout/worktree (version-bump
# testing) without moving the production submodule tree that live envs run.
SGLANG_SRC=${SGLANG_SRC:-$THIRDPARTY_DIR/sglang}
MEGATRON_BRIDGE_SRC="$THIRDPARTY_DIR/Megatron-Bridge"

# Verify the hand-owned source pin against the checkout, then fail closed on
# the actual torch ABI. The source may reuse an older wheels bundle when torch
# matches; `git describe` is only diagnostic and may be unavailable in a
# shallow clone.
sub_sglang_base=$(git -C "$SGLANG_SRC" describe --tags --abbrev=0 2>/dev/null || echo "")
if [[ -n "${ORBIT_SGLANG_SOURCE_VERSION:-}" && -n "$sub_sglang_base" \
      && "$sub_sglang_base" != "$ORBIT_SGLANG_SOURCE_VERSION" ]]; then
    echo "[pins] WARN: sglang source is $sub_sglang_base but ORBIT_SGLANG_SOURCE_VERSION=$ORBIT_SGLANG_SOURCE_VERSION" >&2
    echo "[pins]       Update the hand-owned source pin as part of sglang-sync." >&2
fi
# The submodule's own torch pin must equal TORCH_VERSION (catches a hand-set
# TORCH_VERSION override that disagrees with what sglang was built against).
sub_torch=$(grep -oE '"torch==[0-9][^"]*"' "$SGLANG_SRC/python/pyproject.toml" 2>/dev/null \
            | head -1 | tr -d '"' | cut -d= -f3)
if [[ -n "$sub_torch" && "$sub_torch" != "$TORCH_VERSION" ]]; then
    echo "FATAL: thirdparty/sglang pyproject pins torch==$sub_torch but TORCH_VERSION=$TORCH_VERSION." >&2
    echo "       Regenerate pins: python scripts/slurm/setup/extract_pins.py --write" >&2
    exit 1
fi

# shellcheck disable=SC1091
source "$CONDA_ROOT/etc/profile.d/conda.sh"

# ---------- conda env ----------------------------------------------------

if ! conda env list | awk '{print $1}' | grep -qx "$ORBIT_ENV_NAME"; then
    echo "[env] creating conda env '$ORBIT_ENV_NAME' (python=$ORBIT_PY_VERSION)"
    conda create -y -n "$ORBIT_ENV_NAME" "python=$ORBIT_PY_VERSION"
else
    echo "[env] reusing existing conda env '$ORBIT_ENV_NAME'"
fi
conda activate "$ORBIT_ENV_NAME"
echo "[env] python: $(python --version)  prefix: $CONDA_PREFIX"

UV="uv pip install --python $CONDA_PREFIX/bin/python"
PY_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
CUDNN_LIB_DIR="$PY_SITE/nvidia/cudnn/lib"

_prepend_ld_library_path_once() {
    local dir=$1
    [[ -n "$dir" ]] || return 0
    case ":${LD_LIBRARY_PATH:-}:" in
        *":$dir:"*) : ;;
        *) export LD_LIBRARY_PATH="$dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
    esac
}

_prepend_env_cuda_libs() {
    # Keep env-provided CUDA/cuDNN libs ahead of host paths such as
    # /usr/lib/x86_64-linux-gnu, which can contain an older system cuDNN.
    _prepend_ld_library_path_once "$CONDA_PREFIX/lib"
    _prepend_ld_library_path_once "$CUDNN_LIB_DIR"
}

CUDNN_PKG="nvidia-cudnn-cu13"
# Override knob: CUDNN_VERSION. Default: pins.env CUDNN_CU13_VERSION (the Dockerfile's
# explicit nvidia-cudnn-cu13 pin, ahead of the one torch itself declares); if pins.env has
# none, torch's declared pin is used.
CUDNN_VERSION=${CUDNN_VERSION:-${CUDNN_CU13_VERSION:-}}

_torch_declared_cudnn_version() {
    # Parse torch's own `nvidia-cudnn-cuNN==X.Y.Z` requirement (no `packaging` dependency).
    CUDNN_PKG="$CUDNN_PKG" "$CONDA_PREFIX/bin/python" - <<'PY'
from importlib import metadata
import os, re, sys
pkg = os.environ["CUDNN_PKG"]
for req_text in metadata.requires("torch") or []:
    m = re.match(rf"^{re.escape(pkg)}\s*==\s*([0-9][0-9.]*)", req_text.strip())
    if m:
        print(m.group(1)); raise SystemExit(0)
raise SystemExit(f"FATAL: torch metadata does not declare {pkg}==<version>")
PY
}

_effective_cudnn_version() {
    if [[ -n "${CUDNN_VERSION:-}" ]]; then
        printf '%s\n' "$CUDNN_VERSION"
    else
        _torch_declared_cudnn_version
    fi
}

_install_cudnn_for_torch() {
    local context=${1:-"torch-declared pin"}
    local version
    version=$(_effective_cudnn_version)
    export CUDNN_VERSION="$version"
    _prepend_env_cuda_libs
    echo "[deps] $CUDNN_PKG==$CUDNN_VERSION ($context)"
    $UV "$CUDNN_PKG==$CUDNN_VERSION"
}

_prepend_env_cuda_libs

# The LD_LIBRARY_PATH prepend above puts the env's OpenSSL (conda-forge libprotobuf/ffmpeg
# pull it in) ahead of the system one. git's ssh transport — used for ssh:// remotes, and
# for https:// GitHub remotes on accounts with a `url.git@github.com:.insteadOf` rewrite —
# then runs the system `ssh` against the wrong libssl and aborts with "OpenSSL version
# mismatch" (seen as uv's "Git operation failed" on the git+https installs below). Keep
# git's ssh child on the system libraries.
export GIT_SSH_COMMAND=${GIT_SSH_COMMAND:-"env -u LD_LIBRARY_PATH ssh"}

# ---------- torch (pinned, cu130 index) ----------------------------------

echo "[torch] torch==$TORCH_VERSION + torchvision from $TORCH_INDEX_URL"
$UV --index-url "$TORCH_INDEX_URL" "torch==$TORCH_VERSION" torchvision
_install_cudnn_for_torch "after torch install; override with CUDNN_VERSION if needed"

# ---------- patched Megatron-LM + sglang (editable source installs) -------
# (submodules were init'd + validated above, before the torch install)

# Megatron-Core's pyproject deps are just `torch>=2.6.0, numpy, packaging`,
# all already satisfied. --no-deps avoids any chance of pip re-resolving torch.
echo "[src] installing Megatron-LM editable (--no-deps)"
$UV -e "$MEGATRON_SRC" --no-deps

# Megatron-LM's pyproject only declares `megatron.core*` as packages, so the
# editable finder does NOT expose `orbit_megatron_plugins/` (top-level pkg
# inside the same repo, hard-imported from megatron/core/transformer/*.py).
# Drop a .pth file so `import orbit_megatron_plugins` works in any python
# invocation without needing PYTHONPATH set.
echo "$MEGATRON_SRC" > "$PY_SITE/orbit-megatron-source-root.pth"
echo "[src] orbit-megatron-source-root.pth -> $MEGATRON_SRC"

# sglang's pyproject (post-2026-05) declares a setuptools-rust extension that
# builds `thirdparty/sglang/rust/sglang-grpc` via cargo + protoc. The
# `lmsysorg/sglang` docker image installs both via apt; we don't have root, so
# put rustup in $HOME/.cargo and protoc in the conda env's bin/.
# Idempotent: rustup -y is a no-op if already installed; conda install ditto.
if ! command -v cargo &>/dev/null && [[ ! -x "$HOME/.cargo/bin/cargo" ]]; then
    echo "[deps] installing rustup (sglang setuptools-rust ext build dep)"
    curl --proto '=https' --tlsv1.2 --retry 3 --retry-delay 2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
export PATH="$HOME/.cargo/bin:$PATH"
command -v cargo >/dev/null \
    || { echo "FATAL: cargo still not on PATH after rustup install" >&2; exit 1; }

if ! command -v protoc >/dev/null; then
    echo "[deps] installing libprotobuf + protobuf into $ORBIT_ENV_NAME (sglang-grpc build dep)"
    "$CONDA_ROOT/bin/conda" install -n "$ORBIT_ENV_NAME" -c conda-forge -y libprotobuf protobuf
fi

# torchcodec (sglang srt dep, torch-2.11-matched) dlopens FFmpeg shared libs at
# import; without them every engine start dumps a (non-fatal) probe traceback and
# video decode is unavailable. FFmpeg 7 pairs with torchcodec's core7 loader
# (libavutil.so.59). Docker gets ffmpeg from the base image; bare-metal installs it
# into the env, which conda activation puts on the loader path.
if [[ ! -e "$CONDA_PREFIX/lib/libavutil.so.59" ]]; then
    echo "[deps] installing ffmpeg 7 into $ORBIT_ENV_NAME (torchcodec runtime libs)"
    "$CONDA_ROOT/bin/conda" install -n "$ORBIT_ENV_NAME" -c conda-forge -y 'ffmpeg=7'
fi

# SGLang's pyproject declares the full runtime tree (fastapi/uvicorn/orjson/
# flashinfer_python/sglang-kernel/flash-attn-4/cuda-python/...). We DON'T use
# --no-deps because we're not starting from `lmsysorg/sglang:v0.5.10` — pip
# has to resolve those itself. extra-index-url is needed for flashinfer.
echo "[src] installing sglang editable from $SGLANG_SRC (full dep resolution)"
# unsafe-first-match: sglang declares torchao==0.9.0 which is only on PyPI, not
# on $TORCH_INDEX_URL or $FLASHINFER_INDEX_URL. Without this, uv refuses to
# look at PyPI for torchao (dependency-confusion guard). Order of indexes
# still controls preference (cu130/flashinfer before pypi); uv only falls
# through when earlier indexes do not provide a compatible version.
#
# sglang's pyproject is CUDA-13-native (cuda-python>=13, flashinfer_python[cu13], ...), so
# its pins are left alone. The one constraint: sglang pins `torch==$TORCH_VERSION` with no
# local tag, and under unsafe-first-match uv would swap the +cu130 index wheel for PyPI's
# plain torch. Same CUDA build, but the local tag is what verify_env.py's cu-build check
# reads — hold torch at the +cu130 wheel.
mkdir -p "$WHEELS_DIR"
echo "torch==${TORCH_VERSION}+cu${torch_cu_tag}" > "$WHEELS_DIR/torch-cu-constraint.txt"
sglang_dep_args=(--constraint "$WHEELS_DIR/torch-cu-constraint.txt")
# Caller-supplied extra constraints (e.g. pre-pinning resolver-backtrack-prone
# deps like flash-attn-4/quack-kernels/cuda-tile on a version-bump test).
if [[ -n "${SGLANG_EXTRA_CONSTRAINT:-}" ]]; then
    sglang_dep_args+=(--constraint "$SGLANG_EXTRA_CONSTRAINT")
fi
# Optional extra index for +cu130 local-version kernel wheels (sglang-kernel /
# sgl-deep-gemm at docs.sglang.ai/whl/cu130; pins.env derives it from the wheels tag).
if [[ -n "${SGL_WHL_INDEX_URL:-}" ]]; then
    sglang_dep_args+=(--extra-index-url "$SGL_WHL_INDEX_URL")
fi
$UV -e "$SGLANG_SRC/python[all]" \
    "${sglang_dep_args[@]}" \
    --extra-index-url "$FLASHINFER_INDEX_URL" \
    --extra-index-url "$TORCH_INDEX_URL" \
    --index-strategy unsafe-first-match

# transformers==5.6.0 imports integrations.hub_kernels at model-import time and
# constructs kernels.LayerRepository objects without a version/revision. kernels
# 0.15+ made that invalid, which breaks `import megatron.bridge` / `import mbridge`
# before any model code runs. SGLang's pyproject leaves `kernels` unpinned, so
# cap the incompatible API line until transformers catches up.
KERNELS_SPEC=${KERNELS_SPEC:-"kernels>=0.12,<0.15"}
echo "[deps] $KERNELS_SPEC (transformers hub_kernels compatibility)"
$UV "$KERNELS_SPEC"

# docker/Dockerfile reconciliation checks. On bare metal uv resolves sglang's own pins
# directly (no base image to drag older builds in), so these are assertions, not reinstalls:
#  - nvidia-cutlass-dsl 4.6.x's tvm_ffi provider needs apache-tvm-ffi >= 0.1.10
#    (make_kwargs_wrapper(map_dataclass_to_tuple=...)); too old -> flashinfer's CuTe rmsnorm
#    dies during CUDA-graph capture.
#  - every nvidia-cutlass-dsl component must sit at the version sglang pins (4.6.0/4.6.1 hang
#    FA4 CuTe backward on sm_103).
"$CONDA_PREFIX/bin/python" - <<'PY'
import inspect
from importlib.metadata import version
from tvm_ffi.utils import kwargs_wrapper as k
params = inspect.signature(k.make_kwargs_wrapper).parameters
assert "map_dataclass_to_tuple" in params, f"apache-tvm-ffi {version('apache-tvm-ffi')} too old for nvidia-cutlass-dsl 4.6.x: {list(params)}"
print(f"[deps] apache-tvm-ffi {version('apache-tvm-ffi')} provides make_kwargs_wrapper(map_dataclass_to_tuple=...)")
PY
cutlass_pin=$(grep -oE '"nvidia-cutlass-dsl(\[[a-z0-9]+\])?==[^"]+"' "$SGLANG_SRC/python/pyproject.toml" | head -1 | sed -E 's/.*==([^"]+)"/\1/')
if [[ -n "$cutlass_pin" ]]; then
    CUTLASS_PIN="$cutlass_pin" "$CONDA_PREFIX/bin/python" - <<'PY'
import os
from importlib.metadata import version, PackageNotFoundError
want = os.environ["CUTLASS_PIN"]
comps = ["nvidia-cutlass-dsl", "nvidia-cutlass-dsl-libs-base", "nvidia-cutlass-dsl-libs-core",
         "nvidia-cutlass-dsl-libs-cu13"]
got = {}
for c in comps:
    try: got[c] = version(c)
    except PackageNotFoundError: pass
bad = {c: v for c, v in got.items() if v != want}
assert not bad, f"nvidia-cutlass-dsl components not at sglang's pin {want}: {bad}"
print(f"[deps] nvidia-cutlass-dsl components at {want}: {sorted(got)}")
PY
fi

# sglang_router is installed from the miles-wheels release (NOT PyPI) in the
# prebuilt-wheels section below, to match the upstream Dockerfile's wheel source.

# ---------- recipe-required runtime deps (not in requirements.txt) -------

echo "[deps] mbridge @ $MBRIDGE_COMMIT (for tools/convert_hf_to_torch_dist.py)"
$UV "git+https://github.com/ISEEKYAN/mbridge.git@$MBRIDGE_COMMIT" --no-deps

echo "[deps] nvidia-modelopt — required by megatron.bridge's auto_bridge.py top-level import"
# Dockerfile has the [torch] extra but modelopt 0.44+ dropped it (warning-only).
# Side effect: can pull an older nvidia-cudnn-cu13 and clobber the pinned
# cuDNN pin; reassert the effective torch-derived/overridden cuDNN before TE.
$UV --no-build-isolation "nvidia-modelopt>=0.37.0"

echo "[src] installing Megatron-Bridge editable from $MEGATRON_BRIDGE_SRC (--no-deps)"
$UV -e "$MEGATRON_BRIDGE_SRC" --no-deps --no-build-isolation
echo "[deps] megatron-energon (--no-deps; mirrors docker/Dockerfile — optional megatron.bridge import)"
$UV megatron-energon --no-deps

echo "[deps] torch_memory_saver @ $TMS_COMMIT (for orbit/backends/megatron_utils/actor.py)"
# Newer TMS source builds need TMS_CUDA_MAJOR (upstream #1774); derive from the env's torch.
TMS_CUDA_MAJOR=$("$CONDA_PREFIX/bin/python" -c "import torch; print(torch.version.cuda.split('.')[0])") \
    $UV --no-cache-dir --force-reinstall "git+https://github.com/fzyzcjy/torch_memory_saver.git@$TMS_COMMIT"

mkdir -p "$WHEELS_DIR/$MILES_WHEELS_TAG"
_fetch_orbit_wheel() {
    # Usage: _fetch_orbit_wheel <basename-prefix>
    # Downloads matching asset from the miles-wheels release if not already cached.
    local prefix=$1
    local existing
    existing=$(compgen -G "$WHEELS_DIR/$MILES_WHEELS_TAG/${prefix}*" || true)
    if [[ -n "$existing" ]]; then
        echo "$existing" | head -1
        return 0
    fi
    local url
    url=$(curl -fsSL "https://api.github.com/repos/$MILES_WHEELS_REPO/releases/tags/$MILES_WHEELS_TAG" \
          | python3 -c "
import sys, json
prefix = sys.argv[1]
for a in json.load(sys.stdin).get('assets', []):
    if a['name'].startswith(prefix):
        print(a['browser_download_url']); break
" "$prefix")
    if [[ -z "$url" ]]; then
        echo "FATAL: no wheel matching '$prefix*' in $MILES_WHEELS_REPO@$MILES_WHEELS_TAG" >&2
        return 1
    fi
    local name="${url##*/}"
    local bytes
    bytes=$(curl -fsLI "$url" 2>/dev/null | awk '/^[Cc]ontent-[Ll]ength:/ {print $2}' | tr -d '\r' | tail -1 || true)
    if [[ "$bytes" =~ ^[0-9]+$ ]]; then
        echo "[wheels] downloading $name (~$(( bytes / 1024 / 1024 )) MB)" >&2
    else
        echo "[wheels] downloading $name" >&2
    fi
    curl -fSL --retry 3 -o "$WHEELS_DIR/$MILES_WHEELS_TAG/$name" "$url" >&2
    echo "$WHEELS_DIR/$MILES_WHEELS_TAG/$name"
}

_install_cudnn_for_torch "before transformer_engine install"

# ---------- transformer_engine (prebuilt triplet from the wheels bundle) ----------
# docker/Dockerfile ENABLE_CUDA_13=1: the bundle ships transformer_engine, transformer_engine_cu13
# and a transformer_engine_torch built against this bundle's torch — install all three --no-deps
# (no source build, so nvcc is not needed here), restore the torch extension's runtime deps, then
# run the Dockerfile's own triplet checker and apply its cu13 TE patches (a patch that fails
# to apply must fail the install, not ship a silently unpatched TE).
echo "[deps] transformer_engine==$TE_VERSION triplet from $MILES_WHEELS_REPO@$MILES_WHEELS_TAG"
te_wheel=$(_fetch_orbit_wheel transformer_engine-)
te_core_wheel=$(_fetch_orbit_wheel transformer_engine_cu13-)
te_torch_wheel=$(_fetch_orbit_wheel transformer_engine_torch-)
$UV --no-deps --reinstall "$te_wheel" "$te_core_wheel" "$te_torch_wheel"
echo "[deps] einops + onnx>=1.21.0 + onnxscript + pydantic + nvdlfw-inspect (transformer_engine_torch runtime deps)"
$UV einops "onnx>=1.21.0" onnxscript pydantic nvdlfw-inspect
"$CONDA_PREFIX/bin/python" "$ORBIT_REPO/docker/verify_transformer_engine.py" transformer_engine_cu13
if compgen -G "$ORBIT_REPO/docker/patch/cu13/*.patch" >/dev/null; then
    te_dir=$("$CONDA_PREFIX/bin/python" -c 'import importlib.util; print(importlib.util.find_spec("transformer_engine").submodule_search_locations[0])')
    for te_patch in "$ORBIT_REPO"/docker/patch/cu13/*.patch; do
        echo "[deps] TE patch $(basename "$te_patch") -> $te_dir"
        patch -d "$te_dir" -p1 < "$te_patch" \
            || { echo "FATAL: TE patch $(basename "$te_patch") did not apply cleanly against the installed transformer_engine" >&2; exit 1; }
    done
fi

# ---------- prebuilt wheels (flash-attn, flash-attn-3, apex) -------------
# Source builds for these are 30-60 min each on Hopper. The Dockerfile uses
# prebuilt wheels from $MILES_WHEELS_REPO @ $MILES_WHEELS_TAG; we do the same.


if [[ "$INSTALL_FLASH_ATTN" == "1" ]]; then
    echo "[deps] flash-attn (FA2) — Megatron --attention-backend flash"
    fa_wheel=$(_fetch_orbit_wheel flash_attn-)
    $UV "$fa_wheel"
fi

if [[ "$INSTALL_FLASH_ATTN_3" == "1" ]]; then
    echo "[deps] flash-attn-3 (FA3, Hopper) — Megatron attention.py auto-detects HAVE_FA3"
    fa3_wheel=$(_fetch_orbit_wheel flash_attn_3-)
    $UV "$fa3_wheel"
    # The FA3 wheel ships the .so but NOT the python interface module that
    # Megatron imports from. Drop it in (matches docker/Dockerfile pattern).
    fa3_dir=$(python -c "import site; print(site.getsitepackages()[0])")/flash_attn_3
    mkdir -p "$fa3_dir"
    if [[ ! -f "$fa3_dir/flash_attn_interface.py" ]]; then
        curl -fSL --retry 3 -o "$fa3_dir/flash_attn_interface.py" \
            "https://raw.githubusercontent.com/Dao-AILab/flash-attention/$FLASH_ATTN_INTERFACE_COMMIT/hopper/flash_attn_interface.py"
    fi
fi

if [[ "$INSTALL_APEX" == "1" ]]; then
    echo "[deps] apex — Megatron FusedAdam / FastLayerNorm / multi_tensor_applier"
    apex_wheel=$(_fetch_orbit_wheel apex-)
    $UV "$apex_wheel"
fi

# sglang_router from the SAME miles-wheels release as FA/apex — NOT from PyPI.
# The release wheel may be a radixark/patched build; a PyPI `sglang-router==X`
# can share the version number yet diverge from what the upstream Dockerfile
# installs (it COPYs /tmp/wheels/sglang_router-*.whl from this release). orbit
# code version-gates on sglang_router.__version__, so provenance matters. Rust/
# abi3 wheel — NOT torch-ABI-bound — installed here (after the editable sglang
# above) so it wins over any sglang_router pulled transitively from an index.
router_wheel=$(_fetch_orbit_wheel sglang_router-)
router_wheel_base=$(basename "$router_wheel")
case "$router_wheel_base" in
    sglang_router-"$SGLANG_ROUTER_VERSION"-*) : ;;
    *)
        echo "FATAL: release sglang_router wheel $router_wheel_base does not match pinned $SGLANG_ROUTER_VERSION." >&2
        echo "       Update WHEELS_STACK in extract_pins.py for MILES_WHEELS_TAG=$MILES_WHEELS_TAG," >&2
        echo "       or use a miles-wheels release that ships sglang_router-$SGLANG_ROUTER_VERSION." >&2
        exit 1
        ;;
esac

# GLIBC guard: the release wheel is manylinux_2_NN (built on the docker base, e.g.
# Ubuntu 24.04 / GLIBC 2.39). On an older host uv refuses to install it AND the abi3
# .so would fail to load at runtime. When the host GLIBC is below the wheel's floor,
# build the patched router from source (radixark/sgl-router-for-orbit — the Dockerfile's
# SGL_ROUTER_USE_WHEELS=0 path) so the radixark patches survive on an older-GLIBC host.
# (Same root cause as the sgl-model-gateway GLIBC skip below; the router is essential so
# we build rather than skip.)
wheel_glibc_minor=$(sed -n 's/.*manylinux_2_\([0-9]\{1,\}\)_.*/\1/p' <<<"$router_wheel_base")
host_glibc_minor=$(ldd --version 2>/dev/null | awk 'NR==1{print $NF}' | cut -d. -f2)
SGL_ROUTER_REPO=${SGL_ROUTER_REPO:-https://github.com/radixark/sgl-router-for-orbit.git}
SGL_ROUTER_BRANCH=${SGL_ROUTER_BRANCH:-main}
if [[ -n "$wheel_glibc_minor" && -n "$host_glibc_minor" && "$host_glibc_minor" -lt "$wheel_glibc_minor" ]] \
   && ! git ls-remote --exit-code "$SGL_ROUTER_REPO" HEAD >/dev/null 2>&1; then
    # The Dockerfile's SGL_ROUTER_USE_WHEELS=0 source ($SGL_ROUTER_REPO) is not reachable from
    # here (repository not found / private), so the radixark-patched build cannot be rebuilt.
    # Last resort: PyPI's sglang-router at the pinned version — its wheel is manylinux_2_17, so
    # it loads on this host, but it is the upstream build, not the patched one the Dockerfile
    # ships. orbit's OFT rollout forces its own router (orbit/ray/rollout/router_manager.py)
    # and only imports sglang_router for RouterArgs, which the PyPI build provides.
    echo "[deps] sglang_router: release wheel needs GLIBC 2.$wheel_glibc_minor > host 2.$host_glibc_minor, and $SGL_ROUTER_REPO is not reachable" >&2
    echo "[deps] WARN: installing sglang-router==$SGLANG_ROUTER_VERSION from PyPI (upstream build, NOT the radixark-patched release wheel)" >&2
    $UV --force-reinstall "sglang-router==$SGLANG_ROUTER_VERSION"
elif [[ -n "$wheel_glibc_minor" && -n "$host_glibc_minor" && "$host_glibc_minor" -lt "$wheel_glibc_minor" ]]; then
    echo "[deps] sglang_router: release wheel needs GLIBC 2.$wheel_glibc_minor > host 2.$host_glibc_minor — building from source"
    command -v cargo >/dev/null \
        || { echo "FATAL: cargo not on PATH — needed to build sglang_router from source" >&2; exit 1; }
    router_src=$(mktemp -d)
    echo "[deps] cloning $SGL_ROUTER_REPO@$SGL_ROUTER_BRANCH"
    git clone --branch "$SGL_ROUTER_BRANCH" --depth 1 "$SGL_ROUTER_REPO" "$router_src/src"
    $UV maturin
    # maturin >=1.14 refuses readme paths that resolve outside the package dir;
    # the router's bindings/python/pyproject.toml points at ../../README.md.
    # Copy the file in place and rewrite the reference so metadata is self-contained.
    router_pyproject="$router_src/src/bindings/python/pyproject.toml"
    if grep -qE '^readme = "\.\./\.\./README\.md"' "$router_pyproject"; then
        cp "$router_src/src/README.md" "$router_src/src/bindings/python/README.md"
        sed -i 's|^readme = "\.\./\.\./README\.md"|readme = "README.md"|' "$router_pyproject"
        echo "[deps] sglang_router: inlined ../../README.md (maturin metadata-root restriction)"
    fi
    ( cd "$router_src/src/bindings/python" && ulimit -n 65536 \
        && maturin build --release --features vendored-openssl --out "$router_src/wheels" )
    built_router=$(compgen -G "$router_src/wheels/sglang_router-*.whl" | head -1 || true)
    [[ -n "$built_router" ]] \
        || { echo "FATAL: maturin produced no sglang_router wheel under $router_src/wheels" >&2; exit 1; }
    case "$(basename "$built_router")" in
        sglang_router-"$SGLANG_ROUTER_VERSION"-*) : ;;
        *) echo "[deps] WARN: built $(basename "$built_router") != pinned $SGLANG_ROUTER_VERSION (source '$SGL_ROUTER_BRANCH' moved); proceeding (from-source GLIBC fallback)" >&2 ;;
    esac
    echo "[deps] sglang_router <- $(basename "$built_router") (built from source)"
    $UV --force-reinstall "$built_router"
    rm -rf "$router_src"
else
    echo "[deps] sglang_router <- $router_wheel_base (release wheel, matches Dockerfile source)"
    $UV "$router_wheel"
fi

# ---------- sgl-model-gateway binary -------------------------------------
# Standalone Rust binary that fronts multiple sglang servers (multi-replica
# disagg rollout routing). Docker drops it into /usr/local/bin/; bare-metal we
# install into the conda env's bin/ so `conda activate $ORBIT_ENV_NAME` picks
# it up via PATH.
#
# The miles-wheels prebuild is linked against GLIBC 2.38+ (Ubuntu 24.04 base).
# Slinky's compute nodes are Ubuntu 22.04 (GLIBC 2.35) — the prebuilt binary
# will refuse to start. We detect that here and skip with a clear note; build
# from source via SGL_ROUTER_USE_WHEELS=0 path in the Dockerfile if needed.

INSTALL_SGL_GATEWAY=${INSTALL_SGL_GATEWAY:-1}
if [[ "$INSTALL_SGL_GATEWAY" == "1" ]]; then
    GLIBC_VER=$(ldd --version 2>/dev/null | awk 'NR==1{print $NF}')
    GLIBC_MAJOR=${GLIBC_VER%%.*}
    GLIBC_MINOR=${GLIBC_VER#*.}
    GLIBC_MINOR=${GLIBC_MINOR%%.*}
    if [[ "$GLIBC_MAJOR" -ge 2 && "$GLIBC_MINOR" -ge 38 ]]; then
        echo "[deps] sgl-model-gateway binary -> \$CONDA_PREFIX/bin/"
        gateway_tarball=$(_fetch_orbit_wheel sgl-model-gateway-linux-)
        tar xzf "$gateway_tarball" -C "$CONDA_PREFIX/bin/"
        chmod +x "$CONDA_PREFIX/bin/sgl-model-gateway"
    else
        # EXPLICIT HOST EXCEPTION (not an sglang-stack pin issue): the gateway is a
        # standalone Rust binary, NOT torch-ABI-bound, so it's exempt from the ABI
        # guards above. It's skipped purely because this host's GLIBC predates the
        # prebuilt's 2.38 floor. Only needed for multi-server sglang routing.
        echo "[deps] sgl-model-gateway: skipping — host GLIBC $GLIBC_VER < 2.38 (prebuilt needs 2.38+)"
        echo "[deps] sgl-model-gateway: build from source (SGL_ROUTER_USE_WHEELS=0 path) if you need routing"
    fi
fi

# ---------- orbit itself + its python-only requirements ------------------

echo "[deps] requirements.txt"
# nvidia-resiliency-ext (upstream FT stack, #1598) ships manylinux_2_39-only
# wheels; this cluster's glibc is 2.35 and the FT features are flag-gated and
# unused here (orbit references the package only in a docstring). Filter it
# out below that glibc rather than failing the whole install.
glibc_minor=$(getconf GNU_LIBC_VERSION | awk '{split($2, v, "."); print v[2]}')
if [ "${glibc_minor:-0}" -ge 39 ]; then
    $UV -r "$ORBIT_REPO/requirements.txt"
else
    echo "[deps] glibc 2.${glibc_minor} < 2.39 — installing requirements.txt without nvidia-resiliency-ext (FT-only, manylinux_2_39 wheels)"
    _req_filtered=$(mktemp /tmp/orbit-requirements-XXXX.txt)
    grep -v '^nvidia-resiliency-ext' "$ORBIT_REPO/requirements.txt" > "$_req_filtered"
    $UV -r "$_req_filtered"
    rm -f "$_req_filtered"
fi

echo "[deps] orbit editable"
$UV -e "$ORBIT_REPO" --no-deps

# numpy: upstream dropped its late `pip install "numpy<2"` at the sglang-v0.5.14
# bump (#1587) — the current Megatron/sglang line runs on numpy 2.x, so the old
# numpy<2 + scipy<1.16 pairing cap is gone with it (2026-07-24 sync decision).

if [[ "${glibc_minor:-0}" -ge 39 ]]; then
    # docker/Dockerfile cu13: the bundle's mooncake_transfer_engine_cuda13 wheel carries the
    # structured-object-store API orbit's Mooncake backend needs. manylinux_2_39 -> GLIBC >= 2.39 only.
    mk_wheel=$(_fetch_orbit_wheel mooncake_transfer_engine)
    echo "[deps] mooncake <- $(basename "$mk_wheel") (bundle wheel, structured object store)"
    $UV --no-deps --reinstall "$mk_wheel"
else
    echo "[deps] mooncake-transfer-engine==$MOONCAKE_VERSION (PyPI base wheel; the bundle's cu13 structured-object-store wheel needs GLIBC >= 2.39, host has 2.${glibc_minor:-?})"
    $UV "mooncake-transfer-engine==$MOONCAKE_VERSION"
    # PyPI's mooncake is a CUDA-12 build: engine.so NEEDs libcudart.so.12 with an $ORIGIN-only
    # RPATH, and a cu13 env carries only libcudart.so.13. Give it the cu12 runtime from the
    # nvidia wheel and preload that library at interpreter start through a .pth (the same
    # mechanism torch uses for its nvidia libs), so `import mooncake.engine` works in every
    # process of this env without LD_LIBRARY_PATH. This mixes libcudart.so.12 and .so.13 in
    # one process — the known trade-off (precedent: the miles-deploy env) until a cu13 mooncake
    # wheel exists for this host's GLIBC (the bundle's needs GLIBC_2.38 symbols).
    MOONCAKE_CUDART12_VERSION=${MOONCAKE_CUDART12_VERSION:-12.9.79}
    echo "[deps] nvidia-cuda-runtime-cu12==$MOONCAKE_CUDART12_VERSION + orbit-mooncake-cudart12.pth (PyPI mooncake is a CUDA-12 build)"
    $UV "nvidia-cuda-runtime-cu12==$MOONCAKE_CUDART12_VERSION"
    cat > "$PY_SITE/orbit-mooncake-cudart12.pth" <<'EOF'
import ctypes, os, sysconfig; _p = os.path.join(sysconfig.get_paths()["purelib"], "nvidia", "cuda_runtime", "lib", "libcudart.so.12"); os.path.exists(_p) and ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
EOF
fi

_install_cudnn_for_torch "final pre-verify reassert"

# ---------- smoke test + version audit -----------------------------------
# verify_env.py runs the import/CUDA/FA3-symbol checks the old inline heredoc
# did, plus cross-checks installed versions against pins.env and confirms the
# editable installs point at thirdparty/. Failures here fail the install.

_prepend_env_cuda_libs
python3 "$SCRIPT_DIR/verify_env.py"

echo
echo "[done] orbit env ready: $CONDA_PREFIX"
echo "[done] activate:  source $CONDA_ROOT/etc/profile.d/conda.sh && conda activate $CONDA_PREFIX"
echo "[done] PYTHONPATH for train.py: $MEGATRON_SRC"
