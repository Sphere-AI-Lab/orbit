#!/bin/bash
#
# install_env.sh — from-source conda env build for miles.
#
# Follows docs/getting-started/installation.md Method 2, extended with the
# transitive deps that miles-imp recipes actually need at runtime — the bare
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
#   - torch_memory_saver — miles/backends/megatron_utils/actor.py imports it
#   - transformer_engine + source-built torch extension — Megatron training requires it
#   - flash-attn 2/3   — `--attention-backend flash` and FA3-only Megatron paths
#
# Versions for the non-thirdparty deps (TE, flash-attn, mbridge, tms) match
# docker/Dockerfile pins. SGLang + Megatron versions come from the submodule
# pointers in .gitmodules.
#
# Target: H200/Hopper + CUDA 12. Blackwell (B200/GB300/B300) → switch to
# docker/Dockerfile's ENABLE_CUDA_13=1 path (TE 2.12/cu13, cudnn-cu13, cu13 wheels).
#
# Usage:
#   salloc --gres=gpu:1 --cpus-per-task=16 --mem=128G --time=2:00:00 --pty bash
#   bash scripts/slurm/setup/install_env.sh
#
# Knobs (env vars, all optional):
#   MILES_ENV_NAME    conda env name                       [miles]
#   MILES_PY_VERSION  python version                       [3.12]
#   MILES_REPO        this repo                            [$PWD]
#   THIRDPARTY_DIR    submodule dir                        [$MILES_REPO/thirdparty]
#   PULL_REMOTE       `git submodule update --remote`?     [0]  set 1 to bump to branch HEAD
#   CUDA_HOME         override system CUDA toolkit path    [auto-detected]
#   CUDNN_CU12_VERSION override torch-declared cuDNN pin   [derived from torch]
#   TE_BUILD_MAX_JOBS transformer_engine_torch build jobs  [16]
#   KERNELS_SPEC      transformers hub-kernels compat cap  [kernels>=0.12,<0.15]
#   SGLANG_SRC        external sglang checkout/worktree    [$THIRDPARTY_DIR/sglang]
#   SGL_WHL_INDEX_URL extra index for +cuNNN kernel wheels [unset]
#                     (e.g. https://docs.sglang.ai/whl/cu129 for sglang-kernel/
#                      sgl-deep-gemm +cu129 builds; PyPI defaults are cu13)
#   SGLANG_EXTRA_CONSTRAINT  extra uv constraint file for the sglang step [unset]

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# --- Knobs (paths, names, feature toggles) -------------------------------

MILES_ENV_NAME=${MILES_ENV_NAME:-miles}
MILES_PY_VERSION=${MILES_PY_VERSION:-3.12}
MILES_REPO=${MILES_REPO:-$PWD}
THIRDPARTY_DIR=${THIRDPARTY_DIR:-$MILES_REPO/thirdparty}
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
# _fetch_miles_wheel pulls a mismatched-ABI wheel set. WHEELS_STACK in extract_pins.py is
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
    && [[ -f "$MILES_REPO/docker/Dockerfile" ]] \
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

# Torch/flashinfer/prebuilt wheels are pinned to cu129 by default. Keep those
# CUDA build tags aligned, then make sure the driver advertises a compatible
# CUDA runtime before spending time on the install.
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
if [[ -n "$torch_cu_tag" && "${torch_cu_tag:0:2}" != "12" ]]; then
    echo "FATAL: this bare-metal setup mirrors Dockerfile's CUDA 12 path only; got TORCH_INDEX_URL=cu$torch_cu_tag" >&2
    echo "       Dockerfile's CUDA 13 variants also need TE 2.12/cu13 wheels, TE patches, and nvidia-cudnn-cu13." >&2
    exit 1
fi
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

# nvcc is required to compile transformer_engine from source.
# Prefer system CUDA toolkit; if absent, instruct the user (conda-install is
# heavy; we don't auto-install ~3 GB without consent).
if [[ -z "${CUDA_HOME:-}" ]]; then
    for cuda_dir in /usr/local/cuda-12.8 /usr/local/cuda-12.9 /usr/local/cuda; do
        if [[ -x "$cuda_dir/bin/nvcc" ]]; then
            export CUDA_HOME="$cuda_dir"
            break
        fi
    done
fi
if [[ -z "${CUDA_HOME:-}" ]] || [[ ! -x "$CUDA_HOME/bin/nvcc" ]]; then
    echo "FATAL: nvcc not found. transformer_engine needs it to compile." >&2
    echo "       Either:" >&2
    echo "         1. Set CUDA_HOME=/path/to/cuda before re-running, or" >&2
    echo "         2. conda install -n $MILES_ENV_NAME -c nvidia/label/cuda-12.9.0 cuda-nvcc cuda-cudart-dev cuda-libraries-dev" >&2
    exit 1
fi
export PATH="$CUDA_HOME/bin:$PATH"
echo "[preflight] nvcc: $(nvcc --version | tail -1)"
echo "[preflight] CUDA_HOME=$CUDA_HOME"

mkdir -p "$THIRDPARTY_DIR"

# ---------- submodules: init + fail-closed validation BEFORE env mutation -
# Init the source tree and validate the sglang/torch line HERE — before the
# torch install below mutates a (possibly reused) conda env. A wrong submodule
# vs MILES_WHEELS_TAG, or a TORCH_VERSION override that disagrees with the
# submodule's pyproject, must abort before we touch torch. Needs only git +
# pins.env values; no conda/torch dependency, so it is safe this early.
echo "[src] initialising submodules under $THIRDPARTY_DIR"
git -C "$MILES_REPO" submodule update --init --recursive \
    thirdparty/Megatron-LM thirdparty/sglang thirdparty/Megatron-Bridge

if [[ "$PULL_REMOTE" == "1" ]]; then
    echo "[src] PULL_REMOTE=1 — bumping submodules to branch HEAD"
    git -C "$MILES_REPO" submodule update --remote --recursive \
        thirdparty/Megatron-LM thirdparty/sglang thirdparty/Megatron-Bridge
fi

MEGATRON_SRC="$THIRDPARTY_DIR/Megatron-LM"
# SGLANG_SRC may point at an external sglang checkout/worktree (version-bump
# testing) without moving the production submodule tree that live envs run.
SGLANG_SRC=${SGLANG_SRC:-$THIRDPARTY_DIR/sglang}
MEGATRON_BRIDGE_SRC="$THIRDPARTY_DIR/Megatron-Bridge"

# Fail closed on submodule ↔ ACTIVE-pins mismatch (needs the actual submodule
# HEAD, which only git can see — complements the file-derivable preflight ABI
# guard up top). The sglang line just checked out must match the wheels bundle
# the ACTIVE pins describe, else we'd build sglang vX but install vY's torch-ABI
# wheels. Skipped if `git describe` finds no tag (shallow/odd clone).
sub_sglang_base=$(git -C "$SGLANG_SRC" describe --tags --abbrev=0 2>/dev/null || echo "")
if [[ -n "${MILES_WHEELS_SGLANG_VERSION:-}" && -n "$sub_sglang_base" \
      && "$sub_sglang_base" != "$MILES_WHEELS_SGLANG_VERSION" ]]; then
    # A lagging bundle is VALID when torch matches: the bundle wheels
    # (FA2/FA3/apex/router/gateway) are torch-ABI-bound, not sglang-version-bound —
    # upstream's own Dockerfile pairs the v0.5.13 sglang image with the v0.5.12
    # wheels bundle, and the pairing is validated bare-metal (2026-07 sync). The
    # torch pin check below is the real safety and stays FATAL.
    echo "[pins] WARN: sglang source is $sub_sglang_base but the wheels bundle is $MILES_WHEELS_SGLANG_VERSION" >&2
    echo "[pins]       (MILES_WHEELS_TAG=$MILES_WHEELS_TAG). OK while their torch matches (checked next)." >&2
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

if ! conda env list | awk '{print $1}' | grep -qx "$MILES_ENV_NAME"; then
    echo "[env] creating conda env '$MILES_ENV_NAME' (python=$MILES_PY_VERSION)"
    conda create -y -n "$MILES_ENV_NAME" "python=$MILES_PY_VERSION"
else
    echo "[env] reusing existing conda env '$MILES_ENV_NAME'"
fi
conda activate "$MILES_ENV_NAME"
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

_torch_declared_cudnn_cu12_version() {
    "$CONDA_PREFIX/bin/python" - <<'PY'
from importlib import metadata
from packaging.requirements import Requirement
import sys

for req_text in metadata.requires("torch") or []:
    req = Requirement(req_text)
    if req.name != "nvidia-cudnn-cu12":
        continue
    versions = [spec.version for spec in req.specifier if spec.operator == "=="]
    if len(versions) == 1:
        print(versions[0])
        raise SystemExit(0)
    raise SystemExit(f"FATAL: torch requirement {req_text!r} does not contain exactly one == pin")

raise SystemExit("FATAL: torch metadata does not declare nvidia-cudnn-cu12")
PY
}

_effective_cudnn_cu12_version() {
    if [[ -n "${CUDNN_CU12_VERSION:-}" ]]; then
        printf '%s\n' "$CUDNN_CU12_VERSION"
    else
        _torch_declared_cudnn_cu12_version
    fi
}

_install_cudnn_for_torch() {
    local context=${1:-"torch-declared pin"}
    local version
    version=$(_effective_cudnn_cu12_version)
    export CUDNN_CU12_VERSION="$version"
    _prepend_env_cuda_libs
    echo "[deps] nvidia-cudnn-cu12==$CUDNN_CU12_VERSION ($context)"
    $UV "nvidia-cudnn-cu12==$CUDNN_CU12_VERSION"
}

_prepend_env_cuda_libs

# ---------- torch (pinned, cu129) ----------------------------------------

echo "[torch] torch==$TORCH_VERSION + torchvision from $TORCH_INDEX_URL"
$UV --index-url "$TORCH_INDEX_URL" "torch==$TORCH_VERSION" torchvision
_install_cudnn_for_torch "after torch install; override with CUDNN_CU12_VERSION if needed"

# ---------- patched Megatron-LM + sglang (editable source installs) -------
# (submodules were init'd + validated above, before the torch install)

# Megatron-Core's pyproject deps are just `torch>=2.6.0, numpy, packaging`,
# all already satisfied. --no-deps avoids any chance of pip re-resolving torch.
echo "[src] installing Megatron-LM editable (--no-deps)"
$UV -e "$MEGATRON_SRC" --no-deps

# Megatron-LM's pyproject only declares `megatron.core*` as packages, so the
# editable finder does NOT expose `miles_megatron_plugins/` (top-level pkg
# inside the same repo, hard-imported from megatron/core/transformer/*.py).
# Drop a .pth file so `import miles_megatron_plugins` works in any python
# invocation without needing PYTHONPATH set.
echo "$MEGATRON_SRC" > "$PY_SITE/miles-megatron-source-root.pth"
echo "[src] miles-megatron-source-root.pth -> $MEGATRON_SRC"

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
    echo "[deps] installing libprotobuf + protobuf into $MILES_ENV_NAME (sglang-grpc build dep)"
    "$CONDA_ROOT/bin/conda" install -n "$MILES_ENV_NAME" -c conda-forge -y libprotobuf protobuf
fi

# torchcodec (sglang srt dep, torch-2.11-matched) dlopens FFmpeg shared libs at
# import; without them every engine start dumps a (non-fatal) probe traceback and
# video decode is unavailable. FFmpeg 7 pairs with torchcodec's core7 loader
# (libavutil.so.59). Docker gets ffmpeg from the base image; bare-metal installs it
# into the env, which conda activation puts on the loader path.
if [[ ! -e "$CONDA_PREFIX/lib/libavutil.so.59" ]]; then
    echo "[deps] installing ffmpeg 7 into $MILES_ENV_NAME (torchcodec runtime libs)"
    "$CONDA_ROOT/bin/conda" install -n "$MILES_ENV_NAME" -c conda-forge -y 'ffmpeg=7'
fi

# SGLang's pyproject declares the full runtime tree (fastapi/uvicorn/orjson/
# flashinfer_python/sglang-kernel/flash-attn-4/cuda-python/...). We DON'T use
# --no-deps because we're not starting from `lmsysorg/sglang:v0.5.10` — pip
# has to resolve those itself. extra-index-url is needed for flashinfer.
echo "[src] installing sglang editable from $SGLANG_SRC (full dep resolution)"
# unsafe-first-match: sglang declares torchao==0.9.0 which is only on PyPI, not
# on $TORCH_INDEX_URL or $FLASHINFER_INDEX_URL. Without this, uv refuses to
# look at PyPI for torchao (dependency-confusion guard). Order of indexes
# still controls preference (cu129/flashinfer before pypi); uv only falls
# through when earlier indexes do not provide a compatible version.
#
# cu12 sglang dependency reconciliation. Two coupled problems on a cu12 host (cu129
# prebuilt wheels + a CUDA-12.x driver that cannot run CUDA 13):
#   1. sglang's pyproject pins `cuda-python>=13.0` (CUDA 13). The cu129 sglang docker
#      variant (`lmsysorg/sglang:<ver>-cu129`) actually runs on the cu12.9 cuda-python
#      line, so the floor is wrong for the build we ship. A --constraint can't satisfy
#      `>=13` AND `<13`; an --override REPLACES the declared bound.
#   2. sglang pins `torch==$TORCH_VERSION` (no local tag); under unsafe-first-match uv will
#      otherwise swap the +cu$torch_cu_tag wheel for PyPI's plain torch (the CUDA-13 build
#      at >= 2.11), leaving torch(cu130) vs torchvision(cu$torch_cu_tag) — a CUDA-major
#      mismatch that breaks `import torchvision` (hence sglang). The +cu<tag> --constraint
#      pins the cu-index build (unsatisfiable by PyPI's plain wheel).
# Together: torch is held at +cu$torch_cu_tag, whose `cuda-bindings>=12.9.4,<13` requirement
# pulls cuda-python up to the 12.9.x line (12.6.x has no cuda-bindings dep, so `<13` alone
# under-selects) — the whole env lands CUDA-${torch_cu_tag}-consistent with the wheels.
# Applied only on cu12; a genuine CUDA-13 bundle (cu130 tag) leaves sglang's pins alone.
sglang_dep_args=()
if [[ "${torch_cu_tag:0:2}" == "12" ]]; then
    mkdir -p "$WHEELS_DIR"
    echo "cuda-python<13" > "$WHEELS_DIR/sglang-cuda-python-override.txt"
    echo "torch==${TORCH_VERSION}+cu${torch_cu_tag}" > "$WHEELS_DIR/torch-cu-constraint.txt"
    sglang_dep_args=(
        --override "$WHEELS_DIR/sglang-cuda-python-override.txt"
        --constraint "$WHEELS_DIR/torch-cu-constraint.txt"
    )
fi
# Caller-supplied extra constraints (e.g. pre-pinning resolver-backtrack-prone
# deps like flash-attn-4/quack-kernels/cuda-tile on a version-bump test).
if [[ -n "${SGLANG_EXTRA_CONSTRAINT:-}" ]]; then
    sglang_dep_args+=(--constraint "$SGLANG_EXTRA_CONSTRAINT")
fi
# Optional extra index for +cuNNN local-version kernel wheels (sglang-kernel /
# sgl-deep-gemm publish cu12 builds only at docs.sglang.ai/whl/cuNNN; the PyPI
# default wheels for those are cu13-linked).
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

# sglang_router is installed from the miles-wheels release (NOT PyPI) in the
# prebuilt-wheels section below, to match the upstream Dockerfile's wheel source.

# ---------- recipe-required runtime deps (not in requirements.txt) -------

echo "[deps] mbridge @ $MBRIDGE_COMMIT (for tools/convert_hf_to_torch_dist.py)"
$UV "git+https://github.com/ISEEKYAN/mbridge.git@$MBRIDGE_COMMIT" --no-deps

echo "[deps] nvidia-modelopt — required by megatron.bridge's auto_bridge.py top-level import"
# Dockerfile has the [torch] extra but modelopt 0.44+ dropped it (warning-only).
# Side effect: can pull an older nvidia-cudnn-cu12 and clobber torch's declared
# cuDNN pin; reassert the effective torch-derived/overridden cuDNN before TE.
$UV --no-build-isolation "nvidia-modelopt>=0.37.0"

echo "[src] installing Megatron-Bridge editable from $MEGATRON_BRIDGE_SRC (--no-deps)"
$UV -e "$MEGATRON_BRIDGE_SRC" --no-deps --no-build-isolation

echo "[deps] torch_memory_saver @ $TMS_COMMIT (for miles/backends/megatron_utils/actor.py)"
# Newer TMS source builds need TMS_CUDA_MAJOR (upstream #1774); derive from the env's torch.
TMS_CUDA_MAJOR=$("$CONDA_PREFIX/bin/python" -c "import torch; print(torch.version.cuda.split('.')[0])") \
    $UV --no-cache-dir --force-reinstall "git+https://github.com/fzyzcjy/torch_memory_saver.git@$TMS_COMMIT"

_install_cudnn_for_torch "before transformer_engine_torch source build"

echo "[deps] transformer_engine==$TE_VERSION + transformer_engine_cu12==$TE_VERSION"
# Do NOT install the `transformer_engine[pytorch]` extra here: pip/uv will prefer
# PyPI's prebuilt transformer_engine_torch wheel when one exists, and that wheel is
# torch-ABI-bound. For torch 2.11.0+cu129, TE 2.10.0's prebuilt torch extension
# imports with an undefined c10::cuda symbol. Mirror Docker's safer CUDA-13 shape:
# install the pure/runtime TE packages, then force the torch extension to compile
# against the torch already installed in this env.
$UV --no-deps "transformer_engine==$TE_VERSION" "transformer_engine_cu12==$TE_VERSION"

TE_BUILD_MAX_JOBS=${TE_BUILD_MAX_JOBS:-${MAX_JOBS:-16}}
echo "[deps] transformer_engine_torch==$TE_VERSION (source build, MAX_JOBS=$TE_BUILD_MAX_JOBS)"
_prepend_env_cuda_libs
MAX_JOBS="$TE_BUILD_MAX_JOBS" NVTE_FRAMEWORK=pytorch \
    $UV --no-deps --no-build-isolation --no-cache --no-binary :all: \
        --reinstall-package transformer_engine_torch \
        "transformer_engine_torch==$TE_VERSION"

echo "[deps] onnx>=1.21.0 + onnxscript (transformer_engine_torch runtime deps)"
# The source-build path intentionally uses --no-deps to avoid PyPI's prebuilt
# torch-ABI-bound transformer_engine_torch wheel, so restore the pure/runtime
# deps declared by transformer_engine_torch metadata explicitly.
$UV "onnx>=1.21.0" onnxscript

# ---------- prebuilt wheels (flash-attn, flash-attn-3, apex) -------------
# Source builds for these are 30-60 min each on Hopper. The Dockerfile uses
# prebuilt wheels from $MILES_WHEELS_REPO @ $MILES_WHEELS_TAG; we do the same.

mkdir -p "$WHEELS_DIR/$MILES_WHEELS_TAG"
_fetch_miles_wheel() {
    # Usage: _fetch_miles_wheel <basename-prefix>
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

if [[ "$INSTALL_FLASH_ATTN" == "1" ]]; then
    echo "[deps] flash-attn (FA2) — Megatron --attention-backend flash"
    fa_wheel=$(_fetch_miles_wheel flash_attn-)
    $UV "$fa_wheel"
fi

if [[ "$INSTALL_FLASH_ATTN_3" == "1" ]]; then
    echo "[deps] flash-attn-3 (FA3, Hopper) — Megatron attention.py auto-detects HAVE_FA3"
    fa3_wheel=$(_fetch_miles_wheel flash_attn_3-)
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
    apex_wheel=$(_fetch_miles_wheel apex-)
    $UV "$apex_wheel"
fi

# sglang_router from the SAME miles-wheels release as FA/apex — NOT from PyPI.
# The release wheel may be a radixark/patched build; a PyPI `sglang-router==X`
# can share the version number yet diverge from what the upstream Dockerfile
# installs (it COPYs /tmp/wheels/sglang_router-*.whl from this release). miles
# code version-gates on sglang_router.__version__, so provenance matters. Rust/
# abi3 wheel — NOT torch-ABI-bound — installed here (after the editable sglang
# above) so it wins over any sglang_router pulled transitively from an index.
router_wheel=$(_fetch_miles_wheel sglang_router-)
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
# build the patched router from source (radixark/sgl-router-for-miles — the Dockerfile's
# SGL_ROUTER_USE_WHEELS=0 path) so the radixark patches survive on an older-GLIBC host.
# (Same root cause as the sgl-model-gateway GLIBC skip below; the router is essential so
# we build rather than skip.)
wheel_glibc_minor=$(sed -n 's/.*manylinux_2_\([0-9]\{1,\}\)_.*/\1/p' <<<"$router_wheel_base")
host_glibc_minor=$(ldd --version 2>/dev/null | awk 'NR==1{print $NF}' | cut -d. -f2)
if [[ -n "$wheel_glibc_minor" && -n "$host_glibc_minor" && "$host_glibc_minor" -lt "$wheel_glibc_minor" ]]; then
    echo "[deps] sglang_router: release wheel needs GLIBC 2.$wheel_glibc_minor > host 2.$host_glibc_minor — building from source"
    SGL_ROUTER_REPO=${SGL_ROUTER_REPO:-https://github.com/radixark/sgl-router-for-miles.git}
    SGL_ROUTER_BRANCH=${SGL_ROUTER_BRANCH:-main}
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
# install into the conda env's bin/ so `conda activate $MILES_ENV_NAME` picks
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
        gateway_tarball=$(_fetch_miles_wheel sgl-model-gateway-linux-)
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

# ---------- miles itself + its python-only requirements ------------------

echo "[deps] requirements.txt"
# nvidia-resiliency-ext (upstream FT stack, #1598) ships manylinux_2_39-only
# wheels; this cluster's glibc is 2.35 and the FT features are flag-gated and
# unused here (miles references the package only in a docstring). Filter it
# out below that glibc rather than failing the whole install.
glibc_minor=$(getconf GNU_LIBC_VERSION | awk '{split($2, v, "."); print v[2]}')
if [ "${glibc_minor:-0}" -ge 39 ]; then
    $UV -r "$MILES_REPO/requirements.txt"
else
    echo "[deps] glibc 2.${glibc_minor} < 2.39 — installing requirements.txt without nvidia-resiliency-ext (FT-only, manylinux_2_39 wheels)"
    _req_filtered=$(mktemp /tmp/miles-requirements-XXXX.txt)
    grep -v '^nvidia-resiliency-ext' "$MILES_REPO/requirements.txt" > "$_req_filtered"
    $UV -r "$_req_filtered"
    rm -f "$_req_filtered"
fi

echo "[deps] miles editable"
$UV -e "$MILES_REPO" --no-deps

# numpy: upstream dropped its late `pip install "numpy<2"` at the sglang-v0.5.14
# bump (#1587) — the current Megatron/sglang line runs on numpy 2.x, so the old
# numpy<2 + scipy<1.16 pairing cap is gone with it (2026-07-24 sync decision).

echo "[deps] mooncake-transfer-engine==$MOONCAKE_VERSION (sglang docker base)"
$UV "mooncake-transfer-engine==$MOONCAKE_VERSION"

_install_cudnn_for_torch "final pre-verify reassert"

# ---------- smoke test + version audit -----------------------------------
# verify_env.py runs the import/CUDA/FA3-symbol checks the old inline heredoc
# did, plus cross-checks installed versions against pins.env and confirms the
# editable installs point at thirdparty/. Failures here fail the install.

_prepend_env_cuda_libs
python3 "$SCRIPT_DIR/verify_env.py"

echo
echo "[done] miles env ready: $CONDA_PREFIX"
echo "[done] activate:  source $CONDA_ROOT/etc/profile.d/conda.sh && conda activate $MILES_ENV_NAME"
echo "[done] PYTHONPATH for train.py: $MEGATRON_SRC"
