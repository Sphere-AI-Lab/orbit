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
#   - transformer_engine[pytorch] — Megatron training requires it
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
SGLANG_SRC="$THIRDPARTY_DIR/sglang"
MEGATRON_BRIDGE_SRC="$THIRDPARTY_DIR/Megatron-Bridge"

# Fail closed on submodule ↔ ACTIVE-pins mismatch (needs the actual submodule
# HEAD, which only git can see — complements the file-derivable preflight ABI
# guard up top). The sglang line just checked out must match the wheels bundle
# the ACTIVE pins describe, else we'd build sglang vX but install vY's torch-ABI
# wheels. Skipped if `git describe` finds no tag (shallow/odd clone).
sub_sglang_base=$(git -C "$SGLANG_SRC" describe --tags --abbrev=0 2>/dev/null || echo "")
if [[ -n "${MILES_WHEELS_SGLANG_VERSION:-}" && -n "$sub_sglang_base" \
      && "$sub_sglang_base" != "$MILES_WHEELS_SGLANG_VERSION" ]]; then
    echo "FATAL: thirdparty/sglang is at $sub_sglang_base but pins.env expects $MILES_WHEELS_SGLANG_VERSION" >&2
    echo "       (MILES_WHEELS_TAG=$MILES_WHEELS_TAG). The wheels won't match the sglang you build." >&2
    echo "       Run sglang-sync to realign, or set MILES_WHEELS_TAG to the matching release." >&2
    exit 1
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

# ---------- torch (pinned, cu129) ----------------------------------------

echo "[torch] torch==$TORCH_VERSION + torchvision from $TORCH_INDEX_URL"
$UV --index-url "$TORCH_INDEX_URL" "torch==$TORCH_VERSION" torchvision

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
PY_SITE=$(python -c "import site; print(site.getsitepackages()[0])")
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
$UV -e "$SGLANG_SRC/python[all]" \
    --extra-index-url "$FLASHINFER_INDEX_URL" \
    --extra-index-url "$TORCH_INDEX_URL" \
    --index-strategy unsafe-first-match

# sglang_router is installed from the miles-wheels release (NOT PyPI) in the
# prebuilt-wheels section below, to match the upstream Dockerfile's wheel source.

# ---------- recipe-required runtime deps (not in requirements.txt) -------

echo "[deps] mbridge @ $MBRIDGE_COMMIT (for tools/convert_hf_to_torch_dist.py)"
$UV "git+https://github.com/ISEEKYAN/mbridge.git@$MBRIDGE_COMMIT" --no-deps

echo "[deps] nvidia-modelopt — required by megatron.bridge's auto_bridge.py top-level import"
# Dockerfile has the [torch] extra but modelopt 0.44+ dropped it (warning-only).
# Side effect: pulls nvidia-cudnn-cu12==9.10.2.21 which clobbers the 9.16.0.29
# pin we need for pytorch/pytorch#168167; the cudnn step below restores it.
$UV --no-build-isolation "nvidia-modelopt>=0.37.0"

echo "[src] installing Megatron-Bridge editable from $MEGATRON_BRIDGE_SRC (--no-deps)"
$UV -e "$MEGATRON_BRIDGE_SRC" --no-deps --no-build-isolation

echo "[deps] torch_memory_saver @ $TMS_COMMIT (for miles/backends/megatron_utils/actor.py)"
$UV --no-cache-dir --force-reinstall "git+https://github.com/fzyzcjy/torch_memory_saver.git@$TMS_COMMIT"

echo "[deps] transformer_engine[pytorch]==$TE_VERSION (compiles, ~10 min)"
$UV --no-build-isolation "transformer_engine[pytorch]==$TE_VERSION"

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
echo "[deps] sglang_router <- $(basename "$router_wheel") (release wheel, matches Dockerfile source)"
case "$(basename "$router_wheel")" in
    sglang_router-"$SGLANG_ROUTER_VERSION"-*) : ;;
    *)
        echo "FATAL: release sglang_router wheel $(basename "$router_wheel") does not match pinned $SGLANG_ROUTER_VERSION." >&2
        echo "       Update WHEELS_STACK in extract_pins.py for MILES_WHEELS_TAG=$MILES_WHEELS_TAG," >&2
        echo "       or use a miles-wheels release that ships sglang_router-$SGLANG_ROUTER_VERSION." >&2
        exit 1
        ;;
esac
$UV "$router_wheel"

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
$UV -r "$MILES_REPO/requirements.txt"

echo "[deps] miles editable"
$UV -e "$MILES_REPO" --no-deps

# Megatron-Core insists on numpy <2.
$UV 'numpy<2'

# pytorch/pytorch#168167 — torch 2.9.x ships an older cudnn that segfaults on
# some cu12 setups; pin to the same cudnn the Dockerfile uses.
CUDNN_CU12_VERSION=${CUDNN_CU12_VERSION:-9.16.0.29}
echo "[deps] nvidia-cudnn-cu12==$CUDNN_CU12_VERSION (pytorch/pytorch#168167 workaround)"
$UV "nvidia-cudnn-cu12==$CUDNN_CU12_VERSION"

echo "[deps] mooncake-transfer-engine==$MOONCAKE_VERSION (sglang docker base)"
$UV "mooncake-transfer-engine==$MOONCAKE_VERSION"

# ---------- smoke test + version audit -----------------------------------
# verify_env.py runs the import/CUDA/FA3-symbol checks the old inline heredoc
# did, plus cross-checks installed versions against pins.env and confirms the
# editable installs point at thirdparty/. Failures here fail the install.

python3 "$SCRIPT_DIR/verify_env.py"

echo
echo "[done] miles env ready: $CONDA_PREFIX"
echo "[done] activate:  source $CONDA_ROOT/etc/profile.d/conda.sh && conda activate $MILES_ENV_NAME"
echo "[done] PYTHONPATH for train.py: $MEGATRON_SRC"
