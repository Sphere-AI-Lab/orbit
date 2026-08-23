#!/usr/bin/env bash
# Install Orbit on CUDA 13 from prebuilt wheels plus editable Sphere-Lab sources.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
source "$SCRIPT_DIR/pins.env"

ENV_PREFIX=${ENV_PREFIX:-/fast/zqiu/orbit-iclr/orbit/envs/orbit-cu130-v1}
SOURCE_ROOT=${SOURCE_ROOT:-/fast/zqiu/orbit-iclr/orbit/sources/orbit-cu130-v1}
CACHE_DIR=${CACHE_DIR:-/fast/zqiu/orbit-iclr/orbit/cache/orbit-cu130-v1}
CONDA_EXE=${CONDA_EXE:-/home/zqiu/anaconda3/bin/conda}
UV_EXE=${UV_EXE:-/home/zqiu/.local/bin/uv}
TOOL_PYTHON=${TOOL_PYTHON:-/home/zqiu/anaconda3/bin/python}
JOBS=${JOBS:-32}
DRY_RUN=0
PREFLIGHT_ONLY=0

usage() {
    cat <<EOF
Usage: $0 [options]
  --env-prefix PATH
  --source-root PATH
  --cache-dir PATH
  --conda-exe PATH
  --uv-exe PATH
  --tool-python PATH
  --jobs N
  --preflight-only
  --dry-run
EOF
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --env-prefix) ENV_PREFIX=$2; shift 2 ;;
        --source-root) SOURCE_ROOT=$2; shift 2 ;;
        --cache-dir) CACHE_DIR=$2; shift 2 ;;
        --conda-exe) CONDA_EXE=$2; shift 2 ;;
        --uv-exe) UV_EXE=$2; shift 2 ;;
        --tool-python) TOOL_PYTHON=$2; shift 2 ;;
        --jobs) JOBS=$2; shift 2 ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "FATAL: unknown argument: $1" >&2; exit 2 ;;
    esac
done

WHEEL_DIR="$CACHE_DIR/miles-wheels/$MILES_WHEELS_TAG"
LOCK_DIR="$ENV_PREFIX.install.lock"

cat <<EOF
[plan] Orbit root:       $REPO_ROOT
[plan] Conda prefix:     $ENV_PREFIX
[plan] Sources:          $SOURCE_ROOT
[plan] Cache:            $CACHE_DIR
[plan] Miles recipe:     $RADIXARK_MILES_COMMIT
[plan] Miles wheels:     $MILES_WHEELS_REPO@$MILES_WHEELS_TAG
[plan] SGLang baseline:  $SGLANG_BASE_VERSION with sglang-kernel $SGLANG_KERNEL_VERSION+cu130
[plan] Sphere SGLang:    $ORBIT_SGLANG_COMMIT
[plan] Megatron-Core:    $ORBIT_MEGATRON_COMMIT
[plan] Megatron-Bridge:  $ORBIT_MEGATRON_BRIDGE_COMMIT
EOF
[ "$DRY_RUN" -eq 1 ] && exit 0

for executable in "$CONDA_EXE" "$UV_EXE" "$TOOL_PYTHON" git nvidia-smi; do
    command -v "$executable" >/dev/null 2>&1 || [ -x "$executable" ] || {
        echo "FATAL: missing executable: $executable" >&2
        exit 1
    }
done
"$TOOL_PYTHON" "$SCRIPT_DIR/extract_pins.py" --check
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
case "$gpu_name" in
    *H100*) ;;
    *) echo "FATAL: expected H100, got $gpu_name" >&2; exit 1 ;;
esac
echo "[preflight] GPU=$gpu_name"
[ "$PREFLIGHT_ONLY" -eq 1 ] && exit 0

if [ -e "$LOCK_DIR" ]; then
    echo "FATAL: installer lock exists: $LOCK_DIR" >&2
    exit 1
fi
mkdir -p "$(dirname "$ENV_PREFIX")" "$SOURCE_ROOT" "$CACHE_DIR" "$WHEEL_DIR"
mkdir "$LOCK_DIR"
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

# uv requires working file locks; the MPI /fast filesystem does not provide them.
export UV_CACHE_DIR="${UV_CACHE_DIR:-${HOME}/.cache/orbit-cu130-v1/uv}"
export PIP_CACHE_DIR="$CACHE_DIR/pip"
export MAX_JOBS="$JOBS"

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
    if [ -e "$ENV_PREFIX" ] && [ -n "$(find "$ENV_PREFIX" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]; then
        echo "FATAL: target is not a recognizable Conda prefix: $ENV_PREFIX" >&2
        exit 1
    fi
    echo "[1/10] create Conda Python $PYTHON_VERSION prefix"
    "$CONDA_EXE" create -y -p "$ENV_PREFIX" "python=$PYTHON_VERSION" pip
else
    echo "[1/10] resume $ENV_PREFIX"
fi

PYTHON="$ENV_PREFIX/bin/python"
uv_install() {
    "$UV_EXE" pip install --python "$PYTHON" "$@"
}

echo "[2/10] install pinned PyTorch CUDA 13 foundation"
uv_install --upgrade \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION" \
    "triton==$TRITON_VERSION" \
    "cuda-python==$CUDA_PYTHON_VERSION"

echo "[3/10] install official prebuilt SGLang CUDA 13 baseline"
uv_install --force-reinstall --no-deps "$SGLANG_KERNEL_WHEEL_URL"
uv_install --prerelease=allow --only-binary=:all: "sglang==$SGLANG_BASE_VERSION"
uv_install --only-binary=:all: "sgl-deep-gemm==$SGL_DEEP_GEMM_VERSION"

echo "[4/10] download RadixArk Miles release assets"
"$PYTHON" - \
    "$MILES_WHEELS_REPO" \
    "$MILES_WHEELS_TAG" \
    "$WHEEL_DIR" \
    "$SCRIPT_DIR/miles-wheels-$MILES_WHEELS_TAG.sha256" <<'PY'
import hashlib
import os
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import quote

repo, tag, output, manifest = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4])
if not manifest.is_file():
    raise SystemExit(f"missing pinned asset manifest: {manifest}")
output.mkdir(parents=True, exist_ok=True)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            value.update(chunk)
    return value.hexdigest()


assets = []
for raw_line in manifest.read_text().splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    checksum, name = line.split(maxsplit=1)
    assets.append((checksum, name))
if not assets:
    raise SystemExit(f"empty pinned asset manifest: {manifest}")

timeout = int(os.environ.get("ORBIT_DOWNLOAD_TIMEOUT", "600"))
for expected, name in assets:
    target = output / name
    if target.exists() and digest(target) == expected:
        print("[cache] reuse " + target.name)
        continue
    if target.exists():
        print("[cache] discard checksum mismatch: " + target.name)
        target.unlink()
    temporary = Path(str(target) + ".part")
    url = (
        f"https://github.com/{repo}/releases/download/"
        f"{quote(tag, safe='')}/{quote(name, safe='')}"
    )
    print("[cache] download " + target.name)
    for attempt in range(1, 4):
        temporary.unlink(missing_ok=True)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "orbit-cu130-installer"})
            with urllib.request.urlopen(request, timeout=timeout) as source, temporary.open("wb") as sink:
                while chunk := source.read(8 * 1024 * 1024):
                    sink.write(chunk)
            break
        except Exception:
            temporary.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2**attempt)
    actual = digest(temporary)
    if actual != expected:
        temporary.unlink(missing_ok=True)
        raise SystemExit(f"checksum mismatch for {name}: expected {expected}, got {actual}")
    os.replace(temporary, target)
PY

pick_one() {
    matches=$(compgen -G "$1" || true)
    count=$(printf '%s\n' "$matches" | sed '/^$/d' | wc -l | tr -d ' ')
    if [ "$count" -ne 1 ]; then
        echo "FATAL: expected one prebuilt wheel matching $1; found $count" >&2
        return 1
    fi
    printf '%s\n' "$matches"
}

pick_optional() {
    matches=$(compgen -G "$1" || true)
    count=$(printf '%s\n' "$matches" | sed '/^$/d' | wc -l | tr -d ' ')
    if [ "$count" -gt 1 ]; then
        echo "FATAL: multiple wheels match $1" >&2
        return 1
    fi
    [ "$count" -eq 0 ] || printf '%s\n' "$matches"
}

install_optional() {
    wheel=$(pick_optional "$1")
    if [ -n "$wheel" ]; then
        uv_install --force-reinstall --no-deps "$wheel"
    fi
}

echo "[5/10] install prebuilt Miles Hopper wheels"
flash_attn_wheel=$(pick_one "$WHEEL_DIR/flash_attn-*.whl")
flash_attn_3_wheel=$(pick_one "$WHEEL_DIR/flash_attn_3-*.whl")
uv_install --force-reinstall --no-deps "$flash_attn_wheel"
uv_install --force-reinstall --no-deps "$flash_attn_3_wheel"
transformer_engine_wheel=$(pick_one "$WHEEL_DIR/transformer_engine-$TRANSFORMER_ENGINE_VERSION-*.whl")
transformer_engine_cu13_wheel=$(pick_one "$WHEEL_DIR/transformer_engine_cu13-$TRANSFORMER_ENGINE_VERSION-*.whl")
transformer_engine_torch_wheel=$(pick_one "$WHEEL_DIR/transformer_engine_torch-$TRANSFORMER_ENGINE_VERSION-*.whl")
uv_install --force-reinstall --no-deps \
    "$transformer_engine_wheel" \
    "$transformer_engine_cu13_wheel" \
    "$transformer_engine_torch_wheel"
uv_install einops onnx onnxscript pydantic nvdlfw-inspect
apex_wheel=$(pick_one "$WHEEL_DIR/apex-*.whl")
uv_install --force-reinstall --no-deps "$apex_wheel"
install_optional "$WHEEL_DIR/fast_hadamard_transform-*.whl"
install_optional "$WHEEL_DIR/causal_conv1d-*.whl"
install_optional "$WHEEL_DIR/mamba_ssm-*.whl"
install_optional "$WHEEL_DIR/deep_ep-*.whl"
install_optional "$WHEEL_DIR/ring_flash_attn-*.whl"
install_optional "$WHEEL_DIR/sglang_router-*.whl"
install_optional "$WHEEL_DIR/mooncake_transfer_engine_cuda13-*.whl"

echo "[6/10] reconcile SGLang CUDA runtime pins"
uv_install --no-cache --link-mode copy --force-reinstall --no-deps \
    "https://flashinfer.ai/whl/flashinfer-python/flashinfer_python-${FLASHINFER_VERSION}-py3-none-any.whl"
uv_install --force-reinstall --no-deps \
    --extra-index-url https://flashinfer.ai/whl \
    --extra-index-url https://flashinfer.ai/whl/cu130 \
    "flashinfer-cubin==$FLASHINFER_VERSION" \
    "flashinfer-jit-cache==$FLASHINFER_VERSION"
uv_install --force-reinstall --no-deps \
    "apache-tvm-ffi==$APACHE_TVM_FFI_VERSION" \
    "nvidia-cutlass-dsl==$CUTLASS_DSL_VERSION" \
    "nvidia-cutlass-dsl-libs-base==$CUTLASS_DSL_VERSION" \
    "nvidia-cutlass-dsl-libs-core==$CUTLASS_DSL_VERSION" \
    "nvidia-cutlass-dsl-libs-cu12==$CUTLASS_DSL_VERSION" \
    "nvidia-cutlass-dsl-libs-cu13==$CUTLASS_DSL_VERSION" \
    "nvidia-cudnn-cu13==$CUDNN_CU13_VERSION"

ensure_checkout() {
    url=$1
    commit=$2
    destination=$3
    if [ -e "$destination" ]; then
        [ -d "$destination/.git" ] || { echo "FATAL: non-git source path: $destination" >&2; return 1; }
        [ -z "$(git -C "$destination" status --porcelain)" ] || {
            echo "FATAL: dirty source checkout: $destination" >&2
            return 1
        }
        [ "$(git -C "$destination" rev-parse HEAD)" = "$commit" ] || {
            echo "FATAL: source checkout at wrong commit: $destination" >&2
            return 1
        }
        return
    fi
    git clone --filter=blob:none --no-checkout "$url" "$destination"
    git -C "$destination" fetch --depth 1 origin "$commit"
    git -C "$destination" checkout --detach "$commit"
}

echo "[7/10] materialize exact Sphere-Lab sources"
SGLANG_SRC="$SOURCE_ROOT/sglang"
MEGATRON_SRC="$SOURCE_ROOT/Megatron-LM"
BRIDGE_SRC="$SOURCE_ROOT/Megatron-Bridge"
ensure_checkout "$ORBIT_SGLANG_REPO" "$ORBIT_SGLANG_COMMIT" "$SGLANG_SRC"
ensure_checkout "$ORBIT_MEGATRON_REPO" "$ORBIT_MEGATRON_COMMIT" "$MEGATRON_SRC"
ensure_checkout "$ORBIT_MEGATRON_BRIDGE_REPO" "$ORBIT_MEGATRON_BRIDGE_COMMIT" "$BRIDGE_SRC"

echo "[compat] compare Sphere-Lab sgl-kernel with upstream $SGLANG_IMAGE_TAG"
git -C "$SGLANG_SRC" fetch --depth 1 https://github.com/sgl-project/sglang.git "refs/tags/$SGLANG_IMAGE_TAG"
if ! git -C "$SGLANG_SRC" diff --quiet FETCH_HEAD "$ORBIT_SGLANG_COMMIT" -- sgl-kernel; then
    echo "FATAL: Sphere-Lab sgl-kernel differs from upstream $SGLANG_IMAGE_TAG; refusing the prebuilt wheel" >&2
    exit 1
fi
echo "[compat] sgl-kernel subtree matches the prebuilt $SGLANG_KERNEL_VERSION+cu130 wheel"

echo "[8/10] install Orbit runtime dependencies"
RUNTIME_REQUIREMENTS="$CACHE_DIR/orbit-runtime-requirements.txt"
"$PYTHON" - "$REPO_ROOT/pyproject.toml" "$RUNTIME_REQUIREMENTS" <<'PY'
import re
import sys
import tomllib
from pathlib import Path

controlled = {
    "deep-ep",
    "megatron-bridge",
    "megatron-core",
    "nvidia-resiliency-ext",
    "ring-flash-attn",
    "sglang",
    "sglang-router",
    "transformer-engine",
}
data = tomllib.loads(Path(sys.argv[1]).read_text())
requirements = []
for requirement in data["project"]["dependencies"]:
    name = re.split(r"[<>=!~ \[]", requirement, maxsplit=1)[0].lower().replace("_", "-")
    if name not in controlled:
        requirements.append(requirement)
Path(sys.argv[2]).write_text("\n".join(requirements) + "\n")
PY
uv_install -r "$RUNTIME_REQUIREMENTS"
uv_install "nvidia-modelopt==0.44.0" "torch-memory-saver==0.0.9.post1"

echo "[9/10] install editable Sphere-Lab and Orbit overlays"
uv_install --reinstall -e "$MEGATRON_SRC" --no-deps
uv_install -e "$BRIDGE_SRC" --no-deps --no-build-isolation
# Reuse the prebuilt sglang_router wheel; do not compile Rust for the Python overlay.
SGLANG_BUILD_RUST_EXTS=none uv_install -e "$SGLANG_SRC/$ORBIT_SGLANG_SUBDIRECTORY" --no-deps
uv_install -e "$REPO_ROOT" --no-deps

echo "[10/10] reassert ABI pins and verify"
uv_install --no-deps \
    "torch==$TORCH_VERSION" \
    "torchvision==$TORCHVISION_VERSION" \
    "torchaudio==$TORCHAUDIO_VERSION" \
    "triton==$TRITON_VERSION" \
    "cuda-python==$CUDA_PYTHON_VERSION" \
    "nvidia-cudnn-cu13==$CUDNN_CU13_VERSION"
"$PYTHON" "$SCRIPT_DIR/verify_env.py" --source-root "$SOURCE_ROOT" --full-h100

echo "[done] activate with:"
echo "source /home/zqiu/anaconda3/etc/profile.d/conda.sh"
echo "conda activate $ENV_PREFIX"
