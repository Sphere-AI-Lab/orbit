#!/usr/bin/env bash
set -euo pipefail

: "${PUBLIC_ORBIT_URL:?set PUBLIC_ORBIT_URL to the public Orbit Git URL}"

WORK_ROOT=${WORK_ROOT:-$(mktemp -d)}
CHECKOUT_DIR="${WORK_ROOT}/orbit-public-gate"

git clone "${PUBLIC_ORBIT_URL}" "${CHECKOUT_DIR}"
cd "${CHECKOUT_DIR}"

uv python pin 3.12
uv sync

uv run python - <<'PY'
import importlib.metadata as md
import torch
import cuda.bindings
import orbit
import megatron.bridge
import megatron.core
import sglang

print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda-python", md.version("cuda-python"))
print("orbit", md.version("orbit"))
print("megatron-bridge", md.version("megatron-bridge"))
print("megatron-core", md.version("megatron-core"))
print("sglang", md.version("sglang"))
print("imports ok")
PY

uv run pytest tests/fast/productization -q
uv run pytest tests/fast/scripts/test_release_launcher_smoke.py -q
find examples scripts -type f -name '*.sh' -print0 | xargs -0 -n1 bash -n
