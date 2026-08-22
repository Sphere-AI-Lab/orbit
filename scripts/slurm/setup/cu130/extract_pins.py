#!/usr/bin/env python3
"""Generate the Orbit CUDA 13 native installation contract.

RadixArk Miles supplies the pinned prebuilt binary recipe. Orbit metadata
supplies Python, PyTorch, and the Sphere-Lab source revisions layered over it.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import os
import re
import sys
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
PYPROJECT = REPO_ROOT / "pyproject.toml"
PINS_FILE = SCRIPT_DIR / "pins.env"
RADIXARK_MILES_COMMIT = "128cdfb99ba4816eb01eee01e77aed767296ed25"
DEFAULT_DOCKERFILE_URL = (
    "https://raw.githubusercontent.com/radixark/miles/"
    + RADIXARK_MILES_COMMIT
    + "/docker/Dockerfile"
)


def fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "orbit-cu130-pins"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode()


def one(text: str, pattern: str, label: str) -> str:
    found = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if not found:
        raise SystemExit(f"FATAL: cannot extract {label}; source layout changed")
    return found.group(1)


def exact_requirement(text: str, name: str) -> str:
    return one(text, rf'"{re.escape(name)}==([^"]+)"', name)


def backend(text: str, name: str) -> tuple[str, str, str]:
    block = one(
        text,
        rf"\[tool\.orbit\.release\.backend-pins\.{re.escape(name)}\](.*?)(?=\n\[|\Z)",
        name,
    )
    source = one(block, r'^source\s*=\s*"([^"]+)"', name + " source")
    revision = one(block, r'^tested-ref\s*=\s*"([^"]+)"', name + " tested-ref")
    subdirectory_match = re.search(r'^subdirectory\s*=\s*"([^"]+)"', block, re.MULTILINE)
    return source, revision, subdirectory_match.group(1) if subdirectory_match else ""


def extract(dockerfile: str, dockerfile_url: str) -> dict[str, str]:
    project_bytes = PYPROJECT.read_bytes()
    project = project_bytes.decode()
    sglang_repo, sglang_commit, sglang_subdir = backend(project, "sglang")
    megatron_repo, megatron_commit, _ = backend(project, "megatron-core")
    bridge_repo, bridge_commit, _ = backend(project, "megatron-bridge")
    image_tag = one(dockerfile, r"^ARG SGLANG_IMAGE_TAG=(\S+)", "SGLANG_IMAGE_TAG")
    sglang_pyproject_url = (
        "https://raw.githubusercontent.com/sgl-project/sglang/"
        + image_tag
        + "/python/pyproject.toml"
    )
    sglang_project = fetch(sglang_pyproject_url)
    kernel_version = exact_requirement(project, "sglang-kernel")

    return {
        "RADIXARK_MILES_COMMIT": RADIXARK_MILES_COMMIT,
        "RADIXARK_DOCKERFILE_URL": dockerfile_url,
        "RADIXARK_DOCKERFILE_SHA256": hashlib.sha256(dockerfile.encode()).hexdigest(),
        "ORBIT_PYPROJECT_SHA256": hashlib.sha256(project_bytes).hexdigest(),
        "SGLANG_PYPROJECT_URL": sglang_pyproject_url,
        "SGLANG_PYPROJECT_SHA256": hashlib.sha256(sglang_project.encode()).hexdigest(),
        "PYTHON_VERSION": one(project, r'requires-python\s*=\s*">=(\d+\.\d+)', "Python version"),
        "TORCH_VERSION": exact_requirement(project, "torch"),
        "TORCHVISION_VERSION": exact_requirement(project, "torchvision"),
        "TORCHAUDIO_VERSION": exact_requirement(project, "torchaudio"),
        "TRITON_VERSION": exact_requirement(project, "triton"),
        "CUDA_PYTHON_VERSION": exact_requirement(project, "cuda-python"),
        "SGLANG_IMAGE_TAG": image_tag,
        "SGLANG_BASE_VERSION": image_tag.removeprefix("v"),
        "SGLANG_KERNEL_VERSION": kernel_version,
        "SGLANG_KERNEL_WHEEL_URL": (
            "https://github.com/sgl-project/whl/releases/download/v"
            + kernel_version
            + "/sglang_kernel-"
            + kernel_version
            + "+cu130-cp310-abi3-manylinux2014_x86_64.whl"
        ),
        "SGL_DEEP_GEMM_VERSION": exact_requirement(sglang_project, "sgl-deep-gemm"),
        "SGLANG_ROUTER_VERSION": exact_requirement(project, "sglang-router"),
        "MILES_WHEELS_REPO": one(dockerfile, r"^ARG WHEELS_REPO=(\S+)", "WHEELS_REPO"),
        "MILES_WHEELS_TAG": one(dockerfile, r"^ARG WHEELS_TAG_X86=(\S+)", "WHEELS_TAG_X86"),
        "TRANSFORMER_ENGINE_VERSION": one(
            dockerfile, r"transformer_engine-([0-9][^-]+)-py3-none-any\.whl", "Transformer Engine"
        ),
        "CUDNN_CU13_VERSION": one(
            dockerfile, r"nvidia-cudnn-cu13==([0-9.]+)", "nvidia-cudnn-cu13"
        ),
        "CUTLASS_DSL_VERSION": one(
            dockerfile, r'"nvidia-cutlass-dsl==([0-9.]+)"', "nvidia-cutlass-dsl"
        ),
        "FLASHINFER_VERSION": one(
            dockerfile, r'"flashinfer-python==([0-9A-Za-z.]+)"', "FlashInfer"
        ),
        "APACHE_TVM_FFI_VERSION": one(
            dockerfile, r'"apache-tvm-ffi==([0-9.]+)"', "apache-tvm-ffi"
        ),
        "TORCH_MEMORY_SAVER_COMMIT": one(
            dockerfile, r"torch_memory_saver\.git@([0-9a-f]{40})", "torch-memory-saver"
        ),
        "ORBIT_SGLANG_REPO": sglang_repo,
        "ORBIT_SGLANG_COMMIT": sglang_commit,
        "ORBIT_SGLANG_SUBDIRECTORY": sglang_subdir or "python",
        "ORBIT_MEGATRON_REPO": megatron_repo,
        "ORBIT_MEGATRON_COMMIT": megatron_commit,
        "ORBIT_MEGATRON_BRIDGE_REPO": bridge_repo,
        "ORBIT_MEGATRON_BRIDGE_COMMIT": bridge_commit,
    }


ORDER = (
    "RADIXARK_MILES_COMMIT",
    "RADIXARK_DOCKERFILE_URL",
    "RADIXARK_DOCKERFILE_SHA256",
    "ORBIT_PYPROJECT_SHA256",
    "SGLANG_PYPROJECT_URL",
    "SGLANG_PYPROJECT_SHA256",
    "PYTHON_VERSION",
    "TORCH_VERSION",
    "TORCHVISION_VERSION",
    "TORCHAUDIO_VERSION",
    "TRITON_VERSION",
    "CUDA_PYTHON_VERSION",
    "SGLANG_IMAGE_TAG",
    "SGLANG_BASE_VERSION",
    "SGLANG_KERNEL_VERSION",
    "SGLANG_KERNEL_WHEEL_URL",
    "SGL_DEEP_GEMM_VERSION",
    "SGLANG_ROUTER_VERSION",
    "MILES_WHEELS_REPO",
    "MILES_WHEELS_TAG",
    "TRANSFORMER_ENGINE_VERSION",
    "CUDNN_CU13_VERSION",
    "CUTLASS_DSL_VERSION",
    "FLASHINFER_VERSION",
    "APACHE_TVM_FFI_VERSION",
    "TORCH_MEMORY_SAVER_COMMIT",
    "ORBIT_SGLANG_REPO",
    "ORBIT_SGLANG_COMMIT",
    "ORBIT_SGLANG_SUBDIRECTORY",
    "ORBIT_MEGATRON_REPO",
    "ORBIT_MEGATRON_COMMIT",
    "ORBIT_MEGATRON_BRIDGE_REPO",
    "ORBIT_MEGATRON_BRIDGE_COMMIT",
)


def render(values: dict[str, str]) -> str:
    lines = [
        "# AUTO-GENERATED by extract_pins.py --write. Do not edit.",
        "# Binary pins: radixark/miles Dockerfile at RADIXARK_MILES_COMMIT.",
        "# Source pins: Orbit pyproject.toml backend metadata.",
        "",
    ]
    opening = "$" + "{"
    for key in ORDER:
        value = values[key]
        if re.search(r"\s", value):
            raise SystemExit(f"FATAL: shell-unsafe whitespace in {key}: {value!r}")
        lines.append(key + "=" + opening + key + ":-" + value + "}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    parser.add_argument("--dockerfile", type=Path)
    parser.add_argument(
        "--dockerfile-url",
        default=os.environ.get("RADIXARK_DOCKERFILE_URL", DEFAULT_DOCKERFILE_URL),
    )
    args = parser.parse_args()

    if args.dockerfile:
        dockerfile = args.dockerfile.read_text()
        source = str(args.dockerfile.resolve())
    else:
        dockerfile = fetch(args.dockerfile_url)
        source = args.dockerfile_url
    rendered = render(extract(dockerfile, source))

    if args.write:
        PINS_FILE.write_text(rendered)
        print("[pins] wrote " + str(PINS_FILE.relative_to(REPO_ROOT)), file=sys.stderr)
        return 0
    if args.check:
        if not PINS_FILE.exists():
            print("FATAL: pins.env is missing", file=sys.stderr)
            return 1
        current = PINS_FILE.read_text()
        if current != rendered:
            print("FATAL: pins.env is stale", file=sys.stderr)
            sys.stderr.writelines(
                difflib.unified_diff(
                    current.splitlines(keepends=True),
                    rendered.splitlines(keepends=True),
                    fromfile="pins.env",
                    tofile="fresh extraction",
                )
            )
            return 1
        print("[pins] current", file=sys.stderr)
        return 0
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
