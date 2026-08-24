#!/usr/bin/env python3
"""Audit Orbit's native CUDA 13 environment and H100 runtime."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as metadata
import json
import os
import re
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]


def pins() -> dict[str, str]:
    values: dict[str, str] = {}
    opening = "$" + "{"
    pattern = re.compile(r"^([A-Z0-9_]+)=" + re.escape(opening) + r"\1:-(.+)\}$")
    for line in (SCRIPT_DIR / "pins.env").read_text().splitlines():
        found = pattern.match(line)
        if found:
            values[found.group(1)] = os.environ.get(found.group(1), found.group(2))
    return values


def version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def direct_url(package: str) -> dict:
    try:
        distribution = metadata.distribution(package)
    except metadata.PackageNotFoundError:
        return {}
    entry = next((item for item in distribution.files or [] if item.name == "direct_url.json"), None)
    return json.loads(entry.locate().read_text()) if entry else {}


def editable_at(package: str, expected: Path) -> bool:
    info = direct_url(package)
    url = info.get("url", "")
    return (
        info.get("dir_info", {}).get("editable") is True
        and url.startswith("file://")
        and os.path.realpath(url.removeprefix("file://")) == os.path.realpath(expected)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--full-h100", action="store_true")
    args = parser.parse_args()

    expected = pins()
    checks: list[tuple[str, bool, str]] = []

    def check(label: str, result: bool, detail: object) -> None:
        checks.append((label, result, str(detail)))

    for module_name in (
        "torch",
        "sglang",
        "sgl_kernel",
        "sglang_router.launch_router",
        "deep_gemm",
        "megatron.core",
        "megatron.bridge",
        "orbit",
        "transformer_engine.pytorch",
        "flash_attn",
        "flash_attn_3.flash_attn_interface",
        "apex",
    ):
        try:
            module = importlib.import_module(module_name)
            check("import " + module_name, True, getattr(module, "__version__", "ok"))
        except Exception as error:
            check("import " + module_name, False, type(error).__name__ + ": " + str(error))

    exact_versions = {
        "torch": expected["TORCH_VERSION"],
        "torchvision": expected["TORCHVISION_VERSION"],
        "torchaudio": expected["TORCHAUDIO_VERSION"],
        "triton": expected["TRITON_VERSION"],
        "cuda-python": expected["CUDA_PYTHON_VERSION"],
        "transformer-engine": expected["TRANSFORMER_ENGINE_VERSION"],
        "transformer-engine-cu13": expected["TRANSFORMER_ENGINE_VERSION"],
        "transformer-engine-torch": expected["TRANSFORMER_ENGINE_VERSION"],
        "nvidia-cudnn-cu13": expected["CUDNN_CU13_VERSION"],
        "apache-tvm-ffi": expected["APACHE_TVM_FFI_VERSION"],
    }
    for package, wanted in exact_versions.items():
        actual = version(package)
        check(package + " == " + wanted, bool(actual and actual.split("+")[0] == wanted), actual)

    kernel = version("sglang-kernel")
    check(
        "sglang-kernel prebuilt cu130",
        bool(kernel and kernel.startswith(expected["SGLANG_KERNEL_VERSION"]) and "cu130" in kernel),
        kernel,
    )
    deep_gemm = version("sgl-deep-gemm")
    check(
        "sgl-deep-gemm == " + expected["SGL_DEEP_GEMM_VERSION"],
        deep_gemm == expected["SGL_DEEP_GEMM_VERSION"],
        deep_gemm,
    )

    for package in ("flashinfer-python", "flashinfer-cubin", "flashinfer-jit-cache"):
        actual = version(package)
        wanted = expected["FLASHINFER_VERSION"]
        check(package + " == " + wanted, bool(actual and actual.startswith(wanted)), actual)

    editables = {
        "sglang": args.source_root / "sglang" / expected["ORBIT_SGLANG_SUBDIRECTORY"],
        "megatron-core": args.source_root / "Megatron-LM",
        "megatron-bridge": args.source_root / "Megatron-Bridge",
        "orbit": REPO_ROOT,
    }
    for package, path in editables.items():
        check(package + " editable", editable_at(package, path), path)

    commits = {
        args.source_root / "sglang": expected["ORBIT_SGLANG_COMMIT"],
        args.source_root / "Megatron-LM": expected["ORBIT_MEGATRON_COMMIT"],
        args.source_root / "Megatron-Bridge": expected["ORBIT_MEGATRON_BRIDGE_COMMIT"],
    }
    for path, wanted in commits.items():
        actual = ""
        if (path / ".git").exists():
            actual = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
        check(path.name + " commit", actual == wanted, actual)

    if args.full_h100:
        try:
            import torch

            available = torch.cuda.is_available()
            check("torch.cuda.is_available", available, torch.version.cuda)
            check(
                "torch CUDA major == 13",
                bool(torch.version.cuda and torch.version.cuda.startswith("13.")),
                torch.version.cuda,
            )
            if available:
                name = torch.cuda.get_device_name(0)
                capability = torch.cuda.get_device_capability(0)
                check("GPU is H100, H200 or B200", "H100" in name or "H200" in name or "B200" in name, name)
                check("compute capability in {9.0, 10.0}", capability in {(9, 0), (10, 0)}, capability)
                value = torch.randn((256, 256), device="cuda", dtype=torch.bfloat16)
                result = value @ value
                check("finite BF16 CUDA matmul", bool(torch.isfinite(result).all().item()), result.shape)
        except Exception as error:
            check("GPU runtime", False, type(error).__name__ + ": " + str(error))

    failures = 0
    for label, passed, detail in checks:
        failures += not passed
        print("[" + ("PASS" if passed else "FAIL") + "] " + label + ": " + detail)
    print("[summary] " + str(len(checks) - failures) + "/" + str(len(checks)) + " passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
