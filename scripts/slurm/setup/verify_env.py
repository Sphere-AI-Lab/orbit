#!/usr/bin/env python3
"""Verify the miles conda env matches install_env.sh's declared state.

Combines (a) the smoke-test imports install_env.sh used to run at the end of
its pass and (b) a version/commit cross-check against scripts/slurm/setup/
pins.env, plus the .pth file + editable-install metadata that install_env.sh
relies on.

Usage:
  python scripts/slurm/setup/verify_env.py                 # imports + pinned versions
  python scripts/slurm/setup/verify_env.py --imports-only  # smoke test only (fast)
  python scripts/slurm/setup/verify_env.py --net           # also verify FA3 interface file SHA

Exit code: 0 if everything checks out, 1 otherwise.
install_env.sh runs this with no flags at the end of the install.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as md
import json
import os
import re
import site
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SITE_DIR = Path(site.getsitepackages()[0])
PINS_FILE = REPO_ROOT / "scripts/slurm/setup/pins.env"


def load_pins() -> dict[str, str]:
    """Parse `KEY=${KEY:-VALUE}` lines from pins.env."""
    out: dict[str, str] = {}
    pat = re.compile(r"^(\w+)=\$\{\w+:-(.+?)\}$")
    for line in PINS_FILE.read_text().splitlines():
        m = pat.match(line)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def ver(pkg: str) -> str | None:
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def direct_url(pkg: str) -> dict | None:
    try:
        dist = md.distribution(pkg)
    except md.PackageNotFoundError:
        return None
    f = next((x for x in (dist.files or []) if x.name == "direct_url.json"), None)
    return json.loads(f.locate().read_text()) if f else None


def _realpath(p: str | Path) -> str:
    return os.path.realpath(str(p))


def editable_at(pkg: str, expected_path: Path) -> bool:
    """True iff `pkg` is installed editable and points at `expected_path`.

    Uses os.path.realpath to canonicalise both sides — slinky's /home/$USER
    and /data/home/$USER are bind-mounted to the same inode, so plain string
    compare against direct_url.json's url field is fragile.
    """
    d = direct_url(pkg) or {}
    if not d.get("dir_info", {}).get("editable", False):
        return False
    url = d.get("url", "")
    if not url.startswith("file://"):
        return False
    return _realpath(url[len("file://") :]) == _realpath(expected_path)


def check_imports() -> list[tuple[str, bool]]:
    """Reproduce install_env.sh's old smoke test + the .pth-injected plugins package."""
    mods = [
        "torch",
        "sglang",
        "megatron.core",
        "megatron.bridge",
        "transformer_engine",
        "ray",
        "miles",
        "mbridge",
        "torch_memory_saver",
        "mooncake.engine",
        "miles_megatron_plugins",
    ]
    if os.environ.get("INSTALL_FLASH_ATTN", "1") == "1":
        mods.append("flash_attn")
    if os.environ.get("INSTALL_FLASH_ATTN_3", "1") == "1":
        mods.append("flash_attn_3.flash_attn_interface")
    if os.environ.get("INSTALL_APEX", "1") == "1":
        mods.append("apex")
    out = []
    for m in mods:
        try:
            mod = importlib.import_module(m)
            v = getattr(mod, "__version__", "?")
            out.append((f"import {m:<36s} v={v}", True))
        except Exception as e:
            out.append((f"import {m:<36s} {type(e).__name__}: {e}", False))
    return out


def check_runtime() -> list[tuple[str, bool]]:
    """Importable != linkable. Probe FA3 symbols, apex C extensions, CUDA visibility."""
    out: list[tuple[str, bool]] = []
    if os.environ.get("INSTALL_FLASH_ATTN_3", "1") == "1":
        try:
            fa3 = importlib.import_module("flash_attn_3.flash_attn_interface")
            req = ("_flash_attn_forward", "flash_attn_with_kvcache", "flash_attn_varlen_func")
            missing = [n for n in req if not hasattr(fa3, n)]
            out.append((f"FA3 symbols: {','.join(req)}", not missing))
        except Exception as e:
            out.append((f"FA3 symbols ({type(e).__name__}: {e})", False))
    if os.environ.get("INSTALL_APEX", "1") == "1":
        for ext in ("amp_C", "fused_layer_norm_cuda"):
            try:
                importlib.import_module(ext)
                out.append((f"apex C ext: {ext}", True))
            except Exception as e:
                out.append((f"apex C ext: {ext} ({type(e).__name__})", False))
        try:
            from apex.normalization import FusedLayerNorm  # noqa: F401
            from apex.optimizers import FusedAdam  # noqa: F401

            out.append(("apex FusedAdam + FusedLayerNorm", True))
        except Exception as e:
            out.append((f"apex FusedAdam/LayerNorm ({type(e).__name__})", False))
    try:
        import torch

        out.append((f"torch.cuda.is_available()  (cuda={torch.version.cuda})", torch.cuda.is_available()))
    except Exception as e:
        out.append((f"torch.cuda ({type(e).__name__})", False))
    return out


def check_pins(pins: dict[str, str]) -> list[tuple[str, bool]]:
    out: list[tuple[str, bool]] = []

    t = ver("torch")
    out.append((f"torch == {pins['TORCH_VERSION']}", bool(t and t.split("+")[0] == pins["TORCH_VERSION"])))
    for key, pkg in [
        ("TE_VERSION", "transformer_engine"),
        ("CUDNN_CU12_VERSION", "nvidia-cudnn-cu12"),
        ("MOONCAKE_VERSION", "mooncake-transfer-engine"),
    ]:
        out.append((f"{pkg} == {pins[key]}", ver(pkg) == pins[key]))

    mb = (direct_url("mbridge") or {}).get("vcs_info", {}).get("commit_id")
    out.append((f"mbridge @ {pins['MBRIDGE_COMMIT'][:8]}…", mb == pins["MBRIDGE_COMMIT"]))

    tms = (direct_url("torch-memory-saver") or {}).get("vcs_info", {}).get("commit_id")
    out.append((f"torch_memory_saver @ {pins['TMS_COMMIT']}…", bool(tms and tms.startswith(pins["TMS_COMMIT"]))))

    editable_targets = [
        ("megatron-core", REPO_ROOT / "thirdparty/Megatron-LM"),
        ("sglang", REPO_ROOT / "thirdparty/sglang/python"),
        ("miles", REPO_ROOT),
        ("megatron-bridge", REPO_ROOT / "thirdparty/Megatron-Bridge"),
    ]
    for pkg, path in editable_targets:
        rel = path.relative_to(REPO_ROOT) if path != REPO_ROOT else Path(".")
        out.append((f"{pkg} editable @ {rel}/", editable_at(pkg, path)))

    pth = SITE_DIR / "miles-megatron-source-root.pth"
    expected = REPO_ROOT / "thirdparty/Megatron-LM"
    pth_ok = pth.exists() and _realpath(pth.read_text().strip()) == _realpath(expected)
    out.append((f"miles-megatron-source-root.pth -> {expected.relative_to(REPO_ROOT)}/", pth_ok))
    return out


def check_fa3_content(pins: dict[str, str]) -> list[tuple[str, bool]]:
    """Network-bound: verify the manually-curled flash_attn_interface.py matches pin."""
    local = SITE_DIR / "flash_attn_3" / "flash_attn_interface.py"
    commit = pins["FLASH_ATTN_INTERFACE_COMMIT"]
    url = f"https://raw.githubusercontent.com/Dao-AILab/flash-attention/{commit}/hopper/flash_attn_interface.py"
    try:
        remote = urllib.request.urlopen(url, timeout=10).read()
    except Exception as e:
        return [(f"flash_attn_interface.py @ {commit[:8]} (net err: {type(e).__name__})", False)]
    ok = local.exists() and hashlib.sha256(local.read_bytes()).digest() == hashlib.sha256(remote).digest()
    return [(f"flash_attn_interface.py sha matches {commit[:8]}…", ok)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--imports-only", action="store_true", help="skip pin/version comparison; just do the smoke-test imports"
    )
    ap.add_argument(
        "--net", action="store_true", help="also verify FA3 interface file against github (~1 small request)"
    )
    args = ap.parse_args()

    checks: list[tuple[str, bool]] = []
    print("[verify] imports", file=sys.stderr)
    checks += check_imports()
    print("[verify] runtime symbols / CUDA", file=sys.stderr)
    checks += check_runtime()

    if not args.imports_only:
        pins = load_pins()
        print("[verify] pinned versions / commits / editables", file=sys.stderr)
        checks += check_pins(pins)
        if args.net:
            print("[verify] FA3 interface file (network)", file=sys.stderr)
            checks += check_fa3_content(pins)

    fail = sum(1 for _, ok in checks if not ok)
    for label, ok in checks:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {label}")
    print(f"\n=== {len(checks) - fail}/{len(checks)} pass, {fail} fail ===", file=sys.stderr)
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
