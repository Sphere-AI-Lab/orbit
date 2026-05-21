#!/usr/bin/env python3
"""Extract pinned versions / commits from upstream sources of truth into pins.env.

Sources of truth (the things you actually bump when upgrading):
  - docker/Dockerfile                          (radixark/miles)
  - thirdparty/sglang/docker/Dockerfile        (sglang base image)
  - thirdparty/sglang/python/pyproject.toml    (the patched sglang fork)

`scripts/slurm/setup/install_env.sh` sources `pins.env` to get its version
defaults — the bare-metal install therefore stays in lockstep with the
container build without us re-typing the pins in two places.

Usage:
  # Print pins.env content to stdout (preview before writing):
  python scripts/slurm/setup/extract_pins.py

  # Regenerate pins.env in place (run this after bumping the Dockerfile):
  python scripts/slurm/setup/extract_pins.py --write

  # CI / install-time drift check (exits 1 if pins.env is stale vs sources):
  python scripts/slurm/setup/extract_pins.py --check
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# scripts/slurm/setup/extract_pins.py → repo root = parents[3]
REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = REPO_ROOT / "docker" / "Dockerfile"
SGLANG_DOCKER = REPO_ROOT / "thirdparty" / "sglang" / "docker" / "Dockerfile"
SGLANG_PYPROJ = REPO_ROOT / "thirdparty" / "sglang" / "python" / "pyproject.toml"
PINS_FILE = REPO_ROOT / "scripts" / "slurm" / "setup" / "pins.env"


@dataclass
class Pin:
    key: str
    source: Path
    regex: str
    note: str = ""


# Group → ordered list of pins. Group titles double as section headers in pins.env.
PIN_GROUPS: list[tuple[str, list[Pin]]] = [
    (
        "From thirdparty/sglang/python/pyproject.toml (patched sglang fork)",
        [
            Pin(
                "TORCH_VERSION",
                SGLANG_PYPROJ,
                r'"torch==([^"]+)"',
                "sglang's pyproject pins the entire runtime tree against this torch.",
            ),
        ],
    ),
    (
        "From docker/Dockerfile (radixark/miles)",
        [
            Pin("TE_VERSION", DOCKERFILE, r'transformer_engine\[pytorch\]==([^"\s]+)"'),
            Pin("MBRIDGE_COMMIT", DOCKERFILE, r"ISEEKYAN/mbridge\.git@([0-9a-f]{7,40})"),
            Pin("TMS_COMMIT", DOCKERFILE, r"fzyzcjy/torch_memory_saver\.git@([0-9a-f]{7,40})"),
            Pin(
                "CUDNN_CU12_VERSION",
                DOCKERFILE,
                r"nvidia-cudnn-cu12==([0-9a-z.]+)",
                "pytorch/pytorch#168167 workaround; modelopt later downgrades it, this re-pins.",
            ),
            Pin(
                "FLASH_ATTN_INTERFACE_COMMIT",
                DOCKERFILE,
                r"flash-attention/([0-9a-f]{7,40})/hopper",
                "FA3 ships .so but not the python interface; we drop it in by hand.",
            ),
            Pin("MILES_WHEELS_REPO", DOCKERFILE, r"^ARG\s+WHEELS_REPO=(\S+)"),
            Pin("MILES_WHEELS_TAG", DOCKERFILE, r"^ARG\s+WHEELS_TAG=(\S+)"),
        ],
    ),
    (
        "From thirdparty/sglang/docker/Dockerfile (sglang base image)",
        [
            Pin(
                "MOONCAKE_VERSION",
                SGLANG_DOCKER,
                r"^ARG\s+MOONCAKE_VERSION=(\S+)",
                "miles imports mooncake.engine.TransferEngine at top level — required.",
            ),
        ],
    ),
]


def search(pin: Pin) -> str:
    if not pin.source.exists():
        raise SystemExit(f"FATAL: {pin.source} missing (submodule not checked out?)")
    text = pin.source.read_text()
    flags = re.MULTILINE if pin.regex.startswith("^") else 0
    match = re.search(pin.regex, text, flags)
    if not match:
        raise SystemExit(
            f"FATAL: pattern {pin.regex!r} matched nothing in {pin.source.relative_to(REPO_ROOT)}\n"
            f"       (upstream layout changed? update {Path(__file__).name})"
        )
    return match.group(1)


def derive_index_urls(wheels_tag: str) -> dict[str, str]:
    """Derive torch + flashinfer wheel-index URLs from the MILES_WHEELS_TAG cu prefix."""
    match = re.search(r"cu(\d{3})", wheels_tag)
    if not match:
        raise SystemExit(f"FATAL: cannot derive cu tag from MILES_WHEELS_TAG={wheels_tag!r}")
    cu = f"cu{match.group(1)}"
    return {
        "TORCH_INDEX_URL": f"https://download.pytorch.org/whl/{cu}",
        "FLASHINFER_INDEX_URL": f"https://flashinfer.ai/whl/{cu}",
    }


def extract() -> tuple[dict[str, str], dict[str, Pin]]:
    """Return ({key: value}, {key: Pin}) — pin metadata preserved for header notes."""
    values: dict[str, str] = {}
    meta: dict[str, Pin] = {}
    for _, pins in PIN_GROUPS:
        for pin in pins:
            values[pin.key] = search(pin)
            meta[pin.key] = pin
    values.update(derive_index_urls(values["MILES_WHEELS_TAG"]))
    return values, meta


HEADER = """\
# scripts/slurm/setup/pins.env — pinned versions/commits for install_env.sh.
#
# AUTO-GENERATED — do not edit by hand. Sources of truth:
#   - docker/Dockerfile                          (radixark/miles)
#   - thirdparty/sglang/docker/Dockerfile        (sglang base image)
#   - thirdparty/sglang/python/pyproject.toml    (patched sglang fork)
#
# Regenerate after bumping any of the above:
#   python scripts/slurm/setup/extract_pins.py --write
#
# Each value is overridable: set the env var before invoking install_env.sh
# (e.g. `TE_VERSION=2.11.0 bash scripts/slurm/setup/install_env.sh`).
"""


def render(values: dict[str, str], meta: dict[str, Pin]) -> str:
    out: list[str] = [HEADER]
    for title, pins in PIN_GROUPS:
        out.append(f"\n# {title}")
        for pin in pins:
            if pin.note:
                out.append(f"# {pin.note}")
            out.append(f"{pin.key}=${{{pin.key}:-{values[pin.key]}}}")
    out.append("")
    out.append("# Derived from MILES_WHEELS_TAG cu prefix")
    out.append(f"TORCH_INDEX_URL=${{TORCH_INDEX_URL:-{values['TORCH_INDEX_URL']}}}")
    out.append(f"FLASHINFER_INDEX_URL=${{FLASHINFER_INDEX_URL:-{values['FLASHINFER_INDEX_URL']}}}")
    out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--write", action="store_true", help=f"write to {PINS_FILE.relative_to(REPO_ROOT)} instead of stdout"
    )
    parser.add_argument(
        "--check", action="store_true", help="exit 1 if committed pins.env diverges from extracted pins"
    )
    args = parser.parse_args()

    values, meta = extract()
    rendered = render(values, meta)

    if args.check:
        if not PINS_FILE.exists():
            print(f"FATAL: {PINS_FILE.relative_to(REPO_ROOT)} does not exist", file=sys.stderr)
            return 1
        committed = PINS_FILE.read_text()
        if committed.strip() == rendered.strip():
            print(f"[pins] OK — {PINS_FILE.relative_to(REPO_ROOT)} matches upstream sources", file=sys.stderr)
            return 0
        print(f"FATAL: {PINS_FILE.relative_to(REPO_ROOT)} is stale vs upstream sources.", file=sys.stderr)
        print("       Regenerate with: python scripts/slurm/setup/extract_pins.py --write\n", file=sys.stderr)
        sys.stderr.writelines(
            difflib.unified_diff(
                committed.splitlines(keepends=True),
                rendered.splitlines(keepends=True),
                fromfile=str(PINS_FILE.relative_to(REPO_ROOT)) + " (committed)",
                tofile="(extracted now)",
                n=1,
            )
        )
        return 1

    if args.write:
        PINS_FILE.write_text(rendered)
        print(f"[pins] wrote {PINS_FILE.relative_to(REPO_ROOT)}", file=sys.stderr)
        return 0

    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
