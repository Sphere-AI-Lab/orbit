#!/usr/bin/env python3
"""Extract pinned versions / commits from upstream sources of truth into pins.env.

Sources of truth (the things you actually bump when upgrading):
  - docker/Dockerfile                          (radixark/miles)
  - thirdparty/sglang/docker/Dockerfile        (sglang base image)
  - thirdparty/sglang/python/pyproject.toml    (the patched sglang fork)

`scripts/slurm/setup/install_env.sh` sources `pins.env` to get its version
defaults — the bare-metal install therefore stays in lockstep with the
container build without us re-typing the pins in two places.

The sglang stack is special. The prebuilt wheels (flash-attn / flash-attn-3 /
apex) are torch-ABI-bound, so (sglang version, torch version, wheels) move as
one atomic bundle. There are TWO views of that bundle:

  ACTIVE          — what install_env.sh actually installs. Tracks the
                    thirdparty/sglang submodule. Only sglang-sync advances it.
                    Held in MILES_WHEELS_TAG; torch/sglang/router are DERIVED
                    from it via the WHEELS_STACK mapping below.
  UPSTREAM_TARGET — where upstream docker/Dockerfile points (SGLANG_IMAGE_TAG /
                    WHEELS_TAG). The destination a future sglang-sync advances
                    ACTIVE to. Extracted, recorded, never auto-applied.

`MILES_WHEELS_TAG` is therefore NOT extracted from the Dockerfile — it is
PRESERVED from the existing pins.env. miles-sync must never auto-bump it; doing
so without also bumping the submodule + torch is exactly the ABI-mismatch trap.

Usage:
  # Print pins.env content to stdout (preview before writing):
  python scripts/slurm/setup/extract_pins.py

  # Regenerate pins.env in place (re-derives ACTIVE fields, refreshes UPSTREAM):
  python scripts/slurm/setup/extract_pins.py --write

  # CI / install-time check (see exit-code contract in main()):
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

# ---------------------------------------------------------------------------
# WHEELS_STACK — the single source of truth for "what an sglang wheels tag
# means". Maps a miles-wheels release tag to the (sglang base tag, torch ABI,
# sglang_router version) it was built against. The torch/sglang fields are read
# off the release's own name on GitHub (e.g. cu129-x86_64-v0.5.12 is published
# as "SGLang v0.5.12 / torch 2.11.0"); the old cu129-x86_64 line is the v0.5.10
# / torch 2.9.1 bundle. ADD A ROW HERE as part of sglang-sync whenever you point
# MILES_WHEELS_TAG at a new release.
# ---------------------------------------------------------------------------
WHEELS_STACK: dict[str, dict[str, str]] = {
    "cu129-x86_64": {"sglang": "v0.5.10", "torch": "2.9.1", "router": "0.3.2"},
    "cu129-x86_64-v0.5.12": {"sglang": "v0.5.12", "torch": "2.11.0", "router": "0.3.2"},
}


@dataclass
class Pin:
    key: str
    source: Path
    regex: str
    note: str = ""


# Purely-extracted scalars (overwritten verbatim on --write). The sglang-stack
# ACTIVE block (MILES_WHEELS_TAG + derived fields) is handled separately below —
# it is NOT in here because it is preserved, not extracted.
PIN_GROUPS: list[tuple[str, list[Pin]]] = [
    (
        "From thirdparty/sglang/python/pyproject.toml (patched sglang fork)",
        [
            Pin(
                "TORCH_VERSION",
                SGLANG_PYPROJ,
                r'"torch==([^"]+)"',
                "torch the sglang SUBMODULE is built against — this defines ACTIVE torch.",
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
                "FLASH_ATTN_INTERFACE_COMMIT",
                DOCKERFILE,
                r"flash-attention/([0-9a-f]{7,40})/hopper",
                "FA3 ships .so but not the python interface; we drop it in by hand.",
            ),
            Pin("MILES_WHEELS_REPO", DOCKERFILE, r"^ARG\s+WHEELS_REPO=(\S+)"),
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

# Extracted from docker/Dockerfile — the UPSTREAM TARGET, recorded but not applied.
UPSTREAM_PINS: list[Pin] = [
    Pin("UPSTREAM_SGLANG_IMAGE_TAG", DOCKERFILE, r"^ARG\s+SGLANG_IMAGE_TAG=(\S+)"),
    # Upstream split WHEELS_TAG into per-arch WHEELS_TAG_X86 / WHEELS_TAG_ARM64
    # (2026-06). We track the x86 line; the bare `WHEELS_TAG=` alternative keeps
    # older Dockerfile layouts parsable.
    Pin("UPSTREAM_WHEELS_TAG", DOCKERFILE, r"^ARG\s+WHEELS_TAG(?:_X86)?=(\S+)"),
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


def read_active_wheels_tag(default: str) -> str:
    """ACTIVE MILES_WHEELS_TAG is PRESERVED from the committed pins.env, not
    extracted. Falls back to `default` (the upstream tag) only when pins.env is
    absent / has no pin — i.e. first-time bootstrap, where tracking upstream is
    the sane initial state."""
    if PINS_FILE.exists():
        m = re.search(r"^MILES_WHEELS_TAG=\$\{MILES_WHEELS_TAG:-(\S+)\}", PINS_FILE.read_text(), re.MULTILINE)
        if m:
            return m.group(1)
    return default


def derive_index_urls(wheels_tag: str) -> dict[str, str]:
    """Derive wheel-index URLs from the ACTIVE wheels-tag cu prefix. SGL_WHL_INDEX_URL
    carries sgl-project's +cuNNN local-version builds of sglang-kernel/sgl-deep-gemm —
    the PyPI default wheels of those are cu13-linked, unloadable on a CUDA-12 driver."""
    match = re.search(r"cu(\d{3})", wheels_tag)
    if not match:
        raise SystemExit(f"FATAL: cannot derive cu tag from MILES_WHEELS_TAG={wheels_tag!r}")
    cu = f"cu{match.group(1)}"
    return {
        "TORCH_INDEX_URL": f"https://download.pytorch.org/whl/{cu}",
        "FLASHINFER_INDEX_URL": f"https://flashinfer.ai/whl/{cu}",
        "SGL_WHL_INDEX_URL": f"https://docs.sglang.ai/whl/{cu}",
    }


def extract() -> dict[str, str]:
    values: dict[str, str] = {}
    for _, pins in PIN_GROUPS:
        for pin in pins:
            values[pin.key] = search(pin)
    for pin in UPSTREAM_PINS:
        values[pin.key] = search(pin)

    # ACTIVE sglang stack: preserve MILES_WHEELS_TAG, derive the rest from the map.
    active = read_active_wheels_tag(default=values["UPSTREAM_WHEELS_TAG"])
    if active not in WHEELS_STACK:
        raise SystemExit(
            f"FATAL: MILES_WHEELS_TAG={active!r} is not in WHEELS_STACK.\n"
            f"       Add a row for it in {Path(__file__).name} (this is part of sglang-sync)."
        )
    stack = WHEELS_STACK[active]
    values["MILES_WHEELS_TAG"] = active
    values["MILES_WHEELS_TORCH_VERSION"] = stack["torch"]
    values["MILES_WHEELS_SGLANG_VERSION"] = stack["sglang"]
    values["SGLANG_ROUTER_VERSION"] = stack["router"]

    values.update(derive_index_urls(active))
    return values


HEADER = """\
# scripts/slurm/setup/pins.env — pinned versions/commits for install_env.sh.
#
# AUTO-GENERATED — do not hand-edit, EXCEPT MILES_WHEELS_TAG (the ACTIVE sglang
# bundle), which only sglang-sync changes. Regenerate the rest with:
#   python scripts/slurm/setup/extract_pins.py --write
#
# Sources of truth:
#   - docker/Dockerfile                          (radixark/miles)
#   - thirdparty/sglang/docker/Dockerfile        (sglang base image)
#   - thirdparty/sglang/python/pyproject.toml    (patched sglang fork)
#
# Each value is overridable: set the env var before invoking install_env.sh
# (e.g. `TE_VERSION=2.11.0 bash scripts/slurm/setup/install_env.sh`).
"""

SGLANG_STACK_COMMENT = """\
# --- sglang stack (atomic bundle: sglang version + torch ABI + prebuilt wheels) ---
# ACTIVE = what install_env.sh installs; tracks thirdparty/sglang. UPSTREAM_* =
# where docker/Dockerfile points. When they differ, `extract_pins.py --check`
# prints `[sglang-sync pending]` (exit 0 — deferrable, NOT a sync blocker), and
# install_env.sh hard-fails on any torch-ABI inconsistency.
#
# MILES_WHEELS_TAG is the ONE hand/sglang-sync-owned pin. The three fields under
# it are DERIVED from it via WHEELS_STACK in extract_pins.py — do not hand-edit;
# run `extract_pins.py --write`. To upgrade sglang: add a WHEELS_STACK row, set
# MILES_WHEELS_TAG, bump the submodule, then --write."""


def render(values: dict[str, str]) -> str:
    out: list[str] = [HEADER]
    for title, pins in PIN_GROUPS:
        out.append(f"\n# {title}")
        for pin in pins:
            if pin.note:
                out.append(f"# {pin.note}")
            out.append(f"{pin.key}=${{{pin.key}:-{values[pin.key]}}}")

    out.append("")
    out.append(SGLANG_STACK_COMMENT)
    out.append(f"MILES_WHEELS_TAG=${{MILES_WHEELS_TAG:-{values['MILES_WHEELS_TAG']}}}")
    out.append(f"MILES_WHEELS_TORCH_VERSION=${{MILES_WHEELS_TORCH_VERSION:-{values['MILES_WHEELS_TORCH_VERSION']}}}")
    out.append(
        f"MILES_WHEELS_SGLANG_VERSION=${{MILES_WHEELS_SGLANG_VERSION:-{values['MILES_WHEELS_SGLANG_VERSION']}}}"
    )
    out.append(f"SGLANG_ROUTER_VERSION=${{SGLANG_ROUTER_VERSION:-{values['SGLANG_ROUTER_VERSION']}}}")
    out.append("# UPSTREAM TARGET (from docker/Dockerfile) — advance ACTIVE to this via sglang-sync:")
    out.append(f"UPSTREAM_SGLANG_IMAGE_TAG=${{UPSTREAM_SGLANG_IMAGE_TAG:-{values['UPSTREAM_SGLANG_IMAGE_TAG']}}}")
    out.append(f"UPSTREAM_WHEELS_TAG=${{UPSTREAM_WHEELS_TAG:-{values['UPSTREAM_WHEELS_TAG']}}}")

    out.append("")
    out.append("# Derived from MILES_WHEELS_TAG cu prefix")
    out.append(f"TORCH_INDEX_URL=${{TORCH_INDEX_URL:-{values['TORCH_INDEX_URL']}}}")
    out.append(f"FLASHINFER_INDEX_URL=${{FLASHINFER_INDEX_URL:-{values['FLASHINFER_INDEX_URL']}}}")
    out.append(f"SGL_WHL_INDEX_URL=${{SGL_WHL_INDEX_URL:-{values['SGL_WHL_INDEX_URL']}}}")
    out.append("")
    return "\n".join(out)


def abi_errors(values: dict[str, str]) -> list[str]:
    """Cross-field torch-ABI consistency of the ACTIVE bundle. Non-empty = danger
    (a fresh install_env.sh would install torch-X wheels into a torch-Y env)."""
    errs: list[str] = []
    active = values["MILES_WHEELS_TAG"]
    stack = WHEELS_STACK[active]
    if stack["torch"] != values["TORCH_VERSION"]:
        errs.append(
            f"ABI MISMATCH: MILES_WHEELS_TAG={active} ships torch-{stack['torch']} wheels, "
            f"but the sglang submodule pins torch=={values['TORCH_VERSION']}. "
            f"flash-attn/apex are torch-ABI-bound — run sglang-sync to align."
        )
    return errs


def _tag_sglang_version(tag: str) -> str:
    """The sglang-version component of a wheels tag ('cu129-x86_64-v0.5.12' ->
    'v0.5.12'; legacy suffix-less tags -> '')."""
    m = re.search(r"-(v[0-9][^-]*)$", tag)
    return m.group(1) if m else ""


def pending_notice(values: dict[str, str]) -> str | None:
    """ACTIVE behind UPSTREAM target → a deferrable sglang-sync is due.

    Compared on the sglang-VERSION component of the tags, not the whole tag: the
    cu prefix is a deployment property, not a version lag — upstream's Dockerfile
    moved to cu130 wheels while bare-metal is driver-bound to the cu129 line
    (CUDA-12.8 driver), so a cu129-vs-cu130 difference alone is not 'pending'."""
    if _tag_sglang_version(values["MILES_WHEELS_TAG"]) != _tag_sglang_version(values["UPSTREAM_WHEELS_TAG"]):
        return (
            f"[sglang-sync pending] ACTIVE MILES_WHEELS_TAG={values['MILES_WHEELS_TAG']} "
            f"(sglang {values['MILES_WHEELS_SGLANG_VERSION']} / torch {values['MILES_WHEELS_TORCH_VERSION']}) "
            f"is behind UPSTREAM_WHEELS_TAG={values['UPSTREAM_WHEELS_TAG']} "
            f"(image {values['UPSTREAM_SGLANG_IMAGE_TAG']}). Run sglang-sync when ready to upgrade."
        )
    return None


def resolve_tag(tag: str) -> int:
    """Print the derived MILES_WHEELS_* fields for a wheels tag (pure WHEELS_STACK
    lookup, no file reads). install_env.sh evals this to re-derive the sglang-stack
    fields from the EFFECTIVE (possibly runtime-overridden) MILES_WHEELS_TAG, so an
    override of the tag can't desync from its torch/sglang/router. Exit 2 = unknown
    tag (same code as ABI danger: 'this bundle is not safe / not known')."""
    stack = WHEELS_STACK.get(tag)
    if stack is None:
        print(
            f"FATAL: MILES_WHEELS_TAG={tag!r} is not in WHEELS_STACK "
            f"(add a row in {Path(__file__).name} — this is part of sglang-sync).",
            file=sys.stderr,
        )
        return 2
    print(f"MILES_WHEELS_TORCH_VERSION={stack['torch']}")
    print(f"MILES_WHEELS_SGLANG_VERSION={stack['sglang']}")
    print(f"SGLANG_ROUTER_VERSION={stack['router']}")
    return 0


def tag_for_sglang(base: str) -> int:
    """Reverse-lookup: print the wheels tag whose WHEELS_STACK row has sglang==base.
    sglang-sync's no-arg path uses this instead of string-concatenating
    `cu129-x86_64-<base>` — the legacy v0.5.10 tag is suffix-less (`cu129-x86_64`),
    so the convention only holds for newer bases. Exit 2 on no/ambiguous match."""
    matches = [t for t, s in WHEELS_STACK.items() if s["sglang"] == base]
    if len(matches) == 1:
        print(matches[0])
        return 0
    if not matches:
        print(
            f"FATAL: no WHEELS_STACK tag maps to sglang {base!r}. Add a row in {Path(__file__).name} "
            f"(new miles-wheels releases are named cu129-x86_64-{base}).",
            file=sys.stderr,
        )
        return 2
    print(f"FATAL: ambiguous — multiple WHEELS_STACK tags map to sglang {base!r}: {matches}", file=sys.stderr)
    return 2


def main() -> int:
    # Exit-code contract (shared by --check and --write):
    #   0 = consistent, OR only `[sglang-sync pending]` (ACTIVE behind UPSTREAM —
    #       deferrable, must NOT block CI / install / miles-sync).
    #   1 = pins.env missing, OR drift (committed != freshly extracted; run --write).
    #   2 = torch-ABI inconsistency / unknown wheels tag (DANGER — do NOT --write;
    #       bump the submodule / fix the tag first). Distinct from 1 so callers can
    #       stop on danger instead of blindly regenerating.
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--write", action="store_true", help=f"write to {PINS_FILE.relative_to(REPO_ROOT)}")
    parser.add_argument("--check", action="store_true", help="verify pins.env (exit 0/1/2; see contract in source)")
    parser.add_argument(
        "--resolve", metavar="TAG", help="print derived MILES_WHEELS_* for a wheels TAG (for install_env.sh) and exit"
    )
    parser.add_argument(
        "--tag-for-sglang",
        metavar="BASE",
        help="print the wheels tag whose WHEELS_STACK row matches sglang BASE, and exit",
    )
    args = parser.parse_args()

    if args.resolve is not None:
        return resolve_tag(args.resolve)
    if args.tag_for_sglang is not None:
        return tag_for_sglang(args.tag_for_sglang)

    values = extract()
    rendered = render(values)
    errs = abi_errors(values)

    if args.check:
        if not PINS_FILE.exists():
            print(f"FATAL: {PINS_FILE.relative_to(REPO_ROOT)} does not exist", file=sys.stderr)
            return 1
        if errs:
            for e in errs:
                print(f"FATAL: {e}", file=sys.stderr)
            return 2

        committed = PINS_FILE.read_text()
        if committed.strip() != rendered.strip():
            print(f"FATAL: {PINS_FILE.relative_to(REPO_ROOT)} is stale vs sources.", file=sys.stderr)
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

        pending = pending_notice(values)
        if pending:
            print(pending, file=sys.stderr)
        else:
            print(f"[pins] OK — {PINS_FILE.relative_to(REPO_ROOT)} matches sources", file=sys.stderr)
        return 0

    if args.write:
        if errs:
            for e in errs:
                print(f"FATAL: {e}", file=sys.stderr)
            print(
                "       Refusing to write an ABI-inconsistent bundle. Bump thirdparty/sglang to the\n"
                "       matching line FIRST (so its pyproject torch matches the tag), then --write.",
                file=sys.stderr,
            )
            return 2
        PINS_FILE.write_text(rendered)
        print(f"[pins] wrote {PINS_FILE.relative_to(REPO_ROOT)}", file=sys.stderr)
        pending = pending_notice(values)
        if pending:
            print(pending, file=sys.stderr)
        return 0

    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    sys.exit(main())
