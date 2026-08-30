#!/usr/bin/env python3
"""Miles purity manifest: generate and check the orbit-vs-miles entanglement ratchet.

This repo vendors radixark/miles at MILES_BASE verbatim: the miles/ and
miles_plugins/ packages keep upstream's names and, wherever possible, upstream's
exact bytes. All orbit code lives in the top-level orbit/ home (plus the
orbit-only trees: tools/, scripts/, examples/, docs/, tests/). The manifest
records, for every file shared with the base, which purity class it is in and a
content hash. The companion test (tests/fast/test_miles_purity_ratchet.py) needs
only the manifest and the working tree, so CI never touches the miles repository.

Classes:
  pristine    -- byte-identical to the miles base. Must stay that way.
  budgeted    -- carries orbit modifications. Any further edit changes the
                 recorded hash and fails the ratchet until the manifest is
                 deliberately regenerated and the new delta reviewed.

Regenerating (needs the miles base objects once):
  git fetch https://github.com/radixark/miles.git dbbab1566ae438f7202fff653eae938e07b1d4b6:refs/miles/base
  python3 tools/miles_purity.py --write
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "tests" / "fast" / "miles_purity_manifest.json"
MILES_BASE = "dbbab1566ae438f7202fff653eae938e07b1d4b6"


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "surrogateescape")).hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True
    )
    if result.returncode:
        raise SystemExit(f"git {' '.join(args)}: {result.stderr.strip()[:200]}")
    return result.stdout


def miles_tree() -> dict[str, str]:
    """Miles base files as {path: blob sha}; paths match this repo's verbatim."""
    out = {}
    for line in git("ls-tree", "-r", MILES_BASE).splitlines():
        meta, path = line.split("\t", 1)
        _mode, typ, blob = meta.split()
        if typ == "blob":
            out[path] = blob
    return out


def blob_text(blob: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(REPO), "cat-file", "blob", blob],
        capture_output=True,
        text=True,
        errors="surrogateescape",
    )
    return result.stdout


def working_text(path: str) -> str | None:
    p = REPO / path
    if not p.is_file():
        return None
    return p.read_text(errors="surrogateescape")


def build() -> dict:
    base = miles_tree()
    pristine, budgeted, dropped = {}, {}, []
    for path, blob in sorted(base.items()):
        current = working_text(path)
        if current is None:
            dropped.append(path)
            continue
        base_text = blob_text(blob)
        if current == base_text:
            pristine[path] = sha(base_text)
        else:
            delta = sum(
                1
                for d in difflib.unified_diff(
                    base_text.splitlines(), current.splitlines(), n=0
                )
                if d[:1] in "+-" and d[:3] not in ("+++", "---")
            )
            budgeted[path] = {"sha": sha(current), "delta_lines": delta}
    return {
        "miles_base": MILES_BASE,
        "pristine": pristine,
        "budgeted": budgeted,
        "dropped": dropped,
    }


def home_violations(manifest: dict) -> list[str]:
    """Every file under miles/ traces to the base. Orbit code never lands there
    without going through the seam workflow; new orbit modules live in orbit/."""
    known = set(manifest["pristine"]) | set(manifest["budgeted"])
    out = []
    for path in git("ls-files", "miles/").splitlines():
        if path in known:
            continue
        out.append(
            f"{path}: file under miles/ that is not part of the vendored miles "
            f"base; orbit code belongs in orbit/ (regenerate the manifest only "
            f"if this really is upstreamed miles code)"
        )
    return out


def check(manifest: dict) -> list[str]:
    errors = []
    for path, expect in manifest["pristine"].items():
        current = working_text(path)
        if current is None:
            errors.append(f"{path}: pristine miles file is missing")
        elif sha(current) != expect:
            errors.append(
                f"{path}: was pristine miles code and has been modified; keep "
                f"miles files pristine, or move the change into the orbit/ "
                f"home layer and regenerate the manifest"
            )
    for path, entry in manifest["budgeted"].items():
        current = working_text(path)
        if current is None:
            errors.append(f"{path}: budgeted miles-shared file is missing")
        elif sha(current) != entry["sha"]:
            errors.append(
                f"{path}: miles-shared file changed (recorded delta "
                f"{entry['delta_lines']} lines); regenerate the manifest with "
                f"tools/miles_purity.py --write and review whether the new "
                f"delta shrinks or grows the entanglement"
            )
    errors.extend(home_violations(manifest))
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.write:
        manifest = build()
        MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        n_p, n_b = len(manifest["pristine"]), len(manifest["budgeted"])
        total = sum(e["delta_lines"] for e in manifest["budgeted"].values())
        print(
            f"[miles-purity] wrote {MANIFEST.relative_to(REPO)}: "
            f"{n_p} pristine, {n_b} budgeted ({total} delta lines), "
            f"{len(manifest['dropped'])} dropped",
            file=sys.stderr,
        )
        return 0

    manifest = json.loads(MANIFEST.read_text())
    errors = check(manifest)
    for e in errors:
        print(f"[miles-purity] {e}", file=sys.stderr)
    print(
        f"[miles-purity] {'FAIL' if errors else 'ok'}: "
        f"{len(manifest['pristine'])} pristine, "
        f"{len(manifest['budgeted'])} budgeted checked",
        file=sys.stderr,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
