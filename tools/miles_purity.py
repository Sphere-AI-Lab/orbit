#!/usr/bin/env python3
"""Miles purity manifest: generate and check the orbit-vs-miles entanglement ratchet.

Orbit is a fork of radixark/miles at MILES_BASE, package-renamed miles/ -> orbit/
and miles_plugins/ -> orbit_plugins/. The manifest records, for every file orbit
shares with that base, which purity class it is in and a content hash. The
companion test (tests/fast/test_miles_purity_ratchet.py) needs only the manifest
and the working tree, so CI never touches the miles repository.

Classes:
  pristine    -- byte-identical to the miles base after NORMALIZE (the mechanical
                 miles->orbit token rewrite). Must stay that way.
  budgeted    -- carries orbit modifications. Any further edit changes the
                 recorded hash and fails the ratchet until the manifest is
                 deliberately regenerated and the new delta reviewed.

Regenerating (needs the miles base objects once):
  git fetch https://github.com/radixark/miles.git ef7481ae3bfbcc641d031e7e6113b646bb764382:refs/miles/base
  python3 tools/miles_purity.py --write
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "tests" / "fast" / "miles_purity_manifest.json"
MILES_BASE = "ef7481ae3bfbcc641d031e7e6113b646bb764382"

PATH_MAP = (("miles_plugins/", "orbit_plugins/"), ("miles/", "orbit/"))

NORMALIZE = (
    (r"\bmiles_plugins\b", "orbit_plugins"),
    (r"\bmiles\b", "orbit"),
    (r"\bMILES_", "ORBIT_"),
    (r"\bMiles\b", "Orbit"),
    (r"\bMILES\b", "ORBIT"),
)


def normalize(text: str) -> str:
    for pat, rep in NORMALIZE:
        text = re.sub(pat, rep, text)
    return text


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
    """Miles base files as {orbit-mapped path: blob sha}."""
    out = {}
    for line in git("ls-tree", "-r", MILES_BASE).splitlines():
        meta, path = line.split("\t", 1)
        _mode, typ, blob = meta.split()
        if typ != "blob":
            continue
        for src, dst in PATH_MAP:
            if path.startswith(src):
                path = dst + path[len(src):]
                break
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
        base_norm = normalize(blob_text(blob))
        if current == base_norm:
            pristine[path] = sha(base_norm)
        else:
            delta = sum(
                1
                for d in difflib.unified_diff(
                    base_norm.splitlines(), current.splitlines(), n=0
                )
                if d[:1] in "+-" and d[:3] not in ("+++", "---")
            )
            budgeted[path] = {"sha": sha(current), "delta_lines": delta}
    return {
        "miles_base": MILES_BASE,
        "normalize": [list(p) for p in NORMALIZE],
        "pristine": pristine,
        "budgeted": budgeted,
        "dropped": dropped,
    }


def check(manifest: dict) -> list[str]:
    errors = []
    for path, expect in manifest["pristine"].items():
        current = working_text(path)
        if current is None:
            errors.append(f"{path}: pristine miles file is missing")
        elif sha(current) != expect:
            errors.append(
                f"{path}: was pristine miles code and has been modified; keep "
                f"miles-derived files pristine, or move the change into the "
                f"orbit home layer and regenerate the manifest"
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
