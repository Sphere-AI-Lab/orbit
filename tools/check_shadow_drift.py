#!/usr/bin/env python3
"""Two failure modes of orbit's mixin-override architecture, both silent.

Orbit overrides upstream behaviour by declaring a mixin FIRST in a vendored
class's bases::

    class MegatronTrainRayActor(OrbitTrainActorExtensions, TrainRayActor):
    MRO: [MegatronTrainRayActor, OrbitTrainActorExtensions, TrainRayActor, object]

The mixin therefore shadows **TrainRayActor's** methods -- NOT the vendored
class's own body, which Python checks first. That asymmetry produces two bugs
that neither the test suite nor a merge will ever surface:

1. DEAD OVERRIDE. The mixin defines ``foo`` and the vendored class body ALSO
   defines ``foo``. The vendored one wins; orbit's override never runs. Nothing
   errors -- upstream behaviour just quietly ships. Reported as an error here.

2. UPSTREAM DRIFT. The mixin overrides a base-class method. That base method is
   now effectively dead, and dead code never conflicts: upstream can rewrite it,
   git reports nothing (we did not touch it), and orbit's override goes on
   winning while silently no longer matching what it was written against. So its
   body is hash-pinned; when it changes, this fails and names the method, and
   regenerating the manifest is how a reviewer records "I looked".

Together these are the mixin analogue of a pinned-hash patch: they turn silent
divergence into a loud, reviewable failure, which is what makes it safe to keep
expanding the override surface instead of editing vendored files.

Regenerating (after a deliberate review):
  python3 tools/check_shadow_drift.py --write
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "tests" / "fast" / "shadow_manifest.json"
MIXIN_NAME = re.compile(r"^Orbit\w*Extensions$")
VENDORED_ROOTS = ("miles/", "miles_plugins/")


def tracked_py() -> list[str]:
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "*.py"], capture_output=True, text=True
    ).stdout.splitlines()
    return [f for f in out if f.startswith(VENDORED_ROOTS)]


def parse(path: Path):
    try:
        src = path.read_text(errors="surrogateescape")
        return src, ast.parse(src)
    except (OSError, SyntaxError):
        return None, None


def methods_of(node: ast.ClassDef) -> dict[str, ast.AST]:
    return {
        n.name: n
        for n in node.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def body_hash(src: str, node: ast.AST) -> str:
    """Hash the method's source text, normalized for whitespace-only churn so a
    reindent does not masquerade as a behavioural change."""
    seg = ast.get_source_segment(src, node) or ""
    norm = "\n".join(line.rstrip() for line in seg.splitlines() if line.strip())
    return hashlib.sha256(norm.encode("utf-8", "surrogateescape")).hexdigest()


def mixin_module_for(tree: ast.Module, mixin: str) -> str | None:
    """Resolve `from orbit.x.y import OrbitZExtensions` to the dotted module."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                if (a.asname or a.name) == mixin and node.module.startswith("orbit"):
                    return node.module
    return None


def _resolve_class(tree: ast.Module, name: str) -> tuple[str, Path] | None:
    """Resolve a base-class name to (dotted module, file) via this file's imports."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for a in node.names:
                if (a.asname or a.name) == name and node.module.startswith(
                    ("miles", "miles_plugins", "orbit")
                ):
                    p = REPO / (node.module.replace(".", "/") + ".py")
                    if p.is_file():
                        return node.module, p
    return None


def _mixin_sites():
    """Yield (rel, class node, module tree, source, mixin dotted name, mixin methods)."""
    cache: dict[str, dict[str, ast.AST]] = {}
    for rel in tracked_py():
        src, tree = parse(REPO / rel)
        if tree is None:
            continue
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for base in cls.bases:
                if not (isinstance(base, ast.Name) and MIXIN_NAME.match(base.id)):
                    continue
                dotted = mixin_module_for(tree, base.id)
                if dotted is None:
                    continue
                if dotted not in cache:
                    _msrc, mtree = parse(REPO / (dotted.replace(".", "/") + ".py"))
                    found: dict[str, ast.AST] = {}
                    if mtree is not None:
                        for mcls in [
                            n for n in ast.walk(mtree)
                            if isinstance(n, ast.ClassDef) and n.name == base.id
                        ]:
                            found.update(methods_of(mcls))
                    cache[dotted] = found
                yield rel, cls, tree, src, f"{dotted}.{base.id}", cache[dotted]


def dead_overrides() -> list[str]:
    """Mixin methods the vendored class's OWN body shadows -- they never run."""
    out = []
    for rel, cls, _tree, _src, mixin, mmethods in _mixin_sites():
        own = methods_of(cls)
        for name in sorted(set(mmethods) & set(own)):
            if name.startswith("__") and name.endswith("__"):
                continue
            out.append(
                f"{rel}::{cls.name}.{name}: DEAD OVERRIDE -- {mixin} defines "
                f"{name}, but {cls.name}'s own body defines it too and the own "
                f"body outranks the mixin in the MRO, so orbit's version never "
                f"runs. Delete it from the vendored class body."
            )
    return out


def collect_pairs() -> dict[str, dict]:
    """Base-class methods a mixin genuinely overrides, hash-pinned.

    {"<base module>::<base class>.<method>": {"mixin": ..., "via": ..., "sha": ...}}
    """
    pairs: dict[str, dict] = {}
    for rel, cls, tree, _src, mixin, mmethods in _mixin_sites():
        own = set(methods_of(cls))
        for base in cls.bases:
            if isinstance(base, ast.Name) and MIXIN_NAME.match(base.id):
                continue
            if not isinstance(base, ast.Name):
                continue
            resolved = _resolve_class(tree, base.id)
            if resolved is None:
                continue
            bmod, bpath = resolved
            bsrc, btree = parse(bpath)
            if btree is None:
                continue
            for bcls in [
                n for n in ast.walk(btree)
                if isinstance(n, ast.ClassDef) and n.name == base.id
            ]:
                bmethods = methods_of(bcls)
                for name in sorted(set(mmethods) & set(bmethods) - own):
                    if name.startswith("__") and name.endswith("__"):
                        continue
                    pairs[f"{bmod}::{base.id}.{name}"] = {
                        "mixin": mixin,
                        "via": f"{rel}::{cls.name}",
                        "sha": body_hash(bsrc, bmethods[name]),
                    }
    return pairs


def collect_errors() -> list[str]:
    # A dead override is a bug on its own terms -- report it whether or not a
    # manifest exists, since no hash can make an override that never runs correct.
    errors = dead_overrides()
    if not MANIFEST.is_file():
        return errors + [f"{MANIFEST.relative_to(REPO)} is missing; run --write"]
    recorded = json.loads(MANIFEST.read_text())["shadowed"]
    current = collect_pairs()
    for key, entry in sorted(recorded.items()):
        cur = current.get(key)
        if cur is None:
            errors.append(
                f"{key}: recorded as shadowed by {entry['mixin']} but the vendored "
                f"method is gone; if orbit's override is now dead, delete it, then "
                f"regenerate with tools/check_shadow_drift.py --write"
            )
        elif cur["sha"] != entry["sha"]:
            errors.append(
                f"{key}: UPSTREAM CHANGED a method orbit shadows via "
                f"{entry['mixin']}. Orbit's override still wins, so nothing "
                f"conflicted and nothing failed at runtime -- review whether the "
                f"override needs the same change, then regenerate the manifest to "
                f"record that review"
            )
    for key in sorted(set(current) - set(recorded)):
        errors.append(
            f"{key}: newly shadows a vendored method (mixin "
            f"{current[key]['mixin']}) and is not recorded; regenerate the manifest"
        )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.write:
        pairs = collect_pairs()
        MANIFEST.write_text(
            json.dumps({"shadowed": pairs}, indent=1, sort_keys=True) + "\n"
        )
        print(
            f"[shadow-drift] wrote {MANIFEST.relative_to(REPO)}: "
            f"{len(pairs)} shadowed methods",
            file=sys.stderr,
        )
        return 0

    errors = collect_errors()
    for e in errors:
        print(f"[shadow-drift] {e}", file=sys.stderr)
    print(
        f"[shadow-drift] {'FAIL' if errors else 'ok'}: "
        f"{len(collect_pairs())} shadowed methods checked",
        file=sys.stderr,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
