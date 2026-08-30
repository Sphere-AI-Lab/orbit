#!/usr/bin/env python3
"""Every orbit patch pin still matches the upstream function it replaces.

The runtime does this check too (orbit/patch/runtime.py verifies before it
swaps), but only once torch and megatron are importable and a process is
actually starting. This is the same check done STATICALLY -- pure AST over the
vendored source -- so it runs in the CPU PR gate in a couple of seconds and
fails the moment an upstream merge moves a patched function, instead of at the
top of someone's eight-GPU run.

The two hashes are byte-identical by construction: both normalize from the
``def`` line onward via orbit.patch.runtime.normalize, which is why decorators
(included by inspect.getsource, excluded by ast.get_source_segment) cannot make
them disagree.

Re-pinning after a deliberate review:
  python3 tools/check_patch_pins.py --write
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from orbit.patch.runtime import normalize  # noqa: E402  (needs REPO on the path)

DECORATOR = "patch_function"


def tracked_orbit_py() -> list[str]:
    """Tracked AND untracked-but-not-ignored orbit sources.

    `--others` matters: a brand-new patch module is untracked until it is
    staged, and a gate that cannot see new patches is exactly the vacuously
    green one this exists to prevent.
    """
    out = subprocess.run(
        [
            "git", "-C", str(REPO), "ls-files",
            "--cached", "--others", "--exclude-standard",
            "orbit/*.py",
        ],
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return sorted(set(out))


def _kwargs_of(call: ast.Call) -> dict[str, str]:
    out: dict[str, str] = {}
    for kw in call.keywords:
        if kw.arg and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            out[kw.arg] = kw.value.value
    for i, arg in enumerate(call.args):
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            out["module" if i == 0 else "attr" if i == 1 else f"_{i}"] = arg.value
    return out


def declarations() -> list[dict]:
    """Every @patch_function(...) in orbit/, read statically."""
    found: list[dict] = []
    for rel in tracked_orbit_py():
        path = REPO / rel
        try:
            src = path.read_text(errors="surrogateescape")
            tree = ast.parse(src)
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                if not isinstance(dec, ast.Call):
                    continue
                name = dec.func.attr if isinstance(dec.func, ast.Attribute) else getattr(dec.func, "id", None)
                if name != DECORATOR:
                    continue
                kw = _kwargs_of(dec)
                if not {"module", "attr", "upstream_sha"} <= set(kw):
                    # Never skip quietly: a declaration this gate cannot read is
                    # a patch it cannot verify, which is the vacuously-green
                    # failure this whole file exists to prevent. Keep module and
                    # attr as plain string literals -- an f-string is not a
                    # static constant and lands here.
                    found.append(
                        {
                            "file": rel,
                            "lineno": dec.lineno,
                            "replacement": node.name,
                            "unreadable": sorted(
                                {"module", "attr", "upstream_sha"} - set(kw)
                            ),
                        }
                    )
                    continue
                found.append(
                    {
                        "file": rel,
                        "lineno": dec.lineno,
                        "replacement": node.name,
                        **kw,
                    }
                )
    return found


def upstream_sha(module: str, attr: str) -> str | None:
    """Hash of the vendored function, or None when it does not exist."""
    path = REPO / (module.replace(".", "/") + ".py")
    if not path.is_file():
        path = REPO / (module.replace(".", "/") + "/__init__.py")
        if not path.is_file():
            return None
    try:
        src = path.read_text(errors="surrogateescape")
        tree = ast.parse(src)
    except (OSError, SyntaxError):
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == attr:
            seg = ast.get_source_segment(src, node) or ""
            return hashlib.sha256(
                normalize(seg).encode("utf-8", "surrogateescape")
            ).hexdigest()
    return None


def collect_errors() -> list[str]:
    errors = []
    for d in declarations():
        if "unreadable" in d:
            errors.append(
                f"{d['file']}:{d['lineno']}: patch_function declaration for "
                f"{d['replacement']} is not statically readable "
                f"({', '.join(d['unreadable'])} is not a plain string literal), "
                f"so this patch cannot be pin-checked. Use literals, not "
                f"f-strings or names."
            )
            continue
        actual = upstream_sha(d["module"], d["attr"])
        target = f"{d['module']}.{d['attr']}"
        where = f"{d['file']}:{d['lineno']}"
        if actual is None:
            errors.append(
                f"{where}: orbit patches {target}, but no such module-level "
                f"function exists in the vendored tree. Upstream moved or renamed "
                f"it; re-point or retire the patch (orbit's replacement is "
                f"{d['replacement']})"
            )
        elif actual != d["upstream_sha"]:
            errors.append(
                f"{where}: pin stale for {target} (pinned "
                f"{d['upstream_sha'][:12]}, upstream is now {actual[:12]}). "
                f"Review whether upstream's new body changes why orbit replaced "
                f"it, then re-pin with tools/check_patch_pins.py --write"
            )
    return errors


def rewrite_pins() -> int:
    """Update every stale pin in place. A deliberate act: it records review."""
    changed = 0
    by_file: dict[str, list[dict]] = {}
    for d in declarations():
        if "unreadable" in d:
            continue
        by_file.setdefault(d["file"], []).append(d)
    for rel, decls in by_file.items():
        path = REPO / rel
        lines = path.read_text(errors="surrogateescape").splitlines(keepends=True)
        # Rewrite by LINE, not by text search: several declarations in one file
        # legitimately share a pin value (every unfilled placeholder is the same
        # empty string), so a search-and-replace cannot tell them apart. Walk
        # bottom-up so earlier edits never shift a later declaration's lineno.
        for d in sorted(decls, key=lambda x: -x["lineno"]):
            actual = upstream_sha(d["module"], d["attr"])
            if actual is None or actual == d["upstream_sha"]:
                continue
            old = f'upstream_sha="{d["upstream_sha"]}"'
            new = f'upstream_sha="{actual}"'
            for i in range(d["lineno"] - 1, min(d["lineno"] + 12, len(lines))):
                if old in lines[i]:
                    lines[i] = lines[i].replace(old, new, 1)
                    changed += 1
                    break
        path.write_text("".join(lines))
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()

    if args.write:
        n = rewrite_pins()
        print(f"[patch-pins] re-pinned {n} patch(es)", file=sys.stderr)
        return 0

    errors = collect_errors()
    for e in errors:
        print(f"[patch-pins] {e}", file=sys.stderr)
    print(
        f"[patch-pins] {'FAIL' if errors else 'ok'}: "
        f"{len(declarations())} pins checked",
        file=sys.stderr,
    )
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
