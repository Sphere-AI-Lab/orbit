#!/usr/bin/env python3
"""Static import integrity: every miles.*/miles_plugins.*/orbit.* import in tracked
.py files must resolve to a real module file, and `from X import name` must name a
submodule or a top-level binding in X. Catches the classic file-move failure where
the old package still resolves but the submodule is gone. "orbit" stays in ROOTS
even though the home moved to miles/orbit/: top-level orbit.* no longer exists, so
any surviving orbit.* import is a stale reference and fails here. Exits 1 on any
dangler."""
import ast, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
tracked = subprocess.run(
    ["git", "-C", str(REPO), "ls-files", "*.py"], capture_output=True, text=True
).stdout.splitlines()
ROOTS = ("miles", "miles_plugins", "orbit")

# Pre-existing danglers, tolerated but not expanded (file, dotted module, name).
ALLOWLIST = {
    ("tools/convert_to_hf_legacy.py", "miles.backends.megatron_utils", "update_weight_utils"),
    # upstream ships this script for use after docker/npu_patch is applied
    ("scripts/run_qwen3_4b_npu.py", "miles.utils.external_utils.command_utils", "execute_train_npu"),
}

def module_file(dotted: str):
    p = REPO / dotted.replace(".", "/")
    if (p.with_suffix(".py")).is_file():
        return p.with_suffix(".py")
    if (p / "__init__.py").is_file():
        return p / "__init__.py"
    if p.is_dir():
        return p  # namespace package
    return None

_bindings_cache: dict = {}

def top_bindings(pyfile: Path):
    if pyfile in _bindings_cache:
        return _bindings_cache[pyfile]
    names, star = set(), False
    try:
        tree = ast.parse(pyfile.read_text(errors="surrogateescape"))
    except Exception:
        star = True
        tree = None
    if tree:
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    for n in ast.walk(t):
                        if isinstance(n, ast.Name):
                            names.add(n.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif hasattr(ast, "TypeAlias") and isinstance(node, ast.TypeAlias):
                names.add(node.name.id)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    names.add((a.asname or a.name).split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                for a in node.names:
                    if a.name == "*":
                        star = True
                    else:
                        names.add(a.asname or a.name)
            elif isinstance(node, (ast.If, ast.Try)):
                for sub in ast.walk(node):
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        names.add(sub.name)
                    elif isinstance(sub, ast.Import):
                        for a in sub.names:
                            names.add((a.asname or a.name).split(".")[0])
                    elif isinstance(sub, ast.ImportFrom):
                        for a in sub.names:
                            names.add(a.asname or a.name)
                    elif isinstance(sub, ast.Assign):
                        for t in sub.targets:
                            for n in ast.walk(t):
                                if isinstance(n, ast.Name):
                                    names.add(n.id)
    _bindings_cache[pyfile] = (names, star)
    return names, star

def collect_errors() -> list[str]:
    errors = []
    for f in tracked:
        errors.extend(_check_file(f))
    return errors


def _check_file(f: str) -> list[str]:
    errors = []
    p = REPO / f
    if not p.is_file():
        return errors
    try:
        tree = ast.parse(p.read_text(errors="surrogateescape"), filename=f)
    except SyntaxError as e:
        errors.append(f"{f}: syntax error: {e}")
        return errors
    pkg = f.rsplit("/", 1)[0].split("/") if "/" in f else []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in ROOTS and module_file(a.name) is None:
                    errors.append(f"{f}:{node.lineno}: import {a.name} -> no module")
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg[: len(pkg) - (node.level - 1)] if node.level > 1 else pkg
                if not base or base[0] not in ROOTS:
                    continue
                mod = ".".join(base + (node.module.split(".") if node.module else []))
            else:
                if not node.module or node.module.split(".")[0] not in ROOTS:
                    continue
                mod = node.module
            mf = module_file(mod)
            if mf is None:
                errors.append(f"{f}:{node.lineno}: from {mod} import ... -> no module")
                continue
            for a in node.names:
                if a.name == "*":
                    continue
                if module_file(mod + "." + a.name) is not None:
                    continue
                if mf.is_dir():
                    errors.append(f"{f}:{node.lineno}: from {mod} import {a.name} -> namespace pkg has no such submodule")
                    continue
                names, star = top_bindings(mf)
                if a.name not in names and not star:
                    if (f, mod, a.name) in ALLOWLIST:
                        continue
                    errors.append(f"{f}:{node.lineno}: from {mod} import {a.name} -> not a submodule or top-level binding")
    return errors


if __name__ == "__main__":
    all_errors = collect_errors()
    for e in all_errors:
        print(e)
    print(f"[verify-imports] {'FAIL' if all_errors else 'ok'}: {len(tracked)} files")
    sys.exit(1 if all_errors else 0)
