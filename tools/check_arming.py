#!/usr/bin/env python3
"""Every process that reaches an orbit patch target must arm orbit first.

Orbit's behaviour over the vendored miles tree installs through an import hook
that ``import orbit`` arms. A process that imports a patched module without
importing orbit gets upstream's function with NO error -- the run just quietly
uses different code. Both checkpoint->HF converter CLIs shipped that way once.

The predecessor of this check lived inside tests/fast/test_hf_export_patches.py
and asked a much weaker question: does a file with a ``__main__`` block import a
patched module BY NAME, or its immediate parent package? That is not the
property. Reaching a patched module is transitive -- ``from miles.utils.arguments
import X`` pulls in misc, rm_hub, metric_utils and replay_base, none of which
appear in the importing file -- and ``__main__`` is not the only way a process
starts. Out of 652 tracked files the old rule considered 8, and it called the
repo clean while eight tests asserted against upstream's converters.

So this computes the actual closure over the repo's static import graph, and
treats two things as process starts:

* a ``__main__`` block -- a script someone runs directly;
* a ``@ray.remote`` definition -- Ray starts a worker process for it, and what
  that worker imports is decided by the task it is handed, not by any entrypoint
  in this repo.

The ray case cannot be fixed the way the script case is: four of those modules
are byte-pristine vendored files, so ``import orbit`` cannot be added to them.
They are armed cluster-wide instead, by the ``worker_process_setup_hook`` that
orbit's launch paths pass to ``ray.init`` (orbit/ray_setup.py), which is why this
check verifies that wiring rather than assuming it.

Known limits, in the interest of not being trusted further than it earns:

* Only static ``import``/``from`` statements, and only statically written
  names. ``importlib.import_module(name)`` and ``getattr(mod, attr)`` with
  computed names are invisible.
* ``miles/utils/misc.py`` calls ``ray.init(address="auto")`` with no runtime_env
  and is pristine, so a job started from THERE arms nothing. Its only tasks run
  shell commands and reach no patch target today; that is a fact about the
  current code, not a guarantee, and this check will say so if it changes.
* Shell heredoc programs (scripts/lib/*.sh) have no .py file to parse. The one
  that starts training is checked by string, for the hook only.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOTS = ("miles", "miles_plugins", "orbit", "tools", "tests", "scripts")

# Launch paths orbit controls: each must pass the worker setup hook to ray.init,
# or the ray-remote modules below are unarmed in their workers.
ORBIT_RAY_LAUNCHERS = ("scripts/lib/driver.sh", "tools/convert_torch_dist_to_hf_ray.py")


def tracked_py() -> list[str]:
    """Tracked and untracked-but-not-ignored .py files.

    Untracked matters: a guard that reads only the index passes vacuously on a
    file that has not been committed yet, which is exactly when it is needed.
    """
    out = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "--cached", "--others", "--exclude-standard", "*.py"],
        capture_output=True,
        text=True,
    ).stdout.split()
    return sorted(set(out))


def module_name(rel: str) -> str:
    name = rel[:-3].replace("/", ".")
    return name[: -len(".__init__")] if name.endswith(".__init__") else name


def _repo_prefixes(dotted: str, paths: dict[str, Path]) -> set[str]:
    parts = dotted.split(".")
    return {p for i in range(1, len(parts) + 1) if (p := ".".join(parts[:i])) in paths}


def build_graph(files: list[str]) -> tuple[dict[str, Path], dict[str, set[str]]]:
    """module name -> file, and module name -> repo modules it imports.

    Importing ``a.b.c`` executes ``a`` and ``a.b`` too, so every prefix is an
    edge. That is what makes package ``__init__`` re-exports part of the closure
    without special-casing them.
    """
    paths = {module_name(f): REPO / f for f in files if f.split("/")[0] in ROOTS}
    edges: dict[str, set[str]] = {}
    for mod, path in paths.items():
        try:
            tree = ast.parse(path.read_text(errors="surrogateescape"))
        except (OSError, SyntaxError):
            edges[mod] = set()
            continue
        pkg = mod.rsplit(".", 1)[0] if (path.name != "__init__.py") else mod
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    found |= _repo_prefixes(a.name, paths)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    base = pkg.split(".")
                    base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                    target = ".".join(base + (node.module.split(".") if node.module else []))
                else:
                    target = node.module or ""
                if not target:
                    continue
                found |= _repo_prefixes(target, paths)
                # `from pkg import name` binds pkg.name when that is a module.
                for a in node.names:
                    found |= _repo_prefixes(f"{target}.{a.name}", paths)
        edges[mod] = found
    return paths, edges


def closure(mod: str, edges: dict[str, set[str]]) -> set[str]:
    seen, stack = {mod}, [mod]
    while stack:
        for nxt in edges.get(stack.pop(), ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def arms_orbit(mods: set[str]) -> bool:
    return any(m == "orbit" or m.startswith("orbit.") for m in mods)


def entrypoint_kinds(path: Path) -> set[str]:
    """"main" if the file has a __main__ block, "ray" if it defines a remote."""
    kinds: set[str] = set()
    try:
        tree = ast.parse(path.read_text(errors="surrogateescape"))
    except (OSError, SyntaxError):
        return kinds
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            for cmp_node in ast.walk(node.test):
                if isinstance(cmp_node, ast.Name) and cmp_node.id == "__name__":
                    kinds.add("main")
        if isinstance(node, ast.Attribute) and node.attr == "remote":
            base = node.value
            if isinstance(base, ast.Name) and base.id == "ray":
                kinds.add("ray")
    return kinds


def patch_targets() -> set[tuple[str, str | None]]:
    """(module, attr) pairs whose behaviour only exists once orbit is imported.

    ``attr`` is None for an on_import seam: those act on the module itself, so
    importing it at all is enough to need the arming.
    """
    sys.path.insert(0, str(REPO))
    import orbit  # noqa: F401  -- registers the patches and seams
    from orbit.patch import registry
    from orbit.patch.on_import import registry as seam_registry

    targets = {(p.module, p.attr) for p in registry()}
    targets |= {(name, None) for name, _ in seam_registry()}
    return targets


def names_mentioned(path: Path) -> set[str]:
    """Every identifier the file could be calling a patched function through.

    Attribute accesses (``m2hf.convert_qwen2_to_hf``), imported names
    (``from ... import get_free_port``) and bare uses of them. Computed names
    (``getattr(mod, attr)``) are invisible -- see the limits in the docstring.
    """
    try:
        tree = ast.parse(path.read_text(errors="surrogateescape"))
    except (OSError, SyntaxError):
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.ImportFrom):
            names.update(a.name for a in node.names)
    return names


def target_users(paths, edges, targets):
    """target -> the repo modules that can actually invoke it.

    Reaching a patched MODULE is not the same as calling the patched FUNCTION:
    the newer miles base ships ~39 launcher scripts whose import closure touches
    miles.utils.misc but which never call get_free_port (its only caller in the
    tree is RayActor). Flagging those was over-approximation, so a module counts
    as a user only if it both reaches the target module and mentions the name.
    """
    users: dict[tuple[str, str | None], set[str]] = {t: set() for t in targets}
    closures = {mod: closure(mod, edges) for mod in paths}
    mentions = {mod: names_mentioned(path) for mod, path in paths.items()}
    for (module, attr) in targets:
        for mod in paths:
            if module not in closures[mod]:
                continue
            if attr is None or attr in mentions[mod]:
                users[(module, attr)].add(mod)
    return users


def hook_wiring_errors() -> list[str]:
    """The ray half rests on the setup hook actually being passed. Check it."""
    from orbit.ray_setup import WORKER_SETUP_HOOK

    errors = []
    for rel in ORBIT_RAY_LAUNCHERS:
        path = REPO / rel
        if not path.is_file():
            errors.append(f"{rel}: launch path is gone; re-point ORBIT_RAY_LAUNCHERS")
            continue
        src = path.read_text(errors="surrogateescape")
        # Either spelling counts: the launchers import the constant rather than
        # repeating the dotted path, which is the point of it living in one place.
        names = WORKER_SETUP_HOOK in src or "WORKER_SETUP_HOOK" in src
        if not ("worker_process_setup_hook" in src and names):
            errors.append(
                f"{rel}: starts Ray without worker_process_setup_hook="
                f"{WORKER_SETUP_HOOK!r}; its workers run upstream's unpatched functions"
            )
    return errors


def collect_errors() -> list[str]:
    targets = patch_targets()
    errors = hook_wiring_errors()
    hook_ok = not errors

    conftest = REPO / "conftest.py"
    if not conftest.is_file() or "import orbit" not in conftest.read_text(errors="surrogateescape"):
        errors.append(
            "conftest.py: missing or does not `import orbit`. Tests are skipped below on the "
            "grounds that pytest arms orbit before collection; without this they assert "
            "against upstream and pass or fail by collection order."
        )

    files = tracked_py()
    paths, edges = build_graph(files)
    users = target_users(paths, edges, targets)
    for mod, path in sorted(paths.items()):
        if mod.split(".")[0] == "tests":
            continue  # armed by conftest.py, verified above
        kinds = entrypoint_kinds(path)
        if not kinds:
            continue
        reached = closure(mod, edges)
        hit = sorted(
            f"{module}.{attr}" if attr else module
            for (module, attr), callers in users.items()
            if reached & callers
        )
        if not hit or arms_orbit(reached):
            continue
        rel = path.relative_to(REPO)
        if kinds == {"ray"} and hook_ok:
            continue  # armed cluster-wide by the worker setup hook
        why = "runs as a script" if "main" in kinds else "defines a Ray remote"
        errors.append(
            f"{rel}: {why} and can call patched {', '.join(hit)} "
            f"without importing orbit -- those patches silently do not apply"
        )
    return errors


if __name__ == "__main__":
    all_errors = collect_errors()
    for e in all_errors:
        print(e)
    print(f"[check-arming] {'FAIL' if all_errors else 'ok'}: {len(tracked_py())} files")
    sys.exit(1 if all_errors else 0)
