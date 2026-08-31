#!/usr/bin/env python3
"""Static path-anchor validity: every `Path(__file__)`-derived upward walk
(`.parents[N]` with a constant N, or a chained `.parent` access) must compute
an anchor that is still inside the repo, and an anchor immediately joined with
all-literal path segments (`anchor / "a" / "b.py"`, `anchor.joinpath("a")`)
must resolve to a path that exists in the working tree. Catches the classic
silent-anchor bug: a file moves, a fixed `parents[N]` index still returns SOME
directory, but the wrong one -- with the mistake invisible until something
that lives under it fails to load. Exits 1 on any mismatch."""
import ast, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
def tracked_py() -> list[str]:
    """Tracked AND untracked-but-not-ignored .py files.

    Reading the index alone makes this guard pass VACUOUSLY on a file that has
    not been committed yet, which is exactly when a new anchor is most likely to
    be wrong -- tests/fast/test_arming.py landed carrying a finding this guard
    could not see until the commit that added it.

    Computed per call rather than once at import: as a module-level constant it
    also went stale for any in-process caller that created a file after import,
    which silently defeated the very test written to prove this blind spot closed.
    """
    out = subprocess.run(
        [
            "git", "-C", str(REPO), "ls-files",
            "--cached", "--others", "--exclude-standard",
            "*.py",
        ],
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return sorted(set(out))

# (file, lineno) -> reason. Real, reviewed exceptions only -- not guard false
# positives (the rule is doing exactly what it's supposed to on every one of
# these), but pre-existing findings the guard surfaces without fixing.
ALLOWLIST: dict[tuple[str, int], str] = {
    ("tests/fast/test_import_integrity.py", 36): (
        "deliberately transient: that guard's untracked-file regression test writes "
        "the probe, runs the checker and deletes it in a finally block"
    ),
    ("tests/fast/test_args_dest_consistency.py", 65): (
        "deliberately transient: same shape, for the args-dest guard's own "
        "untracked-file regression test"
    ),
    ("tests/fast/test_path_anchors.py", 39): (
        "deliberately transient, same shape as the arming probe: this guard's own "
        "untracked-file regression test writes the probe, runs the checker and "
        "deletes it in a finally block, so it must NOT exist in the working tree"
    ),
    ("tests/fast/test_arming.py", 76): (
        "deliberately transient: the arming guard's falsification test writes this "
        "probe, runs the checker against it and deletes it in a finally block, so it "
        "must NOT exist in the working tree. The anchor itself is correct -- which is "
        "the point, since the test needs the probe to land somewhere the checker scans"
    ),
    ("miles/rollout/rm_hub/ifbench.py", 16): (
        "stale reference: examples/eval_multi_task/ does not exist in this repo "
        "(inherited from upstream at Orbit's public-release commit); the read is "
        "guarded by .exists() so it silently no-ops rather than crashing"
    ),
    ("miles/rollout/rm_hub/ifbench.py", 15): (
        "intentional: _WORKSPACE_PARENT is the same IFBench-sibling anchor as "
        "line 22's join, reported again here at its own definition site"
    ),
    ("miles/rollout/rm_hub/ifbench.py", 22): (
        "intentional: IFBench is cloned as a sibling of the repo root in the dev "
        "workspace, not a repo path; same convention as orbit/rewards/ultra_agents.py"
    ),
    # Every remaining entry is a DELIBERATE reference to something outside this
    # repo. The three defects this guard found on its first run were fixed, not
    # allowlisted: orbit/rewards/ultra_agents.py's fixed-depth fallback, the
    # dropped examples/swe-agent-harbor-docker/ files an upstream CI test loads,
    # and a stale examples/true_on_policy/ path left by the examples reorg.
    ("tools/adapter_runtime_compare/run_compare.py", 31): (
        "intentional: module docstring says this harness lives outside the "
        "candidate worktrees on purpose (compares two sibling checkouts)"
    ),
    ("tools/check_dsv4_deepgemm_cross_repo_parity.py", 24): (
        "intentional: PROJECT_ROOT is the same cross-repo anchor as lines 25/26's "
        "joins, reported again here at its own definition site"
    ),
    ("tools/check_dsv4_deepgemm_cross_repo_parity.py", 25): (
        "intentional: module docstring names this a cross-repo parity check; "
        "PROJECT_ROOT's sibling sglang/Megatron-LM checkouts are the point"
    ),
    ("tools/check_dsv4_deepgemm_cross_repo_parity.py", 26): (
        "intentional: same cross-repo design as line 25 (sibling Megatron-LM checkout)"
    ),
}


def _is_file_base(node: ast.AST) -> bool:
    """`Path(__file__)` or `Path(os.path.abspath(__file__))` / `Path(os.path.realpath(__file__))`."""
    if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "Path"
            and len(node.args) == 1 and not node.keywords):
        return False
    arg = node.args[0]
    if isinstance(arg, ast.Name) and arg.id == "__file__":
        return True
    if (isinstance(arg, ast.Call) and isinstance(arg.func, ast.Attribute) and arg.func.attr in ("abspath", "realpath")
            and isinstance(arg.func.value, ast.Attribute) and arg.func.value.attr == "path"
            and isinstance(arg.func.value.value, ast.Name) and arg.func.value.value.id == "os"
            and len(arg.args) == 1 and isinstance(arg.args[0], ast.Name) and arg.args[0].id == "__file__"):
        return True
    return False


def _const_int(node: ast.AST):
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return node.value
    return None


def _depth(node: ast.AST, consts: dict):
    """Levels-up-from-the-file this expression anchors to, or None if not
    statically resolvable. depth 0 == the file itself; depth 1 == its
    containing directory (`.parent` / `.parents[0]`); depth N+1 == `.parents[N]`."""
    if _is_file_base(node):
        return 0
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "resolve" \
            and not node.args and not node.keywords:
        return _depth(node.func.value, consts)
    if isinstance(node, ast.Attribute) and node.attr == "parent":
        base = _depth(node.value, consts)
        return None if base is None else base + 1
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and node.value.attr == "parents":
        n = _const_int(node.slice)
        if n is None or n < 0:
            return None
        base = _depth(node.value.value, consts)
        return None if base is None else base + n + 1
    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in consts:
        return consts[node.id]
    return None


def _extends_chain(node: ast.AST, pm: dict) -> bool:
    """True if node's parent continues the SAME __file__-anchor chain (via
    `.parent`, `.resolve()`, or `.parents[...]`), so `node` is not the
    outermost/maximal anchor expression and should not be reported on its own."""
    parent = pm.get(node)
    if isinstance(parent, ast.Attribute) and parent.value is node:
        if parent.attr in ("parent", "resolve"):
            return True
        if parent.attr == "parents":
            gp = pm.get(parent)
            return isinstance(gp, ast.Subscript) and gp.value is parent
    return False


def _literal_join(node: ast.AST):
    """Unwind a `/`-chain or `.joinpath(...)` call made ENTIRELY of string
    literals. Returns (root_expr, [segments]); segments is empty if node
    itself is the root (no join found yet). Returns None the moment a
    non-literal segment is hit."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        base = _literal_join(node.left)
        if base is None:
            return None
        if not (isinstance(node.right, ast.Constant) and isinstance(node.right.value, str)):
            return None
        root, segs = base
        return root, segs + [node.right.value]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "joinpath" \
            and not node.keywords:
        base = _literal_join(node.func.value)
        if base is None:
            return None
        args = []
        for a in node.args:
            if not (isinstance(a, ast.Constant) and isinstance(a.value, str)):
                return None
            args.append(a.value)
        root, segs = base
        return root, segs + args
    return node, []


def _is_join_node(node: ast.AST) -> bool:
    return (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)) or \
        (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "joinpath")


def _embedded_in_larger_join(node: ast.AST, pm: dict) -> bool:
    parent = pm.get(node)
    if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Div) and parent.left is node:
        return True
    if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Attribute) and parent.func.attr == "joinpath" \
            and parent.func.value is node:
        return True
    return False


def _parent_map(tree: ast.AST) -> dict:
    pm = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            pm[child] = node
    return pm


def _looks_like_source_target(segs: list) -> bool:
    """Heuristic: only require existence for joins that plausibly name a
    tracked source artifact, not a runtime-created path (log/checkpoint/output
    dir, something under a gitignored tree, an env var-shaped placeholder)."""
    if not segs:
        return False
    RUNTIME_HINTS = (
        "log", "logs", "checkpoint", "checkpoints", "ckpt", "output", "outputs",
        "tmp", "temp", "cache", "artifact", "artifacts", "result", "results",
        "wandb", "runs", "run_", "dump", "build", "dist", ".venv", "venv",
        "__pycache__", "node_modules", "snapshot", "snapshots",
    )
    joined = "/".join(segs).lower()
    if any(h in joined for h in RUNTIME_HINTS):
        return False
    return True


def _anchor_dir(f: str, depth: int) -> Path:
    """The real directory `depth` levels up from tracked file `f`, computed
    from its ACTUAL location in the repo (not the source's assumed depth).
    Uses REPO's real filesystem ancestors (never `..` string segments) so the
    result is a normalized path that `is_relative_to` can compare correctly."""
    parts = f.split("/")
    keep = len(parts) - depth
    if keep >= 0:
        return REPO.joinpath(*parts[:keep])
    above = -keep
    idx = min(above - 1, len(REPO.parents) - 1)
    return REPO.parents[idx]


def _check_file(f: str, source: str) -> list[str]:
    errors = []
    try:
        tree = ast.parse(source, filename=f)
    except SyntaxError as e:
        errors.append(f"{f}: syntax error: {e}")
        return errors

    pm = _parent_map(tree)

    # Module-level `NAME = <pure anchor expr>` (no literal join baked in) --
    # track NAME so joins/derivations elsewhere in the module resolve too.
    # The definition's own RHS is still walked like any other expression below
    # (rule 2 applies to it too), so an escaping anchor is reported even if
    # NAME turns out to be unused elsewhere in the module.
    consts: dict = {}
    for stmt in tree.body:
        target = None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            target = stmt.targets[0]
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            target = stmt.target
        if target is None:
            continue
        joined = _literal_join(stmt.value)
        if joined is not None and joined[1] == []:  # no join segments: a pure anchor expr
            d = _depth(stmt.value, consts)
            if d is not None:
                consts[target.id] = d

    consumed: set = set()

    # Rule 3 (+ rule 2 on the same root): full-literal joins off a resolvable anchor.
    for node in ast.walk(tree):
        if not _is_join_node(node) or _embedded_in_larger_join(node, pm):
            continue
        joined = _literal_join(node)
        if joined is None:
            continue
        root, segs = joined
        if not segs:
            continue
        depth = _depth(root, consts)
        if depth is None:
            continue
        consumed.add(id(root))
        key = (f, node.lineno)
        if key in ALLOWLIST:
            continue
        anchor = _anchor_dir(f, depth)
        escapes = not anchor.is_relative_to(REPO)
        target = anchor.joinpath(*segs)
        if escapes:
            errors.append(
                f"{f}:{node.lineno}: anchor at depth {depth} escapes the repo root "
                f"(-> {anchor}); join -> {target}"
            )
        elif _looks_like_source_target(segs) and not target.exists():
            errors.append(
                f"{f}:{node.lineno}: anchor {anchor.relative_to(REPO)} "
                f"joined with {'/'.join(segs)!r} -> missing target {target}"
            )

    # Rule 2 only: maximal anchor expressions not covered by a literal join above
    # (this also re-visits the RHS of a tracked constant's own definition, so an
    # escaping anchor is caught even if the constant is never used again).
    for node in ast.walk(tree):
        if id(node) in consumed or _extends_chain(node, pm):
            continue
        depth = _depth(node, consts)
        if depth is None or depth < 1:
            continue
        anchor = _anchor_dir(f, depth)
        if not anchor.is_relative_to(REPO):
            key = (f, node.lineno)
            if key in ALLOWLIST:
                continue
            errors.append(
                f"{f}:{node.lineno}: anchor at depth {depth} escapes the repo root (-> {anchor})"
            )
    return errors


def collect_errors() -> list[str]:
    errors = []
    for f in tracked_py():
        p = REPO / f
        if not p.is_file():
            continue
        try:
            source = p.read_text(errors="surrogateescape")
        except OSError:
            continue
        errors.extend(_check_file(f, source))
    return errors


if __name__ == "__main__":
    all_errors = collect_errors()
    for e in all_errors:
        print(e)
    print(f"[check-path-anchors] {'FAIL' if all_errors else 'ok'}: {len(tracked_py())} files")
    sys.exit(1 if all_errors else 0)
