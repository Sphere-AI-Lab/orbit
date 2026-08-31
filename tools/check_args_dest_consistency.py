#!/usr/bin/env python3
"""Static argparse dest consistency: every bare `args.<name>` read of a parsed
namespace must name something the parser can actually produce. Catches the
mechanical-rename failure where an option is renamed (`--miles-root` ->
`--orbit-root`, so dest `miles_root` -> `orbit_root`) but a reader still spells
the old dest: argparse derives the dest at runtime, no source token spells it,
and the AttributeError is only reachable by running the script.

Two rules, selected per READ, by where that namespace was parsed:

* Rule A -- the namespace was parsed by a call in this same file, and this file
  registers its own options statically. Valid names are the file's own dests,
  the names it writes onto the namespace, and the production training surface
  (tests/fast/args_surface_golden.json).
* Rule B -- the namespace arrived as a parameter (parsed elsewhere), or this
  file's own registration is not statically knowable. Valid names are the
  production surface plus every dest registered and every namespace attribute
  written anywhere in the repo, plus the top-level keys of every YAML a
  `--custom-config-path` names (that merge injects them onto the namespace).

A dest is derived from `add_argument` / `add_subparsers` the way argparse does:
explicit `dest=` wins, else the first long option with dashes stripped and
`-` -> `_`, else the positional name. Parsers built by reflection (dataclass
fields, `DataclassArgparseBridge`, `tap.Tap` subclasses) are resolved to the
field names they generate.

Reads that cannot raise are never flagged: `getattr(args, "x", default)` and
`hasattr` guards, and any namespace written through a dynamic key
(`setattr(args, key, v)`, `vars(args)[k] = v`, `Namespace(**payload)`).

SCOPE: tracked non-test .py files. `tests/` is excluded on purpose. Not for
noise -- including it costs only two hits -- but because folding test files into
the repo-wide valid set adds 64 names harvested from fabricated namespaces,
among them `debug`, `default`, `name`, `flag` and `labels`: exactly the generic
names a real typo would land on. Tests also run in CI, so a stale dest read
there fails loudly already; this guard exists for the code CI never executes.

Exits 1 on any inconsistency."""
import ast, json, re, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GOLDEN = REPO / "tests" / "fast" / "args_surface_golden.json"


def _ls_files(*patterns: str) -> list[str]:
    return subprocess.run(
        ["git", "-C", str(REPO), "ls-files", *patterns], capture_output=True, text=True
    ).stdout.splitlines()


tracked = [f for f in _ls_files("*.py") if not f.startswith("tests/")]

# Reads the guard is right to flag but that it must not fail on today, as
# (file, attribute). Every entry is a standing debt, not an exemption class.
ALLOWLIST = {
    # UPSTREAM BUG, inherited verbatim (present at refs/miles/base): nothing
    # registers --offload-optimizer-states, so this assert raises on the
    # --stream-optimizer-state-to-disk path. Left unfixed deliberately -- a fix
    # here is a behavioural divergence in a vendored file, and it belongs
    # upstream. Report it to radixark/miles rather than seaming around it.
    ("miles/utils/arguments.py", "offload_optimizer_states"),
    # megatron.training.arguments.validate_args() derives args.data_parallel_size
    # after parsing, so it is a real namespace attribute but not a parser dest.
    ("miles/backends/megatron_utils/initialize.py", "data_parallel_size"),
    # Megatron flags newer than the pinned Megatron-LM commit; the CLI test for
    # them skips when absent (see tests/fast/test_megatron_cli_flags.py).
    ("miles_plugins/models/glm4.py", "post_self_attn_layernorm"),
    ("miles_plugins/models/glm4.py", "post_mlp_layernorm"),
    # UPSTREAM BUG on upstream's DEFAULT path, inherited verbatim: nothing in
    # orbit, miles or the pinned Megatron-LM registers --moe-use-legacy-grouped-gemm,
    # so `--megatron-to-hf-mode raw` -- which is the DEFAULT -- dies in
    # MegatronTrainRayActor.init() with AttributeError. Confirmed by a raw-mode
    # GPU smoke on 2026-08-31 and present identically at orbit-main. It has
    # bit-rotted because every recipe in the repo passes `bridge`. Left unfixed
    # deliberately: it is upstream's code on upstream's default path, so the fix
    # belongs upstream, not as a local divergence in a merge-hot file.
    ("miles/backends/megatron_utils/model_provider.py", "moe_use_legacy_grouped_gemm"),
    ("miles_plugins/models/glm4.py", "moe_use_legacy_grouped_gemm"),
}

# Calls whose result is a parsed argparse namespace. `SimpleNamespace` is
# deliberately absent: argparse never returns one, and it is the standard way to
# fake an arbitrary domain object, so treating it as a namespace is pure noise.
NS_CALLS = ("parse_args", "parse_arguments", "Namespace")
# Calls that register options on a parser.
REGISTRARS = ("add_argument", "add_subparsers")


def callee(call: ast.Call) -> str | None:
    f = call.func
    if isinstance(f, ast.Attribute):
        return f.attr
    if isinstance(f, ast.Name):
        return f.id
    return None


def annotation_verdict(annotation) -> str:
    """`nsparam` for `Namespace` / `argparse.Namespace | None`; `unknown` for
    `Any` and `object`, which annotate a namespace without naming it (the
    parameter name then decides); `other` for a real type -- `SimpleNamespace`
    and typer/dataclass config objects included."""
    parts = [
        p.strip().rsplit(".", 1)[-1]
        for p in ast.unparse(annotation).replace("Optional[", "").replace("]", "|").split("|")
    ]
    if "Namespace" in parts:
        return "nsparam"
    return "unknown" if set(parts) <= {"Any", "object", "None", ""} else "other"


def ns_producers(tree: ast.Module) -> set[str]:
    """Names that return a parsed namespace: the argparse spellings plus any
    local alias of them (`from x import parse_args as megatron_parse_args`)."""
    names = set(NS_CALLS)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                if al.asname and al.name.rsplit(".", 1)[-1] in NS_CALLS:
                    names.add(al.asname)
    return names


class Scope:
    """One lexical scope; `kind[name]` is the set of ways `name` was bound.
    `base` holds the bindings that do not depend on classification order
    (parameters), so a second classification round can start from them while
    still consulting the previous round through `prev`."""

    def __init__(self, parent):
        self.parent = parent
        self.kind: dict[str, set[str]] = {}
        self.base: dict[str, set[str]] = {}
        self.prev: dict[str, set[str]] = {}

    def mark(self, name: str, kind: str) -> None:
        self.kind.setdefault(name, set()).add(kind)

    def new_round(self) -> None:
        self.prev = self.kind
        self.kind = {k: set(v) for k, v in self.base.items()}

    def was_ns(self, name: str) -> bool:
        s = self
        while s is not None:
            if name in s.kind or name in s.prev:
                return bool(
                    {"ns", "nsparam", "nsdyn"}
                    & (set(s.kind.get(name, ())) | set(s.prev.get(name, ())))
                )
            s = s.parent
        return False

    def lookup(self, name: str):
        s = self
        while s is not None:
            if name in s.kind:
                return s
            s = s.parent
        return None


def _bind(target, scope: Scope, kind: str) -> None:
    if isinstance(target, ast.Name):
        scope.mark(target.id, kind)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for e in target.elts:
            _bind(e, scope, "other")
    elif isinstance(target, ast.Starred):
        _bind(target.value, scope, "other")


def build_scopes(tree: ast.Module) -> dict[int, Scope]:
    """Map every node to its enclosing scope, classifying each binding as
    `ns` (parsed by a call in this file), `nsparam`/`argsparam` (a parameter
    annotated `Namespace`, or an un-annotated one named `args`), `nsdyn` (a
    namespace whose attributes cannot be enumerated) or `other` (everything
    else -- `*args` tuples, dataclass parameters, loop variables, imports)."""
    scope_of: dict[int, Scope] = {}
    producers = ns_producers(tree)
    scopes: list[Scope] = []

    def visit(node, scope: Scope) -> None:
        scope_of[id(node)] = scope
        inner = scope
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            inner = Scope(scope)
            scopes.append(inner)
            a = node.args
            for p in list(a.posonlyargs) + list(a.args) + list(a.kwonlyargs):
                # A real annotation settles it: `args: Namespace` is the parsed
                # namespace, `args: ScriptArgs` (typer/dataclass) is not.
                verdict = annotation_verdict(p.annotation) if p.annotation else "unknown"
                if verdict == "unknown":
                    inner.mark(p.arg, "argsparam" if p.arg == "args" else "other")
                else:
                    inner.mark(p.arg, verdict)
            for extra in (a.vararg, a.kwarg):  # `*args` is a tuple, never a namespace
                if extra is not None:
                    inner.mark(extra.arg, "other")
        elif isinstance(node, ast.ClassDef):
            inner = Scope(scope)
            scopes.append(inner)
        for child in ast.iter_child_nodes(node):
            visit(child, inner)

    root = Scope(None)
    scopes.append(root)
    visit(tree, root)
    for s in scopes:
        s.base = {k: set(v) for k, v in s.kind.items()}

    def rhs_kind(value, scope: Scope, target=None) -> str:
        if isinstance(value, ast.Call):
            if callee(value) in producers:
                # `Namespace(**payload)` is a namespace whose attributes cannot
                # be enumerated: still a namespace, never checked.
                return "nsdyn" if any(k.arg is None for k in value.keywords) else "ns"
            if callee(value) in ("deepcopy", "copy") and value.args:
                return rhs_kind(value.args[0], scope)
            # `args = set_default_megatron_args(args)` threads the namespace
            # through a helper; it is still the same namespace.
            if isinstance(target, ast.Name) and any(
                isinstance(a, ast.Name) and a.id == target.id for a in value.args
            ):
                return "ns" if scope.was_ns(target.id) else "other"
            return "other"
        if isinstance(value, ast.Name):
            return "ns" if scope.was_ns(value.id) else "other"
        return "other"

    # Two rounds so an alias or pass-through classified before its producer
    # (ast.walk is breadth-first, so a nested producer can come later) settles.
    for _ in range(2):
        for s in scopes:
            s.new_round()
        _classify(tree, scope_of, rhs_kind)
    return scope_of


def _classify(tree, scope_of, rhs_kind) -> None:
    for node in ast.walk(tree):
        sc = scope_of.get(id(node))
        if sc is None:
            continue
        if isinstance(node, ast.Assign):
            for t in node.targets:
                _bind(t, sc, rhs_kind(node.value, sc, t))
        elif isinstance(node, ast.AnnAssign):
            _bind(node.target, sc, rhs_kind(node.value, sc, node.target) if node.value else "other")
        elif isinstance(node, ast.NamedExpr):
            _bind(node.target, sc, rhs_kind(node.value, sc, node.target))
        elif isinstance(node, (ast.AugAssign, ast.For, ast.AsyncFor)):
            _bind(node.target, sc, "other")
        elif isinstance(node, ast.comprehension):
            _bind(node.target, sc, "other")
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            _bind(node.optional_vars, sc, "other")
        elif isinstance(node, ast.ExceptHandler) and node.name:
            sc.mark(node.name, "other")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for al in node.names:
                sc.mark((al.asname or al.name).split(".")[0], "other")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            sc.mark(node.name, "other")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for n in node.names:
                sc.mark(n, "other")


def _kinds(node: ast.Name, scope_of: dict[int, Scope]) -> set[str]:
    sc = scope_of.get(id(node))
    owner = sc.lookup(node.id) if sc else None
    return owner.kind.get(node.id, set()) if owner else set()


def read_scope(node: ast.Name, scope_of: dict[int, Scope]) -> str | None:
    """`local` when the namespace was parsed by THIS file (its own parser
    defines the valid dests), `foreign` when it arrived as a parameter (parsed
    elsewhere, so only the repo-wide surface can judge it), None when the name
    is not a namespace or is ambiguously bound."""
    kinds = _kinds(node, scope_of)
    if not kinds:
        return None
    if kinds == {"ns"}:
        return "local"
    if node.id != "args" and "argsparam" in kinds:
        return None
    # A parameter, or a parameter reassigned from a parser: parsed elsewhere.
    return "foreign" if kinds <= {"ns", "nsparam", "argsparam"} else None


def is_namespace(node: ast.Name, scope_of: dict[int, Scope]) -> bool:
    """The name is a parsed namespace whose attribute set is enumerable."""
    return read_scope(node, scope_of) is not None


def is_namespace_like(node: ast.Name, scope_of: dict[int, Scope]) -> bool:
    """The name is a namespace, enumerable or not (`Namespace(**payload)`).
    Writes onto it still tell us a dest by that name exists somewhere."""
    kinds = _kinds(node, scope_of)
    if node.id != "args" and "argsparam" in kinds:
        return False
    return bool(kinds) and kinds <= {"ns", "nsparam", "nsdyn", "argsparam"}


def _ns_dict_owner(func, scope_of: dict[int, Scope]) -> ast.Name | None:
    """`<namespace>.__dict__.<method>` -- a dynamic write channel."""
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Attribute)
        and func.value.attr == "__dict__"
        and isinstance(func.value.value, ast.Name)
        and is_namespace_like(func.value.value, scope_of)
    ):
        return func.value.value
    return None


def _mutated_vars_calls(tree: ast.Module) -> set[int]:
    """ids of `vars(x)` calls whose result is written through -- `vars(x)["k"]=v`,
    `vars(x).update(...)`. A read-only `set(vars(args))` mutates nothing."""
    mutated = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and not isinstance(node.ctx, ast.Load):
            if isinstance(node.value, ast.Call) and callee(node.value) == "vars":
                mutated.add(id(node.value))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            inner = node.func.value
            if (
                node.func.attr in ("update", "setdefault", "pop", "clear", "__setitem__")
                and isinstance(inner, ast.Call)
                and callee(inner) == "vars"
            ):
                mutated.add(id(inner))
    return mutated


def dest_of(call: ast.Call) -> str | None:
    """argparse's own dest derivation. None means "not statically knowable"."""
    for kw in call.keywords:
        if kw.arg is None:
            return None  # **kwargs may carry dest=
        if kw.arg == "dest":
            v = kw.value
            return v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else None
    if callee(call) == "add_subparsers":
        return ""  # subparsers without dest= store nothing
    longs, shorts, positionals = [], [], []
    for a in call.args:
        if not (isinstance(a, ast.Constant) and isinstance(a.value, str)):
            return None
        if a.value.startswith("--"):
            longs.append(a.value)
        elif a.value.startswith("-"):
            shorts.append(a.value)
        else:
            positionals.append(a.value)
    for group in (longs, shorts, positionals):
        if group:
            return group[0].lstrip("-").replace("-", "_")
    return None


def annotated_attrs(cls: ast.ClassDef) -> list[str]:
    return [
        n.target.id
        for n in cls.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name)
    ]


def reflected_dests(tree: ast.Module, classes: dict[str, ast.ClassDef]) -> set[str]:
    """Dests produced by parsers built from a class instead of literal flags:
    a `for f in fields(X): parser.add_argument(f"--{f.name}...")` loop, a
    `Bridge(X, prefix="script")` (dest `script_<field>`), and `tap.Tap`
    subclasses whose class annotations are the dests."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.AsyncFor)):
            it = node.iter
            if (
                isinstance(it, ast.Call)
                and callee(it) == "fields"
                and it.args
                and isinstance(it.args[0], ast.Name)
                and it.args[0].id in classes
                and any(
                    isinstance(n, ast.Call) and callee(n) in REGISTRARS for n in ast.walk(node)
                )
            ):
                found.update(annotated_attrs(classes[it.args[0].id]))
        elif isinstance(node, ast.Call):
            prefix = next(
                (
                    kw.value.value
                    for kw in node.keywords
                    if kw.arg == "prefix"
                    and isinstance(kw.value, ast.Constant)
                    and isinstance(kw.value.value, str)
                ),
                None,
            )
            if prefix is not None and node.args and isinstance(node.args[0], ast.Name):
                cls = classes.get(node.args[0].id)
                if cls is not None:
                    stem = f"{prefix}_" if prefix else ""
                    found.update(stem + a for a in annotated_attrs(cls))
            # `Args().parse_args()` (tap.Tap): the class annotations are the dests.
            f = node.func
            if (
                callee(node) == "parse_args"
                and isinstance(f, ast.Attribute)
                and isinstance(f.value, ast.Call)
                and isinstance(f.value.func, ast.Name)
                and f.value.func.id in classes
            ):
                found.update(annotated_attrs(classes[f.value.func.id]))
    return found


_CUSTOM_CONFIG = re.compile(r"--custom-config-path[= ]+\"?([\w./-]+\.ya?ml)")
_YAML_KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*):")


def custom_config_keys() -> set[str]:
    """`--custom-config-path some.yaml` merges that file's top-level keys onto
    the namespace at run time, so they are dests no `add_argument` spells."""
    referenced: set[str] = set()
    for f in _ls_files("*.py", "*.sh", "*.md"):
        p = REPO / f
        if p.is_file():
            referenced.update(_CUSTOM_CONFIG.findall(p.read_text(errors="surrogateescape")))
    keys: set[str] = set()
    for rel in sorted(referenced):
        p = REPO / rel
        if not p.is_file():
            continue
        for line in p.read_text(errors="surrogateescape").splitlines():
            m = _YAML_KEY.match(line)
            if m:
                keys.add(m.group(1))
    return keys


def literal_dict_keys(tree: ast.Module) -> dict[str, set[str] | None]:
    """Module-level `NAME = {"k": ...}` tables, so `set_defaults(**NAME)` is
    still a knowable set of dests. None means the keys are not all literals."""
    tables: dict[str, set[str] | None] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        if not isinstance(value, ast.Dict):
            continue
        keys = {k.value for k in value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        ok = len(keys) == len(value.keys)
        for t in targets:
            if isinstance(t, ast.Name):
                tables[t.id] = keys if ok else None
    return tables


def scan(rel: str, tree: ast.Module) -> dict:
    scope_of = build_scopes(tree)
    classes = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)}
    tables = literal_dict_keys(tree)
    dests: set[str] = set()
    written: set[str] = set()  # names put onto a namespace, or read tolerantly
    reads: list[tuple[int, str, str, str]] = []
    opaque = False  # this file's parser registers dests the guard cannot see
    opaque_vars: set[str] = set()  # namespaces written through a dynamic key
    mutated_vars = _mutated_vars_calls(tree)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = callee(node)
            if name in REGISTRARS:
                d = dest_of(node)
                if d is None:
                    opaque = True
                elif d:
                    dests.add(d)
            elif name == "set_defaults":
                for kw in node.keywords:
                    if kw.arg is not None:
                        written.add(kw.arg)
                        continue
                    keys = (
                        tables.get(kw.value.id) if isinstance(kw.value, ast.Name) else None
                    )
                    if keys is None:
                        opaque = True  # set_defaults(**cfg) can define any dest
                    else:
                        written |= keys
            elif name == "Namespace":
                # Literal kwargs are dests; `Namespace(**payload)` only makes
                # the resulting variable unenumerable (kind `nsdyn`).
                written.update(kw.arg for kw in node.keywords if kw.arg)
            elif name in ("setattr", "getattr", "hasattr") and len(node.args) >= 2:
                if not isinstance(node.args[0], ast.Name) or not is_namespace_like(
                    node.args[0], scope_of
                ):
                    continue
                key = node.args[1]
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    written.add(key.value)
                elif name == "setattr":
                    # A dynamic READ tells us nothing; a dynamic WRITE can put
                    # any name on that one namespace.
                    opaque_vars.add(node.args[0].id)
            elif name == "vars" and id(node) in mutated_vars and node.args:
                if isinstance(node.args[0], ast.Name) and is_namespace_like(
                    node.args[0], scope_of
                ):
                    opaque_vars.add(node.args[0].id)
            elif name in ("setdefault", "update", "pop"):
                owner = _ns_dict_owner(node.func, scope_of)
                if owner is None:
                    continue
                key = node.args[0] if node.args else None
                if name == "setdefault" and isinstance(key, ast.Constant) and isinstance(key.value, str):
                    written.add(key.value)  # args.__dict__.setdefault("x", ...)
                else:
                    opaque_vars.add(owner.id)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                if (
                    isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and is_namespace_like(t.value, scope_of)
                ):
                    written.add(t.attr)
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and not node.attr.startswith("_")
        ):
            where = read_scope(node.value, scope_of)
            if where is not None:
                reads.append((node.lineno, node.value.id, node.attr, where))

    dests |= reflected_dests(tree, classes)
    reads = [r for r in reads if r[1] not in opaque_vars]
    return dict(rel=rel, dests=dests, written=written, reads=reads, opaque=opaque)


def collect_errors() -> list[str]:
    golden = {r["dest"] for r in json.loads(GOLDEN.read_text(encoding="utf-8"))}
    infos, errors = [], []
    for f in tracked:
        p = REPO / f
        if not p.is_file():
            continue
        try:
            tree = ast.parse(p.read_text(errors="surrogateescape"), filename=f)
        except SyntaxError as e:
            errors.append(f"{f}: syntax error: {e}")
            continue
        infos.append(scan(f, tree))

    repo_wide = golden | custom_config_keys()
    for info in infos:
        repo_wide |= info["dests"] | info["written"]

    for info in infos:
        # Rule A: a file that registers its own options is held to its own
        # surface. Rule B: a file that only consumes a namespace parsed
        # elsewhere -- or whose own registration is not statically knowable --
        # sees every dest the repo registers.
        own = golden | info["dests"] | info["written"]
        wide = repo_wide | info["dests"] | info["written"]
        rule_a = bool(info["dests"]) and not info["opaque"]
        for lineno, var, attr, where in sorted(set(info["reads"])):
            local = rule_a and where == "local"
            if attr in (own if local else wide) or (info["rel"], attr) in ALLOWLIST:
                continue
            errors.append(
                f"{info['rel']}:{lineno}: {var}.{attr} -> not a registered dest "
                f"({'own parser' if local else 'repo-wide'})"
            )
    return errors


if __name__ == "__main__":
    all_errors = collect_errors()
    for e in all_errors:
        print(e)
    print(f"[verify-args-dest] {'FAIL' if all_errors else 'ok'}: {len(tracked)} files")
    sys.exit(1 if all_errors else 0)
