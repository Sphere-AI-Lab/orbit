#!/usr/bin/env python3
"""Static call-signature agreement: for the calls whose callee can be resolved
uniquely inside this repo, check that every required parameter is supplied and no
unexpected keyword is passed. Catches the classic mechanical-refactor failure where
a callee gains a required parameter (upstream `forward_only(..., rollout_id)`) and one
caller is left behind -> TypeError only reachable at run time. Exits 1 on any mismatch.

Under-reporting is the design goal: anything whose callee cannot be resolved uniquely
and whose signature cannot be trusted (decorators, conditional defs, rebindings,
*args/**kwargs, star-args at the call site) is skipped, not guessed at."""
import ast, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# `--others` matters as much as `--cached`: a brand-new module is untracked until
# it is staged, and a file this guard cannot see is a file whose callees it
# resolves to the wrong definition -- or not at all. Same reasoning, same flags,
# as tools/check_patch_pins.py.
tracked = sorted(set(subprocess.run(
    ["git", "-C", str(REPO), "ls-files", "--cached", "--others", "--exclude-standard", "*.py"],
    capture_output=True, text=True,
).stdout.splitlines()))

# Decorators that provably preserve the wrapped callable's call signature. A callee
# carrying any other decorator is skipped: the decorator may rewrite the signature.
SAFE_FUNC_DECORATORS = {
    "torch.no_grad", "torch.enable_grad", "torch.inference_mode", "torch.compile",
    "no_grad", "enable_grad", "inference_mode",
    "abc.abstractmethod", "abstractmethod",
    "typing.override", "typing_extensions.override", "override",
    "functools.cache", "functools.lru_cache", "cache", "lru_cache",
    "contextlib.contextmanager", "contextlib.asynccontextmanager",
    "contextmanager", "asynccontextmanager",
    "typing.final", "final",
}
# Class decorators that do not rewrite user-defined methods.
SAFE_CLASS_DECORATORS = {"dataclasses.dataclass", "dataclass", "functools.total_ordering", "total_ordering"}

# Attribute lookups that defeat static method resolution entirely.
DYNAMIC_ATTR_HOOKS = ("__getattr__", "__getattribute__")

# Pre-existing mismatches, tolerated but not expanded (file, call label, verdict).
# Empty by design: this guard only reports a call it can prove is a TypeError
# against every candidate definition, so an entry here means a defect is being
# tolerated. The first one it found (a stale `entropy_no_grad` kwarg in
# tests/test_true_on_policy_logprobs.py, renamed with inverted sense by the
# phase-4 upstream merge) was fixed rather than allowlisted.
ALLOWLIST: set[tuple[str, str, str]] = set()

STATS = {"checked": 0, "skipped": 0, "files": 0}

AMBIGUOUS = object()  # a name that resolves to more than one thing, or to something untrustworthy


def _dec_key(d: ast.expr) -> str:
    return ast.unparse(d.func if isinstance(d, ast.Call) else d)


def _safe_decorators(node, allowed: set[str]) -> bool:
    return all(_dec_key(d) in allowed for d in node.decorator_list)


def _signature(fn) -> dict:
    """Positional/keyword shape of a def, as far as a call site needs it."""
    a = fn.args
    pos = a.posonlyargs + a.args
    first_default = len(pos) - len(a.defaults)
    return {
        "pos": [(p.arg, i >= first_default) for i, p in enumerate(pos)],
        "n_posonly": len(a.posonlyargs),
        "kwonly": [(p.arg, a.kw_defaults[i] is not None) for i, p in enumerate(a.kwonlyargs)],
        "vararg": a.vararg is not None,
        "kwarg": a.kwarg is not None,
        "where": f"{fn._src_file}:{fn.lineno}",
        "name": fn.name,
    }


def _mismatch(call: ast.Call, sig: dict, drop_self: bool) -> str | None:
    """Return a description of a definite TypeError, or None when the call is fine
    or cannot be judged. `*args`/`**kwargs` on either side relaxes the check entirely."""
    if sig["vararg"] or sig["kwarg"]:
        return None
    if any(isinstance(a, ast.Starred) for a in call.args):
        return None
    if any(k.arg is None for k in call.keywords):
        return None
    pos, n_posonly = sig["pos"], sig["n_posonly"]
    if drop_self:
        if not pos or pos[0][0] != "self":
            return None  # unbound-looking method (staticmethod-ish); not judgeable
        pos, n_posonly = pos[1:], max(0, n_posonly - 1)
    given = len(call.args)
    kw = [k.arg for k in call.keywords]
    if len(set(kw)) != len(kw):
        return None
    if given > len(pos):
        return f"{sig['name']}() takes {len(pos)} positional arg(s), {given} given"
    byname = {n for n, _ in pos[n_posonly:]} | {n for n, _ in sig["kwonly"]}
    for k in kw:
        if k not in byname:
            return f"{sig['name']}() got an unexpected keyword argument '{k}'"
    for i, (n, _) in enumerate(pos):
        if i < given and n in kw:
            return f"{sig['name']}() got multiple values for argument '{n}'"
    missing = [n for i, (n, dflt) in enumerate(pos) if i >= given and not dflt and n not in kw]
    missing += [n for n, dflt in sig["kwonly"] if not dflt and n not in kw]
    if missing:
        return f"{sig['name']}() missing required argument(s): " + ", ".join(missing)
    return None


# ---------------------------------------------------------------------------
# Repo index
# ---------------------------------------------------------------------------

class ClassInfo:
    __slots__ = ("name", "module", "file", "lineno", "bases", "methods", "poisoned", "usable", "mro")

    def __init__(self, name, module, file, lineno):
        self.name, self.module, self.file, self.lineno = name, module, file, lineno
        self.bases: list = []          # raw ast base expressions
        self.methods: dict = {}        # name -> signature dict, or AMBIGUOUS
        self.poisoned: set = set()     # names shadowed by an attribute/class-level assignment
        self.usable = True             # False when a decorator may have rewritten the class
        self.mro = False               # False = not computed, None = uncomputable, list = C3 order


class Unresolved:
    """Opaque stand-in for an out-of-repo base class, deduplicated by source text."""
    __slots__ = ("text",)

    def __init__(self, text):
        self.text = text


class ModIndex:
    __slots__ = ("dotted", "file", "funcs", "classes", "aliases", "bound_once", "mod_attr_stores")

    def __init__(self, dotted, file):
        self.dotted, self.file = dotted, file
        self.funcs: dict = {}          # top-level def name -> signature dict, or AMBIGUOUS
        self.classes: dict = {}        # top-level class name -> ClassInfo, or AMBIGUOUS
        self.aliases: dict = {}        # local name -> ("mod", dotted) | ("obj", dotted, origname)
        self.bound_once: set = set()   # names bound exactly once in the whole file
        self.mod_attr_stores: set = set()  # names assigned as <something>.<name> = ... in this file


def _dotted(path: str) -> str:
    d = path[:-3].replace("/", ".")
    return d[: -len(".__init__")] if d.endswith(".__init__") else d


PACKAGE_ROOTS = {p.name for p in REPO.iterdir() if p.is_dir() and (p / "__init__.py").is_file()}
MODULES: dict[str, ModIndex] = {}


def _file_scan(tree: ast.Module) -> tuple[set, set]:
    """One walk yielding (names bound exactly once in the file, attribute-assignment
    targets). The first lets a single-import alias be trusted; the second flags names
    that some `x.<name> = ...` / `setattr(x, "<name>", ...)` may have monkey-patched."""
    counts: dict = {}
    attr_stores: set = set()

    def bump(n):
        counts[n] = counts.get(n, 0) + 1

    def store_targets(targets):
        for t in targets:
            for sub in ast.walk(t):
                if isinstance(sub, ast.Attribute):
                    attr_stores.add(sub.attr)

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, (ast.Store, ast.Del)):
                bump(node.id)
        elif isinstance(node, ast.arg):
            bump(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bump(node.name)
        elif isinstance(node, ast.Import):
            for a in node.names:
                bump((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                if a.name != "*":
                    bump(a.asname or a.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bump(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name:
            bump(node.name)
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bump(node.rest)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for n in node.names:
                bump(n)
        elif isinstance(node, ast.Assign):
            store_targets(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
            store_targets([node.target])
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and isinstance(node.args[1].value, str)
        ):
            attr_stores.add(node.args[1].value)
    return {n for n, c in counts.items() if c == 1}, attr_stores


def _self_attr_stores(node) -> set:
    """`self.<name> = ...` anywhere under a class: such a name may shadow a method."""
    out = set()
    for n in ast.walk(node):
        tgts = []
        if isinstance(n, ast.Assign):
            tgts = n.targets
        elif isinstance(n, (ast.AnnAssign, ast.AugAssign, ast.For, ast.AsyncFor)):
            tgts = [n.target]
        elif isinstance(n, ast.withitem) and n.optional_vars is not None:
            tgts = [n.optional_vars]
        for t in tgts:
            for sub in ast.walk(t):
                if (
                    isinstance(sub, ast.Attribute)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id == "self"
                ):
                    out.add(sub.attr)
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "setattr"
            and len(n.args) >= 2
            and isinstance(n.args[0], ast.Name)
            and n.args[0].id == "self"
        ):
            if isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, str):
                out.add(n.args[1].value)
            else:
                out.add("*")  # dynamic setattr on self: distrust the whole class
    return out


def _index_file(f: str) -> ast.Module | None:
    p = REPO / f
    if not p.is_file():
        return None
    try:
        tree = ast.parse(p.read_text(errors="surrogateescape"), filename=f)
    except (SyntaxError, ValueError, RecursionError):
        return None  # unparseable for this interpreter: nothing to say about it
    dotted = _dotted(f)
    mod = ModIndex(dotted, f)
    mod.bound_once, mod.mod_attr_stores = _file_scan(tree)

    # Only *direct* children of the module body are trustworthy definitions; anything
    # nested in an `if`/`try` is conditional and gets poisoned instead.
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node._src_file = f
            if node.name in mod.funcs or node.name in mod.classes:
                mod.funcs[node.name] = AMBIGUOUS
            elif not _safe_decorators(node, SAFE_FUNC_DECORATORS):
                mod.funcs[node.name] = AMBIGUOUS
            else:
                mod.funcs[node.name] = _signature(node)
        elif isinstance(node, ast.ClassDef):
            if node.name in mod.classes or node.name in mod.funcs:
                mod.classes[node.name] = AMBIGUOUS
                continue
            ci = ClassInfo(node.name, dotted, f, node.lineno)
            ci.usable = _safe_decorators(node, SAFE_CLASS_DECORATORS)
            ci.bases = list(node.bases) if not node.keywords else []
            if node.keywords:  # metaclass=/other class kwargs: distrust the hierarchy
                ci.usable = False
            ci.poisoned = _self_attr_stores(node)
            node._ci = ci  # marks this exact node as the module-level class, not a same-named local
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    sub._src_file = f
                    sub._is_method = True
                    if sub.name in ci.methods:
                        ci.methods[sub.name] = AMBIGUOUS
                    elif not _safe_decorators(sub, SAFE_FUNC_DECORATORS):
                        ci.methods[sub.name] = AMBIGUOUS
                    else:
                        ci.methods[sub.name] = _signature(sub)
                elif isinstance(sub, (ast.Assign, ast.AnnAssign)):
                    for t in (sub.targets if isinstance(sub, ast.Assign) else [sub.target]):
                        for nm in ast.walk(t):
                            if isinstance(nm, ast.Name):
                                ci.poisoned.add(nm.id)
                elif isinstance(sub, (ast.If, ast.Try)):
                    for nested in ast.walk(sub):
                        if isinstance(nested, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                            ci.poisoned.add(nested.name)
            mod.classes[node.name] = ci
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for t in (node.targets if isinstance(node, ast.Assign) else [node.target]):
                for nm in ast.walk(t):
                    if isinstance(nm, ast.Name):
                        mod.funcs[nm.id] = AMBIGUOUS
                        mod.classes[nm.id] = AMBIGUOUS
        elif isinstance(node, (ast.If, ast.Try)):
            for nested in ast.walk(node):
                if isinstance(nested, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    mod.funcs[nested.name] = AMBIGUOUS
                    mod.classes[nested.name] = AMBIGUOUS
    MODULES[dotted] = mod
    return tree


# ---------------------------------------------------------------------------
# orbit's patch layer: what actually runs at a patched target
# ---------------------------------------------------------------------------

PATCHED: dict = {}  # (target module, attr) -> signature of orbit's replacement


def _build_patch_overrides(trees: dict) -> None:
    """Index every `@patch_function("module", "attr", ...)` in orbit/.

    orbit/patch/runtime.py swaps a vendored function for one declared in orbit/,
    so at a patched target the signature a call site must satisfy is the
    REPLACEMENT's, not the vendored `def`'s. That matters in both directions: a
    replacement that widens the signature (orbit's `compute_pass_rate` takes
    `k_values`/`scale`, which upstream's parameters have no room for) makes its
    call sites correct, and one that narrows makes them wrong even though the
    vendored text would accept them. Reading the vendored `def` alone gets both
    backwards. tools/check_patch_pins.py is what keeps the declaration itself
    honest, and it rejects any target it cannot read as a plain string literal,
    so a declaration this loop skips is already a hard failure over there.
    """
    for f, tree in trees.items():
        if not f.startswith("orbit/"):
            continue
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                if not isinstance(dec, ast.Call):
                    continue
                dname = dec.func.attr if isinstance(dec.func, ast.Attribute) else getattr(dec.func, "id", None)
                if dname != "patch_function":
                    continue
                pos = [a.value for a in dec.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                kw = {
                    k.arg: k.value.value
                    for k in dec.keywords
                    if k.arg and isinstance(k.value, ast.Constant) and isinstance(k.value.value, str)
                }
                module = kw.get("module") or (pos[0] if len(pos) > 0 else None)
                attr = kw.get("attr") or (pos[1] if len(pos) > 1 else None)
                if not module or not attr:
                    continue
                node._src_file = f
                PATCHED[(module, attr)] = _signature(node)


def _resolve_module(dotted: str) -> ModIndex | None:
    if dotted.split(".")[0] not in PACKAGE_ROOTS:
        return None
    return MODULES.get(dotted)


def _abs_module(mod: ModIndex, node: ast.ImportFrom) -> str | None:
    if not node.level:
        return node.module
    pkg = mod.dotted.split(".")
    if not mod.file.endswith("__init__.py"):
        pkg = pkg[:-1]
    if node.level > 1:
        pkg = pkg[: len(pkg) - (node.level - 1)]
    if not pkg:
        return None
    return ".".join(pkg + (node.module.split(".") if node.module else []))


def _build_aliases(mod: ModIndex, tree: ast.Module) -> None:
    """Top-level imports only; conditional (`if`/`try`) and function-local ones are ignored."""
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                local = a.asname or a.name.split(".")[0]
                target = a.name if a.asname else a.name.split(".")[0]
                if local in mod.bound_once and _resolve_module(target) is not None:
                    mod.aliases[local] = ("mod", target)
        elif isinstance(node, ast.ImportFrom):
            base = _abs_module(mod, node)
            if not base:
                continue
            for a in node.names:
                if a.name == "*":
                    continue
                local = a.asname or a.name
                if local not in mod.bound_once:
                    continue
                if _resolve_module(f"{base}.{a.name}") is not None:
                    mod.aliases[local] = ("mod", f"{base}.{a.name}")
                elif _resolve_module(base) is not None:
                    mod.aliases[local] = ("obj", base, a.name)


def _resolve_class(mod: ModIndex, expr: ast.expr):
    """A base-class expression -> ClassInfo, or None when it is not an in-repo class."""
    if isinstance(expr, ast.Name):
        local = mod.classes.get(expr.id)
        if isinstance(local, ClassInfo):
            return local
        if local is AMBIGUOUS:
            return None
        a = mod.aliases.get(expr.id)
        if a and a[0] == "obj":
            tgt = _resolve_module(a[1])
            c = tgt.classes.get(a[2]) if tgt else None
            return c if isinstance(c, ClassInfo) else None
        return None
    if isinstance(expr, ast.Attribute) and isinstance(expr.value, ast.Name):
        a = mod.aliases.get(expr.value.id)
        if a and a[0] == "mod":
            tgt = _resolve_module(a[1])
            c = tgt.classes.get(expr.attr) if tgt else None
            return c if isinstance(c, ClassInfo) else None
    return None


_UNRESOLVED: dict[str, Unresolved] = {}


def _base_nodes(ci: ClassInfo) -> list:
    out = []
    mod = MODULES[ci.module]
    for b in ci.bases:
        r = _resolve_class(mod, b)
        if r is None:
            text = ast.unparse(b)
            out.append(_UNRESOLVED.setdefault(text, Unresolved(text)))
        else:
            out.append(r)
    return out


def _linearize(ci, stack=()):
    """C3 over in-repo classes; out-of-repo bases are opaque leaf nodes."""
    if isinstance(ci, Unresolved):
        return [ci]
    if ci.mro is not False:
        return ci.mro
    if ci in stack or not ci.usable:
        return None  # cyclic or decorated hierarchy: uncached, the answer is context-bound
    seqs, direct = [], _base_nodes(ci)
    for b in direct:
        lin = _linearize(b, stack + (ci,))
        if lin is None:
            ci.mro = None
            return None
        seqs.append(list(lin))
    seqs.append(list(direct))
    result = [ci]
    while True:
        seqs = [s for s in seqs if s]
        if not seqs:
            break
        for s in seqs:
            head = s[0]
            if not any(head in t[1:] for t in seqs):
                break
        else:
            ci.mro = None
            return None  # inconsistent hierarchy
        result.append(head)
        for s in seqs:
            if s and s[0] is head:
                del s[0]
    ci.mro = result
    return result


_DESCENDANTS: dict = {}


def _build_descendants() -> None:
    children: dict = {}
    for mod in MODULES.values():
        for ci in mod.classes.values():
            if not isinstance(ci, ClassInfo):
                continue
            for b in _base_nodes(ci):
                if isinstance(b, ClassInfo):
                    children.setdefault(b, []).append(ci)
    for mod in MODULES.values():
        for ci in mod.classes.values():
            if not isinstance(ci, ClassInfo):
                continue
            seen, queue = set(), list(children.get(ci, ()))
            while queue:
                c = queue.pop()
                if c in seen:
                    continue
                seen.add(c)
                queue.extend(children.get(c, ()))
            _DESCENDANTS[ci] = seen


def _method_target(ci: ClassInfo, name: str):
    """MRO lookup for `self.<name>`: the resolved signature, or None when unsure."""
    if name.startswith("__") and name.endswith("__"):
        return None
    lin = _linearize(ci)
    if lin is None:
        return None
    for node in lin:
        if isinstance(node, Unresolved):
            return None  # an out-of-repo class could define or override it first
        if not node.usable:
            return None
        if "*" in node.poisoned or name in node.poisoned:
            return None
        if any(h in node.methods for h in DYNAMIC_ATTR_HOOKS):
            return None
        sig = node.methods.get(name)
        if sig is AMBIGUOUS:
            return None
        if sig is not None:
            return sig
    return None


# ---------------------------------------------------------------------------
# Call-site checking
# ---------------------------------------------------------------------------

def _self_targets(ci: ClassInfo, name: str):
    """Every signature `self.<name>(...)` could dispatch to: the MRO target plus every
    in-repo override on a subclass. None when any of them is untrustworthy."""
    sig = _method_target(ci, name)
    if sig is None:
        return None
    candidates = [sig]
    for d in _DESCENDANTS.get(ci, ()):
        if not d.usable or "*" in d.poisoned or name in d.poisoned:
            return None
        override = d.methods.get(name)
        if override is AMBIGUOUS:
            return None
        if override is not None:
            candidates.append(override)
    return candidates


def _resolve_call(mod: ModIndex, call: ast.Call, ci):
    """(label, candidate signatures, drop_self) for a resolvable call, else None."""
    fn = call.func
    if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id == "self":
        if ci is None:
            return None
        cands = _self_targets(ci, fn.attr)
        return (f"self.{fn.attr}", cands, True) if cands else None
    if isinstance(fn, ast.Name):
        a = mod.aliases.get(fn.id)
        if not a or a[0] != "obj":
            return None
        tgt, oname, label = _resolve_module(a[1]), a[2], fn.id
    elif isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
        a = mod.aliases.get(fn.value.id)
        if not a or a[0] != "mod":
            return None
        tgt, oname, label = _resolve_module(a[1]), fn.attr, f"{fn.value.id}.{fn.attr}"
    else:
        return None
    if tgt is None:
        return None
    sig = tgt.funcs.get(oname)
    # A declared orbit patch replaces the vendored def at import time, so the
    # replacement is the signature this call has to satisfy.
    sig = PATCHED.get((tgt.dotted, oname), sig)
    if not isinstance(sig, dict):
        return None
    if oname in tgt.mod_attr_stores or oname in mod.mod_attr_stores:
        return None  # possibly monkey-patched
    return (label, [sig], False)


def _check_file(f: str, tree: ast.Module) -> list[str]:
    mod = MODULES[_dotted(f)]
    errors, class_stack = [], []

    def visit(node):
        pushed = False
        if isinstance(node, ast.ClassDef):
            ci = getattr(node, "_ci", None)
            # A class nested in a function or another class carries no _ci and is skipped;
            # a module-level one only counts while it is still the binding for its name.
            ok = isinstance(ci, ClassInfo) and mod.classes.get(node.name) is ci
            class_stack.append(ci if ok else None)
            pushed = True
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not getattr(node, "_is_method", False):
            # A nested def taking its own `self` (a monkey-patch replacement, say) is not
            # bound to the enclosing class.
            if any(a.arg == "self" for a in node.args.posonlyargs + node.args.args) and class_stack:
                class_stack.append(None)
                pushed = True
        if isinstance(node, ast.Call):
            resolved = _resolve_call(mod, node, class_stack[-1] if class_stack else None)
            if resolved is None:
                STATS["skipped"] += 1
            else:
                label, cands, drop_self = resolved
                STATS["checked"] += 1
                verdicts = [_mismatch(node, c, drop_self) for c in cands]
                if (
                    verdicts
                    and all(v is not None for v in verdicts)
                    and (f, label, verdicts[0]) not in ALLOWLIST
                ):
                    errors.append(
                        f"{f}:{node.lineno}: {label}(...) -> {verdicts[0]}  [def {cands[0]['where']}]"
                    )
        for child in ast.iter_child_nodes(node):
            visit(child)
        if pushed:
            class_stack.pop()

    try:
        visit(tree)
    except RecursionError:
        return []  # pathologically nested source: say nothing rather than something wrong
    return errors


def collect_errors() -> list[str]:
    MODULES.clear()
    _UNRESOLVED.clear()
    _DESCENDANTS.clear()
    PATCHED.clear()
    STATS.update(checked=0, skipped=0, files=0)
    trees: dict[str, ast.Module] = {}
    for f in tracked:
        tree = _index_file(f)
        if tree is not None:
            trees[f] = tree
    _build_patch_overrides(trees)
    for f, tree in trees.items():
        _build_aliases(MODULES[_dotted(f)], tree)
    _build_descendants()
    errors = []
    for f, tree in trees.items():
        STATS["files"] += 1
        errors.extend(_check_file(f, tree))
    return errors


if __name__ == "__main__":
    all_errors = collect_errors()
    for e in all_errors:
        print(e)
    print(
        f"[verify-call-signatures] {'FAIL' if all_errors else 'ok'}: "
        f"{STATS['files']} files, {STATS['checked']} calls checked, {STATS['skipped']} skipped"
    )
    sys.exit(1 if all_errors else 0)
