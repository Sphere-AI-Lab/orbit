"""Hash-pinned function patching.

A patch declares the upstream function it replaces AND a hash of that function's
source. ``apply_all()`` verifies the hash before swapping, so the two ways a
monkey-patch normally rots both become loud:

* the target moved or was renamed  -> AttributeError naming the patch
* the target's body changed        -> UpstreamDrift naming the patch

Either way the process stops instead of silently running upstream's code (or
orbit's stale replacement of it). Re-pinning is the deliberate act that records
"a human compared the new upstream body against orbit's replacement".

Hashing note: the hash covers the function's source from its ``def`` line
onward, with blank lines dropped and trailing whitespace stripped. Decorators
are EXCLUDED, because ``inspect.getsource`` includes them while
``ast.get_source_segment`` does not, and the static CI gate must agree with the
runtime check byte for byte.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import inspect
import sys
from dataclasses import dataclass, field
from typing import Any, Callable


class UpstreamDrift(RuntimeError):
    """An upstream function orbit patches is not the one the patch was written against."""


def normalize(source: str) -> str:
    """Body text a hash is taken over: from `def`/`async def` onward, no blank
    lines, no trailing whitespace, no leading indentation drift."""
    lines = source.splitlines()
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(("def ", "async def ")):
            lines = lines[i:]
            break
    return "\n".join(line.rstrip() for line in lines if line.strip())


def source_sha(obj: Any) -> str:
    return hashlib.sha256(
        normalize(inspect.getsource(obj)).encode("utf-8", "surrogateescape")
    ).hexdigest()


@dataclass(frozen=True)
class Patch:
    module: str
    attr: str
    upstream_sha: str
    reason: str
    replacement: Callable = field(compare=False)

    @property
    def target(self) -> str:
        return f"{self.module}.{self.attr}"


_REGISTRY: list[Patch] = []
_APPLIED: set[str] = set()


def registry() -> list[Patch]:
    return list(_REGISTRY)


def original(module: str, attr: str) -> Callable:
    """The pristine upstream function a patch replaced.

    A replacement that DELEGATES -- handling only its own cases and letting
    upstream's body do the rest -- is strongly preferred over one that copies
    upstream's logic: it stays small, and upstream's future fixes keep flowing
    through instead of being shadowed by a stale copy.
    """
    mod = importlib.import_module(module)
    saved = getattr(mod, f"_orbit_unpatched_{attr}", None)
    if saved is None:
        raise UpstreamDrift(
            f"{module}.{attr}: no saved original -- orbit.patch.apply_all() has "
            f"not run in this process, so the replacement is executing without "
            f"the function it delegates to"
        )
    return saved


def patch_function(module: str, attr: str, *, upstream_sha: str, reason: str):
    """Declare that the decorated function replaces ``module.attr``.

    ``reason`` is not decoration: it is what a reader needs when the pin fails
    and they must decide whether upstream's new body changes orbit's intent.
    """

    def decorate(fn: Callable) -> Callable:
        _REGISTRY.append(
            Patch(
                module=module,
                attr=attr,
                upstream_sha=upstream_sha,
                reason=reason,
                replacement=fn,
            )
        )
        return fn

    return decorate


def _repoint_reexports(patch: "Patch", original_fn) -> None:
    """Re-point already-imported re-exports of a patched function.

    Patching `module.attr` is not enough when a package did
    `from .submodule import attr` at import time: that bound the ORIGINAL object
    into the package namespace, and callers dispatching through the package keep
    getting it. The import hook normally avoids this by patching before anything
    imports the target -- but a target already in sys.modules when the hook is
    armed has re-exports that are already stale.

    That is not theoretical. `miles.backends.megatron_utils.megatron_to_hf`
    re-exports every converter and `_convert_to_hf_core` dispatches through the
    PACKAGE binding, so a Ray actor -- which imports the converter package while
    unpickling its worker class, before the actor body runs `import orbit` --
    would call upstream's unpatched function despite the patch being installed.

    Only an attribute that is still identical to the object just replaced is
    re-pointed, so an unrelated same-named attribute is never clobbered.
    """
    parts = patch.module.split(".")
    for depth in range(len(parts) - 1, 0, -1):
        pkg = sys.modules.get(".".join(parts[:depth]))
        if pkg is not None and getattr(pkg, patch.attr, None) is original_fn:
            setattr(pkg, f"_orbit_unpatched_{patch.attr}", original_fn)
            setattr(pkg, patch.attr, patch.replacement)


class _ApplyOnImport:
    """Apply each patch when its target module is first imported -- never sooner.

    Applying eagerly (importing every target at ``import orbit`` time) is the
    obvious design and it is wrong: it makes ``import orbit`` drag in torch and
    the whole vendored backend, which breaks lightweight tooling that only
    wanted a constant and slows every test that touches orbit. Measured: the
    fast suite went 5m20s -> 14m07s, and a helper spawning
    ``python -c "from tools.lora_regret.arms import ..."`` started failing.

    So instead this finder sits on ``sys.meta_path`` and does nothing until
    something actually imports a patched module; it then lets the normal
    machinery load it and applies the patch immediately afterwards. ``import
    orbit`` stays as cheap as it was.
    """

    def __init__(self) -> None:
        self._loading: set[str] = set()

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self._loading:
            return None  # re-entrancy: our own find_spec below
        if not any(p.module == fullname for p in _REGISTRY):
            return None
        self._loading.add(fullname)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._loading.discard(fullname)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _PatchingLoader(spec.loader, fullname)
        return spec


class _PatchingLoader:
    def __init__(self, inner, fullname: str) -> None:
        self._inner = inner
        self._fullname = fullname

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        apply_all(only=self._fullname)

    def __getattr__(self, item):
        return getattr(self._inner, item)


_HOOK: _ApplyOnImport | None = None


def install_hook() -> None:
    """Arm deferred patching. Idempotent; safe to call from ``orbit/__init__``."""
    global _HOOK
    if _HOOK is None:
        _HOOK = _ApplyOnImport()
        sys.meta_path.insert(0, _HOOK)
    # Anything already imported missed the hook, so patch it now.
    for patch in _REGISTRY:
        if patch.module in sys.modules:
            apply_all(only=patch.module)


def apply_all(*, reapply: bool = False, only: str | None = None) -> int:
    """Verify every pin, then install every replacement. Returns the count.

    Idempotent: applying twice is a no-op unless ``reapply`` is set, so an
    entrypoint may call it defensively (ray starts each worker in a fresh
    process, and each one must patch before it touches miles).
    """
    applied = 0
    for patch in _REGISTRY:
        if only is not None and patch.module != only:
            continue
        if patch.target in _APPLIED and not reapply:
            continue
        module = importlib.import_module(patch.module)
        try:
            current = getattr(module, patch.attr)
        except AttributeError as exc:
            raise UpstreamDrift(
                f"{patch.target}: orbit patches this function but it no longer "
                f"exists upstream. Orbit's replacement was written because: "
                f"{patch.reason}. Find where the behaviour moved, then re-point "
                f"or retire the patch."
            ) from exc
        # A previously-applied patch would hash as orbit's own body; compare
        # against the pristine upstream object when re-applying.
        if patch.target in _APPLIED:
            current = getattr(module, f"_orbit_unpatched_{patch.attr}", current)
        actual = source_sha(current)
        if actual != patch.upstream_sha:
            raise UpstreamDrift(
                f"{patch.target}: upstream's body changed (pinned "
                f"{patch.upstream_sha[:12]}, found {actual[:12]}). Orbit replaces "
                f"it because: {patch.reason}. Review whether the new upstream "
                f"body changes that reasoning, then re-pin with "
                f"tools/check_patch_pins.py --write."
            )
        setattr(module, f"_orbit_unpatched_{patch.attr}", current)
        setattr(module, patch.attr, patch.replacement)
        _repoint_reexports(patch, current)
        _APPLIED.add(patch.target)
        applied += 1
    return applied
