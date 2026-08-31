"""Run an orbit side effect the moment a vendored module is first imported.

orbit/patch/runtime.py moves orbit's change to a vendored FUNCTION out of the
miles tree. Not every orbit change is a function: some vendored modules carried a
bare module-level ``import X`` whose only purpose was X's import side effect --
installing a shim, registering a subclass, widening a class. There is no body to
swap, so ``patch_function`` cannot express it, and the import line left in place
is precisely what keeps that vendored file dirty.

This is runtime.py's deferred mechanism one level up: register a callback against
a module NAME and it runs immediately after that module is first executed, which
is the exact moment the deleted ``import`` line used to run at. Registration is
by string and the callbacks do their own imports lazily, so ``import orbit``
stays as cheap as runtime.py's docstring demands -- no torch, no vendored
backend, no megatron.

Three properties, two shared with the function patches and one not:

* It fires only in a process that ran ``import orbit`` (which is what arms it).
  An entrypoint reaching the vendored module without importing orbit gets
  upstream's behaviour with no error -- the same trap runtime.py documents, and
  the reason tests/fast/test_hf_export_patches.py checks entrypoints.
* A module already in ``sys.modules`` when the callback is registered has
  already missed the hook, so it fires immediately instead. Import ORDER must
  not decide whether the seam takes effect.
* There is no hash pin, because there is nothing upstream to drift from: the
  callback ADDS behaviour next to the vendored module rather than replacing any
  of it. What can rot is the module NAME, so a callback registered against a
  module that no longer exists is a silent no-op -- ``check_seam_targets`` makes
  that loud, and tests/fast/test_miles_utils_patches.py runs it.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from typing import Callable


# (module name, callback). A list rather than a dict: two independent seams may
# legitimately hang off the same vendored module, and dropping one silently is
# exactly the failure this layer exists to avoid.
_REGISTRY: list[tuple[str, Callable[[], object]]] = []
_FIRED: set[tuple[str, int]] = set()


def registry() -> list[tuple[str, Callable[[], object]]]:
    return list(_REGISTRY)


def fired() -> set[tuple[str, int]]:
    return set(_FIRED)


def on_import(
    module: str, callback: Callable[[], object], relevant_names: tuple[str, ...] = ()
) -> None:
    """Run ``callback()`` right after ``module`` is first imported.

    Idempotent per (module, callback): the vendored ``import`` line this
    replaces ran once, because the module it lived in was cached after its first
    execution. Re-importing the target does not re-run the callback.

    ``relevant_names`` narrows who has to be armed for the seam. A function patch
    changes what an existing call returns, so tools/check_arming.py can ask which
    processes call that name; a seam has no such name, and asking instead "who
    imports the module" over-approximates badly -- miles' ~39 launcher scripts
    import ``miles.backends.megatron_utils`` to build a command line and never
    touch a bridge. Naming what the seam actually installs (e.g. the classes
    ``AutoBridge`` resolves) gives the guard the same precision it has for
    patches. Left empty, module reachability is used, which is conservative.

    Stashed on the callback rather than widening the registry tuple, so
    ``registry()`` keeps its shape for every existing reader.
    """
    if relevant_names:
        callback.orbit_relevant_names = tuple(relevant_names)
    _REGISTRY.append((module, callback))
    _arm()
    if module in sys.modules:
        # Registered too late to be hooked -- fire now, as install_hook() does.
        _run(module)


def _run(module: str) -> None:
    for name, callback in _REGISTRY:
        if name != module:
            continue
        key = (name, id(callback))
        if key in _FIRED:
            continue
        _FIRED.add(key)
        callback()


class _RunOnImport:
    """Sits on sys.meta_path and does nothing until a registered module loads.

    Deliberately mirrors orbit.patch.runtime._ApplyOnImport rather than
    importing the targets eagerly: importing them here would drag torch and the
    whole vendored backend into ``import orbit``, which is measured to more than
    double the fast suite.
    """

    def __init__(self) -> None:
        self._loading: set[str] = set()

    def find_spec(self, fullname, path=None, target=None):
        if fullname in self._loading:
            return None  # re-entrancy: our own find_spec below
        if not any(name == fullname for name, _ in _REGISTRY):
            return None
        self._loading.add(fullname)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._loading.discard(fullname)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _RunningLoader(spec.loader, fullname)
        return spec


class _RunningLoader:
    def __init__(self, inner, fullname: str) -> None:
        self._inner = inner
        self._fullname = fullname

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        _run(self._fullname)

    def __getattr__(self, item):
        return getattr(self._inner, item)


_HOOK: _RunOnImport | None = None


def _arm() -> None:
    global _HOOK
    if _HOOK is None:
        _HOOK = _RunOnImport()
        sys.meta_path.insert(0, _HOOK)


def check_seam_targets() -> list[str]:
    """Registered module names that no longer resolve to a real module.

    A callback hung off a module that upstream renamed or deleted never runs and
    never complains, so the seam it carries just stops happening. Resolution is
    by spec lookup, without importing, so this is safe in the CPU gate.
    """
    missing = []
    for name, callback in _REGISTRY:
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, AttributeError, ValueError):
            spec = None
        if spec is None:
            missing.append(
                f"{name}: orbit runs {getattr(callback, '__name__', callback)!r} "
                f"when this module is imported, but no such module exists. Its "
                f"side effect is silently not happening; re-point or retire the "
                f"seam."
            )
    return missing
