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
import inspect
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


def apply_all(*, reapply: bool = False) -> int:
    """Verify every pin, then install every replacement. Returns the count.

    Idempotent: applying twice is a no-op unless ``reapply`` is set, so an
    entrypoint may call it defensively (ray starts each worker in a fresh
    process, and each one must patch before it touches miles).
    """
    applied = 0
    for patch in _REGISTRY:
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
        _APPLIED.add(patch.target)
        applied += 1
    return applied
