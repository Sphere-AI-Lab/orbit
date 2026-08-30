"""Orbit's hash-pinned patch layer.

Lets orbit override a vendored miles function WITHOUT editing the miles file, so
the vendored tree stays byte-pristine and every orbit line lives under orbit/.

The reason plain monkey-patching was rejected until now is that it fails
SILENTLY: upstream renames or rewrites the target, the patch stops matching what
it was written against, and nothing errors -- the run just quietly uses different
code. This layer removes that failure mode by pinning a hash of the upstream
function each patch replaces and verifying it before the swap, so a moved target
aborts loudly at startup instead of drifting.

See orbit/patch/runtime.py for the mechanism and tools/check_patch_pins.py for
the CI gate that verifies every pin statically, without importing torch.
"""

from orbit.patch.runtime import (
    UpstreamDrift,
    apply_all,
    patch_function,
    registry,
    source_sha,
)

__all__ = [
    "UpstreamDrift",
    "apply_all",
    "patch_function",
    "registry",
    "source_sha",
]
