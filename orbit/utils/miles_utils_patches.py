"""Orbit's two additions to ``miles.utils``, moved out of the vendored files.

Both used to be edits inside miles/, and both are now expressed from orbit's
side so those two files are byte-pristine again:

* ``miles/utils/reloadable_process_group.py`` gained a whole function that does
  not exist upstream, plus the one line that called it at module import. The
  function is LIFTED here verbatim; the call is now an import-time seam
  (orbit.patch.on_import) that fires at exactly the same moment the deleted
  module-level call did.
* ``miles/utils/misc.py``'s ``get_free_port`` gained a second condition in its
  scan loop. That is a DELEGATING patch: orbit re-runs upstream's scan and only
  adds the cross-process lock, so upstream keeps owning what "available" means.

Nothing here imports torch or miles at module scope -- every dependency is
imported inside the function that needs it -- because ``import orbit`` executes
this module and must stay cheap (see orbit/patch/runtime.py).
"""

from __future__ import annotations

from orbit.patch import original, patch_function
from orbit.patch.on_import import on_import


_RPG = "miles.utils.reloadable_process_group"

# Attributes of torch's ProcessGroup that are NOT collectives to forward:
# identity/lifecycle accessors ReloadableProcessGroup answers itself, and the
# property it already defines.
_NOT_A_COLLECTIVE = frozenset(
    {"rank", "size", "name", "abort", "shutdown", "bound_device_id"}
)


def forward_remaining_collectives() -> list[str]:
    """Forward every ProcessGroup collective ReloadableProcessGroup lacks.

    Callers that resolved a collective before the monkey patch went on --
    Megatron binds `dist_reduce_scatter_func` at import time -- hand the wrapper
    straight to torch, which invokes the method on the group object. Anything
    not overridden reaches the C++ base, whose own backend map is empty, and
    dies as "No backend type associated with device type cuda". A hand-written
    forward list silently regrows that hole whenever torch renames or adds a
    collective, which is how torch 2.13's *_single family got through.

    Returns the names it installed, so a test can prove the seam fired rather
    than trusting that it did.
    """
    import torch.distributed as dist

    from miles.utils.reloadable_process_group import ReloadableProcessGroup

    installed = []
    for name in dir(dist.ProcessGroup):
        if name.startswith("__") or name in _NOT_A_COLLECTIVE:
            continue
        if name in vars(ReloadableProcessGroup):
            continue
        if not callable(getattr(dist.ProcessGroup, name, None)):
            continue

        def make(method):
            def forward(self, *args, **kwargs):
                return self._fwd(method, *args, **kwargs)

            forward.__name__ = method
            return forward

        setattr(ReloadableProcessGroup, name, make(name))
        installed.append(name)
    return installed


# The vendored module ran this at the bottom of its own body; run it at the same
# point from out here. Re-running is harmless -- an already-installed forward is
# in vars(ReloadableProcessGroup) and gets skipped -- but on_import fires once.
on_import(_RPG, forward_remaining_collectives)


_FREE_PORT_REASON = (
    "is_port_available only observes a port, it does not claim one, so two "
    "concurrent launches on the same host both saw the same range free and "
    "both took it; orbit flocks the range before returning it"
)


@patch_function(
    "miles.utils.misc",
    "get_free_port",
    upstream_sha="0898e029010c13cade9bf190c3dbc18d59c9d3f78f3bfa7b36213c944b3eab6b",
    reason=_FREE_PORT_REASON,
)
def get_free_port(start_port=10000, consecutive=1):
    """Upstream's scan, then orbit's cross-process claim on what it found.

    Delegation rather than a copied body, even though orbit owned half the lines
    of the original edit: upstream's loop is "advance until `consecutive` ports
    look free", which composes exactly with "...and we won the lock on them,
    otherwise keep advancing". Re-entering upstream's scan one port further on a
    lost lock is the same sequence the inlined `and` produced, and it leaves
    upstream owning the definition of "available".
    """
    from miles.utils.http_utils import _try_lock_port_range

    scan = original("miles.utils.misc", "get_free_port")
    port = start_port
    while True:
        port = scan(port, consecutive)
        if _try_lock_port_range(port, consecutive):
            return port
        port += 1
