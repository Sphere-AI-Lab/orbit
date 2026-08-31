"""Orbit's addition to ``miles.utils.misc``, moved out of the vendored file.

``get_free_port`` gained a second condition in its scan loop. That is a
DELEGATING patch: orbit re-runs upstream's scan and only adds the cross-process
lock, so upstream keeps owning what "available" means.

Orbit's other ``miles.utils`` change is NOT here, and its absence is the
mechanism working rather than an omission: ``reloadable_process_group.py``'s
``_forward_remaining_collectives`` went upstream. miles @ dbbab1566 defines it
and calls it at module scope itself, so orbit's copy would be a stale duplicate
of code that already runs. The vendored file keeps upstream's version and orbit
carries nothing. (orbit-main-isolated, which sits on the older miles base, still
has to lift it.)

Nothing here imports torch or miles at module scope -- every dependency is
imported inside the function that needs it -- because ``import orbit`` executes
this module and must stay cheap (see orbit/patch/runtime.py).
"""

from __future__ import annotations

from orbit.patch import original, patch_function


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
