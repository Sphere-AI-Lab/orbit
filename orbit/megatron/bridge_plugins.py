"""Load orbit's megatron.bridge plugin patches when the Megatron backend loads.

``miles/backends/megatron_utils/__init__.py`` carried a best-effort
``import miles_plugins.megatron_bridge.patches.bridges`` -- an import whose only
purpose is the side effect of importing it: the package installs orbit's
``MegatronParamMapping.broadcast_obj_from_pp_rank`` unwrap (megatron.bridge's pp
broadcast cannot see orbit's ReloadableProcessGroup wrapper) and registers
orbit's bridge subclasses.

There is no function to patch there, so the seam is an import-time one: the
callback below runs the instant ``miles.backends.megatron_utils`` is first
imported, which is exactly when the deleted line ran. The vendored file is
pristine again and the timing is unchanged.

Why the whole thing stays best-effort, as it was in the vendored file: not every
environment has megatron.bridge, and the backend must still import where it does
not. A failure is logged, never raised.

Note this fires at BACKEND import, earlier than base miles' own hook -- base
imports the (empty) ``miles_plugins.megatron_bridge`` plugin package at each
bridge call site instead. Orbit needs the shims installed before the first
``AutoBridge`` call in the actor, and the actor reaches those call sites through
paths base's two hooks do not cover.
"""

from __future__ import annotations

import logging

from orbit.patch.on_import import on_import


logger = logging.getLogger(__name__)

MEGATRON_BACKEND = "miles.backends.megatron_utils"


def load_bridge_plugins() -> bool:
    """Import orbit's bridge plugin package for its side effects. Never raises."""
    try:
        import miles_plugins.megatron_bridge.patches.bridges  # noqa: F401
    except Exception as exc:  # best-effort; not every environment uses megatron.bridge
        logger.warning("orbit megatron.bridge plugins failed to load: %s", exc)
        return False
    return True


# Only processes that actually resolve a bridge need these registrations; the
# launcher scripts import this backend package to build a command line and never
# construct one. See on_import's docstring for why the guard needs telling.
on_import(MEGATRON_BACKEND, load_bridge_plugins, relevant_names=("AutoBridge",))
