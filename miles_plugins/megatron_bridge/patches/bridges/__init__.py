"""Orbit-owned reusable Megatron Bridge subclasses and conversion shims.

Importing this package installs orbit's ``megatron.bridge`` integration shims and
registers orbit's bridge subclasses (each submodule self-installs on import).
Every step is behind try/except so a failure in one does not break the others or
the import of orbit's megatron utils.
"""

import logging

logger = logging.getLogger(__name__)

__all__: list[str] = []


def _install_bridge_pp_group_unwrap() -> None:
    """Let ``MegatronParamMapping.broadcast_obj_from_pp_rank`` work with orbit's
    :class:`~miles.utils.reloadable_process_group.ReloadableProcessGroup`.

    ``broadcast_obj_from_pp_rank`` calls ``broadcast_object_list`` on
    ``self.pp_group``, which goes through ``_world.pg_group_ranks``. Orbit wraps
    every ``ProcessGroup`` in ``ReloadableProcessGroup`` for reload-safety; that
    wrapper is not in ``pg_group_ranks`` so ``get_group_rank`` raises
    ``"Group ... is not registered"``. Orbit's global ``monkey_patch_torch_dist``
    does not cover ``broadcast_object_list`` (it resolves ``get_group_rank`` from
    ``distributed_c10d``'s own globals), so this targeted shim is still needed for
    ``pp_size > 1``. Temporarily swap in the inner group for the broadcast.
    """
    from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping

    from miles.utils.reloadable_process_group import ReloadableProcessGroup

    if getattr(MegatronParamMapping, "_orbit_pp_group_unwrap_installed", False):
        return

    _orig = MegatronParamMapping.broadcast_obj_from_pp_rank

    def broadcast_obj_from_pp_rank(self, obj, name=None):
        if not isinstance(self.pp_group, ReloadableProcessGroup):
            return _orig(self, obj, name)
        saved = self.pp_group
        self.pp_group = saved.group
        try:
            return _orig(self, obj, name)
        finally:
            self.pp_group = saved

    MegatronParamMapping.broadcast_obj_from_pp_rank = broadcast_obj_from_pp_rank
    MegatronParamMapping._orbit_pp_group_unwrap_installed = True


try:
    _install_bridge_pp_group_unwrap()
except Exception as _e:  # best-effort
    logger.warning("orbit bridge shim _install_bridge_pp_group_unwrap not applied: %s", _e)

try:
    from . import nemotron_h  # noqa: F401
except Exception as _e:  # pragma: no cover - defensive
    logger.warning("orbit nemotron_h bridge plugin failed to load: %s", _e)
