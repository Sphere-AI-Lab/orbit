"""Deferred import-time side effects for the megatron backend.

These two hooks used to run in this package's ``__init__``, which made ANY
``miles.backends.megatron_utils.*`` import (including the featherweight
``ft.types`` enum that the audit event models need) pay the full
megatron.bridge + transformers import: ~75s per process on this cluster's
WekaFS. They are only needed by processes that actually run the megatron
trainer, so they now install explicitly from ``initialize.init()``.

Fork note (2026-08 sync): upstream carries the same side effects in
``__init__``; this split is an upstream candidate.
"""

import logging

logger = logging.getLogger(__name__)

_installed = False


def install_runtime_hooks() -> None:
    """Patch deep_ep for torch_memory_saver and register bridge plugins.

    Idempotent. Must run before any ``deep_ep.Buffer`` is constructed and
    before bridge model providers are resolved — ``initialize.init()``
    precedes both in every trainer entrypoint.
    """
    global _installed
    if _installed:
        return
    _installed = True

    import torch

    try:
        import deep_ep
        from torch_memory_saver import torch_memory_saver

        old_init = deep_ep.Buffer.__init__

        def new_init(self, *args, **kwargs):
            if torch_memory_saver._impl is not None:
                torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(False)
            old_init(self, *args, **kwargs)
            torch.cuda.synchronize()
            if torch_memory_saver._impl is not None:
                torch_memory_saver._impl._binary_wrapper.cdll.tms_set_interesting_region(True)

        deep_ep.Buffer.__init__ = new_init
    except ImportError:
        logger.warning("deep_ep is not installed, some functionalities may be limited.")

    try:
        import miles_plugins.megatron_bridge  # noqa: F401
    except Exception as _e:  # best-effort; not every environment uses megatron.bridge
        logger.warning("miles megatron.bridge plugins failed to load: %s", _e)
