"""Resolve the OFT adapter exporter across the two Megatron-Bridge API shapes.

The Bridge revision this tree pins (Sphere-AI-Lab/Megatron-Bridge 988d6426,
"remove legacy import compatibility shims") exposes the exporter as a method,
``AutoBridge.export_oft_adapter_weights(model, cpu=..., show_progress=...)``.
orbit-main's Bridge lineage (bb937216) moved it out to the free function
``megatron.bridge.orbit.conversion.oft_export.export_oft_adapter_weights(auto_bridge,
model, ...)``. The weight-sync iterator and the PEFT checkpoint writer were ported
from orbit-main with the free-function shape, which does not exist on the pinned
Bridge -- the first OFT weight sync then dies with an ImportError.

``oft_adapter_exporter(bridge)`` returns ``exporter(model, *, cpu, show_progress)``
bound to whichever shape the installed Bridge provides, free function first (the
unit tests stub that module), so the same call sites run against either pin.
Kept free of heavy imports: it is called from inside the weight-sync path.
"""

from __future__ import annotations

import functools
import importlib
from collections.abc import Callable
from typing import Any

OFT_EXPORT_MODULE = "megatron.bridge.orbit.conversion.oft_export"
EXPORTER_NAME = "export_oft_adapter_weights"


def _free_function() -> Callable[..., Any] | None:
    try:
        module = importlib.import_module(OFT_EXPORT_MODULE)
    except ImportError:
        return None
    return getattr(module, EXPORTER_NAME, None)


def oft_adapter_exporter(bridge: Any) -> Callable[..., Any]:
    """Return the OFT adapter exporter for ``bridge`` as ``exporter(model, *, cpu, show_progress)``."""
    free_function = _free_function()
    if free_function is not None:
        return functools.partial(free_function, bridge)
    method = getattr(bridge, EXPORTER_NAME, None)
    if method is None:
        raise ImportError(
            f"No OFT adapter exporter available: neither {OFT_EXPORT_MODULE}.{EXPORTER_NAME} "
            "(orbit-main Bridge lineage) nor AutoBridge.export_oft_adapter_weights (pinned Bridge "
            "988d6426) is present -- check the Megatron-Bridge checkout against pyproject's "
            "tool.orbit.release.backend-pins."
        )
    return method
