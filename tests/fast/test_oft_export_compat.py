"""The OFT exporter resolver binds whichever Megatron-Bridge shape is installed.

Pinned Bridge 988d6426: ``AutoBridge.export_oft_adapter_weights`` method only.
orbit-main's Bridge lineage: free function in
``megatron.bridge.orbit.conversion.oft_export`` taking the bridge first.
"""

import sys
from types import ModuleType

import pytest

from orbit.megatron import oft_export_compat


class _Bridge:
    def __init__(self):
        self.calls = []

    def export_oft_adapter_weights(self, model, cpu=False, show_progress=True):
        self.calls.append(("method", model, cpu, show_progress))
        return ("method-result",)


def _stub_oft_export_module(monkeypatch, *, with_free_function: bool):
    module = ModuleType(oft_export_compat.OFT_EXPORT_MODULE)
    if with_free_function:
        def export_oft_adapter_weights(auto_bridge, model, cpu=False, show_progress=True):
            auto_bridge.calls.append(("free", model, cpu, show_progress))
            return ("free-result",)

        module.export_oft_adapter_weights = export_oft_adapter_weights
    monkeypatch.setitem(sys.modules, oft_export_compat.OFT_EXPORT_MODULE, module)
    return module


def test_pinned_bridge_shape_uses_the_bound_method(monkeypatch):
    _stub_oft_export_module(monkeypatch, with_free_function=False)
    bridge = _Bridge()

    exporter = oft_export_compat.oft_adapter_exporter(bridge)

    assert exporter("model", cpu=False, show_progress=False) == ("method-result",)
    assert bridge.calls == [("method", "model", False, False)]


def test_orbit_main_bridge_shape_prefers_the_free_function(monkeypatch):
    _stub_oft_export_module(monkeypatch, with_free_function=True)
    bridge = _Bridge()

    exporter = oft_export_compat.oft_adapter_exporter(bridge)

    assert exporter("model", cpu=True, show_progress=False) == ("free-result",)
    assert bridge.calls == [("free", "model", True, False)]


def test_missing_module_falls_back_to_the_method(monkeypatch):
    monkeypatch.setitem(sys.modules, oft_export_compat.OFT_EXPORT_MODULE, None)  # import raises ImportError
    bridge = _Bridge()

    exporter = oft_export_compat.oft_adapter_exporter(bridge)

    assert exporter("model") == ("method-result",)


def test_neither_shape_is_a_loud_import_error(monkeypatch):
    _stub_oft_export_module(monkeypatch, with_free_function=False)

    with pytest.raises(ImportError, match="No OFT adapter exporter available"):
        oft_export_compat.oft_adapter_exporter(object())
