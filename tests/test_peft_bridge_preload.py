from types import SimpleNamespace

import pytest

import miles.orbit.audit.peft_wrap as peft_audit
import miles.orbit.megatron.bridge_peft_helpers as bridge_peft_helpers
from miles.orbit.megatron.bridge_peft_helpers import (
    _make_peft_pre_wrap_hook,
    _propagate_preloaded_checkpoint_identity,
)
from miles.orbit.megatron.low_precision_bootstrap import _dist_checkpoint_already_loaded


def test_peft_replacement_model_inherits_preloaded_checkpoint_identity():
    source = SimpleNamespace(
        _orbit_loaded_dist_checkpoint_path="/checkpoint/release",
        _orbit_loaded_dist_checkpoint_prefix="",
        _orbit_restored_modelopt_checkpoint_path="/checkpoint/release",
    )
    transformed = SimpleNamespace()

    _propagate_preloaded_checkpoint_identity([source], [transformed])

    assert transformed._orbit_loaded_dist_checkpoint_path == "/checkpoint/release"
    assert transformed._orbit_loaded_dist_checkpoint_prefix == ""
    assert transformed._orbit_restored_modelopt_checkpoint_path == "/checkpoint/release"


def test_peft_preload_identity_rejects_changed_model_chunk_count():
    with pytest.raises(RuntimeError, match="changed the number of model chunks"):
        _propagate_preloaded_checkpoint_identity([SimpleNamespace()], [])


def test_peft_pre_wrap_identity_reaches_final_replacement_chunks(monkeypatch):
    sources = [SimpleNamespace(), SimpleNamespace()]
    peft_replacements = [SimpleNamespace(), SimpleNamespace()]
    final_replacements = [SimpleNamespace(), SimpleNamespace()]

    class _ReplacementPeft:
        def __call__(self, _model, *, training):
            assert training is True
            return peft_replacements

        def set_params_to_save(self, model):
            assert model == final_replacements

    def mark_preloaded(model, load_path, *, is_value_model):
        assert load_path == "/checkpoint"
        assert is_value_model is False
        for chunk in model:
            chunk._orbit_loaded_dist_checkpoint_path = "/checkpoint/release"
            chunk._orbit_loaded_dist_checkpoint_prefix = ""

    monkeypatch.setattr(bridge_peft_helpers, "is_distributed_checkpoint", lambda _path: True)
    monkeypatch.setattr(bridge_peft_helpers, "load_dist_checkpoint", mark_preloaded)
    monkeypatch.setattr(bridge_peft_helpers, "_assert_peft_wrapped_modules", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(bridge_peft_helpers, "_materialize_runtime_device", lambda _model: None)
    monkeypatch.setattr(peft_audit, "dump_megatron_audit", lambda _model: None)

    hook = _make_peft_pre_wrap_hook(
        _ReplacementPeft(),
        load_path="/checkpoint",
        is_value_model=False,
        peft_method="lora",
        post_peft_hooks=[lambda _model: final_replacements],
    )
    transformed = hook(sources)

    assert transformed == final_replacements
    assert _dist_checkpoint_already_loaded(transformed, "/checkpoint/release")
