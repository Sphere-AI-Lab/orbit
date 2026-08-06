from types import SimpleNamespace

import pytest

from orbit.backends.megatron_utils.bridge_peft_helpers import _propagate_preloaded_checkpoint_identity


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
