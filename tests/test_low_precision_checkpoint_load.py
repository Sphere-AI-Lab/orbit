import sys
from types import ModuleType, SimpleNamespace

import pytest

import orbit.backends.megatron_utils.low_precision_bootstrap as low_precision_bootstrap


class _FakeDistCheckpointing:
    def __init__(self):
        self.load_calls = []

    @staticmethod
    def load_tensors_metadata(_path):
        return {}

    def load(self, sharded_state_dict, path, **kwargs):
        self.load_calls.append((sharded_state_dict, path, kwargs))
        return sharded_state_dict


def _install_fake_module(monkeypatch, name, module):
    parts = name.split(".")
    for index in range(1, len(parts)):
        package_name = ".".join(parts[:index])
        if package_name not in sys.modules:
            package = ModuleType(package_name)
            package.__path__ = []
            monkeypatch.setitem(sys.modules, package_name, package)
        if index > 1:
            parent_name = ".".join(parts[: index - 1])
            setattr(sys.modules[parent_name], parts[index - 1], sys.modules[package_name])
    monkeypatch.setitem(sys.modules, name, module)
    setattr(sys.modules[".".join(parts[:-1])], parts[-1], module)


def _patch_dist_checkpoint_loader(monkeypatch, tmp_path, model, *, int4=False):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / ".metadata").touch()

    fake_dist_checkpointing = _FakeDistCheckpointing()
    monkeypatch.setattr(low_precision_bootstrap, "_unwrap_parallel_model", lambda _model: [model])
    monkeypatch.setattr(low_precision_bootstrap, "_restore_modelopt_state_before_load", lambda *_args: None)
    monkeypatch.setattr(low_precision_bootstrap, "_import_dist_checkpointing", lambda: fake_dist_checkpointing)
    monkeypatch.setattr(low_precision_bootstrap, "_get_assume_ok_unexpected_strict_handling", lambda: None)
    monkeypatch.setattr(low_precision_bootstrap, "detect_int4_checkpoint", lambda _path: (int4, False))
    monkeypatch.setattr(low_precision_bootstrap, "detect_nvfp4_checkpoint", lambda _path: False)
    monkeypatch.setattr(low_precision_bootstrap, "detect_fp8_checkpoint", lambda _path: False)
    monkeypatch.setattr(low_precision_bootstrap, "_materialize_meta_sharded_tensor_data", lambda _state: None)
    monkeypatch.setattr(
        low_precision_bootstrap, "_materialize_loaded_meta_module_tensors", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(low_precision_bootstrap, "_zero_oft_adapter_parameters", lambda *_args: None)
    monkeypatch.setattr(low_precision_bootstrap, "_raise_if_meta_tensors_remain", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(low_precision_bootstrap, "_mark_dist_checkpoint_as_loaded", lambda *_args: None)
    return checkpoint_dir, fake_dist_checkpointing


def test_int4_checkpoint_load_forwards_process_groups_to_swiglu_factory(monkeypatch, tmp_path):
    calls = []
    tp_group = object()
    dp_group = object()

    def apply_swiglu(original_sh_ten, sharded_offsets, singleton_local_shards=False, *, tp_group=None, dp_group=None):
        calls.append((original_sh_ten, sharded_offsets, singleton_local_shards, tp_group, dp_group))
        return original_sh_ten

    experts = ModuleType("megatron.core.transformer.moe.experts")
    experts.apply_swiglu_sharded_factory = apply_swiglu
    _install_fake_module(monkeypatch, "megatron.core.transformer.moe.experts", experts)

    int4_utils = ModuleType("megatron.bridge.orbit.quant.int4_utils")
    int4_utils.register_int4_buffers_after_load = lambda *_args: None
    int4_utils.transform_sharded_state_dict_for_int4 = lambda state: state
    _install_fake_module(monkeypatch, "megatron.bridge.orbit.quant.int4_utils", int4_utils)

    class _Model:
        def sharded_state_dict(self, **_kwargs):
            tensor = SimpleNamespace(local_shape=(2, 2))
            experts.apply_swiglu_sharded_factory(
                tensor,
                ((0, 0, 1),),
                tp_group=tp_group,
                dp_group=dp_group,
            )
            return {}

    model = _Model()
    checkpoint_dir, _ = _patch_dist_checkpoint_loader(monkeypatch, tmp_path, model, int4=True)
    monkeypatch.setattr(low_precision_bootstrap.torch.distributed, "is_initialized", lambda: False)

    low_precision_bootstrap.load_dist_checkpoint(model, str(checkpoint_dir))

    assert len(calls) == 1
    _, sharded_offsets, singleton_local_shards, received_tp_group, received_dp_group = calls[0]
    assert sharded_offsets == ((0, 0, 1),)
    assert singleton_local_shards is False
    assert received_tp_group is tp_group
    assert received_dp_group is dp_group


def test_int4_checkpoint_load_supports_pinned_swiglu_factory_signature(monkeypatch, tmp_path):
    calls = []

    def apply_swiglu(original_sh_ten, sharded_offsets, singleton_local_shards=False):
        calls.append((original_sh_ten, sharded_offsets, singleton_local_shards))
        return original_sh_ten

    experts = ModuleType("megatron.core.transformer.moe.experts")
    experts.apply_swiglu_sharded_factory = apply_swiglu
    _install_fake_module(monkeypatch, "megatron.core.transformer.moe.experts", experts)

    int4_utils = ModuleType("megatron.bridge.orbit.quant.int4_utils")
    int4_utils.register_int4_buffers_after_load = lambda *_args: None
    int4_utils.transform_sharded_state_dict_for_int4 = lambda state: state
    _install_fake_module(monkeypatch, "megatron.bridge.orbit.quant.int4_utils", int4_utils)

    class _Model:
        def sharded_state_dict(self, **_kwargs):
            tensor = SimpleNamespace(local_shape=(2, 2))
            experts.apply_swiglu_sharded_factory(tensor, ((0, 0, 1),))
            return {}

    model = _Model()
    checkpoint_dir, _ = _patch_dist_checkpoint_loader(monkeypatch, tmp_path, model, int4=True)
    monkeypatch.setattr(low_precision_bootstrap.torch.distributed, "is_initialized", lambda: False)

    low_precision_bootstrap.load_dist_checkpoint(model, str(checkpoint_dir))

    assert len(calls) == 1
    _, sharded_offsets, singleton_local_shards = calls[0]
    assert sharded_offsets == ((0, 0, 1),)
    assert singleton_local_shards is False


@pytest.mark.parametrize(("world_size", "expected"), [(1, False), (2, True)])
def test_checkpoint_load_validates_shard_access_only_across_ranks(monkeypatch, tmp_path, world_size, expected):
    model = SimpleNamespace(sharded_state_dict=lambda **_kwargs: {})
    checkpoint_dir, fake_dist_checkpointing = _patch_dist_checkpoint_loader(monkeypatch, tmp_path, model)
    monkeypatch.setattr(low_precision_bootstrap.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(low_precision_bootstrap.torch.distributed, "get_world_size", lambda: world_size)

    low_precision_bootstrap.load_dist_checkpoint(model, str(checkpoint_dir))

    assert fake_dist_checkpointing.load_calls[0][2]["validate_access_integrity"] is expected
