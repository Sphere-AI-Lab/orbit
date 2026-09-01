import sys
from types import ModuleType, SimpleNamespace

import orbit.backends.megatron_utils.low_precision_bootstrap as low_precision_bootstrap


class _FakeDistCheckpointing:
    @staticmethod
    def load_tensors_metadata(_path):
        return {}

    @staticmethod
    def load(sharded_state_dict, _path, **_kwargs):
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


def test_int4_checkpoint_load_supports_pinned_swiglu_factory_signature(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    (checkpoint_dir / ".metadata").touch()
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
    monkeypatch.setattr(low_precision_bootstrap, "_unwrap_parallel_model", lambda _model: [model])
    monkeypatch.setattr(low_precision_bootstrap, "_restore_modelopt_state_before_load", lambda *_args: None)
    monkeypatch.setattr(low_precision_bootstrap, "_import_dist_checkpointing", lambda: _FakeDistCheckpointing())
    monkeypatch.setattr(low_precision_bootstrap, "_get_assume_ok_unexpected_strict_handling", lambda: None)
    monkeypatch.setattr(low_precision_bootstrap, "detect_int4_checkpoint", lambda _path: (True, False))
    monkeypatch.setattr(low_precision_bootstrap, "detect_nvfp4_checkpoint", lambda _path: False)
    monkeypatch.setattr(low_precision_bootstrap, "detect_fp8_checkpoint", lambda _path: False)
    monkeypatch.setattr(low_precision_bootstrap, "_materialize_meta_sharded_tensor_data", lambda _state: None)
    monkeypatch.setattr(
        low_precision_bootstrap, "_materialize_loaded_meta_module_tensors", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(low_precision_bootstrap, "_zero_oft_adapter_parameters", lambda *_args: None)
    monkeypatch.setattr(low_precision_bootstrap, "_raise_if_meta_tensors_remain", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(low_precision_bootstrap, "_mark_dist_checkpoint_as_loaded", lambda *_args: None)
    monkeypatch.setattr(low_precision_bootstrap.torch.distributed, "is_initialized", lambda: False)

    low_precision_bootstrap.load_dist_checkpoint(model, str(checkpoint_dir))

    assert len(calls) == 1
    _, sharded_offsets, singleton_local_shards = calls[0]
    assert sharded_offsets == ((0, 0, 1),)
    assert singleton_local_shards is False
