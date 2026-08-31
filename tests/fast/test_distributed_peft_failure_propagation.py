"""Regression coverage for distributed PEFT source-rank failures.

The break caught here is a source-only adapter send, Ray resolution, or
result-validation failure letting peer trainer ranks enter the next global
barrier and hang forever.
"""

import importlib
import sys
from argparse import Namespace
from dataclasses import dataclass
from types import ModuleType, SimpleNamespace

import pytest
import torch


@dataclass(frozen=True)
class _SyncSpec:
    method: str
    adapter_name: str
    adapter_config: dict
    sync_transport: str


@pytest.fixture
def update_mod(monkeypatch):
    """Load the real orchestrator while stubbing unavailable runtime packages."""
    existing_modules = set(sys.modules)
    megatron = ModuleType("megatron")
    megatron.__path__ = []
    megatron_core = ModuleType("megatron.core")
    megatron_core.mpu = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", megatron_core)

    megatron_bridge = ModuleType("megatron.bridge")
    megatron_bridge.__path__ = []
    bridge_orbit = ModuleType("megatron.bridge.orbit")
    bridge_orbit.__path__ = []
    bridge_oft = ModuleType("megatron.bridge.orbit.oft")
    bridge_oft.__path__ = []
    param_names = ModuleType("megatron.bridge.orbit.oft.param_names")
    param_names.CANONICAL_OFT_SLICE_NAMES = ()
    param_names.is_peft_adapter_param_name = lambda _name: False
    for name, module in {
        "megatron.bridge": megatron_bridge,
        "megatron.bridge.orbit": bridge_orbit,
        "megatron.bridge.orbit.oft": bridge_oft,
        "megatron.bridge.orbit.oft.param_names": param_names,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    # The available repo environment is Python 3.11, while this unrelated
    # helper uses Python 3.12's ``type`` statement. Its symbols are import-only
    # for the orchestration exercised below.
    adapter_tensors = ModuleType("orbit.utils.adapter_tensors")
    adapter_tensors.AdapterTensorKey = tuple
    adapter_tensors.adapter_named_parameters = lambda *_args, **_kwargs: ()
    adapter_tensors.adapter_tensor_key_digest = lambda *_args, **_kwargs: ""
    monkeypatch.setitem(sys.modules, adapter_tensors.__name__, adapter_tensors)

    sglang = ModuleType("orbit.backends.megatron_utils.sglang")
    sglang.FlattenedTensorBucket = object
    sglang.MultiprocessingSerializer = object
    monkeypatch.setitem(sys.modules, sglang.__name__, sglang)

    broadcast = ModuleType(
        "orbit.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast"
    )
    broadcast.connect_rollout_engines_from_distributed = lambda *_args, **_kwargs: None
    broadcast.disconnect_rollout_engines_from_distributed = lambda *_args, **_kwargs: None
    broadcast.update_weights_from_distributed = lambda *_args, **_kwargs: []
    monkeypatch.setitem(sys.modules, broadcast.__name__, broadcast)

    module_name = "orbit.backends.megatron_utils.update_weight.update_weight_from_tensor"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    module = importlib.import_module(module_name)
    yield module
    for name in set(sys.modules) - existing_modules:
        if name.startswith("orbit.backends.megatron_utils"):
            sys.modules.pop(name, None)


class _RemoteMethod:
    def remote(self, **_kwargs):
        return {"success": True}


class _Engine:
    pause_generation = _RemoteMethod()
    flush_cache = _RemoteMethod()
    continue_generation = _RemoteMethod()


class _Iterator:
    def get_hf_weight_chunks(self, _weights):
        yield [("layer.oft_R", torch.ones(1))]


class _Transport:
    def __init__(self, stage, ray_ref, sync_spec):
        self.stage = stage
        self.ray_ref = ray_ref
        self.sync_spec = sync_spec
        self.runtime_mode = SimpleNamespace(adapter_double_buffer=False)

    def send_adapter(self, _tensors, *, weight_version):
        assert weight_version == 1
        if self.stage == "send":
            raise RuntimeError("transport send exploded")
        if self.stage == "validation":
            return SimpleNamespace(refs=[], results=[{"success": False, "error": "adapter rejected"}])
        return SimpleNamespace(refs=[self.ray_ref], results=None)


def _make_updater(update_mod, transport, sync_spec):
    updater = object.__new__(update_mod.UpdateWeightFromTensor)
    updater.args = Namespace(pause_generation_mode="retract", peft_method="oft")
    updater._peft_args = updater.args
    updater.weight_version = 0
    updater.peft_method = "oft"
    updater._peft_sync_spec = sync_spec
    updater._peft_transport = transport
    updater.quantization_config = None
    updater.rollout_engines = []
    updater._all_rollout_engines = [_Engine()]
    updater.distributed_rollout_engines = [_Engine()]
    updater.use_distribute = True
    updater._is_distributed_src_rank = True
    updater._hf_weight_iterator = _Iterator()
    updater.weights_getter = lambda: {}
    return updater


@pytest.mark.parametrize("failure_stage", ["send", "ray", "validation"])
def test_distributed_peft_source_failure_reaches_peer_before_next_barrier(
    monkeypatch, update_mod, failure_stage
):
    rank = {"value": 0}
    barrier_calls = {0: 0, 1: 0}
    source_record = {"value": None}
    gloo_group = object()
    ray_ref = object()
    sync_spec = _SyncSpec(
        method="oft",
        adapter_name="orbit_oft",
        adapter_config={"peft_type": "OFT"},
        sync_transport="oft_adapter",
    )
    transport = _Transport(failure_stage, ray_ref, sync_spec)
    updater = _make_updater(update_mod, transport, sync_spec)

    monkeypatch.setattr(update_mod.dist, "get_rank", lambda: rank["value"])
    monkeypatch.setattr(update_mod.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(update_mod, "get_gloo_group", lambda: gloo_group)
    monkeypatch.setattr(update_mod, "post_process_weights", lambda **_kwargs: None)
    monkeypatch.setattr(update_mod, "sum_metrics_across_ranks", lambda values, **_kwargs: values)
    monkeypatch.setattr(update_mod, "emit_update_weights_metrics", lambda **_kwargs: None)
    monkeypatch.setattr(update_mod, "emit_timeline_event", lambda *_args, **_kwargs: None)

    def barrier(*, group):
        assert group is gloo_group
        barrier_calls[rank["value"]] += 1
        if barrier_calls[rank["value"]] > 1:
            pytest.fail("trainer rank reached the post-chunk barrier after a source failure")

    def all_gather_object(records, record, *, group):
        assert group is gloo_group
        if record is not None:
            source_record["value"] = record
        records[:] = [source_record["value"], None]

    def ray_get(refs):
        if refs == [ray_ref]:
            raise RuntimeError("ray resolution exploded")
        return refs

    monkeypatch.setattr(update_mod.dist, "barrier", barrier)
    monkeypatch.setattr(update_mod.dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(update_mod.ray, "get", ray_get)

    with pytest.raises(RuntimeError) as source_failure:
        updater.update_weights()

    rank["value"] = 1
    updater._is_distributed_src_rank = False
    updater._peft_transport = None
    with pytest.raises(RuntimeError, match=r"PEFT adapter dispatch failed on source rank 0") as peer_failure:
        updater.update_weights()

    assert barrier_calls == {0: 1, 1: 1}
    assert source_failure.value.__cause__ is not None
    assert peer_failure.value.__cause__ is None
