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
    transformer_layer = ModuleType("megatron.core.transformer.transformer_layer")
    transformer_layer.get_transformer_layer_offset = lambda *_args, **_kwargs: 0
    for name, module in {
        "megatron": ModuleType("megatron"),
        "megatron.core": ModuleType("megatron.core"),
        "megatron.core.transformer": ModuleType("megatron.core.transformer"),
        "megatron.core.transformer.transformer_layer": transformer_layer,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    sglang = ModuleType("miles.backends.megatron_utils.sglang")
    sglang.FlattenedTensorBucket = object
    sglang.MultiprocessingSerializer = object
    monkeypatch.setitem(sys.modules, sglang.__name__, sglang)

    broadcast = ModuleType("miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast")
    broadcast.connect_rollout_engines_from_distributed = lambda *_args, **_kwargs: None
    broadcast.disconnect_rollout_engines_from_distributed = lambda *_args, **_kwargs: None
    broadcast.update_weights_from_distributed = lambda *_args, **_kwargs: []
    monkeypatch.setitem(sys.modules, broadcast.__name__, broadcast)

    module_name = "miles.backends.megatron_utils.update_weight.update_weight_from_tensor"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    module = importlib.import_module(module_name)
    yield module
    for name in set(sys.modules) - existing_modules:
        if name.startswith("miles.backends.megatron_utils"):
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
    def __init__(self, stage, ray_ref):
        self.stage = stage
        self.ray_ref = ray_ref

    def send_adapter(self, _tensors, *, weight_version):
        assert weight_version == 1
        if self.stage == "send":
            raise RuntimeError("transport send exploded")
        if self.stage == "validation":
            return SimpleNamespace(refs=[], results=[{"success": False, "error": "adapter rejected"}])
        return SimpleNamespace(refs=[self.ray_ref], results=None)


def _make_updater(update_mod, transport):
    updater = object.__new__(update_mod.UpdateWeightFromTensor)
    updater.args = Namespace(pause_generation_mode="retract")
    updater.weight_version = 0
    updater.peft_method = "oft"
    updater._peft_sync_spec = _SyncSpec(
        method="oft",
        adapter_name="miles_oft",
        adapter_config={"peft_type": "OFT"},
        sync_transport="oft_adapter",
    )
    updater._peft_transport = transport
    updater.rollout_engines = []
    updater._all_rollout_engines = [_Engine()]
    updater.use_distribute = True
    updater._is_distributed_src_rank = True
    updater._hf_weight_iterator = _Iterator()
    updater.weights_getter = lambda: {}
    return updater


@pytest.mark.parametrize("failure_stage", ["send", "ray", "validation"])
def test_distributed_peft_source_failure_reaches_peer_before_next_barrier(monkeypatch, update_mod, failure_stage):
    rank = {"value": 0}
    barrier_calls = {0: 0, 1: 0}
    source_record = {"value": None}
    gloo_group = object()
    ray_ref = object()
    transport = _Transport(failure_stage, ray_ref)
    updater = _make_updater(update_mod, transport)

    monkeypatch.setattr(update_mod.dist, "get_rank", lambda: rank["value"])
    monkeypatch.setattr(update_mod.dist, "get_world_size", lambda group=None: 2)
    monkeypatch.setattr(update_mod, "get_gloo_group", lambda: gloo_group)

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
