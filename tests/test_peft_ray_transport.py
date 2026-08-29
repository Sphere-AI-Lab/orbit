from argparse import Namespace
from dataclasses import dataclass

import pytest
import torch

from orbit.peft.megatron.peft_utils import PeftSyncSpec
from orbit.peft.transport import build_peft_transport
from orbit.peft.transport.backends import ray_object as ray_backend
from orbit.peft.transport.backends.ray_object import RayObjectBackend
from orbit.peft.transport.registry import PEFT_METHODS, PeftMethodSpec
from orbit.peft.transport.runtime import resolve_peft_runtime_mode


class _RemoteMethod:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def remote(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class _FakeLock:
    def __init__(self):
        self.acquire = _RemoteMethod(True)
        self.release = _RemoteMethod(True)


class _FakeEngine:
    def __init__(self):
        self.unload_lora_adapter = _RemoteMethod({"unloaded": True})
        self.load_lora_adapter_from_ray_tensors = _RemoteMethod({"loaded": True})
        self.update_adapter_from_ray_tensor = _RemoteMethod({"loaded_shaped": True})
        self.update_weight_version = _RemoteMethod({"versioned": True})


def _args(peft_method="lora", transport="ray", double_buffer=False):
    return Namespace(
        peft_method=peft_method,
        peft_distributed_transport=transport,
        adapter_double_buffer=double_buffer,
        lora_adapter_path=None,
        peft_adapter_path=None,
    )


def _sync_spec(method="lora"):
    return PeftSyncSpec(
        method=method,
        adapter_name=f"orbit_{method}",
        adapter_config={"peft_type": method.upper()},
        sync_transport=f"{method}_adapter",
    )


def _fake_ray_get(value):
    return value


def test_build_peft_transport_selects_ray_backend(monkeypatch):
    monkeypatch.setattr(
        "orbit.peft.transport.build_peft_sync_spec",
        lambda _args: _sync_spec("lora"),
    )

    transport = build_peft_transport(_args(), use_distribute=True)

    assert isinstance(transport, RayObjectBackend)
    assert transport.runtime_mode.transport == "ray"


def test_ray_transport_rejects_adapter_double_buffer():
    with pytest.raises(ValueError, match="adapter-double-buffer"):
        resolve_peft_runtime_mode(_args(double_buffer=True), use_distribute=True)


def test_ray_backend_sends_lora_adapter_and_weight_version(monkeypatch):
    monkeypatch.setattr(ray_backend.ray, "get", _fake_ray_get)
    engine = _FakeEngine()
    backend = RayObjectBackend(
        args=_args(),
        method_spec=PEFT_METHODS["lora"],
        sync_spec=_sync_spec("lora"),
    )
    backend.connect([engine], _FakeLock())

    result = backend.send_adapter(
        [("model.layers.0.self_attn.q_proj.lora_A.weight", torch.ones(2, 2))],
        weight_version=7,
    )

    # LoRA carries a payload_shaper too, so it takes the shaped path -- not the
    # per-tensor load_lora_adapter_from_ray_tensors one.
    assert result.results == [{"loaded_shaped": True}, {"versioned": True}]
    assert engine.load_lora_adapter_from_ray_tensors.calls == []
    assert len(engine.update_adapter_from_ray_tensor.calls) == 1
    load_call = engine.update_adapter_from_ray_tensor.calls[0]
    # The tag must follow the method. sglang's normalize_lora_weight_payload
    # asserts payload[0] == "flattened_lora_payload"; sending the OFT tag here
    # (as this path did when it was hardcoded) fails the adapter load outright.
    assert load_call["payload_tag"] == "flattened_lora_payload"
    assert load_call["load_format"] == "lora_adapter"
    assert load_call["adapter_config"] == {"peft_type": "LORA"}
    assert load_call["adapter_name"] == "orbit_lora"
    assert load_call["flat_tensor"].device.type == "cpu"
    assert engine.update_weight_version.calls == [{"weight_version": "7"}]


@dataclass
class _FakeOftPayload:
    flat_tensor: torch.Tensor
    metadata: dict
    extra: dict


def test_ray_backend_sends_oft_adapter_and_weight_version(monkeypatch):
    monkeypatch.setattr(ray_backend.ray, "get", _fake_ray_get)

    def shape_oft(weight_tensors):
        return _FakeOftPayload(
            flat_tensor=torch.cat([tensor.flatten() for _, tensor in weight_tensors]),
            metadata={"entries": ["m0"]},
            extra={"entries": [("m0", 0)]},
        )

    method_spec = PeftMethodSpec(
        name="oft",
        sglang_load_format="oft_adapter",
        weight_name_predicate=lambda name: ".oft_" in name,
        dedupe_by_storage=True,
        payload_shaper=shape_oft,
        sample_names="oft_R",
        label="OFT",
    )
    engine = _FakeEngine()
    backend = RayObjectBackend(
        args=_args(peft_method="oft"),
        method_spec=method_spec,
        sync_spec=_sync_spec("oft"),
    )
    backend.connect([engine], _FakeLock())

    result = backend.send_adapter(
        [("model.layers.0.self_attn.q_proj.oft_R", torch.ones(2, 2))],
        weight_version=11,
    )

    assert result.results == [{"loaded_shaped": True}, {"versioned": True}]
    assert len(engine.update_adapter_from_ray_tensor.calls) == 1
    load_call = engine.update_adapter_from_ray_tensor.calls[0]
    assert load_call["payload_tag"] == "flattened_oft_payload"
    assert load_call["flat_tensor"].device.type == "cpu"
    assert load_call["metadata"] == {"entries": ["m0"]}
    assert load_call["entries"] == [("m0", 0)]
    assert load_call["load_format"] == "oft_adapter"
    assert load_call["adapter_config"] == {"peft_type": "OFT"}
    assert load_call["adapter_name"] == "orbit_oft"
    assert engine.update_weight_version.calls == [{"weight_version": "11"}]
