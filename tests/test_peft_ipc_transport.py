from argparse import Namespace
from dataclasses import dataclass

import pytest
import torch
import torch.multiprocessing as torch_mp

from orbit.transport.backends import ipc as ipc_backend
from orbit.transport.backends.ipc import IpcBackend
from orbit.transport.registry import PeftMethodSpec
from orbit.megatron.peft_utils import PeftSyncSpec
from miles.backends.sglang_utils import sglang_engine as engine_module
from miles.backends.sglang_utils.sglang_engine import SGLangEngine


class _RemoteMethod:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def remote(self, **kwargs):
        self.calls.append(kwargs)
        return self.result


class _FailingRemoteMethod:
    def __init__(self, error):
        self.error = error

    def remote(self, **_kwargs):
        raise self.error


class _FakeEngine:
    def __init__(self):
        self.update_adapter_from_rank_tensors = _RemoteMethod({"loaded": True})
        self.update_weight_version = _RemoteMethod({"versioned": True})


class _FailingEngine:
    def __init__(self):
        self.update_adapter_from_rank_tensors = _FailingRemoteMethod(
            RuntimeError("scheduler load failed")
        )
        self.update_weight_version = _RemoteMethod({"versioned": True})


class _FailingVersionEngine(_FakeEngine):
    def __init__(self):
        super().__init__()
        self.update_weight_version = _FailingRemoteMethod(
            RuntimeError("weight version failed")
        )


@dataclass
class _FakeOftPayload:
    flat_tensor: torch.Tensor
    metadata: dict
    extra: dict


def _method_spec():
    def shape_oft(weight_tensors):
        return _FakeOftPayload(
            flat_tensor=torch.cat([tensor.flatten() for _, tensor in weight_tensors]),
            metadata={"entries": ["m0"]},
            extra={"entries": [("m0", 0)]},
        )

    return PeftMethodSpec(
        name="oft",
        sglang_load_format="oft_adapter",
        weight_name_predicate=lambda name: ".oft_" in name,
        dedupe_by_storage=True,
        payload_shaper=shape_oft,
        sample_names="oft_R",
        label="OFT",
    )


def test_ipc_oft_gathers_cpu_rank_tensors_before_calling_engine(monkeypatch):
    gathered_objects = []

    def gather_object(obj, object_gather_list, **_kwargs):
        gathered_objects.append(obj)
        object_gather_list[:] = [obj, obj]

    def all_gather_object(objects, obj, **_kwargs):
        objects[:] = [obj, None]

    monkeypatch.setenv("ORBIT_PEFT_ADAPTER_TRANSPORT", "cpu_gather")
    monkeypatch.setattr(ipc_backend.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(ipc_backend.dist, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(ipc_backend.dist, "gather_object", gather_object)
    monkeypatch.setattr(ipc_backend.dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(ipc_backend.dist, "barrier", lambda **_kwargs: None)
    monkeypatch.setattr(ipc_backend, "get_gloo_group", lambda: object(), raising=False)
    monkeypatch.setattr(ipc_backend.ray, "get", lambda value: value)

    engine = _FakeEngine()
    backend = IpcBackend(
        args=Namespace(
            peft_method="oft",
            peft_distributed_transport="nccl",
            adapter_double_buffer=False,
            peft_adapter_path=None,
        ),
        method_spec=_method_spec(),
        sync_spec=PeftSyncSpec(
            method="oft",
            adapter_name="orbit_oft",
            adapter_config={"peft_type": "OFT"},
            sync_transport="oft_adapter",
        ),
        ipc_gather_group=object(),
        ipc_gather_src=0,
    )
    backend.connect([engine], object())

    result = backend.send_adapter(
        [("model.layers.0.self_attn.q_proj.oft_R", torch.ones(2, 2))],
        weight_version=3,
    )

    assert result.results == [{"loaded": True}, {"versioned": True}]
    assert len(gathered_objects) == 1
    flat_tensor, metadata, entries = gathered_objects[0]
    assert flat_tensor.device.type == "cpu"
    assert metadata == {"entries": ["m0"]}
    assert entries == [("m0", 0)]
    assert engine.update_adapter_from_rank_tensors.calls == [
        {
            "rank_payloads": [gathered_objects[0], gathered_objects[0]],
            "payload_tag": "flattened_oft_payload",
            "load_format": "oft_adapter",
            "adapter_config": {"peft_type": "OFT"},
            "adapter_name": "orbit_oft",
        }
    ]
    assert engine.update_weight_version.calls == [{"weight_version": "3"}]


def test_engine_serializes_each_oft_rank_tensor_under_file_system(monkeypatch):
    calls = []

    class _Serializer:
        @staticmethod
        def serialize(value, output_str=False):
            calls.append((value, output_str, torch_mp.get_sharing_strategy()))
            return f"serialized-{len(calls)}"

    monkeypatch.setattr(engine_module, "MultiprocessingSerializer", _Serializer)
    engine = SGLangEngine.__new__(SGLangEngine)
    engine.nnodes = 1
    engine.args = Namespace(num_gpus_per_node=8, rollout_num_gpus_per_engine=2)
    engine.num_gpus_per_engine = 2
    captured = {}

    def update_weights_from_tensor(**kwargs):
        captured.update(kwargs)
        return {"success": True}

    engine.update_weights_from_tensor = update_weights_from_tensor
    old_strategy = torch_mp.get_sharing_strategy()
    rank_payloads = [
        (torch.arange(4), {"rank": 0}, [("m0", 0)]),
        (torch.arange(4, 8), {"rank": 1}, [("m1", 0)]),
    ]

    result = engine.update_adapter_from_rank_tensors(
        rank_payloads=rank_payloads,
        payload_tag="flattened_oft_payload",
        load_format="oft_adapter",
        adapter_config={"peft_type": "OFT"},
        adapter_name="orbit_oft",
    )

    assert result == {"success": True}
    assert captured == {
        "serialized_named_tensors": ["serialized-2", "serialized-4"],
        "load_format": "oft_adapter",
        "adapter_config": {"peft_type": "OFT"},
        "adapter_name": "orbit_oft",
    }
    assert [call[2] for call in calls] == ["file_system"] * 4
    assert torch_mp.get_sharing_strategy() == old_strategy


def test_engine_init_persists_launched_nnodes_from_server_args(monkeypatch):
    initialized = []

    monkeypatch.setattr(engine_module, "_to_local_gpu_id", lambda gpu_id: gpu_id)
    monkeypatch.setattr(
        SGLangEngine,
        "_init_normal",
        lambda _self, actual_server_args: initialized.append(actual_server_args),
    )
    engine = SGLangEngine(
        args=Namespace(
            env_report=None,
            sglang_router_ip=None,
            sglang_router_port=None,
            rollout_external=False,
            num_gpus_per_node=4,
            hf_checkpoint="test-model",
            seed=1,
            offload_rollout=False,
            sglang_dp_size=1,
            sglang_attn_cp_size=1,
            sglang_moe_dp_size=1,
            sglang_pp_size=1,
            sglang_ep_size=1,
            use_rollout_routing_replay=False,
            fp16=False,
            peft_method="none",
        ),
        rank=1,
        base_gpu_id=0,
        sglang_overrides={"nnodes": 3},
        num_gpus_per_engine=8,
    )

    engine.init(
        dist_init_addr="127.0.0.1:30000",
        port=30001,
        nccl_port=30002,
        host="127.0.0.1",
    )

    assert engine.num_gpus_per_engine // engine.args.num_gpus_per_node == 2
    assert engine.nnodes == 3
    assert len(initialized) == 1
    assert initialized[0]["nnodes"] == 3


def test_engine_rejects_multi_node_oft_rank_tensor_serialization(monkeypatch):
    class _Serializer:
        @staticmethod
        def serialize(_value, output_str=False):
            return "serialized"

    monkeypatch.setattr(engine_module, "MultiprocessingSerializer", _Serializer)
    monkeypatch.setattr(
        torch_mp,
        "set_sharing_strategy",
        lambda _strategy: pytest.fail(
            "multi-node rejection must precede sharing-strategy changes"
        ),
    )
    engine = SGLangEngine.__new__(SGLangEngine)
    engine.nnodes = 2
    engine.args = Namespace(num_gpus_per_node=4, rollout_num_gpus_per_engine=8)
    engine.num_gpus_per_engine = 8
    engine.update_weights_from_tensor = lambda **_kwargs: {"success": True}

    with pytest.raises(RuntimeError, match="single-host"):
        engine.update_adapter_from_rank_tensors(
            rank_payloads=[(torch.arange(4), {"rank": 0}, [("m0", 0)])],
            payload_tag="flattened_oft_payload",
            load_format="oft_adapter",
            adapter_config={"peft_type": "OFT"},
            adapter_name="orbit_oft",
        )


@pytest.mark.parametrize(
    ("failure_point", "expected_error"),
    [
        ("load_dispatch", "scheduler load failed"),
        ("load_result", "ray get failed"),
        ("version_dispatch", "weight version failed"),
        ("version_result", "weight version ray get failed"),
    ],
)
def test_ipc_oft_propagates_source_load_failure_across_engine_groups(
    monkeypatch, failure_point, expected_error
):
    rank = {"value": 0}
    source_record = {"value": None}
    all_gather_groups = []
    first_local_group = object()
    second_local_group = object()
    global_group = object()

    def gather_object(obj, object_gather_list, **_kwargs):
        if object_gather_list is not None:
            object_gather_list[:] = [obj, obj]

    def get_world_size(group):
        return 4 if group is global_group else 2

    def all_gather_object(objects, obj, **kwargs):
        all_gather_groups.append(kwargs["group"])
        if rank["value"] == 0:
            source_record["value"] = obj
        objects[:] = [source_record["value"], None, None, None]

    ray_get_calls = []

    def ray_get(value):
        ray_get_calls.append(value)
        if failure_point == "load_result" and len(ray_get_calls) == 1:
            raise RuntimeError("ray get failed")
        if failure_point == "version_result" and len(ray_get_calls) == 2:
            raise RuntimeError("weight version ray get failed")
        return value

    monkeypatch.setenv("ORBIT_PEFT_ADAPTER_TRANSPORT", "cpu_gather")
    monkeypatch.setattr(ipc_backend.dist, "get_rank", lambda: rank["value"])
    monkeypatch.setattr(ipc_backend.dist, "get_world_size", get_world_size)
    monkeypatch.setattr(ipc_backend.dist, "gather_object", gather_object)
    monkeypatch.setattr(ipc_backend.dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(
        ipc_backend.dist,
        "barrier",
        lambda **_kwargs: pytest.fail("OFT failure sync must not use a barrier"),
    )
    monkeypatch.setattr(ipc_backend.ray, "get", ray_get)
    monkeypatch.setattr(
        ipc_backend, "get_gloo_group", lambda: global_group, raising=False
    )

    if failure_point == "load_dispatch":
        source_engine = _FailingEngine()
    elif failure_point == "version_dispatch":
        source_engine = _FailingVersionEngine()
    else:
        source_engine = _FakeEngine()

    source_backend = IpcBackend(
        args=Namespace(
            peft_method="oft",
            peft_distributed_transport="nccl",
            adapter_double_buffer=False,
            peft_adapter_path=None,
        ),
        method_spec=_method_spec(),
        sync_spec=PeftSyncSpec(
            method="oft",
            adapter_name="orbit_oft",
            adapter_config={"peft_type": "OFT"},
            sync_transport="oft_adapter",
        ),
        ipc_gather_group=first_local_group,
        ipc_gather_src=0,
    )
    source_backend.connect([source_engine], object())

    peer_backend = IpcBackend(
        args=Namespace(
            peft_method="oft",
            peft_distributed_transport="nccl",
            adapter_double_buffer=False,
            peft_adapter_path=None,
        ),
        method_spec=_method_spec(),
        sync_spec=PeftSyncSpec(
            method="oft",
            adapter_name="orbit_oft",
            adapter_config={"peft_type": "OFT"},
            sync_transport="oft_adapter",
        ),
        ipc_gather_group=second_local_group,
        ipc_gather_src=2,
    )
    peer_backend.connect([_FakeEngine()], object())

    tensors = [("model.layers.0.self_attn.q_proj.oft_R", torch.ones(2, 2))]
    with pytest.raises(RuntimeError, match=expected_error):
        source_backend.send_adapter(tensors, weight_version=3)

    rank["value"] = 3
    with pytest.raises(RuntimeError, match=expected_error):
        peer_backend.send_adapter(tensors, weight_version=3)

    assert all_gather_groups == [global_group, global_group]


def test_ipc_oft_propagates_failed_engine_result_to_peer_rank(monkeypatch):
    rank = {"value": 0}
    source_record = {"value": None}
    first_local_group = object()
    second_local_group = object()
    global_group = object()

    def gather_object(obj, object_gather_list, **_kwargs):
        if object_gather_list is not None:
            object_gather_list[:] = [obj, obj]

    def get_world_size(group):
        return 4 if group is global_group else 2

    second_source_record = {
        "source_rank": 2,
        "results": [{"loaded": "second"}, {"versioned": "second"}],
        "error": None,
    }

    def all_gather_object(objects, obj, **_kwargs):
        if rank["value"] == 0:
            source_record["value"] = obj
        objects[:] = [source_record["value"], None, second_source_record, None]

    monkeypatch.setenv("ORBIT_PEFT_ADAPTER_TRANSPORT", "cpu_gather")
    monkeypatch.setattr(ipc_backend.dist, "get_rank", lambda: rank["value"])
    monkeypatch.setattr(ipc_backend.dist, "get_world_size", get_world_size)
    monkeypatch.setattr(ipc_backend.dist, "gather_object", gather_object)
    monkeypatch.setattr(ipc_backend.dist, "all_gather_object", all_gather_object)
    monkeypatch.setattr(ipc_backend.dist, "barrier", lambda **_kwargs: None)
    monkeypatch.setattr(ipc_backend.ray, "get", lambda value: value)
    monkeypatch.setattr(
        ipc_backend, "get_gloo_group", lambda: global_group, raising=False
    )

    failed_engine = _FakeEngine()
    failed_engine.update_adapter_from_rank_tensors = _RemoteMethod(
        {"success": False, "error": "adapter rejected"}
    )
    source_backend = IpcBackend(
        args=Namespace(
            peft_method="oft",
            peft_distributed_transport="nccl",
            adapter_double_buffer=False,
            peft_adapter_path=None,
        ),
        method_spec=_method_spec(),
        sync_spec=PeftSyncSpec(
            method="oft",
            adapter_name="orbit_oft",
            adapter_config={"peft_type": "OFT"},
            sync_transport="oft_adapter",
        ),
        ipc_gather_group=first_local_group,
        ipc_gather_src=0,
    )
    source_backend.connect([failed_engine], object())

    peer_backend = IpcBackend(
        args=Namespace(
            peft_method="oft",
            peft_distributed_transport="nccl",
            adapter_double_buffer=False,
            peft_adapter_path=None,
        ),
        method_spec=_method_spec(),
        sync_spec=PeftSyncSpec(
            method="oft",
            adapter_name="orbit_oft",
            adapter_config={"peft_type": "OFT"},
            sync_transport="oft_adapter",
        ),
        ipc_gather_group=second_local_group,
        ipc_gather_src=2,
    )
    peer_backend.connect([_FakeEngine()], object())

    tensors = [("model.layers.0.self_attn.q_proj.oft_R", torch.ones(2, 2))]
    source_result = source_backend.send_adapter(tensors, weight_version=3)

    rank["value"] = 3
    peer_result = peer_backend.send_adapter(tensors, weight_version=3)

    expected_results = [
        {"success": False, "error": "adapter rejected"},
        {"versioned": True},
        {"loaded": "second"},
        {"versioned": "second"},
    ]
    assert source_result.results == expected_results
    assert peer_result.results == expected_results
