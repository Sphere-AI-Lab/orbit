"""CPU tests for sync-cost instrumentation in the weight-update paths.

Exercises UpdateWeightFromTensor.update_weights with mocked engines/dist/ray
(full-model colocated path end-to-end, PEFT orchestration with a stubbed
adapter send) and the per-transport payload-recording sites (NCCL, IPC, Ray).
"""

import json
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

import orbit.transport.backends.ipc as ipc_mod
import orbit.transport.backends.nccl as nccl_mod
import orbit.transport.backends.ray_object as ray_mod
import miles.backends.megatron_utils.update_weight.update_weight_from_tensor as uw_mod
from orbit.transport.backends.ipc import IpcBackend
from orbit.transport.backends.nccl import NcclBackend
from orbit.transport.backends.ray_object import RayObjectBackend
from orbit.transport.interface import PeftPayload
from orbit.transport.registry import PeftMethodSpec
from orbit.transport.runtime import PeftRuntimeMode
from orbit.megatron.peft_utils import PeftSyncSpec
from orbit.megatron.sync_metrics import (
    NUM_CHUNKS_KEY,
    PAUSE_TIMER_KEY,
    PAYLOAD_BYTES_KEY,
    PAYLOAD_NUM_TENSORS_KEY,
    TIMELINE_EVENTS_ENV_VAR,
    get_payload_tracker,
)
from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import (
    UpdateWeightFromTensor,
)
from miles.utils.timer import Timer


@pytest.fixture(autouse=True)
def _clean_metric_state():
    timer = Timer()
    timer.reset()
    timer.perf_scalars = {}
    get_payload_tracker().reset()
    yield
    timer.reset()
    timer.perf_scalars = {}
    get_payload_tracker().reset()


class _RemoteMethod:
    def __init__(self, result, call_log=None, name=None):
        self.result = result
        self.calls = []
        self._call_log = call_log
        self._name = name

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if self._call_log is not None:
            self._call_log.append(self._name)
        return self.result


class _FakeEngine:
    def __init__(self, call_log=None):
        ok = {"success": True}
        self.pause_generation = _RemoteMethod(ok, call_log, "pause_generation")
        self.flush_cache = _RemoteMethod(ok, call_log, "flush_cache")
        self.continue_generation = _RemoteMethod(ok, call_log, "continue_generation")
        self.update_weights_from_tensor = _RemoteMethod(ok, call_log, "update_weights_from_tensor")
        self.update_weight_version = _RemoteMethod(ok, call_log, "update_weight_version")
        self.unload_lora_adapter = _RemoteMethod(ok, call_log, "unload_lora_adapter")
        self.load_lora_adapter_from_tensors = _RemoteMethod(ok, call_log, "load_lora_adapter_from_tensors")
        self.load_lora_adapter_from_ray_tensors = _RemoteMethod(ok, call_log, "load_lora_adapter_from_ray_tensors")
        self.update_adapter_from_ray_tensor = _RemoteMethod(ok, call_log, "update_adapter_from_ray_tensor")


class _FakeLock:
    def __init__(self):
        self.acquire = _RemoteMethod(True)
        self.release = _RemoteMethod(True)


def _fake_ray_get(refs):
    return refs


class _FakeIterator:
    def __init__(self, chunks):
        self.chunks = chunks

    # upstream added the weight_type kwarg ("base" / adapter) to the real signature.
    def get_hf_weight_chunks(self, weights, weight_type="base"):
        yield from self.chunks


def _make_updater(engine, chunks, *, peft_method="none", peft_sync_spec=None, peft_transport=None):
    updater = object.__new__(UpdateWeightFromTensor)
    updater.args = Namespace(pause_generation_mode="retract")
    updater._peft_args = updater.args
    updater.weight_version = 0
    updater.peft_method = peft_method
    # upstream's update_weights now reads self.is_lora (set alongside peft_method in __init__).
    updater.is_lora = peft_method == "lora"
    updater._peft_sync_spec = peft_sync_spec
    updater._peft_transport = peft_transport
    updater.quantization_config = None
    updater.rollout_engines = [engine]
    updater._all_rollout_engines = [engine]
    updater.distributed_rollout_engines = []
    updater.use_distribute = False
    updater._is_distributed_src_rank = False
    updater._ipc_engine = engine
    updater._ipc_gather_src = 0
    updater._ipc_gather_group = object()
    updater._hf_weight_iterator = _FakeIterator(chunks)
    updater.weights_getter = lambda: {}
    return updater


def _patch_single_rank_dist(monkeypatch):
    monkeypatch.setattr(uw_mod.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(uw_mod.dist, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(uw_mod.dist, "barrier", lambda group=None: None)

    def fake_gather_object(obj, object_gather_list=None, dst=0, group=None):
        if object_gather_list is not None:
            object_gather_list[0] = obj

    monkeypatch.setattr(uw_mod.dist, "gather_object", fake_gather_object)
    monkeypatch.setattr(uw_mod, "get_gloo_group", lambda: None)
    monkeypatch.setattr(uw_mod.ray, "get", _fake_ray_get)
    # upstream replaced post_process_weights(restore_weights_before_load=/
    # post_process_quantization=) with an explicit begin/end weight-update session.
    monkeypatch.setattr(uw_mod, "begin_weight_update", lambda *args, **kwargs: None)
    monkeypatch.setattr(uw_mod, "end_weight_update", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        uw_mod.MultiprocessingSerializer, "serialize", staticmethod(lambda obj, output_str=False: "blob")
    )


def _patch_perf_counter(monkeypatch, values):
    remaining = list(values)

    def fake_perf_counter():
        if len(remaining) > 1:
            return remaining.pop(0)
        return remaining[0]

    monkeypatch.setattr(uw_mod.time, "perf_counter", fake_perf_counter)


# ---------------------------------------------------------------------------
# Full-model colocated path (real _send_to_colocated_engine)
# ---------------------------------------------------------------------------


def test_full_model_update_emits_payload_pause_and_events(monkeypatch, tmp_path):
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv(TIMELINE_EVENTS_ENV_VAR, str(events_file))

    call_log = []
    engine = _FakeEngine(call_log)
    chunks = [
        [("w1", torch.zeros(4, 4, dtype=torch.float32)), ("w2", torch.zeros(8, dtype=torch.float32))],
        [("w3", torch.zeros(2, 3, dtype=torch.float32))],
    ]
    expected_bytes = (4 * 4 + 8 + 2 * 3) * 4

    updater = _make_updater(engine, chunks)
    _patch_single_rank_dist(monkeypatch)
    _patch_perf_counter(monkeypatch, [100.0, 103.5])

    updater.update_weights()

    # Pause window: dispatch of pause_generation -> completion of continue.
    assert Timer().log_dict()[PAUSE_TIMER_KEY] == pytest.approx(3.5)
    scalars = Timer().perf_scalars
    assert scalars[PAYLOAD_BYTES_KEY] == expected_bytes
    assert scalars[PAYLOAD_NUM_TENSORS_KEY] == 2  # one flat bucket per chunk
    assert scalars[NUM_CHUNKS_KEY] == 2

    # Lifecycle ordering: pause -> flush -> sends -> continue.
    assert call_log.index("pause_generation") < call_log.index("update_weights_from_tensor")
    assert call_log.index("update_weights_from_tensor") < call_log.index("continue_generation")

    lines = [json.loads(line) for line in events_file.read_text().splitlines()]
    assert [rec["event"] for rec in lines] == ["update_start", "update_end"]
    for rec in lines:
        assert rec["weight_version"] == 1
        assert rec["mode"] == "full"


def test_full_model_update_without_events_env_writes_nothing(monkeypatch, tmp_path):
    monkeypatch.delenv(TIMELINE_EVENTS_ENV_VAR, raising=False)
    engine = _FakeEngine()
    updater = _make_updater(engine, [[("w1", torch.zeros(2, dtype=torch.float32))]])
    _patch_single_rank_dist(monkeypatch)

    updater.update_weights()

    assert list(tmp_path.iterdir()) == []
    assert Timer().perf_scalars[PAYLOAD_BYTES_KEY] == 8


def test_payload_tracker_resets_between_updates(monkeypatch):
    engine = _FakeEngine()
    updater = _make_updater(engine, [[("w1", torch.zeros(2, dtype=torch.float32))]])
    _patch_single_rank_dist(monkeypatch)

    updater.update_weights()
    first = Timer().perf_scalars[PAYLOAD_BYTES_KEY]
    updater._hf_weight_iterator = _FakeIterator([[("w1", torch.zeros(2, dtype=torch.float32))]])
    updater.update_weights()
    # Scalars accumulate across updates within one flush window (2 updates
    # here), but the tracker itself must reset per update: 8 + 8, not 8 + 16.
    assert Timer().perf_scalars[PAYLOAD_BYTES_KEY] == first * 2


# ---------------------------------------------------------------------------
# PEFT orchestration (stubbed adapter send) — mode labels
# ---------------------------------------------------------------------------


def _oft_sync_spec():
    return PeftSyncSpec(
        method="oft",
        adapter_name="orbit_oft",
        adapter_config={"peft_type": "OFT"},
        sync_transport="oft_adapter",
    )


def _run_peft_update(monkeypatch, *, double_buffer):
    engine = _FakeEngine()
    transport = SimpleNamespace(
        runtime_mode=SimpleNamespace(adapter_double_buffer=double_buffer)
    )
    chunks = [[("l.oft_R", torch.zeros(3, dtype=torch.float32))]]
    updater = _make_updater(
        engine,
        chunks,
        peft_method="oft",
        peft_sync_spec=_oft_sync_spec(),
        peft_transport=transport,
    )
    sent = []

    def fake_send_adapter_params(hf_named_tensors):
        sent.append(list(hf_named_tensors))
        get_payload_tracker().record(num_bytes=44, num_tensors=1)
        return [], [], [{"success": True}]

    updater._send_adapter_params = fake_send_adapter_params
    _patch_single_rank_dist(monkeypatch)
    updater.update_weights()
    return sent


def test_peft_single_slot_mode_label_and_metrics(monkeypatch, tmp_path):
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv(TIMELINE_EVENTS_ENV_VAR, str(events_file))

    sent = _run_peft_update(monkeypatch, double_buffer=False)

    assert len(sent) == 1  # coalesced into one adapter chunk
    scalars = Timer().perf_scalars
    assert scalars[PAYLOAD_BYTES_KEY] == 44
    assert scalars[PAYLOAD_NUM_TENSORS_KEY] == 1
    assert scalars[NUM_CHUNKS_KEY] == 1
    modes = {json.loads(line)["mode"] for line in events_file.read_text().splitlines()}
    assert modes == {"adapter_single_slot"}


def test_peft_double_buffer_mode_label(monkeypatch, tmp_path):
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv(TIMELINE_EVENTS_ENV_VAR, str(events_file))

    _run_peft_update(monkeypatch, double_buffer=True)

    modes = {json.loads(line)["mode"] for line in events_file.read_text().splitlines()}
    assert modes == {"adapter_double_buffer"}


# ---------------------------------------------------------------------------
# Transport send sites — payload recording
# ---------------------------------------------------------------------------


def _flat_payload_shaper(weight_tensors):
    return PeftPayload(
        flat_tensor=torch.cat([tensor.flatten() for _, tensor in weight_tensors]),
        metadata=[],
        extra={"entries": [(name, i) for i, (name, _) in enumerate(weight_tensors)]},
    )


def _oft_method_spec():
    return PeftMethodSpec(
        name="oft",
        sglang_load_format="oft_adapter",
        weight_name_predicate=lambda name: ".oft_" in name,
        dedupe_by_storage=True,
        payload_shaper=_flat_payload_shaper,
        sample_names="oft_R",
        label="OFT",
    )


def _runtime_mode(*, transport, double_buffer=False, use_distribute=True):
    return PeftRuntimeMode(
        peft_method="oft",
        use_distribute=use_distribute,
        distributed_transport=transport,
        adapter_versioning=True,
        adapter_double_buffer=double_buffer,
    )


class _FakeBroadcastHandle:
    def wait(self):
        return None


def test_nccl_send_adapter_records_flat_tensor_bytes(monkeypatch):
    monkeypatch.setattr(nccl_mod.ray, "get", _fake_ray_get)
    monkeypatch.setattr(
        nccl_mod.dist, "broadcast", lambda t, src, group=None, async_op=False: _FakeBroadcastHandle()
    )

    staged_ok = {
        "success": True,
        "staged_adapter_version": "1",
        "active_adapter_version": "1",
    }
    engine = _FakeEngine()
    engine.update_adapter_from_distributed = _RemoteMethod(staged_ok)

    backend = NcclBackend(
        args=Namespace(),
        method_spec=_oft_method_spec(),
        sync_spec=_oft_sync_spec(),
        runtime_mode=_runtime_mode(transport="nccl"),
    )
    backend._engines = [engine]
    backend._lock = _FakeLock()
    backend._group_name = "orbit-peft-pp_0"
    backend._model_update_group = object()

    named = [
        ("a.oft_R", torch.zeros(4, 4, dtype=torch.float32)),
        ("b.oft_R", torch.zeros(2, dtype=torch.float32)),
    ]
    backend.send_adapter(named, weight_version=1)

    tracker = get_payload_tracker()
    # ONE flat wire tensor carrying all adapter elements.
    assert tracker.payload_bytes == (16 + 2) * 4
    assert tracker.num_tensors == 1


def test_nccl_double_buffer_send_records_same_payload(monkeypatch):
    monkeypatch.setattr(nccl_mod.ray, "get", _fake_ray_get)
    monkeypatch.setattr(
        nccl_mod.dist, "broadcast", lambda t, src, group=None, async_op=False: _FakeBroadcastHandle()
    )

    staged_ok = {"success": True, "staged_adapter_version": "1"}
    active_ok = {"success": True, "active_adapter_version": "1"}
    engine = _FakeEngine()
    engine.update_adapter_from_distributed = _RemoteMethod(staged_ok)
    engine.activate_adapter_version = _RemoteMethod(active_ok)

    backend = NcclBackend(
        args=Namespace(),
        method_spec=_oft_method_spec(),
        sync_spec=_oft_sync_spec(),
        runtime_mode=_runtime_mode(transport="nccl", double_buffer=True),
    )
    backend._engines = [engine]
    backend._lock = _FakeLock()
    backend._group_name = "orbit-peft-pp_0"
    backend._model_update_group = object()

    backend.send_adapter([("a.oft_R", torch.zeros(8, dtype=torch.float32))], weight_version=1)

    # Stage + tensor-free activate: payload is the flat tensor only.
    assert len(engine.activate_adapter_version.calls) == 1
    tracker = get_payload_tracker()
    assert tracker.payload_bytes == 8 * 4
    assert tracker.num_tensors == 1


def test_ipc_send_adapter_records_flat_tensor_bytes(monkeypatch):
    monkeypatch.setattr(ipc_mod.ray, "get", _fake_ray_get)
    monkeypatch.setattr(ipc_mod.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(ipc_mod.dist, "get_world_size", lambda group=None: 1)
    monkeypatch.setattr(ipc_mod.dist, "barrier", lambda group=None: None)

    def fake_gather_object(obj, object_gather_list=None, dst=0, group=None):
        if object_gather_list is not None:
            object_gather_list[0] = obj

    monkeypatch.setattr(ipc_mod.dist, "gather_object", fake_gather_object)
    monkeypatch.setattr(
        ipc_mod.MultiprocessingSerializer, "serialize", staticmethod(lambda obj, output_str=False: "blob")
    )

    engine = _FakeEngine()
    backend = IpcBackend(
        args=Namespace(peft_method="oft", lora_adapter_path=None, peft_adapter_path=None),
        method_spec=_oft_method_spec(),
        sync_spec=_oft_sync_spec(),
        ipc_gather_group=object(),
        ipc_gather_src=0,
        runtime_mode=_runtime_mode(transport="nccl", use_distribute=False),
    )
    backend.connect([engine], _FakeLock())

    backend.send_adapter([("a.oft_R", torch.zeros(5, dtype=torch.float32))], weight_version=2)

    tracker = get_payload_tracker()
    assert tracker.payload_bytes == 5 * 4
    assert tracker.num_tensors == 1


def test_ray_send_adapter_records_flat_tensor_bytes(monkeypatch):
    monkeypatch.setattr(ray_mod.ray, "get", _fake_ray_get)

    engine = _FakeEngine()
    backend = RayObjectBackend(
        args=Namespace(peft_method="oft", lora_adapter_path=None, peft_adapter_path=None),
        method_spec=_oft_method_spec(),
        sync_spec=_oft_sync_spec(),
        runtime_mode=_runtime_mode(transport="ray"),
    )
    backend.connect([engine], _FakeLock())

    backend.send_adapter([("a.oft_R", torch.zeros(6, dtype=torch.float32))], weight_version=3)

    tracker = get_payload_tracker()
    assert tracker.payload_bytes == 6 * 4
    assert tracker.num_tensors == 1
