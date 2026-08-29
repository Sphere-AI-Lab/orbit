"""The distributed (non-colocated) full-model broadcast path must emit the same
perf/update_weights_* metrics and timeline markers as the tensor path, or A1's
full-model arm has no payload/pause series and A2's full-model trace has no
update windows. Mirrors tests/fast/test_update_weights_sync_metrics.py."""

import json
from argparse import Namespace

import pytest
import torch

import miles.backends.megatron_utils.update_weight.update_weight_from_distributed.mixin as mixin_mod
from orbit.megatron.sync_metrics import (
    NUM_CHUNKS_KEY,
    PAUSE_TIMER_KEY,
    PAYLOAD_BYTES_KEY,
    PAYLOAD_NUM_TENSORS_KEY,
    TIMELINE_EVENTS_ENV_VAR,
    get_payload_tracker,
)
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed.mixin import (
    DistBucketedWeightUpdateMixin,
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
    def __init__(self, call_log, name):
        self._call_log = call_log
        self._name = name

    def remote(self, *args, **kwargs):
        self._call_log.append(self._name)
        return {"success": True}


class _FakeEngine:
    def __init__(self, call_log):
        self.pause_generation = _RemoteMethod(call_log, "pause_generation")
        self.flush_cache = _RemoteMethod(call_log, "flush_cache")
        self.continue_generation = _RemoteMethod(call_log, "continue_generation")


def _make_updater(engine, chunks, call_log):
    updater = object.__new__(DistBucketedWeightUpdateMixin)
    updater.args = Namespace(pause_generation_mode="retract")
    updater.weight_version = 0
    updater.quantization_config = None
    updater.rollout_engines = [engine]
    updater._is_source = True
    updater._group_name = "test"
    updater._update_weight_implementation = lambda *a, **k: None

    def fake_non_expert(update_func, pbar):
        # What broadcast.py does per bucket on the source rank.
        for chunk in chunks:
            call_log.append("broadcast")
            get_payload_tracker().record(chunk)

    updater._gather_and_update_non_expert_weights = fake_non_expert
    updater._gather_and_update_expert_weights = lambda update_func, pbar: None
    return updater


def _patch_single_rank(monkeypatch, perf_values):
    monkeypatch.setattr(mixin_mod.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(mixin_mod.dist, "barrier", lambda group=None: None)
    monkeypatch.setattr(mixin_mod, "get_gloo_group", lambda: None)
    monkeypatch.setattr(mixin_mod.ray, "get", lambda refs: refs)
    monkeypatch.setattr(mixin_mod, "post_process_weights", lambda **kwargs: None)
    monkeypatch.setattr(mixin_mod, "sum_metrics_across_ranks", lambda values, group=None: list(values))
    remaining = list(perf_values)

    def fake_perf_counter():
        if len(remaining) > 1:
            return remaining.pop(0)
        return remaining[0]

    monkeypatch.setattr(mixin_mod.time, "perf_counter", fake_perf_counter)


def test_distributed_full_model_update_emits_metrics_and_events(monkeypatch, tmp_path):
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv(TIMELINE_EVENTS_ENV_VAR, str(events_file))

    call_log = []
    engine = _FakeEngine(call_log)
    chunks = [
        [("w1", torch.zeros(4, 4, dtype=torch.float32)), ("w2", torch.zeros(8, dtype=torch.float32))],
        [("w3", torch.zeros(2, 3, dtype=torch.float32))],
    ]
    expected_bytes = (4 * 4 + 8 + 2 * 3) * 4

    updater = _make_updater(engine, chunks, call_log)
    _patch_single_rank(monkeypatch, [100.0, 103.5])

    updater.update_weights()

    assert updater.weight_version == 1
    assert Timer().log_dict()[PAUSE_TIMER_KEY] == pytest.approx(3.5)
    scalars = Timer().perf_scalars
    assert scalars[PAYLOAD_BYTES_KEY] == expected_bytes
    assert scalars[PAYLOAD_NUM_TENSORS_KEY] == 3
    assert scalars[NUM_CHUNKS_KEY] == 2  # one tracker record per broadcast bucket

    assert call_log.index("pause_generation") < call_log.index("broadcast")
    assert call_log.index("broadcast") < call_log.index("continue_generation")

    lines = [json.loads(line) for line in events_file.read_text().splitlines()]
    assert [rec["event"] for rec in lines] == ["update_start", "update_end"]
    for rec in lines:
        assert rec["weight_version"] == 1
        assert rec["mode"] == "full"


def test_tracker_counts_records():
    tracker = get_payload_tracker()
    tracker.record([("a", torch.zeros(2, dtype=torch.float32))])
    tracker.record(num_bytes=10, num_tensors=1)
    assert tracker.num_records == 2
    tracker.reset()
    assert tracker.num_records == 0
