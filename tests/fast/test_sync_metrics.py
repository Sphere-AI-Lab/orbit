"""CPU unit tests for weight-sync instrumentation helpers.

Covers byte accounting on synthetic tensors, the Timer perf-scalar staging
channel and its pickup by log_perf_data_raw, defensive (never-raise) behavior,
and env-gated timeline event emission.
"""

import json
from argparse import Namespace

import pytest
import torch

from orbit.backends.megatron_utils.update_weight import sync_metrics
from orbit.backends.megatron_utils.update_weight.sync_metrics import (
    NUM_CHUNKS_KEY,
    PAUSE_TIMER_KEY,
    PAYLOAD_BYTES_KEY,
    PAYLOAD_NUM_TENSORS_KEY,
    WeightSyncPayloadTracker,
    emit_timeline_event,
    emit_update_weights_metrics,
    get_payload_tracker,
    named_tensors_num_bytes,
    record_perf_scalar,
    sum_metrics_across_ranks,
    tensor_num_bytes,
)
from orbit.utils import train_metric_utils
from orbit.utils.timer import Timer


@pytest.fixture(autouse=True)
def _clean_timer_singleton():
    timer = Timer()
    timer.reset()
    timer.perf_scalars = {}
    get_payload_tracker().reset()
    yield
    timer.reset()
    timer.perf_scalars = {}
    get_payload_tracker().reset()


# ---------------------------------------------------------------------------
# Byte accounting
# ---------------------------------------------------------------------------


def test_tensor_num_bytes_fp32_and_bf16():
    assert tensor_num_bytes(torch.zeros(3, 4, dtype=torch.float32)) == 3 * 4 * 4
    assert tensor_num_bytes(torch.zeros(5, dtype=torch.bfloat16)) == 5 * 2
    assert tensor_num_bytes(torch.zeros(0, dtype=torch.float32)) == 0


def test_named_tensors_num_bytes_pairs_and_bare_tensors():
    named = [
        ("a", torch.zeros(2, 2, dtype=torch.float32)),  # 16 B
        ("b", torch.zeros(8, dtype=torch.bfloat16)),  # 16 B
    ]
    assert named_tensors_num_bytes(named) == 32
    bare = [torch.zeros(4, dtype=torch.uint8), torch.zeros(2, dtype=torch.float64)]
    assert named_tensors_num_bytes(bare) == 4 + 16
    assert named_tensors_num_bytes([]) == 0
    assert named_tensors_num_bytes([("skip", None), None]) == 0  # None entries skipped


def test_named_tensors_num_bytes_accepts_generator():
    gen = ((f"t{i}", torch.zeros(i, dtype=torch.float32)) for i in range(4))
    assert named_tensors_num_bytes(gen) == (0 + 1 + 2 + 3) * 4


# ---------------------------------------------------------------------------
# Payload tracker
# ---------------------------------------------------------------------------


def test_tracker_accumulates_and_resets():
    tracker = WeightSyncPayloadTracker()
    tracker.record([("a", torch.zeros(4, dtype=torch.float32))])
    tracker.record(num_bytes=100, num_tensors=2)
    assert tracker.payload_bytes == 16 + 100
    assert tracker.num_tensors == 1 + 2
    tracker.reset()
    assert tracker.payload_bytes == 0
    assert tracker.num_tensors == 0


def test_tracker_explicit_counts_override_derived():
    tracker = WeightSyncPayloadTracker()
    tracker.record([("a", torch.zeros(4, dtype=torch.float32))], num_tensors=7)
    assert tracker.payload_bytes == 16
    assert tracker.num_tensors == 7


def test_tracker_record_never_raises():
    tracker = WeightSyncPayloadTracker()

    class Broken:
        def numel(self):
            raise RuntimeError("boom")

        def element_size(self):
            return 4

    tracker.record([("bad", Broken())])  # must not raise
    assert tracker.payload_bytes == 0
    assert tracker.num_tensors == 0


def test_get_payload_tracker_is_process_wide():
    assert get_payload_tracker() is get_payload_tracker()


# ---------------------------------------------------------------------------
# Perf scalar staging + log_perf_data_raw pickup
# ---------------------------------------------------------------------------


def test_record_perf_scalar_accumulates_on_timer_singleton():
    record_perf_scalar("update_weights_payload_bytes", 10)
    record_perf_scalar("update_weights_payload_bytes", 5)
    assert Timer().perf_scalars == {"update_weights_payload_bytes": 15}


def test_emit_update_weights_metrics_stages_all_keys():
    emit_update_weights_metrics(
        pause_seconds=1.5, payload_bytes=1024, num_tensors=3, num_chunks=2
    )
    assert Timer().log_dict()[PAUSE_TIMER_KEY] == pytest.approx(1.5)
    scalars = Timer().perf_scalars
    assert scalars[PAYLOAD_BYTES_KEY] == 1024
    assert scalars[PAYLOAD_NUM_TENSORS_KEY] == 3
    assert scalars[NUM_CHUNKS_KEY] == 2


def test_emit_update_weights_metrics_no_pause_key_when_none():
    emit_update_weights_metrics(
        pause_seconds=None, payload_bytes=1, num_tensors=1, num_chunks=1
    )
    assert PAUSE_TIMER_KEY not in Timer().log_dict()


def test_log_perf_data_raw_emits_perf_scalars(monkeypatch):
    logged = {}
    monkeypatch.setattr(
        train_metric_utils.tracking_utils, "log", lambda args, metrics, step_key: logged.update(metrics)
    )
    monkeypatch.setattr(train_metric_utils, "compute_rollout_step", lambda args, rollout_id: 7)

    emit_update_weights_metrics(
        pause_seconds=0.25, payload_bytes=2048, num_tensors=4, num_chunks=1
    )
    train_metric_utils.log_perf_data_raw(
        rollout_id=3,
        args=Namespace(),
        is_primary_rank=True,
        compute_total_fwd_flops=None,
    )

    assert logged["perf/update_weights_pause_time"] == pytest.approx(0.25)
    assert logged["perf/update_weights_payload_bytes"] == 2048
    assert logged["perf/update_weights_payload_num_tensors"] == 4
    assert logged["perf/update_weights_num_chunks"] == 1
    assert logged["rollout/step"] == 7
    # Flush must clear the staged scalars so the next window starts clean.
    assert Timer().perf_scalars == {}


def test_log_perf_data_raw_clears_scalars_on_non_primary_rank(monkeypatch):
    monkeypatch.setattr(
        train_metric_utils.tracking_utils, "log", lambda *a, **k: pytest.fail("must not log")
    )
    record_perf_scalar("update_weights_payload_bytes", 99)
    train_metric_utils.log_perf_data_raw(
        rollout_id=0,
        args=Namespace(),
        is_primary_rank=False,
        compute_total_fwd_flops=None,
    )
    assert Timer().perf_scalars == {}


def test_log_perf_data_raw_without_scalars_still_works(monkeypatch):
    timer = Timer()
    if hasattr(timer, "perf_scalars"):
        del timer.perf_scalars
    logged = {}
    monkeypatch.setattr(
        train_metric_utils.tracking_utils, "log", lambda args, metrics, step_key: logged.update(metrics)
    )
    monkeypatch.setattr(train_metric_utils, "compute_rollout_step", lambda args, rollout_id: 0)
    timer.add("update_weights", 2.0)
    train_metric_utils.log_perf_data_raw(
        rollout_id=0,
        args=Namespace(),
        is_primary_rank=True,
        compute_total_fwd_flops=None,
    )
    assert logged["perf/update_weights_time"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Cross-rank reduction fallback
# ---------------------------------------------------------------------------


def test_sum_metrics_across_ranks_passthrough_without_dist():
    # torch.distributed is not initialized in CPU tests: local passthrough.
    assert sum_metrics_across_ranks([1.5, 2, 3]) == [1.5, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Timeline events
# ---------------------------------------------------------------------------


def test_emit_timeline_event_noop_without_env(monkeypatch, tmp_path):
    monkeypatch.delenv(sync_metrics.TIMELINE_EVENTS_ENV_VAR, raising=False)
    emit_timeline_event("update_start", weight_version=1, mode="full")
    assert list(tmp_path.iterdir()) == []
    assert not sync_metrics.timeline_events_enabled()


def test_emit_timeline_event_appends_valid_jsonl(monkeypatch, tmp_path):
    events_file = tmp_path / "events.jsonl"
    monkeypatch.setenv(sync_metrics.TIMELINE_EVENTS_ENV_VAR, str(events_file))
    assert sync_metrics.timeline_events_enabled()

    emit_timeline_event("update_start", weight_version=3, mode="adapter_single_slot")
    emit_timeline_event("update_end", weight_version=3, mode="adapter_single_slot")

    lines = events_file.read_text().splitlines()
    assert len(lines) == 2
    start, end = (json.loads(line) for line in lines)
    assert start["event"] == "update_start"
    assert end["event"] == "update_end"
    for record in (start, end):
        assert record["weight_version"] == 3
        assert record["mode"] == "adapter_single_slot"
        assert isinstance(record["t_wall"], float)
    assert start["t_wall"] <= end["t_wall"]


def test_emit_timeline_event_never_raises_on_bad_path(monkeypatch, tmp_path):
    monkeypatch.setenv(
        sync_metrics.TIMELINE_EVENTS_ENV_VAR, str(tmp_path / "no_such_dir" / "events.jsonl")
    )
    emit_timeline_event("update_start", weight_version=1, mode="full")  # must not raise
