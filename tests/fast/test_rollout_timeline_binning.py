"""CPU unit tests for tools/rollout_timeline/binning.py.

Synthetic counter series (steady rates, counter resets, scrape failures,
multi-engine merges) -> expected bins, plus event pairing and annotation.
"""

import json

import pytest

from tools.rollout_timeline import binning
from tools.rollout_timeline.binning import (
    Bin,
    annotate_bins,
    bin_engine_samples,
    build_timeline,
    combine_engine_bins,
    counter_intervals,
    extract_engine_series,
    load_jsonl,
    make_bin_edges,
    update_windows,
)


def _probe_record(t, url="http://e0", value=None, ok=True, counter="c"):
    record = {"t_wall": t, "engine_url": url, "ok": ok}
    if ok:
        record["counters"] = {} if value is None else {counter: value}
    else:
        record["error"] = "timeout"
    return record


# ---------------------------------------------------------------------------
# load_jsonl
# ---------------------------------------------------------------------------


def test_load_jsonl_skips_blank_and_truncated_lines(tmp_path):
    path = tmp_path / "probe.jsonl"
    path.write_text(
        json.dumps({"a": 1}) + "\n\n" + json.dumps({"b": 2}) + "\n" + '{"trunca'
    )
    assert load_jsonl(str(path)) == [{"a": 1}, {"b": 2}]


# ---------------------------------------------------------------------------
# Series extraction + intervals
# ---------------------------------------------------------------------------


def test_extract_engine_series_orders_and_maps_failures_to_none():
    records = [
        _probe_record(2.0, value=20.0),
        _probe_record(1.0, value=10.0),
        _probe_record(3.0, ok=False),
        _probe_record(4.0, value=None),  # ok but counter missing
        _probe_record(1.5, url="http://e1", value=5.0),
    ]
    series = extract_engine_series(records, "c")
    assert series["http://e0"] == [(1.0, 10.0), (2.0, 20.0), (3.0, None), (4.0, None)]
    assert series["http://e1"] == [(1.5, 5.0)]


def test_counter_intervals_steady_series():
    samples = [(0.0, 0.0), (0.1, 10.0), (0.2, 20.0)]
    intervals, failures = counter_intervals(samples)
    assert failures == []
    assert len(intervals) == 2
    assert intervals[0].tokens == 10.0 and not intervals[0].is_reset
    assert intervals[1].tokens == 10.0


def test_counter_intervals_bridge_over_failed_scrape():
    # Failed scrape at t=0.1 does not lose tokens: 0.0 -> 0.2 delta survives.
    samples = [(0.0, 0.0), (0.1, None), (0.2, 30.0)]
    intervals, failures = counter_intervals(samples)
    assert failures == [0.1]
    assert len(intervals) == 1
    assert (intervals[0].t_start, intervals[0].t_end, intervals[0].tokens) == (0.0, 0.2, 30.0)


def test_counter_intervals_reset_yields_gap_not_negative():
    samples = [(0.0, 50.0), (0.1, 60.0), (0.2, 5.0), (0.3, 15.0)]
    intervals, _ = counter_intervals(samples)
    assert [i.tokens for i in intervals] == [10.0, 0.0, 10.0]
    assert [i.is_reset for i in intervals] == [False, True, False]


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------


def test_make_bin_edges_covers_range():
    edges = make_bin_edges(0.0, 0.35, 0.1)
    assert edges == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])
    assert make_bin_edges(5.0, 5.0, 0.1) == pytest.approx([5.0, 5.1])


def test_bin_engine_samples_steady_rate():
    # 100 tokens/s, sampled every 0.1s over 4 bins.
    samples = [(0.0 + i * 0.1, i * 10.0) for i in range(5)]
    bins = bin_engine_samples(samples, make_bin_edges(0.0, 0.4, 0.1))
    assert len(bins) == 4
    for b in bins:
        assert b.tokens == pytest.approx(10.0)
        assert b.tokens_per_s == pytest.approx(100.0)
        assert not b.has_gap


def test_bin_engine_samples_spreads_interval_across_bins():
    # One interval [0.0, 0.25] with 25 tokens -> 10/10/5 across 0.1 bins.
    samples = [(0.0, 0.0), (0.25, 25.0)]
    bins = bin_engine_samples(samples, make_bin_edges(0.0, 0.3, 0.1))
    assert [b.tokens for b in bins] == pytest.approx([10.0, 10.0, 5.0])
    # Last bin only covered for 0.05s -> flagged as (partial) gap but the
    # rate over the covered time is still the true 100 tok/s.
    assert [b.has_gap for b in bins] == [False, False, True]
    assert bins[2].tokens_per_s == pytest.approx(100.0)


def test_bin_engine_samples_failed_scrape_flags_bin_but_keeps_tokens():
    samples = [(0.0, 0.0), (0.1, 10.0), (0.15, None), (0.2, 20.0), (0.3, 30.0)]
    bins = bin_engine_samples(samples, make_bin_edges(0.0, 0.3, 0.1))
    assert [b.tokens for b in bins] == pytest.approx([10.0, 10.0, 10.0])
    assert bins[1].scrape_failures == 1
    assert bins[1].has_gap
    assert not bins[0].has_gap and not bins[2].has_gap


def test_bin_engine_samples_reset_marks_gap_bins():
    samples = [(0.0, 100.0), (0.1, 110.0), (0.2, 0.0), (0.3, 10.0)]
    bins = bin_engine_samples(samples, make_bin_edges(0.0, 0.3, 0.1))
    assert bins[0].tokens == pytest.approx(10.0)
    assert bins[1].tokens == pytest.approx(0.0)
    assert bins[1].has_gap  # reset interval
    assert bins[2].tokens == pytest.approx(10.0)
    assert not bins[2].has_gap


def test_bin_engine_samples_uncovered_edges_are_gaps():
    # Sampling starts at t=0.15: bins before coverage are gaps.
    samples = [(0.15, 0.0), (0.25, 10.0)]
    bins = bin_engine_samples(samples, make_bin_edges(0.0, 0.3, 0.1))
    assert bins[0].has_gap and bins[0].tokens == 0.0
    assert bins[1].has_gap  # only half covered
    assert bins[2].has_gap  # only half covered
    assert bins[1].tokens + bins[2].tokens == pytest.approx(10.0)


def test_combine_engine_bins_sums_rates_and_propagates_gaps():
    edges = make_bin_edges(0.0, 0.2, 0.1)
    e0 = bin_engine_samples([(0.0, 0.0), (0.1, 10.0), (0.2, 20.0)], edges)
    e1 = bin_engine_samples([(0.0, 0.0), (0.1, 30.0), (0.15, None), (0.2, 60.0)], edges)
    combined = combine_engine_bins({"e0": e0, "e1": e1})
    assert len(combined) == 2
    assert combined[0].tokens == pytest.approx(40.0)
    assert combined[0].tokens_per_s == pytest.approx(400.0)
    assert not combined[0].has_gap
    assert combined[1].has_gap  # e1 failed a scrape in bin 1
    assert combined[1].tokens == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def _event(t, event, version=1, mode="full"):
    return {"t_wall": t, "event": event, "weight_version": version, "mode": mode}


def test_update_windows_pairs_by_version():
    events = [
        _event(1.0, "update_start", version=1),
        _event(1.5, "update_end", version=1),
        _event(3.0, "update_start", version=2, mode="adapter_single_slot"),
        _event(3.2, "update_end", version=2, mode="adapter_single_slot"),
    ]
    windows = update_windows(events)
    assert [(w["t_start"], w["t_end"], w["weight_version"]) for w in windows] == [
        (1.0, 1.5, 1),
        (3.0, 3.2, 2),
    ]
    assert windows[1]["mode"] == "adapter_single_slot"


def test_update_windows_unmatched_start_stays_open():
    windows = update_windows([_event(1.0, "update_start", version=9)])
    assert windows == [
        {"t_start": 1.0, "t_end": None, "weight_version": 9, "mode": "full"}
    ]


def test_annotate_bins_marks_update_overlap_and_attaches_events():
    bins = [Bin(0.0, 0.1), Bin(0.1, 0.2), Bin(0.2, 0.3)]
    events = [
        _event(0.05, "update_start", version=4),
        _event(0.17, "update_end", version=4),
    ]
    annotate_bins(bins, events)
    assert [b.in_update for b in bins] == [True, True, False]
    assert bins[0].update_versions == [4]
    assert [e["event"] for e in bins[0].events] == ["update_start"]
    assert [e["event"] for e in bins[1].events] == ["update_end"]
    assert bins[2].events == []


# ---------------------------------------------------------------------------
# Top-level assembly
# ---------------------------------------------------------------------------


def test_build_timeline_end_to_end():
    counter = "sglang:realtime_tokens_total{mode=decode}"
    probe_records = []
    # Engine generates 10 tokens per 0.1s, pauses (scrape failures) during
    # [0.3, 0.5), resumes after.
    values = [0, 10, 20, None, None, 40, 50]
    for i, value in enumerate(values):
        t = i * 0.1
        if value is None:
            probe_records.append(_probe_record(t, ok=False, counter=counter))
        else:
            probe_records.append(_probe_record(t, value=float(value), counter=counter))
    events = [
        _event(0.31, "update_start", version=2, mode="adapter_double_buffer"),
        _event(0.47, "update_end", version=2, mode="adapter_double_buffer"),
    ]

    timeline = build_timeline(probe_records, events, counter=counter, bin_s=0.1)

    bins = timeline["bins"]
    assert len(bins) == 6
    assert bins[0]["tokens_per_s"] == pytest.approx(100.0)
    assert not bins[0]["has_gap"] and not bins[0]["in_update"]
    # Publication window bins: flagged in_update, scrape gaps recorded.
    assert bins[3]["in_update"] and bins[3]["has_gap"]
    assert bins[4]["in_update"] and bins[4]["has_gap"]
    # Tokens across the failure window survive via the counter bridge.
    assert sum(b["tokens"] for b in bins) == pytest.approx(50.0)
    assert timeline["windows"] == [
        {"t_start": 0.31, "t_end": 0.47, "weight_version": 2, "mode": "adapter_double_buffer"}
    ]
    assert set(timeline["per_engine"].keys()) == {"http://e0"}


def test_build_timeline_no_samples():
    timeline = build_timeline([], [_event(1.0, "update_start")], counter="c")
    assert timeline["bins"] == []
    assert timeline["per_engine"] == {}
    assert len(timeline["windows"]) == 1


def test_bin_dataclass_serializes():
    b = Bin(0.0, 0.1, tokens=5.0, covered_s=0.1)
    d = b.to_dict()
    json.dumps(d)  # must be JSON-serializable
    assert d["tokens_per_s"] == pytest.approx(50.0)


def test_default_bin_width_is_100ms():
    assert binning.DEFAULT_BIN_S == pytest.approx(0.1)
