"""Pure binning/merging logic for the rollout throughput timeline.

Consumes:
- probe JSONL (see probe.py): one record per scrape,
  ``{"t_wall": float, "engine_url": str, "ok": bool, "counters": {spec: value}}``
  (failed scrapes carry ``"ok": false`` and no counters — a scrape gap IS
  signal: the engine was unresponsive, e.g. paused for a weight update);
- events JSONL (emitted by the trainer via ORBIT_TIMELINE_EVENTS_FILE):
  ``{"t_wall": float, "event": "update_start"|"update_end",
  "weight_version": ..., "mode": ...}``.

Produces per-bin tokens/s series with gap flags and weight-publication
windows, ready for the figure script. Counter semantics:

- deltas are taken between consecutive SUCCESSFUL samples, so a missed scrape
  never loses tokens (cumulative counters are robust to missed reads); the
  tokens of an interval are spread uniformly over its wall-clock span;
- a counter DECREASE (engine restart / counter reset) contributes no tokens
  and marks the interval as a gap; the chain restarts from the reset sample;
- bins containing failed scrapes, reset intervals, or lying (partly) outside
  the sampled range are flagged ``has_gap``.

Stdlib only; no orbit imports.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

DEFAULT_BIN_S = 0.1


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file, skipping blank and corrupt lines (a run that is
    still writing may leave a truncated last line)."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


# ---------------------------------------------------------------------------
# Counter series extraction
# ---------------------------------------------------------------------------


def extract_engine_series(
    probe_records: list[dict],
    counter: str,
) -> dict[str, list[tuple[float, float | None]]]:
    """Group probe records by engine_url into time-sorted sample lists.

    Each sample is ``(t_wall, value)``; value is None for failed scrapes or
    scrapes where the counter was missing.
    """
    series: dict[str, list[tuple[float, float | None]]] = {}
    for record in probe_records:
        url = record.get("engine_url")
        t_wall = record.get("t_wall")
        if url is None or t_wall is None:
            continue
        value: float | None = None
        if record.get("ok"):
            counters = record.get("counters") or {}
            raw = counters.get(counter)
            if raw is not None:
                value = float(raw)
        series.setdefault(url, []).append((float(t_wall), value))
    for samples in series.values():
        samples.sort(key=lambda s: s[0])
    return series


@dataclass
class Interval:
    t_start: float
    t_end: float
    tokens: float
    is_reset: bool = False


def counter_intervals(
    samples: list[tuple[float, float | None]],
) -> tuple[list[Interval], list[float]]:
    """Turn one engine's samples into token intervals + failed-scrape times.

    Intervals connect consecutive successful samples (missed scrapes in
    between do not break the chain — the cumulative counter preserves the
    tokens). A counter decrease yields a zero-token interval flagged
    ``is_reset``.
    """
    intervals: list[Interval] = []
    failures: list[float] = []
    prev: tuple[float, float] | None = None
    for t_wall, value in samples:
        if value is None:
            failures.append(t_wall)
            continue
        if prev is not None and t_wall > prev[0]:
            delta = value - prev[1]
            if delta >= 0:
                intervals.append(Interval(prev[0], t_wall, delta))
            else:
                intervals.append(Interval(prev[0], t_wall, 0.0, is_reset=True))
        prev = (t_wall, value)
    return intervals, failures


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------


@dataclass
class Bin:
    t_start: float
    t_end: float
    tokens: float = 0.0
    covered_s: float = 0.0
    has_gap: bool = False
    scrape_failures: int = 0
    events: list[dict] = field(default_factory=list)
    in_update: bool = False
    update_versions: list[Any] = field(default_factory=list)

    @property
    def tokens_per_s(self) -> float | None:
        if self.covered_s <= 0:
            return None
        return self.tokens / self.covered_s

    def to_dict(self) -> dict:
        return {
            "t_start": self.t_start,
            "t_end": self.t_end,
            "tokens": self.tokens,
            "covered_s": self.covered_s,
            "tokens_per_s": self.tokens_per_s,
            "has_gap": self.has_gap,
            "scrape_failures": self.scrape_failures,
            "events": self.events,
            "in_update": self.in_update,
            "update_versions": self.update_versions,
        }


def make_bin_edges(t_start: float, t_end: float, bin_s: float = DEFAULT_BIN_S) -> list[float]:
    """Edges of contiguous bins of width ``bin_s`` covering [t_start, t_end]."""
    if t_end <= t_start:
        return [t_start, t_start + bin_s]
    n = max(1, math.ceil((t_end - t_start) / bin_s - 1e-9))
    return [t_start + i * bin_s for i in range(n + 1)]


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def bin_engine_samples(
    samples: list[tuple[float, float | None]],
    bin_edges: list[float],
) -> list[Bin]:
    """Bin one engine's counter samples into tokens / covered seconds.

    Interval tokens are spread uniformly over the interval's span; reset
    intervals contribute gap-time instead of tokens. Bin time not covered by
    any (non-reset) interval — before the first sample, after the last, or
    under failed scrapes at the chain edges — leaves ``covered_s`` short and
    flags the bin as a gap.
    """
    bins = [Bin(bin_edges[i], bin_edges[i + 1]) for i in range(len(bin_edges) - 1)]
    intervals, failures = counter_intervals(samples)

    for interval in intervals:
        span = interval.t_end - interval.t_start
        if span <= 0:
            continue
        for b in bins:
            overlap = _overlap(interval.t_start, interval.t_end, b.t_start, b.t_end)
            if overlap <= 0:
                continue
            if interval.is_reset:
                b.has_gap = True
            else:
                b.tokens += interval.tokens * (overlap / span)
                b.covered_s += overlap

    for t_fail in failures:
        for b in bins:
            if b.t_start <= t_fail < b.t_end:
                b.scrape_failures += 1
                b.has_gap = True

    epsilon = 1e-9
    for b in bins:
        if b.covered_s + epsilon < (b.t_end - b.t_start):
            b.has_gap = True
    return bins


def combine_engine_bins(per_engine_bins: dict[str, list[Bin]]) -> list[Bin]:
    """Sum aligned per-engine bins into a cluster-level series.

    ``tokens`` sum; ``covered_s`` averages across engines so ``tokens_per_s``
    stays the SUM of per-engine rates; a gap in any engine flags the combined
    bin.
    """
    if not per_engine_bins:
        return []
    engine_lists = list(per_engine_bins.values())
    n_bins = min(len(bins) for bins in engine_lists)
    n_engines = len(engine_lists)
    combined = []
    for i in range(n_bins):
        first = engine_lists[0][i]
        merged = Bin(first.t_start, first.t_end)
        for bins in engine_lists:
            b = bins[i]
            merged.tokens += b.tokens
            merged.covered_s += b.covered_s / n_engines
            merged.has_gap = merged.has_gap or b.has_gap
            merged.scrape_failures += b.scrape_failures
        combined.append(merged)
    return combined


# ---------------------------------------------------------------------------
# Weight-publication events
# ---------------------------------------------------------------------------


def update_windows(event_records: list[dict]) -> list[dict]:
    """Pair update_start/update_end markers into publication windows.

    Pairing is by weight_version when present, else first-in-first-out. An
    unmatched start yields an open window (``t_end`` None) — e.g. a run
    killed mid-publication.
    """
    windows: list[dict] = []
    open_by_version: dict[Any, dict] = {}
    open_fifo: list[dict] = []
    for record in sorted(event_records, key=lambda r: r.get("t_wall", 0.0)):
        event = record.get("event")
        version = record.get("weight_version")
        if event == "update_start":
            window = {
                "t_start": record.get("t_wall"),
                "t_end": None,
                "weight_version": version,
                "mode": record.get("mode"),
            }
            windows.append(window)
            if version is not None:
                open_by_version[version] = window
            else:
                open_fifo.append(window)
        elif event == "update_end":
            window = None
            if version is not None and version in open_by_version:
                window = open_by_version.pop(version)
            elif open_fifo:
                window = open_fifo.pop(0)
            if window is not None:
                window["t_end"] = record.get("t_wall")
    return windows


def annotate_bins(bins: list[Bin], event_records: list[dict]) -> list[Bin]:
    """Attach raw events to their bins and flag bins inside update windows."""
    for record in event_records:
        t_wall = record.get("t_wall")
        if t_wall is None:
            continue
        for b in bins:
            if b.t_start <= t_wall < b.t_end:
                b.events.append(record)
    for window in update_windows(event_records):
        w_start = window["t_start"]
        w_end = window["t_end"]
        if w_start is None:
            continue
        if w_end is None:
            w_end = float("inf")
        for b in bins:
            if _overlap(w_start, w_end, b.t_start, b.t_end) > 0:
                b.in_update = True
                if window["weight_version"] is not None:
                    b.update_versions.append(window["weight_version"])
    return bins


# ---------------------------------------------------------------------------
# Top-level assembly
# ---------------------------------------------------------------------------


def build_timeline(
    probe_records: list[dict],
    event_records: list[dict],
    *,
    counter: str,
    bin_s: float = DEFAULT_BIN_S,
) -> dict:
    """probe + event records -> binned tokens/s timeline with annotations.

    Returns ``{"bins": [dict...], "per_engine": {url: [dict...]},
    "windows": [dict...]}`` — everything JSON-serializable for the figure
    script.
    """
    series = extract_engine_series(probe_records, counter)
    all_times = [t for samples in series.values() for t, _ in samples]
    if not all_times:
        return {"bins": [], "per_engine": {}, "windows": update_windows(event_records)}
    edges = make_bin_edges(min(all_times), max(all_times), bin_s)
    per_engine = {url: bin_engine_samples(samples, edges) for url, samples in series.items()}
    combined = combine_engine_bins(per_engine)
    annotate_bins(combined, event_records)
    return {
        "bins": [b.to_dict() for b in combined],
        "per_engine": {url: [b.to_dict() for b in bins] for url, bins in per_engine.items()},
        "windows": update_windows(event_records),
    }
