#!/usr/bin/env python3
"""Render the A2 rollout-throughput timeline from probe + event JSONL.

One trace per invocation (one arm); overlaying arms is the caller's job
(run once per arm with --label, or import render() and compose). Update
windows are shaded; bins flagged has_gap are marked — a scrape gap IS
signal (engine unresponsive during an update).
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tools.rollout_timeline.binning import DEFAULT_BIN_S, build_timeline, load_jsonl  # noqa: E402

DEFAULT_COUNTER = "sglang:realtime_tokens_total{mode=decode}"


def render(probe_path: str, events_path: str, out_path: str, *,
           counter: str = DEFAULT_COUNTER, bin_s: float = DEFAULT_BIN_S,
           label: str = "") -> dict:
    timeline = build_timeline(load_jsonl(probe_path), load_jsonl(events_path),
                              counter=counter, bin_s=bin_s)
    bins = timeline["bins"]
    windows = timeline["windows"]

    fig, ax = plt.subplots(figsize=(10, 3.2))
    t0 = bins[0]["t_start"] if bins else 0.0
    xs = [(b["t_start"] + b["t_end"]) / 2.0 - t0 for b in bins]
    ys = [b["tokens_per_s"] for b in bins]
    ax.plot(xs, ys, lw=1.0, label=label or None)
    for b in bins:
        if b.get("has_gap"):
            ax.axvspan(b["t_start"] - t0, b["t_end"] - t0, color="0.85", zorder=0)
    for w in windows:
        if w.get("t_start") is not None and w.get("t_end") is not None:
            ax.axvspan(w["t_start"] - t0, w["t_end"] - t0, alpha=0.25, color="tab:red",
                       zorder=1, label="_update")
    ax.set_xlabel("wall time (s)")
    ax.set_ylabel("rollout tokens/s")
    if label:
        ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return {"n_bins": len(bins), "n_windows": len(windows),
            "gap_bins": sum(1 for b in bins if b.get("has_gap"))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--counter", default=DEFAULT_COUNTER)
    parser.add_argument("--bin-s", type=float, default=DEFAULT_BIN_S)
    parser.add_argument("--label", default="")
    args = parser.parse_args(argv)
    stats = render(args.probe, args.events, args.out,
                   counter=args.counter, bin_s=args.bin_s, label=args.label)
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
