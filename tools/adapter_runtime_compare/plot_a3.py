#!/usr/bin/env python3
"""Draw the A3 figures from ``analyze_a3.py --json`` output.

Three panels, one shared reward axis for the first two (never a dual axis):

1. reward vs samples   -- parity: arms should coincide within the seed band;
2. reward vs wall-clock -- speedup: the same curves against training wall-clock,
   x = seed-mean wall-clock at each rollout (each arm has its own clock);
3. train/rollout log-prob gap vs samples -- the train/inference mismatch per
   arm with the qualified envelope, so a parity claim is read next to the
   evidence that both stacks agree about the sampling policy.

Curves are the across-seed mean of a centred moving average over ``--smooth``
rollouts (raw per-rollout rewards on 32-sample batches are noise-dominated);
the band is ± one across-seed standard deviation of that smoothed value. Colors
follow the fixed categorical order of the reference palette (sync = blue,
async_fullft = orange, async_db = aqua), assigned by arm identity, never by
rank; lines are direct-labeled at their ends and a legend is present.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

# Reference palette, validated with dataviz/scripts/validate_palette.js on the light
# surface (#fcfcfb): lightness band, chroma floor, CVD and normal-vision separation all
# PASS; aqua sits below 3:1 contrast, which the direct labels and the tables relieve.
ARM_COLORS = {"sync": "#2a78d6", "async_fullft": "#eb6834", "async_db": "#1baf7a", "async": "#eda100"}
ARM_LABELS = {
    "sync": "sync OFT (colocated)",
    "async_fullft": "async full-FT",
    "async_db": "async OFT, double-buffer",
    "async": "async OFT, single slot",
}
# End-of-line direct labels: short so they fit in the right margin of each panel.
ARM_SHORT = {"sync": "sync OFT", "async_fullft": "full-FT", "async_db": "async OFT", "async": "async 1-slot"}
ARM_ORDER = ("sync", "async_fullft", "async_db", "async")
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def moving_average(values: list[float], window: int) -> list[float]:
    """Centred moving average, shrinking the window at both ends (no NaN padding)."""
    half = max(window // 2, 0)
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        out.append(statistics.mean(values[lo:hi]))
    return out


def arm_curves(series: list[dict[str, Any]], *, smooth: int) -> dict[str, dict[str, list[float]]]:
    """Per arm: samples, seed-mean wall-clock, mean/std of smoothed reward, mean/std of the gap.

    Only the rollouts every seed of the arm reached are used, so the band is a true
    across-seed spread at each x.
    """
    curves: dict[str, dict[str, list[float]]] = {}
    for mode in ARM_ORDER:
        runs = [r for r in series if r["mode"] == mode and r["rollouts"]]
        if not runs:
            continue
        n = min(len(r["rollouts"]) for r in runs)
        smoothed = [moving_average([p["reward"] for p in r["rollouts"][:n]], smooth) for r in runs]
        gaps = [[p.get("logprob_gap") for p in r["rollouts"][:n]] for r in runs]
        walls = [[p.get("wall_s") for p in r["rollouts"][:n]] for r in runs]
        curve = {"samples": [runs[0]["rollouts"][i]["samples"] for i in range(n)], "wall_s": [], "reward": [],
                 "reward_std": [], "gap": [], "gap_std": []}
        for i in range(n):
            col = [s[i] for s in smoothed]
            curve["reward"].append(statistics.mean(col))
            curve["reward_std"].append(statistics.stdev(col) if len(col) > 1 else 0.0)
            w = [x[i] for x in walls if x[i] is not None]
            curve["wall_s"].append(statistics.mean(w) if w else float("nan"))
            g = [x[i] for x in gaps if x[i] is not None]
            curve["gap"].append(statistics.mean(g) if g else float("nan"))
            curve["gap_std"].append(statistics.stdev(g) if len(g) > 1 else 0.0)
        curve["n_seeds"] = [len(runs)]
        curves[mode] = curve
    return curves


def _style_axis(ax, *, xlabel: str, ylabel: str | None) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, axis="y", color=GRID, linewidth=0.8)
    ax.grid(False, axis="x")
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=10)


def _direct_label(ax, x: float, y: float, text: str, color: str) -> None:
    ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", fontsize=9, color=INK, va="center",
                bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": color, "lw": 0.8})


def draw(curves: dict[str, dict[str, list[float]]], *, lp_gap_envelope: float, title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE, gridspec_kw={"wspace": 0.32})
    ax_samples, ax_wall, ax_gap = axes
    for mode, c in curves.items():
        color, label = ARM_COLORS[mode], f"{ARM_LABELS[mode]} (n={c['n_seeds'][0]})"
        for ax, xs in ((ax_samples, c["samples"]), (ax_wall, c["wall_s"])):
            lo = [m - s for m, s in zip(c["reward"], c["reward_std"], strict=True)]
            hi = [m + s for m, s in zip(c["reward"], c["reward_std"], strict=True)]
            ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.plot(xs, c["reward"], color=color, linewidth=2, label=label)
            _direct_label(ax, xs[-1], c["reward"][-1], ARM_SHORT[mode], color)
    for ax, key in ((ax_samples, "samples"), (ax_wall, "wall_s")):
        xmax = max(max(c[key]) for c in curves.values())
        ax.set_xlim(0, xmax * 1.25)  # room for the end-of-line labels inside the panel
        glo = [m - s for m, s in zip(c["gap"], c["gap_std"], strict=True)]
        ghi = [m + s for m, s in zip(c["gap"], c["gap_std"], strict=True)]
        ax_gap.fill_between(c["samples"], glo, ghi, color=color, alpha=0.15, linewidth=0)
        ax_gap.plot(c["samples"], c["gap"], color=color, linewidth=2, label=label)
    ax_gap.axhline(lp_gap_envelope, color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax_gap.annotate(f"envelope {lp_gap_envelope:g}", (0, lp_gap_envelope), xytext=(4, 4), textcoords="offset points",
                    fontsize=8, color=INK_2)
    _style_axis(ax_samples, xlabel="samples", ylabel="raw reward (smoothed)")
    _style_axis(ax_wall, xlabel="training wall-clock, s (seed mean)", ylabel=None)
    _style_axis(ax_gap, xlabel="samples", ylabel="train/rollout |Δ log p| per step")
    ax_wall.sharey(ax_samples)
    ax_gap.set_ylim(bottom=0)
    ax_samples.set_title("reward vs samples (parity)", color=INK, fontsize=11, loc="left")
    ax_wall.set_title("reward vs wall-clock (speedup)", color=INK, fontsize=11, loc="left")
    ax_gap.set_title("train/inference mismatch", color=INK, fontsize=11, loc="left")
    handles, labels = ax_samples.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.04), labelcolor=INK)
    fig.suptitle(title, color=INK, fontsize=12, x=0.01, ha="left")
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("series_json", help="output of analyze_a3.py --json")
    parser.add_argument("--out", required=True, help="output path without extension; writes .svg and .png")
    parser.add_argument("--smooth", type=int, default=10, help="moving-average window in rollouts")
    parser.add_argument("--lp-gap-envelope", type=float, default=0.014)
    parser.add_argument("--title", default="A3 — Qwen3-4B, GRPO on OpenR1 math, 3 seeds")
    args = parser.parse_args(argv)

    series = json.loads(Path(args.series_json).read_text())
    curves = arm_curves(series, smooth=args.smooth)
    if not curves:
        print("no complete runs in the series file", file=sys.stderr)
        return 1
    fig = draw(curves, lp_gap_envelope=args.lp_gap_envelope, title=args.title)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=160, bbox_inches="tight", facecolor=SURFACE)
    print(f"wrote {out.with_suffix('.svg')} and {out.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
