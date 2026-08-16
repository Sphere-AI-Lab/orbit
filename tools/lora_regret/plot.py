"""Figures from `analyze --json`, one PNG per panel of the post.

A pure function of the ledgers: no network, no state, no side channel. The
input is the JSON document `analyze --json` writes, so a figure can never show
a number the analysis declined to quote -- an edge-of-grid argmin is absent
from the payload and is therefore absent from the plot.

Every `_draw_*` below reads the keys `analyze.py` actually writes into
`payload`, and `tests/fast/utils/test_lora_regret_plot.py` draws all six panels
from a fixture built out of those same blocks. That pairing is the point: a
panel reading an invented key passes every test that never draws it, then
KeyErrors on the one real payload it was written for.

matplotlib is imported inside `render` rather than at module scope so
`available_panels` and the CLI's argument handling stay importable in an
environment that has not installed it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Panel key -> the payload key it needs. A panel is drawn only when its data is
# present: an empty axes reads as "measured, and flat", which is a lie the
# reader has no way to detect.
PANELS: dict[str, str] = {
    "lr_vs_loss": "argmins",
    "learning_curves": "c1",
    "batch_size": "c3",
    "placement": "c4",
    "rl_accuracy": "c5",
    "short_run_multiplier": "c8",
}

# Figures to compare a panel against, by panel name.
#
# **Empty since 2026-08-02.** These pointed at
# `third_party/lora-without-regret/figures/`, which was a *community
# reproduction* of the blog post -- michaelbzhu/lora-without-regret, run on
# Qwen3-1.7B -- not the post's own output, and it was deleted for having been
# read as the post throughout the campaign. Its four PNGs were that repository
# author's plots.
#
# Left as an empty dict rather than deleted, because the consumer already
# handles "no reference for this panel" and prints nothing: a panel with no
# comparison is the honest state, while a `compare:` line pointing at someone
# else's reproduction is the failure that cost this campaign a re-plan. To
# restore the feature, put figures from the post itself here -- with a comment
# saying where each came from.
REFERENCE_FIGURES: dict[str, str] = {}


def _batch_points(payload: dict) -> list[dict]:
    """C3 rows that can be placed on a batch-size axis.

    `analyze.batch_gaps` groups on `record.get("global_batch_size")`, so any arm
    that left the batch at the launcher's default lands in a `None` group and is
    written out as `"global_batch_size": null`. Those are real measurements with
    no x position: plotting them would mean inventing the batch they ran at, and
    sorting them beside real ints raises outright. They are dropped here and
    counted by the caller instead.
    """
    return [row for row in payload.get("c3", []) if row.get("global_batch_size") is not None]


def available_panels(payload: dict) -> list[str]:
    """Panels whose data the payload actually carries, in PANELS order."""
    names = []
    for name, key in PANELS.items():
        if not payload.get(key):
            continue
        # C3 is the one panel whose rows can be present but unplottable -- see
        # _batch_points. An axes drawn from zero usable rows would read as
        # "measured, and flat", which is the failure this whole module avoids.
        if name == "batch_size" and not _batch_points(payload):
            continue
        names.append(name)
    return names


def _label(method: str, size) -> str:
    return "FullFT" if method == "full" else f"LoRA r{size}"


def _draw_lr_vs_loss(ax, payload: dict) -> None:
    rows = payload["argmins"]
    for row in sorted(rows, key=lambda r: (r["method"], r.get("size") or 0)):
        ax.scatter(row["lr"], row["test_nll"], label=_label(row["method"], row.get("size")))
    ax.set_xscale("log")
    ax.set_xlabel("argmin learning rate")
    ax.set_ylabel("held-out NLL (nats)")
    ax.set_title("Optimal LR by method and rank")
    ax.legend(fontsize="small")


def _draw_learning_curves(ax, payload: dict) -> None:
    rows = payload["c1"]
    names = [r["arm"] for r in rows]
    # `None` means "no departure within the budget", which is a different
    # statement from "departed at the last step" -- plot it at the budget and
    # mark it, rather than dropping the arm.
    values = [r["departure_step"] if r["departure_step"] is not None else r["step_budget"]
              for r in rows]
    colors = ["tab:blue" if r["departure_step"] is not None else "tab:grey" for r in rows]
    ax.barh(names, values, color=colors)
    ax.set_xlabel("departure step (grey = no departure within budget)")
    ax.set_title("Where each rank leaves the envelope")


def _draw_batch_size(ax, payload: dict) -> None:
    """C3's rows are one per (batch size, arm), so they are grouped into a line
    per arm here. The keys are `global_batch_size` and `delta_sigma`, which is
    what analyze's c3 block writes -- the claim is a gap that GROWS with batch,
    and one point per arm could not show a slope."""
    rows = _batch_points(payload)
    dropped = len(payload["c3"]) - len(rows)
    by_arm: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        by_arm.setdefault(row.get("arm", ""), []).append(
            (row["global_batch_size"], row["delta_sigma"])
        )
    for arm, points in sorted(by_arm.items()):
        points.sort()
        ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=arm)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(
        "global batch size"
        + (f"   ({dropped} unlabelled row(s) omitted)" if dropped else "")
    )
    ax.set_ylabel("best LoRA - best FullFT (sigma)")
    ax.set_title("Batch-size penalty")
    ax.legend(fontsize="small")


def _draw_placement(ax, payload: dict) -> None:
    """C4's payload is two labelled groups, not one flat dict: `attn_minus_mlp`
    is the claim that attention-only underperforms, `all_minus_mlp` the separate
    claim that all-modules adds nothing on top. They are read in opposite
    directions, so they are coloured apart rather than pooled into one series."""
    groups = payload["c4"]
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for group, color in (("attn_minus_mlp", "tab:blue"), ("all_minus_mlp", "tab:orange")):
        for label, delta in sorted(groups.get(group, {}).items()):
            labels.append(label)
            values.append(delta)
            colors.append(color)
    ax.bar(labels, values, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylabel("delta (sigma)")
    ax.set_title("Layer placement at matched parameters")
    ax.tick_params(axis="x", labelrotation=20, labelsize="small")


def _draw_rl_accuracy(ax, payload: dict) -> None:
    """C5 is two statements at once: peak parity and band width. The payload
    carries `peak_accuracy` and the band endpoints rather than a per-LR curve,
    so each arm is drawn as its band at the height of its peak -- the width IS
    the second half of the claim, and a bare peak marker would drop it."""
    for row in payload["c5"]:
        peak = row["peak_accuracy"]
        low, high = row["band_low"], row["band_high"]
        line, = ax.plot([low, high], [peak, peak], marker="|", linewidth=2,
                        label=f"{row.get('arm', '')} ({high / low:.0f}x wide)")
        ax.scatter([(low * high) ** 0.5], [peak], color=line.get_color(), zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("learning rate (bar spans the within-2-sigma band)")
    ax.set_ylabel("peak accuracy")
    ax.set_title("RL: peak parity and band width")
    ax.legend(fontsize="small")


def _draw_short_run_multiplier(ax, payload: dict) -> None:
    c8 = payload["c8"]
    ax.bar(["~100 steps", "long horizon"], [c8["short_ratio"], c8["long_ratio"]])
    ax.axhline(c8["predicted_short"], linestyle="--", label=f"post: {c8['predicted_short']:g}x")
    ax.axhline(c8["predicted_long"], linestyle=":", label=f"post: {c8['predicted_long']:g}x")
    ax.set_ylabel("argmin_LR(LoRA r256) / argmin_LR(FullFT)")
    ax.set_title("LR multiplier by horizon")
    ax.legend(fontsize="small")


_DRAW = {
    "lr_vs_loss": _draw_lr_vs_loss,
    "learning_curves": _draw_learning_curves,
    "batch_size": _draw_batch_size,
    "placement": _draw_placement,
    "rl_accuracy": _draw_rl_accuracy,
    "short_run_multiplier": _draw_short_run_multiplier,
}


def render(payload: dict, out_dir: Path) -> list[Path]:
    """Write one PNG per available panel. Returns the paths written."""
    panels = available_panels(payload)
    if not panels:
        return []
    import matplotlib

    matplotlib.use("Agg")  # no display on a compute node
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in panels:
        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        _DRAW[name](ax, payload)
        fig.tight_layout()
        path = out_dir / f"{name}.png"
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True,
                        help="the JSON document `analyze --json` wrote")
    parser.add_argument("--out", type=Path, default=Path("results/figures"))
    args = parser.parse_args(argv)

    payload = json.loads(args.analysis.read_text(encoding="utf-8"))
    written = render(payload, args.out)
    if not written:
        print(f"no plottable claims in {args.analysis}; nothing written")
        return 0
    for path in written:
        reference = REFERENCE_FIGURES.get(path.stem)
        suffix = f"   (compare: {reference})" if reference else ""
        print(f"wrote {path}{suffix}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
