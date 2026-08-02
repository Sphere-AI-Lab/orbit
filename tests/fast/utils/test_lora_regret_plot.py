"""plot.py must be a pure function of the ledgers: no network, no state."""

import json
from pathlib import Path

import pytest

from tools.lora_regret.plot import PANELS, available_panels, render

ALL = "linear_qkv,linear_proj,linear_fc1,linear_fc2"


def _payload():
    """Deliberately carries no `c3`, so the absent-panel path is exercised."""
    return {
        "command": "all",
        "sigma": 0.000992,
        "argmins": [
            {"arm": "full", "method": "full", "size": None, "target_modules": "",
             "lr": 2.5e-5, "test_nll": 1.00, "lr_grid": [1.5e-5, 2.5e-5, 4e-5], "edge_of_grid": False},
            {"arm": "lora r256", "method": "lora", "size": 256, "target_modules": ALL,
             "lr": 2.5e-4, "test_nll": 1.05, "lr_grid": [1.5e-4, 2.5e-4, 4e-4], "edge_of_grid": False},
            {"arm": "lora r16", "method": "lora", "size": 16, "target_modules": ALL,
             "lr": 2.5e-4, "test_nll": 1.12, "lr_grid": [1.5e-4, 2.5e-4, 4e-4], "edge_of_grid": False},
        ],
        "c2": {"lora_r256_argmin_lr": 2.5e-4, "fullft_argmin_lr": 2.5e-5, "ratio": 10.0},
        "c8": {"long_ratio": 10.0, "short_ratio": 15.0, "upholds": True,
               "predicted_long": 9.8, "predicted_short": 15.0},
        "c1": [
            {"arm": "lora-r1-all", "departure_step": 400, "step_budget": 2000},
            {"arm": "lora-r256-all", "departure_step": None, "step_budget": 2000},
        ],
    }


def _full_payload():
    """Every payload key `analyze --json` can emit, in the shape it emits.

    Separate from `_payload` because that one's value is the *absence* of c3.
    Without this second fixture the c3, c4 and c5 panels would never be drawn
    by any test, which is how a panel comes to read a key analyze never writes.
    """
    payload = _payload()
    payload["c3"] = [
        {"global_batch_size": 32, "arm": "lora r256 all", "delta_sigma": 0.4},
        {"global_batch_size": 128, "arm": "lora r256 all", "delta_sigma": 2.1},
        {"global_batch_size": 512, "arm": "lora r256 all", "delta_sigma": 5.8},
        {"global_batch_size": 32, "arm": "lora r16 all", "delta_sigma": 0.9},
        {"global_batch_size": 128, "arm": "lora r16 all", "delta_sigma": 2.6},
        {"global_batch_size": 512, "arm": "lora r16 all", "delta_sigma": 6.3},
    ]
    payload["c4"] = {
        "attn_minus_mlp": {"attn(r256) - mlp(r92)": 3.4, "attn(r256) - mlp(r128)": 2.9},
        "all_minus_mlp": {"all(r256) - mlp(r92)": -0.3},
    }
    payload["c5"] = [
        {"arm": "full", "peak_accuracy": 0.51, "band_low": 1e-6, "band_high": 3.16e-6,
         "sigma_measured": False},
        {"arm": "lora r1 all", "peak_accuracy": 0.50, "band_low": 1e-5, "band_high": 1e-4,
         "sigma_measured": False},
    ]
    return payload


def test_the_fixture_matches_what_analyze_actually_emits():
    """The payload keys are copied from analyze.py's own `payload[...]` blocks.
    A fixture that invents a key lets plot.py pass its tests and KeyError on the
    real pipeline -- which is the whole failure this file exists to prevent.

    Every panel is covered, not only `argmins`: a panel whose data the fixtures
    never carry is a panel no test has ever drawn."""
    payload = _full_payload()
    assert set(payload["argmins"][0]) >= {
        "arm", "method", "size", "target_modules", "lr", "test_nll",
        "lr_grid", "edge_of_grid",
    }
    assert set(payload["c1"][0]) >= {"arm", "departure_step", "step_budget"}
    assert set(payload["c3"][0]) >= {"global_batch_size", "arm", "delta_sigma"}
    assert set(payload["c4"]) == {"attn_minus_mlp", "all_minus_mlp"}
    assert set(payload["c5"][0]) >= {"arm", "peak_accuracy", "band_low", "band_high"}
    assert set(payload["c8"]) >= {"long_ratio", "short_ratio", "predicted_long", "predicted_short"}
    # And every panel PANELS declares has data here, so render() below draws
    # all of them rather than silently skipping the ones with a wrong key.
    assert set(available_panels(payload)) == set(PANELS)


def test_available_panels_reports_only_what_the_payload_supports():
    """A payload with no c3 must not produce an empty batch-size figure -- an
    axes with no data reads as 'measured, and flat'."""
    payload = _payload()
    panels = available_panels(payload)
    assert "lr_vs_loss" in panels
    assert "short_run_multiplier" in panels
    assert "batch_size" not in panels


def test_render_writes_one_png_per_available_panel(tmp_path):
    payload = _payload()
    written = render(payload, tmp_path)
    assert len(written) == len(available_panels(payload))
    assert all(p.exists() and p.suffix == ".png" and p.stat().st_size > 0 for p in written)


def test_render_draws_every_panel_from_a_complete_payload(tmp_path):
    """The c3, c4 and c5 panels read nested and differently-named keys than the
    others; drawing them is the only way to find out whether they read the ones
    analyze writes."""
    written = render(_full_payload(), tmp_path)
    assert {p.stem for p in written} == set(PANELS)
    assert all(p.stat().st_size > 0 for p in written)


class TestUnlabelledBatchRows:
    """`analyze.batch_gaps` groups on `record.get("global_batch_size")`, so an
    arm that left the batch at the launcher's default is emitted with
    `"global_batch_size": null`. Found by running a real `analyze all --json`
    through the CLI, not by inspection -- sorting one of those beside an int
    raises `TypeError: '<' not supported between 'int' and 'NoneType'`."""

    @staticmethod
    def _with_c3(rows):
        payload = _payload()
        payload["c3"] = rows
        return payload

    def test_a_mixed_payload_plots_the_labelled_rows_and_drops_the_rest(self, tmp_path):
        payload = self._with_c3([
            {"global_batch_size": None, "arm": "lora r4 all", "delta_sigma": 300.0},
            {"global_batch_size": 32, "arm": "lora r256 all", "delta_sigma": 0.4},
            {"global_batch_size": 512, "arm": "lora r256 all", "delta_sigma": 5.8},
        ])
        assert "batch_size" in available_panels(payload)
        written = render(payload, tmp_path)
        assert (tmp_path / "batch_size.png") in written

    def test_an_entirely_unlabelled_c3_draws_no_panel_at_all(self, tmp_path):
        """Zero usable rows must not produce an empty axes: that reads as
        'measured, and flat', which the reader cannot tell from the truth."""
        payload = self._with_c3([
            {"global_batch_size": None, "arm": "lora r4 all", "delta_sigma": 300.0},
        ])
        assert "batch_size" not in available_panels(payload)
        render(payload, tmp_path)
        assert not (tmp_path / "batch_size.png").exists()


def test_render_is_idempotent(tmp_path):
    payload = _payload()
    first = render(payload, tmp_path)
    second = render(payload, tmp_path)
    assert sorted(first) == sorted(second)
    assert len(list(tmp_path.glob("*.png"))) == len(first)


def test_empty_payload_writes_nothing_and_does_not_raise(tmp_path):
    assert render({"command": "sigma"}, tmp_path) == []
    assert list(tmp_path.glob("*.png")) == []


def test_cli_reads_a_json_file(tmp_path):
    import subprocess
    import sys

    repo_root = Path(__file__).resolve().parents[3]
    payload_path = tmp_path / "analysis.json"
    payload_path.write_text(json.dumps(_payload()), encoding="utf-8")
    out = tmp_path / "figures"
    proc = subprocess.run(
        [sys.executable, "-m", "tools.lora_regret.plot",
         "--analysis", str(payload_path), "--out", str(out)],
        capture_output=True, text=True, cwd=repo_root,
    )
    assert proc.returncode == 0, proc.stderr
    assert list(out.glob("*.png"))


def test_no_reference_figure_points_at_a_community_reproduction():
    """`third_party/lora-without-regret` was michaelbzhu's reproduction on
    Qwen3-1.7B, not the blog post's own output, and reading it as the post
    mis-set the RL schedule and the FullFT learning-rate grid before anyone
    noticed. It is deleted; nothing may point back into it."""
    from tools.lora_regret.plot import REFERENCE_FIGURES

    repo_root = Path(__file__).resolve().parents[3]
    assert not (repo_root / "third_party" / "lora-without-regret").exists()
    assert not any("third_party" in path for path in REFERENCE_FIGURES.values())


def test_every_reference_figure_named_by_a_panel_exists():
    """A stale REFERENCE_FIGURES entry prints a `compare:` path that is not
    there, which is worse than printing nothing. Vacuous while the dict is
    empty, and that is the point: it is what makes refilling it safe."""
    from tools.lora_regret.plot import REFERENCE_FIGURES

    repo_root = Path(__file__).resolve().parents[3]
    for panel, relative in REFERENCE_FIGURES.items():
        assert panel in PANELS, panel
        assert (repo_root / relative).is_file(), relative



@pytest.mark.parametrize("panel", sorted(PANELS))
def test_every_panel_has_a_drawing_function(panel):
    from tools.lora_regret.plot import _DRAW

    assert panel in _DRAW
