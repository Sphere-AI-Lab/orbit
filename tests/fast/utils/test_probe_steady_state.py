"""The steady-state estimate must exclude one-off costs, and price them separately.

Measured on 2026-08-01, the FullFT RL probe recorded rollouts
`[308.0, 59.0, 677.0]`:

    rollout 0   308s   cold start + the probe's forced per-rollout eval
    rollout 1    59s   steady
    rollout 2   677s   steady + a 616.5s checkpoint write (15 GB to Lustre)

`statistics.median` over `[59, 677]` is 368 -- on two samples a median IS the
mean, so the checkpoint leaked straight into the per-rollout figure and the
campaign estimate came out at 931 h against a true ~453 h. The same distortion
hit every OFT row (`median(115, 266) = 190.5`); LoRA all-modules escaped only
because its adapter checkpoint is negligible.

Two fixes, and both are needed. Taking the MINIMUM makes the estimator immune to
any one-off that lands in a single rollout -- compile, eval, allocator growth,
checkpoint -- on the reasoning that a rollout's time is a fixed steady cost plus
optional extras, so the cheapest observed rollout is the least contaminated.

But removing the checkpoint from `steady` would then drop it from the estimate
altogether, and real arms do checkpoint: the launcher's `SAVE_INTERVAL` is 50,
so a 500-rollout arm writes 10 of them while the probe wrote 1. So the saves are
priced explicitly instead of being smeared across every rollout.
"""

from __future__ import annotations

import pytest

from tools.lora_regret.probe import steady_seconds


def _record(**overrides) -> dict:
    """The FullFT RL probe row as actually written on 2026-08-01, minus the
    fields the report does not read. Built from the real shape rather than
    invented, so the report's own keying (matrix/method/dataset/target) matches
    a planned run instead of falling through to "not run"."""
    from tools.lora_regret.probe import probe_plan

    # Derived, not typed: the report keys on (matrix, arm), so a hardcoded name
    # that drifts from the plan would silently report "not run" instead of
    # failing loudly.
    arm = next(r.arm for r in probe_plan("method") if r.matrix == "e4" and r.method == "full")
    record = {
        "matrix": "e4", "arm": arm, "method": "full",
        "dataset": "math_gsm8k", "target_modules": "", "status": "ok",
        "seconds": 1604.0, "rollout_seconds": [308.0, 59.0, 677.0],
        "gpus": 8, "probe_rollouts": 3, "metric": "accuracy",
    }
    record.update(overrides)
    return record


def _e4_full_row(text: str) -> str:
    """The report prints every matrix, most of them "not run". Pick the one row
    under test rather than indexing into the block by position."""
    for line in text.splitlines():
        if line.startswith("e4 ") and " full " in line:
            return line
    raise AssertionError(f"no e4/full row in:\n{text}")


class TestSteadyIgnoresOneOffCosts:
    def test_the_checkpoint_rollout_does_not_move_it(self):
        """The case that motivated this. 59, not 368."""
        assert steady_seconds([308.0, 59.0, 677.0]) == 59.0

    def test_the_first_rollout_is_still_dropped(self):
        """Rollout 0 carries compile, weight load and the first allocator
        growth. It was already excluded and must stay excluded -- if it were
        the cheapest rollout, a naive minimum would pick it."""
        assert steady_seconds([10.0, 90.0, 92.0]) == 90.0

    def test_a_clean_run_is_unchanged(self):
        """Where nothing is contaminated, min and median agree, so no
        previously-correct row moves."""
        assert steady_seconds([373.0, 89.0, 89.0]) == 89.0

    def test_one_steady_rollout_still_yields_an_estimate(self):
        """A 2-rollout probe leaves a single sample. It is worse than two, but
        refusing to report it would lose the row entirely."""
        assert steady_seconds([300.0, 95.0]) == 95.0

    def test_no_steady_rollout_yields_nothing(self):
        """One rollout is all cold start. Inventing a steady figure from it
        would be reporting the number the probe exists to measure."""
        assert steady_seconds([300.0]) is None
        assert steady_seconds([]) is None


class TestSavesArePricedSeparately:
    def test_the_launcher_save_interval_is_the_one_the_estimate_uses(self):
        """Pinned against the launcher rather than retyped: if the cadence
        changes there, this estimate is silently wrong until it changes here."""
        from pathlib import Path

        from tools.lora_regret.probe import SAVE_INTERVAL

        launcher = (
            Path(__file__).resolve().parents[3]
            / "examples/high_precision/run-llama3_1-8b-bf16-rl-math-gsm8k.sh"
        )
        assert f'--save-interval "${{SAVE_INTERVAL:-{SAVE_INTERVAL}}}"' in launcher.read_text(
            encoding="utf-8"
        )

    @pytest.mark.parametrize(
        "full_rollouts,probe_rollouts,expected",
        [
            (500, 3, 9),    # 10 saves in the real arm, 1 already in the probe
            (50, 3, 0),     # one save, which the probe already paid for
            (2000, 3, 39),
        ],
    )
    def test_extra_saves_beyond_the_one_the_probe_paid(
        self, full_rollouts, probe_rollouts, expected
    ):
        """The probe's own elapsed time already contains one checkpoint, and
        that lands in `overhead`. Only the ADDITIONAL ones a longer arm writes
        get added, or the first would be counted twice."""
        from tools.lora_regret.probe import extra_saves

        assert extra_saves(full_rollouts, probe_rollouts) == expected

    def test_a_ledger_without_save_timings_still_estimates(self):
        """Rows written before save timings were recorded must not vanish from
        the report. They fall back to the old behaviour -- one save, priced
        inside `overhead` -- and the estimate is low rather than absent."""
        from tools.lora_regret.probe import format_report

        record = _record()
        row = _e4_full_row(format_report([record]))
        assert "59.0s" in row, row
        assert "?" not in row, row


class TestTheReportedNumbersMoved:
    def test_the_fullft_row_no_longer_shows_the_checkpoint(self):
        from tools.lora_regret.probe import format_report

        record = _record(save_seconds=[616.5])
        row = _e4_full_row(format_report([record]))
        assert "59.0s" in row, row
        assert "368" not in row, row
