"""The NLL curve behind C1's departure step.

parse_final_nll answers "what did this arm score"; parse_trace answers "how did
it get there". The fixture is the real 2026-07-30 smoke's three eval lines, not
synthesized text, so a parser that only satisfies its own format string fails
here.
"""

from pathlib import Path

from tools.lora_regret.trace import (
    PHASE_AFTER_TRAIN,
    PHASE_BEFORE_TRAIN,
    NllPoint,
    parse_trace,
    parse_trace_file,
    trace_is_consistent,
)

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "lora_regret"
    / "smoke_lora_r256_eval_lines.log"
)


class TestParseTrace:
    def test_parses_the_real_smoke_log(self):
        points = parse_trace_file(FIXTURE)
        assert [p.nll for p in points] == [1.209810, 1.199709, 1.194836]
        assert [p.phase for p in points] == [
            PHASE_BEFORE_TRAIN,
            PHASE_AFTER_TRAIN,
            PHASE_AFTER_TRAIN,
        ]
        assert [p.step for p in points] == [0, 0, 1]

    def test_carries_every_field(self):
        first = parse_trace_file(FIXTURE)[0]
        assert first == NllPoint(
            rollout_id=0,
            step=0,
            phase=PHASE_BEFORE_TRAIN,
            nll=1.209810,
            sample_mean=1.478078,
            tokens=308760,
            samples=1000,
        )

    def test_before_train_sorts_ahead_of_after_train_at_the_same_step(self):
        """The base-model measurement precedes the post-step one at step 0.

        Multi-rank log buffering can place them in either physical order, so the
        ordering must come from (step, phase), not from file position.
        """
        text = "\n".join(reversed(FIXTURE.read_text().splitlines()))
        points = parse_trace(text)
        assert [p.phase for p in points[:2]] == [PHASE_BEFORE_TRAIN, PHASE_AFTER_TRAIN]

    def test_a_log_with_no_eval_lines_is_an_empty_trace(self):
        assert parse_trace("Traceback (most recent call last):\n  boom\n") == []


class TestTraceIsConsistent:
    def _point(self, step, nll, samples=1000, tokens=308760):
        return NllPoint(step, step, PHASE_AFTER_TRAIN, nll, nll, tokens, samples)

    def test_accepts_a_constant_held_out_set(self):
        ok, why = trace_is_consistent([self._point(0, 1.2), self._point(1, 1.1)])
        assert ok, why

    def test_rejects_a_shrinking_sample_count(self):
        """1000 rows at global batch 32 silently becoming 992 is floor division.

        That makes the metric depend on batch size, which is exactly what E2
        varies -- so it must be caught, not averaged over.
        """
        ok, why = trace_is_consistent(
            [self._point(0, 1.2, samples=1000), self._point(1, 1.1, samples=992)]
        )
        assert not ok
        assert "992" in why and "1000" in why

    def test_rejects_a_changing_token_count(self):
        ok, why = trace_is_consistent(
            [self._point(0, 1.2, tokens=308760), self._point(1, 1.1, tokens=306000)]
        )
        assert not ok

    def test_rejects_an_empty_trace(self):
        ok, why = trace_is_consistent([])
        assert not ok
        assert "empty" in why


class TestSweepSharesOneRegex:
    def test_sweep_reuses_the_trace_regex_object(self):
        """One definition, pinned to EVAL_NLL_METRIC_KEY. Not two copies."""
        from tools.lora_regret import sweep, trace

        assert sweep._NLL_LINE is trace.NLL_LINE
