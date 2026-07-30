"""P3: the DP>1 held-out NLL reduction must equal the DP=1 answer.

A differing `tokens` means the reduction double-counts or drops a shard, and no
amount of averaging fixes the FullFT numbers downstream -- so this exits
non-zero rather than warning.
"""

from tools.lora_regret.p3_check import compare_traces
from tools.lora_regret.trace import PHASE_AFTER_TRAIN, PHASE_BEFORE_TRAIN, NllPoint


def _point(step, nll, phase=PHASE_AFTER_TRAIN, tokens=308760, samples=1000):
    return NllPoint(step, step, phase, nll, nll + 0.2, tokens, samples)


class TestCompareTraces:
    def test_identical_traces_compare_equal(self):
        trace = [_point(0, 1.209810, PHASE_BEFORE_TRAIN), _point(1, 1.194836)]
        assert compare_traces(trace, list(trace)) == []

    def test_a_differing_nll_is_reported_with_both_values(self):
        a = [_point(1, 1.194836)]
        b = [_point(1, 1.194837)]
        problems = compare_traces(a, b)
        assert len(problems) == 1
        assert "1.194836" in problems[0] and "1.194837" in problems[0]

    def test_a_differing_token_count_names_the_shard_failure(self):
        a = [_point(1, 1.194836, tokens=308760)]
        b = [_point(1, 1.194836, tokens=617520)]
        problems = compare_traces(a, b)
        assert len(problems) == 1
        assert "tokens" in problems[0]
        assert "shard" in problems[0]

    def test_nll_equality_is_to_six_decimals_not_exact_float(self):
        """The logs print %.6f, so comparing beyond six decimals compares noise."""
        assert compare_traces([_point(1, 1.1948360000001)], [_point(1, 1.194836)]) == []

    def test_a_missing_measurement_is_a_problem_not_a_silent_skip(self):
        a = [_point(0, 1.2, PHASE_BEFORE_TRAIN), _point(1, 1.1)]
        b = [_point(1, 1.1)]
        problems = compare_traces(a, b)
        assert any("only in" in p for p in problems)

    def test_two_empty_traces_are_a_problem_not_a_pass(self):
        """Two runs that logged nothing must not read as two runs that agreed."""
        problems = compare_traces([], [])
        assert problems
