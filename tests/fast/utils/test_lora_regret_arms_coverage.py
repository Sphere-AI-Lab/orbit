"""The three matrices that close the post's coverage gaps on Llama-3.1-8B."""

import pytest

from tools.lora_regret.arms import (
    ALL_MODULES,
    ATTN_MODULES,
    MATRICES,
    MLP_MODULES,
    e1ot_arms,
)

HIDDEN, FFN = 4096, 14336


class TestE1Ot:
    def test_forty_arms_matching_e1s_shape(self):
        arms = e1ot_arms()
        assert len(arms) == 40
        assert sum(1 for a in arms if a.method == "full") == 5
        assert {a.rank for a in arms if a.method == "lora"} == {1, 4, 16, 64, 128, 256, 512}

    def test_every_arm_reads_openthoughts3(self):
        """E1 is Tulu3; this matrix exists precisely to be the other dataset."""
        assert {a.dataset for a in e1ot_arms()} == {"openthoughts3"}

    def test_the_epoch_is_short_enough_that_no_second_long_matrix_is_needed(self):
        """10,000 rows at batch 32 is 312 steps, so these arms run a full epoch
        and yield both the argmins and the curves. `full_epoch` must be set, or
        the launcher caps them at its own NUM_ROLLOUT default."""
        assert all(a.full_epoch for a in e1ot_arms())

    def test_eval_interval_is_about_one_percent_of_the_epoch(self):
        """~100 trace points, which is what C1's departure detector needs."""
        assert {a.eval_nll_interval for a in e1ot_arms()} == {3}

    def test_it_is_registered(self):
        assert len(MATRICES["e1ot"](HIDDEN, FFN, 0, None, None)) == 40


class TestE1Short:
    def test_fourteen_arms_two_methods_seven_lrs(self):
        from tools.lora_regret.arms import e1short_arms

        arms = e1short_arms()
        assert len(arms) == 14
        assert sum(1 for a in arms if a.method == "full") == 7
        assert {a.rank for a in arms if a.method == "lora"} == {256}

    def test_the_grid_resolves_fifteen_from_ten(self):
        """The claim is a 15x multiplier against a long-run 10x. That is a
        factor of 1.5 == 0.176 decades. On the campaign's standard 0.3-decade
        grid, adjacent points differ by 2x and the effect is invisible, so the
        spacing is a requirement of the claim, not a preference."""
        import math

        from tools.lora_regret.arms import e1short_arms

        lrs = sorted({a.lr for a in e1short_arms() if a.method == "full"})
        steps = [math.log10(b / a) for a, b in zip(lrs, lrs[1:])]
        assert max(steps) <= 0.155, steps
        assert math.log10(1.5) > max(steps), "grid cannot resolve 15x from 10x"

    def test_both_methods_get_the_fine_grid(self):
        """The claim is a ratio of two argmins; a coarse denominator ruins it
        as surely as a coarse numerator."""
        from tools.lora_regret.arms import e1short_arms

        arms = e1short_arms()
        assert len({a.lr for a in arms if a.method == "full"}) == 7
        assert len({a.lr for a in arms if a.method == "lora"}) == 7

    def test_one_hundred_rollouts_and_a_cheap_eval_interval(self):
        """At interval 1 a 100-step arm spends ~113 min evaluating against ~14
        min training. The trace is not what this stage measures."""
        from tools.lora_regret.arms import e1short_arms

        arms = e1short_arms()
        assert {a.num_rollout for a in arms} == {100}
        assert {a.eval_nll_interval for a in arms} == {10}
        assert not any(a.full_epoch for a in arms)

    def test_it_runs_on_tulu3_so_e1s_sigma_applies(self):
        from tools.lora_regret.arms import e1short_arms

        assert {a.dataset for a in e1short_arms()} == {"tulu3"}
