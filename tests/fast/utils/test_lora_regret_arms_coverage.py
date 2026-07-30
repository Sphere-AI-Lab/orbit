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


class TestE4Place:
    def test_eight_arms_two_placements_four_lrs(self):
        from tools.lora_regret.arms import e4place_arms

        arms = e4place_arms(HIDDEN, FFN)
        assert len(arms) == 8
        assert {a.target_modules for a in arms} == {ATTN_MODULES, MLP_MODULES}

    def test_it_does_not_restate_any_arm_e4_already_runs(self):
        """e4's LoRA r256 all-modules cell uses this exact grid, so an
        all-modules cell here would be four byte-identical arm names -- four
        re-run RL arms at 8 GPUs each, and a duplicate key if both ledgers are
        ever globbed into analyze together."""
        from tools.lora_regret.arms import e4_arms, e4place_arms

        assert not ({a.name for a in e4_arms()} & {a.name for a in e4place_arms(HIDDEN, FFN)})

    def test_the_mlp_rank_is_e3s_solved_match_not_a_round_number(self):
        """Comparing attention r256 against MLP r256 would compare placement and
        capacity at once. Orbit fuses qkv and gate+up, so the post's own
        attention-256/MLP-128 pair is not matched in this layout either."""
        from orbit.utils.peft_param_match import matched_mlp_rank
        from tools.lora_regret.arms import LLAMA31_8B_QKV_OUTPUT, e4place_arms

        expected = matched_mlp_rank(256, HIDDEN, FFN, LLAMA31_8B_QKV_OUTPUT)
        mlp = {a.rank for a in e4place_arms(HIDDEN, FFN) if a.target_modules == MLP_MODULES}
        assert mlp == {expected}
        assert expected != 256 and expected != 128

    def test_no_fullft_arm(self):
        """The post's RL placement panel is a comparison within LoRA."""
        from tools.lora_regret.arms import e4place_arms

        assert all(a.method == "lora" for a in e4place_arms(HIDDEN, FFN))

    def test_it_shares_e4s_data_and_half_decade_grid(self):
        """So the placement result and the rank result are read off comparable
        arms rather than off two differently-shaped grids."""
        import math

        from tools.lora_regret.arms import RL_MIX_DATASET, e4_arms, e4place_arms

        place = e4place_arms(HIDDEN, FFN)
        assert {a.dataset for a in place} == {RL_MIX_DATASET}
        lrs = sorted({a.lr for a in place if a.target_modules == ATTN_MODULES})
        steps = [math.log10(b / a) for a, b in zip(lrs, lrs[1:])]
        assert all(abs(s - 0.5) < 0.01 for s in steps), steps
        e4_lora = sorted({a.lr for a in e4_arms() if a.method == "lora"})
        assert lrs == e4_lora

    def test_it_is_registered_and_scored_by_accuracy(self):
        from tools.lora_regret.sweep import MATRIX_LAUNCHERS, MATRIX_METRICS

        assert MATRIX_METRICS["e4place"] == "accuracy"
        assert "rl-math-gsm8k" in MATRIX_LAUNCHERS["e4place"]
        assert len(MATRICES["e4place"](HIDDEN, FFN, 0, None, None)) == 8
