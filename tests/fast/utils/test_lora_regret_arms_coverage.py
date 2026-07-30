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
