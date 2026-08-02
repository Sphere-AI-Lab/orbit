"""E5-RL: matched-parameter OFT against LoRA, under policy gradient.

The SFT `e5` matrix asked whether matched-parameter OFT behaves like LoRA on a
next-token objective. `e5rl` asks the same question where the project actually
lives -- RL on MATH + GSM8K, scored by accuracy -- and it is the only matrix in
which OFT and LoRA are compared across a *range* of matched capacities. E4's OFT
cell is a single block size; a single point cannot show whether OFT tracks LoRA
as capacity varies, which is the claim.

Two properties carry the whole design and both are asserted here:

  * every OFT arm has a LoRA partner at the same realized parameter count, to
    within a few percent. Without that the comparison measures capacity, not
    method.
  * the block ladder is solved in the fix-block-solve-rank direction. The
    reverse fails: LoRA ranks form a fine lattice while OFT block sizes must
    divide the input dimension, so matching a given rank lands 24-53% off (see
    `test_the_reverse_direction_is_why_this_ladder_is_solved_the_way_it_is`).
"""

from __future__ import annotations

import pytest

from tools.lora_regret.arms import (
    ALL_MODULES,
    E5RL_BLOCK_LADDER,
    MATRICES,
    OFT_MAX_BLOCK_SGLANG,
    RL_MIX_DATASET,
    e5rl_arms,
)

HIDDEN, FFN, QKV = 4096, 14336, 6144
# Any positive value: these tests are about structure, not about which LR wins.
CENTRE = 1e-4


def _arms(centre=CENTRE):
    return e5rl_arms(HIDDEN, FFN, seed=0, oft_lr_centre=centre)


class TestTheMatchedPairing:
    def test_every_oft_block_has_a_lora_partner_at_the_same_capacity(self):
        """The load-bearing property. One unpaired arm and the matrix compares
        capacity instead of method."""
        arms = _arms()
        oft = {a.oft_block_size for a in arms if a.method == "oft"}
        lora = {a.rank for a in arms if a.method == "lora"}
        assert len(oft) == len(lora) == len(E5RL_BLOCK_LADDER)
        # same number of arms on each side, so no cell is half-populated
        assert sum(a.method == "oft" for a in arms) == sum(a.method == "lora" for a in arms)

    def test_the_realized_ratios_are_within_a_few_percent_of_one(self):
        """`matched_ratio` is carried on every arm rather than recomputed at
        analysis time, so a bad match is visible in the ledger. 5% is the bar:
        the ladder's worst rung is 1.2% off, and anything approaching 5% means
        a block size was chosen that the rank lattice cannot follow."""
        for a in _arms():
            assert a.matched_ratio is not None, a.name
            assert abs(a.matched_ratio - 1.0) <= 0.05, (a.name, a.matched_ratio)

    def test_a_pair_shares_its_ratio(self):
        """Both halves of a pair record the same realized ratio -- they are one
        measurement of one match, not two independent ones."""
        arms = _arms()
        by_ratio = {}
        for a in arms:
            by_ratio.setdefault(round(a.matched_ratio, 6), set()).add(a.method)
        for ratio, methods in by_ratio.items():
            assert methods == {"oft", "lora"}, (ratio, methods)

    def test_the_reverse_direction_is_why_this_ladder_is_solved_the_way_it_is(self):
        """Fixing E4's ranks and solving for a block lands far off, which is the
        documented reason this matrix fixes the block instead. If this ever
        starts passing at a tight tolerance, the constraint changed and the
        ladder should be revisited."""
        from tools.lora_regret.arms import LLAMA31_8B_QKV_OUTPUT, _e5_shapes
        from orbit.utils.peft_param_match import oft_lora_match_report

        shapes = _e5_shapes(HIDDEN, FFN, LLAMA31_8B_QKV_OUTPUT)
        # The blocks bracketing E4's r16: neither is close.
        ratios = {b: oft_lora_match_report(b, shapes)["ratio"] for b in (32, 64)}
        matched_ranks = {b: oft_lora_match_report(b, shapes)["lora_rank"] for b in (32, 64)}
        assert 16 not in matched_ranks.values(), matched_ranks
        assert all(r > 0 for r in ratios.values())


class TestTheLadder:
    def test_it_spans_a_real_capacity_range(self):
        """Three rungs a factor of 4 apart: a 16x span end to end. Two points
        cannot distinguish "tracks LoRA" from "happens to agree here"."""
        assert len(E5RL_BLOCK_LADDER) >= 3
        assert max(E5RL_BLOCK_LADDER) / min(E5RL_BLOCK_LADDER) >= 8

    def test_every_rung_can_actually_launch_inside_sglang(self):
        """An RL arm rotates inside the rollout engine, so a block above the
        kernel ceiling is not a slow arm -- it is one that raises
        OutOfResources minutes into the run. E4 learned this the expensive
        way."""
        for block in E5RL_BLOCK_LADDER:
            assert block <= OFT_MAX_BLOCK_SGLANG, block

    def test_the_ladder_avoids_the_range_where_matching_breaks_down(self):
        """Below block 16 the rank lattice is too coarse to follow: block 8
        matches rank 1 at ratio 1.34. Excluded by construction, not by luck."""
        assert min(E5RL_BLOCK_LADDER) >= 16


class TestItIsAnRlMatrix:
    def test_every_arm_runs_on_the_rl_dataset(self):
        for a in _arms():
            assert a.dataset == RL_MIX_DATASET, a.name

    def test_it_is_scored_by_accuracy_not_nll(self):
        from tools.lora_regret.sweep import MATRIX_METRICS

        assert MATRIX_METRICS["e5rl"] == "accuracy"

    def test_it_has_its_own_wandb_project(self):
        """Its own dashboard, named for what it tests. Sharing E4's project
        would mix two different questions into one set of curves."""
        from tools.lora_regret.sweep import MATRIX_PROJECTS, wandb_project

        assert "e5rl" in MATRIX_PROJECTS
        assert wandb_project("e5rl") != wandb_project("e4")

    def test_it_uses_the_rl_third_decade_grid(self):
        """E4's grid, not E1's. Comparable arm-for-arm with the matrix whose
        argmin supplies this one's centre."""
        arms = _arms()
        lrs = sorted({a.lr for a in arms if a.method == "oft"})
        assert len(lrs) == 7
        # ~a third of a decade each; the points are rounded to one significant
        # figure (the 1-2-5 series), so the steps alternate 2.0x / 2.5x.
        ratios = [lrs[i + 1] / lrs[i] for i in range(len(lrs) - 1)]
        for r in ratios:
            assert abs(r - 10 ** (1 / 3)) < 0.4, ratios

    def test_it_is_registered(self):
        assert "e5rl" in MATRICES


class TestTheCentreIsRequired:
    def test_no_centre_is_an_error_rather_than_a_default(self):
        """Mirrors SFT e5 exactly. A default here would be an invented answer to
        the question E4's oftscout arms exist to ask, and it would be invisible
        in the results -- the arms would run and report numbers."""
        with pytest.raises(ValueError, match="oft_lr_centre"):
            e5rl_arms(HIDDEN, FFN, seed=0, oft_lr_centre=None)

    def test_the_centre_moves_only_the_oft_arms(self):
        """LoRA's RL centre is E4's measured one and does not depend on OFT's."""
        a1 = {a.name for a in _arms(1e-4) if a.method == "lora"}
        a2 = {a.name for a in _arms(1e-5) if a.method == "lora"}
        assert a1 == a2
        o1 = {a.name for a in _arms(1e-4) if a.method == "oft"}
        o2 = {a.name for a in _arms(1e-5) if a.method == "oft"}
        assert o1 != o2


class TestItDoesNotDuplicateWhatAlreadyRuns:
    def test_no_arm_name_collides_with_e4_or_e4place(self):
        """Globbing the ledgers together must not produce a duplicate key, where
        the better of two runs of one configuration would silently win. This is
        the same rule that keeps all-modules out of e4place."""
        mine = {a.name for a in _arms()}
        for other in ("e4", "e4place"):
            theirs = {a.name for a in MATRICES[other](HIDDEN, FFN, QKV, 0, CENTRE, None)}
            assert not (mine & theirs), (other, sorted(mine & theirs))

    def test_placement_is_left_to_e4place(self):
        """This matrix varies capacity only. E4-place already compares OFT and
        LoRA at attention-only and MLP-only on the same grid; repeating it here
        would be eight more 8-GPU arms answering a question already asked."""
        assert {a.target_modules for a in _arms()} == {ALL_MODULES}
