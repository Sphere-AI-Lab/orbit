"""The three matrices that close the post's coverage gaps on Llama-3.1-8B."""

import pytest

from tools.lora_regret.arms import (
    ALL_MODULES,
    ATTN_MODULES,
    MATRICES,
    MLP_MODULES,
    e1ot_arms,
)

HIDDEN, FFN, QKV = 4096, 14336, 6144


class TestE1Ot:
    def test_the_rank_ladder_matches_e1s_shape(self):
        """40 LoRA/FullFT arms as E1 has, plus the r256-anchored OFT cell that
        gives this task's dashboard all three methods."""
        arms = e1ot_arms()
        assert len(arms) == 45
        assert sum(1 for a in arms if a.method == "full") == 5
        assert sum(1 for a in arms if a.method == "oft") == 5
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
        assert len(MATRICES["e1ot"](HIDDEN, FFN, QKV, 0, None, None)) == 45


class TestE1Short:
    def test_fourteen_arms_two_methods_seven_lrs(self):
        from tools.lora_regret.arms import e1short_arms

        arms = e1short_arms()
        assert len(arms) == 21
        assert sum(1 for a in arms if a.method == "full") == 7
        assert sum(1 for a in arms if a.method == "oft") == 7
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
    def test_fourteen_arms_two_placements_seven_lrs(self):
        from tools.lora_regret.arms import e4place_arms

        arms = e4place_arms(HIDDEN, FFN)
        assert len(arms) == 35
        peft = [a for a in arms if a.method != "full"]
        assert {a.target_modules for a in peft} == {ATTN_MODULES, MLP_MODULES}
        assert len([a for a in peft if a.method == "lora"]) == 14

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
        mlp = {a.rank for a in e4place_arms(HIDDEN, FFN)
               if a.method == "lora" and a.target_modules == MLP_MODULES}
        assert mlp == {expected}
        assert expected != 256 and expected != 128

    def test_the_fullft_arms_are_a_reference_line_not_a_placement_cell(self):
        """The post's RL placement panel is a comparison within PEFT: FullFT has
        no adapter to place. Its arms are here as the baseline the placement
        cells are read against inside this task's own dashboard, so they target
        no modules and duplicate E4's grid under a distinguishing tag."""
        from tools.lora_regret.arms import e4place_arms

        full = [a for a in e4place_arms(HIDDEN, FFN) if a.method == "full"]
        assert len(full) == 7
        assert {a.target_modules for a in full} == {""}
        assert all("place" in a.name for a in full)

    def test_it_shares_e4s_data_and_lr_grid(self):
        """So the placement result and the rank result are read off comparable
        arms rather than off two differently-shaped grids."""
        import math

        from tools.lora_regret.arms import RL_MIX_DATASET, e4_arms, e4place_arms

        place = e4place_arms(HIDDEN, FFN)
        assert {a.dataset for a in place} == {RL_MIX_DATASET}
        lrs = sorted({a.lr for a in place
                      if a.method == "lora" and a.target_modules == ATTN_MODULES})
        # ~0.384 decades per step, to within the one-significant-figure
        # rounding that keeps the points readable in an arm name.
        steps = [math.log10(b / a) for a, b in zip(lrs, lrs[1:])]
        assert all(abs(s - 0.3835) < 0.15 for s in steps), steps
        assert math.log10(lrs[-1] / lrs[0]) == pytest.approx(2.301, abs=0.03)
        e4_lora = sorted({a.lr for a in e4_arms() if a.method == "lora"})
        assert lrs == e4_lora

    def test_it_is_registered_and_scored_by_accuracy(self):
        from tools.lora_regret.sweep import MATRIX_LAUNCHERS, MATRIX_METRICS

        assert MATRIX_METRICS["e4place"] == "accuracy"
        assert "rl-math-gsm8k" in MATRIX_LAUNCHERS["e4place"]
        assert len(MATRICES["e4place"](HIDDEN, FFN, QKV, 0, None, None)) == 35


class TestE4OftCapacityLadder:
    EXPECTED_LRS = {2e-6, 5e-6, 1e-5, 3e-5, 7e-5, 2e-4, 4e-4}

    def test_each_dataset_has_three_blocks_on_the_lora_lr0_lr6_window(self):
        """Dropping a block or drifting an LR leaves a hole in the OFT curves."""
        from tools.lora_regret.arms import RL_DATASETS, e4_arms

        oft = [arm for arm in e4_arms() if arm.method == "oft"]
        assert len(oft) == 42
        for dataset in RL_DATASETS:
            panel = [arm for arm in oft if arm.dataset == dataset]
            assert {arm.oft_block_size for arm in panel} == {8, 128, 1024}
            assert {arm.lr for arm in panel} == self.EXPECTED_LRS
            assert len(panel) == 21

    def test_every_arm_is_an_all_modules_scout_with_recorded_match(self):
        """The sweep varies only block, LR, and dataset; capacity stays auditable."""
        from tools.lora_regret.arms import e4_arms

        oft = [arm for arm in e4_arms() if arm.method == "oft"]
        assert all(arm.name.startswith("oftscout-") for arm in oft)
        assert {arm.target_modules for arm in oft} == {ALL_MODULES}
        assert all(arm.matched_ratio is not None for arm in oft)

    def test_the_capacity_reports_remain_visible_and_stable(self):
        """A wrong rung changes adapter capacity even when every arm still runs."""
        from orbit.utils.peft_param_match import megatron_module_shapes, oft_lora_match_report
        from tools.lora_regret.arms import E4_OFT_BLOCK_LADDER

        shapes = megatron_module_shapes(HIDDEN, FFN, QKV)
        reports = [oft_lora_match_report(block, shapes) for block in E4_OFT_BLOCK_LADDER]
        assert [(r["block_size"], r["oft_params"], r["lora_rank"]) for r in reports] == [
            (8, 136192, 2),
            (128, 2470912, 35),
            (1024, 19903488, 286),
        ]
        assert [r["ratio"] for r in reports] == pytest.approx(
            [0.9779411765, 1.0138655462, 0.9994343891]
        )


class TestMethodCoverage:
    """Every grid matrix carries FullFT, LoRA and OFT, so each task's wandb
    project shows all three.

    The hazard this class exists for is the OFT learning rate. OFT parameterizes
    a *rotation*, not an additive update, so nothing about LoRA's optimal LR
    transfers to it -- not the value, not the decade. `sft82` put 35 of its 40
    OFT arms on LoRA's grid and the module docstring calls that unjustified. The
    arms added here must therefore be a labelled *scout* until a centre has been
    measured, never a centred measurement wearing a scout's uncertainty.
    """

    GRID_MATRICES = ("e1", "e1short", "e1ot", "e2", "e3", "e4", "e4place")

    @pytest.mark.parametrize("matrix", GRID_MATRICES)
    def test_all_three_methods_are_present(self, matrix):
        assert {a.method for a in MATRICES[matrix](HIDDEN, FFN, QKV, 0, None, None)} == {
            "full", "lora", "oft"
        }

    @pytest.mark.parametrize("matrix", GRID_MATRICES)
    def test_without_a_scouted_centre_the_oft_arms_say_so_in_their_name(self, matrix):
        """`oftscout-...` in the ledger and the dashboard. An arm named `oft-`
        on an unscouted grid would be quoted as a measurement of OFT's optimum
        when it is a search for it."""
        arms = MATRICES[matrix](HIDDEN, FFN, QKV, 0, None, None)
        oft = [a for a in arms if a.method == "oft"]
        assert oft
        assert all(a.name.startswith("oftscout-") for a in oft), [a.name for a in oft]

    @pytest.mark.parametrize("matrix", GRID_MATRICES)
    def test_with_a_centre_they_become_measurements_on_a_centred_grid(self, matrix):
        arms = MATRICES[matrix](HIDDEN, FFN, QKV, 0, 1e-4, None)
        oft = [a for a in arms if a.method == "oft"]
        assert oft
        assert all(a.name.startswith("oft-") for a in oft), [a.name for a in oft]

    @pytest.mark.parametrize("matrix", GRID_MATRICES)
    def test_the_oft_grid_is_never_loras_grid(self, matrix):
        """The specific mistake sft82 made -- 35 of its 40 OFT arms sat on
        LoRA's own point set.

        Overlap is not the same failure and is not banned: a scout that spans
        the plausible region necessarily crosses the LoRA grid, and refusing to
        would push the scout off the answer. What is banned is *being* that
        grid, and being too narrow to find anything a decade away.
        """
        import math

        arms = MATRICES[matrix](HIDDEN, FFN, QKV, 0, None, None)
        oft_lrs = sorted({a.lr for a in arms if a.method == "oft"})
        lora_lrs = sorted({a.lr for a in arms if a.method == "lora"})
        # E4 deliberately scouts the completed LoRA lr0-lr6 window. Its own
        # in-matrix LoRA cell is lr1-lr7, so the two overlap at six points but
        # are not identical; the literal E4 window is pinned separately above.
        if matrix not in {"e4", "e4place"}:
            assert set(oft_lrs) != set(lora_lrs)
        span = math.log10(max(oft_lrs) / min(oft_lrs))
        assert span >= 1.0, f"{matrix} OFT scout spans only {span:.2f} decades"
        lora_span = math.log10(max(lora_lrs) / min(lora_lrs))
        assert span >= lora_span, (
            f"{matrix} OFT scout ({span:.2f} decades) is narrower than the LoRA "
            f"grid ({lora_span:.2f}), so it is a measurement, not a search"
        )

    @pytest.mark.parametrize("matrix", GRID_MATRICES)
    def test_the_oft_cell_mirrors_the_width_of_the_lora_cell_it_sits_beside(self, matrix):
        """Same number of learning rates per cell, so an OFT cell cannot be
        quietly cheaper or finer than the LoRA cell it is compared against."""
        arms = MATRICES[matrix](HIDDEN, FFN, QKV, 0, None, None)
        oft_cells = {}
        lora_cells = {}
        for arm in arms:
            if arm.method == "oft":
                oft_cells.setdefault((arm.oft_block_size, arm.target_modules,
                                      arm.global_batch_size), set()).add(arm.lr)
            elif arm.method == "lora":
                lora_cells.setdefault((arm.rank, arm.target_modules,
                                       arm.global_batch_size), set()).add(arm.lr)
        widths_oft = {len(v) for v in oft_cells.values()}
        widths_lora = {len(v) for v in lora_cells.values()}
        assert widths_oft == widths_lora, (matrix, widths_oft, widths_lora)

    @pytest.mark.parametrize("matrix", GRID_MATRICES)
    def test_every_oft_arm_records_the_match_it_actually_achieved(self, matrix):
        """`matched_ratio` is the block against its own implied LoRA rank, and
        must be near 1 -- that is the pairing the arm really runs.

        It is NOT the ratio against the anchor rank, and cannot be: on
        Llama-3.1-8B all-modules, block 1024 carries 0.764 of r256's parameters
        and the next block up carries 1.529, so no block matches r256 at all.
        Asking for one and taking the nearest would ship a 24%-undersized
        adapter labelled 'matched'. The next test pins the neighbourhood instead.
        """
        from orbit.utils.peft_param_match import megatron_module_shapes, oft_lora_match_report
        from tools.lora_regret.arms import LLAMA31_8B_QKV_OUTPUT

        shapes = megatron_module_shapes(HIDDEN, FFN, LLAMA31_8B_QKV_OUTPUT)
        arms = MATRICES[matrix](HIDDEN, FFN, QKV, 0, None, None)
        oft = [a for a in arms if a.method == "oft"]
        assert oft
        for arm in oft:
            selected = {n: s for n, s in shapes.items()
                        if n in arm.target_modules.split(",")}
            report = oft_lora_match_report(arm.oft_block_size, selected)
            assert arm.matched_ratio == pytest.approx(report["ratio"]), arm.name
            # Every rung now lands inside the band, b8 included. It used to sit
            # outside at 1.338, matched to rank 1 because the rank lattice ran
            # out below it; under canonical accounting b8 carries three
            # rotations on the fused qkv, matches rank 2, and lands at 0.978.
            assert 0.85 <= arm.matched_ratio <= 1.15, (
                matrix,
                arm.name,
                arm.matched_ratio,
            )

    @pytest.mark.parametrize("matrix", GRID_MATRICES)
    def test_the_oft_capacity_is_in_the_neighbourhood_of_a_lora_arm_it_sits_beside(
        self, matrix
    ):
        """The block's implied rank is within a factor of 2 of some LoRA rank
        run on the same modules in the same matrix. Wider than that and the OFT
        arm would be comparing method and capacity at once.

        `e4place` was exempt while SGLang's kernel capped the block at 128,
        which reached only r28 against its r256 attention cell. The kernel fix
        removed that cap, so it is checked like every other matrix again.
        """
        from orbit.utils.peft_param_match import megatron_module_shapes, oft_lora_match_report
        from tools.lora_regret.arms import LLAMA31_8B_QKV_OUTPUT

        shapes = megatron_module_shapes(HIDDEN, FFN, LLAMA31_8B_QKV_OUTPUT)
        arms = MATRICES[matrix](HIDDEN, FFN, QKV, 0, None, None)
        ranks_for: dict[str, set] = {}
        for arm in arms:
            if arm.method == "lora":
                ranks_for.setdefault(arm.target_modules, set()).add(arm.rank)
        for arm in (a for a in arms if a.method == "oft"):
            selected = {n: s for n, s in shapes.items()
                        if n in arm.target_modules.split(",")}
            implied = oft_lora_match_report(arm.oft_block_size, selected)["lora_rank"]
            neighbours = ranks_for[arm.target_modules]
            if matrix == "e4" and arm.oft_block_size == 128:
                # A documented gap, not a tolerance to widen. Under canonical
                # accounting b128 implies rank 35, which falls BETWEEN this
                # matrix's r16 and r256 -- 2.19x the former, 0.14x the latter --
                # so E4's middle OFT rung has no capacity-comparable LoRA arm.
                # The ladder (8/128/1024) was chosen against the old
                # one-rotation-per-module count, where b128 implied rank 24 and
                # sat a comfortable 1.5x from r16. Any capacity claim about this
                # rung has to say so. Closing it means changing
                # E4_OFT_BLOCK_LADDER, which would strand the arms already run.
                assert implied == 35, (matrix, arm.name, implied)
                continue
            assert any(0.5 <= implied / rank <= 2.0 for rank in neighbours), (
                matrix, arm.name, implied, sorted(neighbours)
            )

    def test_the_frozen_legacy_matrix_is_untouched(self):
        """sft82's dry run is recorded in the gate log; it must stay 82 arms."""
        assert len(MATRICES["sft82"](HIDDEN, FFN, QKV, 0, None, None)) == 82

    def test_the_oft_scout_stage_is_not_turned_into_a_sweep(self):
        """e5scout exists to find OFT's learning rate. Adding FullFT and LoRA
        arms to it would make the scout a sweep and delay every OFT number."""
        arms = MATRICES["e5scout"](HIDDEN, FFN, QKV, 0, None, None)
        assert len(arms) == 5
        assert {a.method for a in arms} == {"oft"}

    def test_the_added_fullft_arms_do_not_collide_with_e1s_or_e4s(self):
        """E1 and E4 already run FullFT on the grids E3 and E4-place now borrow.
        Untagged, all of those names would be byte-identical -- a re-run at 8
        GPUs for E4-place, and a duplicate key the moment two ledgers are
        globbed into `analyze`, where the better of two runs of one
        configuration wins. Hence the `place` tag on both."""
        from tools.lora_regret.arms import e1_arms, e3_arms, e4_arms, e4place_arms

        def full_names(arms):
            return {a.name for a in arms if a.method == "full"}

        assert not (full_names(e1_arms()) & full_names(e3_arms(HIDDEN, FFN)))
        assert not (full_names(e4_arms()) & full_names(e4place_arms(HIDDEN, FFN)))

    def test_the_only_cross_matrix_duplicate_is_the_one_e3_always_had(self):
        """E3's all-modules cell IS E1's r256 rung -- same five names, same five
        runs -- and predates this change. It is recorded here rather than fixed:
        E3 needs that cell for C4's second half (all-modules against MLP-only),
        and renaming it would orphan any E1 ledger already carrying it.

        The value of pinning it is that the set cannot grow unnoticed.
        """
        from tools.lora_regret.arms import e1_arms, e3_arms

        shared = {a.name for a in e1_arms()} & {a.name for a in e3_arms(HIDDEN, FFN)}
        assert shared == {
            f"lora-r256-all-lr{lr:g}-s0"
            for lr in (6.28e-05, 0.000125, 0.00025, 0.000499, 0.000995)
        }

    def test_the_new_counts(self):
        expected = {"e1": 45, "e1short": 21, "e1ot": 45, "e2": 48,
                    "e3": 35, "e4": 98, "e4place": 35}
        actual = {m: len(MATRICES[m](HIDDEN, FFN, QKV, 0, None, None)) for m in expected}
        assert actual == expected


class TestOftBlockCeilingUnderRl:
    """SGLang's fused OFT kernel used to be unable to launch above block 128.

    `sglang/srt/oft/triton_ops/fused_rotate_project.py::fused_rotate_project_qkv`
    stages the BS x BS rotation block in shared memory. Measured on an H100
    (232,448 B limit) with Llama-3.1-8B's fused QKV shape:

        BS  16/32/64/128 -> OK, numerically exact
        BS  256          -> needs   589,824 B
        BS  512          -> needs 1,966,080 B
        BS 1024          -> needs 7,077,888 B

    Every working OFT RL example in examples/high_precision ships 32, 64 or
    128, and the kernel's own `_pick_qkv_tiles` mitigation is tuned for 128.
    Nothing rejects a larger block: it fails inside Triton as an opaque
    `OutOfResources` after the SGLang server has already started.

    The e4/e4place OFT cells asked for 1024 (matched to LoRA r256) and died
    exactly there -- discovered by the coverage probe on 2026-07-31.

    SFT is deliberately NOT capped: it runs no rollout engine, never reaches
    this kernel, and its b1024 arms completed normally in the same probe.
    """

    @pytest.mark.parametrize("matrix", ["e4", "e4place"])
    def test_rl_oft_blocks_fit_the_kernel(self, matrix):
        from tools.lora_regret.arms import OFT_MAX_BLOCK_SGLANG

        arms = MATRICES[matrix](HIDDEN, FFN, QKV, 0, None, None)
        blocks = {a.oft_block_size for a in arms if a.method == "oft"}
        assert blocks, matrix
        assert max(blocks) <= OFT_MAX_BLOCK_SGLANG, (matrix, sorted(blocks))

    def test_the_ceiling_is_the_measured_one(self):
        """Raised from 128 once Sphere-AI-Lab/sglang made every rotation
        kernel's shared-memory footprint independent of the block size --
        893f329a2 for the fused QKV/gate_up kernel, 166041d28 for the un-fused
        gemm_oft_r/sgemm_oft_r pair that o_proj and down_proj take. The first
        alone was NOT enough: a --target all arm still died at
        `Required: 2228224`. Verified through the installed package: all of
        16/32/64/128/256/512/1024 launch."""
        from tools.lora_regret.arms import OFT_MAX_BLOCK_SGLANG

        assert OFT_MAX_BLOCK_SGLANG == 1024

    def test_the_ceiling_matches_every_working_example_launcher(self):
        """Pinned against the launchers rather than retyped: if someone ships
        an example at a larger block, either the kernel improved or that
        example is broken, and this should be revisited either way."""
        import re
        from pathlib import Path

        from tools.lora_regret.arms import OFT_MAX_BLOCK_SGLANG

        repo = Path(__file__).resolve().parents[3]
        seen = set()
        for script in (repo / "examples/high_precision").glob("*oft*.sh"):
            for m in re.finditer(r"--oft-block-size\s+(\d+)", script.read_text(encoding="utf-8")):
                seen.add(int(m.group(1)))
        assert seen, "no example pins an OFT block size"
        assert max(seen) <= OFT_MAX_BLOCK_SGLANG, sorted(seen)

    def test_the_e4_capacity_ladder_stays_within_the_measured_ceiling(self):
        """Every selected E4 rung must remain launchable by the RL kernel."""
        from tools.lora_regret.arms import OFT_MAX_BLOCK_SGLANG

        sft = {a.oft_block_size for a in MATRICES["e1"](HIDDEN, FFN, QKV, 0, None, None)
               if a.method == "oft"}
        rl = {a.oft_block_size for a in MATRICES["e4"](HIDDEN, FFN, QKV, 0, None, None)
              if a.method == "oft"}
        assert sft == {1024}
        assert rl == {8, 128, 1024}
        assert max(sft | rl) <= OFT_MAX_BLOCK_SGLANG

    def test_every_e4_rung_records_an_implied_lora_partner(self):
        """Each fixed block stays auditable even where the rank lattice is coarse."""
        from orbit.utils.peft_param_match import megatron_module_shapes, oft_lora_match_report
        from tools.lora_regret.arms import LLAMA31_8B_QKV_OUTPUT

        shapes = megatron_module_shapes(HIDDEN, FFN, LLAMA31_8B_QKV_OUTPUT)
        for arm in MATRICES["e4"](HIDDEN, FFN, QKV, 0, None, None):
            if arm.method != "oft":
                continue
            sel = {n: s for n, s in shapes.items() if n in arm.target_modules.split(",")}
            report = oft_lora_match_report(arm.oft_block_size, sel)
            assert report["lora_rank"] >= 1, (arm.name, report)
            # Every rung lands inside the band now. b8 was the exception at
            # 1.338 under one-rotation-per-module counting, where it matched
            # rank 1 because the lattice ran out below it; three rotations on
            # the fused qkv put it at rank 2 and 0.978.
            assert 0.85 <= report["ratio"] <= 1.15, (arm.name, report)
            assert arm.matched_ratio == pytest.approx(report["ratio"])

    def test_sglang_runtime_supports_power_of_two_blocks_from_four(self):
        import tomllib
        from pathlib import Path

        from tools.lora_regret.arms import OFT_MAX_BLOCK_SGLANG

        expected_sha = "0ab4a2de7e4e25042d3910396449fe94fc960423"
        supported = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
        assert supported[0] == 4
        assert all(block & (block - 1) == 0 for block in supported)
        assert supported[-1] == OFT_MAX_BLOCK_SGLANG

        repo = Path(__file__).resolve().parents[3]
        config = tomllib.loads((repo / "pyproject.toml").read_text())
        sources = config["tool"]["uv"]["sources"]
        pins = config["tool"]["orbit"]["release"]["backend-pins"]
        lock = tomllib.loads((repo / "uv.lock").read_text())
        packages = {package["name"]: package for package in lock["package"]}
        orbit_requires = {
            requirement["name"]: requirement
            for requirement in packages["orbit"]["metadata"]["requires-dist"]
        }
        sglang_git = "https://github.com/Sphere-AI-Lab/sglang.git"
        # The v0.5.18 base bump moved the kernel tree to python/sglang/kernels/
        # aot/ inside the sglang repo; it stays source-built from the same
        # pinned rev (the upstream prebuilt cu130 wheel targets the torch 2.13
        # ABI and cannot load on this stack's torch 2.11).
        kernel_subdir_enc = "python%2Fsglang%2Fkernels%2Faot"
        bridge_sha = "988d642688b46ccf68796b0eb9c22aacc59593bc"
        assert sources["sglang"]["rev"] == expected_sha
        assert pins["sglang"]["tested-ref"] == expected_sha
        assert packages["sglang"]["version"] == "0.0.0.dev16812+g0ab4a2de7"
        assert packages["sglang"]["source"]["git"] == (
            f"{sglang_git}?subdirectory=python&rev={expected_sha}#{expected_sha}"
        )
        assert orbit_requires["sglang"]["git"] == (
            f"{sglang_git}?subdirectory=python&rev={expected_sha}"
        )
        assert sources["sglang-kernel"]["rev"] == expected_sha
        assert sources["sglang-kernel"]["subdirectory"] == "python/sglang/kernels/aot"
        assert packages["sglang-kernel"]["source"]["git"] == (
            f"{sglang_git}?subdirectory={kernel_subdir_enc}&rev={expected_sha}#{expected_sha}"
        )
        assert orbit_requires["sglang-kernel"]["git"] == (
            f"{sglang_git}?subdirectory={kernel_subdir_enc}&rev={expected_sha}"
        )
        assert sources["megatron-bridge"]["rev"] == bridge_sha
        assert packages["megatron-bridge"]["source"]["git"].endswith(
            f"rev={bridge_sha}#{bridge_sha}"
        )
        assert orbit_requires["megatron-bridge"]["git"].endswith(
            f"rev={bridge_sha}"
        )

    def test_flashinfer_override_matches_what_sglang_declares(self):
        """orbit forces flashinfer through [tool.uv] override-dependencies, and
        a uv override silently outvotes the requirement a package declares for
        itself. That is exactly how sglang v0.5.16 -- which declares
        flashinfer_python[cu13]==0.6.14, aligned with its Dockerfile jit-cache
        -- ran against 0.6.3 here: the override predated sglang's bump and
        nothing shouted. flashinfer is the attention backend, so a silent
        downgrade is a runtime difference, not a packaging nicety.

        Guard: the override must say exactly what the installed sglang
        declares, and the environment must actually contain that version. The
        next sglang flashinfer bump then fails here, loudly, instead of being
        overridden back down."""
        import importlib.metadata as md
        import re
        import tomllib
        from pathlib import Path

        import pytest

        repo = Path(__file__).resolve().parents[3]
        config = tomllib.loads((repo / "pyproject.toml").read_text())
        overrides = [
            o
            for o in config["tool"]["uv"]["override-dependencies"]
            if re.match(r"flashinfer[-_]python\s*==", o)
        ]
        assert overrides, (
            "the flashinfer override vanished from pyproject.toml; if dropping "
            "it is deliberate (letting sglang's own pin resolve), delete this "
            "test with it"
        )
        override_version = overrides[0].split("==", 1)[1].strip()

        try:
            declared = [
                r
                for r in (md.requires("sglang") or [])
                if re.match(r"flashinfer[-_]python\b", r)
            ]
        except md.PackageNotFoundError:
            pytest.skip("sglang is not installed in this environment")
        assert declared, "the installed sglang no longer declares flashinfer"
        wanted = re.search(r"==\s*([0-9][0-9a-zA-Z.\-]*)", declared[0])
        assert wanted, f"unparseable flashinfer requirement: {declared[0]!r}"
        # Which side is wrong? The override is measured against the INSTALLED
        # sglang, so a stale environment looks exactly like a bad override -- and
        # "align the override" is then precisely the wrong advice, because it
        # would pin flashinfer to what an outdated sglang wanted. Compare the
        # installed sglang to the rev uv.lock pins and say which one to fix.
        locked_rev = re.search(
            r"sglang\.git\?subdirectory=python&rev=([0-9a-f]+)",
            (repo / "uv.lock").read_text(),
        )
        installed_sglang = md.version("sglang")
        env_is_stale = bool(
            locked_rev and locked_rev.group(1)[:9] not in installed_sglang
        )
        stale_note = (
            f" NOTE: the installed sglang is {installed_sglang}, which is NOT the "
            f"rev uv.lock pins ({locked_rev.group(1)[:9]}) -- this environment is "
            f"stale, so refresh it rather than editing the override."
            if env_is_stale
            else ""
        )
        assert override_version == wanted.group(1), (
            f"orbit overrides flashinfer=={override_version} but the installed "
            f"sglang declares {declared[0]!r}; the override wins silently, so "
            f"align the override (and re-lock) instead of running sglang "
            f"against the wrong attention backend.{stale_note}"
        )
        assert md.version("flashinfer-python") == override_version
