"""Arm enumeration and the resume ledger for the LoRA-without-regret sweep.

The log-line parser tests are deliberately paranoid: Task 10's brief shipped
with a regex (``eval/test_nll step=(\\d+) nll=([0-9.]+)``) that matches zero
lines against the format ``train.py`` actually emits, which would have made
the entire 82-run sweep look like a total failure after burning the compute.
So every fixture line here is built from templates pinned, by a source-text
assertion, to ``train.py`` and ``orbit/utils/logging_utils.py`` themselves --
not hand-typed strings that would trivially satisfy this module's own regex.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.lora_regret.arms import (
    MATRICES_REQUIRING_OFT_CENTRE,
    ALL_MODULES,
    ATTN_MODULES,
    MLP_MODULES,
    Arm,
    LORA_LR_GRID,
    FULL_LR_GRID,
    arm_env,
    e1_arms,
    e2_arms,
    e3_arms,
    e4_arms,
    e5_arms,
    e5_scout_arms,
    OFT_SCOUT_GRID,
    sft_arms,
)
from tools.lora_regret import sweep
from tools.lora_regret.sweep import append_result, load_ledger, parse_final_nll, run_arm

H, FFN = 2560, 9728
# Only feeds adapter_param_count, which scales linearly in it -- so any positive
# value exercises the CLI path. Llama-3.1-8B's 32 keeps it recognisable.
NUM_LAYERS = 32
REPO_ROOT = Path(__file__).resolve().parents[3]

# The exact %-style template train.py:_log_eval_nll feeds to logger.info.
# Pinned below against train.py's own source so this constant (and every
# fixture built from it) cannot silently drift from what the real training
# loop emits.
_TRAIN_PY_LOG_TEMPLATE = (
    "eval/test_nll rollout_id=%d step=%d phase=%s nll=%.6f sample_mean=%.6f tokens=%d samples=%d"
)
# The literal phase labels train.py picks between.
_PHASE_BEFORE_TRAIN = "before_train"
_PHASE_AFTER_TRAIN = "after_train"

# configure_logger()'s logging.basicConfig format, pinned the same way.
_LOG_PREFIX_FRAGMENT = "%(filename)s:%(lineno)d - %(message)s"


def _render(rollout_id: int, step: int, phase: str, nll: float, *, sample_mean: float = 1.9,
            tokens: int = 4096, samples: int = 32, prefixed: bool = True) -> str:
    """One log line, formatted exactly like train.py's logger.info call."""
    message = _TRAIN_PY_LOG_TEMPLATE % (rollout_id, step, phase, nll, sample_mean, tokens, samples)
    if not prefixed:
        return message
    return f"[2026-07-28 10:00:00] train.py:40 - {message}"


def _build_log(lines: list[str]) -> str:
    """A realistic multi-line run log: real eval lines interleaved with the
    unrelated startup/timing/progress chatter train.py also emits."""
    noise = [
        "[2026-07-28 09:59:00] train.py:104 - startup: placement groups start",
        "[2026-07-28 09:59:05] train.py:104 - startup: placement groups done elapsed=5.00s",
        "[2026-07-28 09:59:10] train.py:104 - rollout 0: generate start",
        "[2026-07-28 09:59:20] train.py:104 - rollout 0: actor train start",
        "[2026-07-28 09:59:30] train.py:104 - rollout 0: actor train done elapsed=10.00s",
        "[2026-07-28 09:59:31] train.py:270 - progress rollout=9 last=10.0s avg=10.0s eta=0:30:00",
    ]
    return "\n".join(noise + lines + noise)


class TestLrGrids:
    def test_lora_grid_brackets_every_published_optimum(self):
        # published LoRA optima span 1.2e-4 .. 3.5e-4
        assert min(LORA_LR_GRID) < 1.2e-4
        assert max(LORA_LR_GRID) > 3.5e-4
        assert len(LORA_LR_GRID) == 7

    def test_full_grid_brackets_the_fullft_optimum(self):
        assert min(FULL_LR_GRID) < 2.5e-5 < max(FULL_LR_GRID)
        assert len(FULL_LR_GRID) == 7

    def test_grids_are_monotonic(self):
        assert LORA_LR_GRID == sorted(LORA_LR_GRID)
        assert FULL_LR_GRID == sorted(FULL_LR_GRID)


class TestSftArms:
    def test_lora_and_full_arm_count_is_42(self):
        arms = [a for a in sft_arms(H, FFN) if a.method in ("lora", "full")]
        assert len(arms) == 42

    def test_one_full_finetune_config(self):
        full = [a for a in sft_arms(H, FFN) if a.method == "full"]
        assert len(full) == 7
        assert all(a.rank is None for a in full)

    def test_layer_ablation_target_modules(self):
        arms = sft_arms(H, FFN)
        targets = {a.target_modules for a in arms if a.method == "lora" and a.rank == 256}
        assert targets == {
            "linear_qkv,linear_proj,linear_fc1,linear_fc2",
            "linear_qkv,linear_proj",
            "linear_fc1,linear_fc2",
        }

    def test_ranks_present(self):
        ranks = {a.rank for a in sft_arms(H, FFN) if a.method == "lora"}
        assert ranks == {1, 16, 256}

    def test_oft_arm_count_is_40(self):
        oft = [a for a in sft_arms(H, FFN) if a.method == "oft"]
        assert len(oft) == 40

    def test_oft_block_sizes_come_from_the_solver(self):
        from orbit.utils.peft_param_match import matched_oft_block_size

        oft = [a for a in sft_arms(H, FFN) if a.method == "oft"]
        blocks = {a.oft_block_size for a in oft}
        assert matched_oft_block_size(1, H, H) in blocks
        assert matched_oft_block_size(16, H, H) in blocks

    def test_arm_names_are_unique(self):
        names = [a.name for a in sft_arms(H, FFN)]
        assert len(names) == len(set(names))

    def test_total_arm_count_is_82(self):
        assert len(sft_arms(H, FFN)) == 82


class TestArmEnv:
    def test_full_finetune_env(self):
        env = arm_env(Arm("x", "full", None, None, "", 2.5e-5, 0))
        assert env["PEFT_METHOD"] == "none"
        assert env["LR"] == "2.5e-05"
        assert "LORA_RANK" not in env

    def test_lora_env_sets_alpha_and_init(self):
        env = arm_env(Arm("x", "lora", 16, None, "linear_fc1", 2e-4, 3))
        assert env["PEFT_METHOD"] == "lora"
        assert env["LORA_RANK"] == "16"
        assert env["LORA_ALPHA"] == "32"
        assert env["LORA_A_INIT_METHOD"] == "kaiming"
        assert env["TARGET_MODULES"] == "linear_fc1"
        assert env["SEED"] == "3"

    def test_oft_env_sets_block_size(self):
        env = arm_env(Arm("x", "oft", None, 64, "linear_fc1", 1e-4, 0))
        assert env["PEFT_METHOD"] == "oft"
        assert env["OFT_BLOCK_SIZE"] == "64"
        assert "LORA_RANK" not in env

    def test_no_env_sets_rollout_seed(self):
        # The launcher ties ROLLOUT_SEED to SEED itself (scripts/lib/rollout.sh);
        # arm_env must not set it, or a seed sweep would stop varying data order.
        for arm in sft_arms(H, FFN)[:5]:
            assert "ROLLOUT_SEED" not in arm_env(arm)


class TestLauncherPath:
    def test_the_launcher_the_sweep_shells_out_to_exists(self):
        """The single cheapest way to lose a reserved node: sweep.LAUNCHER
        naming a script that is not in this repo. Every arm would fail
        identically, and the ledger would record 82 failures with no NLL."""
        assert (REPO_ROOT / sweep.LAUNCHER).is_file()


class TestE1Matrix:
    """E1 decides C1 (shared learning curve, rank-dependent departure) and C2
    (the ~10x LR rule). Its grid is 5 points at 0.3-decade spacing centred on
    the post's own prediction, so a confirmation is a hit and not a fit."""

    def test_arm_count_is_forty_five(self):
        """40 LoRA/FullFT arms as before, plus the r256-anchored OFT cell."""
        arms = e1_arms()
        assert len(arms) == 45
        assert sum(1 for a in arms if a.method != "oft") == 40

    def test_ranks_are_the_posts_stated_range(self):
        ranks = {a.rank for a in e1_arms() if a.method == "lora"}
        assert ranks == {1, 4, 16, 64, 128, 256, 512}

    def test_one_full_finetune_arm_per_lr(self):
        full = [a for a in e1_arms() if a.method == "full"]
        assert len(full) == 5

    def test_lora_centre_is_ten_times_the_full_centre(self):
        """C2's prediction, built into the grid rather than fitted afterwards."""
        full_lrs = sorted({a.lr for a in e1_arms() if a.method == "full"})
        lora_lrs = sorted({a.lr for a in e1_arms() if a.method == "lora"})
        assert lora_lrs[2] == pytest.approx(10 * full_lrs[2], rel=0.02)

    def test_grid_spacing_is_zero_point_three_decades(self):
        lrs = sorted({a.lr for a in e1_arms() if a.method == "full"})
        ratios = [b / a for a, b in zip(lrs, lrs[1:], strict=False)]
        assert all(r == pytest.approx(10**0.3, rel=0.02) for r in ratios)

    def test_every_lora_arm_targets_all_four_projections(self):
        assert {a.target_modules for a in e1_arms() if a.method == "lora"} == {ALL_MODULES}


class TestE2Matrix:
    """E2 decides C3 (LoRA tolerates large batches worse, independent of rank)."""

    def test_arm_count_is_forty_eight(self):
        """36 LoRA/FullFT arms as before, plus one OFT cell per batch size."""
        arms = e2_arms()
        assert len(arms) == 48
        assert sum(1 for a in arms if a.method != "oft") == 36

    def test_batch_sizes_are_the_posts_three(self):
        assert {a.global_batch_size for a in e2_arms()} == {32, 128, 512}

    def test_rank_independence_needs_two_lora_ranks(self):
        """E2-2: the post blames the parametrization, not capacity, so the gap
        must be measured at a second rank -- if it shrinks with rank, the
        post's mechanism is wrong and that is the finding."""
        assert {a.rank for a in e2_arms() if a.method == "lora"} == {16, 256}

    def test_four_lrs_per_cell(self):
        """12 cells now: the original 9 plus one OFT cell per batch size. Every
        cell is still 4 LRs wide, OFT included -- an OFT cell with fewer points
        would get a worse argmin than the LoRA cell it is compared against."""
        cells = {}
        for arm in e2_arms():
            key = (arm.method, arm.rank, arm.oft_block_size, arm.global_batch_size)
            cells.setdefault(key, []).append(arm.lr)
        assert all(len(lrs) == 4 for lrs in cells.values())
        assert len(cells) == 12

    def test_lr_centre_rises_with_batch_size(self):
        """Re-centred per batch, as the plan requires: holding the update-to-
        weight ratio fixed as gradient noise falls scales the optimum by
        sqrt(batch). The acceptance rule still applies -- an argmin on a grid
        edge is re-run on a re-centred grid, never quoted."""
        by_batch = {}
        for arm in e2_arms():
            if arm.method == "lora" and arm.rank == 256:
                by_batch.setdefault(arm.global_batch_size, []).append(arm.lr)
        centres = {batch: sorted(lrs)[1] for batch, lrs in by_batch.items()}
        assert centres[128] > centres[32]
        assert centres[512] > centres[128]

    def test_env_carries_both_batch_knobs(self):
        """--global-batch-size alone would leave --rollout-batch-size at 32, so
        a "batch 512" arm would still draw 32 prompts per rollout and take 16
        optimizer steps' worth of data per step."""
        arm = next(a for a in e2_arms() if a.global_batch_size == 512)
        env = arm_env(arm)
        assert env["GLOBAL_BATCH_SIZE"] == "512"
        assert env["ROLLOUT_BATCH_SIZE"] == "512"

    def test_env_points_at_openthoughts3(self):
        """The post's C3 setup is a 10,000-example OpenThoughts3 subset, not
        Tulu3 -- the launcher's own default."""
        env = arm_env(e2_arms()[0])
        assert "openthoughts3_train.jsonl" in env["TRAIN_JSONL"]
        assert "openthoughts3_test.jsonl" in env["TEST_JSONL"]


class TestE3Matrix:
    """E3 decides C4 (attention-only underperforms MLP-only at MATCHED
    parameter count). The earlier plan compared them at equal rank, which in a
    transformer is unequal parameters -- confounding placement with capacity."""

    def test_arm_count_is_thirty_five(self):
        """20 LoRA placement arms as before, plus a FullFT reference line (5)
        and an OFT cell at each placement (10)."""
        arms = e3_arms(H, FFN)
        assert len(arms) == 35
        assert sum(1 for a in arms if a.method == "lora") == 20

    def test_the_matched_pair_is_attention_r256_against_mlp_r92_on_llama(self):
        arms = [a for a in e3_arms(4096, 14336) if a.method == "lora"]
        attn = {a.rank for a in arms if a.target_modules == ATTN_MODULES}
        mlp = {a.rank for a in arms if a.target_modules == MLP_MODULES}
        assert attn == {256}
        # 18432r attention vs 51200r MLP per layer in Orbit's fused layout.
        assert mlp == {92, 128}

    def test_the_matched_ranks_really_are_matched(self):
        from orbit.utils.peft_param_match import lora_param_count

        attn = lora_param_count(256, 4096, 6144) + lora_param_count(256, 4096, 4096)
        mlp = lora_param_count(92, 4096, 2 * 14336) + lora_param_count(92, 14336, 4096)
        assert mlp / attn == pytest.approx(1.0, abs=0.01)

    def test_it_keeps_the_posts_own_pair_too(self):
        """MLP r128 is the post's own comparison. Keeping it means a
        disagreement can be attributed to parameter accounting rather than to
        physics."""
        arms = e3_arms(4096, 14336)
        assert any(a.rank == 128 and a.target_modules == MLP_MODULES for a in arms)

    def test_all_modules_arm_is_present_for_the_second_half_of_the_claim(self):
        """C4 also says all-modules adds nothing on top of MLP-only."""
        arms = e3_arms(H, FFN)
        assert any(a.target_modules == ALL_MODULES and a.rank == 256 for a in arms)


class TestLedger:
    def test_load_ledger_of_missing_file_is_empty(self, tmp_path: Path):
        assert load_ledger(tmp_path / "nope.jsonl") == set()

    def test_append_then_load_round_trip(self, tmp_path: Path):
        path = tmp_path / "r.jsonl"
        append_result(path, {"arm": "a1", "status": "ok", "test_nll": 1.84})
        append_result(path, {"arm": "a2", "status": "ok", "test_nll": 1.85})
        assert load_ledger(path) == {"a1", "a2"}

    def test_failed_arms_are_not_treated_as_done(self, tmp_path: Path):
        path = tmp_path / "r.jsonl"
        append_result(path, {"arm": "a1", "status": "failed", "test_nll": None})
        assert load_ledger(path) == set()

    def test_ledger_survives_a_truncated_final_line(self, tmp_path: Path):
        path = tmp_path / "r.jsonl"
        append_result(path, {"arm": "a1", "status": "ok"})
        with path.open("a") as fh:
            fh.write('{"arm": "a2", "sta')
        assert load_ledger(path) == {"a1"}


class TestLogFormatPins:
    """Prove the fixtures below match the real, current source -- not a
    hand-maintained guess of what train.py logs."""

    def test_template_matches_train_py_source(self):
        train_py = (REPO_ROOT / "train.py").read_text()
        assert _TRAIN_PY_LOG_TEMPLATE in train_py

    def test_phase_labels_match_train_py_source(self):
        train_py = (REPO_ROOT / "train.py").read_text()
        assert '"before_train" if before_train else "after_train"' in train_py

    def test_log_prefix_matches_logging_utils_source(self):
        logging_utils = (REPO_ROOT / "orbit" / "utils" / "logging_utils.py").read_text()
        assert _LOG_PREFIX_FRAGMENT in logging_utils

    def test_metric_key_constant_matches_the_wire_format(self):
        # sweep.py builds its regex from this constant instead of re-spelling
        # "eval/test_nll" -- confirm the constant is in fact the literal text
        # train.py's format string starts with.
        from orbit.utils.eval_nll import EVAL_NLL_METRIC_KEY

        assert _TRAIN_PY_LOG_TEMPLATE.startswith(EVAL_NLL_METRIC_KEY + " ")


class TestParseFinalNll:
    def test_parses_a_realistic_multiline_log(self):
        # A normal 200-step run, eval_nll_interval=10: after_train rows at
        # rollout_id 9,19,...,199 (the last one forced regardless of interval).
        lines = [_render(0, 0, _PHASE_BEFORE_TRAIN, 5.9)]
        for rollout_id in range(9, 199, 10):
            lines.append(_render(rollout_id, rollout_id, _PHASE_AFTER_TRAIN, 3.0 - rollout_id * 0.005))
        lines.append(_render(199, 199, _PHASE_AFTER_TRAIN, 1.845700, sample_mean=1.801234))
        log_text = _build_log(lines)

        nll, step = parse_final_nll(log_text)

        assert step == 199
        assert nll == pytest.approx(1.845700)

    def test_before_train_row_cannot_win_even_when_it_is_last_in_the_file(self):
        # Simulates interleaved multi-rank log buffering: the pristine
        # before-train measurement physically appears AFTER a real
        # post-training row. It must still lose.
        log_text = _build_log([
            _render(50, 50, _PHASE_AFTER_TRAIN, 2.0),
            _render(0, 0, _PHASE_BEFORE_TRAIN, 5.0),
        ])

        nll, step = parse_final_nll(log_text)

        assert (nll, step) == (2.0, 50)

    def test_only_before_train_row_present_returns_none(self):
        # An arm that crashed before its first periodic eval: only the
        # pristine base-model number was ever logged. The study wants the
        # final post-training number, so this must NOT be treated as a result.
        log_text = _build_log([_render(0, 0, _PHASE_BEFORE_TRAIN, 5.9)])

        assert parse_final_nll(log_text) == (None, None)

    def test_no_eval_nll_lines_at_all_returns_none(self):
        assert parse_final_nll(_build_log([])) == (None, None)

    def test_picks_the_highest_step_not_the_last_occurrence(self):
        # Two after_train rows out of chronological order in the text.
        log_text = _build_log([
            _render(199, 199, _PHASE_AFTER_TRAIN, 1.5),
            _render(99, 99, _PHASE_AFTER_TRAIN, 9.9),
        ])

        nll, step = parse_final_nll(log_text)

        assert (nll, step) == (1.5, 199)

    def test_single_step_run_where_both_phases_share_step_zero(self):
        # num_rollout=1, eval_nll_interval=1: before_train and after_train
        # both fire at rollout/step 0. The after_train row must still win.
        log_text = _build_log([
            _render(0, 0, _PHASE_BEFORE_TRAIN, 5.9),
            _render(0, 0, _PHASE_AFTER_TRAIN, 5.7),
        ])

        nll, step = parse_final_nll(log_text)

        assert (nll, step) == (5.7, 0)

    def test_ignores_unprefixed_message_text_too(self):
        # The regex must not depend on the logging.basicConfig prefix being
        # present -- exercise the bare message form as well.
        line = _render(12, 12, _PHASE_AFTER_TRAIN, 1.23, prefixed=False)
        assert parse_final_nll(line) == (1.23, 12)


class TestRunArm:
    def test_before_train_only_log_marks_the_arm_failed(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            sweep.subprocess, "run", lambda cmd, env, cwd: subprocess.CompletedProcess(cmd, 0)
        )
        arm = Arm("full-na-na-lr2.5e-05-s0", "full", None, None, "", 2.5e-5, 0)
        results_path = tmp_path / "results.jsonl"
        log_path = tmp_path / "logs" / "lora_regret" / f"{arm.name}.log"
        log_path.parent.mkdir(parents=True)
        log_path.write_text(_render(0, 0, _PHASE_BEFORE_TRAIN, 5.0) + "\n")

        run_arm(arm, tmp_path, results_path, dry_run=False)

        record = json.loads(results_path.read_text().splitlines()[0])
        assert record["status"] == "failed"
        assert record["test_nll"] is None

    def test_after_train_log_marks_the_arm_ok_and_records_the_final_nll(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            sweep.subprocess, "run", lambda cmd, env, cwd: subprocess.CompletedProcess(cmd, 0)
        )
        arm = Arm("full-na-na-lr2.5e-05-s0", "full", None, None, "", 2.5e-5, 0)
        results_path = tmp_path / "results.jsonl"
        log_path = tmp_path / "logs" / "lora_regret" / f"{arm.name}.log"
        log_path.parent.mkdir(parents=True)
        log_path.write_text(
            _render(0, 0, _PHASE_BEFORE_TRAIN, 5.9) + "\n" + _render(199, 199, _PHASE_AFTER_TRAIN, 1.84) + "\n"
        )

        run_arm(arm, tmp_path, results_path, dry_run=False)

        record = json.loads(results_path.read_text().splitlines()[0])
        assert record["status"] == "ok"
        assert record["test_nll"] == pytest.approx(1.84)
        assert record["steps"] == 199


class TestDryRunOutput:
    def test_dry_run_prints_exactly_82_lines_matching_the_matrix(self, capsys, monkeypatch, tmp_path):
        # No dimension flags: they are derived from the arm's model now, and
        # this module's H/FFN are Qwen3-4B's, which the CLI would (correctly)
        # refuse as contradicting the arm's llama3.1-8b default.
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "sweep.py",
                "--dry-run",
                "--results",
                str(tmp_path / "r.jsonl"),
            ],
        )

        sweep.main()

        lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
        assert len(lines) == 82

        full_lines = [line for line in lines if "PEFT_METHOD=none" in line]
        assert len(full_lines) == 7
        assert all("LORA_RANK" not in line for line in full_lines)

        oft_lines = [line for line in lines if "PEFT_METHOD=oft" in line]
        assert len(oft_lines) == 40
        assert all("OFT_BLOCK_SIZE=" in line for line in oft_lines)


class TestE4Matrix:
    """E4 decides C5 (LoRA matches FullFT under policy gradient even at rank 1,
    with a wider band of performant LRs)."""

    def test_arm_count_is_twenty(self):
        """16 LoRA/FullFT arms as before, plus the RL OFT scout cell."""
        arms = e4_arms()
        assert len(arms) == 20
        assert sum(1 for a in arms if a.method != "oft") == 16

    def test_rank_one_is_present(self):
        """C5's whole point. Not the arm to drop under budget pressure."""
        assert {a.rank for a in e4_arms() if a.method == "lora"} == {1, 16, 256}

    def test_four_lrs_per_arm(self):
        """5 cells now: FullFT, LoRA r1/r16/r256, and the RL OFT scout. All four
        LRs wide -- the OFT cell mirrors the width it is compared against."""
        cells = {}
        for arm in e4_arms():
            cells.setdefault((arm.method, arm.rank, arm.oft_block_size), []).append(arm.lr)
        assert len(cells) == 5
        assert all(len(lrs) == 4 for lrs in cells.values())

    def test_lora_centre_is_ten_times_the_full_centre(self):
        full = sorted({a.lr for a in e4_arms() if a.method == "full"})
        lora = sorted({a.lr for a in e4_arms() if a.method == "lora"})
        assert lora[1] == pytest.approx(10 * full[1], rel=0.02)

    def test_grid_is_wider_than_the_sft_grids(self):
        """Half-decade steps, not E1's 0.3: the RL optimum is less well
        predicted than the SFT one, and C5's claim is about the *width* of the
        performant band, which needs coverage more than resolution."""
        lrs = sorted({a.lr for a in e4_arms() if a.method == "full"})
        ratios = [b / a for a, b in zip(lrs, lrs[1:], strict=False)]
        assert all(r == pytest.approx(10**0.5, rel=0.02) for r in ratios)

    def test_env_points_at_the_combined_rl_training_file(self):
        env = arm_env(e4_arms()[0])
        assert env["TRAIN_JSONL"].endswith("math_gsm8k_train.jsonl")

    def test_env_sets_no_test_jsonl(self):
        """There is no math_gsm8k_test.jsonl: E4 evaluates the MATH and GSM8K
        test splits separately so per-dataset accuracy stays visible. Exporting
        a TEST_JSONL the launcher never reads would just mislead whoever reads
        the dry run."""
        assert "TEST_JSONL" not in arm_env(e4_arms()[0])


class TestMatrixLaunchers:
    def test_every_matrix_has_a_launcher_that_exists(self):
        assert set(sweep.MATRIX_LAUNCHERS) == set(sweep.MATRICES)
        for matrix, launcher in sweep.MATRIX_LAUNCHERS.items():
            assert (REPO_ROOT / launcher).is_file(), f"{matrix} -> {launcher}"

    def test_e4_uses_the_rl_launcher_and_the_sft_matrices_do_not(self):
        assert "rl-math-gsm8k" in sweep.MATRIX_LAUNCHERS["e4"]
        for matrix in ("sft82", "e1", "e2", "e3"):
            assert "rl-math-gsm8k" not in sweep.MATRIX_LAUNCHERS[matrix]

    def test_every_matrix_has_a_metric(self):
        assert set(sweep.MATRIX_METRICS) == set(sweep.MATRICES)
        assert sweep.MATRIX_METRICS["e4"] == "accuracy"
        assert sweep.MATRIX_METRICS["e1"] == "nll"


class TestEvalAccuracyFormatPins:
    """The RL eval emits a Python dict repr, not a formatted metric line, so the
    parser is pinned to the source that produces it -- the same discipline the
    NLL pins use, and for the same reason: a parser that silently matches
    nothing turns a whole sweep into uniform 'failed'."""

    def _rollout_py(self) -> str:
        return (REPO_ROOT / "orbit" / "ray" / "rollout.py").read_text(encoding="utf-8")

    def test_log_line_template_matches_rollout_py_source(self):
        assert 'logger.info(f"eval {rollout_id}: {log_dict}")' in self._rollout_py()

    def test_per_dataset_score_key_matches_rollout_py_source(self):
        assert 'log_dict[f"eval/{key}"] = score' in self._rollout_py()

    def test_cross_dataset_average_key_matches_rollout_py_source(self):
        assert 'log_dict["eval/avg"] = sum(per_dataset_scores) / len(per_dataset_scores)' in self._rollout_py()

    def test_step_is_added_after_the_log_call(self):
        """eval/step is assigned *after* logger.info, so it is not in the line.
        The rollout id in the prefix is the only ordering key available."""
        source = self._rollout_py()
        assert source.index('logger.info(f"eval {rollout_id}: {log_dict}")') < source.index(
            'log_dict["eval/step"] = step'
        )


def _render_eval(rollout_id: int, scores: dict, prefixed: bool = True) -> str:
    """One eval log line, built the way rollout.py builds it: a dict repr."""
    log_dict = {f"eval/{name}": score for name, score in scores.items()}
    if len(scores) > 1:
        log_dict["eval/avg"] = sum(scores.values()) / len(scores)
    message = f"eval {rollout_id}: {log_dict}"
    if not prefixed:
        return message
    return f"[2026-07-30 09:59:00] rollout.py:1227 - {message}"


class TestParseFinalAccuracy:
    def test_parses_per_dataset_scores_and_the_average(self):
        line = _render_eval(25, {"math_test": 0.31, "gsm8k_test": 0.43})
        score, rollout_id, per_dataset = sweep.parse_final_accuracy(line)
        assert rollout_id == 25
        assert per_dataset == {"math_test": pytest.approx(0.31), "gsm8k_test": pytest.approx(0.43)}
        assert score == pytest.approx(0.37)

    def test_picks_the_highest_rollout_id_not_the_last_line(self):
        text = "\n".join([
            _render_eval(50, {"math_test": 0.5, "gsm8k_test": 0.5}),
            _render_eval(25, {"math_test": 0.1, "gsm8k_test": 0.1}),
        ])
        score, rollout_id, _ = sweep.parse_final_accuracy(text)
        assert (rollout_id, score) == (50, pytest.approx(0.5))

    def test_single_dataset_run_has_no_avg_key_and_still_parses(self):
        """rollout.py only emits eval/avg when more than one dataset is
        configured, so the parser cannot depend on it."""
        line = _render_eval(3, {"math_test": 0.25})
        score, rollout_id, per_dataset = sweep.parse_final_accuracy(line)
        assert (rollout_id, score) == (3, pytest.approx(0.25))
        assert per_dataset == {"math_test": pytest.approx(0.25)}

    def test_ignores_sub_metric_keys(self):
        """eval/<name>/<metric> and eval/<name>-truncated_ratio are sub-metrics,
        not dataset scores -- counting them would corrupt the average."""
        message = (
            "eval 7: {'eval/math_test': 0.4, 'eval/math_test/response_length': 812.5, "
            "'eval/math_test-truncated_ratio': 0.02}"
        )
        score, _, per_dataset = sweep.parse_final_accuracy(message)
        assert per_dataset == {"math_test": pytest.approx(0.4)}
        assert score == pytest.approx(0.4)

    def test_no_eval_lines_at_all_returns_none(self):
        assert sweep.parse_final_accuracy("nothing here") == (None, None, {})

    def test_works_without_the_logging_prefix(self):
        line = _render_eval(9, {"math_test": 0.6}, prefixed=False)
        assert sweep.parse_final_accuracy(line)[1] == 9


class TestRunArmAccuracyMetric:
    def test_accuracy_arm_records_accuracy_and_no_nll(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            sweep.subprocess, "run", lambda cmd, env, cwd: subprocess.CompletedProcess(cmd, 0)
        )
        arm = e4_arms()[0]
        results_path = tmp_path / "results.jsonl"
        log_path = tmp_path / "logs" / "lora_regret" / f"{arm.name}.log"
        log_path.parent.mkdir(parents=True)
        log_path.write_text(_render_eval(100, {"math_test": 0.33, "gsm8k_test": 0.55}) + "\n")

        run_arm(arm, tmp_path, results_path, dry_run=False, metric="accuracy")

        record = json.loads(results_path.read_text().splitlines()[0])
        assert record["status"] == "ok"
        assert record["accuracy"] == pytest.approx(0.44)
        assert record["accuracy_per_dataset"]["gsm8k_test"] == pytest.approx(0.55)
        assert record["test_nll"] is None

    def test_accuracy_arm_with_no_eval_line_is_failed(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(
            sweep.subprocess, "run", lambda cmd, env, cwd: subprocess.CompletedProcess(cmd, 0)
        )
        arm = e4_arms()[0]
        results_path = tmp_path / "results.jsonl"
        log_path = tmp_path / "logs" / "lora_regret" / f"{arm.name}.log"
        log_path.parent.mkdir(parents=True)
        log_path.write_text("startup: placement groups done\n")

        run_arm(arm, tmp_path, results_path, dry_run=False, metric="accuracy")

        record = json.loads(results_path.read_text().splitlines()[0])
        assert record["status"] == "failed"
        assert record["accuracy"] is None

    def test_e4_dry_run_shells_out_to_the_rl_launcher(self, capsys, tmp_path):
        run_arm(e4_arms()[0], tmp_path, tmp_path / "r.jsonl", dry_run=True, launcher=sweep.RL_LAUNCHER)
        assert "rl-math-gsm8k" in capsys.readouterr().out


class TestRlEvalDatasetNames:
    def test_the_rl_launcher_configures_exactly_the_datasets_the_parser_expects(self):
        """Cross-file pin. The parser reads `eval/<name>` keys by exact name, so a
        rename in the launcher's --eval-prompt-data would make it match nothing
        and every E4 arm would be recorded as failed for one silent reason."""
        launcher = (REPO_ROOT / sweep.RL_LAUNCHER).read_text(encoding="utf-8")
        eval_line = next(line for line in launcher.splitlines() if "--eval-prompt-data" in line)
        for name in sweep.RL_EVAL_DATASETS:
            assert f" {name} " in eval_line, f"{name} not configured in the launcher"

    def test_explicit_names_exclude_passrate_and_truncation_submetrics(self):
        """With --log-passrate and n_samples_per_eval_prompt > 1, rollout.py emits
        `eval/<name>-pass@k` beside the dataset score. Those must not enter the
        mean; exact-name matching is what keeps them out."""
        message = (
            "eval 40: {'eval/math_test': 0.4, 'eval/math_test-pass@1': 0.4, "
            "'eval/math_test-pass@2': 0.6, 'eval/gsm8k_test': 0.6, "
            "'eval/gsm8k_test-truncated_ratio': 0.01, 'eval/avg': 0.5}"
        )
        score, rollout_id, per_dataset = sweep.parse_final_accuracy(message, sweep.RL_EVAL_DATASETS)
        assert (rollout_id, score) == (40, pytest.approx(0.5))
        assert per_dataset == {"math_test": pytest.approx(0.4), "gsm8k_test": pytest.approx(0.6)}

    def test_a_half_reported_eval_is_skipped_rather_than_averaged(self):
        """One dataset missing means the mean would be over a different set of
        splits than every other arm's -- not comparable, so not a number."""
        text = "\n".join([
            "eval 40: {'eval/math_test': 0.4, 'eval/gsm8k_test': 0.6}",
            "eval 60: {'eval/math_test': 0.9}",
        ])
        score, rollout_id, _ = sweep.parse_final_accuracy(text, sweep.RL_EVAL_DATASETS)
        assert (rollout_id, score) == (40, pytest.approx(0.5))

    def test_a_hyphenated_dataset_name_works_when_named_explicitly(self):
        """The heuristic fallback cannot see this one; the explicit form can."""
        message = "eval 5: {'eval/math-500': 0.42}"
        assert sweep.parse_final_accuracy(message, ("math-500",))[0] == pytest.approx(0.42)
        assert sweep.parse_final_accuracy(message)[0] is None


class TestOftDiagnosticScope:
    def _run(self, matrix: str, capsys, monkeypatch, tmp_path) -> str:
        # Dimensions derived from the arm's model -- see TestDryRunOutput.
        monkeypatch.setattr(
            sys, "argv",
            ["sweep.py", "--matrix", matrix,
             "--dry-run", "--results", str(tmp_path / f"{matrix}.jsonl")],
        )
        sweep.main()
        return capsys.readouterr().err

    def test_oft_match_report_is_printed_for_the_matrix_with_oft_arms(self, capsys, monkeypatch, tmp_path):
        assert "oft match rank=" in self._run("sft82", capsys, monkeypatch, tmp_path)

    def test_oft_match_report_is_absent_when_no_oft_arm_is_selected(
        self, capsys, monkeypatch, tmp_path
    ):
        """Printing a block-size ratio next to a run with no OFT arm invites
        reading it as a property of the arms about to execute.

        Every matrix now carries an OFT cell, so the case that exercises the
        guard is `--only`: selecting just the LoRA arms of a matrix must not
        print a diagnostic about the OFT arms it filtered out.
        """
        for matrix in ("e1", "e4"):
            monkeypatch.setattr(
                sys, "argv",
                ["sweep.py", "--matrix", matrix, "--only", "^lora-",
                 "--dry-run", "--results", str(tmp_path / f"{matrix}.jsonl")],
            )
            sweep.main()
            assert "oft match rank=" not in capsys.readouterr().err


LLAMA_H, LLAMA_FFN = 4096, 14336


class TestE5ScoutMatrix:
    """E5 asks whether matched-parameter OFT behaves like LoRA on C1/C2/C4. Its
    LR scale is unknown a priori -- OFT parameterizes a rotation, not an additive
    update -- so the scout comes first and the refinement grid is centred on what
    the scout finds."""

    def test_scout_is_five_arms_on_the_half_decade_grid(self):
        arms = e5_scout_arms(LLAMA_H, LLAMA_FFN)
        assert len(arms) == 5
        assert sorted(a.lr for a in arms) == sorted(OFT_SCOUT_GRID)

    def test_scout_is_oft_only(self):
        assert {a.method for a in e5_scout_arms(LLAMA_H, LLAMA_FFN)} == {"oft"}

    def test_scout_block_size_is_one_of_the_refinement_ladder(self):
        """Scouting at a block size the refinement never uses would locate the LR
        for a model that is not then measured."""
        scout_blocks = {a.oft_block_size for a in e5_scout_arms(LLAMA_H, LLAMA_FFN)}
        refine_blocks = {a.oft_block_size for a in e5_arms(LLAMA_H, LLAMA_FFN, oft_lr_centre=1e-4)}
        assert scout_blocks <= refine_blocks


class TestE5Matrix:
    def _arms(self, centre=1e-4):
        return e5_arms(LLAMA_H, LLAMA_FFN, oft_lr_centre=centre)

    def test_arm_count_is_fifty(self):
        assert len(self._arms()) == 50

    def test_every_oft_arm_has_a_lora_partner_at_matched_parameters(self):
        """The point of the whole experiment. An unmatched pair would compare
        capacity, not parametrization."""
        arms = self._arms()
        oft = [a for a in arms if a.method == "oft"]
        lora = [a for a in arms if a.method == "lora"]
        assert len(oft) == len(lora) == 25
        for arm in oft:
            partners = [b for b in lora if b.target_modules == arm.target_modules]
            assert partners, f"no LoRA partner for {arm.name}"

    def test_realized_match_ratio_is_recorded_on_every_arm(self):
        """Recorded, not assumed: a pair at 0.93 must not be described as matched,
        and the direction of the miss decides how a result may be read."""
        for arm in self._arms():
            assert arm.matched_ratio is not None
            assert 0.9 < arm.matched_ratio < 1.1, arm

    def test_oft_grid_is_centred_on_the_scout_result(self):
        centre = 3e-4
        oft_lrs = sorted({a.lr for a in self._arms(centre) if a.method == "oft"})
        assert len(oft_lrs) == 5
        assert oft_lrs[2] == pytest.approx(centre, rel=0.02)

    def test_lora_partners_use_the_known_lora_scale_not_the_oft_one(self):
        """LoRA's optimal LR is already known from E1; re-scouting it would spend
        arms to rediscover a number this campaign has measured."""
        lora_lrs = sorted({a.lr for a in self._arms(1e-3) if a.method == "lora"})
        assert lora_lrs[2] == pytest.approx(2.5e-4, rel=0.02)

    def test_capacity_axis_spans_three_block_sizes_on_all_modules(self):
        all_module_oft = [
            a for a in self._arms() if a.method == "oft" and a.target_modules == ALL_MODULES
        ]
        assert {a.oft_block_size for a in all_module_oft} == {32, 64, 256}

    def test_placement_axis_is_a_two_by_two_at_one_capacity(self):
        """C4 for OFT. attention-only and MLP-only at the *same* block size are
        not equal-capacity, so the MLP block size is solved to match attention's
        realized parameter count -- E3's lesson, one method over."""
        # Aliased: peft_param_match exports MLP_MODULES as a *tuple* of module
        # names, while this module's MLP_MODULES is the comma-joined string the
        # launcher takes. Importing it unaliased here shadows the string and every
        # `target_modules ==` comparison silently becomes string-vs-tuple, i.e.
        # always False.
        from orbit.utils.peft_param_match import ATTENTION_MODULES as ATTN_NAMES
        from orbit.utils.peft_param_match import MLP_MODULES as MLP_NAMES
        from orbit.utils.peft_param_match import megatron_module_shapes, oft_param_count_for_modules

        arms = self._arms()
        shapes = megatron_module_shapes(LLAMA_H, LLAMA_FFN, 6144)
        attn = {n: shapes[n] for n in ATTN_NAMES}
        mlp = {n: shapes[n] for n in MLP_NAMES}

        attn_blocks = {a.oft_block_size for a in arms if a.method == "oft" and a.target_modules == ATTN_MODULES}
        mlp_blocks = {a.oft_block_size for a in arms if a.method == "oft" and a.target_modules == MLP_MODULES}
        assert len(attn_blocks) == len(mlp_blocks) == 1
        assert attn_blocks != mlp_blocks, "same block size would mean unequal capacity"

        attn_params = oft_param_count_for_modules(attn_blocks.pop(), attn)
        mlp_params = oft_param_count_for_modules(mlp_blocks.pop(), mlp)
        assert mlp_params / attn_params == pytest.approx(1.0, abs=0.02)

    def test_arm_names_are_unique(self):
        names = [a.name for a in self._arms()]
        assert len(names) == len(set(names))

    def test_oft_arms_carry_a_block_size_and_lora_arms_do_not(self):
        for arm in self._arms():
            if arm.method == "oft":
                assert arm.oft_block_size and arm.rank is None
                assert arm_env(arm)["OFT_BLOCK_SIZE"] == str(arm.oft_block_size)
            else:
                assert arm.rank and arm.oft_block_size is None
                assert "OFT_BLOCK_SIZE" not in arm_env(arm)


class TestE5Wiring:
    def test_refining_without_a_scouted_centre_is_refused(self):
        """You cannot refine before you scout. A default centre here would be an
        invented answer to the question the scout exists to ask."""
        with pytest.raises(ValueError, match="oft_lr_centre"):
            e5_arms(LLAMA_H, LLAMA_FFN, oft_lr_centre=None)

    def test_both_e5_matrices_are_registered_with_the_sft_launcher_and_nll(self):
        for matrix in ("e5scout", "e5"):
            assert sweep.MATRIX_LAUNCHERS[matrix] == sweep.LAUNCHER
            assert sweep.MATRIX_METRICS[matrix] == "nll"


class TestE5CliGuards:
    def _argv(self, *extra):
        return ["sweep.py", "--hidden-size", str(LLAMA_H), "--ffn-size", str(LLAMA_FFN),
                "--num-layers", str(NUM_LAYERS), "--dry-run", *extra]

    def test_e5_without_a_scouted_centre_exits_cleanly(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setattr(sys, "argv", self._argv("--matrix", "e5", "--results", str(tmp_path / "r.jsonl")))
        with pytest.raises(SystemExit) as excinfo:
            sweep.main()
        assert excinfo.value.code == 2
        assert "e5scout" in capsys.readouterr().err

    def test_a_scouted_centre_re_centres_another_matrixs_oft_cell(
        self, monkeypatch, capsys, tmp_path
    ):
        """This used to be refused, because e5 was the only matrix with OFT
        arms. Every matrix carries one now, so the centre is honoured rather
        than rejected -- and honouring it is visible in the arm names, which go
        from `oftscout-` (a search) to `oft-` (a measurement)."""
        monkeypatch.setattr(
            sys, "argv",
            self._argv("--matrix", "e1", "--oft-lr-centre", "1e-4",
                       "--results", str(tmp_path / "r.jsonl")),
        )
        sweep.main()
        printed = capsys.readouterr().out
        assert "LAUNCHER_NAME=oft-b" in printed
        assert "LAUNCHER_NAME=oftscout-" not in printed

    def test_without_a_centre_that_same_cell_is_a_labelled_scout(
        self, monkeypatch, capsys, tmp_path
    ):
        monkeypatch.setattr(
            sys, "argv",
            self._argv("--matrix", "e1", "--results", str(tmp_path / "r.jsonl")),
        )
        sweep.main()
        printed = capsys.readouterr().out
        assert "LAUNCHER_NAME=oftscout-b" in printed

    def test_e5_with_a_centre_runs(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setattr(
            sys, "argv",
            self._argv("--matrix", "e5", "--oft-lr-centre", "1e-4", "--results", str(tmp_path / "r.jsonl")),
        )
        sweep.main()
        assert len(capsys.readouterr().out.strip().splitlines()) == 50


class TestRunArmRecordsTheTrace:
    """The ledger carries the whole curve, not only its last point.

    C1's departure step cannot be recovered from a single final NLL, and the
    logs it would otherwise have to be re-parsed from are gitignored and
    routinely cleaned.
    """

    def _arm(self):
        return Arm("lora-r16-all-lr0.00025-s0", "lora", 16, None, ALL_MODULES, 2.5e-4, 0)

    def _run(self, tmp_path, monkeypatch, log_body):
        results = tmp_path / "results.jsonl"

        def fake_run(cmd, env, cwd):
            Path(env["RUN_LOG"]).parent.mkdir(parents=True, exist_ok=True)
            Path(env["RUN_LOG"]).write_text(log_body)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(sweep.subprocess, "run", fake_run)
        run_arm(self._arm(), tmp_path, results, dry_run=False)
        return json.loads(results.read_text().splitlines()[0])

    def test_the_trace_lands_in_the_record(self, tmp_path, monkeypatch):
        body = _build_log([
            _render(0, 0, _PHASE_BEFORE_TRAIN, 1.209810, tokens=308760, samples=1000),
            _render(0, 0, _PHASE_AFTER_TRAIN, 1.199709, tokens=308760, samples=1000),
            _render(1, 1, _PHASE_AFTER_TRAIN, 1.194836, tokens=308760, samples=1000),
        ])
        record = self._run(tmp_path, monkeypatch, body)
        assert [p["nll"] for p in record["nll_trace"]] == [1.209810, 1.199709, 1.194836]
        assert record["trace_consistent"] is True
        assert record["trace_warning"] is None
        assert record["test_nll"] == 1.194836

    def test_a_floor_divided_held_out_set_is_flagged_but_still_recorded(
        self, tmp_path, monkeypatch
    ):
        body = _build_log([
            _render(0, 0, _PHASE_AFTER_TRAIN, 1.2, tokens=308760, samples=1000),
            _render(1, 1, _PHASE_AFTER_TRAIN, 1.1, tokens=306000, samples=992),
        ])
        record = self._run(tmp_path, monkeypatch, body)
        assert record["trace_consistent"] is False
        assert "992" in record["trace_warning"]
        # The arm still succeeded; it is analyze.py that refuses to quote it.
        assert record["status"] == "ok"


class TestRunArmRecordsTheArmsIdentity:
    """C3 groups by batch size, so the batch size has to be in the record.

    Arm carries global_batch_size and dataset and e2_arms sets both, but the
    ledger dropped them -- leaving the batch an E2 arm ran at recoverable only
    by parsing its name.
    """

    def test_batch_size_and_dataset_reach_the_ledger(self, tmp_path, monkeypatch):
        results = tmp_path / "results.jsonl"

        def fake_run(cmd, env, cwd):
            Path(env["RUN_LOG"]).parent.mkdir(parents=True, exist_ok=True)
            Path(env["RUN_LOG"]).write_text(
                _build_log([_render(0, 0, _PHASE_AFTER_TRAIN, 1.5)])
            )
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(sweep.subprocess, "run", fake_run)
        arm = e2_arms()[0]
        assert arm.global_batch_size is not None, "fixture assumes e2 sets a batch"
        run_arm(arm, tmp_path, results, dry_run=False)
        record = json.loads(results.read_text().splitlines()[0])
        assert record["global_batch_size"] == arm.global_batch_size
        assert record["dataset"] == arm.dataset

    def test_an_arm_with_neither_records_null(self, tmp_path, monkeypatch):
        """E1's arms leave the batch at the launcher's default; null says so."""
        results = tmp_path / "results.jsonl"

        def fake_run(cmd, env, cwd):
            Path(env["RUN_LOG"]).parent.mkdir(parents=True, exist_ok=True)
            Path(env["RUN_LOG"]).write_text(
                _build_log([_render(0, 0, _PHASE_AFTER_TRAIN, 1.5)])
            )
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(sweep.subprocess, "run", fake_run)
        arm = Arm("lora-r16-all-lr0.00025-s0", "lora", 16, None, ALL_MODULES, 2.5e-4, 0)
        run_arm(arm, tmp_path, results, dry_run=False)
        record = json.loads(results.read_text().splitlines()[0])
        assert record["global_batch_size"] is None


class TestDryRunPrintsAPasteableCommand:
    """A previewed command must be the command, including its isolation.

    The launcher's default SAVE_DIR is one directory per recipe, so two arms
    pasted from a dry run would overwrite each other's checkpoints -- the
    runbook's hazard #1, arriving via the preview tool.
    """

    def test_the_sweep_set_variables_are_in_the_printed_line(self, tmp_path, capsys):
        arm = Arm("lora-r16-all-lr0.00025-s0", "lora", 16, None, ALL_MODULES, 2.5e-4, 0)
        run_arm(arm, tmp_path, tmp_path / "r.jsonl", dry_run=True)
        line = capsys.readouterr().out.strip()
        assert f"SAVE_DIR={tmp_path}/orbit_ckpts/lora_regret/{arm.name}" in line
        assert f"RUN_LOG={tmp_path}/logs/lora_regret/{arm.name}.log" in line
        assert f"LAUNCHER_NAME={arm.name}" in line
        # No matrix given, so this arm is unrouted: the bare campaign project,
        # never a real task's. Group is the method. See TestWandbRouting.
        assert "WANDB_PROJECT=lora-without-regret " in line + " "
        assert "WANDB_GROUP=lora" in line
        # and still the arm's own knobs
        assert "LORA_RANK=16" in line
        assert line.endswith("bash examples/sft/run-llama3_1-8b-bf16-lora-sft-tulu3.sh")

    def test_rl_arms_are_previewed_against_the_rl_launcher_and_group(self, tmp_path, capsys):
        arm = Arm("lora-r1-all-lr1e-05-s0", "lora", 1, None, ALL_MODULES, 1e-5, 0)
        run_arm(
            arm, tmp_path, tmp_path / "r.jsonl", dry_run=True,
            launcher=sweep.RL_LAUNCHER, metric="accuracy", matrix="e4",
        )
        line = capsys.readouterr().out.strip()
        # The project carries the task; the sft/rl distinction the old group
        # spelled out is already implied by which launcher runs.
        assert "WANDB_PROJECT=math-gsm8k-rl-rank" in line
        assert "WANDB_GROUP=lora" in line
        assert line.endswith(f"bash {sweep.RL_LAUNCHER}")


from tools.lora_regret.arms import LLAMA31_8B_QKV_OUTPUT, adapter_param_count

# Counted from the real adapter written by the 2026-07-30 smoke:
# 256 tensors, 32 layers, all bf16. Analytic and measured agree exactly, and
# E3's and E5's matched-parameter claims rest on that agreement.
SMOKE_R256_ALL_MODULES_PARAMS = 570_425_344


class TestAdapterParamCount:
    def test_matches_the_real_r256_adapter(self):
        arm = Arm("lora-r256-all-lr0.00025-s0", "lora", 256, None, ALL_MODULES, 2.5e-4, 0)
        assert (
            adapter_param_count(arm, 4096, 14336, 32, LLAMA31_8B_QKV_OUTPUT)
            == SMOKE_R256_ALL_MODULES_PARAMS
        )

    def test_attention_only_counts_only_attention_modules(self):
        arm = Arm("lora-r256-attn-lr0.00025-s0", "lora", 256, None, ATTN_MODULES, 2.5e-4, 0)
        # linear_qkv 256*(4096+6144) + linear_proj 256*(4096+4096), times 32.
        assert adapter_param_count(arm, 4096, 14336, 32, LLAMA31_8B_QKV_OUTPUT) == (
            256 * (4096 + 6144) + 256 * (4096 + 4096)
        ) * 32

    def test_full_finetuning_has_no_adapter(self):
        arm = Arm("full-na-na-lr2.5e-05-s0", "full", None, None, "", 2.5e-5, 0)
        assert adapter_param_count(arm, 4096, 14336, 32, LLAMA31_8B_QKV_OUTPUT) is None

    def test_oft_uses_the_block_size_not_a_rank(self):
        arm = Arm("oft-b64-all-lr0.0001-s0", "oft", None, 64, ALL_MODULES, 1e-4, 0)
        count = adapter_param_count(arm, 4096, 14336, 32, LLAMA31_8B_QKV_OUTPUT)
        assert count > 0
        # OFT's count follows d_in and ignores d_out, so it must NOT equal the
        # LoRA count for any rank that happens to share the arm's tag.
        lora = Arm("lora-r64-all-x", "lora", 64, None, ALL_MODULES, 1e-4, 0)
        assert count != adapter_param_count(lora, 4096, 14336, 32, LLAMA31_8B_QKV_OUTPUT)

    def test_an_unknown_target_module_raises(self):
        arm = Arm("lora-r16-na-x", "lora", 16, None, "linear_nonexistent", 1e-4, 0)
        with pytest.raises(ValueError, match="no known module"):
            adapter_param_count(arm, 4096, 14336, 32, LLAMA31_8B_QKV_OUTPUT)


class TestLedgerCarriesAdapterParams:
    def test_the_record_reports_the_count(self, tmp_path, monkeypatch):
        results = tmp_path / "results.jsonl"

        def fake_run(cmd, env, cwd):
            Path(env["RUN_LOG"]).parent.mkdir(parents=True, exist_ok=True)
            Path(env["RUN_LOG"]).write_text(
                _build_log([_render(0, 0, _PHASE_AFTER_TRAIN, 1.5)])
            )
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(sweep.subprocess, "run", fake_run)
        arm = Arm("lora-r256-all-lr0.00025-s0", "lora", 256, None, ALL_MODULES, 2.5e-4, 0)
        run_arm(
            arm, tmp_path, results, dry_run=False,
            adapter_params=SMOKE_R256_ALL_MODULES_PARAMS,
        )
        record = json.loads(results.read_text().splitlines()[0])
        assert record["adapter_params"] == SMOKE_R256_ALL_MODULES_PARAMS


from tools.lora_regret.arms import E1LONG_EVAL_INTERVAL, e1long_arms

E1LONG_ARGMINS = {
    ("full", None): 2.5e-5,
    ("lora", 1): 5.0e-4,
    ("lora", 4): 4.0e-4,
    ("lora", 16): 2.5e-4,
    ("lora", 64): 2.5e-4,
    ("lora", 128): 2.5e-4,
    ("lora", 256): 2.5e-4,
    ("lora", 512): 1.5e-4,
}


class TestE1LongMatrix:
    def test_one_arm_per_rank_at_its_own_argmin(self):
        arms = e1long_arms(E1LONG_ARGMINS)
        assert len(arms) == 8
        by_key = {(a.method, a.rank): a for a in arms}
        assert set(by_key) == set(E1LONG_ARGMINS)
        assert by_key[("lora", 512)].lr == 1.5e-4
        assert by_key[("full", None)].lr == 2.5e-5

    def test_every_arm_runs_a_full_epoch(self):
        assert all(a.full_epoch for a in e1long_arms(E1LONG_ARGMINS))

    def test_num_rollout_is_emptied_not_omitted(self):
        """A NUM_ROLLOUT=2000 left exported from E1-1 must not shorten the curve.

        The launcher spells it ${NUM_ROLLOUT:-$((...))} -- the colon form -- so an
        empty value re-derives the full epoch, while omitting the key would let
        the stale export through and turn a 29,323-step curve into a 2,000-step
        one. Every rank would then look like it never departs.
        """
        env = arm_env(e1long_arms(E1LONG_ARGMINS)[0])
        assert env["NUM_ROLLOUT"] == ""

    def test_the_eval_interval_is_about_one_percent_of_the_epoch(self):
        env = arm_env(e1long_arms(E1LONG_ARGMINS)[0])
        assert env["EVAL_NLL_INTERVAL"] == str(E1LONG_EVAL_INTERVAL)
        assert 250 <= E1LONG_EVAL_INTERVAL <= 350

    def test_ordinary_arms_set_neither_knob(self):
        """The non-tautology case: e1's arms must be unchanged by this."""
        env = arm_env(e1_arms()[0])
        assert "NUM_ROLLOUT" not in env
        assert "EVAL_NLL_INTERVAL" not in env

    def test_a_missing_rank_is_refused(self):
        partial = {k: v for k, v in E1LONG_ARGMINS.items() if k != ("lora", 512)}
        with pytest.raises(ValueError, match="missing"):
            e1long_arms(partial)

    def test_arms_train_on_tulu3(self):
        assert all(a.dataset == "tulu3" for a in e1long_arms(E1LONG_ARGMINS))


class TestArgminsFrom:
    def _ledger(self, tmp_path, rows):
        path = tmp_path / "e1.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    def _row(self, method, rank, lr, nll, seed=0):
        return {
            "arm": f"{method}-r{rank}-all-lr{lr:g}-s{seed}", "method": method, "rank": rank,
            "oft_block_size": None,
            "target_modules": "" if method == "full" else ALL_MODULES,
            "lr": lr, "seed": seed, "metric": "nll", "test_nll": nll, "status": "ok",
            "trace_consistent": True, "global_batch_size": None, "dataset": None,
        }

    def _complete(self):
        rows = []
        for lr, nll in [(1e-5, 1.52), (2.5e-5, 1.47), (6.3e-5, 1.51)]:
            rows.append(self._row("full", None, lr, nll))
        for rank in (1, 4, 16, 64, 128, 256, 512):
            for lr, nll in [(1e-4, 1.60), (2.5e-4, 1.50), (6.3e-4, 1.58)]:
                rows.append(self._row("lora", rank, lr, nll))
        return rows

    def test_recovers_one_lr_per_arm(self, tmp_path):
        path = self._ledger(tmp_path, self._complete())
        found = sweep.argmins_from([str(path)], allow_edge=False)
        assert len(found) == 8
        assert found[("lora", 256)] == 2.5e-4
        assert found[("full", None)] == 2.5e-5

    def test_a_partial_ledger_is_refused(self, tmp_path):
        """Three arms that look like a completed stage is the failure to avoid."""
        rows = [r for r in self._complete() if r["rank"] in (None, 1, 4)]
        path = self._ledger(tmp_path, rows)
        with pytest.raises(SystemExit):
            sweep.argmins_from([str(path)], allow_edge=False)

    def test_an_edge_of_grid_argmin_is_refused(self, tmp_path):
        rows = self._complete()
        for row in rows:  # make r512's lowest LR win
            if row["rank"] == 512:
                row["test_nll"] = 1.40 if row["lr"] == 1e-4 else 1.60
        path = self._ledger(tmp_path, rows)
        with pytest.raises(SystemExit):
            sweep.argmins_from([str(path)], allow_edge=False)

    def test_the_edge_override_lets_it_through(self, tmp_path):
        rows = self._complete()
        for row in rows:
            if row["rank"] == 512:
                row["test_nll"] = 1.40 if row["lr"] == 1e-4 else 1.60
        path = self._ledger(tmp_path, rows)
        found = sweep.argmins_from([str(path)], allow_edge=True)
        assert found[("lora", 512)] == 1e-4


class TestE1LongCliGuards:
    def _run(self, tmp_path, extra):
        return subprocess.run(
            [sys.executable, "-m", "tools.lora_regret.sweep",
             "--hidden-size", "4096", "--ffn-size", "14336", "--num-layers", "32",
             "--dry-run", *extra],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )

    def test_e1long_without_argmins_exits_two(self, tmp_path):
        result = self._run(tmp_path, ["--matrix", "e1long"])
        assert result.returncode == 2
        assert "--argmins-from" in result.stderr

    def test_argmins_from_on_another_matrix_exits_two(self, tmp_path):
        result = self._run(tmp_path, ["--matrix", "e1", "--argmins-from", "results/x.jsonl"])
        assert result.returncode == 2
        assert "e1long" in result.stderr


class TestModelRegistryWiring:
    """The three dimension flags are derived, and a contradicting value is a
    hard error rather than a silent preference for one of two sources."""

    def test_every_existing_arm_defaults_to_llama(self):
        from tools.lora_regret.arms import MATRICES

        for name in ("e1", "e2", "e3", "e4", "e5scout", "sft82"):
            built = MATRICES[name](4096, 14336, 0, 1e-4 if name in MATRICES_REQUIRING_OFT_CENTRE else None, None)
            assert {arm.model for arm in built} == {"llama3.1-8b"}, name

    def test_dry_run_exports_the_models_checkpoint_and_mask_type(self, tmp_path, capsys):
        from tools.lora_regret.arms import ALL_MODULES, Arm
        from tools.lora_regret.sweep import run_arm

        arm = Arm("probe", "lora", 16, None, ALL_MODULES, 2.5e-4, 0, dataset="tulu3")
        run_arm(arm, tmp_path, tmp_path / "r.jsonl", dry_run=True)
        printed = capsys.readouterr().out
        assert "LOSS_MASK_TYPE=llama3" in printed
        assert "MIN_GPUS_FULLFT=4" in printed
        assert "Llama-3.1-8B_torch_dist" in printed

    def test_num_rollout_reaches_the_launcher_environment(self):
        from tools.lora_regret.arms import ALL_MODULES, Arm, arm_env

        arm = Arm("probe", "lora", 256, None, ALL_MODULES, 2.5e-4, 0, num_rollout=100)
        assert arm_env(arm)["NUM_ROLLOUT"] == "100"

    def test_full_epoch_still_wins_over_num_rollout(self):
        """`full_epoch` sets NUM_ROLLOUT to the empty string so the launcher
        re-derives the epoch. A stale num_rollout must not resurrect a cap."""
        from tools.lora_regret.arms import ALL_MODULES, Arm, arm_env

        arm = Arm("probe", "lora", 256, None, ALL_MODULES, 2.5e-4, 0,
                  num_rollout=100, full_epoch=True)
        assert arm_env(arm)["NUM_ROLLOUT"] == ""

    def test_contradicting_hidden_size_exits_two(self, tmp_path):
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-m", "tools.lora_regret.sweep", "--matrix", "e1",
             "--hidden-size", "9999", "--dry-run", "--results", str(tmp_path / "r.jsonl")],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert proc.returncode == 2
        assert "9999" in proc.stderr and "llama3.1-8b" in proc.stderr

    def test_dimension_flags_are_now_optional(self, tmp_path):
        import subprocess
        import sys

        proc = subprocess.run(
            [sys.executable, "-m", "tools.lora_regret.sweep", "--matrix", "e1",
             "--dry-run", "--results", str(tmp_path / "r.jsonl")],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        assert proc.returncode == 0
        assert len(proc.stdout.strip().splitlines()) == 45


class TestWandbRouting:
    """One wandb project per task, one group per method inside it.

    Before this, every arm of every matrix landed in the launcher's single
    default project and the only split was sft-vs-rl -- so E1's rank ladder,
    E3's placement pair and E5's OFT arms were 112 runs in one flat namespace,
    and the run that decided C2 was indistinguishable in the sidebar from the
    one that decided C6.
    """

    def test_every_matrix_gets_its_own_project(self):
        from tools.lora_regret.arms import MATRICES
        from tools.lora_regret.sweep import MATRIX_PROJECTS, wandb_project

        assert set(MATRIX_PROJECTS) == set(MATRICES)
        projects = {name: wandb_project(name) for name in MATRICES}
        # Distinct, or two tasks would silently share a dashboard.
        assert len(set(projects.values())) == len(MATRICES)
        assert projects["e1"] == "tulu3-sft-rank"
        assert projects["e1ot"] == "openthoughts3-sft-rank"
        assert projects["e1short"] == "tulu3-sft-lr-horizon"
        assert projects["e4place"] == "math-gsm8k-rl-placement"
        assert projects["e5scout"] == "tulu3-sft-oft-scout"

    @pytest.mark.parametrize(
        "matrix", sorted(set(sweep.MATRIX_PROJECTS) - {"e1long"})
    )
    def test_the_project_name_describes_the_arms_it_routes(self, matrix):
        """The `<dataset>-<sft|rl>` head is checked against what the matrix
        actually builds. A name is a claim about the runs inside it, and a
        project called `tulu3-sft-...` holding OpenThoughts3 RL arms is a worse
        lie than an opaque code would have been.

        `e1long` is excluded because it cannot be built without a real E1-1
        ledger; its dataset is pinned by the e1long tests instead.
        """
        from tools.lora_regret.arms import MATRICES
        from tools.lora_regret.sweep import MATRIX_METRICS, wandb_project

        arms = MATRICES[matrix](4096, 14336, 0, 1e-4 if matrix in MATRICES_REQUIRING_OFT_CENTRE else None, None)
        # `None` means the arm takes the launcher's default, which is tulu3.
        datasets = {(a.dataset or "tulu3") for a in arms}
        assert len(datasets) == 1, f"{matrix} mixes datasets: {sorted(datasets)}"
        dataset = datasets.pop().replace("_", "-")
        mode = "rl" if MATRIX_METRICS[matrix] == "accuracy" else "sft"
        assert wandb_project(matrix).startswith(f"{dataset}-{mode}-")

    def test_e4_and_e4place_do_not_share_a_project(self):
        """They run the same launcher at the same four learning rates. Pooling
        them would put the placement panel and the rank panel on one axis."""
        from tools.lora_regret.sweep import wandb_project

        assert wandb_project("e4") != wandb_project("e4place")

    def test_an_unrouted_arm_lands_where_a_hand_run_one_does(self):
        """`run_arm` is callable directly, and a made-up default matrix would
        write those runs into a real task's dashboard. None means "no task", so
        it gets the launchers' own campaign-wide default."""
        from tools.lora_regret.arms import MATRICES
        from tools.lora_regret.sweep import UNROUTED_WANDB_PROJECT, wandb_project

        assert wandb_project(None) == UNROUTED_WANDB_PROJECT
        assert wandb_project(None) not in {wandb_project(m) for m in MATRICES}
        launcher = (REPO_ROOT / sweep.LAUNCHER).read_text(encoding="utf-8")
        assert f"WANDB_PROJECT:-{UNROUTED_WANDB_PROJECT}" in launcher

    def test_an_unknown_matrix_names_the_valid_ones(self):
        """Adding a matrix without a project would otherwise route it silently."""
        from tools.lora_regret.sweep import wandb_project

        with pytest.raises(KeyError, match="e4place"):
            wandb_project("e9")

    def test_the_dry_run_exports_the_matrixs_project(self, tmp_path, capsys):
        arm = Arm("lora-r16-all-lr0.00025-s0", "lora", 16, None, ALL_MODULES, 2.5e-4, 0)
        run_arm(arm, tmp_path, tmp_path / "r.jsonl", dry_run=True, matrix="e1ot")
        assert "WANDB_PROJECT=openthoughts3-sft-rank" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "method,rank,block,modules",
        [("full", None, None, ""), ("lora", 16, None, ALL_MODULES), ("oft", None, 64, ALL_MODULES)],
    )
    def test_the_group_is_the_arms_method(self, tmp_path, capsys, method, rank, block, modules):
        """FullFT, LoRA and OFT arms of one task group apart inside its project.
        The old group repeated the sft/rl split, which the project now states."""
        arm = Arm(f"{method}-probe", method, rank, block, modules, 2.5e-4, 0)
        run_arm(arm, tmp_path, tmp_path / "r.jsonl", dry_run=True, matrix="e5")
        assert f"WANDB_GROUP={method}" in capsys.readouterr().out

    def test_the_ledger_records_where_the_run_went(self, tmp_path, monkeypatch):
        """A ledger row that cannot name its wandb project cannot be traced back
        to the dashboard it was read off, which is the whole point of splitting
        them."""
        import subprocess

        monkeypatch.setattr(
            subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(a, 0)
        )
        arm = Arm("lora-r16-all-lr0.00025-s0", "lora", 16, None, ALL_MODULES, 2.5e-4, 0)
        results = tmp_path / "r.jsonl"
        run_arm(arm, tmp_path, results, dry_run=False, matrix="e3")
        record = json.loads(results.read_text().splitlines()[0])
        assert record["wandb_project"] == "tulu3-sft-placement"
        assert record["wandb_group"] == "lora"


class TestSmokeRunsAreQuarantined:
    """A probe writes a real-looking loss curve after three rollouts. In a task
    project it would sit beside the arms deciding C2, indistinguishable in the
    sidebar -- so every probe goes to one smoke project instead, and the task
    moves into the group so the runs stay separable."""

    def test_a_probe_run_never_lands_in_a_task_project(self, tmp_path, capsys):
        from tools.lora_regret.arms import MATRICES
        from tools.lora_regret.sweep import MATRIX_PROJECTS, SMOKE_WANDB_PROJECT

        arm = Arm("lora-r16-all-lr0.00025-s0", "lora", 16, None, ALL_MODULES, 2.5e-4, 0)
        for matrix in MATRICES:
            run_arm(arm, tmp_path, tmp_path / "r.jsonl", dry_run=True,
                    matrix=matrix, probe_rollouts=3)
            printed = capsys.readouterr().out
            assert f"WANDB_PROJECT={SMOKE_WANDB_PROJECT}" in printed, matrix
            assert f"WANDB_PROJECT={MATRIX_PROJECTS[matrix]}" not in printed, matrix

    def test_the_group_still_separates_task_and_method(self, tmp_path, capsys):
        arm = Arm("oftscout-b1024-attn-lr2.15e-05-s0", "oft", None, 1024,
                  ATTN_MODULES, 2.15e-5, 0)
        run_arm(arm, tmp_path, tmp_path / "r.jsonl", dry_run=True,
                matrix="e4place", probe_rollouts=3)
        assert "WANDB_GROUP=e4place-oft" in capsys.readouterr().out

    def test_a_real_run_is_unaffected(self, tmp_path, capsys):
        arm = Arm("lora-r16-all-lr0.00025-s0", "lora", 16, None, ALL_MODULES, 2.5e-4, 0)
        run_arm(arm, tmp_path, tmp_path / "r.jsonl", dry_run=True, matrix="e1")
        printed = capsys.readouterr().out
        assert "WANDB_PROJECT=tulu3-sft-rank" in printed
        assert "WANDB_GROUP=lora" in printed

    def test_the_smoke_project_is_not_a_task_project(self):
        from tools.lora_regret.sweep import MATRIX_PROJECTS, SMOKE_WANDB_PROJECT

        assert SMOKE_WANDB_PROJECT not in set(MATRIX_PROJECTS.values())
