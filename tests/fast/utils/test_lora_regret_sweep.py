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

from tools.lora_regret.arms import Arm, LORA_LR_GRID, FULL_LR_GRID, arm_env, sft_arms
from tools.lora_regret import sweep
from tools.lora_regret.sweep import append_result, load_ledger, parse_final_nll, run_arm

H, FFN = 2560, 9728
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
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "sweep.py",
                "--hidden-size",
                str(H),
                "--ffn-size",
                str(FFN),
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
