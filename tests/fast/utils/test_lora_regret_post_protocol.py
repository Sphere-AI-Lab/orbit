"""Running a matrix on a base model other than the campaign's anchor.

**Correction, 2026-08-02.** This file was written to support switching the
campaign to Qwen3-1.7B "to match the blog post". That was wrong: the source read
as the post was `third_party/lora-without-regret`, a community reproduction
(michaelbzhu) run on Qwen3-1.7B. The post itself uses **Llama-3.1-8B base on
MATH + GSM8K** for its RL experiments and explicitly avoids Qwen, whose
pretraining data inflates math performance and confounds what RL is measured to
teach. The campaign's anchor was already the post's setup; the vendored
directory is deleted.

What survives is worth keeping on its own terms, because `--model` is real
machinery and each of these is a place where being wrong is silent:

  * selecting a model must move the *shapes* as well as the checkpoint -- a
    matrix solved for Llama's 6144-wide fused QKV and run on another model
    produces identically-named arms with the wrong adapter sizes;
  * the OFT/LoRA capacity ladder must be re-solved per model, because a block
    size means a different parameter count on every set of shapes;
  * the model must reach the wandb project and the ledger row, since arm names
    do not carry it.
"""

from __future__ import annotations

import json
import sys

import pytest

from tools.lora_regret import sweep
from tools.lora_regret.arms import (
    E5RL_BLOCK_LADDER,
    MATRICES,
    arm_env,
    e5rl_matched_ladder,
)
from tools.lora_regret.models import get as get_model
from tools.lora_regret.prepare_data import (
    COMPETITION_MATH_TRAIN_ROWS,
    COMPETITION_MATH_VAL_END,
    COMPETITION_MATH_VAL_START,
    prepare_competition_math,
)

LLAMA = get_model("llama3.1-8b")
QWEN = get_model("qwen3-1.7b")


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class TestModelSelection:
    """`--model` has to move the shapes, not only the checkpoint."""

    def _argv(self, *extra):
        return ["sweep.py", "--dry-run", *extra]

    def test_selecting_a_model_points_every_arm_at_its_checkpoint(
        self, monkeypatch, capsys, tmp_path
    ):
        monkeypatch.setattr(
            sys, "argv",
            self._argv("--model", "qwen3-1.7b", "--matrix", "e4",
                       "--results", str(tmp_path / "r.jsonl")),
        )
        sweep.main()
        printed = capsys.readouterr().out
        assert f"MEGATRON_LOAD={QWEN.megatron_load}" in printed
        assert "MODEL_KEY=qwen3-1.7b" in printed
        assert LLAMA.megatron_load not in printed

    def test_the_default_is_still_the_campaigns_anchor(self, monkeypatch, capsys, tmp_path):
        """Every pre-existing ledger and runbook command assumes this."""
        monkeypatch.setattr(
            sys, "argv",
            self._argv("--matrix", "e4", "--results", str(tmp_path / "r.jsonl")),
        )
        sweep.main()
        assert "MODEL_KEY=llama3.1-8b" in capsys.readouterr().out

    def test_the_fused_qkv_width_follows_the_model(self):
        """The silent failure this exists to prevent.

        `qkv_output_size` is not derivable from `hidden_size` under GQA, and it
        decides every matched-parameter block size and rank. Before it was
        threaded, a non-Llama model got its own hidden/FFN and **Llama's** 6144,
        which changes the adapters without changing a single arm name.
        """
        wrong = MATRICES["e4place"](QWEN.hidden_size, QWEN.ffn_size, LLAMA.qkv_output_size, 0, None, None)
        right = MATRICES["e4place"](QWEN.hidden_size, QWEN.ffn_size, QWEN.qkv_output_size, 0, None, None)
        assert {a.name for a in wrong} != {a.name for a in right}

    def test_a_contradicting_shape_flag_is_refused_against_the_selected_model(
        self, monkeypatch, tmp_path
    ):
        """--hidden-size 4096 is right for Llama and wrong for Qwen3-1.7B."""
        monkeypatch.setattr(
            sys, "argv",
            self._argv("--model", "qwen3-1.7b", "--hidden-size", "4096",
                       "--matrix", "e4", "--results", str(tmp_path / "r.jsonl")),
        )
        with pytest.raises(SystemExit) as excinfo:
            sweep.main()
        assert excinfo.value.code == 2

    def test_an_agreeing_shape_flag_still_passes(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setattr(
            sys, "argv",
            self._argv("--model", "qwen3-1.7b", "--hidden-size", str(QWEN.hidden_size),
                       "--matrix", "e4", "--results", str(tmp_path / "r.jsonl")),
        )
        sweep.main()
        assert "MODEL_KEY=qwen3-1.7b" in capsys.readouterr().out

    def test_the_arm_records_which_model_it_ran_on(self):
        arms = MATRICES["e4"](QWEN.hidden_size, QWEN.ffn_size, QWEN.qkv_output_size, 0, None, None)
        stamped = [sweep.replace(a, model="qwen3-1.7b") for a in arms]
        assert all(a.model == "qwen3-1.7b" for a in stamped)
        assert all(arm_env(a) is not None for a in stamped)


class TestTheOftLadderIsResolvedPerModel:
    def test_the_same_block_pairs_with_a_different_rank_on_each_model(self):
        """A block size is not a capacity. OFT's count follows `d_in` and LoRA's
        follows `d_in + d_out`, so the partner rank moves with the shapes."""
        llama = {r["block_size"]: r["lora_rank"]
                 for r in e5rl_matched_ladder(LLAMA.hidden_size, LLAMA.ffn_size, LLAMA.qkv_output_size)}
        qwen = {r["block_size"]: r["lora_rank"]
                for r in e5rl_matched_ladder(QWEN.hidden_size, QWEN.ffn_size, QWEN.qkv_output_size)}
        assert set(llama) == set(qwen) == set(E5RL_BLOCK_LADDER)
        assert llama[512] == 98 and qwen[512] == 96

    def test_every_rung_is_matched_on_both_models(self):
        for model in (LLAMA, QWEN):
            ladder = e5rl_matched_ladder(model.hidden_size, model.ffn_size, model.qkv_output_size)
            assert all(abs(r["ratio"] - 1.0) <= 0.05 for r in ladder), model.key

    def test_an_unmatched_ladder_is_refused_rather_than_built(self):
        """The whole point of the guard.

        `oft_lora_match_report` returns a pair at any ratio, and arms built from a
        0.75 pair run, finish and report accuracies -- so "OFT does not track
        LoRA" would read as a method difference when it is a capacity difference.
        A tolerance tight enough to bite proves the guard is load-bearing.
        """
        with pytest.raises(ValueError, match="not matched"):
            e5rl_matched_ladder(
                LLAMA.hidden_size, LLAMA.ffn_size, LLAMA.qkv_output_size, tolerance=0.001
            )

    def test_the_arms_carry_the_re_solved_rank_not_the_llama_one(self):
        arms = MATRICES["e5rl"](QWEN.hidden_size, QWEN.ffn_size, QWEN.qkv_output_size, 0, 1e-5, None)
        ranks = {a.rank for a in arms if a.method == "lora"}
        assert 96 in ranks and 98 not in ranks

    def test_the_pairing_is_recorded_on_every_arm(self):
        arms = MATRICES["e5rl"](QWEN.hidden_size, QWEN.ffn_size, QWEN.qkv_output_size, 0, 1e-5, None)
        assert all(a.matched_ratio is not None for a in arms)


class TestTheCompetitionMathSplit:
    """A positional split needs its bounds and its source count asserted.

    Not the post's protocol -- see the module docstring. The assertions are worth
    having anyway: the split is by row index, so a changed upstream row count
    silently changes which problems are trained on."""

    @staticmethod
    def _fake_source(monkeypatch, n=12_500):
        import tools.lora_regret.prepare_data as pd

        rows = [{"problem": f"q{i}", "solution": f"so \\boxed{{{i}}}"} for i in range(n)]
        monkeypatch.setattr(pd, "_load_split", lambda *_a, **_k: rows)
        return rows

    def test_the_split_boundaries_are_positional_and_fixed(self, tmp_path, monkeypatch):
        self._fake_source(monkeypatch)
        result = prepare_competition_math(tmp_path)
        assert result.train_rows == COMPETITION_MATH_TRAIN_ROWS
        assert result.test_rows == COMPETITION_MATH_VAL_END - COMPETITION_MATH_VAL_START

    def test_train_and_validation_do_not_overlap(self, tmp_path, monkeypatch):
        self._fake_source(monkeypatch)
        result = prepare_competition_math(tmp_path)
        train = {r["label"] for r in _rows(result.train_path)}
        val = {r["label"] for r in _rows(result.test_path)}
        assert not (train & val)

    def test_without_a_template_the_problem_text_is_untouched(self, tmp_path, monkeypatch):
        """The library must not mutate source text silently."""
        self._fake_source(monkeypatch)
        result = prepare_competition_math(tmp_path)
        assert _rows(result.train_path)[0]["prompt"] == "q0"

    def test_a_changed_source_row_count_is_refused(self, tmp_path, monkeypatch):
        """The split is positional, so a changed dataset changes which problems
        are trained on without changing anything visible in the output."""
        self._fake_source(monkeypatch, n=12_499)
        with pytest.raises(ValueError, match="source rows"):
            prepare_competition_math(tmp_path)

    def test_overlapping_bounds_are_refused(self, tmp_path, monkeypatch):
        self._fake_source(monkeypatch)
        with pytest.raises(ValueError, match="do not hold"):
            prepare_competition_math(tmp_path, n_train=8_000, val_start=7_500, val_end=8_500)

    def test_rows_carry_the_dataset_tag_the_rl_eval_keys_on(self, tmp_path, monkeypatch):
        self._fake_source(monkeypatch)
        result = prepare_competition_math(tmp_path)
        assert all(r["metadata"]["dataset"] == "competition_math" for r in _rows(result.train_path))

    def test_ungradeable_rows_are_dropped_and_counted(self, tmp_path, monkeypatch):
        import tools.lora_regret.prepare_data as pd

        rows = [{"problem": f"q{i}", "solution": "no box here"} if i < 3
                else {"problem": f"q{i}", "solution": f"\\boxed{{{i}}}"} for i in range(20)]
        monkeypatch.setattr(pd, "_load_split", lambda *_a, **_k: rows)
        result = prepare_competition_math(
            tmp_path, n_train=10, val_start=10, val_end=20, expected_source_rows=20
        )
        assert result.filtered_rows == 3
        assert result.train_rows == 7


class TestTheModelIsVisibleInTheResults:
    """`--model` made two experiments share one identity. This is the fix.

    Arm names carry method, capacity, placement, LR and seed -- never the base
    model, because every matrix was single-model when the names were designed.
    So `lora-r1-all-lr1e-05-s0` is the same string on both models, and without
    the two assertions below a Qwen run and a Llama run are one run everywhere a
    human or `analyze` would look.
    """

    def test_two_models_do_not_share_a_wandb_project(self):
        assert sweep.wandb_project("e4", "qwen3-1.7b", "gsm8k", "lora") != sweep.wandb_project(
            "e4", "llama3.1-8b", "gsm8k", "lora"
        )

    def test_the_campaigns_own_dashboards_do_not_move(self):
        """The anchor model keeps the bare name, so every project the runbook
        already names still exists and every pre-`--model` ledger row still
        points at a real dashboard."""
        for matrix in ("e4", "e4place", "e5rl"):
            assert sweep.wandb_project(matrix, "llama3.1-8b") == sweep.wandb_project(matrix)

    def test_the_dataset_and_mode_stay_at_the_front(self):
        """Suffixed, not prefixed: `test_the_project_name_describes_the_arms_it_routes`
        reads the `<dataset>-<sft|rl>-` head, and a model prefix would push the
        claim the name is making out of the position a reader looks at first."""
        assert sweep.wandb_project("e4", "qwen3-1.7b", "gsm8k", "lora").startswith("gsm8k-rl-")

    def test_the_ledger_records_which_model_produced_the_number(self, tmp_path, monkeypatch):
        """Globbing two models' ledgers into `analyze` must not merge their arms
        into one argmin. The row has to say which model it came from; nothing
        else in it does -- the arm name is byte-identical across models."""
        import subprocess

        monkeypatch.setattr(
            sweep.subprocess, "run",
            lambda cmd, env, cwd: subprocess.CompletedProcess(cmd, 0),
        )
        arm = sweep.replace(
            MATRICES["e4"](QWEN.hidden_size, QWEN.ffn_size, QWEN.qkv_output_size, 0, None, None)[0],
            model="qwen3-1.7b",
        )
        results = tmp_path / "r.jsonl"
        log = tmp_path / "logs" / "lora_regret" / f"{arm.name}.log"
        log.parent.mkdir(parents=True)
        log.write_text(
            "eval/rollout_id=0 eval/math_test=0.5 eval/gsm8k_test=0.5\n"
        )

        sweep.run_arm(arm, tmp_path, results, False, launcher=sweep.RL_LAUNCHER,
                      metric="accuracy", matrix="e4")

        row = json.loads(results.read_text().splitlines()[0])
        assert row["model"] == "qwen3-1.7b"
        assert row["wandb_project"] == "gsm8k-rl-rank-ft-qwen3-1.7b"

    def test_the_same_arm_on_the_anchor_model_is_told_apart_only_by_that_field(self):
        """Both halves of the hazard in one assertion: the names collide, and
        the recorded model is what separates them."""
        build = lambda m: MATRICES["e4"](m.hidden_size, m.ffn_size, m.qkv_output_size, 0, None, None)[0]
        assert build(QWEN).name == build(LLAMA).name
        assert sweep.wandb_project("e4", "qwen3-1.7b", "gsm8k", "lora") != sweep.wandb_project(
            "e4", "llama3.1-8b", "gsm8k", "lora"
        )
