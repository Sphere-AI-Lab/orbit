"""The smoke test, the backfill, and the log parsers both of them stand on.

Every case here is a defect that actually shipped. On 2026-08-03 seven gsm8k
columns ran 150 rollouts each, exited 0, and recorded `accuracy: null,
status: "failed"` in every row -- so the tests that matter are the ones that
distinguish "ran" from "measured", which is exactly the distinction the
coverage probe does not make.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from orbit.utils.misc import should_run_periodic_action
from tools.lora_regret import backfill, smoke
from tools.lora_regret.probe_log import (
    RUN_START_MARKER,
    last_run_segment,
    parse_reward_trace,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _rollout(rollout_id: int, reward: float, truncated: float = 0.01, length: float = 200.0) -> str:
    return (
        f"[ts] log_utils.py:54 - rollout {rollout_id}: "
        f"{{'rollout/response_lengths': {length}, 'rollout/rewards': 0.0, "
        f"'rollout/truncated': {truncated}, 'rollout/raw_reward': {reward}, "
        f"'rollout/advantages': 0.0}}"
    )


def _eval(rollout_id: int, scores: dict[str, float]) -> str:
    body = ", ".join(f"'eval/{name}': {value}" for name, value in scores.items())
    return f"[ts] rollout.py:1 - eval {rollout_id}: {{{body}}}"


class TestLastRunSegment:
    def test_a_retried_arm_is_read_as_its_newest_run_only(self):
        """RUN_LOG is a fixed path per arm and the launcher opens it with
        `tee -a`, so attempt N+1 appends to attempt N. `full-na-na-gsm8k-lr5e-07`
        holds three invocations and its ledger row recorded 258 rollout timings
        for a 150-rollout run."""
        text = "\n".join([
            f"{RUN_START_MARKER}/logs/arm.log",
            _rollout(0, 0.1),
            _rollout(1, 0.2),
            f"{RUN_START_MARKER}/logs/arm.log",
            _rollout(0, 0.5),
        ])
        assert [p["reward"] for p in parse_reward_trace(last_run_segment(text))] == [0.5]

    def test_a_log_with_no_marker_reads_as_one_run(self):
        """An older log, or a caller's synthetic text, must read as a single run
        rather than as nothing."""
        text = _rollout(0, 0.3)
        assert last_run_segment(text) == text

    def test_the_run_start_marker_is_the_line_the_launcher_actually_writes(self):
        """Cross-file pin. Everything that segments a log by run depends on this
        string, and the launcher is free to reword it."""
        launcher = (REPO_ROOT / "scripts" / "lib" / "launcher.sh").read_text(encoding="utf-8")
        assert f'echo "{RUN_START_MARKER}${{RUN_LOG}}"' in launcher
        assert 'tee -a "${RUN_LOG}"' in launcher, "append is why segmenting is needed at all"


class TestRewardTrace:
    def test_the_uncentred_reward_is_read_not_the_centred_one(self):
        """With GRPO centring the advantage is the reward minus its group mean,
        so `rollout/rewards` is ~0 on every healthy rollout and reads as a dead
        run. `raw_reward` is the uncentred mean, and with --rm-type math the
        reward is exactly 1 or 0 -- so it is accuracy on the training batch."""
        trace = parse_reward_trace(_rollout(7, 0.687))
        assert trace == [{
            "rollout": 7, "reward": 0.687, "truncated": 0.01, "response_len": 200.0,
        }]

    def test_a_line_without_raw_reward_is_skipped(self):
        assert parse_reward_trace("rollout 3: {'eval/gsm8k_test': 0.4}") == []


class TestSummarize:
    def test_an_arm_that_rose_and_died_is_collapsed_with_a_rollout(self):
        """`full-na-na-gsm8k-lr1e-06` in miniature: it reached 0.70 and was at
        0.000 with 99% truncation by rollout 90. Where an arm dies is the
        measurement -- a run that peaks then collapses says something different
        about its learning rate than one that never rose."""
        trace = (
            [{"rollout": i, "reward": 0.7, "truncated": 0.0, "response_len": 200} for i in range(20)]
            + [{"rollout": i, "reward": 0.0, "truncated": 1.0, "response_len": 2048}
               for i in range(20, 40)]
        )
        summary = backfill.summarize(trace)
        assert summary["verdict"] == "collapsed"
        assert summary["reward_peak"] == pytest.approx(0.7)
        assert summary["collapse_rollout"] == 20

    def test_an_arm_that_never_rose_is_not_called_collapsed(self):
        """`full-na-na-gsm8k-lr2e-05` sat at 0.001 from rollout 0. "Collapsed"
        would claim it had something to lose."""
        trace = [{"rollout": i, "reward": 0.001, "truncated": 1.0, "response_len": 2048}
                 for i in range(30)]
        assert backfill.summarize(trace)["verdict"] == "never-learned"

    def test_a_healthy_arm_is_learned(self):
        trace = [{"rollout": i, "reward": 0.02 + 0.02 * i, "truncated": 0.0, "response_len": 200}
                 for i in range(30)]
        assert backfill.summarize(trace)["verdict"] == "learned"

    def test_a_single_lucky_batch_does_not_set_the_peak(self):
        """The peak is over windowed means. One outlier rollout on an otherwise
        dead arm would otherwise set it and make everything after look like a
        collapse from a height the arm never held."""
        trace = [{"rollout": i, "reward": 0.9 if i == 5 else 0.01, "truncated": 0.0,
                  "response_len": 200} for i in range(40)]
        summary = backfill.summarize(trace)
        assert summary["reward_peak"] < 0.15
        assert summary["verdict"] == "never-learned"


class TestBackfillRow:
    def _log(self, tmp_path: Path, arm: str, text: str) -> Path:
        logs = tmp_path / "logs"
        logs.mkdir(exist_ok=True)
        (logs / f"{arm}.log").write_text(text, encoding="utf-8")
        return logs

    def test_a_reward_curve_alone_never_promotes_a_row_to_ok(self, tmp_path: Path):
        """`campaign.sh` skips ok arms on resume. Promoting an arm on the
        strength of a training-reward curve would quietly retire exactly the
        arms that still need re-running for a held-out number."""
        logs = self._log(tmp_path, "a1", "\n".join(
            [_eval(0, {"gsm8k_test": 0.03})] + [_rollout(i, 0.5) for i in range(20)]
        ))
        row = backfill.backfill_row(
            {"arm": "a1", "dataset": "gsm8k", "status": "failed", "accuracy": None}, logs
        )
        assert row["status"] == "failed"
        assert row["accuracy"] is None
        assert row["accuracy_before_train"] == pytest.approx(0.03)
        assert row["reward_final"] == pytest.approx(0.5)

    def test_reward_is_never_written_to_the_accuracy_field(self, tmp_path: Path):
        """`analyze` picks argmins off `accuracy`. A figure built from training
        reward while labelled held-out accuracy is worse than a missing one."""
        logs = self._log(tmp_path, "a2", "\n".join(_rollout(i, 0.77) for i in range(20)))
        row = backfill.backfill_row({"arm": "a2", "dataset": "gsm8k", "status": "failed"}, logs)
        assert row["accuracy"] is None
        assert row["reward_peak"] == pytest.approx(0.77)

    def test_a_real_post_training_eval_does_promote_the_row(self, tmp_path: Path):
        """The case the live campaigns will land in: train.py is fixed so the
        eval happens, but their already-imported sweep.py still writes
        `accuracy: null`. The log has the number; this recovers it."""
        logs = self._log(tmp_path, "a3", "\n".join(
            [_eval(0, {"gsm8k_test": 0.03})]
            + [_rollout(i, 0.5) for i in range(20)]
            + [_eval(19, {"gsm8k_test": 0.61})]
        ))
        row = backfill.backfill_row(
            {"arm": "a3", "dataset": "gsm8k", "status": "failed", "accuracy": None}, logs
        )
        assert row["status"] == "ok"
        assert row["accuracy"] == pytest.approx(0.61)
        assert row["accuracy_per_dataset"] == {"gsm8k_test": pytest.approx(0.61)}

    def test_a_math_arm_is_read_with_math_keys(self, tmp_path: Path):
        """The dataset comes off the row, not from a constant. Assuming the pair
        is what recorded 11 healthy arms as failed."""
        logs = self._log(tmp_path, "a4", "\n".join(
            [_rollout(i, 0.2) for i in range(20)] + [_eval(19, {"math_test": 0.29})]
        ))
        row = backfill.backfill_row({"arm": "a4", "dataset": "math", "status": "failed"}, logs)
        assert row["accuracy"] == pytest.approx(0.29)


class TestSmokeArms:
    def test_three_arms_one_per_method_all_from_the_real_matrix(self):
        """Read out of `e4` rather than named, so a renamed arm surfaces as a
        missing method instead of as a passing run of something else."""
        arms = smoke.smoke_arms()
        assert [a.method for a in arms] == ["full", "lora", "oft"]
        assert {a.dataset for a in arms} == {smoke.SMOKE_DATASET}

    def test_the_selection_is_deterministic(self):
        assert [a.name for a in smoke.smoke_arms()] == [a.name for a in smoke.smoke_arms()]


class TestPostTrainEvalCount:
    def test_an_eval_before_train_only_log_counts_zero(self):
        """THE defect, stated as a test. Rollout 0's eval comes from train.py's
        eval-before-train branch, which fires regardless of interval, so a log
        containing exactly one eval line looks complete and describes the
        UNTRAINED policy. All seven gsm8k columns ended in this state."""
        text = _eval(0, {"gsm8k_test": 0.032})
        assert smoke.post_train_eval_rollouts(text, ("gsm8k_test",)) == []

    def test_the_final_rollout_eval_is_counted(self):
        text = "\n".join([_eval(0, {"gsm8k_test": 0.03}), _eval(4, {"gsm8k_test": 0.2}),
                          _eval(9, {"gsm8k_test": 0.3})])
        assert smoke.post_train_eval_rollouts(text, ("gsm8k_test",)) == [4, 9]

    def test_an_eval_missing_a_configured_dataset_does_not_count(self):
        """Same fail-closed rule the ledger applies, so the smoke cannot pass on
        an eval line the ledger will go on to reject."""
        text = _eval(9, {"math_test": 0.4})
        assert smoke.post_train_eval_rollouts(text, ("math_test", "gsm8k_test")) == []


class TestSmokeSchedule:
    def test_the_schedule_separates_the_periodic_and_final_rollout_branches(self):
        """The fixed train.py evaluates at [3, 7, 9]; one with the num_rollout
        argument dropped again evaluates at [3, 7]. The schedules DIFFER, which
        is the entire diagnostic content of the smoke's eval check -- and the
        broken schedule is computed here with the genuinely broken call shape,
        not assumed."""
        fixed = [
            rollout_id
            for rollout_id in range(smoke.SMOKE_ROLLOUTS)
            if should_run_periodic_action(
                rollout_id, smoke.SMOKE_EVAL_INTERVAL, None, smoke.SMOKE_ROLLOUTS
            )
        ]
        broken = [
            rollout_id
            for rollout_id in range(smoke.SMOKE_ROLLOUTS)
            if should_run_periodic_action(rollout_id, smoke.SMOKE_EVAL_INTERVAL, None)
        ]
        assert fixed == [3, 7, 9]
        assert broken == [3, 7]
        assert len(fixed) == smoke.EXPECTED_POST_TRAIN_EVALS
        assert smoke.SMOKE_ROLLOUTS - 1 in fixed
        assert smoke.SMOKE_ROLLOUTS - 1 not in broken

    def test_the_interval_must_not_divide_the_rollout_count(self):
        """The property the test above rests on, pinned directly. This file's
        first version used interval 5 against 10 rollouts, and because 10 % 5
        == 0 the periodic branch fired on the final rollout too -- the broken
        and the fixed train.py produced the IDENTICAL schedule [4, 9], and the
        smoke could not detect the very defect it was written for. Anyone
        retuning these numbers hits this assertion before shipping that."""
        assert smoke.SMOKE_ROLLOUTS % smoke.SMOKE_EVAL_INTERVAL != 0

    def test_a_save_fires_exactly_once_via_the_final_rollout_branch(self):
        """SAVE_INTERVAL=999999 never matches the modulo, so the smoke's one
        checkpoint isolates the final-rollout branch of the save call."""
        fires = [
            rollout_id
            for rollout_id in range(smoke.SMOKE_ROLLOUTS)
            if should_run_periodic_action(
                rollout_id, smoke.SMOKE_SAVE_INTERVAL, None, smoke.SMOKE_ROLLOUTS
            )
        ]
        assert fires == [smoke.SMOKE_ROLLOUTS - 1]
        assert len(fires) == smoke.EXPECTED_SAVES

    def test_the_script_reads_the_schedule_from_smoke_py_and_exports_before_sourcing(self):
        """Two properties. The numbers must come from smoke.py -- the eval
        interval is only diagnostic while it does not divide the rollout count,
        and a hand-copied pair in the script would not stay that way. And the
        exports must precede the protocol source: every protocol value is
        `: "${VAR=default}"`, which assigns only when unset, so sourced first
        EVAL_INTERVAL would be 100000 and the smoke would run zero periodic
        evals."""
        script = (REPO_ROOT / "scripts" / "lora_regret" / "smoke_e4_8gpu.sh").read_text(
            encoding="utf-8"
        )
        assert "print(s.SMOKE_ROLLOUTS, s.SMOKE_EVAL_INTERVAL, s.SMOKE_SAVE_INTERVAL)" in script
        assert 'export EVAL_INTERVAL="${SMOKE_EVAL_INTERVAL}"' in script
        assert 'export SAVE_INTERVAL="${SMOKE_SAVE_INTERVAL}"' in script
        export_at = script.index('export EVAL_INTERVAL="${SMOKE_EVAL_INTERVAL}"')
        source_at = script.index('source "${ORBIT_ROOT}/scripts/lora_regret/e4_protocol.sh"')
        assert export_at < source_at

    def test_the_script_sources_the_real_protocol_rather_than_setting_its_own_knobs(self):
        """A smoke that set its own configuration would clear a protocol nothing
        is going to run. Exactly two overrides are allowed, because each IS a
        thing under test rather than a preference: EVAL_INTERVAL (the schedule
        whose final-rollout eval detects the dead branch) and SAVE_INTERVAL
        (the campaign runs with saves off, so the smoke is the only exercise
        the save path gets). The knobs that shape the update itself -- the
        advantage, the clipping, where the metrics go -- must come from the
        protocol untouched."""
        script = (REPO_ROOT / "scripts" / "lora_regret" / "smoke_e4_8gpu.sh").read_text(
            encoding="utf-8"
        )
        assert "e4_protocol.sh" in script
        for knob in ("RL_EXTRA_ARGS", "EPS_CLIP", "EPS_CLIP_HIGH", "NUM_ROLLOUT", "WANDB_MODE"):
            assert f"export {knob}=" not in script, f"{knob} must come from the protocol"

    def test_smoke_rows_can_never_reach_a_real_analysis(self):
        """Ten rollouts produce a real-looking accuracy. `--probe-rollouts`
        stamps the rows and `analyze` refuses any ledger containing one."""
        script = (REPO_ROOT / "scripts" / "lora_regret" / "smoke_e4_8gpu.sh").read_text(
            encoding="utf-8"
        )
        assert "--probe-rollouts" in script


class TestCheckArm:
    def _setup(self, tmp_path: Path, arm_name: str, log_text: str) -> Path:
        (tmp_path / "logs" / "lora_regret").mkdir(parents=True, exist_ok=True)
        (tmp_path / "logs" / "lora_regret" / f"{arm_name}.log").write_text(
            log_text, encoding="utf-8"
        )
        return tmp_path

    def _healthy(self, tmp_path: Path, arm) -> tuple[str, dict]:
        """A log and row shaped exactly like a fully working smoke arm."""
        run_dir = tmp_path / "wandb" / "offline-run-20260803_000000-abcdefgh"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run-abcdefgh.wandb").write_bytes(b"x" * 4096)
        (run_dir / "run-abcdefgh.wandb.synced").write_text("")
        ckpt = tmp_path / "orbit_ckpts" / "lora_regret" / arm.name / "iter_0000010"
        ckpt.mkdir(parents=True, exist_ok=True)
        (ckpt / "model.pt").write_bytes(b"x")
        log = "\n".join(
            [f"{RUN_START_MARKER}/logs/arm.log", _eval(0, {"gsm8k_test": 0.03})]
            + [_rollout(i, 0.3) for i in range(smoke.SMOKE_ROLLOUTS)]
            + [
                f"progress rollout={i}/9 completed={i + 1}/10 remaining=0 elapsed=00:01:00 "
                f"last=00:00:30 avg=00:00:30 eta_remaining=00:00:00 eta_at=x"
                for i in range(smoke.SMOKE_ROLLOUTS)
            ]
            + [_eval(3, {"gsm8k_test": 0.1}), _eval(7, {"gsm8k_test": 0.2}),
               _eval(9, {"gsm8k_test": 0.31})]
            + ["[ts] timer.py:32 - Timer save_model end (elapsed: 12.5s)"]
            + [f"wandb sync \x1b[0m{run_dir}\x1b[0m"]
        )
        row = {
            "arm": arm.name, "accuracy": 0.31, "status": "ok", "steps": 9,
            "accuracy_per_dataset": {"gsm8k_test": 0.31}, "save_seconds": [12.5],
        }
        return log, row

    def test_a_healthy_smoke_passes_every_link(self, tmp_path: Path):
        arm = smoke.smoke_arms()[0]
        log, row = self._healthy(tmp_path, arm)
        self._setup(tmp_path, arm.name, log)
        results = smoke.check_arm(arm, row, tmp_path)
        assert all(ok for ok, _ in results), [d for ok, d in results if not ok]

    def test_a_missing_final_rollout_eval_fails_even_though_two_evals_ran(self, tmp_path: Path):
        """The regression of bug 1 in the smoke's own terms: periodic evals at
        3 and 7 both fire while the final-rollout branch is dead. The count
        alone (2 of 3) fails too, but the named check has to point AT the
        final rollout, because that is the branch to go look at."""
        arm = smoke.smoke_arms()[0]
        log, row = self._healthy(tmp_path, arm)
        log = "\n".join(
            line for line in log.splitlines() if not line.startswith("[ts] rollout.py:1 - eval 9")
        )
        row = {**row, "accuracy": 0.2, "steps": 7}
        self._setup(tmp_path, arm.name, log)
        failed = [d for ok, d in smoke.check_arm(arm, row, tmp_path) if not ok]
        assert any("final-rollout branch" in d for d in failed)
        assert any("from rollout 7" in d for d in failed)

    def test_an_accuracy_from_an_intermediate_eval_is_caught(self, tmp_path: Path):
        """All three evals in the log, but the ledger's number came from
        rollout 7. Every other link is green; only the steps check sees it."""
        arm = smoke.smoke_arms()[0]
        log, row = self._healthy(tmp_path, arm)
        row = {**row, "accuracy": 0.2, "steps": 7, "accuracy_per_dataset": {"gsm8k_test": 0.2}}
        self._setup(tmp_path, arm.name, log)
        failed = [d for ok, d in smoke.check_arm(arm, row, tmp_path) if not ok]
        assert failed == ["ledger accuracy is from rollout 7 (final = 9)"]

    def test_a_save_that_never_ran_fails_both_save_links(self, tmp_path: Path):
        arm = smoke.smoke_arms()[0]
        log, row = self._healthy(tmp_path, arm)
        log = "\n".join(line for line in log.splitlines() if "save_model" not in line)
        row = {**row, "save_seconds": []}
        shutil.rmtree(tmp_path / "orbit_ckpts")
        self._setup(tmp_path, arm.name, log)
        failed = [d for ok, d in smoke.check_arm(arm, row, tmp_path) if not ok]
        assert any("0 save(s) for expected 1" in d for d in failed)
        assert any("MISSING/EMPTY" in d for d in failed)

    def test_expect_saves_zero_skips_the_save_links_rather_than_failing_them(
        self, tmp_path: Path
    ):
        """SMOKE_SAVE=0 means unexercised, not broken; the script prints the
        distinction and the checker must not contradict it."""
        arm = smoke.smoke_arms()[0]
        log, row = self._healthy(tmp_path, arm)
        log = "\n".join(line for line in log.splitlines() if "save_model" not in line)
        row = {**row, "save_seconds": []}
        shutil.rmtree(tmp_path / "orbit_ckpts")
        self._setup(tmp_path, arm.name, log)
        results = smoke.check_arm(arm, row, tmp_path, expect_saves=0)
        assert all(ok for ok, _ in results), [d for ok, d in results if not ok]

    def test_the_real_failure_is_reported_link_by_link(self, tmp_path: Path):
        """The 2026-08-03 shape: a complete, healthy log with one rollout-0 eval
        and a null-accuracy ledger row. The smoke must name which links broke,
        not merely fail."""
        arm = smoke.smoke_arms()[0]
        log = "\n".join(
            [_eval(0, {"gsm8k_test": 0.032})]
            + [_rollout(i, 0.3) for i in range(smoke.SMOKE_ROLLOUTS)]
        )
        self._setup(tmp_path, arm.name, log)
        row = {"arm": arm.name, "accuracy": None, "status": "failed", "accuracy_per_dataset": {}}
        failed = [d for ok, d in smoke.check_arm(arm, row, tmp_path) if not ok]
        assert any("post-training evals" in d for d in failed)
        assert any("accuracy = None" in d for d in failed)
        assert any("'failed'" in d for d in failed)

    def test_an_arm_that_ran_and_recorded_nothing_is_distinguished_from_one_that_did_not_run(
        self, tmp_path: Path
    ):
        """Different defects, different fixes. A missing log means the launcher
        died; a missing ROW with a full log means the ledger write is broken."""
        arm = smoke.smoke_arms()[0]
        self._setup(tmp_path, arm.name, "\n".join(_rollout(i, 0.3) for i in range(10)))
        ran_no_row = [d for ok, d in smoke.check_arm(arm, None, tmp_path) if not ok]
        assert any("NO LEDGER ROW" in d for d in ran_no_row)

        never_ran = smoke.check_arm(smoke.smoke_arms()[1], None, tmp_path)
        assert len(never_ran) == 1 and "no log" in never_ran[0][1]

    def test_the_wandb_directory_survives_the_ansi_codes_wandb_prints(self, tmp_path: Path):
        """wandb bolds the path in its shutdown banner, so `\\S+` takes the
        escape along and the directory misses by four characters -- which
        presents as an EMPTY wandb dir rather than as a parse error."""
        run_dir = tmp_path / "wandb" / "offline-run-20260803_045859-vw4cx1yv"
        text = f"wandb: \x1b[1mwandb sync {run_dir}\x1b[0m\n"
        assert smoke.offline_run_dir(text, tmp_path) == run_dir


class TestSmokeCli:
    def test_plan_prints_three_anchored_regexes(self, capsys):
        assert smoke.main(["plan"]) == 0
        lines = capsys.readouterr().out.strip().splitlines()
        assert len(lines) == 3
        for line, arm in zip(lines, smoke.smoke_arms(), strict=True):
            method, name, only = line.split("\t")
            assert (method, name) == (arm.method, arm.name)
            # Anchored, so `^lora-r1-` cannot also select `lora-r16-`.
            assert only.startswith("^") and only.endswith("$")

    def test_check_exits_non_zero_on_an_absent_ledger(self, capsys, tmp_path):
        assert smoke.main(["check", "--ledger", str(tmp_path / "nope.jsonl"),
                           "--repo-root", str(tmp_path)]) == 1

    def test_check_reads_the_newest_row_for_a_retried_arm(self, tmp_path):
        ledger = tmp_path / "l.jsonl"
        ledger.write_text("\n".join([
            json.dumps({"arm": "x", "accuracy": None, "status": "failed"}),
            json.dumps({"arm": "x", "accuracy": 0.4, "status": "ok"}),
        ]) + "\n")
        assert smoke.load_rows(ledger)["x"]["accuracy"] == pytest.approx(0.4)
