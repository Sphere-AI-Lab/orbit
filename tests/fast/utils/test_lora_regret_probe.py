"""One run per (task, method): does it work, and how long is the real arm?

The probe answers two questions and must not be able to answer a third. Its
rows are three-rollout runs; if one were ever read as a grid point, an argmin
would be decided by a learning rate that trained for 90 seconds.
"""

import json

import pytest

from tools.lora_regret.probe import (
    EXCLUDED_MATRICES,
    FULL_RUN_ROLLOUTS,
    PROBE_ROLLOUTS,
    format_report,
    parse_rollout_seconds,
    probe_plan,
)


class TestPlan:
    def test_the_method_level_is_one_run_per_task_per_method(self):
        """Rank, block size, placement and batch size exercise the same code at
        different shapes, so probing them separately re-runs a path that already
        passed. 24 runs, not 61 -- and `path` collapses further still."""
        assert len(probe_plan("method")) == 24

    def test_config_level_launches_every_distinct_configuration_once(self):
        """The opt-in level, for hunting a shape-dependent failure rather than a
        code-path one. A configuration is everything but the learning rate."""
        from tools.lora_regret.arms import MATRICES
        from tools.lora_regret.probe import config_key

        runs = probe_plan("config")
        assert len({(r.matrix, r.arm) for r in runs}) == len(runs)
        for matrix in MATRICES:
            if matrix in EXCLUDED_MATRICES:
                continue
            centre = 1e-4 if matrix == "e5" else None
            arms = MATRICES[matrix](4096, 14336, 0, centre, None)
            wanted = {config_key(a) for a in arms}
            probed = {
                config_key(next(a for a in arms if a.name == r.arm))
                for r in runs if r.matrix == matrix
            }
            assert probed == wanted, matrix

    def test_method_level_is_the_cheap_subset_and_covers_less(self):
        cheap, full = probe_plan("method"), probe_plan("config")
        assert len(cheap) < len(full)
        assert {(r.matrix, r.method) for r in cheap} == {
            (r.matrix, r.method) for r in full
        }

    def test_an_unknown_level_is_refused(self):
        with pytest.raises(ValueError, match="unknown probe level"):
            probe_plan("everything")

    def test_the_largest_shapes_are_reachable_at_config_level(self):
        """Not in the default plan, and deliberately: they are the same code as
        the shapes that are. This pins that `--level config` can still reach
        them when a shape-dependent failure is what you are hunting."""
        labels = {(r.matrix, r.label) for r in probe_plan("config")}
        assert ("e2", "lora/r256/all/batch512") in labels
        assert ("e1", "lora/r512/all") in labels
        assert ("e5", "oft/b256/all") in labels

    def test_one_run_per_task_per_method(self):
        seen = [(run.matrix, run.method) for run in probe_plan("method")]
        assert len(seen) == len(set(seen)), "a (task, method) pair is probed twice"
        by_matrix = {}
        for matrix, method in seen:
            by_matrix.setdefault(matrix, set()).add(method)
        from tools.lora_regret.arms import MATRICES

        for matrix in MATRICES:
            if matrix in EXCLUDED_MATRICES:
                continue
            assert matrix in by_matrix, matrix

    def test_every_probed_matrix_covers_the_methods_it_actually_has(self):
        """Not a fixed {full, lora, oft}: e5scout is OFT-only and e5 has no
        FullFT arm, so demanding three from them would be demanding a run that
        does not exist."""
        from tools.lora_regret.arms import MATRICES

        planned = {}
        for run in probe_plan("method"):
            planned.setdefault(run.matrix, set()).add(run.method)
        for matrix, methods in planned.items():
            centre = 1e-4 if matrix in ("e5",) else None
            built = MATRICES[matrix](4096, 14336, 0, centre, None)
            assert methods == {a.method for a in built}, matrix

    def test_the_excluded_matrices_say_why(self):
        assert set(EXCLUDED_MATRICES) == {"e1long", "sft82"}
        assert all(reason for reason in EXCLUDED_MATRICES.values())

    def test_each_run_names_a_real_arm_of_that_matrix(self):
        from tools.lora_regret.arms import MATRICES

        for run in probe_plan('config'):
            centre = 1e-4 if run.matrix == "e5" else None
            names = {a.name for a in MATRICES[run.matrix](4096, 14336, 0, centre, None)}
            assert run.arm in names, (run.matrix, run.arm)

    def test_the_only_regex_matches_exactly_one_arm(self):
        """`--only` takes a regex and `run.arm` is fed to it. An arm name
        containing a regex metacharacter, or one that is a prefix of another,
        would silently probe two arms and bill the second to the first."""
        import re

        from tools.lora_regret.arms import MATRICES

        for run in probe_plan('config'):
            centre = 1e-4 if run.matrix == "e5" else None
            arms = MATRICES[run.matrix](4096, 14336, 0, centre, None)
            pattern = re.compile(run.only)
            matched = [a.name for a in arms if pattern.search(a.name)]
            assert matched == [run.arm], (run.matrix, run.method, matched)

    def test_gpu_counts_are_the_ones_the_real_sweep_uses(self):
        """The probe's timings are only estimates of the real arms if the real
        arms get the same GPUs. RL is 8, SFT FullFT is the registry's floor,
        every other SFT arm is 1."""
        from tools.lora_regret.models import get

        floor = get("llama3.1-8b").min_gpus_fullft()
        for run in probe_plan('config'):
            if run.metric == "accuracy":
                assert run.gpus == 8, run.arm
            elif run.method == "full":
                assert run.gpus == floor, run.arm
            else:
                assert run.gpus == 1, run.arm

    def test_every_run_carries_the_real_arms_rollout_count(self):
        """Without it the probe measures a per-step time and cannot turn it into
        an estimate of anything."""
        for run in probe_plan('config'):
            assert run.full_rollouts >= PROBE_ROLLOUTS, run.arm

    def test_the_short_horizon_matrix_extrapolates_to_its_own_cap(self):
        """e1short's arms carry num_rollout=100, so its full run IS 100 -- not
        the runbook's 2000 for the other Tulu3 stages."""
        assert FULL_RUN_ROLLOUTS["e1short"] == 100

    def test_the_openthoughts_ladder_extrapolates_to_one_epoch(self):
        """10,000 rows at rollout batch 32, ceilinged: (10000 + 31) // 32."""
        assert FULL_RUN_ROLLOUTS["e1ot"] == (10_000 + 31) // 32


class TestRolloutSeconds:
    LINE = (
        "[2026-07-31 09:15:22,101] train.py:261 - progress rollout={i}/2 "
        "completed={done}/3 remaining={left} elapsed=00:0{i}:00 last={last} "
        "avg=00:01:30 eta_remaining=00:00:00 eta_at=2026-07-31 09:20:00"
    )

    def _log(self, lasts):
        return "\n".join(
            self.LINE.format(i=i, done=i + 1, left=2 - i, last=last)
            for i, last in enumerate(lasts)
        )

    def test_reads_every_rollouts_own_duration(self):
        assert parse_rollout_seconds(self._log(["00:03:20", "00:01:30", "00:01:32"])) == [
            200.0, 90.0, 92.0
        ]

    def test_a_log_with_no_progress_line_yields_nothing(self):
        assert parse_rollout_seconds("nothing here") == []

    def test_multi_day_durations_parse(self):
        """`format_duration` switches to `2d 03:04:05` past 24 h, and an ETA on
        a 29,323-rollout arm crosses that."""
        assert parse_rollout_seconds(self._log(["2d 03:04:05"])) == [
            2 * 86400 + 3 * 3600 + 4 * 60 + 5
        ]


class TestReport:
    @staticmethod
    def _record(matrix, method, status="ok", seconds=600.0, rollout_seconds=None):
        """Uses a real planned arm name -- the report keys on it, because at
        config level one (task, method) has several rows."""
        arm = next(
            r.arm for r in probe_plan("method")
            if r.matrix == matrix and r.method == method
        )
        return {
            "arm": arm, "method": method, "matrix": matrix,
            "status": status, "seconds": seconds, "probe_rollouts": 3,
            "rollout_seconds": rollout_seconds if rollout_seconds is not None else [200.0, 90.0, 92.0],
            "full_rollouts": 2000, "gpus": 1, "wandb_project": f"{matrix}-proj",
            "metric": "nll", "test_nll": 1.2, "accuracy": None,
        }

    def test_the_steady_step_drops_the_first_rollout(self):
        """Rollout 1 carries compile, weight load and the first allocator
        growth. Averaging it in inflates a 2000-rollout estimate by hours."""
        text = format_report([self._record("e1", "lora")])
        assert "91" in text  # median of 90 and 92, not of 200/90/92

    def test_a_failed_probe_is_reported_as_failed_not_as_a_zero(self):
        text = format_report([self._record("e1", "full", status="failed",
                                           rollout_seconds=[])])
        assert "FAILED" in text
        assert "e1" in text and "full" in text

    def test_the_estimate_is_absent_when_no_rollout_completed(self):
        """An arm that died in startup has no per-step time, and printing one
        anyway would be inventing the number the probe exists to measure."""
        text = format_report([self._record("e1", "oft", rollout_seconds=[])])
        assert "?" in text

    def test_it_reports_every_probed_pair_even_the_missing_ones(self):
        """A pair absent from the ledger never ran. Silently omitting it makes
        a partial probe look complete -- the failure the sweep's own resume
        ledger is built to avoid."""
        text = format_report([self._record("e1", "lora")])
        assert "not run" in text

    def test_the_total_is_the_sum_over_planned_arms_not_over_probes(self):
        """The point of the probe: 3 rollouts x 24 runs tells you nothing about
        the campaign unless it is multiplied out by each arm count."""
        text = format_report([self._record("e1", "lora")])
        assert "campaign estimate" in text.lower()


class TestLedgerRowsCannotBeMistakenForMeasurements:
    def test_probe_rows_are_marked(self, tmp_path):
        """`analyze` reads any ledger it is pointed at. A 3-rollout row with a
        real-looking test_nll in a globbed ledger would win an argmin."""
        from tools.lora_regret.analyze import load_records

        path = tmp_path / "probe.jsonl"
        row = TestReport._record("e1", "lora")
        row.update({"lr": 2.5e-4, "seed": 0, "rank": 256, "target_modules": "x",
                    "dataset": "tulu3"})
        path.write_text(json.dumps(row) + "\n")
        loaded = load_records([path])
        assert loaded and loaded[0]["probe_rollouts"] == 3

    def test_analyze_refuses_a_ledger_of_probe_rows(self, tmp_path):
        import subprocess
        import sys
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        path = tmp_path / "probe.jsonl"
        rows = []
        for lr in (1.5e-4, 2.5e-4, 4.0e-4):
            row = TestReport._record("e1", "lora")
            row.update({"arm": f"lora-r256-all-lr{lr:g}-s0", "lr": lr, "seed": 0,
                        "rank": 256, "target_modules": "all", "dataset": "tulu3"})
            rows.append(row)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        proc = subprocess.run(
            [sys.executable, "-m", "tools.lora_regret.analyze", "argmins",
             "--ledgers", str(path), "--sigma", "0.001"],
            capture_output=True, text=True, cwd=repo_root,
        )
        assert proc.returncode == 4, proc.stdout + proc.stderr
        assert "probe" in proc.stderr.lower()


def test_probe_rollouts_is_short_enough_to_be_cheap_and_long_enough_to_average():
    """Two steady rollouts after dropping the first. One would give a per-step
    time with no spread; the report prints a median, which needs at least two."""
    assert PROBE_ROLLOUTS == 3


class TestPathLevel:
    """The default. One run per distinct code path, deduplicated ACROSS tasks."""

    def test_it_is_the_default_and_smaller_than_one_per_task_per_method(self):
        assert probe_plan() == probe_plan("path")
        assert len(probe_plan("path")) < len(probe_plan("method"))

    def test_two_tasks_running_identical_code_are_probed_once(self):
        """e4/full and e4place/full are the same script over the same data
        wrapping the same (empty) module set. Probing both proves nothing the
        first did not, at 8 GPUs a time."""
        from tools.lora_regret.probe import path_key

        runs = probe_plan("path")
        rl_full = [r for r in runs if r.metric == "accuracy" and r.method == "full"]
        assert len(rl_full) == 1, [r.arm for r in rl_full]
        # ...and the pair it stands for really is one path.
        from tools.lora_regret.arms import e4_arms, e4place_arms

        a = next(x for x in e4_arms() if x.method == "full")
        b = next(x for x in e4place_arms(4096, 14336) if x.method == "full")
        assert path_key("e4", a) == path_key("e4place", b)

    def test_target_modules_are_not_collapsed(self):
        """`linear_fc1` is Orbit's fused gate+up. Wrapping it is not the same
        code as wrapping `linear_qkv`, so attn/mlp/all stay separate paths --
        this is the axis that must NOT be deduplicated away."""
        labels = {r.label for r in probe_plan("path")}
        for modules in ("attn", "mlp", "all"):
            assert any(label.endswith(f"/{modules}") for label in labels), modules

    def test_it_covers_a_path_the_method_level_never_launched(self):
        """e4place's MLP placement under RL: absent from the 24, present here.
        Fewer runs AND more coverage, which is the whole point."""
        assert "rl/math_gsm8k/lora/mlp" in {r.label for r in probe_plan("path")}
        method_arms = {(r.matrix, r.arm) for r in probe_plan("method")}
        mlp = next(r for r in probe_plan("path") if r.label == "rl/math_gsm8k/lora/mlp")
        assert (mlp.matrix, mlp.arm) not in method_arms

    def test_datasets_stay_separate_even_though_the_code_is_shared(self):
        """Not a code difference but a shape one: OpenThoughts3 rows are ~62 KB
        against Tulu3's ~3 KB, a 20x sequence length that moves both memory and
        step time. Collapsing them would make the estimate meaningless."""
        labels = {r.label for r in probe_plan("path")}
        assert "sft/tulu3/lora/all" in labels
        assert "sft/openthoughts3/lora/all" in labels

    def test_the_cheapest_task_is_chosen_as_the_representative(self):
        """Same code either way, so probe it where the arm is shortest."""
        from tools.lora_regret.probe import FULL_RUN_ROLLOUTS

        run = next(r for r in probe_plan("path") if r.label == "sft/tulu3/lora/all")
        # e1short (100 rollouts) beats e1 and e5 (2000) for the same path.
        assert run.matrix == "e1short"
        assert FULL_RUN_ROLLOUTS[run.matrix] == 100

    def test_the_report_still_prints_every_task_and_method(self):
        """17 measurements, 24 rows. If a task row could not find its path's
        measurement it would read `not run`, which is the failure this guards."""
        records = []
        for run in probe_plan("path"):
            arm = next(a for a in _build_arms(run.matrix) if a.name == run.arm)
            records.append({
                "arm": run.arm, "method": run.method, "matrix": run.matrix,
                "status": "ok", "seconds": 500.0, "probe_rollouts": 3,
                "rollout_seconds": [200.0, 60.0, 61.0], "metric": run.metric,
                "dataset": arm.dataset, "target_modules": arm.target_modules,
            })
        text = format_report(records, "path")
        assert "not run" not in text
        for matrix in ("e1", "e1ot", "e1short", "e2", "e3", "e4", "e4place",
                       "e5scout", "e5"):
            assert matrix in text, matrix


def _build_arms(matrix):
    from tools.lora_regret.probe import _build

    return _build(matrix)


class TestTheGpuSplit:
    """Three scripts, one per GPU size, so each can be booked on a differently
    sized node. The split must partition the plan: an overlap bills a run twice
    and a gap means a path nobody runs."""

    SIZES = (1, 4, 8)

    def test_the_three_sizes_partition_the_plan(self):
        runs = probe_plan("path")
        by_size = {n: [r for r in runs if r.gpus == n] for n in self.SIZES}
        assert sum(len(v) for v in by_size.values()) == len(runs)
        assert {r.gpus for r in runs} == set(self.SIZES)

    def test_each_size_is_non_empty_so_no_script_is_a_no_op(self):
        for size in self.SIZES:
            assert [r for r in probe_plan("path") if r.gpus == size], size

    def test_the_eight_gpu_script_is_exactly_the_rl_paths(self):
        """The subset worth running first: every path in it has never executed
        in any form, while the SFT paths have a passing smoke behind them."""
        eight = [r for r in probe_plan("path") if r.gpus == 8]
        assert {r.metric for r in eight} == {"accuracy"}
        assert len(eight) == 7

    def test_the_four_gpu_script_is_exactly_the_sft_fullft_paths(self):
        four = [r for r in probe_plan("path") if r.gpus == 4]
        assert {r.method for r in four} == {"full"}
        assert {r.metric for r in four} == {"nll"}

    def test_the_one_gpu_script_is_every_peft_sft_path(self):
        one = [r for r in probe_plan("path") if r.gpus == 1]
        assert {r.method for r in one} == {"lora", "oft"}
        assert {r.metric for r in one} == {"nll"}

    @pytest.mark.parametrize("size", SIZES)
    def test_a_wrapper_exists_for_each_size_and_pins_only_that_size(self, size):
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[3]
        script = repo_root / f"scripts/lora_regret/coverage_probe_{size}gpu.sh"
        assert script.is_file(), script
        text = script.read_text(encoding="utf-8")
        assert f"ONLY_GPUS={size}" in text
        # Delegates rather than duplicating: three copies of the run loop would
        # drift, and the drift would be invisible until a node was booked.
        assert "coverage_probe.sh" in text

    def test_the_preflight_stage_of_each_wrapper_fits_its_node(self):
        """A 1-GPU reservation must not fail an audit demanding 8 cards it was
        never going to use."""
        from pathlib import Path

        from tools.lora_regret.preflight import STAGE_GPU_REQUIREMENTS

        repo_root = Path(__file__).resolve().parents[3]
        common = (repo_root / "scripts/lora_regret/coverage_probe.sh").read_text(
            encoding="utf-8"
        )
        for size, stage in ((1, "e1-lora"), (4, "e1-full"), (8, "e4")):
            assert f"{size}) PREFLIGHT_STAGE={stage}" in common or stage == "e4", size
            assert STAGE_GPU_REQUIREMENTS[stage] <= size, (stage, size)
