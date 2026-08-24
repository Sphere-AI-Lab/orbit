import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tools.adapter_runtime_compare import run_compare


class ParseOutputTest(unittest.TestCase):
    def test_json_output_writes_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            log_path = run_dir / "run.log"
            log_path.write_text("perf 0: {'perf/rollout_time': 1.0}\n")
            (run_dir / "run.json").write_text(
                json.dumps(
                    {
                        "run_id": "run",
                        "branch": "runtime",
                        "model": "qwen3_4b",
                        "precision": "bf16",
                        "peft": "oft",
                        "mode": "sync",
                        "repeat": 0,
                        "gpu_ids": [0],
                        "num_rollout": 1,
                        "eval_enabled": False,
                        "env": {"RUN_LOG": str(log_path)},
                    }
                )
            )
            (run_dir / "status.json").write_text(json.dumps({"status": "ok", "returncode": 0, "wall_s": 1.0}))
            output = root / "summary.json"

            with contextlib.redirect_stdout(io.StringIO()):
                rc = run_compare.parse_runs(argparse.Namespace(path=str(root), format="json", output=str(output)))

            self.assertEqual(rc, 0)
            rows = json.loads(output.read_text())
            self.assertEqual(rows[0]["run_id"], "run")


class CaseConfigTest(unittest.TestCase):
    def test_qwen25_dense_lora_uses_csgmv_backend(self) -> None:
        case = next(case for case in run_compare.CASES if case.key == "qwen25_05b_bf16_lora")

        self.assertEqual(case.lora_backend, "csgmv")


def make_args(**overrides):
    base = dict(
        profile="main",
        branches=None,
        models=None,
        precisions=None,
        pefts=None,
        modes=None,
        repeats=1,
        num_rollout=None,
        eval=None,
        eval_interval=10,
        skip_eval_before_train=True,
        batch_profile="bench",
        output_dir="/tmp/adapter_runtime_compare_test",
        campaign="test",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def qwen3_4b_bf16_oft_case():
    return next(case for case in run_compare.CASES if case.key == "qwen3_4b_bf16_oft")


def make_job(case, mode):
    return run_compare.make_job(
        output_root=Path("/tmp/adapter_runtime_compare_test"),
        branch=run_compare.BRANCHES["runtime"],
        case=case,
        mode=mode,
        repeat=0,
        gpu_ids=(0, 1, 2, 3),
        num_rollout=3,
        eval_enabled=False,
        eval_interval=10,
        skip_eval_before_train=True,
        batch_profile="bench",
    )


class ArmSelectionTest(unittest.TestCase):
    def test_arm_registry_covers_modes(self) -> None:
        self.assertEqual(set(run_compare.MODES), set(run_compare.ARMS))
        self.assertIn("async_fullft", run_compare.MODES)

    def test_default_modes_exclude_fullft(self) -> None:
        modes = run_compare.selected_modes(make_args())

        self.assertEqual(modes, ["sync", "async", "async_db"])

    def test_explicit_fullft_mode_is_selectable(self) -> None:
        modes = run_compare.selected_modes(make_args(modes="async,async_fullft"))

        self.assertEqual(modes, ["async", "async_fullft"])

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            run_compare.selected_modes(make_args(modes="fullft"))


class FullftArmTest(unittest.TestCase):
    def test_fullft_job_uses_dedicated_launcher_and_effective_peft(self) -> None:
        case = qwen3_4b_bf16_oft_case()

        job = make_job(case, "async_fullft")

        self.assertEqual(job.peft, "none")
        self.assertEqual(job.script, case.fullft_script)
        self.assertIn("_qwen3_4b_bf16_none_async_fullft_", job.run_id)

    def test_fullft_launcher_exists_in_repo(self) -> None:
        case = qwen3_4b_bf16_oft_case()

        self.assertTrue((run_compare.REPO_ROOT / case.fullft_script).is_file())

    def test_fullft_env_disables_peft_and_double_buffer(self) -> None:
        job = make_job(qwen3_4b_bf16_oft_case(), "async_fullft")

        env = run_compare.job_env(job)

        self.assertEqual(env["PEFT_METHOD"], "none")
        self.assertEqual(env["ADAPTER_DOUBLE_BUFFER"], "0")
        self.assertEqual(env["ORBIT_COLOCATE"], "0")
        self.assertTrue(env["ORBIT_ENTRYPOINT"].endswith("train_async.py"))
        for key in run_compare.PEFT_ONLY_ENV_KEYS:
            self.assertNotIn(key, env)

    def test_async_db_env_unchanged(self) -> None:
        case = qwen3_4b_bf16_oft_case()

        job = make_job(case, "async_db")
        env = run_compare.job_env(job)

        self.assertEqual(job.script, case.script)
        self.assertEqual(job.peft, "oft")
        self.assertIn("_qwen3_4b_bf16_oft_async_db_", job.run_id)
        self.assertEqual(env["PEFT_METHOD"], "oft")
        self.assertEqual(env["ADAPTER_DOUBLE_BUFFER"], "1")
        self.assertEqual(env["TARGET_MODULES"], case.target_modules)
        self.assertTrue(env["ORBIT_ENTRYPOINT"].endswith("train_async.py"))

    def test_sync_env_unchanged(self) -> None:
        case = qwen3_4b_bf16_oft_case()

        env = run_compare.job_env(make_job(case, "sync"))

        self.assertEqual(env["ORBIT_COLOCATE"], "1")
        self.assertEqual(env["GPUS_PER_NODE"], str(case.gpu_total))
        self.assertEqual(env["ADAPTER_DOUBLE_BUFFER"], "0")
        self.assertNotIn("ROLLOUT_NUM_GPUS", env)
        self.assertTrue(env["ORBIT_ENTRYPOINT"].endswith("train.py"))


class BuildWavesFullftTest(unittest.TestCase):
    def test_fullft_waves_only_cover_cases_with_fullft_launcher(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            waves = run_compare.build_waves(make_args(modes="async_fullft"))

        jobs = [job for wave in waves for job in wave]
        self.assertTrue(jobs)
        for job in jobs:
            self.assertEqual(job.mode, "async_fullft")
            self.assertEqual(job.peft, "none")
            self.assertIsNotNone(job.case.fullft_script)
            self.assertEqual(job.script, job.case.fullft_script)

    def test_fullft_without_supporting_case_raises(self) -> None:
        # qwen25_05b now has a fullft_script on its oft case (Task 2), so pin
        # this to the lora case, which still has none, to keep testing the
        # "no supporting case" SystemExit path.
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                run_compare.build_waves(make_args(modes="async_fullft", models="qwen25_05b", pefts="lora"))

    def test_default_waves_do_not_include_fullft(self) -> None:
        waves = run_compare.build_waves(make_args())

        modes = {job.mode for wave in waves for job in wave}
        self.assertEqual(modes, {"sync", "async", "async_db"})


def test_case_scripts_exist():
    from tools.adapter_runtime_compare import run_compare

    missing = []
    for case in run_compare.CASES:
        for attr in ("script", "fullft_script"):
            rel = getattr(case, attr, None)
            if rel and not (run_compare.REPO_ROOT / rel).exists():
                missing.append(f"{case.model}/{case.precision}: {rel}")
    assert not missing, f"CASES reference missing launchers: {missing}"


def test_a1_rungs_have_fullft_arms():
    from tools.adapter_runtime_compare import run_compare

    by_key = {(c.model, c.precision, c.peft): c for c in run_compare.CASES}
    for key in [("qwen25_05b", "bf16", "oft"), ("qwen25_3b", "bf16", "oft"),
                ("qwen3_4b", "bf16", "oft"), ("qwen3_30b", "bf16", "oft")]:
        case = by_key.get(key)
        assert case is not None, f"missing A1 case {key}"
        assert case.fullft_script, f"A1 case {key} has no fullft_script"
        assert (run_compare.REPO_ROOT / case.fullft_script).exists()


if __name__ == "__main__":
    unittest.main()


def test_rollout_gpus_per_engine_defaults_to_rollout_gpus_async():
    case = qwen3_4b_bf16_oft_case()
    env = run_compare.job_env(make_job(case, "async"))
    assert env["ROLLOUT_NUM_GPUS_PER_ENGINE"] == str(case.rollout_gpus_async)


def test_rollout_gpus_per_engine_override_is_honored():
    import dataclasses

    case = dataclasses.replace(qwen3_4b_bf16_oft_case(), rollout_gpus_per_engine=1)
    env = run_compare.job_env(make_job(case, "async"))
    assert env["ROLLOUT_NUM_GPUS_PER_ENGINE"] == "1"
    assert env["ROLLOUT_NUM_GPUS"] == str(case.rollout_gpus_async)
