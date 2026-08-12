"""Behavioral contract for the two-GPU tiny-block OFT smoke wrapper."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
WRAPPER = REPO_ROOT / "scripts/lora_regret/smoke_oft_tiny_bs_2gpu.sh"

FAKE_SUCCESS = r'''#!/usr/bin/env bash
set -eu
printf '%s\t%s\t%s\t%s\t%s\t<%s>\t%s\t%s\t<%s>\t<%s>\t<%s>\t<%s>\n' \
  "${OFT_BLOCK_SIZE}" "${GPUS_PER_NODE}" \
  "${TENSOR_MODEL_PARALLEL_SIZE}" "${ROLLOUT_NUM_GPUS_PER_ENGINE}" \
  "${NUM_ROLLOUT}" "${SAVE_INTERVAL}" "${EVAL_INTERVAL}" "${ORBIT_RAY_LIFECYCLE}" \
  "${ORBIT_RAY_ADDRESS-}" "${RAY_ADDRESS-}" "${RAY_HEAD_PORT-}" \
  "${RAY_DASHBOARD_AGENT_LISTEN_PORT-}" >> "${CALL_LEDGER}"
printf '%s\n' \
  'weight_sync stage=update_weights_complete rank=0' \
  'progress rollout=2/2 completed=3/3 remaining=0 elapsed=00:00:03' \
  'Training driver exited with code 0' >> "${RUN_LOG}"
'''

FAKE_MISSING_ROLLOUT_MARKER = r'''#!/usr/bin/env bash
set -eu
printf '%s\t%s\t%s\t%s\t%s\t<%s>\t%s\t%s\t<%s>\t<%s>\t<%s>\t<%s>\n' \
  "${OFT_BLOCK_SIZE}" "${GPUS_PER_NODE}" \
  "${TENSOR_MODEL_PARALLEL_SIZE}" "${ROLLOUT_NUM_GPUS_PER_ENGINE}" \
  "${NUM_ROLLOUT}" "${SAVE_INTERVAL}" "${EVAL_INTERVAL}" "${ORBIT_RAY_LIFECYCLE}" \
  "${ORBIT_RAY_ADDRESS-}" "${RAY_ADDRESS-}" "${RAY_HEAD_PORT-}" \
  "${RAY_DASHBOARD_AGENT_LISTEN_PORT-}" >> "${CALL_LEDGER}"
printf '%s\n' \
  'weight_sync stage=update_weights_complete rank=0' \
  'Training driver exited with code 0' >> "${RUN_LOG}"
'''


def _write_launcher(tmp_path: Path, body: str = FAKE_SUCCESS) -> Path:
    launcher = tmp_path / "fake_launcher.sh"
    launcher.write_text(body, encoding="utf-8")
    launcher.chmod(0o755)
    return launcher


def _write_failing_tee(tmp_path: Path) -> Path:
    executable = tmp_path / "fake-bin" / "tee"
    executable.parent.mkdir()
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "set -uo pipefail\n"
        "/usr/bin/tee \"$@\"\n"
        "exit 23\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable.parent


def _run_wrapper(
    tmp_path: Path,
    run_root: Path,
    launcher: Path,
    *,
    cuda_visible_devices: str = "GPU-a,GPU-b",
    extra_environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = os.environ | {
        "RUN_ROOT": str(run_root),
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
        "OFT_TINY_SMOKE_LAUNCHER": str(launcher),
        "OFT_TINY_SMOKE_ARM_TIMEOUT": "",
        "CALL_LEDGER": str(tmp_path / "calls.tsv"),
        "SECRET_SENTINEL": "must-not-reach-environment-record",
        "ORBIT_RAY_ADDRESS": "inherited-orbit-ray-address",
        "RAY_ADDRESS": "inherited-ray-address",
        "RAY_HEAD_PORT": "31001",
        "RAY_DASHBOARD_AGENT_LISTEN_PORT": "31002",
    }
    if extra_environment is not None:
        env |= extra_environment
    return subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _status(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines())


class TestOftTinyBsTwoGpuSmoke(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._temporary_directory.name)

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def test_success_runs_all_blocks_with_the_two_gpu_topology_and_durable_evidence(self):
        """Wrong topology, ordering, or missing arm evidence must fail this test."""
        run_root = self.tmp_path / "run-root"
        run_root.mkdir()
        result = _run_wrapper(self.tmp_path, run_root, _write_launcher(self.tmp_path))

        self.assertEqual(result.returncode, 0, result.stderr)
        rows = (self.tmp_path / "calls.tsv").read_text(encoding="utf-8").splitlines()
        self.assertEqual(
            [row.split("\t") for row in rows],
            [
                ["4", "2", "2", "2", "3", "<>", "2", "private", "<>", "<>", "<>", "<>"],
                ["8", "2", "2", "2", "3", "<>", "2", "private", "<>", "<>", "<>", "<>"],
                ["16", "2", "2", "2", "3", "<>", "2", "private", "<>", "<>", "<>", "<>"],
            ],
        )
        for block_size in (4, 8, 16):
            arm_dir = run_root / f"bs{block_size}"
            self.assertEqual(_status(arm_dir / "completion.status")["final_exit_code"], "0")
            for name in ("console.log", "orbit.log", "environment.txt", "timings.txt"):
                self.assertTrue((arm_dir / name).is_file())
            self.assertNotIn(
                "must-not-reach-environment-record",
                (arm_dir / "environment.txt").read_text(encoding="utf-8"),
            )
        self.assertEqual(_status(run_root / "completion.status")["final_exit_code"], "0")

    def test_rejects_anything_other_than_two_visible_gpus_before_launching(self):
        """Accepting one GPU would run a topology the smoke does not validate."""
        run_root = self.tmp_path / "run-root"
        run_root.mkdir()
        result = _run_wrapper(
            self.tmp_path,
            run_root,
            _write_launcher(self.tmp_path),
            cuda_visible_devices="GPU-a",
        )

        self.assertEqual(result.returncode, 2)
        self.assertFalse((self.tmp_path / "calls.tsv").exists())
        self.assertFalse(any(run_root.iterdir()))

    def test_rejects_duplicate_and_empty_edge_gpu_masks_before_launching(self):
        """A duplicate or empty identifier does not reserve two distinct GPUs."""
        for cuda_visible_devices in ("GPU-a,GPU-a", ",GPU-a", "GPU-a,", "GPU-a,,GPU-b"):
            with self.subTest(cuda_visible_devices=cuda_visible_devices):
                case_path = self.tmp_path / f"case-{cuda_visible_devices.replace(',', '_')}"
                case_path.mkdir()
                run_root = case_path / "run-root"
                run_root.mkdir()
                result = _run_wrapper(
                    case_path,
                    run_root,
                    _write_launcher(case_path),
                    cuda_visible_devices=cuda_visible_devices,
                )

                self.assertEqual(result.returncode, 2)
                self.assertFalse((case_path / "calls.tsv").exists())
                self.assertFalse(any(run_root.iterdir()))

    def test_rejects_an_existing_arm_directory_before_launching_or_modifying_it(self):
        """Reusing an arm directory would corrupt the evidence from a prior run."""
        run_root = self.tmp_path / "run-root"
        arm_dir = run_root / "bs4"
        arm_dir.mkdir(parents=True)
        sentinel = arm_dir / "keep.txt"
        sentinel.write_text("previous evidence", encoding="utf-8")
        result = _run_wrapper(self.tmp_path, run_root, _write_launcher(self.tmp_path))

        self.assertEqual(result.returncode, 2)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "previous evidence")
        self.assertFalse((self.tmp_path / "calls.tsv").exists())
        self.assertFalse((run_root / "bs8").exists())

    def test_launcher_failure_is_recorded_and_does_not_prevent_the_later_arm(self):
        """Stopping at BS8 would hide whether the final tiny block can train."""
        run_root = self.tmp_path / "run-root"
        run_root.mkdir()
        launcher = _write_launcher(
            self.tmp_path,
            FAKE_SUCCESS + '\nif [[ "${OFT_BLOCK_SIZE}" == "8" ]]; then exit 7; fi\n',
        )
        result = _run_wrapper(self.tmp_path, run_root, launcher)

        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(
            [line.split("\t", 1)[0] for line in (self.tmp_path / "calls.tsv").read_text().splitlines()],
            ["4", "8", "16"],
        )
        bs8_status = _status(run_root / "bs8" / "completion.status")
        self.assertEqual(bs8_status["launcher_exit_code"], "7")
        self.assertEqual(bs8_status["final_exit_code"], "7")
        campaign_status = _status(run_root / "completion.status")
        self.assertEqual(campaign_status["launcher_exit_code"], "7")
        self.assertEqual(campaign_status["console_exit_code"], "0")
        self.assertEqual(campaign_status["verification_exit_code"], "0")
        self.assertEqual(campaign_status["final_exit_code"], "7")

    def test_missing_rollout_marker_fails_verification_after_a_zero_exit_launcher(self):
        """A clean process exit without completed-rollout evidence is not success."""
        run_root = self.tmp_path / "run-root"
        run_root.mkdir()
        launcher = _write_launcher(self.tmp_path, FAKE_MISSING_ROLLOUT_MARKER)
        result = _run_wrapper(self.tmp_path, run_root, launcher)

        self.assertNotEqual(result.returncode, 0)
        bs4_status = _status(run_root / "bs4" / "completion.status")
        self.assertEqual(bs4_status["launcher_exit_code"], "0")
        self.assertEqual(bs4_status["verification_exit_code"], "1")
        self.assertEqual(bs4_status["final_exit_code"], "1")
        campaign_status = _status(run_root / "completion.status")
        self.assertEqual(campaign_status["launcher_exit_code"], "0")
        self.assertEqual(campaign_status["console_exit_code"], "0")
        self.assertEqual(campaign_status["verification_exit_code"], "1")
        self.assertEqual(campaign_status["final_exit_code"], "1")

    def test_console_write_failure_fails_the_arm_while_preserving_launcher_success(self):
        """Discarding tee's exit code would falsely mark lost console evidence as success."""
        run_root = self.tmp_path / "run-root"
        run_root.mkdir()
        fake_bin = _write_failing_tee(self.tmp_path)
        result = _run_wrapper(
            self.tmp_path,
            run_root,
            _write_launcher(self.tmp_path),
            extra_environment={"PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"},
        )

        self.assertEqual(result.returncode, 23)
        bs4_status = _status(run_root / "bs4" / "completion.status")
        self.assertEqual(bs4_status["launcher_exit_code"], "0")
        self.assertEqual(bs4_status["console_exit_code"], "23")
        self.assertEqual(bs4_status["verification_exit_code"], "0")
        self.assertEqual(bs4_status["final_exit_code"], "23")
        campaign_status = _status(run_root / "completion.status")
        self.assertEqual(campaign_status["launcher_exit_code"], "0")
        self.assertEqual(campaign_status["console_exit_code"], "23")
        self.assertEqual(campaign_status["verification_exit_code"], "0")
        self.assertEqual(campaign_status["final_exit_code"], "23")
