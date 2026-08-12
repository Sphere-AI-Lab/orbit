"""Behavioral contract for the two-GPU tiny-block OFT smoke wrapper."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
WRAPPER = REPO_ROOT / "scripts/lora_regret/smoke_oft_tiny_bs_2gpu.sh"

FAKE_SUCCESS = r'''#!/usr/bin/env bash
set -eu
printf '%s\t%s\t%s\t%s\t%s\t<%s>\t%s\t%s\t<%s>\t<%s>\t<%s>\t<%s>\t<%s>\t%s\n' \
  "${OFT_BLOCK_SIZE}" "${GPUS_PER_NODE}" \
  "${TENSOR_MODEL_PARALLEL_SIZE}" "${ROLLOUT_NUM_GPUS_PER_ENGINE}" \
  "${NUM_ROLLOUT}" "${SAVE_INTERVAL}" "${EVAL_INTERVAL}" "${ORBIT_RAY_LIFECYCLE}" \
  "${ORBIT_RAY_ADDRESS-}" "${RAY_ADDRESS-}" "${RAY_HEAD_PORT-}" \
  "${RAY_DASHBOARD_AGENT_LISTEN_PORT-}" "${MASTER_ADDR-}" "${WANDB_DIR}" >> "${CALL_LEDGER}"
printf '%s\n' \
  'weight_sync stage=update_weights_complete rank=0' \
  'progress rollout=2/2 completed=3/3 remaining=0 elapsed=00:00:03' \
  'Training driver exited with code 0' >> "${RUN_LOG}"
'''

FAKE_MISSING_ROLLOUT_MARKER = r'''#!/usr/bin/env bash
set -eu
printf '%s\t%s\t%s\t%s\t%s\t<%s>\t%s\t%s\t<%s>\t<%s>\t<%s>\t<%s>\t<%s>\t%s\n' \
  "${OFT_BLOCK_SIZE}" "${GPUS_PER_NODE}" \
  "${TENSOR_MODEL_PARALLEL_SIZE}" "${ROLLOUT_NUM_GPUS_PER_ENGINE}" \
  "${NUM_ROLLOUT}" "${SAVE_INTERVAL}" "${EVAL_INTERVAL}" "${ORBIT_RAY_LIFECYCLE}" \
  "${ORBIT_RAY_ADDRESS-}" "${RAY_ADDRESS-}" "${RAY_HEAD_PORT-}" \
  "${RAY_DASHBOARD_AGENT_LISTEN_PORT-}" "${MASTER_ADDR-}" "${WANDB_DIR}" >> "${CALL_LEDGER}"
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


def _write_racing_mkdir(tmp_path: Path, arm_path: Path, attacker_path: Path) -> Path:
    executable = tmp_path / "race-bin" / "mkdir"
    executable.parent.mkdir()
    real_mkdir = shutil.which("mkdir")
    if real_mkdir is None:
        raise RuntimeError("mkdir is required by this test")
    executable.write_text(
        "#!/usr/bin/env bash\n"
        "set -eu\n"
        "for argument in \"$@\"; do\n"
        "  case \"${argument}\" in\n"
        "    \"${RACE_ARM}\"|\"${RACE_ARM}\"/*)\n"
        "      if [[ ! -e \"${RACE_ARM}\" && ! -L \"${RACE_ARM}\" ]]; then\n"
        "        ln -s \"${RACE_TARGET}\" \"${RACE_ARM}\"\n"
        "      fi\n"
        "      break\n"
        "      ;;\n"
        "  esac\n"
        "done\n"
        f'exec "{real_mkdir}" "$@"\n',
        encoding="utf-8",
    )
    executable.chmod(0o755)
    return executable.parent


def _wrapper_environment(
    tmp_path: Path,
    run_root: Path,
    launcher: Path,
    *,
    cuda_visible_devices: str = "GPU-a,GPU-b",
    extra_environment: dict[str, str] | None = None,
) -> dict[str, str]:
    env = os.environ | {
        "RUN_ROOT": str(run_root),
        "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
        "OFT_TINY_SMOKE_LAUNCHER": str(launcher),
        "OFT_TINY_SMOKE_ARM_TIMEOUT": "",
        "CALL_LEDGER": str(tmp_path / "calls.tsv"),
        "SECRET_SENTINEL": "must-not-reach-environment-record",
        "MASTER_ADDR": "inherited-master-address",
        "ORBIT_RAY_ADDRESS": "inherited-orbit-ray-address",
        "RAY_ADDRESS": "inherited-ray-address",
        "RAY_HEAD_PORT": "31001",
        "RAY_DASHBOARD_AGENT_LISTEN_PORT": "31002",
    }
    if extra_environment is not None:
        env |= extra_environment
    return env


def _run_wrapper(
    tmp_path: Path,
    run_root: Path,
    launcher: Path,
    *,
    cuda_visible_devices: str = "GPU-a,GPU-b",
    extra_environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=REPO_ROOT,
        env=_wrapper_environment(
            tmp_path,
            run_root,
            launcher,
            cuda_visible_devices=cuda_visible_devices,
            extra_environment=extra_environment,
        ),
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
                ["4", "2", "2", "2", "3", "<>", "2", "private", "<>", "<>", "<>", "<>", "<>", str(run_root / "bs4/wandb")],
                ["8", "2", "2", "2", "3", "<>", "2", "private", "<>", "<>", "<>", "<>", "<>", str(run_root / "bs8/wandb")],
                ["16", "2", "2", "2", "3", "<>", "2", "private", "<>", "<>", "<>", "<>", "<>", str(run_root / "bs16/wandb")],
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

    def test_atomic_arm_reservation_refuses_a_path_inserted_after_preflight(self):
        """A racing symlink must not redirect evidence outside the run root."""
        run_root = self.tmp_path / "run-root"
        run_root.mkdir()
        attacker_path = self.tmp_path / "attacker"
        attacker_path.mkdir()
        arm_path = run_root / "bs4"
        fake_bin = _write_racing_mkdir(self.tmp_path, arm_path, attacker_path)

        result = _run_wrapper(
            self.tmp_path,
            run_root,
            _write_launcher(self.tmp_path),
            extra_environment={
                "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
                "RACE_ARM": str(arm_path),
                "RACE_TARGET": str(attacker_path),
            },
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.tmp_path / "calls.tsv").exists())
        self.assertEqual(list(attacker_path.iterdir()), [])
        self.assertTrue(arm_path.is_symlink())
        self.assertFalse((run_root / "bs8").exists())
        self.assertFalse((run_root / "completion.status").exists())

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

    def test_arm_status_publication_failure_stops_and_marks_campaign_evidence_failure(self):
        """Lost atomic status evidence must not be reported as a successful campaign."""
        run_root = self.tmp_path / "run-root"
        run_root.mkdir()
        failed_status = run_root / "bs4" / "completion.status"
        result = _run_wrapper(
            self.tmp_path,
            run_root,
            _write_launcher(self.tmp_path),
            extra_environment={"OFT_TINY_SMOKE_TEST_FAIL_STATUS": str(failed_status)},
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(failed_status.exists())
        self.assertEqual(
            [line.split("\t", 1)[0] for line in (self.tmp_path / "calls.tsv").read_text().splitlines()],
            ["4"],
        )
        self.assertFalse((run_root / "bs8").exists())
        campaign_status = _status(run_root / "completion.status")
        self.assertNotEqual(campaign_status["evidence_exit_code"], "0")
        self.assertEqual(campaign_status["final_exit_code"], campaign_status["evidence_exit_code"])

    def _assert_signal_interrupts_active_launcher(
        self, wrapper_signal: signal.Signals, expected_returncode: int, expected_status: str
    ) -> None:
        run_root = self.tmp_path / "run-root"
        run_root.mkdir()
        ready_file = self.tmp_path / "launcher.ready"
        terminated_file = self.tmp_path / "launcher.terminated"
        launcher = _write_launcher(
            self.tmp_path,
            r'''#!/usr/bin/env bash
set -u
child_pid=
terminate() {
  printf 'terminated\n' > "${TERMINATED_FILE}"
  if [[ -n "${child_pid}" ]]; then
    kill -TERM "${child_pid}" 2>/dev/null || :
    wait "${child_pid}" 2>/dev/null || :
  fi
  exit 143
}
trap terminate TERM
printf '%s\n' "${OFT_BLOCK_SIZE}" >> "${CALL_LEDGER}"
printf 'ready\n' > "${READY_FILE}"
sleep 30 &
child_pid=$!
wait "${child_pid}"
''',
        )
        process = subprocess.Popen(
            ["bash", str(WRAPPER)],
            cwd=REPO_ROOT,
            env=_wrapper_environment(
                self.tmp_path,
                run_root,
                launcher,
                extra_environment={
                    "READY_FILE": str(ready_file),
                    "TERMINATED_FILE": str(terminated_file),
                },
            ),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        stdout = ""
        stderr = ""
        communicated = False
        try:
            deadline = time.monotonic() + 5
            while not ready_file.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(ready_file.exists(), "launcher did not become ready")
            process.send_signal(wrapper_signal)
            stdout, stderr = process.communicate(timeout=5)
            communicated = True
        finally:
            if process.poll() is None or not communicated:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process.poll() is None:
                process.kill()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass

        self.assertEqual(process.returncode, expected_returncode, f"stdout={stdout}\nstderr={stderr}")
        self.assertTrue(terminated_file.exists())
        self.assertEqual((self.tmp_path / "calls.tsv").read_text().splitlines(), ["4"])
        self.assertFalse((run_root / "bs8").exists())
        arm_status = _status(run_root / "bs4" / "completion.status")
        self.assertEqual(arm_status["interrupted"], expected_status)
        self.assertEqual(arm_status["launcher_exit_code"], "143")
        self.assertEqual(arm_status["console_exit_code"], "0")
        self.assertEqual(arm_status["final_exit_code"], str(expected_returncode))
        campaign_status = _status(run_root / "completion.status")
        self.assertEqual(campaign_status["interrupted"], expected_status)
        self.assertEqual(campaign_status["final_exit_code"], str(expected_returncode))

    def test_term_interrupts_the_active_launcher_and_publishes_nonzero_statuses(self):
        """Termination must be forwarded and must not allow a later arm to start."""
        self._assert_signal_interrupts_active_launcher(signal.SIGTERM, 143, "TERM")

    def test_int_uses_term_for_child_cleanup_but_records_the_int_exit_code(self):
        """Async children may ignore INT, so cleanup must use TERM while reporting 130."""
        self._assert_signal_interrupts_active_launcher(signal.SIGINT, 130, "INT")

    def test_term_escalates_to_kill_after_a_bounded_grace_period(self):
        """A TERM-ignoring launcher must not prevent interrupted status publication."""
        run_root = self.tmp_path / "run-root"
        run_root.mkdir()
        ready_file = self.tmp_path / "launcher.ready"
        launcher = _write_launcher(
            self.tmp_path,
            r'''#!/usr/bin/env bash
exec "${PYTHON_FOR_TEST}" -c '
import os
import signal
import time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
with open(os.environ["CALL_LEDGER"], "a", encoding="utf-8") as ledger:
    ledger.write(os.environ["OFT_BLOCK_SIZE"] + "\n")
with open(os.environ["READY_FILE"], "w", encoding="utf-8") as ready:
    ready.write("ready\n")
time.sleep(30)
'
''',
        )
        process = subprocess.Popen(
            ["bash", str(WRAPPER)],
            cwd=REPO_ROOT,
            env=_wrapper_environment(
                self.tmp_path,
                run_root,
                launcher,
                extra_environment={
                    "OFT_TINY_SMOKE_SIGNAL_GRACE_SECONDS": "1",
                    "PYTHON_FOR_TEST": os.sys.executable,
                    "READY_FILE": str(ready_file),
                },
            ),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        stdout = ""
        stderr = ""
        communicated = False
        started = time.monotonic()
        try:
            deadline = started + 5
            while not ready_file.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(ready_file.exists(), "launcher did not become ready")
            process.terminate()
            stdout, stderr = process.communicate(timeout=5)
            communicated = True
        finally:
            if process.poll() is None or not communicated:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process.poll() is None:
                process.kill()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass

        self.assertLess(time.monotonic() - started, 4)
        self.assertEqual(process.returncode, 143, f"stdout={stdout}\nstderr={stderr}")
        self.assertEqual((self.tmp_path / "calls.tsv").read_text().splitlines(), ["4"])
        self.assertFalse((run_root / "bs8").exists())
        arm_status = _status(run_root / "bs4" / "completion.status")
        self.assertEqual(arm_status["interrupted"], "TERM")
        self.assertEqual(arm_status["launcher_exit_code"], "137")
        self.assertEqual(arm_status["final_exit_code"], "143")

    def test_term_kills_a_surviving_launcher_group_after_its_leader_exits(self):
        """A descendant holding the FIFO must not outlive an exited launcher leader."""
        run_root = self.tmp_path / "run-root"
        run_root.mkdir()
        ready_file = self.tmp_path / "launcher.ready"
        child_pid_file = self.tmp_path / "launcher-child.pid"
        launcher = _write_launcher(
            self.tmp_path,
            r'''#!/usr/bin/env bash
set -u
trap 'exit 0' TERM
"${PYTHON_FOR_TEST}" -c '
import os
import signal
import time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
with open(os.environ["CHILD_PID_FILE"], "w", encoding="utf-8") as pid_file:
    pid_file.write(str(os.getpid()))
with open(os.environ["READY_FILE"], "w", encoding="utf-8") as ready:
    ready.write("ready\n")
time.sleep(30)
' &
wait
''',
        )
        process = subprocess.Popen(
            ["bash", str(WRAPPER)],
            cwd=REPO_ROOT,
            env=_wrapper_environment(
                self.tmp_path,
                run_root,
                launcher,
                extra_environment={
                    "OFT_TINY_SMOKE_SIGNAL_GRACE_SECONDS": "1",
                    "PYTHON_FOR_TEST": os.sys.executable,
                    "READY_FILE": str(ready_file),
                    "CHILD_PID_FILE": str(child_pid_file),
                },
            ),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        stdout = ""
        stderr = ""
        child_pid = None
        communicated = False
        try:
            deadline = time.monotonic() + 5
            while not ready_file.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(ready_file.exists(), "launcher descendant did not become ready")
            child_pid = int(child_pid_file.read_text(encoding="utf-8"))
            process.terminate()
            stdout, stderr = process.communicate(timeout=5)
            communicated = True
        finally:
            if process.poll() is None or not communicated:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if child_pid is not None:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            if process.poll() is None:
                process.kill()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                pass

        self.assertEqual(process.returncode, 143, f"stdout={stdout}\nstderr={stderr}")
        self.assertFalse((run_root / "bs8").exists())
        arm_status = _status(run_root / "bs4" / "completion.status")
        self.assertEqual(arm_status["interrupted"], "TERM")
        self.assertEqual(arm_status["final_exit_code"], "143")
