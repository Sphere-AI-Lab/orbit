import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR_HELPERS = REPO_ROOT / "scripts" / "slurm" / "lib" / "run_dir.sh"


class TestPrepareRunDir(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory()
        self.temp_dir = Path(self._temp_dir.name)
        self.caller_dir = self.temp_dir / "caller" / "nested"
        self.caller_dir.mkdir(parents=True)

    def tearDown(self):
        self._temp_dir.cleanup()

    def _prepare(self, requested: str, env=None):
        default = self.temp_dir / "default-run"
        return subprocess.run(
            [
                "bash",
                "-c",
                'source "$1" && prepare_run_dir "$2" "$3"',
                "prepare-run-dir",
                str(RUN_DIR_HELPERS),
                requested,
                str(default),
            ],
            cwd=self.caller_dir,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_relative_override_is_canonicalized(self):
        result = self._prepare("../durable/run")

        self.assertEqual(result.returncode, 0, result.stderr)
        expected = (self.caller_dir / "../durable/run").resolve()
        self.assertEqual(result.stdout.strip(), str(expected))
        self.assertTrue(expected.is_dir())

    def test_comma_in_override_is_rejected(self):
        requested = self.temp_dir / "durable,runs" / "run"

        result = self._prepare(str(requested))

        self.assertEqual(result.returncode, 78)
        self.assertIn("RUN_DIR must not contain ','", result.stderr)
        self.assertFalse(requested.exists())

    def test_comma_in_canonical_symlink_target_is_rejected(self):
        target = self.temp_dir / "durable,runs" / "run"
        target.mkdir(parents=True)
        requested = self.temp_dir / "run-link"
        requested.symlink_to(target, target_is_directory=True)

        result = self._prepare(str(requested))

        self.assertEqual(result.returncode, 78)
        self.assertIn("RUN_DIR must not contain ','", result.stderr)

    def test_newline_in_override_is_rejected(self):
        requested = self.temp_dir / "durable" / "run\n"

        result = self._prepare(str(requested))

        self.assertEqual(result.returncode, 78)
        self.assertIn("RUN_DIR must not contain newlines", result.stderr)
        self.assertFalse(requested.exists())

    def test_nonempty_override_is_rejected(self):
        run_dir = self.temp_dir / "durable" / "existing-run"
        run_dir.mkdir(parents=True)
        (run_dir / "MANIFEST.json").write_text("{}\n")

        result = self._prepare(str(run_dir))

        self.assertEqual(result.returncode, 73)
        self.assertIn("RUN_DIR already contains run artifacts", result.stderr)

    def test_nonempty_symlink_target_is_rejected(self):
        target = self.temp_dir / "durable" / "existing-run"
        target.mkdir(parents=True)
        (target / "MANIFEST.json").write_text("{}\n")
        requested = self.temp_dir / "run-link"
        requested.symlink_to(target, target_is_directory=True)

        result = self._prepare(str(requested))

        self.assertEqual(result.returncode, 73)
        self.assertIn("RUN_DIR already contains run artifacts", result.stderr)

    def test_directory_scan_failure_is_rejected(self):
        run_dir = self.temp_dir / "durable" / "run"
        run_dir.mkdir(parents=True)
        fake_bin = self.temp_dir / "fake-bin"
        fake_bin.mkdir()
        fake_find = fake_bin / "find"
        fake_find.write_text("#!/bin/bash\nexit 2\n")
        fake_find.chmod(0o755)

        result = self._prepare(
            str(run_dir),
            env={"PATH": f"{fake_bin}:/usr/bin:/bin"},
        )

        self.assertEqual(result.returncode, 73)
        self.assertIn("could not inspect RUN_DIR", result.stderr)


if __name__ == "__main__":
    unittest.main()
