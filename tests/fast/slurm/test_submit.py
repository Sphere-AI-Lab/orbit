import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DIR_HELPERS = REPO_ROOT / "scripts" / "slurm" / "lib" / "run_dir.sh"


def _prepare(tmp_path: Path, requested: str):
    caller_dir = tmp_path / "caller" / "nested"
    caller_dir.mkdir(parents=True)
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1" && prepare_run_dir "$2" "$3"',
            "prepare-run-dir",
            str(RUN_DIR_HELPERS),
            requested,
            str(tmp_path / "default-run"),
        ],
        cwd=caller_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    return result, caller_dir


def test_relative_override_is_canonicalized(tmp_path):
    result, caller_dir = _prepare(tmp_path, "../durable/run")

    expected = (caller_dir / "../durable/run").resolve()
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(expected)
    assert expected.is_dir()


def test_comma_in_override_is_rejected(tmp_path):
    result, _ = _prepare(tmp_path, str(tmp_path / "durable,runs" / "run"))

    assert result.returncode == 78
    assert "RUN_DIR must not contain ','" in result.stderr


def test_nonempty_override_is_rejected(tmp_path):
    run_dir = tmp_path / "durable" / "existing-run"
    run_dir.mkdir(parents=True)
    (run_dir / "MANIFEST.json").write_text("{}\n")

    result, _ = _prepare(tmp_path, str(run_dir))

    assert result.returncode == 73
    assert "RUN_DIR already contains run artifacts" in result.stderr
