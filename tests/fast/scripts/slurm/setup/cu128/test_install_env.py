import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[6]
SCRIPT = REPO_ROOT / "scripts/slurm/setup/cu128/install_env.sh"


def write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    orbit = tmp_path / "orbit"
    orbit.mkdir()
    (orbit / "pyproject.toml").write_text("[project]\nname='orbit'\nversion='0.0.0'\n")
    pins = tmp_path / "pins.env"
    pins.write_text(
        "\n".join(
            [
                "CUDA_PROFILE=cu128",
                "CUDA_TOOLKIT_VERSION=12.8",
                "PYTHON_VERSION=3.12",
                "TORCH_VERSION=2.11.0",
                "TORCHVISION_VERSION=0.26.0",
                "TORCHAUDIO_VERSION=2.11.0",
                "TORCH_INDEX_URL=https://example.test/torch",
                "FLASHINFER_INDEX_URL=https://example.test/flashinfer",
                "SGLANG_WHEEL_INDEX_URL=https://example.test/sglang",
                "SGLANG_SOURCE_URL=https://example.test/sglang.git",
                f"SGLANG_COMMIT={'1' * 40}",
                "MEGATRON_SOURCE_URL=https://example.test/megatron.git",
                f"MEGATRON_COMMIT={'2' * 40}",
                "MEGATRON_BRIDGE_SOURCE_URL=https://example.test/bridge.git",
                f"MEGATRON_BRIDGE_COMMIT={'3' * 40}",
                "TRANSFORMER_ENGINE_SOURCE_URL=https://example.test/te.git",
                f"TRANSFORMER_ENGINE_COMMIT={'4' * 40}",
                "APEX_SOURCE_URL=https://example.test/apex.git",
                f"APEX_COMMIT={'5' * 40}",
            ]
        )
        + "\n"
    )
    return orbit, pins


def invoke(tmp_path: Path, *extra: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    orbit, pins = write_fixture(tmp_path)
    command = [
        "bash",
        str(SCRIPT),
        "--workspace",
        str(tmp_path),
        "--orbit-root",
        str(orbit),
        "--pins",
        str(pins),
        "--env-prefix",
        str(tmp_path / "env"),
        "--source-root",
        str(tmp_path / "sources"),
        *extra,
    ]
    return subprocess.run(command, capture_output=True, text=True, env=env)


def test_help_describes_h200_scheduler_requirement() -> None:
    result = subprocess.run(["bash", str(SCRIPT), "--help"], capture_output=True, text=True)
    assert result.returncode == 0
    assert "H200" in result.stdout
    assert "Slurm allocation" in result.stdout


def test_dry_run_prints_twelve_stage_plan_without_creating_prefix(tmp_path: Path) -> None:
    result = invoke(tmp_path, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "[01/12]" in result.stdout
    assert "[12/12]" in result.stdout
    assert "torch==2.11.0+cu128" in result.stdout
    assert "--full-h200" in result.stdout
    assert not (tmp_path / "env").exists()
    assert not (tmp_path / "sources").exists()


def test_real_install_refuses_to_run_outside_slurm(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.pop("SLURM_JOB_ID", None)
    result = invoke(tmp_path, env=environment)
    assert result.returncode == 2
    assert "must run inside a Slurm allocation" in result.stderr
    assert not (tmp_path / "env").exists()


def test_unsafe_environment_prefix_is_rejected(tmp_path: Path) -> None:
    orbit, pins = write_fixture(tmp_path)
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--workspace",
            str(tmp_path),
            "--orbit-root",
            str(orbit),
            "--pins",
            str(pins),
            "--env-prefix",
            "/",
            "--source-root",
            str(tmp_path / "sources"),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "unsafe environment prefix" in result.stderr


def test_preflight_only_accepts_fake_h200_cuda128_tools(tmp_path: Path) -> None:
    tool_dir = tmp_path / "bin"
    tool_dir.mkdir()
    tools = {
        "conda": "#!/bin/sh\necho 'conda 25.0'\n",
        "uv": "#!/bin/sh\necho 'uv 0.8'\n",
        "nvidia-smi": "#!/bin/sh\necho 'NVIDIA H200'\n",
        "nvcc": "#!/bin/sh\necho 'Cuda compilation tools, release 12.8, V12.8.93'\n",
    }
    for name, content in tools.items():
        path = tool_dir / name
        path.write_text(content)
        path.chmod(0o755)
    environment = os.environ.copy()
    environment["SLURM_JOB_ID"] = "12345"
    environment["PATH"] = f"{tool_dir}:{environment['PATH']}"
    result = invoke(tmp_path, "--preflight-only", env=environment)
    assert result.returncode == 0, result.stderr
    assert "preflight passed for Slurm job 12345" in result.stdout
    assert not (tmp_path / "env").exists()
