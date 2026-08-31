import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[6]
MODULE_PATH = REPO_ROOT / "scripts/slurm/setup/cu128/extract_pins.py"
SPEC = importlib.util.spec_from_file_location("orbit_cu128_extract_pins", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
EXTRACT_PINS = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = EXTRACT_PINS
SPEC.loader.exec_module(EXTRACT_PINS)

collect_pins = EXTRACT_PINS.collect_pins


def test_collect_pins_cross_checks_orbit_backend_refs(tmp_path: Path) -> None:
    repo = tmp_path / "orbit"
    sglang = tmp_path / "sglang"
    repo.mkdir()
    (sglang / "python").mkdir(parents=True)

    (repo / "pyproject.toml").write_text(
        """
[project]
name = "orbit"
version = "0.2.1"
requires-python = ">=3.12,<3.13"
dependencies = ["transformers==5.12.1"]

[tool.uv.sources]
sglang = { git = "https://github.com/Sphere-AI-Lab/sglang.git", rev = "a6fe249b3d56dde4bf275f98cc3d9f95813b0f44", subdirectory = "python" }
megatron-core = { git = "https://github.com/radixark/Megatron-LM.git", rev = "235952df607b3820716e5e67728a5ab470ca33ae" }
megatron-bridge = { git = "https://github.com/Sphere-AI-Lab/Megatron-Bridge.git", rev = "bb9372161e016b87dd87f7bb06d19794c31178f7" }
transformer-engine = { git = "https://github.com/NVIDIA/TransformerEngine.git", rev = "71bbefbf153418f943640df0f7373625dc93fa46" }
deep-ep = { git = "https://github.com/deepseek-ai/DeepEP.git", rev = "d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb" }

[tool.orbit.release.backend-pins.sglang]
source = "https://github.com/Sphere-AI-Lab/sglang.git"
tested-ref = "a6fe249b3d56dde4bf275f98cc3d9f95813b0f44"

[tool.orbit.release.backend-pins.megatron-core]
source = "https://github.com/radixark/Megatron-LM.git"
tested-ref = "235952df607b3820716e5e67728a5ab470ca33ae"

[tool.orbit.release.backend-pins.megatron-bridge]
source = "https://github.com/Sphere-AI-Lab/Megatron-Bridge.git"
tested-ref = "bb9372161e016b87dd87f7bb06d19794c31178f7"
""".strip()
    )
    (sglang / "python" / "pyproject.toml").write_text(
        """
[project]
name = "sglang"
dependencies = ["torch==2.11.0", "flashinfer-python==0.6.14"]
""".strip()
    )

    pins = collect_pins(repo, sglang)

    assert pins["CUDA_PROFILE"] == "cu128"
    assert pins["TORCH_VERSION"] == "2.11.0"
    assert pins["SGLANG_COMMIT"] == "a6fe249b3d56dde4bf275f98cc3d9f95813b0f44"
    assert pins["MEGATRON_COMMIT"] == "235952df607b3820716e5e67728a5ab470ca33ae"
    assert pins["MEGATRON_BRIDGE_COMMIT"] == "bb9372161e016b87dd87f7bb06d19794c31178f7"
    assert pins["NCCL_VERSION"] == "2.30.4"
    assert pins["DEEP_EP_SOURCE_URL"] == "https://github.com/deepseek-ai/DeepEP.git"
    assert pins["DEEP_EP_COMMIT"] == "d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb"


import pytest


def _write_mismatched_manifests(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "orbit"
    sglang = tmp_path / "sglang"
    repo.mkdir()
    (sglang / "python").mkdir(parents=True)
    (repo / "pyproject.toml").write_text(
        """
[project]
name = "orbit"
version = "0.2.1"
requires-python = ">=3.12,<3.13"
dependencies = ["transformers==5.12.1"]

[tool.uv.sources]
sglang = { git = "https://github.com/Sphere-AI-Lab/sglang.git", rev = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", subdirectory = "python" }
megatron-core = { git = "https://github.com/radixark/Megatron-LM.git", rev = "235952df607b3820716e5e67728a5ab470ca33ae" }
megatron-bridge = { git = "https://github.com/Sphere-AI-Lab/Megatron-Bridge.git", rev = "bb9372161e016b87dd87f7bb06d19794c31178f7" }
transformer-engine = { git = "https://github.com/NVIDIA/TransformerEngine.git", rev = "71bbefbf153418f943640df0f7373625dc93fa46" }
deep-ep = { git = "https://github.com/deepseek-ai/DeepEP.git", rev = "d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb" }

[tool.orbit.release.backend-pins.sglang]
source = "https://github.com/Sphere-AI-Lab/sglang.git"
tested-ref = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

[tool.orbit.release.backend-pins.megatron-core]
source = "https://github.com/radixark/Megatron-LM.git"
tested-ref = "235952df607b3820716e5e67728a5ab470ca33ae"

[tool.orbit.release.backend-pins.megatron-bridge]
source = "https://github.com/Sphere-AI-Lab/Megatron-Bridge.git"
tested-ref = "bb9372161e016b87dd87f7bb06d19794c31178f7"
""".strip()
    )
    (sglang / "python" / "pyproject.toml").write_text(
        """
[project]
name = "sglang"
dependencies = ["torch==2.11.0", "flashinfer-python==0.6.14"]
""".strip()
    )
    return repo, sglang


def test_collect_pins_rejects_disagreeing_sglang_refs(tmp_path: Path) -> None:
    repo, sglang = _write_mismatched_manifests(tmp_path)
    with pytest.raises(EXTRACT_PINS.PinError, match="sglang.*tested-ref"):
        collect_pins(repo, sglang)


def test_render_pins_is_deterministic_and_shell_safe() -> None:
    pins = {
        "TORCH_VERSION": "2.11.0",
        "CUDA_PROFILE": "cu128",
        "VALUE_WITH_SPACE": "one two",
    }
    hashes = {"pyproject.toml": "f" * 64}

    first = EXTRACT_PINS.render_pins(pins, hashes)
    second = EXTRACT_PINS.render_pins(dict(reversed(list(pins.items()))), hashes)

    assert first == second
    assert "AUTO-GENERATED" in first
    assert "TORCH_VERSION=2.11.0" in first
    assert "VALUE_WITH_SPACE='one two'" in first


def test_write_atomic_replaces_existing_content(tmp_path: Path) -> None:
    output = tmp_path / "pins.env"
    output.write_text("stale\n")

    EXTRACT_PINS.write_atomic(output, "fresh\n")

    assert output.read_text() == "fresh\n"
    assert list(tmp_path.iterdir()) == [output]


def test_check_mode_reports_drift_without_writing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "pins.env"
    output.write_text("stale\n")
    original = output.read_bytes()

    assert EXTRACT_PINS.run_check(output, "fresh\n") == 1
    assert output.read_bytes() == original
    assert "pins.env is stale" in capsys.readouterr().err
