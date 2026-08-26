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
sglang = { git = "https://github.com/Sphere-AI-Lab/sglang.git", rev = "51845dc4acca94507ab184b007c8fcfd656b191f", subdirectory = "python" }
megatron-core = { git = "https://github.com/Sphere-AI-Lab/Megatron-LM.git", rev = "00eb75b0c803b0fc8e5413d736529d9d3b82b6bd" }
megatron-bridge = { git = "https://github.com/Sphere-AI-Lab/Megatron-Bridge.git", rev = "69a8e369e23f522c354f1cd33c2cfd21ef5768d6" }
transformer-engine = { git = "https://github.com/NVIDIA/TransformerEngine.git", rev = "71bbefbf153418f943640df0f7373625dc93fa46" }
deep-ep = { git = "https://github.com/deepseek-ai/DeepEP.git", rev = "d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb" }

[tool.orbit.release.backend-pins.sglang]
source = "https://github.com/Sphere-AI-Lab/sglang.git"
tested-ref = "51845dc4acca94507ab184b007c8fcfd656b191f"

[tool.orbit.release.backend-pins.megatron-core]
source = "https://github.com/Sphere-AI-Lab/Megatron-LM.git"
tested-ref = "00eb75b0c803b0fc8e5413d736529d9d3b82b6bd"

[tool.orbit.release.backend-pins.megatron-bridge]
source = "https://github.com/Sphere-AI-Lab/Megatron-Bridge.git"
tested-ref = "69a8e369e23f522c354f1cd33c2cfd21ef5768d6"
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
    assert pins["SGLANG_COMMIT"] == "51845dc4acca94507ab184b007c8fcfd656b191f"
    assert pins["MEGATRON_COMMIT"] == "00eb75b0c803b0fc8e5413d736529d9d3b82b6bd"
    assert pins["MEGATRON_BRIDGE_COMMIT"] == "69a8e369e23f522c354f1cd33c2cfd21ef5768d6"
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
megatron-core = { git = "https://github.com/Sphere-AI-Lab/Megatron-LM.git", rev = "00eb75b0c803b0fc8e5413d736529d9d3b82b6bd" }
megatron-bridge = { git = "https://github.com/Sphere-AI-Lab/Megatron-Bridge.git", rev = "69a8e369e23f522c354f1cd33c2cfd21ef5768d6" }
transformer-engine = { git = "https://github.com/NVIDIA/TransformerEngine.git", rev = "71bbefbf153418f943640df0f7373625dc93fa46" }
deep-ep = { git = "https://github.com/deepseek-ai/DeepEP.git", rev = "d4f41e4e93602a15e95f55f6ee8df8f1aaa0e4bb" }

[tool.orbit.release.backend-pins.sglang]
source = "https://github.com/Sphere-AI-Lab/sglang.git"
tested-ref = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"

[tool.orbit.release.backend-pins.megatron-core]
source = "https://github.com/Sphere-AI-Lab/Megatron-LM.git"
tested-ref = "00eb75b0c803b0fc8e5413d736529d9d3b82b6bd"

[tool.orbit.release.backend-pins.megatron-bridge]
source = "https://github.com/Sphere-AI-Lab/Megatron-Bridge.git"
tested-ref = "69a8e369e23f522c354f1cd33c2cfd21ef5768d6"
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
