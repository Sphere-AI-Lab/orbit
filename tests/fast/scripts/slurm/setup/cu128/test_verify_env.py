import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[6]
MODULE_PATH = REPO_ROOT / "scripts/slurm/setup/cu128/verify_env.py"
SPEC = importlib.util.spec_from_file_location("orbit_cu128_verify_env", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
VERIFY_ENV = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VERIFY_ENV
SPEC.loader.exec_module(VERIFY_ENV)

Check = VERIFY_ENV.Check
check_editables = VERIFY_ENV.check_editables
check_h200_runtime = lambda *args, **kwargs: VERIFY_ENV.check_h200_runtime(*args, **kwargs)
check_import = lambda *args, **kwargs: VERIFY_ENV.check_import(*args, **kwargs)
check_sources = lambda *args, **kwargs: VERIFY_ENV.check_sources(*args, **kwargs)
check_versions = VERIFY_ENV.check_versions
load_pins = VERIFY_ENV.load_pins
print_summary = lambda *args, **kwargs: VERIFY_ENV.print_summary(*args, **kwargs)
source_revision_inputs = lambda *args, **kwargs: VERIFY_ENV.source_revision_inputs(*args, **kwargs)

PINS = {
    "TORCH_VERSION": "2.11.0",
    "TORCHVISION_VERSION": "0.26.0",
    "TORCHAUDIO_VERSION": "2.11.0",
    "CUDA_PROFILE": "cu128",
}


def test_load_pins_parses_generated_shell_assignments(tmp_path: Path) -> None:
    pins = tmp_path / "pins.env"
    pins.write_text("TORCH_VERSION=2.11.0\nCUDA_PROFILE=cu128\n")
    assert load_pins(pins) == {"TORCH_VERSION": "2.11.0", "CUDA_PROFILE": "cu128"}


def test_version_check_requires_cu128_local_tag() -> None:
    installed = {"torch": "2.11.0+cu130", "torchvision": "0.26.0+cu128"}
    checks = check_versions(PINS, installed)
    torch_check = next(check for check in checks if check.label == "torch build")
    assert not torch_check.ok
    assert torch_check.detail == "expected 2.11.0+cu128, got 2.11.0+cu130"


def test_version_check_audits_non_torch_package_pins() -> None:
    pins = {
        "ORBIT_VERSION": "0.1.0",
        "FLASHINFER_VERSION": "0.6.14",
        "TRANSFORMERS_VERSION": "4.57.6",
    }
    installed = {
        "orbit": "0.1.0+editable",
        "flashinfer-python": "0.6.13",
        "transformers": "4.57.6",
    }
    checks = check_versions(pins, installed)
    assert Check(
        "flashinfer-python version", False, "expected 0.6.14, got 0.6.13"
    ) in checks
    assert Check("transformers version", True, "4.57.6") in checks
    assert Check("orbit version", True, "0.1.0+editable") in checks


def test_editable_check_requires_expected_realpath(tmp_path: Path) -> None:
    expected = tmp_path / "orbit"
    wrong = tmp_path / "other"
    expected.mkdir()
    wrong.mkdir()
    checks = check_editables(
        {"orbit": expected},
        {"orbit": {"url": wrong.as_uri(), "dir_info": {"editable": True}}},
    )
    assert checks == [Check("orbit editable source", False, f"expected {expected}, got {wrong}")]


def test_source_check_reports_commit_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "sglang"
    source.mkdir()
    checks = check_sources(
        {"sglang": source},
        {"sglang": "5" * 40},
        git_head=lambda _: "6" * 40,
        git_dirty=lambda _: False,
    )
    assert not checks[0].ok
    assert f"expected {'5' * 40}" in checks[0].detail


def test_source_revision_inputs_exclude_orbit(tmp_path: Path) -> None:
    source_paths, expected_commits = source_revision_inputs({}, tmp_path)
    assert "orbit" not in source_paths
    assert "orbit" not in expected_commits


class FakeNccl:
    @staticmethod
    def version() -> tuple[int, int, int]:
        return (2, 28, 9)


class FakeCuda:
    nccl = FakeNccl()

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def get_device_name(_: int) -> str:
        return "NVIDIA H200"

    @staticmethod
    def get_device_capability(_: int) -> tuple[int, int]:
        return (9, 0)

    @staticmethod
    def is_bf16_supported() -> bool:
        return True


class FakeCudnn:
    @staticmethod
    def version() -> int:
        return 91002


def test_h200_runtime_checks_cuda_capability_and_libraries() -> None:
    fake_torch = SimpleNamespace(
        cuda=FakeCuda(),
        version=SimpleNamespace(cuda="12.8"),
        backends=SimpleNamespace(cudnn=FakeCudnn()),
    )
    checks = check_h200_runtime(
        {"CUDA_TOOLKIT_VERSION": "12.8", "H200_COMPUTE_CAPABILITY": "9.0"},
        torch_module=fake_torch,
        matmul_probe=lambda _: True,
    )
    assert all(check.ok for check in checks)
    assert Check("NCCL runtime", True, "2.28.9") in checks
    assert Check("BF16 CUDA matmul", True, "finite 512x512 result") in checks


def test_h200_runtime_rejects_wrong_compute_capability() -> None:
    class WrongCapabilityCuda(FakeCuda):
        @staticmethod
        def get_device_capability(_: int) -> tuple[int, int]:
            return (8, 0)

    fake_torch = SimpleNamespace(
        cuda=WrongCapabilityCuda(),
        version=SimpleNamespace(cuda="12.8"),
        backends=SimpleNamespace(cudnn=FakeCudnn()),
    )
    checks = check_h200_runtime(
        {"CUDA_TOOLKIT_VERSION": "12.8", "H200_COMPUTE_CAPABILITY": "9.0"},
        torch_module=fake_torch,
        matmul_probe=lambda _: True,
    )
    assert Check(
        "Hopper compute capability", False, "expected 9.0, got 8.0"
    ) in checks


def test_import_check_rejects_empty_namespace_package() -> None:
    module = SimpleNamespace(__file__=None, __version__="2.11.0")
    check = check_import("torch", importer=lambda _: module)
    assert check == Check(
        "import torch", False, "resolved as namespace package without __file__"
    )


def test_print_summary_returns_failure_count(capsys) -> None:
    failures = print_summary(
        [Check("torch build", True, "2.11.0+cu128"), Check("orbit import", False, "missing")]
    )
    output = capsys.readouterr().out
    assert failures == 1
    assert "[PASS] torch build: 2.11.0+cu128" in output
    assert "[FAIL] orbit import: missing" in output
    assert "1 passed, 1 failed" in output
