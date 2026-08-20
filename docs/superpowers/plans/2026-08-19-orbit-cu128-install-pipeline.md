# Orbit CUDA 12.8/H200 Installation Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible, native Conda/uv CUDA 12.8 installation and verification pipeline for Orbit on Slurm H200 nodes without changing the existing CUDA 13.2 workflow.

**Architecture:** A deterministic extractor reads Orbit and pinned backend manifests plus one explicit CUDA 12.8 profile mapping, then generates a committed `pins.env`. A fail-closed shell installer validates source and ABI state before building a prefix-based environment, and a Python verifier audits metadata, editable paths, compiled imports, and optional H200 runtime behavior.

**Tech Stack:** Python 3.12, Bash, Conda prefix environments, uv pip, TOML via `tomllib`, importlib metadata, Git, Slurm, PyTorch 2.11.0+cu128, pytest.

**Spec:** `docs/superpowers/specs/2026-08-19-orbit-cu128-install-pipeline-design.md`

## Global Constraints

- Keep the existing CUDA 13.2 `pyproject.toml`, `uv.lock`, `env.sh`, and guides behaviorally unchanged.
- Target Python 3.12, CUDA build label `cu128`, Torch 2.11.0+cu128, and NVIDIA H200 compute capability 9.0.
- Keep CUDA 12.8-only code under `scripts/slurm/setup/cu128/`.
- Install Orbit, SGLang, Megatron-LM, and Megatron-Bridge editable from validated source directories.
- Never pull, reset, switch, or repair an existing backend checkout automatically.
- Never delete, rename, or replace an environment automatically.
- Do not mutate `/data/home/zeju/miles-orbit/envs/orbit_cu128`; qualification uses a fresh unique prefix.
- Do not touch the Claude-owned session or Slurm job 43131.
- Perform package builds and full verification only inside a scheduled H200 allocation.
- Require explicit user authorization before the expensive fresh-environment qualification job.
- Generated files must be deterministic, atomically written, and free of credentials and machine-specific paths.
- Full verification failures are failures, never skipped checks represented as success.

---

## File Structure

### Production files

- Create `scripts/slurm/setup/cu128/extract_pins.py`: parse source manifests, cross-check duplicated refs, render deterministic shell pins, and implement `--write` and `--check`.
- Create `scripts/slurm/setup/cu128/pins.env`: generated, committed CUDA 12.8/H200 pins and source hashes.
- Create `scripts/slurm/setup/cu128/install_env.sh`: non-mutating preflight, source validation, Conda/uv installation layers, restart behavior, and verifier handoff.
- Create `scripts/slurm/setup/cu128/verify_env.py`: metadata/import audit and `--full-h200` runtime checks.
- Create `scripts/slurm/setup/cu128/README.md`: installation, activation, pin maintenance, verification, and troubleshooting.
- Modify `README.md`: add one link to the qualified CUDA 12.8/H200 profile without changing the CUDA 13.2 default.

### Test files

- Create `tests/fast/scripts/slurm/setup/cu128/test_extract_pins.py`: manifest parsing, cross-checking, deterministic generation, quoting, atomic writes, and drift behavior.
- Create `tests/fast/scripts/slurm/setup/cu128/test_verify_env.py`: pin loading, versions, direct URLs, editable paths, source commits, namespace-package rejection, and mocked H200 checks.
- Create `tests/fast/scripts/slurm/setup/cu128/test_install_env.py`: help/config/install-plan behavior and fail-closed shell preflight contracts without environment mutation.

---

### Task 1: Deterministic pin extraction

**Files:**
- Create: `scripts/slurm/setup/cu128/extract_pins.py`
- Create: `scripts/slurm/setup/cu128/pins.env`
- Create: `tests/fast/scripts/slurm/setup/cu128/test_extract_pins.py`

**Interfaces:**
- Consumes: Orbit root `pyproject.toml`; exact-commit SGLang `python/pyproject.toml`; optional source-root overrides passed to the CLI.
- Produces: `collect_pins(repo_root: Path, sglang_root: Path) -> dict[str, str]`; `render_pins(pins: Mapping[str, str], source_hashes: Mapping[str, str]) -> str`; `write_atomic(path: Path, content: str) -> None`; CLI modes `--write` and `--check`.

- [ ] **Step 1: Write manifest fixtures and the first failing extraction test**

Add compact TOML fixtures directly in the test so failures identify the exact contract:

```python
from pathlib import Path

from scripts.slurm.setup.cu128.extract_pins import collect_pins


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
megatron-bridge = { git = "https://github.com/Sphere-AI-Lab/Megatron-Bridge.git", rev = "ad26fc46b252e6e53a56052776623499da3dc583" }
transformer-engine = { git = "https://github.com/NVIDIA/TransformerEngine.git", rev = "71bbefbf153418f943640df0f7373625dc93fa46" }

[tool.orbit.release.backend-pins.sglang]
source = "https://github.com/Sphere-AI-Lab/sglang.git"
tested-ref = "51845dc4acca94507ab184b007c8fcfd656b191f"

[tool.orbit.release.backend-pins.megatron-core]
source = "https://github.com/Sphere-AI-Lab/Megatron-LM.git"
tested-ref = "00eb75b0c803b0fc8e5413d736529d9d3b82b6bd"

[tool.orbit.release.backend-pins.megatron-bridge]
source = "https://github.com/Sphere-AI-Lab/Megatron-Bridge.git"
tested-ref = "ad26fc46b252e6e53a56052776623499da3dc583"
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
    assert pins["MEGATRON_BRIDGE_COMMIT"] == "ad26fc46b252e6e53a56052776623499da3dc583"
```

- [ ] **Step 2: Run the focused test and confirm the missing module failure**

Run:

```bash
/data/home/zeju/miles-orbit/envs/orbit_cu128/bin/python -m pytest tests/fast/scripts/slurm/setup/cu128/test_extract_pins.py::test_collect_pins_cross_checks_orbit_backend_refs -v
```

Expected: FAIL because `scripts.slurm.setup.cu128.extract_pins` does not exist.

- [ ] **Step 3: Implement TOML parsing, dependency extraction, and duplicate-ref validation**

Create an explicit profile and focused helpers:

```python
CU128_PROFILE = {
    "CUDA_PROFILE": "cu128",
    "PYTHON_VERSION": "3.12",
    "TORCH_INDEX_URL": "https://download.pytorch.org/whl/cu128",
    "FLASHINFER_INDEX_URL": "https://flashinfer.ai/whl/cu128",
    "SGLANG_WHEEL_INDEX_URL": "https://docs.sglang.ai/whl/cu128",
    "TORCHVISION_VERSION": "0.26.0",
    "TORCHAUDIO_VERSION": "2.11.0",
    "CUDA_PYTHON_VERSION": "12.9.2",
    "H200_COMPUTE_CAPABILITY": "9.0",
}

def read_toml(path: Path) -> dict:
    with path.open("rb") as stream:
        return tomllib.load(stream)

def exact_requirement_version(requirements: Sequence[str], name: str) -> str:
    matches = [
        Requirement(text)
        for text in requirements
        if canonicalize_name(Requirement(text).name) == canonicalize_name(name)
    ]
    if len(matches) != 1:
        raise PinError(f"{name}: expected one requirement, found {len(matches)}")
    exact = [spec.version for spec in matches[0].specifier if spec.operator == "=="]
    if len(exact) != 1:
        raise PinError(f"{name}: expected one exact == version")
    return exact[0]

def require_matching_ref(uv_source: dict, release_pin: dict, name: str) -> str:
    uv_ref = uv_source.get("rev")
    tested_ref = release_pin.get("tested-ref")
    if uv_ref != tested_ref:
        raise PinError(f"{name}: tool.uv.sources rev {uv_ref!r} != tested-ref {tested_ref!r}")
    return tested_ref
```

Populate the output in a fixed key order. Extract Torch and FlashInfer from the pinned SGLang manifest and backend refs from Orbit. Keep CUDA-specific indexes and architecture values in `CU128_PROFILE`.

- [ ] **Step 4: Run the focused test and confirm it passes**

Run the same pytest command.

Expected: PASS.

- [ ] **Step 5: Add failing tests for mismatch, deterministic shell rendering, atomic writes, and `--check`**

Add tests with these exact assertions:

```python
def test_collect_pins_rejects_disagreeing_sglang_refs(tmp_path: Path) -> None:
    repo, sglang = write_manifests(tmp_path, release_ref="a" * 40, uv_ref="b" * 40)
    with pytest.raises(PinError, match="sglang.*tested-ref"):
        collect_pins(repo, sglang)


def test_render_pins_is_deterministic_and_shell_safe(pin_fixture: dict[str, str]) -> None:
    first = render_pins(pin_fixture, {"pyproject.toml": "f" * 64})
    second = render_pins(dict(reversed(list(pin_fixture.items()))), {"pyproject.toml": "f" * 64})
    assert first == second
    assert "AUTO-GENERATED" in first
    assert "TORCH_VERSION='2.11.0'" in first
    assert "\n" not in next(line for line in first.splitlines() if line.startswith("TORCH_VERSION="))


def test_check_mode_reports_drift_without_writing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    output = tmp_path / "pins.env"
    output.write_text("stale\n")
    original = output.read_bytes()
    assert run_check(output, "fresh\n") == 1
    assert output.read_bytes() == original
    assert "pins.env is stale" in capsys.readouterr().err
```

- [ ] **Step 6: Run the extractor test file and confirm the new tests fail**

Run:

```bash
/data/home/zeju/miles-orbit/envs/orbit_cu128/bin/python -m pytest tests/fast/scripts/slurm/setup/cu128/test_extract_pins.py -v
```

Expected: FAIL on unimplemented rendering/check behavior.

- [ ] **Step 7: Implement deterministic rendering, atomic replacement, hashes, and CLI modes**

Use `shlex.quote`, SHA-256 file hashing, sorted fixed-order output, and `tempfile.NamedTemporaryFile` in the destination directory followed by `os.replace`. The CLI contract is:

```python
parser = argparse.ArgumentParser()
mode = parser.add_mutually_exclusive_group(required=True)
mode.add_argument("--write", action="store_true")
mode.add_argument("--check", action="store_true")
parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
parser.add_argument("--sglang-root", type=Path)
parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
```

When `--sglang-root` is omitted, resolve `ORBIT_WORKSPACE` or the parent of the Orbit repository and append `sglang`.

- [ ] **Step 8: Generate the real `pins.env` from validated manifests**

The active SGLang sibling is already at the Orbit-tested ref. Run:

```bash
cd /data/home/zeju/miles-orbit/orbit/.worktrees/cu128-install-pipeline
/data/home/zeju/miles-orbit/envs/orbit_cu128/bin/python scripts/slurm/setup/cu128/extract_pins.py   --write   --sglang-root /data/home/zeju/miles-orbit/sglang
```

Then run `--check` with the same `--sglang-root`.

Expected: exit 0 and no drift message.

- [ ] **Step 9: Run extractor tests**

Run:

```bash
/data/home/zeju/miles-orbit/envs/orbit_cu128/bin/python -m pytest tests/fast/scripts/slurm/setup/cu128/test_extract_pins.py -v
```

Expected: all tests PASS.

- [ ] **Step 10: Commit the extractor, generated pins, and tests**

```bash
git add scripts/slurm/setup/cu128/extract_pins.py   scripts/slurm/setup/cu128/pins.env   tests/fast/scripts/slurm/setup/cu128/test_extract_pins.py
git commit -m "feat: generate CUDA 12.8 environment pins"
```

---

### Task 2: Metadata and source verifier

**Files:**
- Create: `scripts/slurm/setup/cu128/verify_env.py`
- Create: `tests/fast/scripts/slurm/setup/cu128/test_verify_env.py`

**Interfaces:**
- Consumes: generated `pins.env`, current Python environment metadata, and source-root overrides.
- Produces: `Check(label: str, ok: bool, detail: str)`; `load_pins(path: Path) -> dict[str, str]`; `check_versions(...)`; `check_editables(...)`; `check_sources(...)`; `check_imports(...)`; CLI default metadata/import mode and `--full-h200`.

- [ ] **Step 1: Write failing unit tests for pin loading, versions, and editable metadata**

```python
def test_load_pins_parses_generated_shell_assignments(tmp_path: Path) -> None:
    pins = tmp_path / "pins.env"
    pins.write_text("TORCH_VERSION='2.11.0'\nCUDA_PROFILE='cu128'\n")
    assert load_pins(pins) == {"TORCH_VERSION": "2.11.0", "CUDA_PROFILE": "cu128"}


def test_version_check_requires_cu128_local_tag() -> None:
    installed = {"torch": "2.11.0+cu130", "torchvision": "0.26.0+cu128"}
    checks = check_versions(PINS, installed)
    torch_check = next(check for check in checks if check.label == "torch build")
    assert not torch_check.ok
    assert "cu128" in torch_check.detail


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
```

- [ ] **Step 2: Run the focused tests and confirm the missing module failure**

Run:

```bash
/data/home/zeju/miles-orbit/envs/orbit_cu128/bin/python -m pytest tests/fast/scripts/slurm/setup/cu128/test_verify_env.py -v
```

Expected: FAIL because `verify_env.py` does not exist.

- [ ] **Step 3: Implement pin loading and pure metadata/source helpers**

Use a strict parser for generated single-line shell assignments rather than sourcing arbitrary shell text. Define:

```python
@dataclass(frozen=True)
class Check:
    label: str
    ok: bool
    detail: str = ""

def load_pins(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, encoded = line.partition("=")
        if not separator or not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise VerificationError(f"{path}:{number}: invalid pin assignment")
        parsed = shlex.split(encoded, posix=True)
        if len(parsed) != 1:
            raise VerificationError(f"{path}:{number}: expected one shell value")
        result[key] = parsed[0]
    return result
```

Use `importlib.metadata.distribution(name).read_text("direct_url.json")` for editable provenance and `Path.resolve()` for path comparisons.

- [ ] **Step 4: Run verifier tests and confirm the initial checks pass**

Run the verifier test file.

Expected: initial tests PASS.

- [ ] **Step 5: Add failing tests for commit checks, namespace imports, and result formatting**

```python
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
    assert "expected " + "5" * 40 in checks[0].detail


def test_import_check_rejects_empty_namespace_package() -> None:
    module = SimpleNamespace(__file__=None, __version__="2.11.0")
    check = check_import("torch", importer=lambda _: module)
    assert check == Check("import torch", False, "resolved as namespace package without __file__")


def test_print_summary_returns_failure_count(capsys: pytest.CaptureFixture[str]) -> None:
    failures = print_summary([Check("one", True), Check("two", False, "bad")])
    assert failures == 1
    output = capsys.readouterr().out
    assert "[PASS] one" in output
    assert "[FAIL] two: bad" in output
    assert "1 passed, 1 failed" in output
```

- [ ] **Step 6: Implement source, import, and summary checks plus CLI metadata mode**

Default import targets are `torch`, `transformers`, `sglang`, `megatron.core`, `megatron.bridge`, `transformer_engine`, `flash_attn`, `sgl_kernel`, `apex`, `deep_ep`, `deep_gemm`, and `orbit`.

The CLI accepts:

```python
parser.add_argument("--pins", type=Path, default=DEFAULT_PINS)
parser.add_argument("--orbit-root", type=Path, default=REPO_ROOT)
parser.add_argument("--workspace", type=Path)
parser.add_argument("--full-h200", action="store_true")
```

Return exit 1 when any required check fails.

- [ ] **Step 7: Run verifier tests**

Run the verifier test file.

Expected: all metadata/source tests PASS.

- [ ] **Step 8: Commit metadata verification**

```bash
git add scripts/slurm/setup/cu128/verify_env.py   tests/fast/scripts/slurm/setup/cu128/test_verify_env.py
git commit -m "feat: audit CUDA environment metadata"
```

---

### Task 3: Installer preflight and source validation

**Files:**
- Create: `scripts/slurm/setup/cu128/install_env.sh`
- Create: `tests/fast/scripts/slurm/setup/cu128/test_install_env.py`

**Interfaces:**
- Consumes: `pins.env`, extractor `--check`, workspace/source/environment overrides.
- Produces: `--help`, `--print-config`, `--print-install-plan`, and normal install modes; shell functions `die`, `require_command`, `resolve_paths`, `check_gpu_preflight`, and `check_source`.

- [ ] **Step 1: Write failing tests for help and non-mutating configuration output**

```python
INSTALLER = REPO_ROOT / "scripts/slurm/setup/cu128/install_env.sh"


def run_installer(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    if env:
        merged.update(env)
    return subprocess.run(
        ["bash", str(INSTALLER), *args],
        cwd=REPO_ROOT,
        env=merged,
        text=True,
        capture_output=True,
        check=False,
    )


def test_help_is_login_node_safe() -> None:
    result = run_installer("--help")
    assert result.returncode == 0
    assert "--print-config" in result.stdout
    assert "--print-install-plan" in result.stdout


def test_print_config_uses_prefix_environment(tmp_path: Path) -> None:
    result = run_installer(
        "--print-config",
        env={
            "ORBIT_WORKSPACE": str(tmp_path),
            "ORBIT_ENV_PREFIX": str(tmp_path / "envs/orbit_cu128_test"),
        },
    )
    assert result.returncode == 0
    assert f"environment={tmp_path}/envs/orbit_cu128_test" in result.stdout
    assert "cuda_profile=cu128" in result.stdout
```

- [ ] **Step 2: Run installer tests and confirm the missing script failure**

Run:

```bash
/data/home/zeju/miles-orbit/envs/orbit_cu128/bin/python -m pytest tests/fast/scripts/slurm/setup/cu128/test_install_env.py -v
```

Expected: FAIL because `install_env.sh` does not exist.

- [ ] **Step 3: Implement argument handling, path resolution, pin sourcing, and safe display modes**

Start with:

```bash
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
ORBIT_REPO="$(cd -- "$SCRIPT_DIR/../../../.." && pwd -P)"
ORBIT_WORKSPACE="${ORBIT_WORKSPACE:-$(dirname "$ORBIT_REPO")}"
ORBIT_ENV_PREFIX="${ORBIT_ENV_PREFIX:-$ORBIT_WORKSPACE/envs/orbit_cu128}"
SGLANG_SRC="${SGLANG_SRC:-$ORBIT_WORKSPACE/sglang}"
MEGATRON_SRC="${MEGATRON_SRC:-$ORBIT_WORKSPACE/Megatron-LM}"
MEGATRON_BRIDGE_SRC="${MEGATRON_BRIDGE_SRC:-$ORBIT_WORKSPACE/Megatron-Bridge}"
CONDA_ROOT="${CONDA_ROOT:-/data/shared/conda/miniconda3}"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/pins.env"
```

`--help`, `--print-config`, and `--print-install-plan` return before any GPU, Conda, Git mutation, or package command.

- [ ] **Step 4: Run installer tests and confirm display tests pass**

Run the installer test file.

Expected: display tests PASS.

- [ ] **Step 5: Add failing tests for source mismatch and install ordering**

Use temporary Git repositories and the print-only plan:

```python
def test_print_install_plan_orders_torch_before_compiled_extensions(tmp_path: Path) -> None:
    result = run_installer(
        "--print-install-plan",
        env={
            "ORBIT_WORKSPACE": str(tmp_path),
            "ORBIT_ENV_PREFIX": str(tmp_path / "env"),
        },
    )
    assert result.returncode == 0
    lines = result.stdout.splitlines()
    assert lines.index("layer=torch-cu128") < lines.index("layer=transformer-engine")
    assert lines.index("layer=transformer-engine") < lines.index("layer=orbit-editable")


def test_source_validation_rejects_wrong_commit(tmp_path: Path) -> None:
    source = make_git_repo(tmp_path / "sglang")
    result = run_installer(
        "--check-source",
        "sglang",
        str(source),
        "5" * 40,
        env={"ORBIT_WORKSPACE": str(tmp_path)},
    )
    assert result.returncode != 0
    assert "sglang commit mismatch" in result.stderr
    assert "no files were changed" in result.stderr
```

- [ ] **Step 6: Implement fail-closed preflight and source validation**

Preflight must:

- Confirm `SLURM_JOB_ID` is present in normal install mode.
- Query one GPU and require compute capability `9.0`.
- Require `nvidia-smi`, `nvcc`, Conda, uv, Git, CMake, Ninja, Cargo, GCC, and G++.
- Require the CUDA toolkit major/minor to be 12.8.
- Check free space at the workspace and environment parent.
- Run extractor `--check`.
- Validate commit and clean state for each source checkout.

A missing source prints the exact pinned clone command but does not execute it until normal install mode reaches the source preparation step. An existing source mismatch always fails.

- [ ] **Step 7: Run installer tests and shell syntax**

Run:

```bash
bash -n scripts/slurm/setup/cu128/install_env.sh
/data/home/zeju/miles-orbit/envs/orbit_cu128/bin/python -m pytest tests/fast/scripts/slurm/setup/cu128/test_install_env.py -v
```

Expected: syntax success and all tests PASS.

- [ ] **Step 8: Commit installer preflight**

```bash
git add scripts/slurm/setup/cu128/install_env.sh   tests/fast/scripts/slurm/setup/cu128/test_install_env.py
git commit -m "feat: add H200 installer preflight"
```

---

### Task 4: Conda/uv installation layers

**Files:**
- Modify: `scripts/slurm/setup/cu128/install_env.sh`
- Modify: `tests/fast/scripts/slurm/setup/cu128/test_install_env.py`

**Interfaces:**
- Consumes: validated paths and pins from Task 3.
- Produces: restartable functions `ensure_environment`, `install_torch_layer`, `install_sglang_layer`, `install_backend_layer`, `install_compiled_layer`, `install_orbit_layer`, and `run_verifier`.

- [ ] **Step 1: Add a failing complete install-plan test**

```python
EXPECTED_LAYERS = [
    "environment-python312",
    "torch-cu128",
    "torch-cudnn",
    "sglang-runtime",
    "megatron-editable",
    "megatron-bridge-editable",
    "sglang-editable",
    "transformer-engine",
    "flash-attention",
    "sglang-kernel",
    "apex",
    "deep-ep",
    "deep-gemm",
    "orbit-requirements",
    "orbit-editable",
    "abi-reassert",
    "verify-full-h200",
]


def test_print_install_plan_is_complete_and_stable(tmp_path: Path) -> None:
    result = run_installer(
        "--print-install-plan",
        env={"ORBIT_WORKSPACE": str(tmp_path), "ORBIT_ENV_PREFIX": str(tmp_path / "env")},
    )
    assert result.returncode == 0
    assert result.stdout.splitlines() == [f"layer={name}" for name in EXPECTED_LAYERS]
```

- [ ] **Step 2: Run the plan test and confirm it fails on missing layers**

Run the focused test.

Expected: FAIL with an ordering/content difference.

- [ ] **Step 3: Implement environment creation and the uv command wrapper**

Use a prefix environment and target its Python explicitly:

```bash
ensure_environment() {
    if [[ ! -x "$ORBIT_ENV_PREFIX/bin/python" ]]; then
        "$CONDA_ROOT/bin/conda" create -y -p "$ORBIT_ENV_PREFIX" "python=$PYTHON_VERSION"
    fi
    UV=(uv pip install --python "$ORBIT_ENV_PREFIX/bin/python")
}

run_uv() {
    printf '[uv]'
    printf ' %q' "${UV[@]}" "$@"
    printf '\n'
    "${UV[@]}" "$@"
}
```

Do not activate Conda inside the installer. Explicit interpreter paths prevent accidental installation into `base`.

- [ ] **Step 4: Implement Torch and cuDNN layers**

Install exact Torch packages from `TORCH_INDEX_URL`, inspect Torch metadata for its exact `nvidia-cudnn-cu12` requirement, install that version, and fail if `torch.__version__` does not equal `TORCH_VERSION+cu128`.

Use a temporary override file containing the exact Torch, TorchVision, TorchAudio, CUDA Python, and FlashInfer choices so SGLang resolution cannot replace the cu128 Torch wheel.

- [ ] **Step 5: Implement source preparation and editable backend layers**

For missing repositories only, clone their generated source URLs and detach at exact commits. Existing sources must already pass Task 3 validation.

Install:

```bash
run_uv -e "$MEGATRON_SRC" --no-deps
run_uv -e "$MEGATRON_BRIDGE_SRC" --no-deps --no-build-isolation
run_uv -e "$SGLANG_SRC/python[all]" --override "$override_file"
```

Write a source-root `.pth` only when an Orbit or backend package is otherwise omitted by editable package discovery. The target must be derived from `site.getsitepackages()`.

- [ ] **Step 6: Implement fixed-order compiled layers**

Use source installs at generated refs for Transformer Engine, FlashAttention, SGLang kernel, Apex, DeepEP, and DeepGEMM. Export SM90-only build variables:

```bash
export TORCH_CUDA_ARCH_LIST=9.0
export NVTE_CUDA_ARCHS=90
export FLASH_ATTN_CUDA_ARCHS=90
export CMAKE_CUDA_ARCHITECTURES=90a
export UV_CONCURRENT_BUILDS=1
export MAX_JOBS="${MAX_JOBS:-16}"
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-8}"
```

Build one ABI-sensitive package at a time. Do not launch package installs in the background.

- [ ] **Step 7: Implement Orbit dependency and editable layers**

Install Orbit's Python requirements through uv with the generated override file so stale `requirements.txt` entries cannot replace the profile's Transformers, Torch, SGLang, router, or FlashInfer choices. Then install Orbit editable with `--no-deps`.

Reinstall or reassert the ABI-sensitive packages after broad dependency resolution, then invoke:

```bash
"$ORBIT_ENV_PREFIX/bin/python" "$SCRIPT_DIR/verify_env.py"   --pins "$SCRIPT_DIR/pins.env"   --orbit-root "$ORBIT_REPO"   --workspace "$ORBIT_WORKSPACE"   --full-h200
```

- [ ] **Step 8: Run print-plan tests and shell syntax**

Run:

```bash
bash -n scripts/slurm/setup/cu128/install_env.sh
/data/home/zeju/miles-orbit/envs/orbit_cu128/bin/python -m pytest tests/fast/scripts/slurm/setup/cu128/test_install_env.py -v
```

Expected: syntax success and all tests PASS. Do not run normal install mode in this task.

- [ ] **Step 9: Commit installation layers**

```bash
git add scripts/slurm/setup/cu128/install_env.sh   tests/fast/scripts/slurm/setup/cu128/test_install_env.py
git commit -m "feat: install the Orbit CUDA 12.8 stack"
```

---

### Task 5: Full H200 verification and user documentation

**Files:**
- Modify: `scripts/slurm/setup/cu128/verify_env.py`
- Modify: `tests/fast/scripts/slurm/setup/cu128/test_verify_env.py`
- Create: `scripts/slurm/setup/cu128/README.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: metadata checks from Task 2 and an importable Torch runtime.
- Produces: `check_h200_runtime(torch_module) -> list[Check]`; documented install/update/activation commands; root-guide link.

- [ ] **Step 1: Write failing mocked H200 runtime tests**

```python
def test_h200_runtime_checks_cuda_arch_and_bf16_matmul() -> None:
    torch = fake_torch(
        version="2.11.0+cu128",
        cuda_version="12.8",
        available=True,
        device_name="NVIDIA H200",
        capability=(9, 0),
        cudnn=91900,
        nccl=(2, 28, 9),
    )
    checks = check_h200_runtime(torch)
    assert all(check.ok for check in checks)


def test_h200_runtime_rejects_blackwell() -> None:
    torch = fake_torch(
        version="2.11.0+cu128",
        cuda_version="12.8",
        available=True,
        device_name="NVIDIA B200",
        capability=(10, 0),
        cudnn=91900,
        nccl=(2, 28, 9),
    )
    failed = [check for check in check_h200_runtime(torch) if not check.ok]
    assert any(check.label == "compute capability" for check in failed)
```

The fake tensor object records `matmul` and `synchronize` calls so the test proves the small BF16 operation is attempted.

- [ ] **Step 2: Run the H200 tests and confirm the missing helper failure**

Run the focused test names.

Expected: FAIL because `check_h200_runtime` is not implemented.

- [ ] **Step 3: Implement full runtime and compiled-symbol checks**

Require:

- `torch.cuda.is_available()`.
- `torch.version.cuda == "12.8"`.
- Device name containing `H200`.
- Capability `(9, 0)`.
- cuDNN equal to the Torch-declared package version.
- NCCL availability and a reported version.
- Required exported symbols from Transformer Engine, FlashAttention, SGLang kernel, Apex, DeepEP, and DeepGEMM.
- A 512x512 BF16 matrix multiplication and `torch.cuda.synchronize()`.

Keep allocations small and release references after the check.

- [ ] **Step 4: Run verifier tests**

Run:

```bash
/data/home/zeju/miles-orbit/envs/orbit_cu128/bin/python -m pytest tests/fast/scripts/slurm/setup/cu128/test_verify_env.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Write the CUDA 12.8 profile README**

Include exact commands:

```bash
# Inside an approved H200 Slurm allocation:
cd /data/home/zeju/miles-orbit/orbit
ORBIT_ENV_PREFIX=/data/home/zeju/miles-orbit/envs/orbit_cu128   bash scripts/slurm/setup/cu128/install_env.sh

# Daily activation:
source /data/home/zeju/miles-orbit/envs/orbit_cu128/bin/activate

# Metadata audit:
python scripts/slurm/setup/cu128/verify_env.py

# Full audit inside an H200 allocation:
python scripts/slurm/setup/cu128/verify_env.py --full-h200

# Maintainer pin refresh:
python scripts/slurm/setup/cu128/extract_pins.py   --write   --sglang-root /data/home/zeju/miles-orbit/sglang
python scripts/slurm/setup/cu128/extract_pins.py   --check   --sglang-root /data/home/zeju/miles-orbit/sglang
```

Document fresh-prefix qualification, restart behavior, no automatic deletion, source mismatch handling, CUDA tag errors, and the distinction from CUDA 13.2.

- [ ] **Step 6: Add one root README link**

Under the existing installation section, preserve CUDA 13.2 as the default and add:

```markdown
For the Slurm H200 CUDA 12.8 profile, see
[`scripts/slurm/setup/cu128/README.md`](scripts/slurm/setup/cu128/README.md).
```

- [ ] **Step 7: Run all fast tests and syntax checks for the new profile**

Run only after Slurm job 43131 and any environment mutation process have ended:

```bash
bash -n scripts/slurm/setup/cu128/install_env.sh
/data/home/zeju/miles-orbit/envs/orbit_cu128/bin/python -m pytest   tests/fast/scripts/slurm/setup/cu128 -v
```

Expected: all tests PASS.

- [ ] **Step 8: Commit verification and documentation**

```bash
git add scripts/slurm/setup/cu128/verify_env.py   scripts/slurm/setup/cu128/README.md   README.md   tests/fast/scripts/slurm/setup/cu128/test_verify_env.py
git commit -m "docs: document the CUDA 12.8 H200 profile"
```

---

### Task 6: Fresh-prefix H200 qualification and idempotence

**Files:**
- Modify only if qualification exposes a source defect: files introduced in Tasks 1-5.
- Preserve evidence outside Git under the canonical remote run store.

**Interfaces:**
- Consumes: committed installer revision and an explicitly approved H200 Slurm resource tuple.
- Produces: authoritative logs, provenance, completion status, a verified fresh environment, and evidence from a second installer run.

- [ ] **Step 1: Confirm no conflicting environment mutation**

Run one bounded status snapshot:

```bash
python3 /Users/zqiu/.codex/skills/control-remote-slurm/scripts/slurm_control.py job 43131
```

From the local controller, also inspect `claude-orbit-iclr-setup` once. Do not proceed while job 43131 or another process is mutating `/data/home/zeju/miles-orbit/envs/orbit_cu128` or shared package caches.

- [ ] **Step 2: Resolve the exact qualification identity and paths**

From the remote worktree:

```bash
cd /data/home/zeju/miles-orbit/orbit/.worktrees/cu128-install-pipeline
git rev-parse HEAD
git status --short
```

Create a collision-resistant execution ID and these authoritative paths before submission:

```bash
execution_id="$(date -u +%Y%m%dT%H%M%SZ)-$(git rev-parse --short HEAD)"
run_dir="${XDG_STATE_HOME:-$HOME/.local/state}/remote-cluster-runs/slurm/orbit/codex-cu128-install-pipeline/$execution_id/fresh-install"
env_prefix="/data/home/zeju/miles-orbit/envs/orbit_cu128_repro_$execution_id"
mkdir -p "$run_dir"
```

Write `provenance.json` with commit, branch, clean state, worktree path, environment prefix, installer command, and source commits before launch. Use `stdout.log`, `stderr.log`, and `completion.status` in `run_dir`.

- [ ] **Step 3: Request explicit authorization for the expensive H200 job**

Present the exact scheduler resource tuple, source commit, environment prefix, run directory, and expected duration. Stop until the user explicitly approves submission.

- [ ] **Step 4: Submit the fresh installation once**

Use the approved project/site resource tuple and make the batch script run:

```bash
ORBIT_ENV_PREFIX="$env_prefix"   bash scripts/slurm/setup/cu128/install_env.sh
```

Bind scheduler stdout and stderr to the pre-created run directory, submit once with `sbatch --parsable`, record the job ID in provenance, and publish `completion.status` atomically.

- [ ] **Step 5: Inspect terminal status and snapshot evidence**

After terminal state, query accounting once, snapshot the run directory to the matching local run-store suffix, and inspect exit code, verifier summary, package/source provenance, and environment existence.

Expected: job exit 0 and full H200 verification reports zero failures.

- [ ] **Step 6: Run the idempotence qualification with explicit approval**

Request authorization for one second job against the same fresh environment prefix. Use the same installer commit and a new `idempotence` run label. Record pre/post package metadata and source commits.

Expected: exit 0, zero pin drift, unchanged source commits, unchanged ABI-critical package versions, and full verification success.

- [ ] **Step 7: Update documentation only if observed commands differ**

If qualification requires a documented site command or environment variable already permitted by the spec, update `scripts/slurm/setup/cu128/README.md` with the exact successful value and commit:

```bash
git add scripts/slurm/setup/cu128/README.md
git commit -m "docs: record qualified H200 installation"
```

Do not commit raw logs, run-store files, environment paths, or process-only qualification scripts.

- [ ] **Step 8: Final implementation commit check**

Confirm the branch contains the design, plan, production files, tests, and qualified documentation. Report all commit SHAs, run-store paths, job IDs, environment prefix, full verifier result, idempotence result, and any remaining risks. Do not push or create a PR without separate user authorization.
