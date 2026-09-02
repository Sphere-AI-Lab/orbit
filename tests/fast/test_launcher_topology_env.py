import os
import subprocess
from pathlib import Path

import pytest

from tools.adapter_runtime_compare import run_compare

REPO = Path(__file__).resolve().parents[2]

OFT_05B_LAUNCHER = "examples/high_precision/run-qwen2_5-0_5b-bf16-math-oft.sh"
FULLFT_4B_LAUNCHER = "examples/high_precision/run-qwen3-4b-instruct-2507-bf16-math-fullft-async.sh"
# Historical literal read from the file before editing (--rollout-num-gpus 4).
FULLFT_4B_ROLLOUT_NUM_GPUS = "4"


def _launcher_paths() -> list[str]:
    paths: list[str] = []
    for case in run_compare.CASES:
        for field in (case.script, case.fullft_script):
            if field and field not in paths:
                paths.append(field)
    return paths


def _base_env(tmp_path: Path) -> dict[str, str]:
    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text('{"prompt": "x", "label": "1"}\n')
    hf = tmp_path / "hf"
    hf.mkdir(exist_ok=True)
    meg = tmp_path / "meg"
    meg.mkdir(exist_ok=True)
    env = dict(os.environ)
    env.update(
        {
            "ORBIT_DRY_RUN_ARGV": "1",
            "HF_CKPT": str(hf),
            "MEGATRON_LOAD": str(meg),
            "TRAIN_JSONL": str(jsonl),
            "SAVE_DIR": str(tmp_path / "save"),
            "DISABLE_EVAL": "1",
        }
    )
    return env


def _run(launcher: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(REPO / launcher)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _argv_lines(stdout: str) -> list[str]:
    return stdout.splitlines()


def _adjacent_pair_present(lines: list[str], flag: str, value: str) -> bool:
    for i, line in enumerate(lines):
        if line == flag and i + 1 < len(lines) and lines[i + 1] == value:
            return True
    return False


LAUNCHERS = _launcher_paths()


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_async_style_env_overrides_topology(launcher, tmp_path):
    env = _base_env(tmp_path)
    env.update(
        {
            "ORBIT_COLOCATE": "0",
            "GPUS_PER_NODE": "2",
            "ROLLOUT_NUM_GPUS": "2",
            "ROLLOUT_NUM_GPUS_PER_ENGINE": "2",
        }
    )
    proc = _run(launcher, env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = _argv_lines(proc.stdout)
    assert "--colocate" not in lines
    assert _adjacent_pair_present(lines, "--rollout-num-gpus", "2")
    assert _adjacent_pair_present(lines, "--rollout-num-gpus-per-engine", "2")


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_colocated_env_overrides_topology(launcher, tmp_path):
    env = _base_env(tmp_path)
    env.update(
        {
            "ORBIT_COLOCATE": "1",
            "GPUS_PER_NODE": "4",
            "ROLLOUT_NUM_GPUS": "0",
            "ROLLOUT_NUM_GPUS_PER_ENGINE": "1",
        }
    )
    proc = _run(launcher, env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = _argv_lines(proc.stdout)
    assert "--colocate" in lines
    assert _adjacent_pair_present(lines, "--rollout-num-gpus", "0")


def test_oft_05b_default_behavior_preserved(tmp_path):
    env = _base_env(tmp_path)
    proc = _run(OFT_05B_LAUNCHER, env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = _argv_lines(proc.stdout)
    assert "--colocate" in lines
    assert _adjacent_pair_present(lines, "--rollout-num-gpus", "0")


def test_fullft_4b_default_behavior_preserved(tmp_path):
    env = _base_env(tmp_path)
    proc = _run(FULLFT_4B_LAUNCHER, env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = _argv_lines(proc.stdout)
    assert "--colocate" not in lines
    assert _adjacent_pair_present(lines, "--rollout-num-gpus", FULLFT_4B_ROLLOUT_NUM_GPUS)


# ---------------------------------------------------------------------------
# Other harness-driven env knobs that launchers must honor
# ---------------------------------------------------------------------------

OFT_LAUNCHERS = [
    "examples/high_precision/run-qwen2_5-0_5b-bf16-math-oft.sh",
    "examples/high_precision/run-qwen2_5-3b-bf16-math-oft.sh",
    "examples/high_precision/run-qwen3-4b-instruct-2507-bf16-math-oft.sh",
]


def test_adapter_double_buffer_env_adds_flag(tmp_path):
    env = _base_env(tmp_path)
    env["ADAPTER_DOUBLE_BUFFER"] = "1"
    proc = _run(OFT_05B_LAUNCHER, env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "--adapter-double-buffer" in _argv_lines(proc.stdout)


def test_adapter_double_buffer_unset_adds_nothing(tmp_path):
    env = _base_env(tmp_path)
    env.pop("ADAPTER_DOUBLE_BUFFER", None)
    proc = _run(OFT_05B_LAUNCHER, env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "--adapter-double-buffer" not in _argv_lines(proc.stdout)


@pytest.mark.parametrize("launcher", OFT_LAUNCHERS)
def test_oft_block_size_env_overrides_default(tmp_path, launcher):
    env = _base_env(tmp_path)
    env["OFT_BLOCK_SIZE"] = "64"
    proc = _run(launcher, env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert _adjacent_pair_present(_argv_lines(proc.stdout), "--oft-block-size", "64")


@pytest.mark.parametrize("launcher", OFT_LAUNCHERS)
def test_oft_block_size_default_is_128(tmp_path, launcher):
    env = _base_env(tmp_path)
    env.pop("OFT_BLOCK_SIZE", None)
    proc = _run(launcher, env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert _adjacent_pair_present(_argv_lines(proc.stdout), "--oft-block-size", "128")


# --- A3 (Qwen3-4B) launcher knobs: SEED, LR and the sglang attention backend ---

A3_4B_LAUNCHERS = [
    "examples/high_precision/run-qwen3-4b-instruct-2507-bf16-math-oft.sh",
    "examples/high_precision/run-qwen3-4b-instruct-2507-bf16-math-oft-async.sh",
    FULLFT_4B_LAUNCHER,
]


def _a3_env(tmp_path: Path) -> dict[str, str]:
    env = _base_env(tmp_path)
    for key in ("SEED", "LR", "SGLANG_ATTENTION_BACKEND"):
        env.pop(key, None)
    return env


@pytest.mark.parametrize("launcher", A3_4B_LAUNCHERS)
def test_a3_launcher_defaults_preserved(tmp_path, launcher):
    proc = _run(launcher, _a3_env(tmp_path))
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = _argv_lines(proc.stdout)
    assert _adjacent_pair_present(lines, "--seed", "1234")
    assert _adjacent_pair_present(lines, "--lr", "3e-6")
    assert _adjacent_pair_present(lines, "--sglang-attention-backend", "flashinfer")


@pytest.mark.parametrize("launcher", A3_4B_LAUNCHERS)
def test_a3_launcher_env_overrides_seed_lr_and_backend(tmp_path, launcher):
    env = _a3_env(tmp_path)
    env.update({"SEED": "7", "LR": "7e-7", "SGLANG_ATTENTION_BACKEND": "triton"})
    proc = _run(launcher, env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    lines = _argv_lines(proc.stdout)
    assert _adjacent_pair_present(lines, "--seed", "7")
    assert _adjacent_pair_present(lines, "--lr", "7e-7")
    assert _adjacent_pair_present(lines, "--sglang-attention-backend", "triton")
    assert "flashinfer" not in lines


@pytest.mark.parametrize("launcher", A3_4B_LAUNCHERS)
def test_a3_launcher_distributed_optimizer_is_env_gated(tmp_path, launcher):
    # Off by default (recipe argv unchanged); USE_DISTRIBUTED_OPTIMIZER=1 adds the flag,
    # which the 4B full-FT arm needs to fit Adam state on 80 GB GPUs at TP=1.
    env = _a3_env(tmp_path)
    env.pop("USE_DISTRIBUTED_OPTIMIZER", None)
    proc = _run(launcher, env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "--use-distributed-optimizer" not in _argv_lines(proc.stdout)

    env["USE_DISTRIBUTED_OPTIMIZER"] = "1"
    proc = _run(launcher, env)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "--use-distributed-optimizer" in _argv_lines(proc.stdout)
