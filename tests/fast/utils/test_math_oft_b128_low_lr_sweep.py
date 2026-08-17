"""Behavioral contract for the dedicated Math OFT BS128 low-LR sweep."""

import os
import re
import subprocess
from pathlib import Path

from tools.lora_regret.arms import ALL_MODULES, MATRICES, e4_arms

HIDDEN, FFN, QKV = 4096, 14336, 6144
SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "lora_regret"
WRAPPER = SCRIPT_DIR / "run_e4_math_oft_b128_low_lr_8gpu.sh"
EXPECTED_LRS = (1e-7, 3e-7, 1e-6, 3e-6, 1e-5)
EXPECTED_NAMES = (
    "oftlow-b128-all-math-lr1e-07-s0",
    "oftlow-b128-all-math-lr3e-07-s0",
    "oftlow-b128-all-math-lr1e-06-s0",
    "oftlow-b128-all-math-lr3e-06-s0",
    "oftlow-b128-all-math-lr1e-05-s0",
)


def _arms():
    from tools.lora_regret.arms import e4_math_oft_b128_low_arms

    return e4_math_oft_b128_low_arms(HIDDEN, FFN, seed=0, qkv_output_size=QKV)


def test_matrix_builds_exactly_the_requested_five_math_bs128_arms():
    arms = _arms()

    assert tuple(arm.lr for arm in arms) == EXPECTED_LRS
    assert tuple(arm.name for arm in arms) == EXPECTED_NAMES
    assert {arm.method for arm in arms} == {"oft"}
    assert {arm.oft_block_size for arm in arms} == {128}
    assert {arm.target_modules for arm in arms} == {ALL_MODULES}
    assert {arm.dataset for arm in arms} == {"math"}
    assert {arm.seed for arm in arms} == {0}
    assert all(arm.matched_ratio is not None for arm in arms)


def test_matrix_is_disjoint_from_the_existing_e4_campaign():
    assert not ({arm.name for arm in _arms()} & {arm.name for arm in e4_arms()})


def test_registry_builds_the_same_five_arms():
    registered = MATRICES["e4oftb128low"](HIDDEN, FFN, QKV, 0, None, None)

    assert registered == _arms()


def test_matrix_routes_through_the_rl_accuracy_stack_with_its_own_project():
    from tools.lora_regret.preflight import EXPECTED_ARMS, STAGE_GPU_REQUIREMENTS
    from tools.lora_regret.sweep import (
        MATRIX_LAUNCHERS,
        MATRIX_METRICS,
        MATRIX_PROJECTS,
        RL_LAUNCHER,
        wandb_project,
    )

    assert MATRIX_LAUNCHERS["e4oftb128low"] == RL_LAUNCHER
    assert MATRIX_METRICS["e4oftb128low"] == "accuracy"
    assert MATRIX_PROJECTS["e4oftb128low"] == "rl-b128-low-lr"
    assert wandb_project("e4oftb128low", None, "math", "oft") == (
        "math-rl-b128-low-lr-oft"
    )
    assert EXPECTED_ARMS["e4oftb128low"] == 5
    assert STAGE_GPU_REQUIREMENTS["e4oftb128low"] == 8


def _fake_campaign_python(tmp_path: Path) -> Path:
    """Record the campaign boundary without importing the unavailable GPU stack."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = fake_bin / "python"
    python.write_text(
        """#!/usr/bin/env bash
if [[ "${1:-}" == "-c" ]]; then
    exit 0
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "tools.lora_regret.preflight" ]]; then
    printf 'preflight\t%s\n' "${4:-}" >> "${CAPTURE_FILE}"
    exit 0
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "tools.lora_regret.sweep" ]]; then
    printf 'sweep\t%s\t%s\t%s\t%s\t%s\n' \\
        "${MATRIX:-}" "${METHOD_RE:-}" "${RESULTS:-}" \\
        "${EXPECT_ARMS:-}" "${ALLOW_OFT:-}" >> "${CAPTURE_FILE}"
    printf '%s\n' \\
        'ARM=oftlow-b128-all-math-lr1e-07-s0 PEFT_METHOD=oft' \\
        'ARM=oftlow-b128-all-math-lr3e-07-s0 PEFT_METHOD=oft' \\
        'ARM=oftlow-b128-all-math-lr1e-06-s0 PEFT_METHOD=oft' \\
        'ARM=oftlow-b128-all-math-lr3e-06-s0 PEFT_METHOD=oft' \\
        'ARM=oftlow-b128-all-math-lr1e-05-s0 PEFT_METHOD=oft'
    printf '5 arms selected, 0 already done, 5 to run\n' >&2
    exit 0
fi
exit 99
""",
        encoding="utf-8",
    )
    python.chmod(0o755)
    return fake_bin


def _campaign_env(tmp_path: Path) -> tuple[dict[str, str], Path]:
    fake_bin = _fake_campaign_python(tmp_path)
    capture = tmp_path / "campaign-boundary.tsv"
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:{env['PATH']}",
            "VIRTUAL_ENV": str(tmp_path / "venv"),
            "CUDA_HOME": str(tmp_path),
            "UV_CACHE_DIR": str(tmp_path / "uv-cache"),
            "SKIP_PREFLIGHT": "0",
            "DRY_RUN": "1",
            "CAPTURE_FILE": str(capture),
        }
    )
    return env, capture


def test_dedicated_wrapper_drives_the_real_campaign_with_the_complete_selection(tmp_path):
    env, capture = _campaign_env(tmp_path)

    result = subprocess.run(
        ["bash", str(WRAPPER)],
        cwd=SCRIPT_DIR.parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, (result.stdout, result.stderr)
    rows = [line.split("\t") for line in capture.read_text(encoding="utf-8").splitlines()]
    assert rows[0] == ["preflight", "e4oftb128low"]
    assert rows[1][0] == "sweep"
    _, matrix, method_re, results, expected_arms, allow_oft = rows[1]
    assert matrix == "e4oftb128low"
    assert results == "results/e4_math_oft_b128_low_lr.jsonl"
    assert expected_arms == "5"
    assert allow_oft == "1"
    assert [arm.name for arm in _arms() if re.search(method_re, arm.name)] == list(
        EXPECTED_NAMES
    )
    assert "5 arms selected, 5 to run" in result.stdout


def test_campaign_honors_a_wrapper_selected_preflight_stage(tmp_path):
    env, capture = _campaign_env(tmp_path)
    env.update(
        {
            "MATRIX": "e4oftb128low",
            "METHOD_RE": "^oftlow-b128-all-math-lr",
            "RESULTS": str(tmp_path / "results.jsonl"),
            "EXPECT_ARMS": "5",
            "ALLOW_OFT": "1",
            "PREFLIGHT_STAGE": "e4oftb128low",
        }
    )

    result = subprocess.run(
        ["bash", str(SCRIPT_DIR / "campaign.sh")],
        cwd=SCRIPT_DIR.parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, (result.stdout, result.stderr)
    first_row = capture.read_text(encoding="utf-8").splitlines()[0].split("\t")
    assert first_row == ["preflight", "e4oftb128low"]
