"""Behavioral contract for the Math OFT BS128 refinement sweep."""

import os
import re
import subprocess
from pathlib import Path

import pytest

from tools.lora_regret.arms import (
    ALL_MODULES,
    MATRICES,
    e4_arms,
    e4_math_oft_b128_low_arms,
)

HIDDEN, FFN, QKV = 4096, 14336, 6144
SCRIPT_DIR = Path(__file__).resolve().parents[3] / "scripts" / "lora_regret"
WRAPPER_A = SCRIPT_DIR / "run_e4_math_oft_b128_refine_a_8gpu.sh"
WRAPPER_B = SCRIPT_DIR / "run_e4_math_oft_b128_refine_b_8gpu.sh"
EXPECTED_LRS = (5e-6, 6e-6, 7e-6, 8e-6, 9e-6, 2e-5)
EXPECTED_NAMES = (
    "oftrefine-b128-all-math-lr5e-06-s0",
    "oftrefine-b128-all-math-lr6e-06-s0",
    "oftrefine-b128-all-math-lr7e-06-s0",
    "oftrefine-b128-all-math-lr8e-06-s0",
    "oftrefine-b128-all-math-lr9e-06-s0",
    "oftrefine-b128-all-math-lr2e-05-s0",
)
SPLITS = (
    (
        WRAPPER_A,
        "results/e4_math_oft_b128_refine_a.jsonl",
        EXPECTED_NAMES[:3],
    ),
    (
        WRAPPER_B,
        "results/e4_math_oft_b128_refine_b.jsonl",
        EXPECTED_NAMES[3:],
    ),
)


def _arms():
    from tools.lora_regret.arms import e4_math_oft_b128_refine_arms

    return e4_math_oft_b128_refine_arms(
        HIDDEN, FFN, seed=0, qkv_output_size=QKV
    )


def test_matrix_builds_the_six_literal_math_bs128_arms():
    arms = _arms()

    assert tuple(arm.lr for arm in arms) == EXPECTED_LRS
    assert tuple(arm.name for arm in arms) == EXPECTED_NAMES
    assert {arm.method for arm in arms} == {"oft"}
    assert {arm.oft_block_size for arm in arms} == {128}
    assert {arm.target_modules for arm in arms} == {ALL_MODULES}
    assert {arm.dataset for arm in arms} == {"math"}
    assert {arm.seed for arm in arms} == {0}
    assert all(arm.matched_ratio is not None for arm in arms)


def test_matrix_names_are_disjoint_from_prior_e4_and_low_lr_arms():
    names = {arm.name for arm in _arms()}

    assert not names & {arm.name for arm in e4_arms()}
    assert not names & {arm.name for arm in e4_math_oft_b128_low_arms()}


def test_registry_builds_the_same_six_arms():
    registered = MATRICES["e4oftb128refine"](
        HIDDEN, FFN, QKV, 0, None, None
    )

    assert registered == _arms()


def test_matrix_routes_through_the_rl_accuracy_stack():
    from tools.lora_regret.preflight import EXPECTED_ARMS, STAGE_GPU_REQUIREMENTS
    from tools.lora_regret.sweep import (
        MATRIX_LAUNCHERS,
        MATRIX_METRICS,
        MATRIX_PROJECTS,
        RL_LAUNCHER,
        wandb_project,
    )

    assert MATRIX_LAUNCHERS["e4oftb128refine"] == RL_LAUNCHER
    assert MATRIX_METRICS["e4oftb128refine"] == "accuracy"
    assert MATRIX_PROJECTS["e4oftb128refine"] == "rl-b128-refine-lr"
    assert wandb_project("e4oftb128refine", None, "math", "oft") == (
        "math-rl-b128-refine-lr-oft"
    )
    assert EXPECTED_ARMS["e4oftb128refine"] == 6
    assert STAGE_GPU_REQUIREMENTS["e4oftb128refine"] == 8


def _fake_campaign_python(tmp_path: Path) -> Path:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = fake_bin / "python"
    python.write_text(
        r'''#!/usr/bin/env bash
if [[ "${1:-}" == "-c" ]]; then
    exit 0
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "tools.lora_regret.preflight" ]]; then
    printf 'preflight\t%s\n' "${4:-}" >> "${CAPTURE_FILE}"
    exit 0
fi
if [[ "${1:-}" == "-m" && "${2:-}" == "tools.lora_regret.sweep" ]]; then
    printf 'sweep\t%s\t%s\t%s\t%s\t%s\n' \
        "${MATRIX:-}" "${METHOD_RE:-}" "${RESULTS:-}" \
        "${EXPECT_ARMS:-}" "${ALLOW_OFT:-}" >> "${CAPTURE_FILE}"
    case "${RESULTS:-}" in
        *refine_a.jsonl)
            printf '%s\n' \
                'ARM=oftrefine-b128-all-math-lr5e-06-s0 PEFT_METHOD=oft' \
                'ARM=oftrefine-b128-all-math-lr6e-06-s0 PEFT_METHOD=oft' \
                'ARM=oftrefine-b128-all-math-lr7e-06-s0 PEFT_METHOD=oft'
            ;;
        *refine_b.jsonl)
            printf '%s\n' \
                'ARM=oftrefine-b128-all-math-lr8e-06-s0 PEFT_METHOD=oft' \
                'ARM=oftrefine-b128-all-math-lr9e-06-s0 PEFT_METHOD=oft' \
                'ARM=oftrefine-b128-all-math-lr2e-05-s0 PEFT_METHOD=oft'
            ;;
        *) exit 98 ;;
    esac
    printf '3 arms selected, 0 already done, 3 to run\n' >&2
    exit 0
fi
exit 99
''',
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


@pytest.mark.parametrize(("wrapper", "ledger", "expected_names"), SPLITS)
def test_each_wrapper_owns_three_arms_and_drives_the_real_campaign(
    tmp_path: Path,
    wrapper: Path,
    ledger: str,
    expected_names: tuple[str, ...],
):
    env, capture = _campaign_env(tmp_path)

    result = subprocess.run(
        ["bash", str(wrapper), "--model", "llama3.1-8b"],
        cwd=SCRIPT_DIR.parents[1],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, (result.stdout, result.stderr)
    rows = [line.split("\t") for line in capture.read_text().splitlines()]
    assert rows[0] == ["preflight", "e4oftb128refine"]
    _, matrix, method_re, results, expected_arms, allow_oft = rows[1]
    assert matrix == "e4oftb128refine"
    assert results == ledger
    assert expected_arms == "3"
    assert allow_oft == "1"
    assert [arm.name for arm in _arms() if re.search(method_re, arm.name)] == list(
        expected_names
    )
    assert all(name in result.stdout for name in expected_names)
    assert not ({*EXPECTED_NAMES} - {*expected_names}) & set(result.stdout.split())
    assert "3 arms selected, 3 to run" in result.stdout
