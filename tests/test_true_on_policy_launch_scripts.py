"""Launcher-contract tests for the Qwen3-4B true-on-policy recipe.

Mirrors the ORBIT_DRY_RUN_ARGV pattern used by test_ppo_launch_scripts.py /
test_search_r1_launch_scripts.py / test_tau_bench_launch_scripts.py: run the
launcher under ORBIT_DRY_RUN_ARGV=1 (scripts/lib/launcher.sh validates the
launcher contract -- required arrays/env vars -- then prints the python argv
and exits 0 before touching Ray or GPUs) and assert on the resulting argv.
Additionally resolves the true-on-policy contract for Qwen3-4B directly via
orbit/true_on_policy/, the way test_true_on_policy_config.py does, using the
exact topology this launcher emits, to confirm the profile lookup succeeds
and the certified-layout constraints hold.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

from orbit.true_on_policy import build_true_on_policy_launch_plan, get_true_on_policy_model_profile


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "examples" / "true_on_policy" / "run-qwen3-4b-top.sh"


def _dry_run(tmp_path: Path, *, top: str) -> list[str]:
    env = os.environ.copy()
    env.update(
        {
            "ORBIT_DRY_RUN_ARGV": "1",
            "ORBIT_LOAD_CUDA_MODULES": "0",
            "DISABLE_EVAL": "1",
            "ENABLE_WANDB": "0",
            "TOP": top,
            "HF_CKPT": str(tmp_path / "hf"),
            "MEGATRON_LOAD": str(tmp_path / "megatron"),
            "TRAIN_JSONL": str(tmp_path / "train.jsonl"),
        }
    )

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.splitlines()


def _value_after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_launcher_passes_shell_syntax():
    subprocess.run(["bash", "-n", str(LAUNCHER)], cwd=REPO_ROOT, check=True)


def test_top_launcher_dry_run_stays_inside_certified_layout(tmp_path):
    argv = _dry_run(tmp_path, top="1")

    # true-on-policy is on, and the topology this recipe picks is entirely
    # inside qwen3_dense's certified layouts (dp/tp/pp train, dp/tp rollout).
    assert "--true-on-policy" in argv
    assert _value_after(argv, "--tensor-model-parallel-size") == "2"
    assert _value_after(argv, "--pipeline-model-parallel-size") == "1"
    assert _value_after(argv, "--context-parallel-size") == "1"
    assert _value_after(argv, "--rollout-num-gpus-per-engine") == "1"
    assert _value_after(argv, "--rollout-num-gpus") == "4"
    assert _value_after(argv, "--sglang-attention-backend") == "triton"

    # The contract rejects --sequence-parallel outright (config.py
    # TrueOnPolicyConfig.validate); the launcher must never emit it.
    assert "--sequence-parallel" not in argv


def test_top_launcher_dry_run_carries_qwen3_4b_model_args(tmp_path):
    argv = _dry_run(tmp_path, top="1")

    assert _value_after(argv, "--num-layers") == "36"
    assert _value_after(argv, "--hidden-size") == "2560"
    assert _value_after(argv, "--num-attention-heads") == "32"
    assert "--qk-layernorm" in argv


def test_top_off_launcher_dry_run_omits_true_on_policy(tmp_path):
    argv = _dry_run(tmp_path, top="0")

    assert "--true-on-policy" not in argv
    # Topology defaults hold regardless of TOP.
    assert _value_after(argv, "--tensor-model-parallel-size") == "2"


def test_top_launcher_topology_resolves_the_qwen3_dense_true_on_policy_contract(tmp_path):
    """The exact topology the launcher emits must clear
    TrueOnPolicyConfig.validate(): the profile lookup for Qwen3-4B succeeds,
    and none of the certified-layout constraints (no cp, no
    sequence-parallel, bf16-only, full-param-only) are violated.
    """
    argv = _dry_run(tmp_path, top="1")
    tp = int(_value_after(argv, "--tensor-model-parallel-size"))
    pp = int(_value_after(argv, "--pipeline-model-parallel-size"))
    cp = int(_value_after(argv, "--context-parallel-size"))
    rollout_gpus_per_engine = int(_value_after(argv, "--rollout-num-gpus-per-engine"))

    profile = get_true_on_policy_model_profile("Qwen3-4B")
    assert profile.family == "qwen3_dense"
    assert "tp" in profile.supported_train_layouts
    assert "cp" not in profile.supported_train_layouts

    args = SimpleNamespace(
        true_on_policy=True,
        true_on_policy_contract=None,
        hf_checkpoint="/fast/groups/ei-slm/hf_models/Qwen3-4B",
        tensor_model_parallel_size=tp,
        context_parallel_size=cp,
        pipeline_model_parallel_size=pp,
        rollout_num_gpus_per_engine=rollout_gpus_per_engine,
        sequence_parallel=False,
        peft_method="none",
        bf16=True,
        fp16=False,
        fp8=None,
    )

    plan = build_true_on_policy_launch_plan(args)

    assert plan.enabled
    assert plan.model_profile is profile
    assert plan.parallel_layout.uses_train_tp
    assert not plan.parallel_layout.uses_train_cp
    assert not plan.parallel_layout.uses_train_pp
    assert not plan.parallel_layout.uses_rollout_tp
