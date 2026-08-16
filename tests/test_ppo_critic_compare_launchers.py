from __future__ import annotations

import hashlib
import itertools
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_DIR = REPO_ROOT / "examples" / "high_precision"
COMMON_LAUNCHER = EXAMPLE_DIR / "ppo_critic_compare_common.sh"
LAUNCHERS = {
    ("controlled", "full"): EXAMPLE_DIR / "run-qwen2_5-3b-math-oft-ppo-full-critic-controlled.sh",
    ("controlled", "adapter"): EXAMPLE_DIR / "run-qwen2_5-3b-math-oft-ppo-adapter-critic-controlled.sh",
    ("budget", "full"): EXAMPLE_DIR / "run-qwen2_5-3b-math-oft-ppo-full-critic-budget.sh",
    ("budget", "adapter"): EXAMPLE_DIR / "run-qwen2_5-3b-math-oft-ppo-adapter-critic-budget.sh",
}
EXPECTED_LAYOUTS = {
    ("controlled", "full"): (1, 1, 2, 4),
    ("controlled", "adapter"): (1, 0, 2, 3),
    ("budget", "full"): (1, 1, 2, 4),
    ("budget", "adapter"): (1, 0, 3, 4),
}

IDENTITY_VALUE_FLAGS = {
    "--save",
    "--critic-save",
    "--wandb-group",
    "--wandb-run-id",
}
CRITIC_ARCHITECTURE_VALUE_FLAGS = {
    "--critic-mode",
    "--critic-load",
    "--critic-num-gpus-per-node",
}


def _launcher_env(script: Path, tmp_path: Path, *, smoke: bool) -> dict[str, str]:
    (tmp_path / "hf").mkdir(exist_ok=True)
    (tmp_path / "megatron").mkdir(exist_ok=True)
    records = {
        "train.jsonl": {"prompt": "1+1", "label": "2"},
        "math500.jsonl": {
            "prompt": "1+1",
            "label": "2",
            "metadata": {"dataset_name": "math500", "rm_type": "math_alignment"},
        },
        "aime24.jsonl": {
            "prompt": "1+1",
            "label": "2",
            "metadata": {"dataset_name": "aime24", "rm_type": "math_alignment"},
        },
        "amc23.jsonl": {
            "prompt": "1+1",
            "label": "2",
            "metadata": {"dataset_name": "amc23", "rm_type": "math_alignment"},
        },
        "smoke-eval.jsonl": {"prompt": "1+1", "label": "2"},
    }
    for filename, record in records.items():
        (tmp_path / filename).write_text(json.dumps(record) + "\n", encoding="utf-8")

    env = os.environ.copy()
    for name in (
        "AIME24_JSONL",
        "AMC23_JSONL",
        "CRITIC_LOAD",
        "DISABLE_EVAL",
        "EVAL_INTERVAL",
        "EVAL_MAX_RESPONSE_LEN",
        "EVAL_ORBIT_DIR",
        "GLOBAL_BATCH_SIZE",
        "MATH500_JSONL",
        "MAX_TOKENS_PER_GPU",
        "NUM_ROLLOUT",
        "N_SAMPLES_PER_PROMPT",
        "ORBIT_PEFT_ARENA_REWARD_TIMEOUT_S",
        "ALLOW_DIRTY_BENCHMARK",
        "PEFT_ARENA_REWARD_TIMEOUT_S",
        "PPO_CRITIC_COMPARE_LOCK_ROOT",
        "PPO_CRITIC_COMPARE_PREPARE_ONLY",
        "RESUME_DIR",
        "ROLLOUT_BATCH_SIZE",
        "ROLLOUT_MAX_RESPONSE_LEN",
        "ROLLOUT_SEED",
        "RUN_LOG",
        "SAVE_DIR",
        "SAVE_INTERVAL",
        "SAVE_ROOT",
        "SEED",
        "SGLANG_MAX_RUNNING_REQUESTS",
        "SMOKE",
        "TEST_JSONL",
        "TRAIN_JSONL",
        "WANDB_GROUP",
        "WANDB_PROJECT",
        "WANDB_RESUME",
        "WANDB_RUN_ID",
    ):
        env.pop(name, None)
    env.update(
        {
            "ORBIT_DRY_RUN_ARGV": "1",
            "ORBIT_LOAD_CUDA_MODULES": "0",
            "ORBIT_TMPDIR": str(tmp_path / "tmp"),
            "DISABLE_EVAL": "0",
            "ENABLE_WANDB": "1",
            "WANDB_MODE": "offline",
            "ALLOW_DIRTY_BENCHMARK": "1",
            "HF_CKPT": str(tmp_path / "hf"),
            "MEGATRON_LOAD": str(tmp_path / "megatron"),
            "TRAIN_JSONL": str(tmp_path / "train.jsonl"),
            "MATH500_JSONL": str(tmp_path / "math500.jsonl"),
            "AIME24_JSONL": str(tmp_path / "aime24.jsonl"),
            "AMC23_JSONL": str(tmp_path / "amc23.jsonl"),
            "TEST_JSONL": str(tmp_path / "smoke-eval.jsonl"),
            "RUN_LOG": str(tmp_path / f"{script.stem}.log"),
            "SAVE_ROOT": str(tmp_path / "checkpoints"),
            "SEED": "17",
            "SMOKE": "1" if smoke else "0",
        }
    )
    return env


def _dry_run(script: Path, tmp_path: Path, *, smoke: bool) -> list[str]:
    env = _launcher_env(script, tmp_path, smoke=smoke)
    env["ORBIT_DRY_RUN_ARGV"] = "1"

    result = subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.splitlines()


def _prepare_only(
    script: Path,
    tmp_path: Path,
    *,
    smoke: bool = False,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    assert "PPO_CRITIC_COMPARE_PREPARE_ONLY" in COMMON_LAUNCHER.read_text(encoding="utf-8"), "prepare-only launcher guard is missing; refusing to risk starting Ray from this test"
    env = _launcher_env(script, tmp_path, smoke=smoke)
    env.update(
        {
            "ORBIT_DRY_RUN_ARGV": "0",
            "PPO_CRITIC_COMPARE_PREPARE_ONLY": "1",
            "PPO_CRITIC_COMPARE_LOCK_ROOT": str(tmp_path / "locks"),
        }
    )
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )


def _save_dir(tmp_path: Path, panel: str, critic_mode: str, *, smoke: bool = False) -> Path:
    model_name = "Qwen2.5-0.5B-Instruct" if smoke else "Qwen2.5-3B-Instruct"
    flavor = "smoke" if smoke else "benchmark"
    return tmp_path / "checkpoints" / f"{model_name}_{panel}_{critic_mode}_seed17_{flavor}"


def _metadata(path: Path) -> dict[str, str]:
    entries = [line.split("\t", 1) for line in path.read_text(encoding="utf-8").splitlines()]
    assert all(len(entry) == 2 for entry in entries)
    metadata = dict(entries)
    assert len(metadata) == len(entries)
    return metadata


def _recorded_argv(path: Path) -> list[list[str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) % 2 == 0
    assert all(lines[index].startswith("# ") for index in range(0, len(lines), 2))
    return [shlex.split(lines[index]) for index in range(1, len(lines), 2)]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def benchmark_runs(tmp_path_factory: pytest.TempPathFactory):
    tmp_path = tmp_path_factory.mktemp("ppo-critic-compare")
    return tmp_path, {key: _dry_run(script, tmp_path, smoke=False) for key, script in LAUNCHERS.items()}


@pytest.fixture(scope="module")
def smoke_runs(tmp_path_factory: pytest.TempPathFactory):
    tmp_path = tmp_path_factory.mktemp("ppo-critic-compare-smoke")
    return tmp_path, {key: _dry_run(script, tmp_path, smoke=True) for key, script in LAUNCHERS.items()}


def _value_after(argv: list[str], flag: str) -> str:
    assert argv.count(flag) == 1, f"expected exactly one {flag!r} in resolved argv"
    index = argv.index(flag)
    assert index + 1 < len(argv), f"{flag!r} has no value"
    return argv[index + 1]


def _values_until_next_flag(argv: list[str], flag: str) -> list[str]:
    assert argv.count(flag) == 1, f"expected exactly one {flag!r} in resolved argv"
    index = argv.index(flag) + 1
    values: list[str] = []
    while index < len(argv) and not argv[index].startswith("--"):
        values.append(argv[index])
        index += 1
    return values


def _without_value_flags(argv: list[str], flags: set[str]) -> list[str]:
    normalized: list[str] = []
    index = 0
    while index < len(argv):
        token = argv[index]
        if token in flags:
            assert index + 1 < len(argv), f"{token!r} has no value"
            index += 2
            continue
        normalized.append(token)
        index += 1
    return normalized


def test_comparison_launchers_pass_shell_syntax():
    subprocess.run(
        ["bash", "-n", str(COMMON_LAUNCHER), *(str(path) for path in LAUNCHERS.values())],
        cwd=REPO_ROOT,
        check=True,
    )


def test_common_launcher_is_source_only():
    result = subprocess.run(
        ["bash", str(COMMON_LAUNCHER)],
        cwd=REPO_ROOT,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 2
    assert result.stdout == ""
    assert "Source this file from a PPO critic-comparison wrapper" in result.stderr


def test_private_ray_cleanup_runs_registered_launcher_hook(tmp_path):
    marker = tmp_path / "cleanup-hook-ran"
    env = os.environ.copy()
    env["HOOK_MARKER"] = str(marker)
    result = subprocess.run(
        [
            "bash",
            "-c",
            """
set -euo pipefail
source scripts/lib/ray.sh
ORBIT_RAY_LIFECYCLE=private
RAY_START_PID=
PORT_LOCK_FDS=()
orbit_launcher_exit_hook() { printf 'released\\n' >"${HOOK_MARKER}"; }
cleanup_private_ray
""",
        ],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert marker.read_text(encoding="utf-8") == "released\n"


def test_main_launcher_rejects_null_training_label(tmp_path):
    script = LAUNCHERS[("controlled", "full")]
    env = _launcher_env(script, tmp_path, smoke=False)
    (tmp_path / "train.jsonl").write_text(
        json.dumps({"prompt": "1+1", "label": None}) + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "dataset preflight failed" in result.stderr
    assert "label is missing or null" in result.stderr


def test_main_launcher_rejects_unaligned_eval_metadata(tmp_path):
    script = LAUNCHERS[("controlled", "adapter")]
    env = _launcher_env(script, tmp_path, smoke=False)
    (tmp_path / "math500.jsonl").write_text(
        json.dumps(
            {
                "prompt": "1+1",
                "label": "2",
                "metadata": {"dataset_name": "math500", "rm_type": "generic"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["bash", str(script)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "dataset preflight failed" in result.stderr
    assert "metadata.rm_type must be 'math_alignment'" in result.stderr


@pytest.mark.parametrize("panel,critic_mode", LAUNCHERS, ids=lambda value: str(value))
def test_dry_run_resource_and_critic_mode_contract(benchmark_runs, panel: str, critic_mode: str):
    _, runs = benchmark_runs
    argv = runs[(panel, critic_mode)]
    actor_gpus, critic_gpus, rollout_gpus, occupied_gpus = EXPECTED_LAYOUTS[(panel, critic_mode)]

    assert argv[0] == str(REPO_ROOT / "train.py")
    assert _value_after(argv, "--actor-num-gpus-per-node") == str(actor_gpus)
    assert _value_after(argv, "--rollout-num-gpus") == str(rollout_gpus)
    assert _value_after(argv, "--rollout-num-gpus-per-engine") == "1"
    assert _value_after(argv, "--num-gpus-per-node") == "4"
    assert _value_after(argv, "--critic-mode") == critic_mode
    assert "--colocate" not in argv

    if critic_mode == "full":
        assert _value_after(argv, "--critic-num-gpus-per-node") == "1"
        assert _value_after(argv, "--critic-load") == _value_after(argv, "--load")
    else:
        assert critic_gpus == 0
        assert "--critic-num-gpus-per-node" not in argv
        assert "--critic-load" not in argv

    requested_gpus = actor_gpus + critic_gpus + rollout_gpus
    assert requested_gpus == occupied_gpus
    assert occupied_gpus == (3 if (panel, critic_mode) == ("controlled", "adapter") else 4)


def test_all_pairwise_argv_differences_are_explicitly_allowed(benchmark_runs):
    _, runs = benchmark_runs

    for left, right in itertools.combinations(LAUNCHERS, 2):
        ignored_flags = set(IDENTITY_VALUE_FLAGS)
        if left[1] != right[1]:
            ignored_flags.update(CRITIC_ARCHITECTURE_VALUE_FLAGS)
        if EXPECTED_LAYOUTS[left][2] != EXPECTED_LAYOUTS[right][2]:
            ignored_flags.add("--rollout-num-gpus")

        assert _without_value_flags(runs[left], ignored_flags) == _without_value_flags(runs[right], ignored_flags), f"unexpected resolved-argv drift between {left} and {right}"


def test_common_model_ppo_seed_eval_and_determinism_contract(benchmark_runs):
    tmp_path, runs = benchmark_runs

    for argv in runs.values():
        # Qwen2.5-3B actor and canonical OFT configuration.
        assert _value_after(argv, "--num-layers") == "36"
        assert _value_after(argv, "--hidden-size") == "2048"
        assert _value_after(argv, "--ffn-hidden-size") == "11008"
        assert _value_after(argv, "--num-attention-heads") == "16"
        assert _value_after(argv, "--num-query-groups") == "2"
        assert _value_after(argv, "--peft-method") == "oft"
        assert _value_after(argv, "--peft-distributed-transport") == "nccl"
        assert _value_after(argv, "--oft-type") == "canonical_oft"
        assert _value_after(argv, "--oft-block-size") == "32"
        assert _value_after(argv, "--oft-eps") == "6e-5"
        assert _value_after(argv, "--target-modules") == "all-linear"
        assert "--adapter-double-buffer" in argv

        # Shared PPO objective and one update pass per rollout.
        assert _value_after(argv, "--advantage-estimator") == "ppo"
        assert _value_after(argv, "--eps-clip") == "0.2"
        assert _value_after(argv, "--eps-clip-high") == "0.28"
        assert _value_after(argv, "--value-clip") == "0.2"
        assert _value_after(argv, "--gamma") == "1.0"
        assert _value_after(argv, "--lambd") == "1.0"
        assert _value_after(argv, "--num-critic-only-steps") == "1"
        assert "--normalize-advantages" in argv
        assert "--calculate-per-token-loss" in argv

        # Matched prompt order and deterministic rollout service behavior.
        assert _value_after(argv, "--seed") == "17"
        assert _value_after(argv, "--rollout-seed") == "17"
        assert "--rollout-shuffle" in argv
        assert _value_after(argv, "--rollout-temperature") == "1.0"
        assert _value_after(argv, "--rollout-top-p") == "1.0"
        assert _value_after(argv, "--rollout-top-k") == "-1"
        assert "--sglang-enable-deterministic-inference" in argv
        assert _value_after(argv, "--sglang-router-policy") == "round_robin"

        # Identical training and held-out data paths across all four runs.
        assert _value_after(argv, "--hf-checkpoint") == str(tmp_path / "hf")
        assert _value_after(argv, "--load") == str(tmp_path / "megatron")
        assert _value_after(argv, "--prompt-data") == str(tmp_path / "train.jsonl")
        assert _values_until_next_flag(argv, "--eval-prompt-data") == [
            "math500",
            str(tmp_path / "math500.jsonl"),
            "aime24",
            str(tmp_path / "aime24.jsonl"),
            "amc23",
            str(tmp_path / "amc23.jsonl"),
        ]
        assert _value_after(argv, "--eval-interval") == "25"
        assert _value_after(argv, "--eval-max-response-len") == "1024"
        assert _values_until_next_flag(argv, "--eval-pass-k-values") == ["1", "2", "4"]

        # Checkpoints must retain optimizer/scheduler and native RNG state where supported.
        assert _value_after(argv, "--save-interval") == "200"
        assert _value_after(argv, "--megatron-to-hf-mode") == "bridge"
        assert _value_after(argv, "--save").endswith("/actor")
        assert _value_after(argv, "--critic-save").endswith("/critic")
        assert "--no-save-optim" not in argv
        assert "--no-save-rng" not in argv


def test_run_identities_and_checkpoint_roots_are_unique(benchmark_runs):
    _, runs = benchmark_runs
    actor_saves: set[str] = set()
    critic_saves: set[str] = set()
    wandb_groups: set[str] = set()
    wandb_run_ids: set[str] = set()

    for (panel, critic_mode), argv in runs.items():
        actor_save = _value_after(argv, "--save")
        critic_save = _value_after(argv, "--critic-save")
        wandb_group = _value_after(argv, "--wandb-group")
        wandb_run_id = _value_after(argv, "--wandb-run-id")
        identity = f"{panel}_{critic_mode}_seed17_benchmark"

        assert identity in actor_save
        assert identity in critic_save
        assert identity in wandb_group
        assert Path(actor_save).parent == Path(critic_save).parent
        assert wandb_run_id.startswith("orbit")
        assert len(wandb_run_id) == 25
        actor_saves.add(actor_save)
        critic_saves.add(critic_save)
        wandb_groups.add(wandb_group)
        wandb_run_ids.add(wandb_run_id)

    assert len(actor_saves) == len(LAUNCHERS)
    assert len(critic_saves) == len(LAUNCHERS)
    assert len(wandb_groups) == len(LAUNCHERS)
    assert len(wandb_run_ids) == len(LAUNCHERS)


def test_smoke_mode_applies_small_model_and_schedule_overrides(smoke_runs):
    tmp_path, runs = smoke_runs

    for (panel, critic_mode), argv in runs.items():
        assert _value_after(argv, "--num-layers") == "24"
        assert _value_after(argv, "--hidden-size") == "896"
        assert _value_after(argv, "--ffn-hidden-size") == "4864"
        assert _value_after(argv, "--num-attention-heads") == "14"
        assert _value_after(argv, "--num-rollout") == "2"
        assert _value_after(argv, "--rollout-batch-size") == "8"
        assert _value_after(argv, "--n-samples-per-prompt") == "1"
        assert _value_after(argv, "--global-batch-size") == "8"
        assert _value_after(argv, "--rollout-max-response-len") == "128"
        assert _value_after(argv, "--max-tokens-per-gpu") == "2048"
        assert _value_after(argv, "--save-interval") == "1"
        assert _value_after(argv, "--eval-interval") == "1"
        assert _value_after(argv, "--eval-max-response-len") == "128"
        assert _value_after(argv, "--sglang-max-running-requests") == "64"
        assert _values_until_next_flag(argv, "--eval-prompt-data") == [
            "math",
            str(tmp_path / "smoke-eval.jsonl"),
        ]
        assert "--eval-pass-k-values" not in argv
        assert f"{panel}_{critic_mode}_seed17_smoke" in _value_after(argv, "--save")
        assert "Qwen2.5-0.5B-Instruct" in _value_after(argv, "--save")
        assert "--no-save-optim" not in argv
        assert "--no-save-rng" not in argv


def test_prepare_only_creates_fresh_metadata_and_records_exact_argv(tmp_path):
    panel = "controlled"
    critic_mode = "full"
    script = LAUNCHERS[(panel, critic_mode)]
    expected_argv = _dry_run(script, tmp_path, smoke=False)

    result = _prepare_only(script, tmp_path)

    assert result.returncode == 0, result.stdout + result.stderr
    save_dir = _save_dir(tmp_path, panel, critic_mode)
    metadata_path = save_dir / "benchmark-metadata.tsv"
    argv_path = save_dir / "launch-argv.log"
    assert metadata_path.is_file()
    assert argv_path.is_file()
    assert list(save_dir.glob(".benchmark-metadata.*")) == []

    metadata = _metadata(metadata_path)
    expected_metadata = {
        "schema": "2",
        "model_tag": "qwen25_3b",
        "model_dir_name": "Qwen2.5-3B-Instruct",
        "run_flavor": "benchmark",
        "panel": "controlled",
        "critic_mode": "full",
        "seed": "17",
        "rollout_seed": "17",
        "git_commit": subprocess.check_output(["git", "rev-parse", "--verify", "HEAD"], cwd=REPO_ROOT, text=True).strip(),
        "allow_dirty_benchmark": "1",
        "common_launcher_sha256": _file_sha256(COMMON_LAUNCHER),
        "wrapper_sha256": _file_sha256(script),
        "orbit_entrypoint": str(REPO_ROOT / "train.py"),
        "orbit_entrypoint_sha256": _file_sha256(REPO_ROOT / "train.py"),
        "hf_checkpoint": str(tmp_path / "hf"),
        "hf_checkpoint_manifest_sha256": hashlib.sha256(b"").hexdigest(),
        "megatron_base": str(tmp_path / "megatron"),
        "megatron_base_manifest_sha256": hashlib.sha256(b"").hexdigest(),
        "train_jsonl": str(tmp_path / "train.jsonl"),
        "train_jsonl_sha256": _file_sha256(tmp_path / "train.jsonl"),
        "math500_jsonl": str(tmp_path / "math500.jsonl"),
        "math500_jsonl_sha256": _file_sha256(tmp_path / "math500.jsonl"),
        "aime24_jsonl": str(tmp_path / "aime24.jsonl"),
        "aime24_jsonl_sha256": _file_sha256(tmp_path / "aime24.jsonl"),
        "amc23_jsonl": str(tmp_path / "amc23.jsonl"),
        "amc23_jsonl_sha256": _file_sha256(tmp_path / "amc23.jsonl"),
        "test_jsonl": str(tmp_path / "smoke-eval.jsonl"),
        "test_jsonl_sha256": _file_sha256(tmp_path / "smoke-eval.jsonl"),
        "disable_eval": "0",
        "reward_function": "orbit.rollout.rm_hub.peft_arena_reward.peft_arena_reward",
        "reward_timeout_seconds": "60",
        "math_eval_semantics": "math_alignment",
        "num_rollout": "500",
        "rollout_batch_size": "64",
        "samples_per_prompt": "4",
        "global_batch_size": "64",
        "rollout_max_response_len": "1024",
        "eval_max_response_len": "1024",
        "max_tokens_per_gpu": "8192",
        "save_interval": "200",
        "eval_interval": "25",
        "actor_gpus": "1",
        "critic_gpus": "1",
        "rollout_gpus": "2",
        "ray_num_gpus": "4",
        "ray_num_cpus": "32",
        "sglang_mem_fraction_static": "0.60",
        "sglang_max_running_requests": "1024",
        "sglang_deterministic_inference": "1",
        "wandb_enabled": "1",
        "wandb_mode": "offline",
        "wandb_project": "orbit-ppo-critic-compare",
        "wandb_group": _value_after(expected_argv, "--wandb-group"),
        "wandb_run_id": _value_after(expected_argv, "--wandb-run-id"),
        "wandb_resume": "allow",
    }
    dynamic_metadata_keys = {"git_dirty", "git_diff_sha256", "git_status_sha256"}
    assert set(metadata) == set(expected_metadata) | dynamic_metadata_keys
    assert {key: metadata[key] for key in expected_metadata} == expected_metadata
    assert metadata["git_dirty"] in {"0", "1"}
    assert len(metadata["git_diff_sha256"]) == 64
    int(metadata["git_diff_sha256"], 16)
    assert len(metadata["git_status_sha256"]) == 64
    int(metadata["git_status_sha256"], 16)
    assert _recorded_argv(argv_path) == [expected_argv]


def test_prepare_only_rejects_unrecognized_artifact_in_fresh_save_dir(tmp_path):
    panel = "budget"
    critic_mode = "full"
    script = LAUNCHERS[(panel, critic_mode)]
    save_dir = _save_dir(tmp_path, panel, critic_mode)
    save_dir.mkdir(parents=True)
    artifact = save_dir / "unexpected-checkpoint.bin"
    artifact.write_bytes(b"not a benchmark checkpoint")

    result = _prepare_only(script, tmp_path)

    assert result.returncode == 2
    output = result.stdout + result.stderr
    assert "SAVE_DIR is not fresh; unrecognized artifact" in output
    assert str(artifact) in output
    assert f"Set RESUME_DIR={save_dir} to resume" in output
    assert artifact.read_bytes() == b"not a benchmark checkpoint"
    assert not (save_dir / "benchmark-metadata.tsv").exists()
    assert not (save_dir / "launch-argv.log").exists()
    assert list(save_dir.glob(".benchmark-metadata.*")) == []


def test_prepare_only_canonicalizes_save_dir_before_shared_lock(tmp_path):
    script = LAUNCHERS[("controlled", "full")]
    canonical_save_dir = tmp_path / "checkpoints" / "canonical-run"
    canonical_save_dir.mkdir(parents=True)
    lock_dir = Path(f"{canonical_save_dir}.launch-lock")
    lock_dir.mkdir()
    (lock_dir / "owner.tsv").write_text("host\ttest-owner\n", encoding="utf-8")
    aliased_save_dir = canonical_save_dir.parent / "nested" / ".." / canonical_save_dir.name

    result = _prepare_only(script, tmp_path, extra_env={"SAVE_DIR": str(aliased_save_dir)})

    assert result.returncode == 2
    output = result.stdout + result.stderr
    assert f"another process is already launching this benchmark run: {canonical_save_dir}" in output
    assert "host\ttest-owner" in output
    assert lock_dir.is_dir()


def test_prepare_only_accepts_synthetic_adapter_resume_with_non_newline_tracker(tmp_path):
    panel = "controlled"
    critic_mode = "adapter"
    script = LAUNCHERS[(panel, critic_mode)]

    fresh_result = _prepare_only(script, tmp_path)
    assert fresh_result.returncode == 0, fresh_result.stdout + fresh_result.stderr
    save_dir = _save_dir(tmp_path, panel, critic_mode)
    metadata_path = save_dir / "benchmark-metadata.tsv"
    argv_path = save_dir / "launch-argv.log"
    original_metadata = metadata_path.read_bytes()

    critic_dir = save_dir / "critic"
    critic_dir.mkdir(parents=True)
    critic_tracker = critic_dir / "latest_checkpointed_iteration.txt"
    critic_tracker.write_bytes(b"7")
    adapter_dir = save_dir / "actor" / "iter_0000007" / "adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_megatron_tp0_pp0.pt").write_bytes(b"adapter")
    (adapter_dir / "training_state_rank0.pt").write_bytes(b"training-state")

    resume_result = _prepare_only(script, tmp_path, extra_env={"RESUME_DIR": str(save_dir)})

    assert resume_result.returncode == 0, resume_result.stdout + resume_result.stderr
    assert critic_tracker.read_bytes() == b"7"
    assert metadata_path.read_bytes() == original_metadata
    launches = _recorded_argv(argv_path)
    assert len(launches) == 2
    fresh_argv, resume_argv = launches
    assert "--peft-adapter-path" not in fresh_argv
    assert "--critic-load" not in fresh_argv
    assert _value_after(resume_argv, "--peft-adapter-path") == str(adapter_dir)
    assert _value_after(resume_argv, "--critic-load") == str(critic_dir)
    assert _value_after(resume_argv, "--save") == str(save_dir / "actor")
    assert _value_after(resume_argv, "--critic-save") == str(critic_dir)
    assert _value_after(resume_argv, "--critic-mode") == "adapter"
