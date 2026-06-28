import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TAU_DIR = REPO_ROOT / "examples" / "tau_bench"
LAUNCHERS = {
    "full": TAU_DIR / "run-qwen3-4b-instruct-2507-bf16-tau-bench-ppo-full.sh",
    "lora": TAU_DIR / "run-qwen3-4b-instruct-2507-bf16-tau-bench-ppo-lora.sh",
    "oft": TAU_DIR / "run-qwen3-4b-instruct-2507-bf16-tau-bench-ppo-oft.sh",
}


def _dry_run(
    script: Path,
    tmp_path: Path,
    *,
    test_data: bool = False,
    extra_env: dict[str, str] | None = None,
) -> list[str]:
    env = os.environ.copy()
    env.update(
        {
            "ORBIT_DRY_RUN_ARGV": "1",
            "ORBIT_LOAD_CUDA_MODULES": "0",
            "DISABLE_EVAL": "0" if test_data else "1",
            "ENABLE_WANDB": "0",
            "TAU_USER_MODEL_PROVIDER": "mock",
            "TAU_USER_MODEL": "mock",
            "HF_CKPT": str(tmp_path / "hf"),
            "MEGATRON_LOAD": str(tmp_path / "megatron"),
            "RUN_LOG": str(tmp_path / "run.log"),
            "TRAIN_DATA": str(tmp_path / "retail_train_tasks.jsonl"),
        }
    )
    if test_data:
        env["TEST_DATA"] = str(tmp_path / "retail_dev_tasks.jsonl")
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        ["bash", str(script)],
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


def test_tau_bench_launchers_pass_shell_syntax():
    scripts = [str(TAU_DIR / "qwen3_4b_tau_bench_ppo_common.sh")]
    scripts.extend(str(script) for script in LAUNCHERS.values())

    subprocess.run(["bash", "-n", *scripts], cwd=REPO_ROOT, check=True)


def test_tau_bench_oft_launcher_dry_run_has_ppo_and_oft_defaults(tmp_path):
    argv = _dry_run(LAUNCHERS["oft"], tmp_path)

    assert _value_after(argv, "--advantage-estimator") == "ppo"
    assert _value_after(argv, "--custom-generate-function-path") == (
        "orbit_plugins.tau_bench.generate_with_tau.generate"
    )
    assert _value_after(argv, "--custom-config-path").endswith("run.tau_bench.yaml")
    assert _value_after(argv, "--input-key") == "index"
    assert _value_after(argv, "--n-samples-per-prompt") == "8"
    assert "--tau-bench-user-model-provider" not in argv
    assert _value_after(argv, "--peft-method") == "oft"
    assert _value_after(argv, "--peft-distributed-transport") == "nccl"
    assert "--adapter-double-buffer" in argv
    assert _value_after(argv, "--oft-block-size") == "32"
    assert _value_after(argv, "--target-modules") == "all-linear"


def test_tau_bench_lora_launcher_dry_run_has_lora_defaults(tmp_path):
    argv = _dry_run(LAUNCHERS["lora"], tmp_path)

    assert _value_after(argv, "--peft-method") == "lora"
    assert _value_after(argv, "--peft-distributed-transport") == "nccl"
    assert "--adapter-double-buffer" in argv
    assert _value_after(argv, "--lora-rank") == "32"
    assert _value_after(argv, "--target-modules") == "all-linear"


def test_tau_bench_launcher_uses_default_dynamic_sampling_filter(tmp_path):
    argv = _dry_run(LAUNCHERS["lora"], tmp_path)

    assert _value_after(argv, "--dynamic-sampling-filter-path") == (
        "orbit.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std"
    )


def test_tau_bench_launcher_can_disable_dynamic_sampling_filter(tmp_path):
    argv = _dry_run(
        LAUNCHERS["lora"],
        tmp_path,
        extra_env={"TAU_BENCH_DYNAMIC_SAMPLING_FILTER_PATH": "none"},
    )

    assert "--dynamic-sampling-filter-path" not in argv


def test_tau_bench_lora_ray_transport_disables_default_double_buffer(tmp_path):
    argv = _dry_run(
        LAUNCHERS["lora"],
        tmp_path,
        extra_env={"PEFT_DISTRIBUTED_TRANSPORT": "ray"},
    )

    assert _value_after(argv, "--peft-distributed-transport") == "ray"
    assert "--adapter-double-buffer" not in argv


def test_tau_bench_full_launcher_dry_run_has_no_peft_adapter(tmp_path):
    argv = _dry_run(LAUNCHERS["full"], tmp_path)

    assert _value_after(argv, "--peft-method") == "none"
    assert "--target-modules" not in argv


def test_tau_bench_launcher_includes_eval_dataset_when_test_data_is_set(tmp_path):
    argv = _dry_run(LAUNCHERS["oft"], tmp_path, test_data=True)

    assert "--eval-prompt-data" in argv
    eval_idx = argv.index("--eval-prompt-data")
    assert argv[eval_idx + 1] == "retail-dev"
    assert argv[eval_idx + 2] == str(tmp_path / "retail_dev_tasks.jsonl")
    assert _value_after(argv, "--eval-input-key") == "index"


def test_tau_bench_launcher_writes_custom_config(tmp_path):
    _dry_run(LAUNCHERS["oft"], tmp_path)

    config = (tmp_path / "run.tau_bench.yaml").read_text()
    assert 'tau_bench_env: "retail"' in config
    assert 'tau_bench_user_model_provider: "mock"' in config
    assert 'tau_bench_user_model: "mock"' in config
    assert "tau_bench_agent_max_steps: 30" in config
