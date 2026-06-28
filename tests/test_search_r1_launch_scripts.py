import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SEARCH_R1_DIR = REPO_ROOT / "examples" / "search_r1"
LAUNCHERS = {
    "full": SEARCH_R1_DIR / "run-qwen2_5-3b-bf16-search-r1-ppo-full.sh",
    "lora": SEARCH_R1_DIR / "run-qwen2_5-3b-bf16-search-r1-ppo-lora.sh",
    "oft": SEARCH_R1_DIR / "run-qwen2_5-3b-bf16-search-r1-ppo-oft.sh",
}
LAUNCHERS_05B = {
    "full": SEARCH_R1_DIR / "run-qwen2_5-0_5b-bf16-search-r1-ppo-full.sh",
    "lora": SEARCH_R1_DIR / "run-qwen2_5-0_5b-bf16-search-r1-ppo-lora.sh",
    "oft": SEARCH_R1_DIR / "run-qwen2_5-0_5b-bf16-search-r1-ppo-oft.sh",
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
            "HF_CKPT": str(tmp_path / "hf"),
            "MEGATRON_LOAD": str(tmp_path / "megatron"),
            "RUN_LOG": str(tmp_path / "run.log"),
            "TRAIN_DATA": str(tmp_path / "train.parquet"),
        }
    )
    if test_data:
        env["TEST_DATA"] = str(tmp_path / "eval.parquet")
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


def test_search_r1_launchers_pass_shell_syntax():
    scripts = [str(SEARCH_R1_DIR / "qwen2_5_3b_search_r1_ppo_common.sh")]
    scripts.extend(str(script) for script in LAUNCHERS.values())
    scripts.extend(str(script) for script in LAUNCHERS_05B.values())

    subprocess.run(["bash", "-n", *scripts], cwd=REPO_ROOT, check=True)


def test_launcher_process_env_sets_short_tmpdir():
    result = subprocess.run(
        [
            "bash",
            "-lc",
            "source scripts/lib/common.sh; configure_process_env; printf '%s\n' \"$TMPDIR\" \"$TMP\" \"$TEMP\"",
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )

    tmpdir, tmp, temp = result.stdout.splitlines()
    assert tmpdir.startswith("/tmp/orbit-")
    assert tmp == tmpdir
    assert temp == tmpdir


def test_search_r1_oft_launcher_dry_run_has_ppo_and_oft_defaults(tmp_path):
    argv = _dry_run(LAUNCHERS["oft"], tmp_path)

    assert _value_after(argv, "--advantage-estimator") == "ppo"
    assert _value_after(argv, "--n-samples-per-prompt") == "8"
    assert _value_after(argv, "--custom-generate-function-path") == (
        "orbit_plugins.search_r1.generate_with_search.generate"
    )
    assert _value_after(argv, "--custom-rm-path") == "orbit_plugins.search_r1.generate_with_search.reward_func"
    assert _value_after(argv, "--custom-config-path").endswith("run.search_r1.yaml")
    assert "--search-r1-timeout" not in argv
    assert _value_after(argv, "--peft-method") == "oft"
    assert _value_after(argv, "--peft-distributed-transport") == "nccl"
    assert "--adapter-double-buffer" in argv
    assert _value_after(argv, "--oft-block-size") == "32"
    assert _value_after(argv, "--target-modules") == "all-linear"


def test_search_r1_lora_launcher_dry_run_has_lora_defaults(tmp_path):
    argv = _dry_run(LAUNCHERS["lora"], tmp_path)

    assert _value_after(argv, "--peft-method") == "lora"
    assert _value_after(argv, "--peft-distributed-transport") == "nccl"
    assert "--adapter-double-buffer" in argv
    assert _value_after(argv, "--lora-rank") == "64"
    assert _value_after(argv, "--target-modules") == "all-linear"


def test_search_r1_lora_ray_transport_disables_default_double_buffer(tmp_path):
    argv = _dry_run(
        LAUNCHERS["lora"],
        tmp_path,
        extra_env={"PEFT_DISTRIBUTED_TRANSPORT": "ray"},
    )

    assert _value_after(argv, "--peft-distributed-transport") == "ray"
    assert "--adapter-double-buffer" not in argv


def test_search_r1_full_launcher_dry_run_has_no_peft_adapter(tmp_path):
    argv = _dry_run(LAUNCHERS["full"], tmp_path)

    assert _value_after(argv, "--peft-method") == "none"
    assert "--target-modules" not in argv


def test_search_r1_launcher_includes_eval_dataset_when_test_data_is_set(tmp_path):
    argv = _dry_run(LAUNCHERS["oft"], tmp_path, test_data=True)

    assert "--eval-prompt-data" in argv
    eval_idx = argv.index("--eval-prompt-data")
    assert argv[eval_idx + 1] == "search_r1"
    assert argv[eval_idx + 2] == str(tmp_path / "eval.parquet")
    assert _value_after(argv, "--eval-label-key") == "reward_model"


def test_search_r1_launcher_writes_custom_config(tmp_path):
    _dry_run(LAUNCHERS["oft"], tmp_path)

    config = (tmp_path / "run.search_r1.yaml").read_text()
    assert 'search_r1_backend: "local"' in config
    assert 'search_r1_local_url: "http://127.0.0.1:8000/retrieve"' in config
    assert "search_r1_timeout: 120" in config


def test_search_r1_qwen25_05b_launcher_uses_model_override(tmp_path):
    argv = _dry_run(LAUNCHERS_05B["oft"], tmp_path)

    assert _value_after(argv, "--num-layers") == "24"
    assert _value_after(argv, "--hidden-size") == "896"
    assert "Qwen2.5-0.5B-Instruct_search_r1_ppo_oft" in _value_after(argv, "--save")
    assert _value_after(argv, "--peft-method") == "oft"
