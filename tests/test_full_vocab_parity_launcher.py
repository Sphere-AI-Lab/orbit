import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = (
    REPO_ROOT
    / "examples"
    / "high_precision"
    / "run-qwen3-1_7b-bf16-openreasoning-opd-full-vocab-lora-fkl.sh"
)


def _value_after(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_qwen3_full_vocab_parity_launcher_dry_run(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "ORBIT_DRY_RUN_ARGV": "1",
            "ORBIT_LOAD_CUDA_MODULES": "0",
            "ENABLE_WANDB": "0",
            "HF_CKPT": str(tmp_path / "student-hf"),
            "MEGATRON_LOAD": str(tmp_path / "student-megatron"),
            "OPD_TEACHER_CKPT": str(tmp_path / "teacher-hf"),
            "TRAIN_JSONL": str(tmp_path / "train.parquet"),
        }
    )

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    argv = result.stdout.splitlines()

    assert argv[0] == str(REPO_ROOT / "train.py")
    assert _value_after(argv, "--opd-jsd-beta") == "0.0"
    assert _value_after(argv, "--rollout-temperature") == "0.7"
    assert _value_after(argv, "--lr") == "5e-6"
    assert _value_after(argv, "--lr-decay-style") == "cosine"
    assert _value_after(argv, "--tensor-model-parallel-size") == "2"
    assert _value_after(argv, "--actor-num-gpus-per-node") == "2"
    assert _value_after(argv, "--num-gpus-per-node") == "2"
    assert _value_after(argv, "--lora-rank") == "64"
    assert _value_after(argv, "--lora-alpha") == "32"
    assert _value_after(argv, "--sglang-attention-backend") == "fa3"
    assert _value_after(argv, "--opd-teacher-max-running-requests") == "8"
    assert _value_after(argv, "--opd-teacher-max-prefill-tokens") == "4096"
    assert _value_after(argv, "--n-samples-per-eval-prompt") == "16"
    assert _value_after(argv, "--eval-top-p") == "0.95"
    assert _value_after(argv, "--eval-temperature") == "1.0"
    assert "--opd-defer-full-vocab-scoring" in argv
    assert "--custom-rm-path" in argv
    assert "--colocate" in argv
