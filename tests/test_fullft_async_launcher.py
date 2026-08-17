import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
FULLFT_LAUNCHER = (
    REPO_ROOT / "examples" / "high_precision" / "run-qwen3-4b-instruct-2507-bf16-math-fullft-async.sh"
)
OFT_ASYNC_LAUNCHER = (
    REPO_ROOT / "examples" / "high_precision" / "run-qwen3-4b-instruct-2507-bf16-math-oft-async.sh"
)


def _dry_run_argv(launcher: Path, tmp_path: Path) -> list[str]:
    env = os.environ.copy()
    env.update(
        {
            "ORBIT_DRY_RUN_ARGV": "1",
            "ORBIT_LOAD_CUDA_MODULES": "0",
            "DISABLE_EVAL": "1",
            "ENABLE_WANDB": "0",
            "TRAIN_ROWS": "1",
            "HF_CKPT": str(tmp_path / "hf"),
            "MEGATRON_LOAD": str(tmp_path / "megatron"),
            "TRAIN_JSONL": str(tmp_path / "train.jsonl"),
        }
    )

    result = subprocess.run(
        ["bash", str(launcher)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.splitlines()


def test_fullft_async_launcher_passes_bash_syntax_check():
    subprocess.run(["bash", "-n", str(FULLFT_LAUNCHER)], cwd=REPO_ROOT, check=True)


def test_fullft_async_launcher_has_no_peft_flags_and_uses_train_async():
    content = FULLFT_LAUNCHER.read_text(encoding="utf-8")

    assert "PEFT_ARGS=()" in content
    assert "--peft-method" not in content
    assert "--adapter-double-buffer" not in content
    assert "--peft-distributed-transport" not in content
    assert "--target-modules" not in content
    assert "--oft-type" not in content
    assert "--oft-block-size" not in content
    assert "train_async.py" in content
    # Full-model Megatron train offload is unimplemented (arguments.py rejects
    # --offload-train with --peft-method none); the launcher must disable it.
    assert "--no-offload-train" in content


def test_fullft_async_launcher_dry_run_argv(tmp_path):
    argv = _dry_run_argv(FULLFT_LAUNCHER, tmp_path)

    assert argv[0] == str(REPO_ROOT / "train_async.py")
    assert "--peft-method" not in argv
    assert "--adapter-double-buffer" not in argv
    assert "--peft-distributed-transport" not in argv
    assert "--target-modules" not in argv
    assert "--colocate" not in argv
    assert "--no-offload-train" in argv
    assert "--rollout-num-gpus" in argv
    assert "--advantage-estimator" in argv
    assert "grpo" in argv


def test_fullft_async_launcher_is_mechanical_copy_of_oft_async(tmp_path):
    """The full-FT arm must differ from the OFT async arm only in the PEFT
    flags (and the save directory), or the benchmark comparison is invalid."""
    oft_argv = _dry_run_argv(OFT_ASYNC_LAUNCHER, tmp_path)
    fullft_argv = _dry_run_argv(FULLFT_LAUNCHER, tmp_path)

    def drop_save_dir(argv: list[str]) -> list[str]:
        out = []
        skip_next = False
        for token in argv:
            if skip_next:
                skip_next = False
                continue
            if token == "--save":
                skip_next = True
                continue
            out.append(token)
        return out

    oft_argv = drop_save_dir(oft_argv)
    fullft_argv = drop_save_dir(fullft_argv)

    # PEFT_ARGS is the last array in the launcher contract, so the OFT argv
    # ends with the PEFT flags; everything before them must match exactly.
    assert "--peft-method" in oft_argv
    shared = oft_argv[: oft_argv.index("--peft-method")]
    assert fullft_argv == shared
