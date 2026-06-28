import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PPO_LAUNCHER = REPO_ROOT / "examples" / "high_precision" / "run-qwen2_5-0_5b-bf16-math-oft-ppo.sh"


def test_ppo_launcher_exists_and_uses_separate_critic_resources():
    content = PPO_LAUNCHER.read_text(encoding="utf-8")

    assert "COLOCATE_ARGS=()" in content
    assert 'GPUS_PER_NODE="${GPUS_PER_NODE:-2}"' in content
    assert 'CRITIC_NUM_GPUS_PER_NODE="${CRITIC_NUM_GPUS_PER_NODE:-2}"' in content
    assert 'ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"' in content
    assert "--advantage-estimator ppo" in content
    assert "--critic-load" in content
    assert "--critic-save" in content
    assert "--critic-num-gpus-per-node" in content
    assert "--num-critic-only-steps 1" in content
    assert "--normalize-advantages" in content
    assert "--no-offload-train" in content
    assert "--no-offload-rollout" in content


def test_ray_defaults_count_ppo_critic_gpus():
    script = """
set -euo pipefail
source scripts/lib/common.sh
source scripts/lib/ray.sh
GPUS_PER_NODE=2
ROLLOUT_NUM_GPUS=4
RL_ARGS=(--advantage-estimator ppo)
MISC_ARGS=(--critic-num-gpus-per-node 2)
COLOCATE_ARGS=()
apply_ray_defaults
printf '%s' "${RAY_NUM_GPUS}"
"""

    result = subprocess.run(
        ["bash", "-c", script],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    assert result.stdout == "8"


def test_ppo_launcher_dry_run_prints_ppo_argv(tmp_path):
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
        ["bash", str(PPO_LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    argv = result.stdout.splitlines()

    assert "--advantage-estimator" in argv
    assert "ppo" in argv
    assert "--peft-distributed-transport" in argv
    assert "nccl" in argv
    assert "--critic-num-gpus-per-node" in argv
    assert "2" in argv
    assert "--rollout-num-gpus" in argv
    assert "4" in argv
    assert "--colocate" not in argv
