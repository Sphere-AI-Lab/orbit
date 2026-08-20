import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_extra_train_args_reach_argv(tmp_path):
    jsonl = tmp_path / "train.jsonl"
    jsonl.write_text('{"prompt": "x", "label": "1"}\n')
    hf = tmp_path / "hf"; hf.mkdir()
    meg = tmp_path / "meg"; meg.mkdir()
    env = dict(os.environ)
    env.update({
        "ORBIT_DRY_RUN_ARGV": "1",
        "EXTRA_TRAIN_ARGS": "--sglang-enable-metrics",
        "HF_CKPT": str(hf), "MEGATRON_LOAD": str(meg),
        "TRAIN_JSONL": str(jsonl), "SAVE_DIR": str(tmp_path / "save"),
        "DISABLE_EVAL": "1",
    })
    proc = subprocess.run(
        ["bash", str(REPO / "examples/high_precision/run-qwen2_5-0_5b-bf16-math-oft.sh")],
        env=env, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "--sglang-enable-metrics" in proc.stdout
