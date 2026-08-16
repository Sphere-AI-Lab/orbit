"""Full fine-tuning under RL needs tensor parallelism to fit; PEFT does not.

Measured on 8xH100 with Llama-3.1-8B at TP=1 (pure data parallel, every GPU
carrying the whole model), after the train-offload fix let the arm reach its
first training step:

    torch.OutOfMemoryError: Tried to allocate 694.00 MiB.
    GPU 2 has 79.18 GiB of which 660.12 MiB is free.
    buf9 = empty_strided_cuda((s10, 1, 128256), ..., torch.float32)

128256 is the vocabulary, so that allocation is the fp32 cross-entropy logits.
Recompute was already `full`/`uniform`, so activations were not the slack. The
standing cost per GPU is `(2+4)*P/TP + 12*P/N` -- bf16 parameters, fp32
main_grad, and the DP-sharded optimizer:

    TP=1   48 + 12 = 60 GB   <- failed, ~19 GB left against a ~20 GB step
    TP=2   24 + 12 = 36 GB
    TP=4   12 + 12 = 24 GB
    TP=8    6 + 12 = 18 GB

TP also shards the vocabulary, so the logits buffer shrinks with it. TP=8 fits
but forces DP=1, which orbit's own preflight treats as degenerate
(`STAGE_GPU_REQUIREMENTS["p3"] == 2`, because DP=1 makes the reduction a
no-op), and pays a per-layer all-reduce across all 32 layers for headroom that
is not needed. Hence GPUS/2, rounded down to a power of two: DP stays >= 2.

These tests execute the launcher under ORBIT_DRY_RUN_ARGV rather than grepping
it, so they pin the value that actually reaches Megatron.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RL_LAUNCHER = REPO_ROOT / "examples" / "high_precision" / "run-llama3_1-8b-bf16-rl-math-gsm8k.sh"


def _argv(tmp_path, **overrides) -> list[str]:
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
    env.update({k: str(v) for k, v in overrides.items()})
    result = subprocess.run(
        ["bash", str(RL_LAUNCHER)],
        cwd=REPO_ROOT, env=env, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout.split()


def _flag(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


class TestFullFineTuningGetsTensorParallelism:
    @pytest.mark.parametrize("gpus,expected_tp", [(8, "4"), (4, "2"), (2, "1"), (16, "8")])
    def test_the_default_is_half_the_gpus_rounded_down_to_a_power_of_two(
        self, tmp_path, gpus, expected_tp
    ):
        """Half, not all: DP must stay >= 2. A power of two because TP has to
        divide 32 attention heads and 8 query groups -- GPUS/2 on a 6-GPU node
        would be 3, which divides neither.

        ALLOW_SMALL_FULLFT because the 2-GPU case is below the launcher's own
        FullFT floor of 4; it is here to pin the arithmetic at the bottom of the
        range, not to suggest anyone run it."""
        argv = _argv(tmp_path, PEFT_METHOD="none", GPUS_PER_NODE=gpus,
                     ALLOW_SMALL_FULLFT=1)
        assert _flag(argv, "--tensor-model-parallel-size") == expected_tp

    def test_data_parallelism_survives(self, tmp_path):
        """The reason this is not TP=GPUS. At TP=8/DP=1 the distributed
        optimizer has nothing to shard across and the gradient all-reduce
        becomes a no-op -- the degenerate case orbit's preflight already
        refuses to test on."""
        argv = _argv(tmp_path, PEFT_METHOD="none", GPUS_PER_NODE=8)
        tp = int(_flag(argv, "--tensor-model-parallel-size"))
        pp = int(_flag(argv, "--pipeline-model-parallel-size"))
        cp = int(_flag(argv, "--context-parallel-size"))
        assert 8 // (tp * pp * cp) >= 2, f"DP={8 // (tp * pp * cp)}"

    def test_tp_never_exceeds_the_query_group_count(self, tmp_path):
        """Llama-3.1-8B has 8 KV heads under GQA. TP above that cannot shard
        them, so the cap is 8 however many GPUs the node has."""
        argv = _argv(tmp_path, PEFT_METHOD="none", GPUS_PER_NODE=64)
        assert int(_flag(argv, "--tensor-model-parallel-size")) <= 8

    def test_an_explicit_override_still_wins(self, tmp_path):
        """The default is a floor for an arm that would otherwise OOM, not a
        policy. Anyone tuning throughput must be able to set it."""
        argv = _argv(tmp_path, PEFT_METHOD="none", GPUS_PER_NODE=8,
                     TENSOR_MODEL_PARALLEL_SIZE=2)
        assert _flag(argv, "--tensor-model-parallel-size") == "2"


class TestPeftIsUntouched:
    @pytest.mark.parametrize("method", ["lora", "oft"])
    def test_peft_arms_stay_at_tp_1(self, tmp_path, method):
        """LoRA and OFT fit at TP=1 -- they carry no fp32 main_grad for the base
        and no full optimizer state. Six RL PEFT arms were measured at TP=1 on
        2026-07-31; changing it would invalidate those timings and change what
        E4 compares."""
        # OFT has no default block size on purpose -- the launcher refuses to
        # invent one -- so supply the value E4 uses.
        argv = _argv(tmp_path, PEFT_METHOD=method, GPUS_PER_NODE=8, OFT_BLOCK_SIZE=1024)
        assert _flag(argv, "--tensor-model-parallel-size") == "1"

    def test_sequence_parallel_is_on_for_everyone(self, tmp_path):
        """Sequence parallelism is what makes TP shard the vocabulary logits
        rather than replicate them, so it must accompany TP > 1. It was already
        unconditional; this pins it, because dropping it would leave the
        FullFT arm with the same fp32 logits buffer TP was raised to shrink."""
        for method in ("none", "lora"):
            argv = _argv(tmp_path, PEFT_METHOD=method, GPUS_PER_NODE=8,
                         OFT_BLOCK_SIZE=1024)
            assert "--sequence-parallel" in argv, method
