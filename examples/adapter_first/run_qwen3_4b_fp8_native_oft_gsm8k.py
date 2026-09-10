"""Qwen3-4B OFT with native FP8 weights in both trainer and serving on GSM8K.

Requires prepared HF and Megatron release checkpoints and GSM8K parquet data.
Run inside a one-node allocation; use ORBIT_SCRIPT_EXTERNAL_RAY=1 for an existing Ray cluster.
See README.md for the precision-specific checkpoint requirements.

Args:
  --peft-method: lora or oft (only oft is supported here).
  --model-dir / --data-dir / --output-dir: Configurable asset and checkpoint roots.
  --hf-checkpoint / --ref-load: Override the inferred checkpoint locations.
  --num-rollout: Training rollouts (default 500); use 3 for a launch check.
  --run-id: Reuse a previous ID to resume its checkpoint directory.

Example:
  python -m examples.adapter_first.run_qwen3_4b_fp8_native_oft_gsm8k --model-dir /models --data-dir /datasets
"""

from dataclasses import dataclass

import typer

import orbit.utils.external_utils.command_utils as U
from examples.adapter_first._common import _Recipe, _ScriptArgs, _execute


@dataclass
class ScriptArgs(_ScriptArgs):
    pass


def execute(args: ScriptArgs):
    _execute(
        args,
        _Recipe(
            model_name="Qwen3-4B-FP8",
            model_type="qwen3-4B",
            checkpoint_suffix="_torch_dist_release",
            response_length=2048,
            max_tokens_per_gpu=16384,
            memory_fraction=0.8,
            oft_block_size=128,
            oft_eps=6e-5,
            native_fp8=True,
        ),
        __file__,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
