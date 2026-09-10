"""Qwen2.5-0.5B LoRA/OFT with BF16 training and serving on GSM8K.

Requires prepared HF and Megatron release checkpoints and GSM8K parquet data.
Run inside a one-node allocation; use ORBIT_SCRIPT_EXTERNAL_RAY=1 for an existing Ray cluster.
See README.md for the precision-specific checkpoint requirements.

Args:
  --peft-method: lora or oft (default: oft).
  --model-dir / --data-dir / --output-dir: Configurable asset and checkpoint roots.
  --hf-checkpoint / --ref-load: Override the inferred checkpoint locations.
  --num-rollout: Training rollouts (default 500); use 3 for a launch check.
  --run-id: Reuse a previous ID to resume its checkpoint directory.

Example:
  python -m examples.adapter_first.run_qwen25_05b_gsm8k --model-dir /models --data-dir /datasets
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
            model_name="Qwen2.5-0.5B-Instruct",
            model_type="qwen2.5-0.5B",
            checkpoint_suffix="_torch_dist",
            response_length=512,
            max_tokens_per_gpu=9216,
            memory_fraction=0.7,
            oft_block_size=32,
            oft_eps=1e-5,
            native_fp8=False,
        ),
        __file__,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
