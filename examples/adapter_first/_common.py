"""Shared configuration for the adapter-first GSM8K recipes."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote
from typing import Literal

import orbit.utils.external_utils.command_utils as U


@dataclass(frozen=True)
class _Recipe:
    model_name: str
    model_type: str
    checkpoint_suffix: str
    response_length: int
    max_tokens_per_gpu: int
    memory_fraction: float
    oft_block_size: int
    oft_eps: float
    native_fp8: bool = False


@dataclass
class _ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=U.create_run_id)
    peft_method: Literal["lora", "oft"] = "oft"
    num_gpus_per_node: int = 8
    actor_gpus: int = 4
    rollout_gpus: int = 4
    tensor_parallel_size: int = 1
    num_rollout: int = 500
    save_interval: int = 50
    eval_interval: int = 50
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    hf_checkpoint: str = ""
    ref_load: str = ""
    train_data: str = ""
    eval_data: str = ""
    megatron_path: str = str(U.repo_base_dir / "thirdparty/Megatron-LM")
    extra_args: str = ""


def _execute(args: _ScriptArgs, recipe: _Recipe, script_file: str):
    if args.peft_method not in {"lora", "oft"}:
        raise ValueError("peft_method must be lora or oft")
    if recipe.native_fp8 and args.peft_method != "oft":
        raise ValueError("The native FP8 recipe supports OFT only")
    if args.num_nodes != 1:
        raise ValueError("These recipes use one disaggregated node")
    if min(args.actor_gpus, args.rollout_gpus) < 1 or args.actor_gpus + args.rollout_gpus != args.num_gpus_per_node:
        raise ValueError("actor_gpus + rollout_gpus must equal num_gpus_per_node, with both pools nonempty")
    if args.tensor_parallel_size < 1 or args.actor_gpus % args.tensor_parallel_size:
        raise ValueError("tensor_parallel_size must divide actor_gpus")
    if min(args.num_rollout, args.save_interval, args.eval_interval) < 1:
        raise ValueError("Rollout count and checkpoint/evaluation intervals must be positive")

    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    checkpoint = Path(args.output_dir) / f"{recipe.model_name}_{args.peft_method}_{args.run_id}"
    ckpt_args = (
        f"--hf-checkpoint {quote(args.hf_checkpoint or str(model_dir / recipe.model_name))} "
        f"--ref-load {quote(args.ref_load or str(model_dir / (recipe.model_name + recipe.checkpoint_suffix)))} "
        f"--load {quote(str(checkpoint))} --save {quote(str(checkpoint))} "
        f"--save-interval {args.save_interval} "
    )
    peft_args = f"--peft-method {args.peft_method} --target-modules all-linear --megatron-to-hf-mode bridge "
    if args.peft_method == "lora":
        peft_args += "--lora-rank 32 --lora-alpha 32 --lora-dropout 0.0 --peft-distributed-transport ray "
    else:
        peft_args += (
            f"--oft-type canonical_oft --oft-block-size {recipe.oft_block_size} "
            f"--oft-eps {recipe.oft_eps} --adapter-double-buffer "
        )
    rollout_args = (
        f"--prompt-data {quote(args.train_data or str(data_dir / 'gsm8k/train.parquet'))} "
        "--input-key messages --label-key label --apply-chat-template --rollout-shuffle --rm-type math "
        f"--num-rollout {args.num_rollout} --rollout-batch-size 16 --n-samples-per-prompt 4 "
        f"--rollout-max-response-len {recipe.response_length} --rollout-temperature 1 --global-batch-size 64 "
    )
    eval_args = (
        f"--eval-interval {args.eval_interval} "
        f"--eval-prompt-data gsm8k {quote(args.eval_data or str(data_dir / 'gsm8k/test.parquet'))} "
        f"--n-samples-per-eval-prompt 1 --eval-max-response-len {recipe.response_length} --eval-top-k 1 "
    )
    perf_args = (
        f"--tensor-model-parallel-size {args.tensor_parallel_size} --sequence-parallel "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        "--expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        f"--use-dynamic-batch-size --max-tokens-per-gpu {recipe.max_tokens_per_gpu} "
    )
    grpo_args = (
        "--advantage-estimator grpo --kl-loss-coef 0.00 --kl-loss-type low_var_kl "
        "--kl-coef 0.00 --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
    )
    optimizer_args = (
        "--optimizer adam --lr 1e-5 --lr-decay-style constant "
        "--weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
    )
    sglang_args = f"--rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static {recipe.memory_fraction} "
    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash "
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {args.actor_gpus} "
        f"--rollout-num-gpus {args.rollout_gpus} --num-gpus-per-node {args.num_gpus_per_node} "
    )
    extra_env = {}
    if recipe.native_fp8:
        extra_env = {
            "ORBIT_RESTORE_MODELOPT_STATE": "0",
            "MEGATRON_OFT_FP8_ACTIVATION_QUANT": "w8a8",
            "MEGATRON_QWEN3_FP8_GEMM_BACKEND": "sglang_native",
            "MEGATRON_KEEP_NATIVE_FP8_WEIGHTS": "True",
            "VERL_KEEP_NATIVE_FP8_WEIGHTS": "True",
            "SGLANG_FP8_GEMM_BACKEND": "auto",
        }
    if entity := os.environ.get("WANDB_ENTITY"):
        extra_env["WANDB_ENTITY"] = entity
    wandb_args = U.get_default_wandb_args(script_file, run_name_prefix=args.peft_method, run_id=args.run_id)
    U.execute_train(
        train_args=" ".join(
            block.strip()
            for block in [
                ckpt_args, peft_args, rollout_args, optimizer_args, grpo_args, perf_args,
                eval_args, sglang_args, misc_args, wandb_args, args.extra_args,
            ]
            if block.strip()
        ),
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=recipe.model_type,
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env,
        config=args,
    )
