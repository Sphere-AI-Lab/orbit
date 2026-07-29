"""
this script requires a vllm instance to be run on the same node with --enable-lora flag
in another terminal, install vllm and then run the following commands
```
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 vllm serve Qwen/Qwen3-1.7B --enable-lora
```

then run this training script after the vllm instance is set up
uv run rl_lora.py --lr 1e-4 --lora_r 1 --model_id Qwen/Qwen3-1.7B --disable_wandb

this script save the LoRA weights to filesystem on each iteration
and then send requests to the vllm instance to load the lora weights and use them during inference
"""

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase
from math_utils import last_boxed_only_string, remove_boxed, is_equiv
from peft import LoraConfig, get_peft_model
from pathlib import Path
import requests
import torch
from tqdm import tqdm
import os
import random
import argparse
import wandb
import time


def parse_args():
    parser = argparse.ArgumentParser(description="Train a language model with GRPO")

    # Model configuration
    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="HuggingFace model ID to use",
    )

    # LoRA configuration
    parser.add_argument("--lora_r", type=int, default=1, help="LoRA rank")
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ],
        help="Target modules for LoRA",
    )

    # Training hyperparameters
    parser.add_argument("--lr", type=float, default=9e-5, help="Learning rate")
    parser.add_argument(
        "--n_grpo_steps",
        type=int,
        default=50,
        help="Number of GRPO training steps",
    )
    parser.add_argument(
        "--n_prompts_per_step",
        type=int,
        default=32,
        help="Number of prompts to sample per step",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=8,
        help="Number of rollouts per prompt",
    )
    parser.add_argument(
        "--epochs_per_step",
        type=int,
        default=1,
        help="Number of epochs to train per step",
    )
    parser.add_argument(
        "--micro_batch_size",
        type=int,
        default=2,
        help="Micro batch size for training",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=128,
        help="Number of gradient accumulation steps",
    )

    # Other configuration
    parser.add_argument(
        "--base_dir",
        type=str,
        default="runs",
        help="Base directory for saving runs",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default="boxed.prompt",
        help="Path to prompt template file",
    )
    parser.add_argument(
        "--vllm_url",
        type=str,
        default="http://localhost:8000",
        help="URL for vLLM API server",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Wandb configuration
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="math-grpo",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Wandb run name (defaults to run directory name)",
    )
    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable wandb logging",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Training configuration:")
    print(f"  Model ID: {args.model_id}")
    print(f"  Learning rate: {args.lr}")
    print(f"  LoRA rank: {args.lora_r}")
    print(f"  Target modules: {args.lora_target_modules}")
    print(f"  W&B logging: {'enabled' if not args.disable_wandb else 'disabled'}")
    print()

    # Setup run directory
    os.makedirs(args.base_dir, exist_ok=True)
    i = 1
    while os.path.exists(f"{args.base_dir}/{i}"):
        i += 1
    run_name = f"{args.base_dir}/{i}"
    os.makedirs(run_name)
    print(f"Created: {run_name}")

    # Initialize wandb
    if not args.disable_wandb:
        wandb_run_name = (
            args.wandb_run_name or f"{args.model_id}_{args.lr:.1e}_r{args.lora_r}"
        )
        wandb.init(
            project=args.wandb_project,
            name=wandb_run_name,
            config=vars(args),
            dir=run_name,
        )
        # Log the run directory
        wandb.config.update({"run_dir": run_name})

    # Load dataset and tokenizer
    train_dataset = load_dataset("qwedsacf/competition_math", split=f"train[:7500]")
    val_dataset = load_dataset("qwedsacf/competition_math", split=f"train[-5000:]")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token_id == None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load prompt template
    with open(args.prompt_template, "r", encoding="utf-8") as f:
        template = f.read().strip()

    def process_data(example):
        with_template = template.replace("{question}", example["problem"])
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": with_template}],
            tokenize=False,
            add_generation_prompt=True,
        )
        answer = remove_boxed(last_boxed_only_string(example["solution"]))
        return {"prompt": prompt, "answer": answer}

    train_dataset = train_dataset.map(process_data)
    val_dataset = val_dataset.map(process_data)

    # generate util for calling vllm
    def generate(
        prompts: list[str], vllm_model_id: str, temperature=0, responses_per_prompt=1
    ):
        """
        takes in a list of prompts list[str]
        and returns a list of outputs list[str]
        """
        api_url = f"{args.vllm_url}/v1/completions"
        headers = {"Content-Type": "application/json"}

        payload = {
            "model": vllm_model_id,
            "prompt": prompts,
            "max_tokens": 1024,
            "temperature": temperature,
            "n": responses_per_prompt,
        }

        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()

        outputs = [choice["text"] for choice in result["choices"]]
        return outputs

    # lora needs to be loaded onto
    loaded_loras = []

    def load_lora(lora_name):
        if lora_name in loaded_loras:
            return
        api_url = f"{args.vllm_url}/v1/load_lora_adapter"
        headers = {"Content-Type": "application/json"}

        lora_path = str(Path.cwd() / lora_name)

        payload = {
            "lora_name": lora_name,
            "lora_path": lora_path,
        }

        response = requests.post(api_url, headers=headers, json=payload)
        response.raise_for_status()
        loaded_loras.append(lora_name)

    def save_lora(model, step):
        lora_name = f"{run_name}/step={step}"
        if not os.path.exists(lora_name):
            model.save_pretrained(lora_name)
        return lora_name

    def tokenize_prompt_and_output(
        prompt_strs: list[str],
        output_strs: list[str],
        tokenizer: PreTrainedTokenizerBase,
    ) -> dict[str, torch.Tensor]:
        """
        INPUT: a list of prompts and outputs and a tokenizer
        OUTPUT: all of shape (len(prompt_strs), max_len - 1)
            - input_ids: input tokens
            - labels: output tokens (basically the inputs shifted 1 to the left)
            - response_mask: 1 indicates token that the model generate, 0 means prompt or padding
        """
        prompt_t = [tokenizer.encode(p) for p in prompt_strs]
        output_t = [tokenizer.encode(o) for o in output_strs]
        full = []
        max_len = 0

        for i in range(len(prompt_t)):
            row_len = len(prompt_t[i]) + len(output_t[i])
            max_len = max(max_len, row_len)

        for i in range(len(prompt_t)):
            padding_size = max_len - len(prompt_t[i]) - len(output_t[i])
            padding = [tokenizer.pad_token_id] * padding_size
            row = torch.tensor(prompt_t[i] + output_t[i] + padding, dtype=torch.long)
            full.append(row.unsqueeze(0))

        f2 = torch.cat(full)
        input_ids = f2[:, :-1]
        labels = f2[:, 1:]

        response_mask = torch.zeros(len(prompt_strs), max_len - 1)
        for i in range(len(prompt_t)):
            response_mask[
                i, len(prompt_t[i]) - 1 : len(prompt_t[i]) + len(output_t[i]) - 1
            ] = 1
        return {
            "input_ids": input_ids,
            "labels": labels,
            "response_mask": response_mask.bool(),
        }

    def get_response_log_probs(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        logits = model(input_ids).logits
        z = logits - logits.max(dim=-1, keepdim=True).values
        exp_z = torch.exp(z)
        denom = exp_z.sum(dim=-1, keepdim=True)
        p = exp_z / denom
        logprobs = z - torch.log(denom)
        logprobs_for_label = torch.gather(
            logprobs, dim=-1, index=labels.unsqueeze(-1)
        ).squeeze(-1)
        return logprobs_for_label

    def eval_model(model, step):
        lora_name = save_lora(model, step)
        load_lora(lora_name)

        val_prompts = val_dataset[:1000]["prompt"]

        eval_start = time.time()
        outputs = generate(val_prompts, lora_name, temperature=0)
        eval_time = time.time() - eval_start

        correct = 0
        idx_correct = []
        idx_wrong = []

        for i in range(len(outputs)):
            correct_answer = val_dataset[i]["answer"]
            generated_answer = remove_boxed(last_boxed_only_string(outputs[i]))

            if generated_answer != None and is_equiv(generated_answer, correct_answer):
                correct += 1
                idx_correct.append(i)
            else:
                idx_wrong.append(i)

        accuracy = correct / len(outputs)
        print(f"step={step}, correct: {correct} / {len(outputs)} ({accuracy:.2%})")

        # Log to wandb
        if not args.disable_wandb:
            wandb.log(
                {
                    "eval/accuracy": accuracy,
                    "eval/correct": correct,
                    "eval/total": len(outputs),
                    "eval/time_seconds": eval_time,
                },
                step=step,
            )

        return correct, idx_correct, idx_wrong

    device = "cuda:0"

    # Load model
    model_kwargs = dict(
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_cache=False,
        device_map=device,
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    # Setup LoRA
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=32,
        target_modules=args.lora_target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Initialize optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    print("Starting initial evaluation...")
    model = model.to(device)
    c, ic, iw = eval_model(model, 0)
    model.train()

    for i in range(args.n_grpo_steps):
        step_start_time = time.time()

        sample_indices = random.sample(
            range(0, len(train_dataset)), args.n_prompts_per_step
        )
        batch = train_dataset[sample_indices]

        # Save the current lora so vllm can use it
        vllm_model_id = save_lora(model, step=i)
        load_lora(vllm_model_id)

        # Generate rollouts
        generation_start = time.time()
        outputs = generate(
            batch["prompt"],
            vllm_model_id=vllm_model_id,
            temperature=1,
            responses_per_prompt=args.group_size,
        )
        generation_time = time.time() - generation_start

        # Extract answers
        generated_answers = [remove_boxed(last_boxed_only_string(o)) for o in outputs]

        # Compute advantages
        raw_reward = [
            a != None and is_equiv(a, batch["answer"][i // args.group_size])
            for i, a in enumerate(generated_answers)
        ]
        raw_reward_tensor = torch.tensor(raw_reward, dtype=torch.float).reshape(
            (args.n_prompts_per_step, args.group_size)
        )
        means = raw_reward_tensor.mean(dim=-1).unsqueeze(1)
        advantages = (raw_reward_tensor - means).reshape(
            (args.n_prompts_per_step * args.group_size,)
        )

        # Reward statistics
        train_accuracy = raw_reward_tensor.mean().item()
        reward_std = raw_reward_tensor.std().item()
        reward_max = raw_reward_tensor.max().item()
        reward_min = raw_reward_tensor.min().item()
        advantage_mean = advantages.mean().item()
        advantage_std = advantages.std().item()
        advantage_max = advantages.max().item()
        advantage_min = advantages.min().item()

        # Tokenize
        prompts_expanded = [x for x in batch["prompt"] for _ in range(args.group_size)]
        data = tokenize_prompt_and_output(prompts_expanded, outputs, tokenizer)
        input_ids = data["input_ids"].to(device)
        labels = data["labels"].to(device)
        response_mask = data["response_mask"].to(device)

        # Compute old log probs
        with torch.inference_mode():
            old_logprobs_all = []
            for b in range(len(input_ids) // args.micro_batch_size):
                idx = b * args.micro_batch_size
                end = idx + args.micro_batch_size
                x = input_ids[idx:end]
                y = labels[idx:end]
                old_logprobs_all.append(get_response_log_probs(model, x, y).detach())
            old_logprobs_all = torch.cat(old_logprobs_all, dim=0)
            assert old_logprobs_all.shape == labels.shape

        # Train for epochs_per_step
        epoch_grad_norms = []
        epoch_policy_ratios = []
        epoch_kl_divs = []

        training_start = time.time()
        for epoch in range(args.epochs_per_step):
            for b in tqdm(
                range(len(input_ids) // args.micro_batch_size),
                desc=f"Step {i + 1}/{args.n_grpo_steps}, Epoch {epoch + 1}/{args.epochs_per_step}",
            ):
                idx = b * args.micro_batch_size
                end = idx + args.micro_batch_size
                x = input_ids[idx:end]
                y = labels[idx:end]
                mask = response_mask[idx:end]
                micro_batch_adv = advantages[idx:end].unsqueeze(-1).to(device)

                # Get the per token log probs of each rollout
                policy_logprobs = get_response_log_probs(model, x, y)
                old_logprobs = old_logprobs_all[idx:end]

                # Compute per token loss
                ratio = torch.exp(policy_logprobs - old_logprobs)
                per_token_loss = -ratio * micro_batch_adv

                # Compute KL divergence (approximate)
                with torch.no_grad():
                    # KL(old || new) ≈ (ratio - 1) - log(ratio)
                    approx_kl = ((ratio - 1) - torch.log(ratio)) * mask
                    approx_kl_mean = approx_kl.sum() / mask.sum()
                    epoch_kl_divs.append(approx_kl_mean.item())

                    # Policy ratio statistics
                    policy_ratio_mean = (ratio * mask).sum() / mask.sum()
                    epoch_policy_ratios.append(policy_ratio_mean.item())

                # Apply mask so we only train on the generated tokens
                masked_loss = per_token_loss * mask

                # Compute a scalar loss and call backward()
                denom = mask.sum(dim=-1).clamp_min(1)
                loss_per_prompt = masked_loss.sum(dim=-1) / denom
                loss = loss_per_prompt.mean() / args.gradient_accumulation_steps

                loss.backward()

                if (b + 1) % args.gradient_accumulation_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    epoch_grad_norms.append(grad_norm.item())
                    optimizer.step()
                    optimizer.zero_grad()

        training_time = time.time() - training_start
        step_time = time.time() - step_start_time

        # Compute statistics
        mean_grad_norm = (
            sum(epoch_grad_norms) / len(epoch_grad_norms) if epoch_grad_norms else 0
        )
        mean_policy_ratio = (
            sum(epoch_policy_ratios) / len(epoch_policy_ratios)
            if epoch_policy_ratios
            else 0
        )
        mean_kl = sum(epoch_kl_divs) / len(epoch_kl_divs) if epoch_kl_divs else 0

        # Count total tokens generated
        total_tokens = response_mask.sum().item()
        tokens_per_sec = total_tokens / generation_time if generation_time > 0 else 0
        avg_generation_len = int(response_mask.sum(dim=-1).float().mean().item())

        # Log metrics to wandb
        if not args.disable_wandb:
            wandb.log(
                {
                    # Reward metrics
                    "train/accuracy": train_accuracy,
                    "train/reward_mean": train_accuracy,
                    "train/reward_std": reward_std,
                    "train/reward_max": reward_max,
                    "train/reward_min": reward_min,
                    # Advantage metrics
                    "train/advantage_mean": advantage_mean,
                    "train/advantage_std": advantage_std,
                    "train/advantage_max": advantage_max,
                    "train/advantage_min": advantage_min,
                    # Training dynamics
                    "train/grad_norm": mean_grad_norm,
                    "train/policy_ratio": mean_policy_ratio,
                    "train/approx_kl": mean_kl,
                    "train/avg_gen_length": avg_generation_len,
                    # Timing metrics
                    "time/step_time": step_time,
                    "time/generation_time": generation_time,
                    "time/training_time": training_time,
                    "time/tokens_per_sec": tokens_per_sec,
                },
                step=i + 1,
            )

        print(
            f"Step {i + 1}/{args.n_grpo_steps} | Train Acc: {train_accuracy:.2%} | KL: {mean_kl:.4f} | Time: {step_time:.1f}s"
        )

        if (i + 1) % 5 == 0:
            eval_model(model, i + 1)

    if not args.disable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
