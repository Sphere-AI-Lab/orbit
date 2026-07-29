"""
CUDA_VISIBLE_DEVICES=0 uv run sft_lora.py \
  --model-id Qwen/Qwen3-4B \
  --lr 2e-4 \
  --lora-rank 64 \
  --lora-type all \
  --output-dir ./lora_model \
  --wandb-project qwen-lora-test \
  --wandb-run-name lr_2e-4
"""

import argparse
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import random
import numpy as np


def set_seed(seed):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make deterministic (may impact performance)
    # but doesn't seem to have any perf impact on my setup
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a language model with LoRA")
    parser.add_argument(
        "--model-id",
        type=str,
        default="Qwen/Qwen3-4B",
        help="Model ID from HuggingFace Hub (default: Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--lora-rank", type=int, default=128, help="LoRA rank (default: 128)"
    )
    parser.add_argument(
        "--lora-type",
        type=str,
        default="all",
        choices=["all", "mlp", "attn"],
        help="LoRA target modules: 'all' (MLP+attention), 'mlp' (gate/up/down), 'attn' (q/k/v/o) (default: all)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="lora-finetuning",
        help="W&B project name (default: lora-finetuning)",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="W&B run name (default: auto-generated)",
    )
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Training batch size (default: 2)"
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=16,
        help="Gradient accumulation steps (default: 16)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=1,
        help="Number of training epochs (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./lora_model",
        help="Directory to save the model (default: ./lora_model)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    set_seed(args.seed)

    # Determine target modules based on lora_type
    if args.lora_type == "all":
        target_modules = [
            "gate_proj",
            "up_proj",
            "down_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ]
    if args.lora_type == "mlp":
        target_modules = ["gate_proj", "up_proj", "down_proj"]
    elif args.lora_type == "attn":
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    # Calculate effective batch size
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps

    # Initialize W&B
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "model_id": args.model_id,
                "learning_rate": args.lr,
                "lora_rank": args.lora_rank,
                "lora_alpha": 32,
                "lora_type": args.lora_type,
                "target_modules": target_modules,
                "batch_size": args.batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_epochs": args.num_epochs,
                "max_length": 2048,
                "optimizer": "AdamW",
                "seed": args.seed,
            },
            tags=["lora", "finetuning"],
        )

    print(f"Training configuration:")
    print(f"  Model ID: {args.model_id}")
    print(f"  Learning rate: {args.lr}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  LoRA type: {args.lora_type}")
    print(f"  Target modules: {target_modules}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  W&B logging: {'enabled' if use_wandb else 'disabled'}")
    print()

    # Load dataset and tokenizer
    dataset = load_dataset("HuggingFaceH4/no_robots", split="train[:6400]")
    val_dataset = load_dataset("HuggingFaceH4/no_robots", split="test[:100]")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # used to help us mask out '<|im_start|>assistant\n' in tokenize_function
    gen_prompt_len = len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "test"}], add_generation_prompt=True
        )
    ) - len(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "test"}], add_generation_prompt=False
        )
    )

    gen_prompt_tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}], add_generation_prompt=True
    )[-gen_prompt_len:]

    # Qwen3-4B tokenizer does not support return_assistant_tokens_mask so we implement it ourselves
    # label_ids and attention_mask should be same as calling tokenizer(messages)
    # but labels are modified such that any token not part of an assistant message is masked with -100
    def tokenize_function(examples):
        all_input_ids = []
        all_labels = []
        all_attention_masks = []

        for messages in examples["messages"]:
            # Start with empty
            current_length = 0
            input_ids = []
            labels = []
            max_length = 2048

            for i, message in enumerate(messages):
                # Tokenize conversation up to and including this message
                text_so_far = tokenizer.apply_chat_template(
                    messages[: i + 1], tokenize=False
                )
                tokens_so_far = tokenizer(
                    text_so_far,
                    add_special_tokens=False,  # Template already added them
                    truncation=True,
                    max_length=max_length,
                )["input_ids"]

                # New tokens from this message
                new_tokens = tokens_so_far[current_length:]

                # Add to input_ids
                input_ids.extend(new_tokens)

                # Create labels for these new tokens
                if message["role"] == "assistant":
                    # For assistant messages, we want to train on the content
                    if new_tokens[:gen_prompt_len] == gen_prompt_tokens:
                        # mask out formatting tokens if they exist
                        new_tokens[:gen_prompt_len] = [-100] * gen_prompt_len
                    labels.extend(new_tokens)
                else:
                    # Mask user/system messages
                    labels.extend([-100] * len(new_tokens))

                current_length = len(tokens_so_far)

            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]

            # Create attention mask
            attention_mask = [1] * len(input_ids)

            all_input_ids.append(input_ids)
            all_labels.append(labels)
            all_attention_masks.append(attention_mask)

        return {
            "input_ids": all_input_ids,
            "labels": all_labels,
            "attention_mask": all_attention_masks,
        }

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    tokenized_val_dataset = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    # Load model
    model_kwargs = dict(
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_cache=False,
        device_map="auto",
    )
    model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)

    # Setup LoRA
    peft_config = LoraConfig(
        r=args.lora_rank, lora_alpha=32, target_modules=target_modules
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Custom collator that handles labels
    def collate_fn(batch):
        # Pad sequences
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids = []
        labels = []
        attention_mask = []

        for item in batch:
            # Pad input_ids and attention_mask
            padding_len = max_len - len(item["input_ids"])

            input_ids.append(item["input_ids"] + [tokenizer.pad_token_id] * padding_len)
            attention_mask.append(item["attention_mask"] + [0] * padding_len)

            # Pad labels with -100 (ignore index)
            labels.append(item["labels"] + [-100] * padding_len)

        return {
            "input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attention_mask),
        }

    # Create DataLoader
    train_dataloader = DataLoader(
        tokenized_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_dataloader = DataLoader(
        tokenized_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    def eval(model, step=None):
        model.eval()
        total_loss = 0
        # ORBIT-LOCAL ADDITION (instrumentation only -- val_loss below is untouched).
        # HF's loss is a mean over the scored tokens WITHIN each batch, so averaging it
        # across batches weights a short batch equally with a long one. Orbit reports a
        # GLOBAL token-weighted NLL, so gate G2 needs that reduction from this side too:
        # accumulate loss * n_scored_tokens and divide once at the end. The denominator
        # matches HF's own -- labels are shifted internally, so position 0 never scores.
        total_weighted_loss = 0.0
        total_tokens = 0
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                loss = model(**batch).loss
                total_loss += loss.item()
                n_tokens = int((batch["labels"][:, 1:] != -100).sum().item())
                total_weighted_loss += loss.item() * n_tokens
                total_tokens += n_tokens
        val_loss = total_loss / len(val_dataloader)
        token_weighted_loss = (
            total_weighted_loss / total_tokens if total_tokens else float("nan")
        )
        print(f"val_loss: {val_loss:.4f} at step {step}")
        print(
            f"val_loss_token_weighted: {token_weighted_loss:.6f} "
            f"tokens: {total_tokens} batches: {len(val_dataloader)} at step {step}"
        )

        if use_wandb and step is not None:
            wandb.log(
                {"val/loss": val_loss, "val/loss_token_weighted": token_weighted_loss},
                step=step,
            )

        model.train()
        return val_loss

    # Training setup
    gradient_accumulation_steps = args.gradient_accumulation_steps
    device = model.device

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    print("Starting initial evaluation...")
    eval(model)
    model.train()
    global_step = 0
    total_loss = 0
    prev_step_loss_acc = 0
    prev_step_loss = 0

    for epoch in range(args.num_epochs):
        progress_bar = tqdm(
            train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}"
        )

        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

            # Backward pass
            loss.backward()

            # Update weights every gradient_accumulation_steps
            if (step + 1) % gradient_accumulation_steps == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                # Log training metrics
                if use_wandb:
                    wandb.log(
                        {
                            "train/loss": prev_step_loss_acc,
                            "train/epoch": epoch + (step + 1) / len(train_dataloader),
                            "train/grad_norm": grad_norm,
                        },
                        step=global_step,
                    )

                # Periodic evaluation
                if global_step % 10 == 0:
                    eval(model, step=global_step)

                prev_step_loss = prev_step_loss_acc
                prev_step_loss_acc = 0

            # Track loss
            prev_step_loss_acc += loss.item()
            total_loss += loss.item() * gradient_accumulation_steps
            avg_loss = total_loss / (step + 1)

            # Update progress bar
            progress_bar.set_postfix(
                {
                    "avg_loss": f"{avg_loss:.4f}",
                    "step": global_step,
                    "prev_step_loss": prev_step_loss,
                }
            )

    print("Training complete!")

    # Final evaluation
    print("Running final evaluation...")
    final_val_loss = eval(model, step=global_step)

    # Log final summary
    if use_wandb:
        wandb.summary["final_val_loss"] = final_val_loss
        wandb.summary["total_steps"] = global_step

    # Save the model
    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)

    # Finish W&B run
    if use_wandb:
        wandb.finish()

    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
