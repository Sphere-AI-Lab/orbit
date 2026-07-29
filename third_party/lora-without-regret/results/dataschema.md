wandb_sft_export.db sqlite tables:
```
run:
- id: str (project_id/run_id from wandb)
- seed: int
- model_id: str
- lora_rank: int | None
- lora_type: str | None
- batch_size: int
- optimizer: str
- lora_alpha: float | None
- max_length: int
- num_epochs: int
- learning_rate: float
- target_modules: str
- effective_batch_size: int
- gradient_accumulation_steps: int
- final_val_loss: float
- fullft_or_lora: str

run_history:
- run_id: str (project_id/run_id)
- step: int
- train_loss: float
- val_loss: float
- train_grad_norm: float
- train_epoch: float
- runtime: float
```

wandb_rl_export.db sqlite tables:
```
runs:
id: should be taken from runs[0].id
lr: float
lora_rank: int | None
seed: int
micro_batch_size: int
gradient_accumulation_steps: int
model_id: str
group_size: int
n_grpo_steps: int
epochs_per_step: int
best_eval_accuracy: float
final_eval_accuracy: float

run history:
step: int
runtime: float
eval_accuracy: float
eval_time_seconds: float
train_avg_gen_length: int
train_accuracy: float
```
NOTE: the `wandb_rl_export.db` also contains runs I tried with models other than Qwen3-1.7B so if you want recreate my data analysis you will have to filter those out.
