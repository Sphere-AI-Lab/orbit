from __future__ import annotations

from argparse import Namespace
from typing import Literal


TrainingStateMode = Literal["adapter", "full_model"]


def resolve_training_state_mode(args: Namespace) -> TrainingStateMode:
    peft_method = getattr(args, "peft_method", "none")
    if peft_method in {"oft", "lora"}:
        return "adapter"
    return "full_model"


def uses_adapter_state(args: Namespace) -> bool:
    return resolve_training_state_mode(args) == "adapter"


def should_backup_actor_after_train(args: Namespace) -> bool:
    if uses_adapter_state(args):
        return bool(getattr(args, "keep_old_actor", False))
    return True
