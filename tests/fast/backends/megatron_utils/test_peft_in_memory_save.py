"""Regression coverage for fault-recovery saves of PEFT models."""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-fast")


from argparse import Namespace
from unittest.mock import MagicMock, patch


_MODEL_MODULE = "miles.backends.megatron_utils.model"


def test_non_persistent_peft_save_uses_the_in_memory_checkpoint_manager():
    """PEFT fault recovery needs the full trainer state, not adapter files on disk."""
    from miles.backends.megatron_utils.model import save

    checkpoint_manager = object()
    checkpointing_context = {"local_checkpoint_manager": checkpoint_manager}
    args = Namespace(ci_test=False, ci_save_model_hash=False)
    model = [MagicMock()]
    optimizer = MagicMock()
    scheduler = MagicMock()

    with (
        patch(f"{_MODEL_MODULE}.get_args", return_value=args),
        patch(f"{_MODEL_MODULE}.should_disable_forward_pre_hook", return_value=False),
        patch(f"{_MODEL_MODULE}.is_peft_model", return_value=True),
        patch(f"{_MODEL_MODULE}.save_checkpoint_with_peft") as save_adapter_checkpoint,
        patch(f"{_MODEL_MODULE}.save_checkpoint") as save_full_checkpoint,
    ):
        save(
            iteration=17,
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            checkpointing_context=checkpointing_context,
            non_persistent_ckpt=True,
        )

    save_adapter_checkpoint.assert_not_called()
    save_full_checkpoint.assert_called_once()
    assert save_full_checkpoint.call_args.kwargs["checkpointing_context"] is checkpointing_context
    assert save_full_checkpoint.call_args.kwargs["non_persistent_ckpt"] is True
