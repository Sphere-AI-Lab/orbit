"""Full fine-tuning must be able to offload train state for colocated RL.

Before this, `--offload-train` was refused outright for `--peft-method none`,
so an 8B FullFT RL arm kept its gradients and optimizer state resident while a
colocated SGLang tried to resume the KV cache it had paused. Measured on
8xH100: 66.69 GB used, 12.48 GB free, against 16.00 GB of paused K+V --
`torch_memory_saver ... cudaError 2 (out of memory) func=resume`, every time,
about seven minutes in. The LoRA arms on the same node sat at 43.88 GB used /
35.30 GB free and resumed fine; the ~22.8 GB between them is exactly the
gradients and optimizer state LoRA does not carry.

**Params stay resident, deliberately.** `update_weights` pushes Megatron
weights into SGLang on every rollout and does not wake the train state, so
offloading `param_data` would hand it a zero-sized storage. Under PEFT only the
adapter is pushed, which is why the frozen base can go. Megatron's own
`offload_grad_buffers` hardcodes `move_params=False` for the same reason.

The refusal was not wrong, either -- it guarded a real failure mode. `sleep()`'s
worker is `offload_megatron_frozen_base_to_cpu`, whose selector skips any param
with `requires_grad`; under FullFT that is every param, so it would plan empty
groups, log "after offload model", and free nothing. Allowing offload WITHOUT
forcing the two sub-flags on would reinstate exactly that silent no-op, which is
why `test_enabling_offload_without_the_sub_flags_would_free_nothing` exists.
"""

from __future__ import annotations

import types

import pytest

from miles.utils.arguments import _is_peft_enabled


def _args(**overrides):
    """The finalised-argument surface these checks read, nothing more."""
    base = dict(
        train_backend="megatron",
        peft_method="none",
        offload_train=True,
        offload_train_grad_buffers=None,
        offload_train_optimizer=None,
        offload_train_adapter=None,
        offload_train_async=None,
        offload_train_frozen_base_mode=None,
        offload_rollout=None,
        # read by the PEFT adapter-offload branch further down finalisation
        megatron_to_hf_mode="bridge",
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


class TestTheRefusalIsGone:
    def test_full_fine_tuning_may_now_offload_train_state(self):
        from miles.utils.arguments import _finalize_train_offload_args as finalize_train_offload_args

        args = _args()
        finalize_train_offload_args(args)  # must not raise
        assert args.offload_train is True

    def test_peft_is_unaffected(self):
        """LoRA and OFT keep the frozen-base path they already use. This change
        must be invisible to them -- every RL PEFT arm measured tonight went
        through it."""
        from miles.utils.arguments import _finalize_train_offload_args as finalize_train_offload_args

        for method in ("lora", "oft"):
            args = _args(peft_method=method)
            finalize_train_offload_args(args)
            assert args.offload_train is True


class TestTheSubFlagsAreForcedOn:
    def test_full_fine_tuning_gets_grad_buffer_and_optimizer_offload(self):
        """The load-bearing assertion. Under FullFT the frozen-base path can
        free nothing, so these two are the only things that release memory. If
        they are left off, `sleep()` is a no-op that logs as a success."""
        from miles.utils.arguments import _finalize_train_offload_args as finalize_train_offload_args

        args = _args()
        finalize_train_offload_args(args)
        assert args.offload_train_grad_buffers is True
        assert args.offload_train_optimizer is True

    def test_an_explicit_opt_out_is_refused_rather_than_silently_honoured(self):
        """Turning either off under FullFT re-creates the original bug with no
        error message. Better to refuse: the operator asked for an offload that
        would not offload."""
        from miles.utils.arguments import _finalize_train_offload_args as finalize_train_offload_args

        for flag in ("offload_train_grad_buffers", "offload_train_optimizer"):
            args = _args(**{flag: False})
            with pytest.raises(ValueError, match="free nothing|full fine-tuning"):
                finalize_train_offload_args(args)

    def test_peft_defaults_are_not_forced(self):
        """PEFT frees memory through the frozen base, so these stay opt-in --
        forcing them on would change the memory and timing profile of every
        LoRA and OFT arm already measured."""
        from miles.utils.arguments import _finalize_train_offload_args as finalize_train_offload_args

        args = _args(peft_method="lora")
        finalize_train_offload_args(args)
        assert args.offload_train_grad_buffers is False
        assert args.offload_train_optimizer is False

    def test_the_flags_stay_off_when_offload_is_off(self):
        """No offload requested, nothing forced -- otherwise the existing
        '--offload-train-grad-buffers requires --offload-train' guard fires on
        an argument the operator never passed."""
        from miles.utils.arguments import _finalize_train_offload_args as finalize_train_offload_args

        args = _args(offload_train=False)
        finalize_train_offload_args(args)
        assert args.offload_train_grad_buffers is False
        assert args.offload_train_optimizer is False


class TestParamsStayResident:
    def test_no_param_offload_flag_is_turned_on_for_full_fine_tuning(self):
        """`update_weights` reads the params every rollout without waking the
        train state. Anything that resized `param_data` to zero would surface as
        corrupt rollouts rather than an error."""
        from miles.utils.arguments import _finalize_train_offload_args as finalize_train_offload_args

        args = _args()
        finalize_train_offload_args(args)
        assert getattr(args, "offload_train_params", False) is False

    def test_adapter_offload_is_off_because_there_is_no_adapter(self):
        from miles.utils.arguments import _finalize_train_offload_args as finalize_train_offload_args

        args = _args()
        finalize_train_offload_args(args)
        assert args.offload_train_adapter is False


class TestTheFrozenBasePathIsSkipped:
    def test_the_selector_yields_nothing_when_everything_is_trainable(self):
        """The mechanism behind the whole bug, asserted directly: this is why
        the frozen-base call cannot help under FullFT, and why skipping it
        loses nothing."""
        import torch

        from orbit.megatron.peft_offload import _iter_frozen_named_params

        model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 4))
        assert list(_iter_frozen_named_params(model)) == []

        for p in model.parameters():
            p.requires_grad_(False)
        assert len(list(_iter_frozen_named_params(model))) == 4

    def test_peft_still_reaches_the_frozen_base_path(self):
        """Guards the skip from over-reaching: a frozen base under LoRA must
        still be offloaded, which is the only thing that frees memory there."""
        from miles.backends.megatron_utils.actor import _should_offload_frozen_base

        assert _should_offload_frozen_base(types.SimpleNamespace(peft_method="lora"))
        assert _should_offload_frozen_base(types.SimpleNamespace(peft_method="oft"))
        assert not _should_offload_frozen_base(types.SimpleNamespace(peft_method="none"))


class TestTheLauncherNoLongerDisablesIt:
    def test_the_rl_launcher_does_not_pass_no_offload_train_for_full(self):
        """The workaround this replaces. It was added when the refusal made
        every FullFT arm die in argument finalisation; leaving it in would keep
        the arms dying for the original reason."""
        from pathlib import Path

        script = (
            Path(__file__).resolve().parents[3]
            / "examples/high_precision/run-llama3_1-8b-bf16-rl-math-gsm8k.sh"
        )
        text = script.read_text(encoding="utf-8")
        none_branch = text.split("none)", 1)[1].split(";;", 1)[0]
        assert "--no-offload-train" not in none_branch, none_branch[:400]
