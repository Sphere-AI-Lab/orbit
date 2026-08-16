"""The `--lora-a-init-method` CLI surface: registered, and with the right vocabulary.

Scoped deliberately to what the lora-without-regret port added. The old repo's
`test_peft_arguments.py` also asserted the whole PEFT CLI surface (--peft-method,
--peft-variant, --oft-type, the YAML validation paths); none of that is this
port's, and copying it wholesale would pin arg-surface details this branch never
touched against a base that has since diverged.

Why `uniform` is not a legal choice, despite being the obvious name: Bridge
routes Megatron parallel linears to `ParallelLinearAdapter`, whose `_get_init_fn`
raises `NotImplementedError` for anything outside {xavier, normal, kaiming, zero}.
PEFT's `kaiming_uniform_(a=sqrt(5))` is spelled `kaiming` there, and its bound is
exactly 1/sqrt(d_in) -- the blog's convention. So `uniform` would name a real
concept with a word Bridge rejects at model-build time, i.e. after the job has
already been scheduled.
"""

import argparse

import pytest

import orbit.utils.arguments as arguments
from orbit.utils.arguments import get_orbit_extra_args_provider


class _RecordingParser:
    """Captures add_argument calls without argparse's parsing machinery."""

    def __init__(self):
        self._actions = []
        self.option_strings = {}
        self.defaults = {}

    def add_argument(self, *option_strings, **kwargs):
        action = argparse.Namespace(
            option_strings=list(option_strings),
            default=kwargs.get("default"),
            choices=kwargs.get("choices"),
            help=kwargs.get("help"),
        )
        self._actions.append(action)
        for option_string in option_strings:
            self.option_strings[option_string] = action
        return action

    def add_argument_group(self, *args, **kwargs):
        return self

    def add_mutually_exclusive_group(self, *args, **kwargs):
        return self

    def set_defaults(self, **kwargs):
        self.defaults.update(kwargs)

    def parse_known_args(self, *args, **kwargs):
        return (
            argparse.Namespace(
                rollout_function_path="orbit.rollout.sglang_rollout.generate_rollout",
                custom_generate_function_path=None,
            ),
            [],
        )


@pytest.fixture
def registered_parser(monkeypatch):
    # The experimental-rollout branch in get_orbit_extra_args_provider would
    # otherwise import orbit.experimental_rollout, which transitively pulls in
    # CUDA-dependent modules. Force the legacy surface for these tests.
    monkeypatch.setattr(arguments, "enable_experimental_rollout_refactor", lambda: False)
    parser = _RecordingParser()
    get_orbit_extra_args_provider()(parser)
    return parser


def test_lora_a_init_method_is_registered_with_the_bridge_vocabulary(registered_parser):
    assert "--lora-a-init-method" in registered_parser.option_strings
    action = registered_parser.option_strings["--lora-a-init-method"]
    assert action.choices == ["xavier", "normal", "kaiming", "zero"]
    assert action.default == "xavier", "changing the default would move every existing run's LR optimum"
    assert "uniform" not in action.choices


def test_loss_mask_type_offers_llama3_and_still_defaults_to_qwen(registered_parser):
    """Adding a choice must not move the default: existing Qwen launchers pass no
    --loss-mask-type and must keep getting the qwen mask."""
    action = registered_parser.option_strings["--loss-mask-type"]
    assert "llama3" in action.choices
    assert action.default == "qwen"


def test_eval_nll_flags_are_registered(registered_parser):
    assert registered_parser.option_strings["--eval-nll-data"].default is None
    assert registered_parser.option_strings["--eval-nll-interval"].default == 0
    assert registered_parser.option_strings["--eval-nll-micro-batch-size"].default is None


def test_lora_a_init_method_real_parser_rejects_uniform(monkeypatch):
    """Regression guard: 'uniform' was the wrong vocabulary (see lora_utils.py's real
    Bridge path, ParallelLinearAdapter._get_init_fn) and must not silently parse."""
    monkeypatch.setattr(arguments, "enable_experimental_rollout_refactor", lambda: False)
    parser = argparse.ArgumentParser()
    get_orbit_extra_args_provider()(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--lora-a-init-method", "uniform"])
