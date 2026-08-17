from argparse import Namespace
from enum import Enum
import sys
import types

import pytest


class _StubRouterArgs:
    @staticmethod
    def add_cli_args(parser, use_router_prefix=True):
        return parser


sglang_router_module = types.ModuleType("sglang_router")
launch_router_module = types.ModuleType("sglang_router.launch_router")
launch_router_module.RouterArgs = _StubRouterArgs
sys.modules.setdefault("sglang_router", sglang_router_module)
sys.modules.setdefault("sglang_router.launch_router", launch_router_module)

sglang_args_module = types.ModuleType("orbit.backends.sglang_utils.arguments")
sglang_args_module.add_sglang_arguments = lambda parser: parser
sglang_args_module.validate_args = lambda args: None
sys.modules.setdefault("orbit.backends.sglang_utils.arguments", sglang_args_module)

chat_template_package = types.ModuleType("orbit.utils.chat_template_utils")
tito_tokenizer_module = types.ModuleType("orbit.utils.chat_template_utils.tito_tokenizer")


class _StubTITOTokenizerType(Enum):
    DEFAULT = "default"
    QWEN3 = "qwen3"


tito_tokenizer_module.TITOTokenizerType = _StubTITOTokenizerType
sys.modules.setdefault("orbit.utils.chat_template_utils", chat_template_package)
sys.modules.setdefault("orbit.utils.chat_template_utils.tito_tokenizer", tito_tokenizer_module)

misc_module = types.ModuleType("orbit.utils.misc")
misc_module.load_function = lambda path: None
sys.modules.setdefault("orbit.utils.misc", misc_module)

from orbit.utils.arguments import (  # noqa: E402
    SFT_ROLLOUT_FUNCTION_PATH,
    _apply_critic_args,
    _validate_ppo_args,
    orbit_validate_args,
)


def _base_args(**overrides):
    values = dict(
        training_mode="rl",
        rollout_function_path="orbit.rollout.sglang_rollout.generate_rollout",
        eval_function_path=None,
        eval_interval=None,
        eval_datasets=[],
        loss_type="policy_loss",
        compute_advantages_and_returns=True,
        n_samples_per_prompt=4,
        rollout_num_gpus=8,
        use_rollout_engines=True,
        offload_rollout=None,
        debug_train_only=False,
        debug_rollout_only=False,
        colocate=False,
        advantage_estimator="grpo",
        kl_coef=0,
        use_kl_loss=False,
    )
    values.update(overrides)
    return Namespace(**values)


def test_sft_mode_applies_sft_defaults_and_disables_plain_rollout_engines(monkeypatch):
    monkeypatch.setattr("orbit.utils.arguments._common_orbit_validate_args", lambda args: None)
    args = _base_args(training_mode="sft")

    orbit_validate_args(args)

    assert args.rollout_function_path == SFT_ROLLOUT_FUNCTION_PATH
    assert args.loss_type == "sft_loss"
    assert args.compute_advantages_and_returns is False
    assert args.n_samples_per_prompt == 1
    assert args.use_rollout_engines is False
    assert args.rollout_num_gpus == 0
    assert args.offload_rollout is False


def test_rl_mode_leaves_existing_defaults_unchanged(monkeypatch):
    monkeypatch.setattr("orbit.utils.arguments._common_orbit_validate_args", lambda args: None)
    args = _base_args(training_mode="rl")

    orbit_validate_args(args)

    assert args.rollout_function_path == "orbit.rollout.sglang_rollout.generate_rollout"
    assert args.loss_type == "policy_loss"
    assert args.compute_advantages_and_returns is True
    assert args.n_samples_per_prompt == 4
    assert args.use_rollout_engines is True
    assert args.rollout_num_gpus == 8


def test_sft_mode_requires_explicit_eval_function_when_eval_is_enabled(monkeypatch):
    monkeypatch.setattr("orbit.utils.arguments._common_orbit_validate_args", lambda args: None)
    args = _base_args(training_mode="sft", eval_interval=10, eval_datasets=[object()])

    with pytest.raises(ValueError, match="--eval-function-path"):
        orbit_validate_args(args)


def test_sft_mode_rejects_ppo(monkeypatch):
    monkeypatch.setattr("orbit.utils.arguments._common_orbit_validate_args", lambda args: None)
    args = _base_args(training_mode="sft", advantage_estimator="ppo")

    with pytest.raises(ValueError, match="--advantage-estimator ppo"):
        orbit_validate_args(args)


def test_ppo_applies_critic_defaults():
    args = Namespace(
        advantage_estimator="ppo",
        actor_num_gpus_per_node=2,
        actor_num_nodes=1,
        critic_mode="full",
        critic_num_gpus_per_node=None,
        critic_num_nodes=None,
        critic_load=None,
        critic_lr=None,
        load="/tmp/actor",
        lr=1e-6,
    )

    _apply_critic_args(args)

    assert args.use_critic is True
    assert args.critic_num_gpus_per_node == 2
    assert args.critic_num_nodes == 1
    assert args.critic_load == "/tmp/actor"
    assert args.critic_lr == 1e-6


def test_ppo_rejects_train_offload():
    # critic_mode defaults to "full", so _validate_ppo_args takes the
    # separate-critic branch and compares actor/critic worker counts before it
    # reaches the --offload-train rejection under test. Equal counts get us there.
    args = Namespace(
        use_critic=True,
        offload_train=True,
        actor_num_nodes=1,
        actor_num_gpus_per_node=1,
        critic_num_nodes=1,
        critic_num_gpus_per_node=1,
    )

    with pytest.raises(ValueError, match="incompatible with --offload-train"):
        _validate_ppo_args(args)
