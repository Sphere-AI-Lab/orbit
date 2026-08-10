import argparse

import pytest

from orbit.utils.arguments import (
    _apply_critic_args,
    _validate_ppo_args,
    uses_adapter_critic,
    uses_separate_critic,
)


def _base_args(**overrides):
    defaults = dict(
        advantage_estimator="ppo",
        critic_mode="full",
        critic_num_gpus_per_node=None,
        critic_num_nodes=None,
        critic_load=None,
        critic_lr=None,
        actor_num_gpus_per_node=4,
        actor_num_nodes=1,
        load="/ckpt/base",
        lr=1e-6,
        peft_method="lora",
        train_backend="megatron",
        keep_old_actor=False,
        num_critic_only_steps=0,
        kl_coef=0.0,
        offload_train=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_full_mode_keeps_existing_defaults():
    args = _base_args()
    _apply_critic_args(args)
    assert args.use_critic
    assert args.critic_num_gpus_per_node == 4
    assert args.critic_num_nodes == 1
    assert args.critic_load == "/ckpt/base"
    assert args.critic_lr == 1e-6
    assert uses_separate_critic(args)
    assert not uses_adapter_critic(args)


def test_grpo_disables_critic_entirely():
    args = _base_args(advantage_estimator="grpo")
    _apply_critic_args(args)
    assert not args.use_critic
    assert not uses_separate_critic(args)
    assert not uses_adapter_critic(args)


def test_adapter_mode_zeroes_critic_gpus_and_skips_load_default():
    args = _base_args(critic_mode="adapter")
    _apply_critic_args(args)
    assert args.use_critic
    assert args.critic_num_gpus_per_node == 0
    assert args.critic_num_nodes == 0
    assert args.critic_load is None
    assert args.critic_lr == 1e-6
    assert uses_adapter_critic(args)
    assert not uses_separate_critic(args)


def test_adapter_mode_preserves_explicit_critic_load_root():
    args = _base_args(critic_mode="adapter", critic_load="/ckpt/critic-input")
    _apply_critic_args(args)
    assert args.critic_load == "/ckpt/critic-input"


def test_adapter_mode_rejects_explicit_critic_gpus():
    args = _base_args(critic_mode="adapter", critic_num_gpus_per_node=2)
    with pytest.raises(ValueError, match="critic-num-gpus-per-node"):
        _apply_critic_args(args)


def test_adapter_mode_requires_ppo():
    args = _base_args(critic_mode="adapter", advantage_estimator="grpo")
    with pytest.raises(ValueError, match="advantage-estimator ppo"):
        _apply_critic_args(args)


def test_adapter_mode_requires_peft():
    args = _base_args(critic_mode="adapter", peft_method="none")
    with pytest.raises(ValueError, match="peft"):
        _apply_critic_args(args)


def test_adapter_mode_requires_megatron_backend():
    args = _base_args(critic_mode="adapter", train_backend="fsdp")
    with pytest.raises(ValueError, match="megatron"):
        _apply_critic_args(args)


def test_adapter_mode_rejects_keep_old_actor():
    args = _base_args(critic_mode="adapter", keep_old_actor=True)
    with pytest.raises(ValueError, match="keep-old-actor"):
        _apply_critic_args(args)


def test_adapter_mode_rejects_routing_replay():
    args = _base_args(critic_mode="adapter", use_rollout_routing_replay=True)
    with pytest.raises(ValueError, match="routing-replay"):
        _apply_critic_args(args)


def test_separate_critic_requires_equal_worker_counts():
    args = _base_args(critic_num_gpus_per_node=2, critic_num_nodes=1)
    _apply_critic_args(args)
    with pytest.raises(ValueError, match="equal actor and critic worker counts"):
        _validate_ppo_args(args)


def test_separate_critic_accepts_equal_total_worker_counts():
    args = _base_args(
        actor_num_gpus_per_node=2,
        actor_num_nodes=2,
        critic_num_gpus_per_node=4,
        critic_num_nodes=1,
    )
    _apply_critic_args(args)
    _validate_ppo_args(args)


def test_critic_only_warmup_rejects_reward_level_kl():
    args = _base_args(num_critic_only_steps=1, kl_coef=0.1)
    _apply_critic_args(args)
    with pytest.raises(ValueError, match="critic-only rollouts"):
        _validate_ppo_args(args)


def test_critic_only_warmup_rejects_negative_steps():
    args = _base_args(num_critic_only_steps=-1)
    _apply_critic_args(args)
    with pytest.raises(ValueError, match="must be nonnegative"):
        _validate_ppo_args(args)


def test_critic_only_warmup_allows_zero_reward_level_kl():
    args = _base_args(num_critic_only_steps=1, kl_coef=0.0)
    _apply_critic_args(args)
    _validate_ppo_args(args)


# --- head mode: value-head-only critic on a detached (read-only aliased) trunk ---

def _head_args(**overrides):
    return _base_args(critic_mode="head", **overrides)


def test_head_mode_allows_full_ft_actor():
    from orbit.utils.arguments import uses_head_critic, uses_one_trunk_critic

    args = _head_args(peft_method="none")
    _apply_critic_args(args)
    assert args.use_critic
    assert args.critic_num_gpus_per_node == 0
    assert args.critic_num_nodes == 0
    assert args.critic_lr == args.lr
    assert uses_head_critic(args)
    assert uses_one_trunk_critic(args)
    assert not uses_adapter_critic(args)
    assert not uses_separate_critic(args)


def test_head_mode_allows_peft_actor():
    from orbit.utils.arguments import uses_head_critic

    args = _head_args(peft_method="oft")
    _apply_critic_args(args)
    assert args.use_critic
    assert uses_head_critic(args)


def test_head_mode_requires_ppo():
    args = _head_args(advantage_estimator="grpo")
    with pytest.raises(ValueError, match="requires --advantage-estimator ppo"):
        _apply_critic_args(args)


def test_head_mode_rejects_keep_old_actor():
    args = _head_args(keep_old_actor=True)
    with pytest.raises(ValueError, match="keep-old-actor"):
        _apply_critic_args(args)


def test_head_mode_rejects_critic_gpu_request():
    args = _head_args(critic_num_gpus_per_node=1)
    with pytest.raises(ValueError, match="critic-num-gpus-per-node"):
        _apply_critic_args(args)


def test_adapter_mode_is_one_trunk_too():
    from orbit.utils.arguments import uses_one_trunk_critic

    args = _base_args(critic_mode="adapter")
    _apply_critic_args(args)
    assert uses_one_trunk_critic(args)
