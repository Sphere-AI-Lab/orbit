import pytest
import torch

from orbit.critic.critic_adapter import (
    alias_trunk_storage,
    assert_trunk_aliased,
)

HIDDEN = 4


class _ActorChunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.trunk.weight.requires_grad_(False)
        self.output_layer = torch.nn.Linear(HIDDEN, 8, bias=False)  # frozen LM head
        self.output_layer.weight.requires_grad_(False)
        self.adapter = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)  # trainable


class _CriticChunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.trunk.weight.requires_grad_(False)
        self.output_layer = torch.nn.Linear(HIDDEN, 1, bias=False)  # trainable value head
        self.adapter = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)  # trainable


def _models():
    return [_CriticChunk()], [_ActorChunk()]


def test_alias_points_frozen_params_at_actor_storage():
    critic, actor = _models()
    count = alias_trunk_storage(critic, actor)
    assert count == 1  # trunk.weight only: value head + adapter are trainable
    assert critic[0].trunk.weight.data_ptr() == actor[0].trunk.weight.data_ptr()
    # trainable value head is role-owned despite the name collision with the frozen LM head
    assert critic[0].output_layer.weight.data_ptr() != actor[0].output_layer.weight.data_ptr()


def test_alias_shares_mutations():
    critic, actor = _models()
    alias_trunk_storage(critic, actor)
    with torch.no_grad():
        actor[0].trunk.weight.fill_(3.0)
    assert torch.equal(critic[0].trunk.weight, actor[0].trunk.weight)


def test_assert_trunk_aliased_detects_broken_alias():
    critic, actor = _models()
    alias_trunk_storage(critic, actor)
    assert_trunk_aliased(critic, actor)  # passes
    critic[0].trunk.weight.data = critic[0].trunk.weight.data.clone()
    with pytest.raises(RuntimeError, match="trunk alias"):
        assert_trunk_aliased(critic, actor)


def test_alias_rejects_shape_mismatch():
    critic, actor = _models()
    critic[0].trunk = torch.nn.Linear(HIDDEN, HIDDEN + 1, bias=False)
    critic[0].trunk.weight.requires_grad_(False)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        alias_trunk_storage(critic, actor)


def test_alias_rejects_missing_actor_param():
    critic, actor = _models()
    critic[0].extra = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
    critic[0].extra.weight.requires_grad_(False)
    with pytest.raises(RuntimeError, match="missing from actor"):
        alias_trunk_storage(critic, actor)


def test_alias_rejects_fully_trainable_critic():
    critic, actor = _models()
    critic[0].trunk.weight.requires_grad_(True)
    with pytest.raises(RuntimeError, match="no frozen"):
        alias_trunk_storage(critic, actor)


def test_gradient_isolation_across_roles():
    critic, actor = _models()
    alias_trunk_storage(critic, actor)
    x = torch.randn(2, HIDDEN)

    actor_loss = actor[0].output_layer(actor[0].adapter(actor[0].trunk(x))).sum()
    actor_loss.backward()
    assert actor[0].adapter.weight.grad is not None
    assert critic[0].adapter.weight.grad is None
    assert critic[0].output_layer.weight.grad is None
    assert actor[0].trunk.weight.grad is None  # frozen shared trunk gets no grads

    critic_loss = critic[0].output_layer(critic[0].adapter(critic[0].trunk(x))).sum()
    critic_loss.backward()
    assert critic[0].adapter.weight.grad is not None
    assert critic[0].output_layer.weight.grad is not None
    assert critic[0].trunk.weight.grad is None


# --- head mode: freeze-all-but-value-head, alias against a FULL-FT (trainable) actor ---

class _FullFTActorChunk(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)   # trainable: full FT
        self.output_layer = torch.nn.Linear(HIDDEN, 8, bias=False)  # trainable LM head


class _PlainCriticChunk(torch.nn.Module):
    """What the plain (non-PEFT) builder produces: everything trainable."""

    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.output_layer = torch.nn.Linear(HIDDEN, 1, bias=False)  # value head


def test_prepare_head_critic_freezes_everything_but_value_head():
    from orbit.critic.critic_adapter import prepare_head_critic

    critic = [_PlainCriticChunk()]
    frozen = prepare_head_critic(critic)
    assert frozen == 1  # trunk.weight
    assert not critic[0].trunk.weight.requires_grad
    assert critic[0].output_layer.weight.requires_grad


def test_head_critic_aliases_full_ft_actor_trunk():
    from orbit.critic.critic_adapter import prepare_head_critic

    critic, actor = [_PlainCriticChunk()], [_FullFTActorChunk()]
    prepare_head_critic(critic)
    count = alias_trunk_storage(critic, actor)
    assert count == 1
    assert critic[0].trunk.weight.data_ptr() == actor[0].trunk.weight.data_ptr()
    assert critic[0].output_layer.weight.data_ptr() != actor[0].output_layer.weight.data_ptr()
    assert_trunk_aliased(critic, actor)


def test_head_critic_value_backward_leaves_actor_trunk_gradless():
    """The safety property of the detached-trunk design: a value-loss backward
    through the critic view produces NO gradient for the shared trunk storage,
    even though the actor's Parameter over that storage is trainable."""
    from orbit.critic.critic_adapter import prepare_head_critic

    critic, actor = [_PlainCriticChunk()], [_FullFTActorChunk()]
    prepare_head_critic(critic)
    alias_trunk_storage(critic, actor)

    x = torch.randn(3, HIDDEN)
    value = critic[0].output_layer(critic[0].trunk(x))
    value.pow(2).sum().backward()

    assert critic[0].output_layer.weight.grad is not None   # head learns
    assert critic[0].trunk.weight.grad is None              # critic view frozen
    assert actor[0].trunk.weight.grad is None               # actor untouched
