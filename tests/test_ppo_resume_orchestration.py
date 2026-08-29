import argparse
import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("ray")

import miles.backends.megatron_utils.actor as actor_mod
import miles.backends.megatron_utils.checkpoint as checkpoint_mod
from miles.ray.actor_group import RayTrainGroup
from miles.ray.placement_group import _single_start_rollout_id


def test_model_only_bridge_load_starts_at_rollout_zero():
    args = argparse.Namespace(_orbit_training_checkpoint_loaded=False)
    assert actor_mod._start_rollout_id_from_checkpoint(args, loaded_iteration=0) == 0


def test_training_checkpoint_starts_after_loaded_iteration():
    args = argparse.Namespace(_orbit_training_checkpoint_loaded=True)
    assert actor_mod._start_rollout_id_from_checkpoint(args, loaded_iteration=7) == 8


def test_training_checkpoint_at_iteration_zero_starts_at_rollout_one():
    args = argparse.Namespace(_orbit_training_checkpoint_loaded=True)
    assert actor_mod._start_rollout_id_from_checkpoint(args, loaded_iteration=0) == 1


def test_all_ranks_must_agree_on_start_rollout_id():
    assert _single_start_rollout_id("actor", [4, 4]) == 4
    with pytest.raises(RuntimeError, match="different rollout ids"):
        _single_start_rollout_id("critic", [4, 5])


def test_connect_rejects_unequal_group_sizes_before_pairing():
    actor_group = object.__new__(RayTrainGroup)
    critic_group = object.__new__(RayTrainGroup)
    actor_group._actor_handles = [object(), object()]
    critic_group._actor_handles = [object()]

    with pytest.raises(RuntimeError, match="equal worker counts"):
        asyncio.run(actor_group.connect(critic_group))


def _checkpoint_args(load_path):
    return argparse.Namespace(
        load=str(load_path),
        megatron_to_hf_mode="bridge",
        peft_method="lora",
        peft_adapter_path="/actor/adapter",
        lora_adapter_path=None,
        oft_adapter_path=None,
    )


def test_distributed_critic_resume_uses_full_megatron_loader(monkeypatch, tmp_path):
    (tmp_path / "payload").write_text("x")
    args = _checkpoint_args(tmp_path)
    calls = []

    monkeypatch.setattr(checkpoint_mod, "get_args", lambda: args)
    monkeypatch.setattr(checkpoint_mod, "validate_low_precision_bootstrap_args", lambda _args: None)
    monkeypatch.setattr(checkpoint_mod, "is_distributed_checkpoint", lambda _path: True)
    monkeypatch.setattr(checkpoint_mod, "_resolve_selected_distributed_checkpoint", lambda _args: tmp_path)
    monkeypatch.setattr(
        checkpoint_mod,
        "_select_megatron_training_checkpoint",
        lambda _args, expected_role, checkpoint_dir: checkpoint_dir,
    )
    monkeypatch.setattr(
        checkpoint_mod,
        "_load_selected_megatron_training_checkpoint",
        lambda *_args, **_kwargs: calls.append("megatron") or (6, 0),
    )
    monkeypatch.setattr(
        checkpoint_mod,
        "_load_checkpoint_dist",
        lambda **kwargs: calls.append("model-only") or (0, 0),
    )
    monkeypatch.setattr(checkpoint_mod, "is_peft_enabled", lambda _args: False)

    result = checkpoint_mod.load_checkpoint(
        [SimpleNamespace(role="critic")],
        object(),
        object(),
        checkpointing_context={},
        skip_load_to_model_and_opt=False,
        load_training_state=True,
    )

    assert result == (6, 0)
    assert calls == ["megatron"]
    assert args._orbit_training_checkpoint_loaded is True


def test_full_critic_does_not_load_actor_peft_adapter(monkeypatch, tmp_path):
    (tmp_path / "payload").write_text("x")
    args = _checkpoint_args(tmp_path)

    monkeypatch.setattr(checkpoint_mod, "get_args", lambda: args)
    monkeypatch.setattr(checkpoint_mod, "validate_low_precision_bootstrap_args", lambda _args: None)
    monkeypatch.setattr(checkpoint_mod, "is_distributed_checkpoint", lambda _path: False)
    monkeypatch.setattr(checkpoint_mod, "_is_megatron_checkpoint", lambda _path: False)
    monkeypatch.setattr(checkpoint_mod, "_load_checkpoint_hf", lambda **kwargs: (0, 0))
    monkeypatch.setattr(checkpoint_mod, "is_peft_enabled", lambda _args: True)
    monkeypatch.setattr(checkpoint_mod, "is_peft_model", lambda _model: False)
    monkeypatch.setattr(
        checkpoint_mod,
        "load_peft_adapter",
        lambda *args, **kwargs: pytest.fail("full critic attempted to load actor PEFT adapter"),
    )

    result = checkpoint_mod.load_checkpoint(
        [object()],
        object(),
        object(),
        checkpointing_context={},
        skip_load_to_model_and_opt=False,
    )

    assert result == (0, 0)
    assert args._orbit_training_checkpoint_loaded is False
