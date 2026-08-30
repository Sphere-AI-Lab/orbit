from argparse import Namespace
from contextlib import contextmanager
from types import MethodType, SimpleNamespace

import pytest

from miles.backends.megatron_utils import actor as actor_utils


@contextmanager
def _null_timer(*args, **kwargs):
    yield


def test_direct_loss_restores_full_ft_actor_after_reference_forward(monkeypatch) -> None:
    """opd_jsd_loss skips advantages, but its optional ref KL must not train the ref."""
    monkeypatch.setattr(actor_utils, "all_replay_managers", [])
    monkeypatch.setattr(actor_utils, "get_data_iterator", lambda *args: ([], []))
    monkeypatch.setattr(actor_utils, "inverse_timer", _null_timer)
    monkeypatch.setattr(actor_utils, "timer", _null_timer)
    monkeypatch.setattr(actor_utils, "log_rollout_data", lambda *args: None)
    monkeypatch.setattr(actor_utils, "uses_one_trunk_critic", lambda args: False)
    monkeypatch.setattr(actor_utils, "uses_separate_critic", lambda args: False)
    monkeypatch.setattr(actor_utils, "should_backup_actor_after_train", lambda args: False)
    # upstream passes extra_metrics= to log_perf_data.
    monkeypatch.setattr(actor_utils, "log_perf_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(actor_utils.train_dump_utils, "save_debug_train_data", lambda *args, **kwargs: None)

    actor = object.__new__(actor_utils.MegatronTrainRayActor)
    actor.args = Namespace(
        compute_advantages_and_returns=False,
        # upstream's train_actor reads skip_actor_forward_only up front and sizes the
        # per-step rollout counts with get_num_rollouts(args, ...).
        skip_actor_forward_only=False,
        global_batch_size=1,
        num_critic_only_steps=0,
        ref_update_interval=None,
    )
    actor.model = object()
    actor.optimizer = object()
    actor.opt_param_scheduler = object()
    actor.rollout_data_postprocess = None
    # upstream's train_actor threads an FT test-action executor into train() and pops
    # weight-sync metrics off the weight_updater before bumping the FT heartbeat.
    actor._ft_test_action_executor = None
    actor.weight_updater = SimpleNamespace(pop_metrics=lambda: {})
    actor._heartbeat = SimpleNamespace(bump=lambda: None)
    actor._active_model_tag = "actor"
    actor._self_teacher = None
    actor.prof = SimpleNamespace(step=lambda **kwargs: None)
    events = []

    def _compute_ref(self, data_iterator, num_microbatches, rollout_id=0):
        self._active_model_tag = "ref"
        events.append("ref_forward")
        return {"ref_log_probs": []}

    def _switch(self, target_tag):
        events.append(f"switch:{target_tag}")
        self._active_model_tag = target_tag

    def _train(*args, **kwargs):
        events.append("train")
        assert actor._active_model_tag == "actor"

    actor.compute_ref_log_probs = MethodType(_compute_ref, actor)
    actor._switch_model = MethodType(_switch, actor)
    monkeypatch.setattr(actor_utils, "train", _train)

    actor.train_actor(rollout_id=0, rollout_data={}, witness_info=None, attempt=0)

    assert events == ["ref_forward", "switch:actor", "train"]


def test_critic_only_warmup_does_not_advance_self_teacher(monkeypatch) -> None:
    monkeypatch.setattr(actor_utils, "all_replay_managers", [])
    monkeypatch.setattr(actor_utils, "get_data_iterator", lambda *args: ([], []))
    monkeypatch.setattr(actor_utils, "inverse_timer", _null_timer)
    monkeypatch.setattr(actor_utils, "timer", _null_timer)
    monkeypatch.setattr(actor_utils, "log_rollout_data", lambda *args: None)
    monkeypatch.setattr(actor_utils, "uses_one_trunk_critic", lambda args: True)
    monkeypatch.setattr(actor_utils, "uses_separate_critic", lambda args: False)
    monkeypatch.setattr(actor_utils, "should_backup_actor_after_train", lambda args: False)
    # upstream passes extra_metrics= to log_perf_data.
    monkeypatch.setattr(actor_utils, "log_perf_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(actor_utils.train_dump_utils, "save_debug_train_data", lambda *args, **kwargs: None)

    updates = []
    actor = object.__new__(actor_utils.MegatronTrainRayActor)
    actor.args = Namespace(
        compute_advantages_and_returns=False,
        # upstream's train_actor reads skip_actor_forward_only up front and sizes the
        # per-step rollout counts with get_num_rollouts(args, ...).
        skip_actor_forward_only=False,
        global_batch_size=1,
        num_critic_only_steps=1,
        ref_update_interval=None,
    )
    actor.model = object()
    actor.optimizer = object()
    actor.opt_param_scheduler = object()
    actor.rollout_data_postprocess = None
    # upstream's train_actor threads an FT test-action executor into train() and pops
    # weight-sync metrics off the weight_updater before bumping the FT heartbeat.
    actor._ft_test_action_executor = None
    actor.weight_updater = SimpleNamespace(pop_metrics=lambda: {})
    actor._heartbeat = SimpleNamespace(bump=lambda: None)
    actor._active_model_tag = "actor"
    actor._self_teacher = SimpleNamespace(update=lambda params: updates.append(params))
    actor.compute_ref_log_probs = MethodType(lambda self, *args, **kwargs: None, actor)
    actor.prof = SimpleNamespace(step=lambda **kwargs: None)
    monkeypatch.setattr(actor_utils, "train", lambda *args, **kwargs: pytest.fail("actor train must be skipped"))

    actor.train_actor(rollout_id=0, rollout_data={}, witness_info=None, attempt=0)

    assert updates == []


def test_first_actor_step_after_critic_warmup_uses_actor_relative_promotion_cadence(monkeypatch) -> None:
    monkeypatch.setattr(actor_utils, "all_replay_managers", [])
    monkeypatch.setattr(actor_utils, "get_data_iterator", lambda *args: ([], []))
    monkeypatch.setattr(actor_utils, "inverse_timer", _null_timer)
    monkeypatch.setattr(actor_utils, "timer", _null_timer)
    monkeypatch.setattr(actor_utils, "log_rollout_data", lambda *args: None)
    monkeypatch.setattr(actor_utils, "uses_one_trunk_critic", lambda args: True)
    monkeypatch.setattr(actor_utils, "uses_separate_critic", lambda args: False)
    monkeypatch.setattr(actor_utils, "should_backup_actor_after_train", lambda args: False)
    # upstream passes extra_metrics= to log_perf_data.
    monkeypatch.setattr(actor_utils, "log_perf_data", lambda *args, **kwargs: None)
    monkeypatch.setattr(actor_utils.train_dump_utils, "save_debug_train_data", lambda *args, **kwargs: None)

    actor = object.__new__(actor_utils.MegatronTrainRayActor)
    actor.args = Namespace(
        compute_advantages_and_returns=False,
        # upstream's train_actor reads skip_actor_forward_only up front and sizes the
        # per-step rollout counts with get_num_rollouts(args, ...).
        skip_actor_forward_only=False,
        global_batch_size=1,
        num_critic_only_steps=2,
        ref_update_interval=None,
        opd_promote_interval=100,
    )
    actor.model = object()
    actor.optimizer = object()
    actor.opt_param_scheduler = object()
    actor.rollout_data_postprocess = None
    # upstream's train_actor threads an FT test-action executor into train() and pops
    # weight-sync metrics off the weight_updater before bumping the FT heartbeat.
    actor._ft_test_action_executor = None
    actor.weight_updater = SimpleNamespace(pop_metrics=lambda: {})
    actor._heartbeat = SimpleNamespace(bump=lambda: None)
    actor._active_model_tag = "actor"
    updates = []
    promotions = []
    actor._self_teacher = SimpleNamespace(update=lambda params: updates.append(params))
    actor._opd_teacher_spec = SimpleNamespace(source="self_ema")
    actor.compute_ref_log_probs = MethodType(lambda self, *args, **kwargs: None, actor)
    actor._adapter_named_params = MethodType(lambda self: {"adapter": object()}, actor)
    actor._promote_self_teacher = MethodType(lambda self: promotions.append(True), actor)
    actor.prof = SimpleNamespace(step=lambda **kwargs: None)
    monkeypatch.setattr(actor_utils, "train", lambda *args, **kwargs: None)

    actor.train_actor(rollout_id=2, rollout_data={}, witness_info=None, attempt=0)

    assert len(updates) == 1
    assert promotions == [True]
