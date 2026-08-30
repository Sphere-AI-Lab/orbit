"""--offload-rollout is inert in the async (disjoint-GPU) topology.

Both async launchers pass --offload-rollout, yet train_async.py never calls
onload_weights/onload_kv (unlike train.py's offload/onload dance). That is
deliberate, not a missing-onload bug: with actor and rollout GPUs disjoint,
``start_rollout_servers`` computes ``needs_offload=False`` for every rollout
ServerGroup (``group_abs_start >= megatron_num_gpus``), so the initial
``rollout_manager.offload()`` issued by ``create_rollout_manager`` releases
nothing and the engines simply stay resident. These tests pin each link of
that chain so a refactor cannot silently turn the no-op into a real offload
that async drivers never undo.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("ray")

import miles.ray.placement_group as pg_mod
from miles.ray.rollout.rollout_server import (
    RolloutServer,
    _compute_megatron_num_gpus,
    _compute_rollout_offset,
)
from miles.ray.rollout.server_group import ServerGroup


def _topology_args(**overrides) -> SimpleNamespace:
    defaults = dict(
        colocate=False,
        debug_train_only=False,
        debug_rollout_only=False,
        critic_train_only=False,
        use_critic=False,
        actor_num_nodes=1,
        actor_num_gpus_per_node=4,
        critic_num_nodes=1,
        critic_num_gpus_per_node=2,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.parametrize(
    "overrides",
    [
        {},  # plain async: actor-only training GPUs
        {"use_critic": True},  # async PPO with separate critic GPUs
        {"critic_train_only": True},  # critic-only warmup topology
        {"debug_rollout_only": True},  # no training GPUs at all
    ],
    ids=["actor_only", "actor_plus_critic", "critic_train_only", "debug_rollout_only"],
)
def test_disjoint_topology_puts_all_rollout_groups_past_megatron_gpus(overrides):
    """In every non-colocate topology the rollout PG offset starts at (or past)
    the last megatron GPU slot, so ``group_abs_start = offset + gpu_offset``
    with ``gpu_offset >= 0`` can never satisfy ``group_abs_start <
    megatron_num_gpus`` -- the needs_offload gate in start_rollout_servers is
    False for every group."""
    args = _topology_args(**overrides)
    assert _compute_rollout_offset(args) >= _compute_megatron_num_gpus(args)


def test_colocate_topology_keeps_the_offload_gate_live():
    """Contrast case: under --colocate the rollout groups start at offset 0,
    below megatron_num_gpus, so needs_offload CAN be True there (train.py's
    onload dance is required). Guards against 'simplifying' the gate away as
    always-false."""
    args = _topology_args(colocate=True)
    assert _compute_rollout_offset(args) < _compute_megatron_num_gpus(args)


def _server_group(needs_offload: bool, engines: list) -> ServerGroup:
    return ServerGroup(
        args=SimpleNamespace(num_gpus_per_node=8, debug_train_only=False, rollout_external=False),
        pg=None,
        all_engines=engines,
        num_gpus_per_engine=1,
        num_new_engines=0,
        needs_offload=needs_offload,
        model_path="/ckpt/base",
    )


def test_server_group_offload_onload_noop_without_needs_offload():
    """With needs_offload=False the group must not touch its engines at all.

    The engine stub has no release/resume attributes, so any attempt to issue
    the RPC raises AttributeError instead of silently passing.
    """
    booby_trapped_engine = object()
    group = _server_group(needs_offload=False, engines=[booby_trapped_engine])
    assert group.offload() == []
    assert group.onload() == []
    assert group.onload(tags=["weights"]) == []
    assert group.onload_weights_from_disk() == []


def _recording_engine(calls: list):
    return SimpleNamespace(
        release_memory_occupation=SimpleNamespace(remote=lambda: calls.append("release") or "release-handle"),
        resume_memory_occupation=SimpleNamespace(
            remote=lambda tags=None: calls.append(("resume", tuple(tags or ()))) or "resume-handle"
        ),
    )


def test_server_group_issues_rpcs_only_when_needs_offload():
    """The protection is the gate, not dead code: with needs_offload=True the
    same methods do issue the release/resume RPCs."""
    calls: list = []
    group = _server_group(needs_offload=True, engines=[_recording_engine(calls)])
    assert group.offload() == ["release-handle"]
    assert group.onload(tags=["weights"]) == ["resume-handle"]
    assert calls == ["release", ("resume", ("weights",))]


def test_rollout_server_offload_onload_paths_all_noop_for_async_groups():
    """RolloutServer aggregates gated groups; with every group at
    needs_offload=False all four memory paths return [] without ray."""
    server = RolloutServer(server_groups=[_server_group(False, [object()]), _server_group(False, [object()])])
    assert server.offload() == []
    assert server.onload() == []
    assert server.onload_weights() == []
    assert server.onload_kv() == []


class _RecordingManagerHandle:
    """Stands in for the RolloutManager ray actor handle."""

    def __init__(self):
        self.offload_calls: list[tuple[tuple, dict]] = []
        self.offload = SimpleNamespace(
            remote=lambda *a, **kw: self.offload_calls.append((a, kw)) or "offload-handle"
        )


def test_create_rollout_manager_initial_offload_takes_the_gated_no_tags_path(monkeypatch):
    """The startup offload in create_rollout_manager must call offload() with
    NO tags: RolloutManager.offload(tags=None) routes through
    ServerGroup.needs_offload (no-op in async), while the tags=... fast path
    bypasses that gate and would release memory the async driver never
    onloads."""
    handle = _RecordingManagerHandle()
    fake_manager_cls = SimpleNamespace(options=lambda **kw: SimpleNamespace(remote=lambda *a, **kw2: handle))
    monkeypatch.setattr(pg_mod, "RolloutManager", fake_manager_cls)
    monkeypatch.setattr(pg_mod, "ray", SimpleNamespace(get=lambda refs: refs))

    args = SimpleNamespace(
        offload_rollout=True,
        num_rollout=4,
        check_weight_update_equal=False,
        pin_rollout_manager_to_head=False,
        use_rollout_engines=True,
    )
    manager, num_rollout_per_epoch = pg_mod.create_rollout_manager(args, pg=None)

    assert manager is handle
    assert num_rollout_per_epoch is None
    assert handle.offload_calls == [((), {})]
