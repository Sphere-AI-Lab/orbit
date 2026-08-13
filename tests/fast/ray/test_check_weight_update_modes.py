"""Mocked call-order tests for the --check-weight-update-equal modes.

boot:         snapshot -> reset_tensors -> [initial update_weights] -> compare
after-update: [initial update_weights] -> snapshot -> reset_tensors -> second
              update_weights -> compare

The initial update_weights lives in the drivers (train.py / train_async.py);
_drive() reproduces that sequence around the two shared helpers in
miles.ray.placement_group and the tests pin the resulting checker call order
plus selector/skip_list propagation. Pure CPU, no Ray or SGLang engines.
"""

import asyncio
from types import SimpleNamespace
from unittest import mock

import pytest

import miles.ray.placement_group as placement_group


class _ImmediateResult:
    """Awaitable stand-in for a Ray ObjectRef that resolves to None."""

    def __await__(self):
        return iter(())


class _FakeRolloutManager:
    def __init__(self, calls):
        self._calls = calls
        self.check_weights = SimpleNamespace(remote=self._check_weights_remote)

    def _check_weights_remote(self, action, **kwargs):
        self._calls.append((action, kwargs))
        return _ImmediateResult()


class _FakeActorModel:
    def __init__(self, calls):
        self._calls = calls

    async def update_weights(self):
        self._calls.append(("update_weights", {}))


def _make_args(mode, equal=True, selector="all", skip_list=None):
    return SimpleNamespace(
        check_weight_update_equal=equal,
        check_weight_update_equal_mode=mode,
        check_weight_update_selector=selector,
        check_weight_update_skip_list=skip_list,
        check_weight_update_allow_quant_error=True,
    )


def _drive(args):
    """Reproduce the driver sequence: boot-time hook (create_rollout_manager),
    then the initial update_weights, then the post-sync check helper."""
    calls = []
    rollout_manager = _FakeRolloutManager(calls)
    actor_model = _FakeActorModel(calls)

    with mock.patch.object(placement_group, "ray", SimpleNamespace(get=lambda ref: None)):
        placement_group.check_weight_update_equal_boot_snapshot(args, rollout_manager)

    async def _driver():
        await actor_model.update_weights()
        if args.check_weight_update_equal:
            await placement_group.check_weight_update_equal_after_initial_sync(args, actor_model, rollout_manager)

    asyncio.run(_driver())
    return calls


def test_boot_mode_call_order():
    calls = _drive(_make_args("boot"))
    assert [action for action, _ in calls] == ["snapshot", "reset_tensors", "update_weights", "compare"]


def test_after_update_mode_call_order():
    calls = _drive(_make_args("after-update"))
    assert [action for action, _ in calls] == [
        "update_weights",
        "snapshot",
        "reset_tensors",
        "update_weights",
        "compare",
    ]


def test_disabled_check_only_updates():
    calls = _drive(_make_args("boot", equal=False))
    assert [action for action, _ in calls] == ["update_weights"]


@pytest.mark.parametrize("mode", ["boot", "after-update"])
def test_selector_and_skip_list_propagation(mode):
    skip_list = ["mtp.", "draft_model."]
    calls = _drive(_make_args(mode, selector="target", skip_list=skip_list))

    checker_calls = [(action, kwargs) for action, kwargs in calls if action != "update_weights"]
    assert {action for action, _ in checker_calls} == {"snapshot", "reset_tensors", "compare"}
    for action, kwargs in checker_calls:
        assert kwargs["selector"] == "target", f"{action} lost the selector"
        if action in ("reset_tensors", "compare"):
            assert kwargs["skip_list"] == skip_list, f"{action} lost the skip_list"

    (compare_kwargs,) = [kwargs for action, kwargs in checker_calls if action == "compare"]
    assert compare_kwargs["allow_quant_error"] is True
