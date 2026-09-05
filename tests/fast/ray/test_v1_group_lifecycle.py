import logging
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from orbit.ray.actor_group import RayTrainGroup

_GROUP_LOGGER = logging.getLogger("orbit.ray.actor_group")


class _RemoteCall:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _EventHandler(logging.Handler):
    def __init__(self, events: list):
        super().__init__()
        self._events = events

    def emit(self, record: logging.LogRecord) -> None:
        self._events.append(record.getMessage())


@contextmanager
def _record_group_logs(events: list):
    handler = _EventHandler(events)
    old_level = _GROUP_LOGGER.level
    _GROUP_LOGGER.setLevel(logging.INFO)
    _GROUP_LOGGER.addHandler(handler)
    try:
        yield
    finally:
        _GROUP_LOGGER.removeHandler(handler)
        _GROUP_LOGGER.setLevel(old_level)


def _make_group(events: list, *, debug_flag: str | None = None) -> RayTrainGroup:
    async def recover_updatable_engines():
        events.append("recover")

    async def get_updatable_engines_and_lock():
        events.append("lock")
        return {"engine": "ready"}

    async def health_monitoring_pause():
        events.append("pause")

    async def broadcast(method, **kwargs):
        events.append((method, kwargs))

    group = RayTrainGroup.__new__(RayTrainGroup)
    group.args = SimpleNamespace(
        debug_train_only=debug_flag == "debug_train_only",
        debug_rollout_only=debug_flag == "debug_rollout_only",
        use_fault_tolerance=True,
        # the synced update_weights gate checks membership before pausing rollout FT
        ft_components=["rollout"],
    )
    group.rollout_manager = SimpleNamespace(
        recover_updatable_engines=_RemoteCall(recover_updatable_engines),
        get_updatable_engines_and_lock=_RemoteCall(get_updatable_engines_and_lock),
        health_monitoring_pause=_RemoteCall(health_monitoring_pause),
    )
    group._broadcast = broadcast
    return group


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("rollout_id", "expected_record"),
    [
        (None, "ft op=update_weights phase=start rollout="),
        (2, "ft op=update_weights phase=start rollout=2"),
    ],
)
async def test_update_weights_records_lifecycle_before_side_effects(
    rollout_id: int | None, expected_record: str
) -> None:
    events: list = []
    group = _make_group(events)

    with _record_group_logs(events):
        await group.update_weights(rollout_id)

    assert events == [
        expected_record,
        "recover",
        "lock",
        "pause",
        ("update_weights", {"info": {"engine": "ready"}}),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("debug_flag", ["debug_train_only", "debug_rollout_only"])
async def test_debug_only_update_is_silent(debug_flag: str) -> None:
    events: list = []
    group = _make_group(events, debug_flag=debug_flag)

    with _record_group_logs(events):
        await group.update_weights(2)

    assert events == []


@pytest.mark.asyncio
async def test_prefetch_train_state_broadcasts_to_every_v1_actor() -> None:
    events: list = []
    group = _make_group(events)

    async def broadcast(method, *args, **kwargs):
        events.append((method, args, kwargs))

    group._broadcast = broadcast

    await group.prefetch_train_state(9)

    assert events == [("prefetch_train_state", (9,), {})]
