"""UpdateWeightFromDistributedBridge carries upstream's engine-connection contract.

miles dbbab1566's actor asks ``weight_updater.is_rollout_engines_fresh()`` before
every sync and calls ``mark_engine_connection_stale()`` on engine recovery; every
upstream distributed updater implements both over a ``_connection_stale`` flag.
orbit's bridge-fed updater (the full-FT disjoint-GPU path) predates that and died
on the second weight sync of every full-FT async run with AttributeError.
"""

from argparse import Namespace

import pytest

from orbit.megatron import update_weight_bridge


@pytest.fixture
def updater(monkeypatch):
    # __init__ builds the bridge weight iterator (needs a real model); stub it.
    monkeypatch.setattr(update_weight_bridge.HfWeightIteratorBase, "create", staticmethod(lambda **_: object()))
    # Non-source rank: connect_rollout_engines records the engines without opening NCCL groups.
    monkeypatch.setattr(update_weight_bridge.dist, "get_rank", lambda: 1)
    return update_weight_bridge.UpdateWeightFromDistributedBridge(
        Namespace(), [], lambda: {}, model_name="qwen3", quantization_config=None
    )


def test_not_fresh_before_any_connection(updater):
    assert updater.rollout_engines is None
    assert updater.is_rollout_engines_fresh() is False


def test_connect_makes_engines_fresh_and_stale_marker_invalidates(updater):
    engines, lock = [object(), object()], object()

    updater.connect_rollout_engines(engines, lock, engine_gpu_counts=[1, 1])
    assert updater.rollout_engines is engines and updater.rollout_engine_lock is lock
    assert updater.is_rollout_engines_fresh() is True

    updater.mark_engine_connection_stale()
    assert updater.is_rollout_engines_fresh() is False

    updater.connect_rollout_engines(engines, lock, engine_gpu_counts=[1, 1])
    assert updater.is_rollout_engines_fresh() is True
