"""orbit/megatron/megatron_utils_patches.py: the additions still happen, and
upstream's own path still runs UPSTREAM's body.

The second half is the point. Both patches delegate, so every test here asserts
against `_orbit_unpatched_*` as well: if someone ever "fixes" one of these by
copying the vendored body into orbit, the delegation assertions fail and
upstream's future changes to those functions stop reaching us silently.
"""

from argparse import Namespace

import pytest

torch = pytest.importorskip("torch")

import orbit  # noqa: F401  -- importing orbit installs the patches
from miles.backends.megatron_utils import parallel
from miles.backends.megatron_utils.update_weight.update_weight_from_distributed import (
    broadcast,
)
from orbit.megatron.sync_metrics import get_payload_tracker


# ---------------------------------------------------------------------------
# parallel.get_packed_seq_params
# ---------------------------------------------------------------------------


def _thd_batch(**extra):
    batch = {
        "cu_seqlens": torch.tensor([0, 4], dtype=torch.int32),
        "max_seqlen": 4,
    }
    batch.update(extra)
    return batch


def test_the_patch_is_actually_installed():
    assert parallel.get_packed_seq_params.__module__ == "orbit.megatron.megatron_utils_patches"
    assert hasattr(parallel, "_orbit_unpatched_get_packed_seq_params"), (
        "the pristine upstream function must be kept so the patch can delegate"
    )


def test_dsv4_cu_seqlens_are_carried_on_the_packed_seq_params():
    cu = torch.tensor([0, 2, 4], dtype=torch.int32)
    valid = torch.tensor([0, 1, 4], dtype=torch.int32)
    batch = _thd_batch(dsv4_cu_seqlens=cu, dsv4_valid_cu_seqlens=valid)

    params = parallel.get_packed_seq_params(batch, Namespace(qkv_format="thd"))

    assert params.dsv4_cu_seqlens is cu
    assert params.dsv4_valid_cu_seqlens is valid
    # ...and prove the patch is what did it: upstream alone cannot.
    upstream = parallel._orbit_unpatched_get_packed_seq_params(
        _thd_batch(dsv4_cu_seqlens=cu), Namespace(qkv_format="thd")
    )
    assert not hasattr(upstream, "dsv4_cu_seqlens")


def test_the_batch_entry_is_the_same_object_the_caller_gets():
    """The vendored edit attached the fields BEFORE storing into the batch. Only
    one PackedSeqParams is ever built, so attaching afterwards must be visible
    through the batch too -- otherwise the model, which reads the batch entry,
    would see a bare object."""
    batch = _thd_batch(dsv4_cu_seqlens=torch.tensor([0, 2], dtype=torch.int32))

    params = parallel.get_packed_seq_params(batch, Namespace(qkv_format="thd"))

    assert batch["packed_seq_params"] is params
    assert batch["packed_seq_params"].dsv4_cu_seqlens is batch["dsv4_cu_seqlens"]


def test_a_batch_without_dsv4_keys_is_exactly_upstreams_result():
    """The delegation property: no dsv4 fields in, upstream's object out."""
    batch = _thd_batch()
    args = Namespace(qkv_format="thd")

    params = parallel.get_packed_seq_params(batch, args)
    upstream = parallel._orbit_unpatched_get_packed_seq_params(_thd_batch(), args)

    assert params.qkv_format == upstream.qkv_format == "thd"
    assert params.max_seqlen_q == upstream.max_seqlen_q
    assert torch.equal(params.cu_seqlens_q, upstream.cu_seqlens_q)
    assert not hasattr(params, "dsv4_cu_seqlens")


def test_non_thd_still_returns_upstreams_none():
    batch = _thd_batch(dsv4_cu_seqlens=torch.tensor([0, 2], dtype=torch.int32))
    args = Namespace(qkv_format="bshd")

    assert parallel.get_packed_seq_params(batch, args) is None
    assert parallel._orbit_unpatched_get_packed_seq_params(batch, args) is None


# ---------------------------------------------------------------------------
# broadcast.update_weights_from_distributed
# ---------------------------------------------------------------------------


class _FakeEngine:
    def __init__(self, calls):
        self.update_weights_from_distributed = _Remote(calls)


class _Remote:
    def __init__(self, calls):
        self._calls = calls

    def remote(self, **kwargs):
        self._calls.append(kwargs)
        return f"ref{len(self._calls)}"


@pytest.fixture
def _fake_broadcast(monkeypatch):
    """Replace only the NCCL call; everything else is upstream's real body."""
    done = []

    class _Handle:
        def wait(self):
            done.append("wait")

    def fake_broadcast(tensor, src, group=None, async_op=False):
        done.append(("broadcast", src, async_op))
        return _Handle()

    monkeypatch.setattr(broadcast.dist, "broadcast", fake_broadcast)
    get_payload_tracker().reset()
    yield done
    get_payload_tracker().reset()


def test_the_broadcast_patch_is_actually_installed():
    assert (
        broadcast.update_weights_from_distributed.__module__
        == "orbit.megatron.megatron_utils_patches"
    )
    assert hasattr(broadcast, "_orbit_unpatched_update_weights_from_distributed")


def test_payload_bytes_are_recorded_once_per_update(_fake_broadcast):
    calls = []
    tensors = [
        ("w1", torch.zeros(4, 4, dtype=torch.float32)),
        ("w2", torch.zeros(8, dtype=torch.float32)),
    ]

    refs = broadcast.update_weights_from_distributed(
        "grp", None, 3, [_FakeEngine(calls), _FakeEngine(calls)], tensors
    )

    tracker = get_payload_tracker()
    assert tracker.payload_bytes == (4 * 4 + 8) * 4
    assert tracker.num_tensors == 2
    # Two engines, one broadcast: the payload is counted once, not per engine.
    assert tracker.num_records == 1
    assert len(refs) == 2


def test_upstream_still_does_the_dispatch_and_the_broadcast(_fake_broadcast):
    """The delegation property: metadata refs and the NCCL calls come out of
    upstream's body, not a copy of it in orbit."""
    calls = []
    tensors = [("w1", torch.zeros(2, 2, dtype=torch.float32))]

    refs = broadcast.update_weights_from_distributed(
        "grp", None, 7, [_FakeEngine(calls)], tensors
    )

    assert refs == ["ref1"]
    assert calls[0]["names"] == ["w1"]
    assert calls[0]["group_name"] == "grp"
    assert calls[0]["weight_version"] == "7"
    assert _fake_broadcast == [("broadcast", 0, True), "wait"]


def test_upstreams_own_path_records_nothing(_fake_broadcast):
    """Guards the assertion above: if the tracker call had been left inside the
    vendored body, this would count too."""
    broadcast._orbit_unpatched_update_weights_from_distributed(
        "grp", None, 1, [_FakeEngine([])], [("w", torch.zeros(4, dtype=torch.float32))]
    )
    assert get_payload_tracker().num_records == 0
