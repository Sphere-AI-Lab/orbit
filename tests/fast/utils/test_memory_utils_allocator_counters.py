"""`reserved - allocated` is not one number, and the difference decides the fix.

On 2026-08-05 the three LoRA arms of E4 gsm8k column 4 died at rollout 2 on
8xH100 in `torch_memory_saver ... func=resume`, while the FullFT arm of the same
column completed 149/149 rollouts on the same node. At the failure rank 0 held
`allocated 0.12 GB` against `reserved 50.01 GB` -- 49.9 GB free in PyTorch's
eyes and unavailable to SGLang's `cuMemCreate` all the same.

`offload_megatron_frozen_base_to_cpu` already calls `gc.collect()` then
`torch.cuda.empty_cache()` every rollout, and the 50.01 GB survives it. Two
things explain that equally well and imply different fixes: the segments are
partially occupied and therefore non-releasable (fragmentation ->
`expandable_segments:True`), or they are fully free and were skipped because
their blocks still carry recorded stream uses (-> a synchronising clear).

`inactive_split_bytes` is exactly the first quantity. Torch's own memory summary
labels it "Non-releasable memory": bytes that are free but sit inside a segment
still holding a live block. Reading it costs a host-side counter lookup on a log
line already being emitted, and it tells the two hypotheses apart.
"""

from __future__ import annotations

import pytest
import torch

from orbit.utils import memory_utils


def _patch_cuda(monkeypatch, stats):
    """A plausible 80 GB device, so only the stats dict varies between tests."""
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (13 * 1024**3, 80 * 1024**3))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 1024**3 // 8)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 50 * 1024**3)
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda device: stats)


def test_reports_the_non_releasable_bytes_empty_cache_cannot_return(monkeypatch):
    _patch_cuda(
        monkeypatch,
        {
            "inactive_split_bytes.all.current": 49 * 1024**3,
            "segment.all.current": 812,
            "num_alloc_retries": 17,
        },
    )

    info = memory_utils.available_memory()

    assert info["inactive_split_GB"] == 49.0
    assert info["segments"] == 812
    assert info["alloc_retries"] == 17


def test_the_existing_fields_are_not_disturbed(monkeypatch):
    _patch_cuda(monkeypatch, {})

    info = memory_utils.available_memory()

    assert info["total_GB"] == 80.0
    assert info["free_GB"] == 13.0
    assert info["used_GB"] == 67.0
    assert info["reserved_GB"] == 50.0


def test_survives_a_device_the_allocator_has_never_served(monkeypatch):
    """`print_memory` runs during setup, before the first allocation, and
    `memory_stats()` returns {} for such a device. A bare subscript would raise
    KeyError at startup, so every lookup must default."""
    _patch_cuda(monkeypatch, {})

    info = memory_utils.available_memory()

    assert info["inactive_split_GB"] == 0.0
    assert info["segments"] == 0
    assert info["alloc_retries"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real CUDA allocator")
def test_the_stat_keys_this_module_reads_exist_in_this_torch(monkeypatch):
    """A misspelled key would return the `.get` default of 0, and a near-zero
    reading is what the diagnostic interprets as evidence against fragmentation
    -- so a typo does not fail loudly, it points at the wrong fix. Record the
    keys `available_memory` actually queries and assert every one of them is
    real, so drift in the module fails here rather than in a campaign's log."""
    torch.zeros(1, device="cuda")  # force the allocator to serve this device
    real_stats = torch.cuda.memory_stats(torch.cuda.current_device())

    queried = []

    class _RecordingStats(dict):
        def get(self, key, default=None):
            queried.append(key)
            return super().get(key, default)

    monkeypatch.setattr(torch.cuda, "memory_stats", lambda device: _RecordingStats(real_stats))

    memory_utils.available_memory()

    assert queried, "available_memory() queried no allocator stats at all"
    missing = [key for key in queried if key not in real_stats]
    assert not missing, f"missing from torch {torch.__version__} memory_stats: {missing}"
