"""Shared-memory refcount balancing for broadcast PEFT adapter payloads.

Home for ``_balance_broadcast_shm_refcounts``, lifted out of
miles/backends/sglang_utils/sglang_engine.py (Phase 3 isolation, slice 3c).
Used by ``miles.orbit.sglang.engine_ext.OrbitEngineExtensions.load_lora_adapter_from_ray_tensors``
and re-exported from the miles file for ``tests/test_peft_broadcast_shm_refcount.py``.
"""


def _balance_broadcast_shm_refcounts(tensors: dict, consumer_count: int) -> int:
    """Pre-pay the shm refcount for a payload that ``consumer_count`` processes rebuild.

    torch's ``file_system`` reduce/rebuild pair is a 1-producer -> 1-consumer
    handshake: ``reduce_storage`` calls ``storage._shared_incref()`` exactly
    once per serialization, and every ``rebuild_storage_filename`` ends in a
    matching ``_shared_decref()`` (torch/multiprocessing/reductions.py). SGLang
    hands ONE payload to EVERY TP scheduler and each deserializes it
    (tp_worker.py:218), so a tp_size=N engine decrefs N times against that
    single incref. The manager unlinks the segment N-1 releases early — while
    this actor still holds ``tensors`` — and the rank that opens last dies with
    ``unable to open shared memory object ... No such file or directory (2)``.
    Ranks skew by roughly a batch, so it fires intermittently and always on the
    slowest rank: three e4 gsm8k LoRA arms died this way at rollouts 28, 65 and
    114 on 2026-08-04.

    One extra incref per ADDITIONAL consumer restores the pairing. Deduped by
    storage because ForkingPickler reduces a storage once however many tensors
    view it — increfing per tensor would leak the segment instead.

    Returns the number of increfs performed, for tests and diagnostics.
    """
    if consumer_count <= 1:
        return 0
    increfs = 0
    seen: set[int] = set()
    for tensor in tensors.values():
        storage = tensor.untyped_storage()
        key = storage.data_ptr()
        if key in seen:
            continue
        seen.add(key)
        for _ in range(consumer_count - 1):
            storage._shared_incref()
            increfs += 1
    return increfs
