"""The broadcast adapter payload must outlive every TP rank that rebuilds it.

torch's ``file_system`` reduce/rebuild pair is a 1-producer -> 1-consumer
handshake: ``reduce_storage`` increfs once per serialization, every
``rebuild_storage_filename`` decrefs once. SGLang broadcasts ONE payload and
each TP scheduler deserializes it, so an unbalanced payload is released
tp_size-1 times too early and the manager unlinks the segment while the engine
still holds the tensors -- the rank that opens last then dies with
"unable to open shared memory object ... No such file or directory (2)".

CPU only: no GPU, no server, no Ray. The bug lives entirely in the shm
accounting, which is why it reproduces here at all.
"""

import base64
import io
import os
import subprocess
import sys
from multiprocessing.reduction import ForkingPickler

import pytest
import torch
import torch.multiprocessing as torch_mp

from orbit.backends.sglang_utils.sglang_engine import _balance_broadcast_shm_refcounts

# Rebuilds the payload the way tp_worker.load_lora_adapter_from_tensors does:
# torch already imported, weights read out, then the payload explicitly
# released. The explicit release matters -- leaving it to interpreter teardown
# skips the storage destructor's decref and the test would pass vacuously.
_CONSUMER = """
import base64, gc, io, pickle, sys
import torch
data = base64.b64decode(open(sys.argv[1]).read(), validate=True)
tensors = pickle.Unpickler(io.BytesIO(data)).load()
print(sum(float(t.sum()) for t in tensors.values()))
tensors.clear()
gc.collect()
"""


def _shm_segments():
    return {f for f in os.listdir("/dev/shm") if f.startswith("torch_")}


def _serialize(tensors, consumers):
    """Serialize under file_system exactly as the engine actor does."""
    old = torch_mp.get_sharing_strategy()
    torch_mp.set_sharing_strategy("file_system")
    try:
        payload = MultiprocessingSerializerStub.serialize(tensors)
        _balance_broadcast_shm_refcounts(tensors, consumers)
    finally:
        torch_mp.set_sharing_strategy(old)
    return payload


class MultiprocessingSerializerStub:
    """SGLang's MultiprocessingSerializer.serialize(obj, output_str=True),
    inlined so the test does not import the sglang server package."""

    @staticmethod
    def serialize(obj):
        buf = io.BytesIO()
        ForkingPickler(buf).dump(obj)
        return base64.b64encode(buf.getvalue()).decode("utf-8")


def _rebuild_in_subprocess(payload, tmp_path, tag):
    payload_file = tmp_path / f"payload_{tag}.b64"
    payload_file.write_text(payload)
    return subprocess.run(
        [sys.executable, "-c", _CONSUMER, str(payload_file)],
        capture_output=True,
        text=True,
    )


@pytest.fixture
def adapter_tensors():
    # Shape of one LoRA push: a handful of small fresh CPU clones.
    return {f"lora_A.{i}": torch.arange(2048, dtype=torch.float32) for i in range(3)}


def test_payload_survives_every_tp_rank_rebuild(adapter_tensors, tmp_path):
    """TP=2 -- the campaign's engine. Both ranks rebuild the one payload; the
    segments must still exist afterwards, because the engine actor still holds
    the tensors until its HTTP POST returns."""
    before = _shm_segments()
    payload = _serialize(adapter_tensors, consumers=2)
    created = _shm_segments() - before
    assert created, "serialization should have created file_system segments"

    for rank in range(2):
        done = _rebuild_in_subprocess(payload, tmp_path, f"tp{rank}")
        assert done.returncode == 0, f"TP{rank} failed to rebuild: {done.stderr[-400:]}"

    alive = {seg for seg in created if os.path.exists(f"/dev/shm/{seg}")}
    assert alive == created, (
        "segments were unlinked while the producer still holds the tensors: " f"{sorted(created - alive)}"
    )


def test_late_rank_can_still_open_the_segment(adapter_tensors, tmp_path):
    """The production failure, made deterministic: with the refcount unbalanced
    a third rebuild finds the file already unlinked. Balanced, it does not."""
    payload = _serialize(adapter_tensors, consumers=3)
    for rank in range(3):
        done = _rebuild_in_subprocess(payload, tmp_path, f"late{rank}")
        assert done.returncode == 0, f"rank {rank} could not open the broadcast segment: {done.stderr[-400:]}"


def test_repeated_pushes_do_not_grow_dev_shm(tmp_path):
    """The pre-paid increfs must not leak.

    Not 'freed the instant the producer lets go' -- torch's incref is a credit
    only a consumer's rebuild redeems, so the newest payload legitimately
    outlives the producer's release until it is recycled. The invariant that
    matters over a 150-push RL run is that /dev/shm does not GROW per push.
    """
    tensors_per_push = 3

    def mine():
        prefix = f"torch_{os.getpid()}_"
        return {f for f in os.listdir("/dev/shm") if f.startswith(prefix)}

    for push in range(6):
        tensors = {f"lora_A.{i}": torch.arange(1024, dtype=torch.float32) for i in range(tensors_per_push)}
        payload = _serialize(tensors, consumers=2)
        for rank in range(2):
            done = _rebuild_in_subprocess(payload, tmp_path, f"p{push}r{rank}")
            assert done.returncode == 0, done.stderr[-400:]
        tensors.clear()
        assert len(mine()) <= tensors_per_push, (
            f"/dev/shm grew to {len(mine())} segments by push {push + 1}; " "the pre-paid increfs are leaking"
        )


def test_single_consumer_is_left_untouched(adapter_tensors):
    """TP=1 needs no pre-payment -- torch's own pairing is already correct."""
    old = torch_mp.get_sharing_strategy()
    torch_mp.set_sharing_strategy("file_system")
    try:
        MultiprocessingSerializerStub.serialize(adapter_tensors)
        assert _balance_broadcast_shm_refcounts(adapter_tensors, 1) == 0
        assert _balance_broadcast_shm_refcounts(adapter_tensors, 0) == 0
    finally:
        torch_mp.set_sharing_strategy(old)


@pytest.mark.parametrize(
    "num_gpus_per_engine, arg_value, expected",
    [
        (2, 8, 2),  # per-engine override wins: 2 schedulers rebuild the payload
        (None, 4, 4),  # falls back to the launch arg
        (None, None, 1),  # unknown -> no-op, torch's own pairing
    ],
)
def test_consumer_count_is_the_engines_tp_size(num_gpus_per_engine, arg_value, expected):
    """The pre-payment is only correct if it counts TP ranks. Counting 1 leaves
    the original bug; counting the whole node's GPUs leaks."""
    from argparse import Namespace

    from orbit.backends.sglang_utils.sglang_engine import SGLangEngine

    engine = SGLangEngine.__new__(SGLangEngine)
    engine.num_gpus_per_engine = num_gpus_per_engine
    engine.args = Namespace(rollout_num_gpus_per_engine=arg_value)
    assert engine._adapter_payload_consumers() == expected


def test_increfs_are_counted_per_storage_not_per_tensor():
    """ForkingPickler reduces each storage once however many tensors view it,
    so the pre-payment must dedupe or it over-increfs and leaks."""
    base = torch.arange(2048, dtype=torch.float32)
    tensors = {"a": base[:1024], "b": base[1024:], "c": torch.zeros(512)}
    old = torch_mp.get_sharing_strategy()
    torch_mp.set_sharing_strategy("file_system")
    try:
        MultiprocessingSerializerStub.serialize(tensors)
        # 2 distinct storages x (4 - 1) additional consumers
        assert _balance_broadcast_shm_refcounts(tensors, 4) == 6
    finally:
        torch_mp.set_sharing_strategy(old)
