"""Orbit's delegating patches over ``miles.backends.megatron_utils``.

Two unrelated additions that used to be edits inside two vendored files. Both
are expressed from orbit's side now, which is why those files are byte-pristine
again -- and in the broadcast case it also removes the ``import orbit`` line the
vendored module carried, restoring the leaf property patched modules must have
(tests/fast/test_hf_export_patches.py checks it).

* ``parallel.get_packed_seq_params`` -- DSV4 sparse attention needs two extra
  cu_seqlens tensors carried on the ``PackedSeqParams`` object. Upstream builds
  that object and returns it, so orbit attaches the extras to what came back.
  The vendored edit attached them BEFORE upstream stored the object into
  ``batch["packed_seq_params"]``, but there is only ever one instance and the
  dict holds that same reference, so attaching afterwards mutates exactly the
  object both the caller and the batch see.

* ``update_weight...broadcast.update_weights_from_distributed`` -- weight-sync
  payload accounting. The tensors to count are the function's own argument, so
  orbit counts them and hands the call straight on. The vendored edit recorded
  between the Ray metadata dispatch and the NCCL broadcast; recording first is
  the same accounting (``record`` only reads dtype/shape and never raises --
  see orbit/megatron/sync_metrics.py) and it cannot be skipped by a failure in
  the dispatch above it.

Nothing here imports torch or miles at module scope: ``import orbit`` executes
this module and must stay cheap (see orbit/patch/runtime.py).
"""

from __future__ import annotations

from orbit.patch import original, patch_function

_PACKED_SEQ_REASON = (
    "DSV4 sparse attention reads dsv4_cu_seqlens / dsv4_valid_cu_seqlens off the "
    "PackedSeqParams object; upstream builds that object from the fixed thd "
    "fields and has no spelling for them"
)

_PAYLOAD_REASON = (
    "orbit reports weight-sync payload volume (perf/update_weights_payload_bytes "
    "and friends) for the distributed full-model path; upstream broadcasts "
    "without counting what it sent"
)

_DSV4_KEYS = ("dsv4_cu_seqlens", "dsv4_valid_cu_seqlens")

# The decorators below must spell these out as plain string literals -- the pin
# gate reads them statically and rejects anything it cannot constant-fold. The
# constants are for the delegating calls in the bodies.
_PARALLEL = "miles.backends.megatron_utils.parallel"
_BROADCAST = (
    "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast"
)


@patch_function(
    "miles.backends.megatron_utils.parallel",
    "get_packed_seq_params",
    upstream_sha="931173e86c025c7df2dceb67d619e1e83dbe6af80bc1df9c12c6a77d551e1af9",
    reason=_PACKED_SEQ_REASON,
)
def get_packed_seq_params(batch, args):
    packed_seq_params = original(_PARALLEL, "get_packed_seq_params")(batch, args)
    # None for non-thd layouts, which have no cu_seqlens to carry anything on.
    if packed_seq_params is not None:
        for key in _DSV4_KEYS:
            if key in batch:
                setattr(packed_seq_params, key, batch[key])
    return packed_seq_params


@patch_function(
    "miles.backends.megatron_utils.update_weight.update_weight_from_distributed.broadcast",
    "update_weights_from_distributed",
    upstream_sha="42d16a07e440d2a985a6df14209be136f7454ea623798cc3ceb84696fc5cf8a1",
    reason=_PAYLOAD_REASON,
)
def update_weights_from_distributed(
    group_name,
    group,
    weight_version,
    rollout_engines,
    converted_named_tensors,
):
    from orbit.megatron.sync_metrics import get_payload_tracker

    # Counted once per update: only the broadcasting source rank reaches this
    # function, and the engine fan-out reuses that one broadcast rather than
    # multiplying the payload.
    get_payload_tracker().record(converted_named_tensors)
    return original(_BROADCAST, "update_weights_from_distributed")(
        group_name,
        group,
        weight_version,
        rollout_engines,
        converted_named_tensors,
    )
