"""Orbit's CP chunk-sizing fix over ``miles.backends.training_utils.cp_utils``.

One change, in two functions that must agree: when a sample is padded to a
``max_seq_len``, the zigzag context-parallel chunk size has to be derived from
that padded length, not from the sample's own token count. Otherwise the slice
each rank takes and the offsets the loss reads back are computed against
different sequence lengths, and the response tokens silently misalign.

Upstream already computes exactly that chunk size -- in its ``bshd`` branch,
which is the padded-batch layout. Both functions treat ``qkv_format`` as
nothing but the switch that chooses which length feeds ``chunk_size``, so
orbit's fix is expressible as "when thd was given an explicit ``max_seq_len``,
delegate through upstream's padded branch". That is why these are patches
rather than copied bodies: orbit carries the CONDITION and upstream keeps
owning the arithmetic, the padding and the slicing.

The one place the two functions differ is the ``cp_size == 1`` early return,
which upstream uses to pad bshd inputs up to ``max_seq_len``. thd does not pad
there and must not start, so ``slice_with_cp`` only re-routes when CP is
actually on. Both re-routings are guarded by the pin: if upstream ever gives
``qkv_format`` a second meaning, the pin goes stale and this file gets reread
before it can silently mean something else.

The vendored edit also asserted that ``max_seq_len`` is not shorter than the
sequence it is meant to pad -- a real footgun, since a too-small value produces
a negative pad and a silently truncated slice. Those asserts move here
unchanged.

Both signatures carry parameters newer than orbit-main-isolated's miles base
(``cp_rank``/``cp_size`` here, ``parallel_state`` in ``slice_with_cp``); they are
forwarded untouched, and ``slice_with_cp`` reads cp_size off the passed state
rather than the ambient one so the re-routing decision matches the body's.

Nothing here imports torch or miles at module scope: ``import orbit`` executes
this module and must stay cheap (see orbit/patch/runtime.py).
"""

from __future__ import annotations

from orbit.patch import original, patch_function

_CP_UTILS = "miles.backends.training_utils.cp_utils"

_REASON = (
    "thd CP chunking must follow the padded max_seq_len, not the sample's own "
    "length, or the rank slices and the logits offsets disagree; upstream ties "
    "the padded chunk size to qkv_format=bshd"
)


@patch_function(
    "miles.backends.training_utils.cp_utils",
    "get_logits_and_tokens_offset_with_cp",
    upstream_sha="0c555a84e5cdb12483ea260fdbf76cde0551be94183dd7c66dca7454355ecb3f",
    reason=_REASON,
)
def get_logits_and_tokens_offset_with_cp(
    total_length: int,
    response_length: int,
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
    cp_rank: int | None = None,
    cp_size: int | None = None,
):
    if qkv_format == "thd" and max_seq_len is not None:
        assert max_seq_len >= total_length, (
            f"max_seq_len must be >= total_length for qkv_format=thd, "
            f"got max_seq_len={max_seq_len}, total_length={total_length}"
        )
        # Upstream's padded-layout branch computes precisely the chunk size a
        # padded thd batch needs; everything after chunk_size is shared.
        qkv_format = "bshd"
    return original(_CP_UTILS, "get_logits_and_tokens_offset_with_cp")(
        total_length, response_length, qkv_format, max_seq_len, cp_rank, cp_size
    )


@patch_function(
    "miles.backends.training_utils.cp_utils",
    "slice_with_cp",
    upstream_sha="1f319f364eddfc0f0e56403b642b91d770c5039ae5ff00d6a57ed5e0a8d501ea",
    reason=_REASON,
)
def slice_with_cp(
    tokens,
    pad_value,
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
    parallel_state=None,
):
    if qkv_format == "thd" and max_seq_len is not None:
        # Read the state first, as upstream's body does, so an uninitialised
        # ParallelState still surfaces as upstream's assertion -- and read it the
        # same way, honouring an explicitly passed state rather than the ambient
        # one (a parameter this miles base added).
        state = parallel_state
        if state is None:
            from miles.backends.training_utils.parallel import get_parallel_state

            state = get_parallel_state()
        cp_size = state.cp.size
        assert max_seq_len >= len(tokens), (
            f"max_seq_len must be >= token length for qkv_format=thd, "
            f"got max_seq_len={max_seq_len}, token_len={len(tokens)}"
        )
        # Only re-route where qkv_format means nothing but the chunk size. With
        # CP off, upstream's bshd path also pads the sequence out to
        # max_seq_len, which thd must not do.
        if cp_size > 1:
            qkv_format = "bshd"
    return original(_CP_UTILS, "slice_with_cp")(
        tokens, pad_value, qkv_format, max_seq_len, parallel_state
    )
