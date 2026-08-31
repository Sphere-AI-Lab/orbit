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
    upstream_sha="195e14878fe6b2ae3188de80b1ae8454f4d3f8bfed5da88021a0b3d1d36c9a06",
    reason=_REASON,
)
def get_logits_and_tokens_offset_with_cp(
    total_length: int,
    response_length: int,
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
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
        total_length, response_length, qkv_format, max_seq_len
    )


@patch_function(
    "miles.backends.training_utils.cp_utils",
    "slice_with_cp",
    upstream_sha="303958ccd86a0180309a1dc882ba60d851b0c1c0cbc7f1cf727a62e8a3397ce1",
    reason=_REASON,
)
def slice_with_cp(
    tokens,
    pad_value,
    qkv_format: str = "thd",
    max_seq_len: int | None = None,
):
    from miles.backends.training_utils.parallel import get_parallel_state

    if qkv_format == "thd" and max_seq_len is not None:
        # Read the state first, as upstream's body does, so an uninitialised
        # ParallelState still surfaces as upstream's assertion.
        cp_size = get_parallel_state().cp.size
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
        tokens, pad_value, qkv_format, max_seq_len
    )
