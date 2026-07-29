"""Match OFT block size to LoRA rank by trainable-parameter count.

OFT stores one skew-symmetric vector per block. Per
``megatron/bridge/peft/oft_layers.py`` (``OFTRotationModule.__init__``, see
``self.oft_r = nn.Parameter(torch.zeros(num_blocks, n_elements, ...))``):

    n_elements  = block_size * (block_size - 1) // 2
    num_blocks  = d_in // block_size          (or 1 when block_share=True)
    oft_r shape = (num_blocks, n_elements)
    params      = num_blocks * n_elements
                = (d_in // block_size) * block_size * (block_size - 1) // 2

which is exactly ``d_in * (block_size - 1) / 2`` when ``block_size`` divides
``d_in`` (verified against the real parameter shape above, not assumed).

LoRA stores two low-rank factors:

    params      = rank * (d_in + d_out)

Equating the two continuous expressions (ignoring the integer floor in
``d_in // block_size``, which only matters once we snap to an actual
divisor) gives the ideal block size

    b = 1 + 2 * rank * (d_in + d_out) / d_in

which reduces to ``b = 1 + 4 * rank`` when ``d_in == d_out``.

Megatron-Bridge's ``OFTRotationModule`` requires ``block_size`` to divide
``d_in`` and silently snaps a non-dividing value to the nearest divisor via
its ``_find_nearest_divisor`` static method (``oft_layers.py``, around
lines 417-423 and 465-473). ``nearest_divisor`` below mirrors that method
byte-for-byte in behaviour, including its tie-breaking rule (a strict ``<``
comparison, so among equidistant candidates whichever is visited first in
the ``i = 1..isqrt(n)`` scan — pairing ``(i, n // i)`` at each step — wins).
Because the snap can move the realized parameter count away from the ideal
(a lot, at large rank), ``match_report`` exposes the *realized* ratio so no
OFT arm is described as "matched" to its LoRA counterpart in prose when the
snap actually left it far off.
"""

from __future__ import annotations

import math


def lora_param_count(rank: int, d_in: int, d_out: int) -> int:
    """Trainable parameters in a LoRA adapter on a (d_out, d_in) linear."""
    return rank * (d_in + d_out)


def oft_param_count(block_size: int, d_in: int, block_share: bool = False) -> int:
    """Trainable parameters in an OFT adapter on a linear with `d_in` inputs.

    Mirrors ``OFTRotationModule``'s ``oft_r`` parameter: shape
    ``(num_blocks, block_size * (block_size - 1) // 2)`` where
    ``num_blocks = d_in // block_size`` (or 1 when all blocks share the same
    parameters).
    """
    per_block = block_size * (block_size - 1) // 2
    num_blocks = 1 if block_share else d_in // block_size
    return num_blocks * per_block


def nearest_divisor(n: int, target: int) -> int:
    """Nearest divisor of `n` to `target`.

    Mirrors ``OFTRotationModule._find_nearest_divisor`` in
    ``megatron/bridge/peft/oft_layers.py``: scans ``i = 1..isqrt(n)``,
    considers both ``i`` and ``n // i`` as candidate divisors, and keeps a
    candidate only on a *strict* improvement. That strictness is what fixes
    the tie-break: among equidistant divisors, whichever this scan order
    visits first is kept.
    """
    best = 1
    for i in range(1, math.isqrt(n) + 1):
        if n % i:
            continue
        for cand in (i, n // i):
            if abs(cand - target) < abs(best - target):
                best = cand
    return best


def _ideal_block_size(rank: int, d_in: int, d_out: int) -> int:
    """Unconstrained (unclamped, unsnapped) block size matching LoRA params."""
    return 1 + 2 * rank * (d_in + d_out) // d_in


def matched_oft_block_size(rank: int, d_in: int, d_out: int) -> int:
    """Block size whose OFT parameter count is closest to LoRA at `rank`.

    The ideal (continuous) block size is clamped to ``[1, d_in]`` and then,
    if it does not divide ``d_in``, snapped to the nearest divisor exactly
    as Bridge does at adapter-construction time.
    """
    if rank <= 0:
        raise ValueError("rank must be positive")
    ideal = _ideal_block_size(rank, d_in, d_out)
    ideal = min(max(ideal, 1), d_in)
    if d_in % ideal == 0:
        return ideal
    return nearest_divisor(d_in, ideal)


def match_report(rank: int, d_in: int, d_out: int) -> dict:
    """Full accounting of a LoRA-to-OFT parameter match, for logging.

    ``ideal_block_size`` is the unconstrained target from equating param
    counts (may exceed ``d_in`` at very large rank); ``block_size`` is the
    realized value after clamping and snapping to a divisor of ``d_in``, the
    same value Bridge would actually construct. ``ratio`` is
    ``oft_params / lora_params`` at the realized block size, so a loose
    match is visible in the numbers rather than hidden behind the word
    "matched".
    """
    ideal = _ideal_block_size(rank, d_in, d_out)
    block_size = matched_oft_block_size(rank, d_in, d_out)
    lora_params = lora_param_count(rank, d_in, d_out)
    oft_params = oft_param_count(block_size, d_in)
    return {
        "rank": rank,
        "d_in": d_in,
        "d_out": d_out,
        "ideal_block_size": ideal,
        "block_size": block_size,
        "lora_params": lora_params,
        "oft_params": oft_params,
        "ratio": oft_params / lora_params,
    }
