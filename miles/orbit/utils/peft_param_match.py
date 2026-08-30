"""Match OFT block size to LoRA rank by trainable-parameter count.

OFT stores one skew-symmetric vector per block. Per
``megatron/bridge/orbit/oft/oft_layers.py`` (``OFTRotationModule.__init__``, see
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
Because the snap can move the realized parameter count away from the ideal,
``match_report`` exposes the *realized* ratio so no OFT arm is described as
"matched" to its LoRA counterpart in prose when the snap actually left it far
off. **The snap hurts most at SMALL rank**, not large: the ideal block size is
``1 + 4·rank`` for a square weight, so the absolute gap to the nearest divisor
stays ~O(1) while the relative gap goes as ``1/(1 + 4·rank)``. Measured on
Llama-3.1-8B's square attention shape (d_in = d_out = 4096, whose divisors are
powers of two), the realized ratio is 0.750 at rank 1, 0.938 at rank 4, 0.984 at
rank 16 and 1.000 at rank 512.

All of the above is the count for ONE rotation, which is what legacy OFT
(``--oft-type oft``) builds per module. Canonical OFT -- what every RL launcher
in this tree actually runs -- builds one rotation per **output slice** of a fused
module, so a module total is ``slices · d_in · (b−1) / 2``. See
``OFT_ROTATION_SLICES`` for the evidence and for what counting it the other way
cost; ``oft_param_count`` stays per-rotation, and
``oft_param_count_for_modules`` applies the slice factor.

**A single global block size cannot match LoRA across mixed shapes**, which is
the constraint that shapes any matched-parameter OFT experiment. A rotation's
count is ``d_in·(b−1)/2`` and does not depend on ``d_out`` at all, while LoRA's is
``rank·(d_in + d_out)``. So the per-module ratio scales as
``slices · (b−1) / (2·rank·(1 + d_out/d_in))``, and one shared ``b`` therefore
starves modules with large ``d_out/d_in`` while overfeeding the rest. On
Llama-3.1-8B at ``b = 64`` against rank 16 the realized per-module ratios are
**2.362** (``linear_qkv``, three rotations over a shared input), 0.984
(``linear_proj``), **0.492** (``linear_fc1``, two rotations but ``d_out = 7·d_in``)
and **1.531** (``linear_fc2``). The two fused modules move in OPPOSITE directions,
so slice-awareness widens the spread rather than closing it. Searching every
divisor of 4096 and 14336 cannot fix it: the best achievable all-modules ratio is
0.996 at rank 4 but drifts to 1.049 at rank 16 and 1.065 by rank 256. Megatron's
``--oft-block-size`` is one integer, so per-module block sizes are not
expressible through the CLI either.

The way out is to invert the match: fix the block size and solve for the **LoRA
rank** with the same parameter count (``oft_matched_lora_rank``). Rank is a much
finer lattice than the divisors of ``d_in``, so this lands within a few percent
for ``b >= 32`` -- 0.962 to 1.014 on all-modules -- and it is what lets an OFT arm
and a LoRA arm be compared at genuinely equal capacity.

What this does NOT rescue is matching two OFT *placements* to each other.
Attention-only carries four rotations (three on ``linear_qkv``) against
MLP-only's three, and ``linear_fc1`` snaps to a divisor of 4096 while
``linear_fc2`` snaps to one of 14336, so no single block brings them together:
the best available lands ~26% high at every attention block size. A placement
comparison has to quote ``oft_block_size_matching_params``' realized ratio rather
than describe the pair as matched.
"""

from __future__ import annotations

import math


def lora_param_count(rank: int, d_in: int, d_out: int) -> int:
    """Trainable parameters in a LoRA adapter on a (d_out, d_in) linear."""
    return rank * (d_in + d_out)


def matched_mlp_rank(attn_rank: int, hidden_size: int, ffn_size: int, qkv_output_size: int) -> int:
    """MLP-only LoRA rank whose adapter parameter count matches attention-only.

    Comparing attention-only against MLP-only at *equal rank* compares unequal
    parameter counts, which confounds placement with capacity -- the MLP is the
    larger of the two by a factor that depends only on the shapes. This solves
    for the MLP rank that matches, per layer, using Orbit's **fused** Megatron
    layout rather than HF's separate projections:

        attention  linear_qkv  r*(hidden + qkv_output)   [q, k and v fused]
                   linear_proj r*(hidden + hidden)
        MLP        linear_fc1  r*(hidden + 2*ffn)        [gate and up fused]
                   linear_fc2  r*(ffn + hidden)

    For Llama-3.1-8B (hidden 4096, ffn 14336, qkv_output 6144) that is 18432r
    against 51200r, a ratio of 2.778, so attention r=256 matches MLP r=92.

    Rounds to the nearest integer, since rank is not continuous: the realized
    ratio is therefore within one rank-step of 1.0, not exactly 1.0.
    """
    if min(attn_rank, hidden_size, ffn_size, qkv_output_size) <= 0:
        raise ValueError("attn_rank, hidden_size, ffn_size and qkv_output_size must be positive")
    attn_per_rank = lora_param_count(1, hidden_size, qkv_output_size) + lora_param_count(1, hidden_size, hidden_size)
    mlp_per_rank = lora_param_count(1, hidden_size, 2 * ffn_size) + lora_param_count(1, ffn_size, hidden_size)
    return max(1, round(attn_rank * attn_per_rank / mlp_per_rank))


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
    ``megatron/bridge/orbit/oft/oft_layers.py``: scans ``i = 1..isqrt(n)``,
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


# Orbit's fused Megatron layout for a Llama-style decoder layer. Fused, so these
# are NOT HF's separate q/k/v/gate/up projections: `linear_qkv` bundles q, k and
# v, and `linear_fc1` bundles gate and up. Adapter accounting has to use these
# shapes, not HF's, or every parameter count is wrong.
ATTENTION_MODULES = ("linear_qkv", "linear_proj")
MLP_MODULES = ("linear_fc1", "linear_fc2")

# How many INDEPENDENT rotations OFT builds on each fused module.
#
# `oft_param_count` above mirrors `oft_layers.py`'s `OFTRotationModule`, which is
# LEGACY OFT (`--oft-type oft`): one shared R per module. Every RL launcher in
# this tree passes `--oft-type canonical_oft` instead -- and has since 46c8e0f6,
# so the published E4 numbers are canonical too -- which builds one rotation per
# OUTPUT SLICE. Megatron-Bridge's R is `(num_slices, num_blocks, block_size,
# block_size)`, and `canonical_oft.py` says plainly that "one shared R across
# gate/up halves is mathematically wrong". SGLang consumes it the same way: its
# dense forward rotates once per slice and splits the result into `num_slices`
# copies, each of width `d_in`.
#
# Counting one rotation per module understated E4's all-modules OFT arms by
# 1.46x -- 208 blocks per layer against the 304 actually built. That number fed
# `matched_ratio` and `oft_matched_lora_rank`, so an OFT/LoRA pair recorded as
# parameter-matched was handing OFT ~46% more capacity than the LoRA arm it was
# being compared against, in the direction that flatters OFT.
#
# Ledgers and reports written before this correction carry the old counts. The
# arms themselves are unaffected -- only the number written down about them.
#
# Unfused names (HF's `q_proj`, `gate_proj`, ...) carry one rotation each, which
# is what the default returns.
OFT_ROTATION_SLICES = {
    "linear_qkv": 3,  # q, k, v
    "linear_fc1": 2,  # gate, up
    "linear_proj": 1,
    "linear_fc2": 1,
}


def oft_rotation_slices(module_name: str) -> int:
    """Independent canonical-OFT rotations on `module_name` (1 if unfused)."""
    return OFT_ROTATION_SLICES.get(module_name, 1)


def megatron_module_shapes(
    hidden_size: int,
    ffn_size: int,
    qkv_output_size: int,
) -> dict[str, tuple[int, int]]:
    """`{module_name: (d_in, d_out)}` for one fused decoder layer.

    `qkv_output_size` is passed rather than derived because under GQA it is
    `hidden + 2·num_query_groups·head_dim`, which `hidden_size` alone does not
    determine (6144 for Llama-3.1-8B, not 3·4096).
    """
    if min(hidden_size, ffn_size, qkv_output_size) <= 0:
        raise ValueError("hidden_size, ffn_size and qkv_output_size must be positive")
    return {
        "linear_qkv": (hidden_size, qkv_output_size),
        "linear_proj": (hidden_size, hidden_size),
        # gate and up are fused, so d_out is 2*ffn_size.
        "linear_fc1": (hidden_size, 2 * ffn_size),
        "linear_fc2": (ffn_size, hidden_size),
    }


def oft_param_count_for_modules(
    block_size: int,
    shapes: dict[str, tuple[int, int]],
    *,
    oft_type: str = "canonical_oft",
) -> int:
    """Realized OFT parameters over a set of modules.

    Two things a plain `sum(oft_param_count(...))` gets wrong, both applied here
    rather than assumed away:

    * Bridge snaps `block_size` to a divisor of each module's own `d_in`
      independently, so a module whose `d_in` does not admit the requested block
      gets a different one -- and the caller never hears about it.
    * Under canonical OFT a fused module carries one rotation per output slice,
      not one per module. See `OFT_ROTATION_SLICES` for why, and for what
      counting it the other way cost.

    `oft_type` mirrors the launcher's `--oft-type` flag: `"canonical_oft"` (the
    default, and the only variant any launcher in this tree runs) applies the
    per-slice factor; `"oft"` -- legacy shared-R, one rotation per module
    regardless of fusion -- skips it. The keyword exists so a legacy arm can
    never silently receive canonical accounting, not because anyone should
    parameter-match against a variant Bridge documents as mathematically wrong
    on fused modules.
    """
    if oft_type not in ("oft", "canonical_oft"):
        raise ValueError(
            f"Unsupported OFT type: {oft_type!r}. Expected 'oft' or 'canonical_oft'."
        )
    canonical = oft_type == "canonical_oft"
    return sum(
        (oft_rotation_slices(name) if canonical else 1)
        * oft_param_count(nearest_divisor(d_in, block_size), d_in)
        for name, (d_in, _) in shapes.items()
    )


def lora_param_count_for_modules(rank: int, shapes: dict[str, tuple[int, int]]) -> int:
    """LoRA parameters over a set of modules."""
    return sum(lora_param_count(rank, d_in, d_out) for d_in, d_out in shapes.values())


def oft_matched_lora_rank(
    block_size: int,
    shapes: dict[str, tuple[int, int]],
    *,
    oft_type: str = "canonical_oft",
) -> int:
    """LoRA rank whose parameter count matches this OFT block size.

    The inverse of `matched_oft_block_size`, and the direction that actually
    works: rank is a finer lattice than the divisors of `d_in`, so rounding to an
    integer rank costs a few percent where snapping a block size costs 24% (see
    the module docstring). Never returns 0 -- rank 0 is not an adapter.
    """
    per_rank = lora_param_count_for_modules(1, shapes)
    oft_params = oft_param_count_for_modules(block_size, shapes, oft_type=oft_type)
    return max(1, round(oft_params / per_rank))


def oft_lora_match_report(
    block_size: int,
    shapes: dict[str, tuple[int, int]],
    *,
    oft_type: str = "canonical_oft",
) -> dict:
    """Everything needed to say honestly whether an OFT/LoRA pair is matched.

    `ratio` is realized OFT parameters over realized LoRA parameters at the
    returned rank. Quote it: a pair at 0.93 is not "matched", and which way it
    misses decides how a result may be read -- an OFT arm carrying *fewer*
    parameters that still keeps up strengthens the claim, while one carrying
    fewer and losing is confounded rather than informative.
    """
    oft_params = oft_param_count_for_modules(block_size, shapes, oft_type=oft_type)
    rank = oft_matched_lora_rank(block_size, shapes, oft_type=oft_type)
    lora_params = lora_param_count_for_modules(rank, shapes)
    return {
        "block_size": block_size,
        "oft_type": oft_type,
        "oft_params": oft_params,
        "lora_rank": rank,
        "lora_params": lora_params,
        "ratio": oft_params / lora_params,
        "snapped_block_sizes": {
            name: nearest_divisor(d_in, block_size) for name, (d_in, _) in shapes.items()
        },
    }


def oft_block_size_matching_params(
    target_params: int,
    shapes: dict[str, tuple[int, int]],
    *,
    oft_type: str = "canonical_oft",
) -> int:
    """Block size whose realized OFT parameter count is closest to `target_params`.

    Needed to compare OFT *against itself* across placements at equal capacity --
    attention-only and MLP-only at the same block size are not equal-capacity,
    because OFT's count scales with `d_in` and the MLP's `d_in` sum is larger. It
    is the same error E3 was written to avoid for LoRA, one method over.

    Searches the divisors of every `d_in` involved, plus every integer up to the
    largest of them that any module could snap to, and scores candidates by the
    *realized* count after per-module snapping -- so the returned value is honest
    about what Bridge will actually build, not about what was requested.
    """
    if target_params <= 0:
        raise ValueError("target_params must be positive")
    d_ins = {d_in for d_in, _ in shapes.values()}
    candidates = sorted({d for d_in in d_ins for d in range(1, d_in + 1) if d_in % d == 0})
    return min(
        candidates,
        key=lambda b: abs(
            oft_param_count_for_modules(b, shapes, oft_type=oft_type) - target_params
        ),
    )
