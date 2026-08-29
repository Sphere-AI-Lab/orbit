"""Parameter-count matching between LoRA rank and OFT block size.

The formulas here must track megatron/bridge/orbit/oft/oft_layers.py: oft_r has
shape (d_in // block_size, block_size * (block_size - 1) // 2), and a block
size that does not divide d_in is snapped to the nearest divisor.
"""

import os
from pathlib import Path

import pytest

from orbit.peft.utils.peft_param_match import (
    lora_param_count,
    match_report,
    matched_oft_block_size,
    nearest_divisor,
    oft_param_count,
)


class TestParamCounts:
    def test_lora_param_count_square(self):
        assert lora_param_count(rank=16, d_in=2560, d_out=2560) == 16 * 5120

    def test_lora_param_count_rectangular(self):
        assert lora_param_count(rank=8, d_in=2560, d_out=9728) == 8 * (2560 + 9728)

    def test_oft_param_count_matches_bridge_shape(self):
        # (d_in // b) blocks, each b(b-1)/2 elements.
        d_in, b = 2560, 64
        assert oft_param_count(b, d_in) == (d_in // b) * (b * (b - 1) // 2)
        assert oft_param_count(b, d_in) == d_in * (b - 1) // 2

    def test_oft_param_count_block_share_ties_all_blocks(self):
        assert oft_param_count(64, 2560, block_share=True) == 64 * 63 // 2


class TestNearestDivisor:
    def test_exact_divisor_is_unchanged(self):
        assert nearest_divisor(2560, 64) == 64

    def test_snaps_below_when_closer(self):
        # divisors of 2560 around 70: 64 and 80 -> 64 is nearer
        assert nearest_divisor(2560, 70) == 64

    def test_snaps_above_when_closer(self):
        assert nearest_divisor(2560, 78) == 80

    def test_never_returns_zero(self):
        assert nearest_divisor(2560, 1) == 1

    def test_tie_prefers_first_found_like_bridge(self):
        # 40 and 64 are both divisors of 2560, equidistant from 52
        # (|52-40|=12, |52-64|=12). Bridge's strict `<` comparison keeps
        # whichever candidate it visits first; pin that behaviour here
        # rather than assuming a "round to nearest even/lower" rule.
        assert nearest_divisor(2560, 52) == 40


class TestMatchedBlockSize:
    def test_rank_1_square_is_exact(self):
        # b = 1 + 4*1 = 5, and 5 divides 2560
        b = matched_oft_block_size(rank=1, d_in=2560, d_out=2560)
        assert b == 5
        assert oft_param_count(b, 2560) == lora_param_count(1, 2560, 2560)

    def test_rank_16_square_snaps_to_64(self):
        # ideal b = 65, nearest divisor of 2560 is 64
        assert matched_oft_block_size(rank=16, d_in=2560, d_out=2560) == 64

    def test_rank_16_match_is_within_two_percent(self):
        rep = match_report(rank=16, d_in=2560, d_out=2560)
        assert 0.98 <= rep["ratio"] <= 1.02

    def test_rank_256_match_is_loose_and_reported_as_such(self):
        rep = match_report(rank=256, d_in=2560, d_out=2560)
        assert rep["ideal_block_size"] == 1025
        # The snap is far away, so the ratio must NOT be near 1 — and the
        # report must expose that rather than hide it.
        assert not (0.9 <= rep["ratio"] <= 1.1)

    def test_report_exposes_all_keys(self):
        rep = match_report(rank=16, d_in=2560, d_out=9728)
        assert set(rep) == {
            "rank", "d_in", "d_out", "ideal_block_size",
            "block_size", "lora_params", "oft_params", "ratio",
        }

    def test_block_size_cannot_exceed_d_in(self):
        b = matched_oft_block_size(rank=4096, d_in=2560, d_out=2560)
        assert b <= 2560
        assert 2560 % b == 0

    def test_rejects_nonpositive_rank(self):
        with pytest.raises(ValueError, match="rank must be positive"):
            matched_oft_block_size(rank=0, d_in=2560, d_out=2560)


class TestAgreesWithBridgeFindNearestDivisor:
    """Cross-check nearest_divisor against Megatron-Bridge's own
    OFTRotationModule._find_nearest_divisor, extracted from its source via
    AST (so the check runs without importing torch/megatron.core) rather
    than trusting a hand-transcribed copy.
    """

    @pytest.fixture(scope="class")
    def bridge_find_nearest_divisor(self):
        import ast
        import math

        bridge_root = os.environ.get("MEGATRON_BRIDGE_ROOT")
        if bridge_root is None:
            pytest.skip("MEGATRON_BRIDGE_ROOT is not set")
        bridge_path = Path(bridge_root) / "src/megatron/bridge/orbit/oft/oft_layers.py"

        try:
            src = bridge_path.read_text()
        except OSError:
            pytest.skip(f"Megatron-Bridge source not available at {bridge_path}")

        tree = ast.parse(src)
        func_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "OFTRotationModule":
                for item in node.body:
                    if (
                        isinstance(item, ast.FunctionDef)
                        and item.name == "_find_nearest_divisor"
                    ):
                        func_node = item
                        break
        if func_node is None:
            pytest.skip("_find_nearest_divisor not found in Bridge source")

        func_src = ast.get_source_segment(src, func_node)
        namespace = {"math": math}
        exec(func_src, namespace)  # noqa: S102 - trusted local source, test-only
        return namespace["_find_nearest_divisor"]

    def test_matches_over_range_of_targets(self, bridge_find_nearest_divisor):
        mismatches = []
        for n in (2560, 4096, 3072, 5120, 9728, 1024):
            for target in range(0, n + 5, 7):  # sample every 7th target
                bridge_val = bridge_find_nearest_divisor(n, target)
                mine_val = nearest_divisor(n, target)
                if bridge_val != mine_val:
                    mismatches.append((n, target, bridge_val, mine_val))
        assert not mismatches, mismatches[:10]

    def test_matches_on_forced_tie(self, bridge_find_nearest_divisor):
        # See test_tie_prefers_first_found_like_bridge: n=2560, target=52 is
        # equidistant from divisors 40 and 64.
        assert bridge_find_nearest_divisor(2560, 52) == nearest_divisor(2560, 52)


# ---------------------------------------------------------------------------
# Matched-parameter OFT (E5). The premise of that experiment is equal capacity,
# so these pin the accounting that decides whether "matched" is true.
# ---------------------------------------------------------------------------

from orbit.peft.utils.peft_param_match import (  # noqa: E402
    ATTENTION_MODULES,
    MLP_MODULES,
    lora_param_count_for_modules,
    megatron_module_shapes,
    oft_block_size_matching_params,
    oft_lora_match_report,
    oft_matched_lora_rank,
    oft_param_count,
    oft_param_count_for_modules,
    oft_rotation_slices,
)

LLAMA31_8B = dict(hidden_size=4096, ffn_size=14336, qkv_output_size=6144)


def _subset(shapes, names):
    return {name: shapes[name] for name in names}


def test_megatron_shapes_are_fused_not_hf_separate():
    """linear_qkv bundles q/k/v and linear_fc1 bundles gate/up. Using HF's
    separate projections here would make every parameter count wrong."""
    shapes = megatron_module_shapes(**LLAMA31_8B)
    assert shapes["linear_qkv"] == (4096, 6144)
    assert shapes["linear_fc1"] == (4096, 2 * 14336)
    assert shapes["linear_fc2"] == (14336, 4096)
    # Same per-rank totals the E3 arithmetic is stated with.
    assert lora_param_count_for_modules(1, _subset(shapes, ATTENTION_MODULES)) == 18432
    assert lora_param_count_for_modules(1, _subset(shapes, MLP_MODULES)) == 51200


def test_block_size_snap_error_is_worst_at_small_rank():
    """The module docstring's claim, pinned as behaviour: the ideal block is
    1+4*rank, so the absolute gap to a divisor stays O(1) while the relative gap
    goes as 1/(1+4*rank)."""
    from orbit.peft.utils.peft_param_match import match_report

    ratios = [match_report(rank, 4096, 4096)["ratio"] for rank in (1, 4, 16, 64, 256)]
    assert ratios[0] < 0.8
    assert ratios[-1] > 0.99
    assert ratios == sorted(ratios), "error must shrink monotonically as rank grows"


def test_one_global_block_size_cannot_match_across_mixed_shapes():
    """The constraint that forces E5's design. OFT's per-rotation count ignores
    d_out, so a shared block size starves linear_fc1 (d_out = 7*d_in) and
    overfeeds linear_fc2 -- and no divisor fixes both.

    Canonical OFT widens the spread rather than closing it: linear_qkv carries
    three rotations and linear_fc1 two, so the two fused modules move in
    OPPOSITE directions relative to LoRA (qkv up to 2.36, fc1 only to 0.49).
    """
    shapes = megatron_module_shapes(**LLAMA31_8B)
    per_module = {
        name: oft_param_count_for_modules(64, {name: shape})
        / lora_param_count_for_modules(16, {name: shape})
        for name, shape in shapes.items()
    }
    assert per_module["linear_fc1"] < 0.6
    assert per_module["linear_fc2"] > 1.5
    assert per_module["linear_qkv"] > 2.0, "3 rotations on a fused qkv"
    whole = oft_param_count_for_modules(64, shapes) / lora_param_count_for_modules(16, shapes)
    assert 1.05 < whole < 1.15, "all-modules lands ~1.10, not 1.0"


def test_inverting_the_match_lands_within_a_few_percent():
    """Rank is a finer lattice than the divisors of d_in, which is the whole
    reason E5 fixes the block size and solves for the rank."""
    shapes = megatron_module_shapes(**LLAMA31_8B)
    for block_size in (32, 64, 256, 1024):
        report = oft_lora_match_report(block_size, shapes)
        assert abs(report["ratio"] - 1.0) < 0.05, report


def test_small_block_sizes_cannot_be_matched_and_say_so():
    """Where the rank lattice runs out, the report must expose it rather than
    round it away.

    The boundary moved with canonical accounting: three rotations on qkv put b=8
    at rank 2 (ratio 0.978, genuinely matched), so the coarseness now bites at
    b <= 4, where the nearest rank is 1 and the match is off by a factor.
    """
    shapes = megatron_module_shapes(**LLAMA31_8B)
    report = oft_lora_match_report(2, shapes)
    assert report["lora_rank"] == 1
    assert report["ratio"] < 0.35, report


def test_matched_lora_rank_is_never_zero():
    shapes = megatron_module_shapes(**LLAMA31_8B)
    assert oft_matched_lora_rank(2, shapes) >= 1


def test_legacy_oft_type_counts_one_rotation_per_module():
    """`--oft-type oft` (legacy shared-R) builds ONE rotation per module no
    matter the fusion, so its count must skip the slice factor.

    Pinned to the number E4's ledgers recorded before the canonical correction
    -- 54,099,968 at b128 all-modules over 32 layers -- because that is exactly
    what those ledgers were counting: legacy accounting applied to canonical
    arms. The keyword exists so the two variants can never be silently
    conflated again, in either direction.
    """
    shapes = megatron_module_shapes(**LLAMA31_8B)
    legacy = oft_param_count_for_modules(128, shapes, oft_type="oft")
    assert legacy * 32 == 54_099_968
    canonical = oft_param_count_for_modules(128, shapes)
    assert canonical * 32 == 79_069_184
    # Unfused (HF-style) names carry one rotation under BOTH variants.
    unfused = {"q_proj": (4096, 4096), "gate_proj": (4096, 14336)}
    assert oft_param_count_for_modules(64, unfused, oft_type="oft") == (
        oft_param_count_for_modules(64, unfused)
    )
    # The report records which accounting produced it.
    assert oft_lora_match_report(128, shapes, oft_type="oft")["oft_params"] == legacy
    assert oft_lora_match_report(128, shapes)["oft_type"] == "canonical_oft"


def test_unsupported_oft_type_raises():
    shapes = megatron_module_shapes(**LLAMA31_8B)
    with pytest.raises(ValueError, match="Unsupported OFT type"):
        oft_param_count_for_modules(64, shapes, oft_type="dora")


def test_oft_placements_cannot_be_matched_by_block_size_alone():
    """attention-only and MLP-only are not equal-capacity at the same block size,
    and under canonical accounting they cannot be BROUGHT to equal capacity by
    choosing one either.

    This inverts what the suite previously asserted. Counting one rotation per
    module, the search matched them to within 2%, and E3/E5's 2x2 placement
    design was built on that. Three rotations on `linear_qkv` make attention-only
    much heavier, and MLP-only's realized counts cannot come down to meet it --
    `linear_fc1` snaps to a divisor of 4096 while `linear_fc2` snaps to one of
    14336, and no single block satisfies both. The best available lands ~26%
    high at every attention block size.

    So a placement comparison has to QUOTE the realized ratio; it cannot claim a
    match. Pinned across several block sizes because the gap is structural, not
    an artifact of one choice.
    """
    shapes = megatron_module_shapes(**LLAMA31_8B)
    attn, mlp = _subset(shapes, ATTENTION_MODULES), _subset(shapes, MLP_MODULES)
    for attn_block in (32, 64, 128, 256):
        attn_params = oft_param_count_for_modules(attn_block, attn)
        same_block = oft_param_count_for_modules(attn_block, mlp) / attn_params
        assert same_block > 1.3, (attn_block, same_block)

        mlp_block = oft_block_size_matching_params(attn_params, mlp)
        ratio = oft_param_count_for_modules(mlp_block, mlp) / attn_params
        assert 1.2 < ratio < 1.3, (attn_block, mlp_block, ratio)


def test_fused_modules_carry_one_rotation_per_output_slice():
    """The regression this suite did not have.

    Canonical OFT (`--oft-type canonical_oft`, which every RL launcher here
    passes) builds one rotation per OUTPUT SLICE: Megatron-Bridge's R is
    `(num_slices, num_blocks, block_size, block_size)`, and sglang's dense
    forward splits the rotated activation into `num_slices` copies of width
    `d_in`. Counting one per module understated every OFT arm -- 54,099,968
    recorded for E4's b128 all-modules arms against 79,069,184 actually built,
    a factor of 1.46 that fed `matched_ratio` and `oft_matched_lora_rank`.
    """
    shapes = megatron_module_shapes(**LLAMA31_8B)
    assert oft_rotation_slices("linear_qkv") == 3
    assert oft_rotation_slices("linear_fc1") == 2
    assert oft_rotation_slices("linear_proj") == 1
    assert oft_rotation_slices("linear_fc2") == 1
    # Unfused names are one rotation each, so HF-style shapes stay correct.
    assert oft_rotation_slices("q_proj") == 1

    # A fused module costs exactly its slice count times one rotation.
    one_rotation = oft_param_count(128, 4096)
    assert oft_param_count_for_modules(128, _subset(shapes, ("linear_qkv",))) == 3 * one_rotation
    assert oft_param_count_for_modules(128, _subset(shapes, ("linear_fc1",))) == 2 * one_rotation
    assert oft_param_count_for_modules(128, _subset(shapes, ("linear_proj",))) == one_rotation

    # The E4 b128 all-modules arm, per layer and over Llama-3.1-8B's 32 layers.
    assert oft_param_count_for_modules(128, shapes) * 32 == 79_069_184


def test_block_size_matching_params_rejects_nonpositive_targets():
    shapes = megatron_module_shapes(**LLAMA31_8B)
    with pytest.raises(ValueError, match="target_params must be positive"):
        oft_block_size_matching_params(0, shapes)
