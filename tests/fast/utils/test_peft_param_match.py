"""Parameter-count matching between LoRA rank and OFT block size.

The formulas here must track megatron/bridge/peft/oft_layers.py: oft_r has
shape (d_in // block_size, block_size * (block_size - 1) // 2), and a block
size that does not divide d_in is snapped to the nearest divisor.
"""

import pytest

from orbit.utils.peft_param_match import (
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

    BRIDGE_PATH = (
        "/lustre/fast/fast/zqiu/NeckariumAI/clthegoat/release/Megatron-Bridge/"
        "src/megatron/bridge/peft/oft_layers.py"
    )

    @pytest.fixture(scope="class")
    def bridge_find_nearest_divisor(self):
        import ast
        import math

        try:
            src = open(self.BRIDGE_PATH).read()
        except OSError:
            pytest.skip("Megatron-Bridge checkout not available in this environment")

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
