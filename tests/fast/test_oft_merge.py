import pytest
import torch

import miles.merge  # noqa: F401  (registers strategies)
from miles.merge.oft_merge import (
    infer_oft_block_size,
    magnitude_corrected_merge,
    oft_params_to_skew_matrix,
    orthomerge_original_merge,
    skew_matrix_to_oft_params,
)
from miles.merge.strategy import get_strategy


def _rand_vec(num_blocks=3, block_size=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    P = block_size * (block_size - 1) // 2
    return torch.randn(num_blocks, P, generator=g)


def _ref_orthomerge(vecs, block_size):
    """Reference: OrthoMerge's merge_cayley_Q_list via full skew matrices (equal weights)."""
    idx = torch.triu_indices(block_size, block_size, 1)

    def to_skew(v):
        B, _ = v.shape
        S = torch.zeros(B, block_size, block_size, dtype=torch.float64)
        S[:, idx[0], idx[1]] = v.double()
        return S - S.transpose(-1, -2)

    Ss = [to_skew(v) for v in vecs]
    stack = torch.stack(Ss, 0)
    merged_sum = stack.sum(0)
    sum_norms = torch.stack([s.flatten().norm() for s in Ss]).sum()
    c = sum_norms / merged_sum.flatten().norm()
    merged = (1.0 / len(vecs)) * c * merged_sum
    merged = 0.5 * (merged - merged.transpose(-1, -2))
    return merged[:, idx[0], idx[1]]  # back to (B, P), float64


def test_single_adapter_is_identity():
    v = _rand_vec(seed=1)
    out = magnitude_corrected_merge([v])
    assert torch.allclose(out, v, atol=1e-6)


def test_identical_adapters_returns_same():
    v = _rand_vec(seed=2)
    out = magnitude_corrected_merge([v, v, v])
    assert torch.allclose(out, v, atol=1e-5)


def test_weights_select_single_adapter():
    a, b = _rand_vec(seed=3), _rand_vec(seed=4)
    out = magnitude_corrected_merge([a, b], weights=[1.0, 0.0])
    assert torch.allclose(out, a, atol=1e-5)


def test_matches_orthomerge_reference():
    vecs = [_rand_vec(num_blocks=3, block_size=4, seed=s) for s in (5, 6, 7)]
    out = magnitude_corrected_merge(vecs).double()
    ref = _ref_orthomerge(vecs, block_size=4)
    assert torch.allclose(out, ref, atol=1e-6)


def test_preserves_shape_and_dtype():
    vecs = [_rand_vec(seed=s).to(torch.bfloat16) for s in (8, 9)]
    out = magnitude_corrected_merge(vecs)
    assert out.shape == vecs[0].shape
    assert out.dtype == torch.bfloat16


def test_strategy_registry_oft_present_and_unknown_raises():
    s = get_strategy("oft")
    assert s.name == "oft"
    with pytest.raises(KeyError):
        get_strategy("procrustes-ties")


def test_oft_strategy_merges_oft_keys_and_averages_others():
    a = {
        "base_model.model.layers.0.self_attn.q_proj.oft_R.weight": _rand_vec(seed=10),
        "extra.scalar": torch.tensor([2.0, 4.0]),
    }
    b = {
        "base_model.model.layers.0.self_attn.q_proj.oft_R.weight": _rand_vec(seed=10),
        "extra.scalar": torch.tensor([4.0, 8.0]),
    }
    merged = get_strategy("oft").merge([a, b])
    # identical oft_R inputs -> unchanged
    k = "base_model.model.layers.0.self_attn.q_proj.oft_R.weight"
    assert torch.allclose(merged[k], a[k], atol=1e-5)
    # non-oft key -> plain mean
    assert torch.allclose(merged["extra.scalar"], torch.tensor([3.0, 6.0]))


def test_oft_strategy_rejects_key_mismatch():
    a = {"x.oft_R.weight": _rand_vec(seed=1)}
    b = {"y.oft_R.weight": _rand_vec(seed=1)}
    with pytest.raises(ValueError):
        get_strategy("oft").merge([a, b])


def test_naive_oft_merge_is_plain_mean():
    a = {"x.oft_R.weight": _rand_vec(seed=1)}
    b = {"x.oft_R.weight": _rand_vec(seed=2)}
    merged = get_strategy("oft-naive").merge([a, b])
    expected = torch.stack([a["x.oft_R.weight"].float(), b["x.oft_R.weight"].float()]).mean(0)
    assert torch.allclose(merged["x.oft_R.weight"], expected.to(a["x.oft_R.weight"].dtype), atol=1e-6)


def test_infer_oft_block_size_from_upper_triangle_width():
    assert infer_oft_block_size(6) == 4
    assert infer_oft_block_size(496) == 32
    with pytest.raises(ValueError, match="not a valid OFT upper-triangle width"):
        infer_oft_block_size(7)


def test_skew_round_trip_preserves_oft_params():
    v = _rand_vec(num_blocks=2, block_size=4, seed=101)
    skew = oft_params_to_skew_matrix(v, block_size=4)
    assert skew.shape == (2, 4, 4)
    assert torch.allclose(skew + skew.transpose(-1, -2), torch.zeros_like(skew))
    got = skew_matrix_to_oft_params(skew)
    assert torch.allclose(got, v)


def test_original_formula_matches_full_skew_reference_for_three_adapters():
    vecs = [_rand_vec(num_blocks=3, block_size=4, seed=s) for s in (111, 112, 113)]
    got = orthomerge_original_merge(vecs, block_size=4).double()
    ref = _ref_orthomerge(vecs, block_size=4)
    assert torch.allclose(got, ref, atol=1e-6)


def test_oft_original_strategy_merges_three_adapters():
    key = "base_model.model.layers.0.self_attn.q_proj.oft_R.weight"
    adapters = [{key: _rand_vec(num_blocks=3, block_size=4, seed=s)} for s in (121, 122, 123)]
    got = get_strategy("oft-original").merge(adapters)[key].double()
    ref = _ref_orthomerge([ad[key] for ad in adapters], block_size=4)
    assert torch.allclose(got, ref, atol=1e-6)


def test_oft_original_strategy_merges_native_oft_r_key():
    key = "module.decoder.layers.0.adapter.oft_r"
    adapters = [{key: _rand_vec(num_blocks=3, block_size=4, seed=s)} for s in (131, 132, 133)]
    got = get_strategy("oft-original").merge(adapters)[key].double()
    ref = _ref_orthomerge([ad[key] for ad in adapters], block_size=4)
    assert torch.allclose(got, ref, atol=1e-6)


@pytest.mark.parametrize("method", ["oft", "oft-original", "oft-naive"])
def test_oft_strategies_reject_dsv4_native_tensor(method):
    key = "decoder.layers.0.mlp.experts.w1_oft_r"
    adapters = [{key: _rand_vec(num_blocks=12, block_size=4, seed=s).reshape(4, 3, 6)} for s in (241, 242)]

    with pytest.raises(NotImplementedError, match="DSV4"):
        get_strategy(method).merge(adapters)


def test_oft_original_strategy_does_not_treat_soft_or_classifier_keys_as_oft():
    keys = [
        "soft_rating.weight",
        "soft_router.weight",
        "some_oft_config",
        "classifier.oft_R.weight",
    ]
    adapters = [{key: _rand_vec(num_blocks=3, block_size=4, seed=s) for key in keys} for s in (201, 202, 203)]
    merged = get_strategy("oft-original").merge(adapters)
    for key in keys:
        expected = torch.stack([ad[key].float() for ad in adapters]).mean(0)
        assert torch.allclose(merged[key], expected.to(adapters[0][key].dtype), atol=1e-6)


def test_oft_original_strategy_rejects_weights():
    key = "x.oft_R.weight"
    adapters = [{key: _rand_vec(seed=s)} for s in (141, 142)]
    with pytest.raises(ValueError, match="does not accept weights"):
        get_strategy("oft-original").merge(adapters, weights=[0.25, 0.75])


def test_oft_original_strategy_averages_non_oft_keys():
    adapters = [
        {"extra.scalar": torch.tensor([2.0, 4.0])},
        {"extra.scalar": torch.tensor([4.0, 8.0])},
    ]
    merged = get_strategy("oft-original").merge(adapters)
    assert torch.allclose(merged["extra.scalar"], torch.tensor([3.0, 6.0]))


def test_oft_original_strategy_rejects_key_mismatch():
    a = {"x.oft_R.weight": _rand_vec(seed=151)}
    b = {"y.oft_R.weight": _rand_vec(seed=152)}
    with pytest.raises(ValueError, match="key set differs"):
        get_strategy("oft-original").merge([a, b])


def test_oft_original_strategy_rejects_shape_mismatch():
    key = "x.oft_R.weight"
    a = {key: _rand_vec(num_blocks=3, block_size=4, seed=161)}
    b = {key: _rand_vec(num_blocks=4, block_size=4, seed=162)}
    with pytest.raises(ValueError, match="shape mismatch"):
        get_strategy("oft-original").merge([a, b])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_oft_original_merge_uses_stable_internal_precision_for_low_precision_inputs(dtype):
    vecs = [(1000 * _rand_vec(num_blocks=64, block_size=32, seed=s)).to(dtype) for s in (171, 172, 173)]
    out = orthomerge_original_merge(vecs, block_size=32)
    assert out.dtype == dtype
    assert torch.isfinite(out).all()


def test_oft_original_merge_rejects_zero_summed_generators():
    v = _rand_vec(num_blocks=3, block_size=4, seed=181)
    with pytest.raises(ValueError, match="zero norm of summed generators"):
        orthomerge_original_merge([v, -v], block_size=4)


def test_oft_original_merge_rejects_malformed_rank_with_value_error():
    with pytest.raises(ValueError, match=r"OFT params must have shape \(\.\.\., P\)"):
        orthomerge_original_merge([torch.ones(6)])
