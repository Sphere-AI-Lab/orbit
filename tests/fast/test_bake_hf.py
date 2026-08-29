import torch
import pytest
from orbit.peft.merge.bake_hf import skew_from_vec, cayley_neumann, bake_linear_weight


def test_skew_is_antisymmetric():
    oft_r = torch.randn(3, 6)  # block_size=4 -> P=6
    S = skew_from_vec(oft_r, 4)
    assert S.shape == (3, 4, 4)
    assert torch.allclose(S, -S.transpose(-1, -2), atol=1e-6)


def test_cayley_neumann_near_orthogonal_for_small_input():
    oft_r = 0.01 * torch.randn(3, 6)  # small -> Neumann-5 close to exact Cayley
    R = cayley_neumann(oft_r, 4)
    eye = torch.eye(4).expand_as(R)
    assert (R.transpose(-1, -2) @ R - eye).abs().max() < 1e-3


def test_bake_orientation_matches_runtime_einsum():
    """W' = W @ blockdiag(R^T) must satisfy W'·x == W·rotate(x) for the
    canonical_oft runtime rotation einsum('...rk,rkc->...rc', x, R)."""
    out_f, nb, b = 5, 3, 4
    W = torch.randn(out_f, nb * b)
    R = torch.linalg.qr(torch.randn(nb, b, b))[0]   # arbitrary orthogonal blocks
    x = torch.randn(7, nb * b)
    x_rot = torch.einsum("...rk,rkc->...rc", x.reshape(7, nb, b), R).reshape(7, nb * b)
    y_runtime = x_rot @ W.t()                       # linear applies W to rotated x
    y_baked = x @ bake_linear_weight(W, R).t()
    assert torch.allclose(y_runtime, y_baked, atol=1e-4)


def test_bake_rejects_dim_mismatch():
    W = torch.randn(5, 11)            # 11 not divisible into nb*b=12
    R = torch.linalg.qr(torch.randn(3, 4, 4))[0]
    with pytest.raises(AssertionError):
        bake_linear_weight(W, R)


def test_hf_weight_key_mapping():
    from orbit.peft.merge.bake_hf import _hf_weight_key
    ok = "base_model.model.model.layers.0.self_attn.q_proj.oft_R.weight"
    assert _hf_weight_key(ok) == "model.layers.0.self_attn.q_proj.weight"


def test_cayley_neumann_matches_exact_series():
    oft_r = 0.3 * torch.randn(3, 6)            # block_size=4, non-trivial magnitude
    Q = skew_from_vec(oft_r.float(), 4)
    Q2 = Q @ Q
    Q3 = Q2 @ Q
    Q4 = Q3 @ Q
    eye = torch.eye(4).expand_as(Q)
    ref = eye + 2.0 * Q + 2.0 * Q2 + 2.0 * Q3 + 1.0 * Q4   # I + 2Q + 2Q^2 + 2Q^3 + Q^4
    assert torch.allclose(cayley_neumann(oft_r, 4), ref, atol=1e-5)
