import pytest
import torch

import orbit.merge.bake_hf as bake_hf
from orbit.merge.bake_hf import bake_linear_weight, cayley_neumann, skew_from_vec


class _FakeSaver:
    def save_pretrained(self, _output_dir):
        return None


class _FakeModel(_FakeSaver):
    def __init__(self, state):
        self.state = state

    def to(self, _device):
        return self

    def state_dict(self):
        return self.state


def _install_fake_hf_model(monkeypatch, state):
    model = _FakeModel(state)
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", lambda *_args, **_kwargs: model)
    monkeypatch.setattr(bake_hf, "load_tokenizer", lambda _base: _FakeSaver())
    return model


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
    R = torch.linalg.qr(torch.randn(nb, b, b))[0]  # arbitrary orthogonal blocks
    x = torch.randn(7, nb * b)
    x_rot = torch.einsum("...rk,rkc->...rc", x.reshape(7, nb, b), R).reshape(7, nb * b)
    y_runtime = x_rot @ W.t()  # linear applies W to rotated x
    y_baked = x @ bake_linear_weight(W, R).t()
    assert torch.allclose(y_runtime, y_baked, atol=1e-4)


def test_bake_rejects_dim_mismatch():
    W = torch.randn(5, 11)  # 11 not divisible into nb*b=12
    R = torch.linalg.qr(torch.randn(3, 4, 4))[0]
    with pytest.raises(AssertionError):
        bake_linear_weight(W, R)


def test_bake_embedding_uses_output_rotation_orientation():
    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    rotation = torch.tensor([[[0.0, -1.0], [1.0, 0.0]]])

    baked = bake_hf.bake_embedding_weight(weight, rotation)

    assert torch.equal(baked, torch.tensor([[2.0, -1.0], [4.0, -3.0]]))


def test_bake_linear_repeats_shared_rotation_for_every_input_block():
    weight = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    rotation = torch.tensor([[[0.0, -1.0], [1.0, 0.0]]])

    baked = bake_linear_weight(weight, rotation, block_share=True)

    assert torch.equal(baked, torch.tensor([[-2.0, 1.0, -4.0, 3.0]]))


def test_bake_hf_model_dispatches_embedding_output_rotation(monkeypatch, tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"block_share": false}')
    state = {"model.embed_tokens.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    model = _install_fake_hf_model(monkeypatch, state)
    adapter = {"base_model.model.model.embed_tokens.oft_R.weight": torch.tensor([[0.5]])}

    bake_hf.bake_hf_model("base", str(adapter_dir), 2, str(tmp_path / "output"), adapter=adapter)

    expected = torch.tensor([[-0.9375, 1.8750], [-1.3125, 4.5000]])
    assert torch.equal(model.state["model.embed_tokens.weight"], expected)


def test_bake_hf_model_honors_block_share_config(monkeypatch, tmp_path):
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text('{"block_share": true}')
    state = {"model.layers.0.q_proj.weight": torch.tensor([[1.0, 2.0, 3.0, 4.0]])}
    model = _install_fake_hf_model(monkeypatch, state)
    adapter = {"base_model.model.model.layers.0.q_proj.oft_R.weight": torch.tensor([[0.5]])}

    bake_hf.bake_hf_model("base", str(adapter_dir), 2, str(tmp_path / "output"), adapter=adapter)

    expected = torch.tensor([[2.0625, 0.3750, 4.6875, 0.0000]])
    assert torch.equal(model.state["model.layers.0.q_proj.weight"], expected)


def test_hf_weight_key_mapping():
    from orbit.merge.bake_hf import _hf_weight_key

    ok = "base_model.model.model.layers.0.self_attn.q_proj.oft_R.weight"
    assert _hf_weight_key(ok) == "model.layers.0.self_attn.q_proj.weight"


def test_cayley_neumann_matches_exact_series():
    oft_r = 0.3 * torch.randn(3, 6)  # block_size=4, non-trivial magnitude
    Q = skew_from_vec(oft_r.float(), 4)
    Q2 = Q @ Q
    Q3 = Q2 @ Q
    Q4 = Q3 @ Q
    eye = torch.eye(4).expand_as(Q)
    ref = eye + 2.0 * Q + 2.0 * Q2 + 2.0 * Q3 + 1.0 * Q4  # I + 2Q + 2Q^2 + 2Q^3 + Q^4
    assert torch.allclose(cayley_neumann(oft_r, 4), ref, atol=1e-5)
