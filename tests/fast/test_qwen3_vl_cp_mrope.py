"""CPU unit test for the Qwen3-VL CP+THD packed mRoPE reconstruction (issue #1296).

Under context parallelism each rank's THD row holds only its zigzag chunks of every packed
segment. `_reassemble_full_row` de-interleaves the all-gathered per-rank rows back to the
full natural-order row so per-segment MRoPE positions can be rebuilt and re-sliced. This
test checks that reconstruction is the exact inverse of `slice_with_cp` (the function miles
uses to shard the tokens), and that re-slicing with `_natural_to_zigzag_slice` round-trips.
"""

import logging
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from miles_plugins.models import qwen3_vl
from miles_plugins.models.qwen3_vl import _natural_to_zigzag_slice, _reassemble_full_row


def _slice_with_cp(tokens, cp_size, cp_rank, pad_value=0):
    """Reference copy of cp_utils.slice_with_cp's THD zigzag slicing (per sample)."""
    token_len = len(tokens)
    chunk = (token_len + 2 * cp_size - 1) // (2 * cp_size)
    pad = 2 * cp_size * chunk - token_len
    if pad:
        tokens = F.pad(tokens, (0, pad), value=pad_value)
    s1, e1 = chunk * cp_rank, chunk * (cp_rank + 1)
    s2, e2 = chunk * (2 * cp_size - cp_rank - 1), chunk * (2 * cp_size - cp_rank)
    return torch.cat([tokens[s1:e1], tokens[s2:e2]])


def _build_like_get_batch(sample_lens, cp_size, pad_size=8):
    """Mimic miles get_batch THD+CP packing: per-sample zigzag slice, concat, pad, cu*cp."""
    samples = []
    base = 1
    for L in sample_lens:
        samples.append(torch.arange(base, base + L))  # unique nonzero ids
        base += L
    per_rank = []
    for r in range(cp_size):
        row = torch.cat([_slice_with_cp(t, cp_size, r) for t in samples])
        per_rank.append(row)
    cu = [0]
    for t in samples:
        cu.append(cu[-1] + _slice_with_cp(t, cp_size, 0).size(0))
    final_pad = (pad_size - per_rank[0].size(0) % pad_size) % pad_size
    if final_pad:
        per_rank = [F.pad(row, (0, final_pad), value=0) for row in per_rank]
        cu.append(cu[-1] + final_pad)
    cu = [x * cp_size for x in cu]
    return samples, per_rank, cu


@pytest.mark.parametrize(
    "cp_size,sample_lens",
    [(2, [10, 7, 13]), (2, [16, 16]), (4, [20, 9, 30, 5]), (2, [3]), (4, [40, 17])],
)
def test_reassemble_is_inverse_of_slice_with_cp(cp_size, sample_lens):
    samples, per_rank, cu = _build_like_get_batch(sample_lens, cp_size)
    local_len = per_rank[0].size(0)
    assert cu[-1] == cp_size * local_len

    full = _reassemble_full_row(per_rank, cu, cp_size)
    assert full is not None and full.numel() == cu[-1]

    # Each real sample's tokens reappear (in order) at the start of its segment.
    for i, t in enumerate(samples):
        seg = full[cu[i] : cu[i + 1]]
        assert torch.equal(seg[: t.numel()], t)

    # Re-slicing the full row per segment recovers exactly each rank's local chunks.
    for r in range(cp_size):
        recon = []
        for i in range(len(cu) - 1):
            recon.append(_natural_to_zigzag_slice(full[cu[i] : cu[i + 1]], cp_size, r, dim=0))
        assert torch.equal(torch.cat(recon), per_rank[r])


def test_reassemble_bails_on_indivisible_segment():
    # A segment length not divisible by 2*cp -> None (caller falls back to dense path).
    cu = [0, 6]  # 6 not divisible by 2*cp=4
    gathered = [torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.long)]
    assert _reassemble_full_row(gathered, cu, cp_size=2) is None


def test_deepstack_patch_is_viewless_differentiable_and_idempotent(monkeypatch):
    class FakeTransformerBlock:
        def _deepstack_process(self, hidden_states):
            return hidden_states.transpose(0, 1)

    helper_calls = []

    def make_viewless_tensor(*, inp, requires_grad, keep_graph):
        helper_calls.append((requires_grad, keep_graph))
        return inp.clone()

    bridge_module = SimpleNamespace(Qwen3VLTransformerBlock=FakeTransformerBlock)
    core_utils = SimpleNamespace(make_viewless_tensor=make_viewless_tensor)
    real_import = qwen3_vl.importlib.import_module

    def fake_import(name):
        if name == "megatron.bridge":
            return SimpleNamespace()
        if name.endswith(".transformer_block"):
            return bridge_module
        if name == "megatron.core.utils":
            return core_utils
        return real_import(name)

    monkeypatch.setattr(qwen3_vl.importlib, "import_module", fake_import)
    monkeypatch.setattr(qwen3_vl, "_patch_rotary_signature", lambda: None)
    monkeypatch.setattr(qwen3_vl, "_patch_model_forward_and_rope_index", lambda: None)
    monkeypatch.setattr(qwen3_vl, "_patch_allgather_vision_embeddings_kwarg", lambda: None)

    qwen3_vl.install_qwen3_vl_packed_mrope_patch()
    wrapped = FakeTransformerBlock._deepstack_process
    qwen3_vl.install_qwen3_vl_packed_mrope_patch()

    assert FakeTransformerBlock._deepstack_process is wrapped

    input_tensor = torch.arange(6.0).reshape(2, 3).requires_grad_()
    output = FakeTransformerBlock()._deepstack_process(input_tensor)

    assert torch.equal(output, input_tensor.transpose(0, 1))
    assert output._base is None
    assert output.requires_grad
    assert helper_calls == [(True, True)]

    output.sum().backward()
    assert torch.equal(input_tensor.grad, torch.ones_like(input_tensor))


def test_deepstack_patch_is_quiet_when_megatron_bridge_is_absent(monkeypatch, caplog):
    def fake_import(name):
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(qwen3_vl.importlib, "import_module", fake_import)
    monkeypatch.setattr(qwen3_vl, "_DEEPSTACK_DRIFT_WARNED", False, raising=False)

    with caplog.at_level(logging.WARNING, logger=qwen3_vl.__name__):
        qwen3_vl._patch_deepstack_output_view()
        qwen3_vl._patch_deepstack_output_view()

    assert caplog.records == []


@pytest.mark.parametrize(
    ("missing_component", "block_class", "viewless_helper"),
    [
        ("Qwen3VLTransformerBlock", None, lambda **kwargs: kwargs["inp"]),
        ("make_viewless_tensor", type("Block", (), {"_deepstack_process": lambda self, value: value}), None),
        ("_deepstack_process", type("Block", (), {}), lambda **kwargs: kwargs["inp"]),
    ],
)
def test_installed_bridge_deepstack_drift_warns_once(
    monkeypatch, caplog, missing_component, block_class, viewless_helper
):
    bridge_root = SimpleNamespace()
    block_module = SimpleNamespace(Qwen3VLTransformerBlock=block_class)
    core_utils = SimpleNamespace(make_viewless_tensor=viewless_helper)

    def fake_import(name):
        modules = {
            "megatron.bridge": bridge_root,
            "megatron.bridge.models.qwen_vl.modelling_qwen3_vl.transformer_block": block_module,
            "megatron.core.utils": core_utils,
        }
        return modules[name]

    monkeypatch.setattr(qwen3_vl.importlib, "import_module", fake_import)
    monkeypatch.setattr(qwen3_vl, "_DEEPSTACK_DRIFT_WARNED", False, raising=False)

    with caplog.at_level(logging.WARNING, logger=qwen3_vl.__name__):
        qwen3_vl._patch_deepstack_output_view()
        qwen3_vl._patch_deepstack_output_view()

    warnings = [record for record in caplog.records if record.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Megatron Bridge" in warnings[0].message
    assert missing_component in warnings[0].message
