import math
import sys
import types
from argparse import Namespace

import pytest
import torch
import torch.nn.functional as F

from miles.backends.training_utils.loss_hub.math_utils import _mask_padded_vocab_logits, compute_log_probs
from miles.backends.training_utils.loss_hub.rkld_dagger import (
    compute_explicit_dagger_loss,
    explicit_topk_cross_entropy_per_token,
)

from .loss_test_utils import make_parallel_state


def _args(**overrides) -> Namespace:
    values = {
        "qkv_format": "thd",
        "rollout_temperature": 1.0,
        "allgather_cp": False,
        "true_on_policy_mode": True,
        "bf16": False,
        "fp16": False,
        "vocab_size": 4,
        "opd_dagger_top_k": 2,
    }
    values.update(overrides)
    return Namespace(**values)


def test_explicit_topk_ce_uses_raw_teacher_mass_and_updates_unsampled_candidates():
    # The rollout action is irrelevant to direct CE. Token 0 can be viewed as
    # sampled, while teacher candidates 1 and 2 still receive dense gradients.
    logits = torch.tensor([[3.0, -2.0, -3.0, -4.0]], requires_grad=True)
    teacher_ids = torch.tensor([[1, 2]], dtype=torch.long)
    teacher_probs = torch.tensor([[0.6, 0.3]], dtype=torch.float32)
    teacher_log_probs = teacher_probs.log()

    outputs = explicit_topk_cross_entropy_per_token(
        logits,
        teacher_ids,
        teacher_log_probs,
        torch.ones((1, 2), dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
        tp_group=None,
        true_on_policy_mode=True,
        vocab_size=4,
    )

    student_log_probs = torch.log_softmax(logits, dim=-1)
    expected = -(teacher_probs * student_log_probs[:, [1, 2]]).sum(dim=-1)
    normalized = -((teacher_probs / teacher_probs.sum(dim=-1, keepdim=True)) * student_log_probs[:, [1, 2]]).sum(
        dim=-1
    )
    torch.testing.assert_close(outputs["per_token_loss"], expected)
    assert not torch.allclose(outputs["per_token_loss"], normalized)
    torch.testing.assert_close(outputs["teacher_topk_mass"], torch.tensor([0.9]))

    outputs["per_token_loss"].sum().backward()
    assert logits.grad[0, 1] < 0
    assert logits.grad[0, 2] < 0
    assert logits.grad[0, 0] > 0


def test_explicit_topk_ce_detaches_teacher_targets():
    logits = torch.tensor([[1.0, 0.0, -1.0, -2.0]], requires_grad=True)
    teacher_log_probs = torch.tensor([[math.log(0.6), math.log(0.3)]], requires_grad=True)

    outputs = explicit_topk_cross_entropy_per_token(
        logits,
        torch.tensor([[1, 2]], dtype=torch.long),
        teacher_log_probs,
        torch.ones((1, 2), dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
        tp_group=None,
        true_on_policy_mode=True,
        vocab_size=4,
    )
    outputs["per_token_loss"].sum().backward()

    assert teacher_log_probs.grad is None
    assert logits.grad is not None


class _FakeProcessGroup:
    def __init__(self, rank: int, size: int):
        self._rank = rank
        self._size = size

    def rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._size


def test_mask_padded_vocab_logits_is_identity_without_an_explicit_vocab_size():
    logits = torch.randn(2, 4, requires_grad=True)

    assert _mask_padded_vocab_logits(logits, _FakeProcessGroup(rank=1, size=2), vocab_size=None) is logits


def test_mask_padded_vocab_logits_masks_only_dummy_ids_on_the_last_tp_shard():
    logits = torch.tensor([[1.0, 2.0, 100.0, 200.0]], requires_grad=True)
    masked = _mask_padded_vocab_logits(logits, _FakeProcessGroup(rank=1, size=2), vocab_size=6)

    torch.testing.assert_close(masked[:, :2], logits[:, :2])
    assert (masked[:, 2:] == torch.finfo(logits.dtype).min).all()

    masked.sum().backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[1.0, 1.0, 0.0, 0.0]]))


def test_compute_log_probs_excludes_padded_vocab_before_fused_ce(monkeypatch):
    fused_module = types.ModuleType("megatron.core.fusions.fused_cross_entropy")

    def fake_fused_cross_entropy(logits, targets, _process_group):
        return F.cross_entropy(logits.squeeze(1), targets.squeeze(1), reduction="none")

    fused_module.fused_vocab_parallel_cross_entropy = fake_fused_cross_entropy
    monkeypatch.setitem(sys.modules, "megatron.core.fusions.fused_cross_entropy", fused_module)

    logits = torch.tensor([[0.0, 1.0, 50.0, 60.0]], requires_grad=True)
    log_prob = compute_log_probs(logits, torch.tensor([1]), None, vocab_size=2)
    expected = torch.log_softmax(logits[:, :2], dim=-1)[:, 1]

    torch.testing.assert_close(log_prob, expected)
    (-log_prob).sum().backward()
    torch.testing.assert_close(logits.grad[:, 2:], torch.zeros_like(logits.grad[:, 2:]))


def test_mask_padded_vocab_logits_rejects_vocab_larger_than_sharded_logits():
    with pytest.raises(ValueError, match=r"vocab_size=9 exceeds.*4 x 2 = 8"):
        _mask_padded_vocab_logits(torch.zeros((1, 4)), _FakeProcessGroup(rank=0, size=2), vocab_size=9)


def test_explicit_topk_ce_masks_padding_and_inactive_response_rows_before_exp():
    logits = torch.tensor([[1.0, 0.0, -1.0, -2.0], [0.5, -0.5, 0.0, -1.0]], requires_grad=True)
    outputs = explicit_topk_cross_entropy_per_token(
        logits,
        torch.tensor([[1, 0], [2, 3]], dtype=torch.long),
        torch.tensor([[-0.2, -torch.inf], [-0.3, -0.4]], dtype=torch.float32),
        torch.tensor([[True, False], [True, True]], dtype=torch.bool),
        torch.tensor([True, False]),
        tp_group=None,
        true_on_policy_mode=True,
        vocab_size=4,
    )

    assert torch.isfinite(outputs["per_token_loss"]).all()
    assert outputs["per_token_loss"][1].item() == 0.0
    torch.testing.assert_close(outputs["teacher_topk_mass"], torch.tensor([math.exp(-0.2), 0.0]))
    torch.testing.assert_close(outputs["valid_candidates"], torch.tensor([1.0, 0.0]))


def test_explicit_topk_ce_all_masked_keeps_a_zero_gradient_graph():
    logits = torch.randn(2, 4, requires_grad=True)
    outputs = explicit_topk_cross_entropy_per_token(
        logits,
        torch.zeros((2, 2), dtype=torch.long),
        torch.full((2, 2), -torch.inf),
        torch.zeros((2, 2), dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
        tp_group=None,
        true_on_policy_mode=True,
        vocab_size=4,
    )

    loss = outputs["per_token_loss"].sum()
    assert loss.item() == 0.0
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_compute_explicit_dagger_loss_aligns_response_logits_and_reports_metrics():
    make_parallel_state()
    args = _args()
    logits = torch.tensor(
        [[[2.0, 0.0, -1.0, -2.0], [0.0, 2.0, -1.0, -2.0], [9.0, 9.0, 9.0, 9.0]]],
        requires_grad=True,
    )
    teacher_probs = torch.tensor([[0.6, 0.3], [0.7, 0.2]], dtype=torch.float32)
    batch = {
        "unconcat_tokens": [torch.tensor([0, 3, 2], dtype=torch.long)],
        "total_lengths": [3],
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
        "teacher_topk_token_ids": [torch.tensor([[1, 2], [0, 2]], dtype=torch.long)],
        "teacher_topk_log_probs": [teacher_probs.log()],
        "teacher_topk_valid_mask": [torch.ones((2, 2), dtype=torch.bool)],
    }

    explicit_ce, metrics = compute_explicit_dagger_loss(args, batch, logits, lambda values: values.mean())

    response_log_probs = torch.log_softmax(logits[0, :2], dim=-1)
    expected_per_token = torch.stack(
        [
            -(teacher_probs[0] * response_log_probs[0, [1, 2]]).sum(),
            -(teacher_probs[1] * response_log_probs[1, [0, 2]]).sum(),
        ]
    )
    torch.testing.assert_close(explicit_ce, expected_per_token.mean())
    torch.testing.assert_close(metrics["explicit_ce"], expected_per_token.mean())
    torch.testing.assert_close(metrics["teacher_topk_mass"], torch.tensor(0.9))
    torch.testing.assert_close(metrics["valid_candidates_mean"], torch.tensor(2.0))
    torch.testing.assert_close(metrics["valid_position_ratio"], torch.tensor(1.0))


def test_compute_explicit_dagger_loss_rejects_configured_width_mismatch():
    make_parallel_state()
    batch = {
        "unconcat_tokens": [torch.tensor([0, 1])],
        "total_lengths": [2],
        "response_lengths": [1],
        "loss_masks": [torch.ones(1)],
        "teacher_topk_token_ids": [torch.tensor([[1]], dtype=torch.long)],
        "teacher_topk_log_probs": [torch.tensor([[-0.2]])],
        "teacher_topk_valid_mask": [torch.ones((1, 1), dtype=torch.bool)],
    }

    with pytest.raises(ValueError, match="configured K=2"):
        compute_explicit_dagger_loss(_args(), batch, torch.zeros((1, 2, 4)), lambda values: values.mean())
