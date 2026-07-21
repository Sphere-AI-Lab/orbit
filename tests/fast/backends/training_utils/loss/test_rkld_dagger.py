import math
import sys
import types
from argparse import Namespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from miles.backends.training_utils.loss_hub.math_utils import (
    _local_full_logsumexp,
    _mask_padded_vocab_logits,
    compute_log_probs,
    vocab_parallel_topk_rest_cross_entropy,
)
from miles.backends.training_utils.loss_hub.rkld_dagger import (
    compute_explicit_dagger_loss,
    compute_topk_rest_dagger_loss,
    explicit_topk_cross_entropy_per_token,
    topk_rest_cross_entropy_per_token,
)

from .loss_test_utils import make_parallel_state
from .rkld_dagger_test_utils import dense_topk_rest_oracle


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


def _run_stable_topk_rest_tp2_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        tp_size = 2
        tp_group = None
        tp_rank = None
        dp_rank = None
        for group_dp_rank, start in enumerate(range(0, world_size, tp_size)):
            ranks = list(range(start, start + tp_size))
            group = dist.new_group(ranks=ranks)
            if rank in ranks:
                tp_group = group
                tp_rank = rank - start
                dp_rank = group_dp_rank
        assert tp_group is not None and tp_rank is not None and dp_rank is not None

        vocab_size = 5
        local_vocab_size = 3
        full_logits = torch.tensor(
            [
                [0.4, -0.3, 0.8, -0.7, 0.1, 50.0],
                [-0.5, 0.6, -0.2, 0.9, -0.4, 50.0],
                [0.2, -0.8, 0.3, -0.1, 0.7, 50.0],
            ],
            dtype=torch.float64,
        )
        full_logits[:, :vocab_size] += dp_rank * torch.tensor(
            [[0.0, 0.4, -0.3, 0.2, -0.1]],
            dtype=torch.float64,
        )
        full_logits[:, vocab_size:] += dp_rank
        teacher_ids = torch.tensor([[0, 4], [2, 3], [1, 4]], dtype=torch.long)
        candidate_mask = torch.tensor(
            [[True, True], [True, True], [True, False]],
            dtype=torch.bool,
        )
        teacher_probs = torch.tensor(
            [[0.45, 0.35], [0.50, 0.20], [0.40, 0.00]],
            dtype=torch.float64,
        )
        teacher_rest_mass = 1.0 - teacher_probs.sum(dim=-1)

        local_start = tp_rank * local_vocab_size
        local_end = local_start + local_vocab_size
        local_logits = full_logits[:, local_start:local_end].clone().requires_grad_()
        actual = vocab_parallel_topk_rest_cross_entropy(
            local_logits,
            teacher_ids,
            teacher_probs,
            teacher_rest_mass,
            candidate_mask,
            tp_group,
            vocab_size=vocab_size,
            row_chunk_size=1,
        )
        actual["per_token_loss"].sum().backward()

        oracle_logits = full_logits[:, :vocab_size].clone().requires_grad_()
        teacher_log_probs = torch.where(
            candidate_mask,
            teacher_probs.log(),
            teacher_probs.new_full((), -torch.inf),
        )
        expected = dense_topk_rest_oracle(
            oracle_logits,
            teacher_ids,
            teacher_log_probs,
            candidate_mask,
            torch.ones(3, dtype=torch.bool),
            vocab_size=vocab_size,
        )
        expected["per_token_loss"].sum().backward()

        torch.testing.assert_close(
            actual["per_token_loss"],
            expected["per_token_loss"],
            atol=1e-10,
            rtol=1e-10,
        )
        expected_local_grad = torch.zeros_like(local_logits)
        real_start = min(local_start, vocab_size)
        real_end = min(local_end, vocab_size)
        if real_end > real_start:
            expected_local_grad[:, : real_end - real_start] = oracle_logits.grad[:, real_start:real_end]
        torch.testing.assert_close(local_logits.grad, expected_local_grad, atol=1e-10, rtol=1e-10)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Stable TP fast test requires the Gloo process-group backend.",
)
def test_stable_topk_rest_tp2_groups_match_dense_oracle_across_real_processes(tmp_path):
    mp.spawn(
        _run_stable_topk_rest_tp2_worker,
        args=(4, str(tmp_path / "stable_tp2_init")),
        nprocs=4,
        join=True,
    )


def test_dense_topk_rest_oracle_matches_hand_computed_loss_and_pdf_gradient():
    student_probs = torch.tensor([[0.2, 0.1, 0.3, 0.4]], dtype=torch.float64)
    logits = student_probs.log().requires_grad_()
    teacher_probs = torch.tensor([[0.6, 0.3]], dtype=torch.float64)

    outputs = dense_topk_rest_oracle(
        logits,
        torch.tensor([[0, 1]], dtype=torch.long),
        teacher_probs.log(),
        torch.ones((1, 2), dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
        vocab_size=4,
    )

    expected_loss = -(0.6 * math.log(0.2) + 0.3 * math.log(0.1) + 0.1 * math.log(0.7))
    torch.testing.assert_close(outputs["per_token_loss"], torch.tensor([expected_loss], dtype=torch.float64))

    outputs["per_token_loss"].sum().backward()
    expected_grad = torch.tensor(
        [[-0.4, -0.2, 0.3 / 0.7 * 0.6, 0.4 / 0.7 * 0.6]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(logits.grad, expected_grad)
    torch.testing.assert_close(logits.grad.sum(dim=-1), torch.zeros(1, dtype=torch.float64))


def test_dense_topk_rest_oracle_passes_gradcheck():
    logits = torch.tensor(
        [[0.3, -0.1, 0.8, -0.4], [-0.2, 0.6, 0.1, -0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    teacher_ids = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    teacher_probs = torch.tensor([[0.45, 0.35], [0.5, 0.2]], dtype=torch.float64)
    valid_mask = torch.ones((2, 2), dtype=torch.bool)
    response_mask = torch.ones(2, dtype=torch.bool)

    def oracle_loss(input_logits):
        return dense_topk_rest_oracle(
            input_logits,
            teacher_ids,
            teacher_probs.log(),
            valid_mask,
            response_mask,
            vocab_size=4,
        )["per_token_loss"]

    assert torch.autograd.gradcheck(oracle_loss, (logits,), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_dense_topk_rest_oracle_identity_has_entropy_loss_and_zero_gradient():
    student_probs = torch.tensor([[0.5, 0.2, 0.1, 0.2]], dtype=torch.float64)
    logits = student_probs.log().requires_grad_()
    teacher_probs = torch.tensor([[0.5, 0.2]], dtype=torch.float64)
    coarse_teacher = torch.tensor([0.5, 0.2, 0.3], dtype=torch.float64)

    outputs = dense_topk_rest_oracle(
        logits,
        torch.tensor([[0, 1]], dtype=torch.long),
        teacher_probs.log(),
        torch.ones((1, 2), dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
        vocab_size=4,
    )

    expected_entropy = -(coarse_teacher * coarse_teacher.log()).sum()
    torch.testing.assert_close(outputs["per_token_loss"], expected_entropy.unsqueeze(0))
    outputs["per_token_loss"].sum().backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_stable_topk_rest_gradient_step_reduces_coarse_cross_entropy():
    teacher_ids = torch.tensor([[1, 2]], dtype=torch.long)
    teacher_probs = torch.tensor([[0.55, 0.25]], dtype=torch.float64)
    teacher_rest_mass = 1.0 - teacher_probs.sum(dim=-1)
    candidate_mask = torch.ones((1, 2), dtype=torch.bool)
    logits = torch.tensor([[1.0, -1.0, 0.2, -0.4]], dtype=torch.float64, requires_grad=True)

    def coarse_ce(input_logits):
        return vocab_parallel_topk_rest_cross_entropy(
            input_logits,
            teacher_ids,
            teacher_probs,
            teacher_rest_mass,
            candidate_mask,
            None,
            vocab_size=4,
        )["per_token_loss"].sum()

    before = coarse_ce(logits)
    gradient = torch.autograd.grad(before, logits)[0]
    after = coarse_ce((logits.detach() - 0.1 * gradient).requires_grad_())
    assert after < before


def test_stable_topk_rest_tp1_matches_dense_oracle_loss_components_and_gradient():
    logits = torch.tensor(
        [[0.3, -0.1, 0.8, -0.4], [-0.2, 0.6, 0.1, -0.7]],
        dtype=torch.float64,
        requires_grad=True,
    )
    oracle_logits = logits.detach().clone().requires_grad_()
    teacher_ids = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    teacher_probs = torch.tensor([[0.45, 0.35], [0.5, 0.2]], dtype=torch.float64)
    candidate_mask = torch.ones((2, 2), dtype=torch.bool)
    response_mask = torch.ones(2, dtype=torch.bool)
    teacher_rest_mass = 1.0 - teacher_probs.sum(dim=-1)

    actual = vocab_parallel_topk_rest_cross_entropy(
        logits,
        teacher_ids,
        teacher_probs,
        teacher_rest_mass,
        candidate_mask,
        None,
        vocab_size=4,
        row_chunk_size=1,
    )
    expected = dense_topk_rest_oracle(
        oracle_logits,
        teacher_ids,
        teacher_probs.log(),
        candidate_mask,
        response_mask,
        vocab_size=4,
    )

    for key in ("per_token_loss", "explicit_ce", "rest_ce", "teacher_topk_mass", "teacher_rest_mass"):
        torch.testing.assert_close(actual[key], expected[key], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual["student_rest_mass"], expected["student_rest_mass"], atol=1e-10, rtol=1e-10)

    actual["per_token_loss"].sum().backward()
    expected["per_token_loss"].sum().backward()
    torch.testing.assert_close(logits.grad, oracle_logits.grad, atol=1e-10, rtol=1e-10)


def test_local_full_logsumexp_bounds_fp32_conversion_by_row_chunk(monkeypatch):
    logits = torch.randn(5, 7, dtype=torch.bfloat16)
    original_logsumexp = torch.logsumexp
    observed_shapes = []

    def recording_logsumexp(input_tensor, *args, **kwargs):
        observed_shapes.append(tuple(input_tensor.shape))
        return original_logsumexp(input_tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "logsumexp", recording_logsumexp)
    actual = _local_full_logsumexp(
        logits,
        local_valid_size=6,
        row_chunk_size=2,
        accumulation_dtype=torch.float32,
    )

    expected = original_logsumexp(logits[:, :6].float(), dim=-1)
    torch.testing.assert_close(actual, expected)
    assert actual.dtype == torch.float32
    assert observed_shapes == [(2, 6), (2, 6), (1, 6)]


def test_stable_topk_rest_tp1_custom_backward_passes_gradcheck():
    logits = torch.tensor([[0.3, -0.1, 0.8, -0.4]], dtype=torch.float64, requires_grad=True)
    teacher_ids = torch.tensor([[0, 2]], dtype=torch.long)
    teacher_probs = torch.tensor([[0.45, 0.35]], dtype=torch.float64)
    teacher_rest_mass = 1.0 - teacher_probs.sum(dim=-1)
    candidate_mask = torch.ones((1, 2), dtype=torch.bool)

    def stable_loss(input_logits):
        return vocab_parallel_topk_rest_cross_entropy(
            input_logits,
            teacher_ids,
            teacher_probs,
            teacher_rest_mass,
            candidate_mask,
            None,
            vocab_size=4,
            row_chunk_size=1,
        )["per_token_loss"]

    assert torch.autograd.gradcheck(stable_loss, (logits,), eps=1e-6, atol=1e-5, rtol=1e-3)


@pytest.mark.parametrize(
    ("dtype", "atol", "rtol"),
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 2e-2, 2e-2),
        (torch.float16, 5e-3, 5e-3),
    ],
)
def test_stable_topk_rest_low_precision_inputs_use_finite_fp32_accumulation(dtype, atol, rtol):
    base_logits = torch.tensor([[0.3, -0.1, 0.8, -0.4]], dtype=torch.float64)
    logits = base_logits.to(dtype).requires_grad_()
    oracle_logits = base_logits.clone().requires_grad_()
    teacher_ids = torch.tensor([[0, 2]], dtype=torch.long)
    teacher_probs = torch.tensor([[0.45, 0.35]], dtype=torch.float32)
    candidate_mask = torch.ones((1, 2), dtype=torch.bool)
    actual = vocab_parallel_topk_rest_cross_entropy(
        logits,
        teacher_ids,
        teacher_probs,
        1.0 - teacher_probs.sum(dim=-1),
        candidate_mask,
        None,
        vocab_size=4,
    )
    expected = dense_topk_rest_oracle(
        oracle_logits,
        teacher_ids,
        teacher_probs.double().log(),
        candidate_mask,
        torch.ones(1, dtype=torch.bool),
        vocab_size=4,
    )

    assert actual["per_token_loss"].dtype is torch.float32
    assert torch.isfinite(actual["per_token_loss"]).all()
    torch.testing.assert_close(
        actual["per_token_loss"],
        expected["per_token_loss"].float(),
        atol=atol,
        rtol=rtol,
    )
    actual["per_token_loss"].sum().backward()
    expected["per_token_loss"].sum().backward()
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(
        logits.grad.float(),
        oracle_logits.grad.float(),
        atol=atol,
        rtol=rtol,
    )


def test_stable_topk_rest_tp1_masks_padded_vocab_and_inactive_rows():
    logits = torch.tensor(
        [[0.2, 0.1, -0.3, 0.7, 50.0, 60.0], [0.5, -0.1, 0.4, -0.2, 70.0, 80.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    outputs = vocab_parallel_topk_rest_cross_entropy(
        logits,
        torch.tensor([[0, 3], [0, 0]], dtype=torch.long),
        torch.tensor([[0.5, 0.3], [0.0, 0.0]], dtype=torch.float64),
        torch.tensor([0.2, 1.0], dtype=torch.float64),
        torch.tensor([[True, True], [False, False]], dtype=torch.bool),
        None,
        vocab_size=4,
        row_chunk_size=1,
    )

    assert torch.isfinite(outputs["per_token_loss"]).all()
    assert outputs["per_token_loss"][1].item() == 0.0
    outputs["per_token_loss"].sum().backward()
    torch.testing.assert_close(logits.grad[1], torch.zeros_like(logits.grad[1]))
    torch.testing.assert_close(logits.grad[:, 4:], torch.zeros_like(logits.grad[:, 4:]))


def test_stable_topk_rest_zero_k_is_exact_zero_loss_and_gradient():
    logits = torch.randn(2, 4, dtype=torch.float64, requires_grad=True)
    outputs = vocab_parallel_topk_rest_cross_entropy(
        logits,
        torch.empty((2, 0), dtype=torch.long),
        torch.empty((2, 0), dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
        torch.empty((2, 0), dtype=torch.bool),
        None,
        vocab_size=4,
    )

    torch.testing.assert_close(outputs["per_token_loss"], torch.zeros(2, dtype=torch.float64))
    outputs["per_token_loss"].sum().backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_stable_topk_rest_handles_zero_and_near_zero_student_rest_mass():
    full_support_logits = torch.tensor([[0.3, -0.2]], dtype=torch.float64, requires_grad=True)
    full_support = vocab_parallel_topk_rest_cross_entropy(
        full_support_logits,
        torch.tensor([[0, 1]], dtype=torch.long),
        torch.tensor([[0.7, 0.3]], dtype=torch.float64),
        torch.zeros(1, dtype=torch.float64),
        torch.ones((1, 2), dtype=torch.bool),
        None,
        vocab_size=2,
    )
    assert torch.isfinite(full_support["per_token_loss"]).all()
    torch.testing.assert_close(full_support["student_rest_mass"], torch.zeros(1, dtype=torch.float64))
    full_support["per_token_loss"].sum().backward()
    assert torch.isfinite(full_support_logits.grad).all()

    tiny_rest_logits = torch.tensor([[20.0, 19.0, -40.0]], dtype=torch.float64, requires_grad=True)
    oracle_logits = tiny_rest_logits.detach().clone().requires_grad_()
    teacher_ids = torch.tensor([[0, 1]], dtype=torch.long)
    teacher_probs = torch.tensor([[0.6, 0.3]], dtype=torch.float64)
    candidate_mask = torch.ones((1, 2), dtype=torch.bool)
    actual = vocab_parallel_topk_rest_cross_entropy(
        tiny_rest_logits,
        teacher_ids,
        teacher_probs,
        torch.tensor([0.1], dtype=torch.float64),
        candidate_mask,
        None,
        vocab_size=3,
    )
    expected = dense_topk_rest_oracle(
        oracle_logits,
        teacher_ids,
        teacher_probs.log(),
        candidate_mask,
        torch.ones(1, dtype=torch.bool),
        vocab_size=3,
    )
    assert torch.isfinite(actual["per_token_loss"]).all()
    torch.testing.assert_close(actual["per_token_loss"], expected["per_token_loss"], atol=1e-10, rtol=1e-10)
    actual["per_token_loss"].sum().backward()
    expected["per_token_loss"].sum().backward()
    torch.testing.assert_close(tiny_rest_logits.grad, oracle_logits.grad, atol=1e-10, rtol=1e-10)


def test_topk_rest_full_vocab_coverage_discards_synthetic_rest_mass():
    logits = torch.tensor([[0.3, -0.2]], dtype=torch.float64, requires_grad=True)
    oracle_logits = logits.detach().clone().requires_grad_()
    # Simulate full-softmax log-probs whose represented mass is just below one.
    # Because both real vocabulary IDs are present, there is structurally no Rest bucket.
    teacher_probs = torch.tensor([[0.6, 0.4]], dtype=torch.float32) * (1.0 - 1e-6)
    teacher_log_probs = teacher_probs.log()
    outputs = topk_rest_cross_entropy_per_token(
        logits,
        torch.tensor([[0, 1]], dtype=torch.long),
        teacher_log_probs,
        torch.ones((1, 2), dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
        tp_group=None,
        vocab_size=2,
    )

    effective_teacher_probs = teacher_log_probs.exp().to(torch.float64)
    expected_loss = -(effective_teacher_probs * torch.log_softmax(oracle_logits, dim=-1)).sum(dim=-1)
    torch.testing.assert_close(outputs["teacher_rest_mass"], torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(outputs["rest_ce"], torch.zeros(1, dtype=torch.float64))
    torch.testing.assert_close(outputs["per_token_loss"], expected_loss, atol=1e-10, rtol=1e-10)
    assert torch.isfinite(outputs["teacher_entropy"]).all()

    outputs["per_token_loss"].sum().backward()
    expected_loss.sum().backward()
    torch.testing.assert_close(logits.grad, oracle_logits.grad, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_stable_topk_rest_low_precision_tiny_student_rest_stays_finite(dtype):
    logits = torch.tensor([[20.0, 19.0, -80.0]], dtype=dtype, requires_grad=True)
    teacher_probs = torch.tensor([[0.6, 0.3]], dtype=torch.float32)
    outputs = vocab_parallel_topk_rest_cross_entropy(
        logits,
        torch.tensor([[0, 1]], dtype=torch.long),
        teacher_probs,
        1.0 - teacher_probs.sum(dim=-1),
        torch.ones((1, 2), dtype=torch.bool),
        None,
        vocab_size=3,
    )

    assert torch.isfinite(outputs["per_token_loss"]).all()
    assert torch.isfinite(outputs["rest_ce"]).all()
    outputs["per_token_loss"].sum().backward()
    assert torch.isfinite(logits.grad).all()


def test_topk_rest_ce_from_teacher_log_probs_matches_dense_oracle_and_reports_mass():
    logits = torch.tensor([[0.3, -0.1, 0.8, -0.4]], dtype=torch.float64, requires_grad=True)
    oracle_logits = logits.detach().clone().requires_grad_()
    teacher_ids = torch.tensor([[0, 2]], dtype=torch.long)
    teacher_probs = torch.tensor([[0.45, 0.35]], dtype=torch.float32)
    valid_mask = torch.ones((1, 2), dtype=torch.bool)
    loss_mask = torch.ones(1, dtype=torch.bool)

    actual = topk_rest_cross_entropy_per_token(
        logits,
        teacher_ids,
        teacher_probs.log(),
        valid_mask,
        loss_mask,
        tp_group=None,
        vocab_size=4,
    )
    expected = dense_topk_rest_oracle(
        oracle_logits,
        teacher_ids,
        teacher_probs.double().log(),
        valid_mask,
        loss_mask,
        vocab_size=4,
    )

    torch.testing.assert_close(actual["per_token_loss"], expected["per_token_loss"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(actual["teacher_topk_mass"], torch.tensor([0.8], dtype=torch.float64))
    torch.testing.assert_close(actual["teacher_rest_mass"], torch.tensor([0.2], dtype=torch.float64))
    coarse_teacher = torch.tensor([[0.45, 0.35, 0.2]], dtype=torch.float64)
    expected_entropy = -(coarse_teacher * coarse_teacher.log()).sum(dim=-1)
    torch.testing.assert_close(actual["teacher_entropy"], expected_entropy, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        actual["coarse_kl"],
        expected["per_token_loss"] - expected_entropy,
        atol=1e-6,
        rtol=1e-6,
    )
    expected_rest_error = (expected["student_rest_mass"] - expected["teacher_rest_mass"]).abs()
    torch.testing.assert_close(actual["rest_mass_abs_error"], expected_rest_error, atol=1e-6, rtol=1e-6)

    actual["per_token_loss"].sum().backward()
    expected["per_token_loss"].sum().backward()
    torch.testing.assert_close(logits.grad, oracle_logits.grad, atol=1e-6, rtol=1e-6)


def test_topk_rest_teacher_rest_uses_expm1_near_unit_mass_without_zeroing_it():
    teacher_log_probs = torch.tensor([[math.log1p(-1e-6)]], dtype=torch.float32)
    outputs = topk_rest_cross_entropy_per_token(
        torch.zeros((1, 4), dtype=torch.float32),
        torch.tensor([[0]], dtype=torch.long),
        teacher_log_probs,
        torch.ones((1, 1), dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
        tp_group=None,
        vocab_size=4,
    )

    expected_rest = -torch.expm1(teacher_log_probs.squeeze(0))
    assert outputs["teacher_rest_mass"].item() > 0.0
    torch.testing.assert_close(outputs["teacher_rest_mass"], expected_rest)


def test_topk_rest_tolerates_small_mass_overshoot_and_sets_rest_to_zero():
    teacher_probs = torch.tensor([[0.6, 0.400001]], dtype=torch.float32)
    outputs = topk_rest_cross_entropy_per_token(
        torch.zeros((1, 4), dtype=torch.float32),
        torch.tensor([[0, 1]], dtype=torch.long),
        teacher_probs.log(),
        torch.ones((1, 2), dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
        tp_group=None,
        vocab_size=4,
    )

    assert teacher_probs.sum().item() > 1.0
    torch.testing.assert_close(outputs["teacher_rest_mass"], torch.zeros(1))
    assert torch.isfinite(outputs["per_token_loss"]).all()


def test_topk_rest_coarse_kl_is_zero_at_the_coarse_teacher_distribution():
    student_probs = torch.tensor([[0.5, 0.2, 0.1, 0.2]], dtype=torch.float32)
    teacher_probs = torch.tensor([[0.5, 0.2]], dtype=torch.float32)
    outputs = topk_rest_cross_entropy_per_token(
        student_probs.log().requires_grad_(),
        torch.tensor([[0, 1]], dtype=torch.long),
        teacher_probs.log(),
        torch.ones((1, 2), dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
        tp_group=None,
        vocab_size=4,
    )

    torch.testing.assert_close(outputs["per_token_loss"], outputs["teacher_entropy"], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(outputs["coarse_kl"], torch.zeros(1), atol=1e-6, rtol=1e-6)


def test_topk_rest_empty_response_has_an_empty_finite_graph():
    logits = torch.empty((0, 4), dtype=torch.float32, requires_grad=True)
    outputs = topk_rest_cross_entropy_per_token(
        logits,
        torch.empty((0, 2), dtype=torch.long),
        torch.empty((0, 2), dtype=torch.float32),
        torch.empty((0, 2), dtype=torch.bool),
        torch.empty((0,), dtype=torch.bool),
        tp_group=None,
        vocab_size=4,
    )

    assert outputs["per_token_loss"].numel() == 0
    outputs["per_token_loss"].sum().backward()
    assert logits.grad is not None and logits.grad.numel() == 0


def test_topk_rest_ce_detaches_teacher_log_probs_and_supports_partial_k():
    logits = torch.tensor([[0.3, -0.1, 0.8, -0.4]], dtype=torch.float64, requires_grad=True)
    teacher_log_probs = torch.tensor([[math.log(0.6), -torch.inf]], requires_grad=True)
    outputs = topk_rest_cross_entropy_per_token(
        logits,
        torch.tensor([[2, 0]], dtype=torch.long),
        teacher_log_probs,
        torch.tensor([[True, False]], dtype=torch.bool),
        torch.ones(1, dtype=torch.bool),
        tp_group=None,
        vocab_size=4,
    )

    assert torch.isfinite(outputs["per_token_loss"]).all()
    torch.testing.assert_close(outputs["teacher_topk_mass"], torch.tensor([0.6], dtype=torch.float64))
    torch.testing.assert_close(outputs["teacher_rest_mass"], torch.tensor([0.4], dtype=torch.float64))
    outputs["per_token_loss"].sum().backward()
    assert teacher_log_probs.grad is None
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    ("teacher_ids", "teacher_probs", "error"),
    [
        ([1, 1], [0.4, 0.3], "duplicate token IDs"),
        ([1, 2], [0.7, 0.4], "probability mass exceeds 1"),
    ],
)
def test_topk_rest_ce_rejects_invalid_teacher_protocol(teacher_ids, teacher_probs, error):
    with pytest.raises(ValueError, match=error):
        topk_rest_cross_entropy_per_token(
            torch.zeros((1, 4), dtype=torch.float32),
            torch.tensor([teacher_ids], dtype=torch.long),
            torch.tensor([teacher_probs], dtype=torch.float32).log(),
            torch.ones((1, 2), dtype=torch.bool),
            torch.ones(1, dtype=torch.bool),
            tp_group=None,
            vocab_size=4,
        )


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


def test_compute_topk_rest_dagger_loss_aligns_response_logits_and_reports_components():
    make_parallel_state()
    args = _args()
    logits = torch.tensor(
        [[[2.0, 0.0, -1.0, -2.0], [0.0, 2.0, -1.0, -2.0], [9.0, 9.0, 9.0, 9.0]]],
        requires_grad=True,
    )
    teacher_probs = torch.tensor([[0.6, 0.3], [0.7, 0.2]], dtype=torch.float32)
    teacher_ids = torch.tensor([[1, 2], [0, 2]], dtype=torch.long)
    batch = {
        "unconcat_tokens": [torch.tensor([0, 3, 2], dtype=torch.long)],
        "total_lengths": [3],
        "response_lengths": [2],
        "loss_masks": [torch.ones(2)],
        "teacher_topk_token_ids": [teacher_ids],
        "teacher_topk_log_probs": [teacher_probs.log()],
        "teacher_topk_valid_mask": [torch.ones((2, 2), dtype=torch.bool)],
    }

    cross_entropy, metrics = compute_topk_rest_dagger_loss(args, batch, logits, lambda values: values.mean())
    expected = dense_topk_rest_oracle(
        logits[0, :2],
        teacher_ids,
        teacher_probs.double().log(),
        torch.ones((2, 2), dtype=torch.bool),
        torch.ones(2, dtype=torch.bool),
        vocab_size=4,
    )

    torch.testing.assert_close(cross_entropy, expected["per_token_loss"].float().mean(), atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(metrics["cross_entropy"], cross_entropy.detach())
    torch.testing.assert_close(metrics["explicit_ce"] + metrics["rest_ce"], metrics["cross_entropy"])
    torch.testing.assert_close(metrics["teacher_entropy"] + metrics["coarse_kl"], metrics["cross_entropy"])
    torch.testing.assert_close(metrics["teacher_topk_mass"], torch.tensor(0.9))
    torch.testing.assert_close(metrics["teacher_rest_mass"], torch.tensor(0.1))
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
