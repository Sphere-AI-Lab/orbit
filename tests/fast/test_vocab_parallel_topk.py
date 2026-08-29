import os

import pytest
import torch
import torch.distributed as dist

from tests.fast.dist_utils import init_gloo, run_multiprocess

from orbit.opd.vocab_parallel import (
    compute_vocab_parallel_topk_log_probs,
    compute_vocab_parallel_topk_log_probs_and_entropy,
    vocab_parallel_topk_indices,
)
from miles.utils.ppo_utils import _gather_true_on_policy_full_logits


def test_single_process_matches_log_softmax_gather_values_and_grad() -> None:
    torch.manual_seed(0)
    v, r, k = 11, 4, 3
    logits = torch.randn(r, v, requires_grad=True)
    ids = torch.randint(0, v, (r, k))

    out = compute_vocab_parallel_topk_log_probs(logits, ids, None)
    expected = torch.log_softmax(logits, -1).gather(-1, ids)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)

    out.sum().backward()
    grad = logits.grad
    logits2 = logits.detach().clone().requires_grad_(True)
    torch.log_softmax(logits2, -1).gather(-1, ids).sum().backward()
    torch.testing.assert_close(grad, logits2.grad, rtol=0, atol=0)


def test_single_process_excludes_padded_vocab_from_values_entropy_and_grad() -> None:
    logits = torch.tensor(
        [[2.0, 1.0, 8.0, 9.0], [-1.0, 3.0, 7.0, 6.0]],
        requires_grad=True,
    )
    ids = torch.tensor([[0, 1], [1, 0]])

    selected, entropy = compute_vocab_parallel_topk_log_probs_and_entropy(
        logits,
        ids,
        vocab_size=2,
    )
    reference_logits = logits.detach()[..., :2].clone().requires_grad_(True)
    reference_log_probs = torch.log_softmax(reference_logits, dim=-1)
    reference_selected = reference_log_probs.gather(-1, ids)
    reference_entropy = -(reference_log_probs.exp() * reference_log_probs).sum(dim=-1)

    torch.testing.assert_close(selected, reference_selected, rtol=0, atol=0)
    torch.testing.assert_close(entropy, reference_entropy, rtol=0, atol=0)

    (selected.sum() + entropy.sum()).backward()
    (reference_selected.sum() + reference_entropy.sum()).backward()
    torch.testing.assert_close(logits.grad[..., :2], reference_logits.grad, rtol=0, atol=0)
    torch.testing.assert_close(logits.grad[..., 2:], torch.zeros_like(logits.grad[..., 2:]), rtol=0, atol=0)


def test_single_process_topk_indices_excludes_padded_vocab() -> None:
    logits = torch.tensor([[1.0, 2.0, 100.0, 99.0]])
    indices = vocab_parallel_topk_indices(logits, k=4, vocab_start=0, group=None, vocab_size=2)
    torch.testing.assert_close(indices, torch.tensor([[1, 0]]))


def _run_tp_case(rank: int, world_size: int) -> None:
    torch.manual_seed(42)
    r, k, v = 4, 3, 12
    shard = v // world_size
    full_logits = torch.randn(r, v)
    # ids span both single-shard and cross-shard rows.
    global_ids = torch.tensor(
        [
            [0, 2, 5],  # shard 0 only
            [6, 9, 11],  # shard 1 only
            [1, 7, 10],  # cross-shard
            [3, 4, 8],  # cross-shard
        ]
    )
    weights = torch.arange(1, r * k + 1, dtype=torch.float32).reshape(r, k)

    vocab_start = rank * shard
    vocab_end = vocab_start + shard
    local_shard = full_logits[:, vocab_start:vocab_end].clone().requires_grad_(True)

    out = compute_vocab_parallel_topk_log_probs(local_shard, global_ids, dist.group.WORLD)

    # The TP path's manual log-sum-exp differs in floating-point op order from the
    # `process_group=None` reference's fused `torch.log_softmax`, so values/grads are
    # mathematically but not bit-for-bit identical -- default assert_close tolerances
    # comfortably separate that noise (~1e-7) from the ×tp_size bug this test targets.
    reference_logits = full_logits.clone().requires_grad_(True)
    reference_out = compute_vocab_parallel_topk_log_probs(reference_logits, global_ids, None)
    torch.testing.assert_close(out, reference_out)

    (out * weights).sum().backward()
    (reference_out * weights).sum().backward()

    gathered = [torch.empty_like(local_shard.grad) for _ in range(world_size)]
    dist.all_gather(gathered, local_shard.grad.contiguous(), group=dist.group.WORLD)
    assembled_grad = torch.cat(gathered, dim=-1)

    torch.testing.assert_close(assembled_grad, reference_logits.grad)
    torch.testing.assert_close(local_shard.grad, reference_logits.grad[:, vocab_start:vocab_end])


def _worker_tp_gradient_equality(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        _run_tp_case(rank, world_size)
    finally:
        dist.destroy_process_group()


def test_tp2_gloo_shard_grad_matches_single_process_reference() -> None:
    run_multiprocess(_worker_tp_gradient_equality, world_size=2)


def _run_tp_padded_vocab_case(rank: int, world_size: int) -> None:
    torch.manual_seed(91)
    rows, padded_vocab, real_vocab = 3, 12, 9
    shard = padded_vocab // world_size
    full_logits = torch.randn(rows, padded_vocab)
    # Make the padding maximally tempting: including it in normalization or
    # diagnostics would fail decisively rather than by round-off.
    full_logits[..., real_vocab:] = 20.0
    global_ids = torch.tensor([[0, 8], [3, 7], [1, 5]])
    weights = torch.tensor([[1.0, -0.5], [0.25, 2.0], [-1.0, 0.75]])

    start = rank * shard
    end = start + shard
    local = full_logits[..., start:end].clone().requires_grad_(True)
    selected, entropy = compute_vocab_parallel_topk_log_probs_and_entropy(
        local,
        global_ids,
        dist.group.WORLD,
        vocab_size=real_vocab,
    )

    reference = full_logits[..., :real_vocab].clone().requires_grad_(True)
    reference_log_probs = torch.log_softmax(reference, dim=-1)
    reference_selected = reference_log_probs.gather(-1, global_ids)
    reference_entropy = -(reference_log_probs.exp() * reference_log_probs).sum(dim=-1)
    torch.testing.assert_close(selected, reference_selected)
    torch.testing.assert_close(entropy, reference_entropy)

    loss = (selected * weights).sum() + entropy.sum()
    reference_loss = (reference_selected * weights).sum() + reference_entropy.sum()
    loss.backward()
    reference_loss.backward()

    gathered = [torch.empty_like(local.grad) for _ in range(world_size)]
    dist.all_gather(gathered, local.grad.contiguous(), group=dist.group.WORLD)
    assembled = torch.cat(gathered, dim=-1)
    expected = torch.cat(
        [reference.grad, torch.zeros(rows, padded_vocab - real_vocab)],
        dim=-1,
    )
    torch.testing.assert_close(assembled, expected)

    topk_ids = vocab_parallel_topk_indices(
        local.detach(),
        k=real_vocab,
        vocab_start=start,
        group=dist.group.WORLD,
        vocab_size=real_vocab,
    )
    expected_topk_ids = torch.topk(full_logits[..., :real_vocab], k=real_vocab, dim=-1).indices
    torch.testing.assert_close(topk_ids, expected_topk_ids)


def _worker_tp_padded_vocab(rank: int, world_size: int, port: int) -> None:
    init_gloo(rank, world_size, port=port)
    try:
        _run_tp_padded_vocab_case(rank, world_size)
    finally:
        dist.destroy_process_group()


def test_tp2_gloo_excludes_padded_vocab_from_values_entropy_diagnostics_and_grad() -> None:
    run_multiprocess(_worker_tp_padded_vocab, world_size=2)


def _worker_tp4_nccl_bf16_padding_only_shard(rank: int, world_size: int, port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        torch.manual_seed(2026)
        rows, shard, real_vocab = 2, 4, 9
        padded_vocab = shard * world_size
        full_logits = torch.randn(rows, padded_vocab, dtype=torch.bfloat16, device=f"cuda:{rank}")
        full_logits[..., real_vocab:] = 20
        ids = torch.tensor([[0, 8, 4], [7, 1, 5]], device=f"cuda:{rank}")
        weights = torch.tensor([[1.0, -0.5, 0.25], [2.0, -1.0, 0.75]], device=f"cuda:{rank}")

        start = rank * shard
        local = full_logits[..., start : start + shard].clone().requires_grad_(True)
        selected, entropy = compute_vocab_parallel_topk_log_probs_and_entropy(
            local,
            ids,
            dist.group.WORLD,
            vocab_size=real_vocab,
        )

        reference = full_logits[..., :real_vocab].float().clone().requires_grad_(True)
        reference_log_probs = torch.log_softmax(reference, dim=-1)
        reference_selected = reference_log_probs.gather(-1, ids)
        reference_entropy = -(reference_log_probs.exp() * reference_log_probs).sum(dim=-1)
        torch.testing.assert_close(selected, reference_selected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(entropy, reference_entropy, rtol=1e-5, atol=1e-5)

        ((selected * weights).sum() + entropy.sum()).backward()
        ((reference_selected * weights).sum() + reference_entropy.sum()).backward()
        gathered = [torch.empty_like(local.grad) for _ in range(world_size)]
        dist.all_gather(gathered, local.grad.contiguous())
        assembled = torch.cat(gathered, dim=-1)
        expected = torch.cat(
            [
                reference.grad.to(torch.bfloat16),
                torch.zeros(
                    rows,
                    padded_vocab - real_vocab,
                    dtype=torch.bfloat16,
                    device=local.device,
                ),
            ],
            dim=-1,
        )
        torch.testing.assert_close(assembled, expected, rtol=1e-2, atol=1e-3)

        diagnostic_ids = vocab_parallel_topk_indices(
            local.detach(),
            k=real_vocab,
            vocab_start=start,
            group=dist.group.WORLD,
            vocab_size=real_vocab,
        )
        expected_ids = torch.topk(full_logits[..., :real_vocab], k=real_vocab, dim=-1).indices
        torch.testing.assert_close(diagnostic_ids, expected_ids)

        # rank 3 owns no real vocabulary columns in this 9-of-16 layout.
        if rank == 3:
            torch.testing.assert_close(local.grad, torch.zeros_like(local.grad), rtol=0, atol=0)

        # The true-on-policy branch intentionally keeps native BF16 and gathers
        # the real full vocabulary. Exercise its replicated-loss backward on the
        # same four-rank layout, including the padding-only rank.
        local_true = full_logits[..., start : start + shard].clone().requires_grad_(True)
        gathered_full = _gather_true_on_policy_full_logits(
            local_true,
            dist.group.WORLD,
            vocab_size=real_vocab,
        )
        true_log_probs = torch.log_softmax(gathered_full, dim=-1)
        true_selected = true_log_probs.gather(-1, ids)
        true_entropy = -(true_log_probs.exp() * true_log_probs).sum(dim=-1)

        true_reference = full_logits[..., :real_vocab].clone().requires_grad_(True)
        true_reference_log_probs = torch.log_softmax(true_reference, dim=-1)
        true_reference_selected = true_reference_log_probs.gather(-1, ids)
        true_reference_entropy = -(true_reference_log_probs.exp() * true_reference_log_probs).sum(dim=-1)
        torch.testing.assert_close(true_selected, true_reference_selected, rtol=0, atol=0)
        torch.testing.assert_close(true_entropy, true_reference_entropy, rtol=0, atol=0)

        ((true_selected * weights).sum() + true_entropy.sum()).backward()
        ((true_reference_selected * weights).sum() + true_reference_entropy.sum()).backward()
        true_gathered_grads = [torch.empty_like(local_true.grad) for _ in range(world_size)]
        dist.all_gather(true_gathered_grads, local_true.grad.contiguous())
        true_assembled_grad = torch.cat(true_gathered_grads, dim=-1)
        true_expected_grad = torch.cat(
            [
                true_reference.grad,
                torch.zeros(
                    rows,
                    padded_vocab - real_vocab,
                    dtype=torch.bfloat16,
                    device=local.device,
                ),
            ],
            dim=-1,
        )
        torch.testing.assert_close(true_assembled_grad, true_expected_grad, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four CUDA GPUs")
def test_tp4_nccl_bf16_excludes_padding_with_completely_padding_only_shard() -> None:
    run_multiprocess(_worker_tp4_nccl_bf16_padding_only_shard, world_size=4)


def test_verl_closed_form_forward_kl_gradient_oracle() -> None:
    torch.manual_seed(7)
    v, k, r = 7, 3, 2

    teacher_logits = torch.randn(r, v)
    teacher_probs = torch.softmax(teacher_logits, dim=-1)
    topk = torch.topk(teacher_probs, k=k, dim=-1)
    a_ids = topk.indices  # [r, k], teacher's top-k token ids (A)
    q = topk.values  # [r, k], teacher's actual (unrenormalized) probs at A

    student_logits = torch.randn(r, v, requires_grad=True)
    gathered_log_p = compute_vocab_parallel_topk_log_probs(student_logits, a_ids, None)

    loss = (q * (torch.log(q) - gathered_log_p)).sum()
    loss.backward()

    p = torch.softmax(student_logits.detach(), dim=-1)
    m_a = q.sum(dim=-1, keepdim=True)
    expected_grad = m_a * p
    expected_grad.scatter_add_(-1, a_ids, -q)

    torch.testing.assert_close(student_logits.grad, expected_grad)
