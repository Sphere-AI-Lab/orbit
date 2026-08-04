import torch
import torch.distributed as dist

from tests.fast.dist_utils import init_gloo, run_multiprocess

from orbit.backends.training_utils.vocab_parallel import compute_vocab_parallel_topk_log_probs


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
