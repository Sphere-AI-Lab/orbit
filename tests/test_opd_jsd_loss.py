"""opd_jsd_loss correctness: TP=1 against an independent GKD Eq.(1) reference,
TP=2 (gloo, CPU) against the TP=1 run, and CP slicing of teacher hidden states
against the 1D log-prob slicing it must mirror.

The reference computes the loss from scratch (own slicing, own softmax /
mixture / clamps) so it validates the implementation's math, not just its
self-consistency.
"""

import math
import os
from argparse import Namespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from orbit.opd import teacher_lm_head as teacher_lm_head_module
from miles.backends.training_utils.cp_utils import slice_log_prob_with_cp
from miles.backends.training_utils.data import _tensorize_cp_sliced_teacher_hidden_states
from miles.backends.training_utils.loss import opd_jsd_loss_function
from miles.backends.training_utils.parallel import GroupInfo, ParallelState, set_parallel_state

CHECKPOINT_KEY = "<test-opd-jsd>"

HIDDEN_SIZE = 16
PADDED_VOCAB_SIZE = 512
TEACHER_VOCAB_SIZE = 500  # < PADDED_VOCAB_SIZE: exercises the -1e4 padding fill
RESPONSE_LENGTHS = [5, 1, 4]
PROMPT_LENGTHS = [3, 2, 6]


def _build_args(beta: float, topk_overlap: bool = False) -> Namespace:
    return Namespace(
        opd_jsd_beta=beta,
        rollout_temperature=0.8,
        opd_log_prob_min_clamp=-20.0,
        opd_loss_max_clamp=100.0,
        opd_jsd_pointwise_clip=0.5,
        opd_log_topk_overlap=topk_overlap,
        opd_topk_overlap_ks=[1, 5, 20],
        use_kl_loss=False,
        teacher_hf_checkpoint=CHECKPOINT_KEY,
        qkv_format="thd",
        allgather_cp=False,
        log_probs_chunk_size=-1,
        true_on_policy_mode=False,
        # Megatron pads the model to 512 columns, but only the first 500 are
        # real.  TP=2 therefore has a partially padded final shard.
        vocab_size=TEACHER_VOCAB_SIZE,
    )


def _build_inputs() -> tuple[torch.Tensor, torch.Tensor, dict]:
    generator = torch.Generator().manual_seed(1234)
    total_lengths = [p + r for p, r in zip(PROMPT_LENGTHS, RESPONSE_LENGTHS, strict=True)]
    logits = torch.randn(1, sum(total_lengths), PADDED_VOCAB_SIZE, generator=generator, dtype=torch.float32)
    teacher_head = torch.randn(TEACHER_VOCAB_SIZE, HIDDEN_SIZE, generator=generator, dtype=torch.float32)
    batch = {
        "unconcat_tokens": [
            torch.randint(0, TEACHER_VOCAB_SIZE, (total,), generator=generator) for total in total_lengths
        ],
        "response_lengths": RESPONSE_LENGTHS,
        "total_lengths": total_lengths,
        # Post-data-layer form: one CPU fp32 tensor per sample, CP=1 so unsliced.
        "teacher_hidden_states": [
            torch.randn(response, HIDDEN_SIZE, generator=generator, dtype=torch.float32)
            for response in RESPONSE_LENGTHS
        ],
    }
    return logits, teacher_head, batch


def _single_state() -> None:
    single = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(ParallelState(intra_dp=single, intra_dp_cp=single, cp=single, tp=single))


def _cp_state(rank: int, size: int) -> None:
    single = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(intra_dp=single, intra_dp_cp=single, cp=GroupInfo(rank=rank, size=size, group=None), tp=single)
    )


def _run_loss(args, logits, head, batch):
    teacher_lm_head_module._TEACHER_LM_HEAD_CACHE[CHECKPOINT_KEY] = head
    teacher_lm_head_module._SHARDED.add(CHECKPOINT_KEY)
    logits = logits.detach().clone().requires_grad_(True)
    loss, metrics = opd_jsd_loss_function(args, batch, logits, lambda x: x.sum())
    loss.backward()
    return loss.detach(), {k: v.clone() for k, v in metrics.items()}, logits.grad.detach()


def _reference_loss(args, logits, head, batch):
    """From-scratch GKD Eq.(1) with plain torch ops."""
    logits = logits.detach().clone().requires_grad_(True)
    flat = logits.squeeze(0)
    temperature = args.rollout_temperature
    beta = args.opd_jsd_beta
    per_position = []
    seq_start = 0
    for i, (prompt, response) in enumerate(zip(PROMPT_LENGTHS, RESPONSE_LENGTHS, strict=True)):
        total = prompt + response
        # logits row t predicts token t+1 -> rows [total-response-1, total-1)
        s_logits = flat[seq_start + total - response - 1 : seq_start + total - 1] / temperature
        seq_start += total
        s_lp = torch.log_softmax(s_logits[:, : args.vocab_size], dim=-1).clamp(min=args.opd_log_prob_min_clamp)

        t_logits = (batch["teacher_hidden_states"][i].float() @ head.T) / temperature
        t_lp_real = torch.log_softmax(t_logits, dim=-1).clamp(min=args.opd_log_prob_min_clamp)
        t_lp = s_lp.new_full(s_lp.shape, -1e4)
        t_lp[:, :TEACHER_VOCAB_SIZE] = t_lp_real

        p_s, p_t = s_lp.exp(), t_lp.exp()
        if beta == 0.0:
            elem = p_t * (t_lp - s_lp)
        elif beta == 1.0:
            elem = p_s * (s_lp - t_lp)
        else:
            mixture = torch.logsumexp(torch.stack([s_lp + math.log1p(-beta), t_lp + math.log(beta)]), dim=0)
            elem = beta * (p_t * (t_lp - mixture)) + (1 - beta) * (p_s * (s_lp - mixture))
        if args.opd_jsd_pointwise_clip is not None:
            elem = elem.clamp(max=args.opd_jsd_pointwise_clip)
        per_position.append(elem.sum(dim=-1).clamp(max=args.opd_loss_max_clamp))
    loss = torch.cat(per_position).sum()
    loss.backward()
    return loss.detach(), logits.grad.detach()


@pytest.mark.parametrize("beta", [0.0, 0.5, 1.0])
def test_jsd_matches_independent_reference(beta):
    _single_state()
    args = _build_args(beta)
    logits, head, batch = _build_inputs()
    impl_loss, _, impl_grad = _run_loss(args, logits, head, batch)
    ref_loss, ref_grad = _reference_loss(args, logits, head, batch)
    assert torch.allclose(impl_loss, ref_loss, atol=1e-5), f"{impl_loss} vs {ref_loss}"
    assert torch.allclose(impl_grad, ref_grad, atol=1e-6)


def _tp_worker(rank: int, tp_size: int, port: int, beta: float, results) -> None:
    dist.init_process_group("gloo", rank=rank, world_size=tp_size, init_method=f"tcp://127.0.0.1:{port}")
    args = _build_args(beta, topk_overlap=True)
    logits, teacher_head, batch = _build_inputs()

    if rank == 0:
        _single_state()
        ref_loss, ref_metrics, ref_grad = _run_loss(args, logits, teacher_head, batch)

    shard = PADDED_VOCAB_SIZE // tp_size
    start, end = rank * shard, (rank + 1) * shard
    single = GroupInfo(rank=0, size=1, group=None)
    set_parallel_state(
        ParallelState(
            intra_dp=single,
            intra_dp_cp=single,
            cp=single,
            tp=GroupInfo(rank=rank, size=tp_size, group=dist.group.WORLD),
        )
    )
    tp_loss, tp_metrics, tp_grad = _run_loss(
        args,
        logits[:, :, start:end],
        teacher_head[start : min(end, TEACHER_VOCAB_SIZE)],
        batch,
    )
    if rank == 0:
        assert torch.allclose(tp_loss, ref_loss, atol=1e-4), f"{tp_loss} vs {ref_loss}"
        assert torch.allclose(tp_grad, ref_grad[:, :, start:end], atol=1e-6)
        for key, ref_value in ref_metrics.items():
            assert torch.allclose(tp_metrics[key], ref_value, atol=1e-4), key
        results.put("ok")
    dist.destroy_process_group()


def test_jsd_tp2_matches_tp1():
    port = 29511 + os.getpid() % 1000
    ctx = mp.get_context("spawn")
    results = ctx.Queue()
    mp.start_processes(_tp_worker, args=(2, port, 0.5, results), nprocs=2, join=True, start_method="spawn")
    assert results.get(timeout=10) == "ok"


def test_jsd_ignores_student_padding_in_loss_and_gradient():
    _single_state()
    args = _build_args(0.0)
    args.rollout_temperature = 1.0
    args.opd_jsd_pointwise_clip = None
    args.opd_log_prob_min_clamp = -1e30
    args.vocab_size = 2
    args.opd_log_topk_overlap = True
    args.opd_topk_overlap_ks = [3]

    # The single response uses logits row 0.  Teacher hidden state and the
    # identity head reconstruct exactly the two real student logits; the two
    # padding columns deliberately carry finite competing scores.
    logits = torch.tensor([[[2.0, 1.0, 0.0, 0.0], [9.0, 9.0, 9.0, 9.0]]])
    head = torch.eye(2)
    batch = {
        "unconcat_tokens": [torch.tensor([0, 1])],
        "response_lengths": [1],
        "total_lengths": [2],
        "teacher_hidden_states": [torch.tensor([[2.0, 1.0]])],
    }

    loss, metrics, grad = _run_loss(args, logits, head, batch)

    torch.testing.assert_close(loss, torch.zeros_like(loss), atol=1e-7, rtol=0)
    torch.testing.assert_close(grad[..., 2:], torch.zeros_like(grad[..., 2:]), atol=0, rtol=0)
    torch.testing.assert_close(metrics["topk_overlap_k3"], torch.ones_like(metrics["topk_overlap_k3"]))


def test_cp_hidden_state_slicing_matches_log_prob_slicing():
    """The 2D hidden-state CP slice must select exactly the rows whose indices the 1D
    log-prob slice selects, per rank, and the two ranks must cover every response row
    exactly once."""
    total_length, response_length = 19, 13
    hidden = np.arange(response_length * 4, dtype=np.float32).reshape(response_length, 4)
    args = Namespace(qkv_format="thd")

    seen_rows = []
    for rank in range(2):
        _cp_state(rank, 2)
        index_slice = slice_log_prob_with_cp(list(range(response_length)), total_length, response_length, "thd", None)
        rollout_data = {
            "teacher_hidden_states": [hidden.copy()],
            "total_lengths": [total_length],
            "response_lengths": [response_length],
        }
        _tensorize_cp_sliced_teacher_hidden_states(args, rollout_data)
        sliced = rollout_data["teacher_hidden_states"][0]
        expected = torch.from_numpy(hidden[index_slice])
        assert torch.equal(sliced, expected), f"rank {rank}: rows {index_slice}"
        seen_rows.extend(index_slice)
    assert sorted(seen_rows) == list(range(response_length))


def test_oversized_teacher_head_is_clipped_to_student_vocab():
    """A bigger same-tokenizer teacher can pad its vocab wider than the student
    (Qwen2.5-7B: 152064 vs 151936). Rows past the student's logit width are
    padding the student cannot emit: the loss must behave exactly as if the
    head were pre-clipped to the student's real vocabulary."""
    _single_state()
    args = _build_args(0.5)
    logits, head, batch = _build_inputs()
    generator = torch.Generator().manual_seed(99)
    oversized = torch.cat([head, torch.randn(8, HIDDEN_SIZE, generator=generator)], dim=0)
    assert (
        oversized.size(0) > PADDED_VOCAB_SIZE - 8
    )  # wider than the student's 512 - real 500? build: head=500 rows; oversized=508 <512! need > 512
    oversized = torch.cat([oversized, torch.randn(8, HIDDEN_SIZE, generator=generator)], dim=0)
    assert oversized.size(0) == TEACHER_VOCAB_SIZE + 16 > PADDED_VOCAB_SIZE

    big_loss, _, big_grad = _run_loss(args, logits, oversized, batch)
    clipped_loss, _, clipped_grad = _run_loss(args, logits, oversized[: args.vocab_size], batch)
    assert torch.equal(big_loss, clipped_loss)
    assert torch.equal(big_grad, clipped_grad)
