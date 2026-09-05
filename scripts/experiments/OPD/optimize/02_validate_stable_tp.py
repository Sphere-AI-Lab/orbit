#!/usr/bin/env python3
"""Distributed parity check for the milestone-02 Stable TP loss.

Run with torchrun. This is intentionally independent of the 3-node E2E job:
it compares each local vocabulary-shard gradient against the matching slice of
the dense FP64 Oracle and can create multiple TP groups to catch accidental
WORLD-group collectives.
"""

import argparse
import math
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
from tests.fast.backends.training_utils.loss.rkld_dagger_test_utils import dense_topk_rest_oracle

from orbit.backends.training_utils.loss_hub.math_utils import vocab_parallel_topk_rest_cross_entropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp-size", type=int, required=True)
    parser.add_argument("--vocab-size", type=int, default=11)
    return parser.parse_args()


def create_tp_group(tp_size: int) -> tuple[dist.ProcessGroup, int, int]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if tp_size <= 0 or world_size % tp_size != 0:
        raise ValueError(f"world_size={world_size} must be divisible by tp_size={tp_size}.")

    selected_group = None
    selected_dp_rank = None
    selected_tp_rank = None
    for dp_rank, start in enumerate(range(0, world_size, tp_size)):
        ranks = list(range(start, start + tp_size))
        group = dist.new_group(ranks=ranks)
        if rank in ranks:
            selected_group = group
            selected_dp_rank = dp_rank
            selected_tp_rank = rank - start

    assert selected_group is not None and selected_dp_rank is not None and selected_tp_rank is not None
    return selected_group, selected_dp_rank, selected_tp_rank


def build_case(
    *,
    dp_rank: int,
    vocab_size: int,
    padded_vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260712 + dp_rank)
    full_logits = torch.randn((3, padded_vocab_size), generator=generator, dtype=torch.float64).to(device)
    if padded_vocab_size > vocab_size:
        full_logits[:, vocab_size:] = 50.0 + dp_rank

    midpoint = vocab_size // 2
    teacher_ids = torch.tensor(
        [[0, vocab_size - 1], [midpoint, min(midpoint + 1, vocab_size - 1)], [1, vocab_size - 2]],
        device=device,
        dtype=torch.long,
    )
    teacher_ids = (teacher_ids + dp_rank) % vocab_size
    candidate_mask = torch.tensor(
        [[True, True], [True, True], [True, False]],
        device=device,
        dtype=torch.bool,
    )
    teacher_probs = torch.tensor(
        [[0.45, 0.35], [0.50, 0.20], [0.40, 0.00]],
        device=device,
        dtype=torch.float64,
    )
    teacher_rest_mass = 1.0 - teacher_probs.sum(dim=-1)
    response_mask = torch.ones(3, device=device, dtype=torch.bool)
    return full_logits, teacher_ids, teacher_probs, teacher_rest_mass, candidate_mask & response_mask.unsqueeze(-1)


def main() -> None:
    args = parse_args()
    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    tp_group, dp_rank, tp_rank = create_tp_group(args.tp_size)
    local_vocab_size = math.ceil(args.vocab_size / args.tp_size)
    padded_vocab_size = local_vocab_size * args.tp_size
    full_logits, teacher_ids, teacher_probs, teacher_rest_mass, candidate_mask = build_case(
        dp_rank=dp_rank,
        vocab_size=args.vocab_size,
        padded_vocab_size=padded_vocab_size,
        device=device,
    )

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
        vocab_size=args.vocab_size,
        row_chunk_size=1,
    )
    actual["per_token_loss"].sum().backward()

    oracle_logits = full_logits[:, : args.vocab_size].clone().requires_grad_()
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
        torch.ones(3, device=device, dtype=torch.bool),
        vocab_size=args.vocab_size,
    )
    expected["per_token_loss"].sum().backward()

    torch.testing.assert_close(actual["per_token_loss"], expected["per_token_loss"], atol=1e-10, rtol=1e-10)
    expected_local_grad = torch.zeros_like(local_logits)
    real_start = min(local_start, args.vocab_size)
    real_end = min(local_end, args.vocab_size)
    if real_end > real_start:
        expected_local_grad[:, : real_end - real_start] = oracle_logits.grad[:, real_start:real_end]
    torch.testing.assert_close(local_logits.grad, expected_local_grad, atol=1e-10, rtol=1e-10)

    dist.barrier()
    if dist.get_rank() == 0:
        dp_size = dist.get_world_size() // args.tp_size
        print(
            "Stable TP parity passed: "
            f"world_size={dist.get_world_size()}, TP={args.tp_size}, DP={dp_size}, "
            f"vocab={args.vocab_size}, padded_vocab={padded_vocab_size}."
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
