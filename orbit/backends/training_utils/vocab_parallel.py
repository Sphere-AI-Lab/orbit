"""Collectives for losses that need the *whole* vocabulary under tensor parallelism.

Megatron's output layer is column-parallel over the vocabulary: with
tensor_model_parallel_size == W each TP rank holds only a `1/W` slice of the logit columns.
Losses that only need the sampled token's log-prob delegate to megatron.core's
fused vocab-parallel kernels, but a full-vocabulary divergence such as `opd_jsd_loss` has to
normalize and reduce across the shards itself -- that is what these helpers provide.
"""

import torch
import torch.distributed as dist

from .parallel import get_parallel_state


def vocab_shard_start(local_vocab_size: int) -> int:
    """Global index of this TP rank's first vocabulary column, given its logit width.

    Megatron splits the output layer's vocabulary into equal, contiguous, rank-ordered
    chunks -- the convention `fused_vocab_parallel_cross_entropy` already relies on -- so the
    local width determines the offset. Derived from the logits rather than from
    `args.padded_vocab_size` because the latter is only what the model was built with on the
    non-bridge path; under `--megatron-to-hf-mode bridge` the vocabulary comes from the HF
    config instead and the two disagree.
    """
    return get_parallel_state().tp.rank * local_vocab_size


class _ReduceFromVocabParallelRegion(torch.autograd.Function):
    """Complete a per-rank partial sum; backward is identity (Megatron's `g` operator).

    Identity is right only because everything downstream of the reduced value is computed
    redundantly -- every TP rank runs the same reduction to the same scalar loss from the
    same replicated total, so the loss is counted once and each rank's partial enters it
    with unit weight.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        x = x.clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


class _AllReduceReplicated(torch.autograd.Function):
    """Complete a per-rank partial sum whose result each rank then consumes *differently*.

    Backward all-reduces as well, unlike `_ReduceFromVocabParallelRegion`. The value
    produced here (a softmax normalizer) is applied to every rank's own vocab shard, so the
    gradient arriving locally accounts for that shard alone and has to be completed with the
    other ranks' shares before it can be pushed back into the local logits.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        x = x.clone()
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        grad_output = grad_output.clone()
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_output, None


def vocab_parallel_log_softmax(logits: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
    """`log_softmax` of shard-local `logits` `[R, V_local]` over the *global* vocabulary.

    `group=None` means tensor parallelism is off and this is plain `torch.log_softmax`. A
    zero-width shard is supported: a rank whose columns all fall past the teacher's real
    vocabulary still has to join both collectives.
    """
    if group is None:
        return torch.log_softmax(logits, dim=-1)

    with torch.no_grad():
        if logits.size(-1) == 0:
            # max() over an empty dim raises; -inf is the identity of a MAX all-reduce.
            local_max = logits.new_full((logits.size(0), 1), -torch.inf)
        else:
            local_max = logits.max(dim=-1, keepdim=True).values.clone()
        dist.all_reduce(local_max, op=dist.ReduceOp.MAX, group=group)

    # Held under no_grad above because the shift cancels symbolically in
    # `x - m - log(sum(exp(x - m)))`; differentiating it would only add a backward collective.
    # Summing an empty dim yields 0, the identity of the SUM all-reduce that follows.
    sum_exp = (logits - local_max).exp().sum(dim=-1, keepdim=True)
    return logits - local_max - _AllReduceReplicated.apply(sum_exp, group).log()


def vocab_parallel_sum(x: torch.Tensor, group: dist.ProcessGroup | None) -> torch.Tensor:
    """Sum `x` `[R, V_local]` over the vocabulary, across TP shards."""
    total = x.sum(dim=-1)
    if group is None:
        return total
    return _ReduceFromVocabParallelRegion.apply(total, group)


def vocab_parallel_topk_indices(
    log_probs: torch.Tensor,
    k: int,
    vocab_start: int,
    group: dist.ProcessGroup | None,
) -> torch.Tensor:
    """Global vocabulary indices of the top-`k` entries of shard-local `log_probs`.

    Returns `[R, min(k, V_global)]` sorted by descending log-prob, identical on every TP rank.
    Diagnostic-only, so the whole thing runs under `no_grad`.
    """
    with torch.no_grad():
        # Shards are equal width, so k_local matches across ranks and all_gather stays regular.
        values, indices = torch.topk(log_probs, k=min(k, log_probs.size(-1)), dim=-1)
        indices = indices + vocab_start
        if group is None:
            return indices

        world_size = dist.get_world_size(group)
        gathered_values = [torch.empty_like(values) for _ in range(world_size)]
        gathered_indices = [torch.empty_like(indices) for _ in range(world_size)]
        dist.all_gather(gathered_values, values.contiguous(), group=group)
        dist.all_gather(gathered_indices, indices.contiguous(), group=group)

        # Each rank's local top-k is a superset of its own contribution to the global top-k,
        # so re-ranking the W*k candidates gives the exact global answer.
        all_values = torch.cat(gathered_values, dim=-1)
        all_indices = torch.cat(gathered_indices, dim=-1)
        winners = torch.topk(all_values, k=min(k, all_values.size(-1)), dim=-1).indices
        return torch.gather(all_indices, -1, winners)
