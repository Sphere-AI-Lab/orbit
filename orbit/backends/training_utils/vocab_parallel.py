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


def compute_vocab_parallel_topk_log_probs(
    logits: torch.Tensor,
    topk_ids: torch.Tensor,
    process_group: dist.ProcessGroup | None = None,
) -> torch.Tensor:
    """Gather (differentiable) log-probs at externally supplied token ids from vocab-parallel logits.

    Used by on_policy_distillation's "topk" loss (`opd_topk_loss_function`) to score the
    student -- with gradients -- at the teacher's top-k token ids for every position, not
    just whichever token the student happened to sample. Deliberately avoids Megatron's
    `fused_vocab_parallel_cross_entropy` (which is wrapped in `@jit_fuser` / torch.compile):
    that kernel recompiles and re-autotunes per new input shape, and calling it once per
    top-k slot in a loop from inside a pipeline-parallel `forward_step` has been observed
    to crash with "CUDA driver error: invalid argument" during Triton autotuning. This
    implementation instead does a single vectorized gather over all K ids at once using
    plain eager ops (mirrors the masked-gather + all-reduce pattern Megatron's own
    vocab-parallel cross-entropy uses internally, and the log-sum-exp all-reduce pattern
    already used by `_VocabParallelEntropy` above), computing the log-normalizer once and
    reusing it for every id.

    Differentiable w.r.t. `logits`. The max used to shift logits for a numerically stable
    log-sum-exp is detached: log-sum-exp's gradient (softmax) is exactly the same
    regardless of which constant shift was used, so detaching the shift only avoids
    differentiating through an arbitrary arg-max tie-break -- it does not change the
    gradient. The sum-of-exp and per-id gather all-reduces use `_ReduceFromVocabParallelRegion`
    (identity backward, not a second all-reduce): everything downstream -- the caller's loss --
    is computed redundantly from the same replicated total on every TP rank, so each rank's
    local partial must enter that loss with unit weight, exactly as `vocab_parallel_sum` above
    already relies on for the same reason.

    `process_group=None` (TP=1 / no tensor parallelism) short-circuits to a plain local
    `log_softmax` + `gather` -- no collectives.

    Adapted from czy/opd @ 0a33680 (`orbit/utils/ppo_utils.py:234-300`), with the backward
    convention above fixed: czy's version used a `_DifferentiableAllReduceSum` whose backward
    re-all-reduces the incoming gradient, which double-counts under a TP-replicated loss and
    was empirically observed to inflate gradients by exactly `tp_size` at TP=2.

    Args:
        logits: `[R, V_local]` vocab-parallel logits (this rank's shard). Requires grad.
        topk_ids: `[R, K]` global (unsharded) token ids to gather log-probs for.
        process_group: Tensor-parallel process group, or `None` if TP is off.

    Returns:
        `[R, K]` log-probs, differentiable w.r.t. `logits`.
    """
    k = topk_ids.size(-1)
    if logits.size(0) == 0:
        return logits.new_zeros((0, k))

    # teacher_topk_ids arrives from TeacherManager via Ray (CPU tensor); logits is
    # the direct output of the student's own forward pass (GPU). Move to match
    # before indexing -- torch.gather requires index and input on the same device.
    topk_ids = topk_ids.to(device=logits.device)
    logits = logits.float()

    if process_group is None:
        return torch.log_softmax(logits, dim=-1).gather(-1, topk_ids)

    tp_rank = dist.get_rank(group=process_group)
    partition_vocab_size = logits.size(-1)
    vocab_start_index = tp_rank * partition_vocab_size
    vocab_end_index = vocab_start_index + partition_vocab_size

    with torch.no_grad():
        logits_max = logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)

    exp_logits = (logits - logits_max).exp()
    sum_exp_logits_local = exp_logits.sum(dim=-1, keepdim=True)
    sum_exp_logits = _ReduceFromVocabParallelRegion.apply(sum_exp_logits_local, process_group)
    log_normalizer = logits_max.squeeze(-1) + sum_exp_logits.squeeze(-1).log()  # [R]

    local_ids = (topk_ids - vocab_start_index).clamp(0, partition_vocab_size - 1)
    owned_mask = (topk_ids >= vocab_start_index) & (topk_ids < vocab_end_index)
    gathered_logit = torch.gather(logits, dim=-1, index=local_ids)  # [R, K]
    gathered_logit = torch.where(owned_mask, gathered_logit, torch.zeros_like(gathered_logit))
    gathered_logit = _ReduceFromVocabParallelRegion.apply(gathered_logit, process_group)

    return gathered_logit - log_normalizer.unsqueeze(-1)


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
