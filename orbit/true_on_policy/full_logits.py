"""True-on-policy full-vocab logit gather (orbit-only).

Lifted verbatim out of ``miles/utils/ppo_utils.py`` (Phase-3 slice 3e, P1
lift-out): the padded-vocab truncation helper, the replicated-loss all-gather
autograd Function and its backward split, plus the TP-aware entry point used by
the true-on-policy log-prob path and by loss.py's full-vocab branch.
``miles.utils.ppo_utils`` re-exports these behind a stamped seam, so existing
importers (``miles/backends/training_utils/loss.py``, the true-on-policy and
vocab-parallel tests) are unchanged.

No module-level ``miles.*`` imports. The single ``megatron.core`` import stays
function-local (exactly as in the base file), so this module imports cleanly in
CPU-only environments.
"""

import torch
import torch.distributed as dist


def _prepare_true_on_policy_full_logits(
    logits_or_shards: torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, ...],
    *,
    vocab_size: int | None = None,
) -> torch.Tensor:
    if isinstance(logits_or_shards, (list, tuple)):
        full_logits = torch.cat([shard.contiguous() for shard in logits_or_shards], dim=-1)
    else:
        full_logits = logits_or_shards.contiguous()

    # Truncate Megatron's padded vocab back to the real tokenizer vocab before
    # log_softmax, matching what SGLang normalizes over.
    if vocab_size is not None and full_logits.size(-1) > vocab_size:
        full_logits = full_logits[..., :vocab_size]

    return full_logits


def _split_replicated_loss_gather_grad(
    grad_output: torch.Tensor,
    *,
    rank: int,
    world_size: int,
    local_last_dim: int,
) -> torch.Tensor:
    if world_size <= 1:
        return grad_output.contiguous()

    expected_last_dim = local_last_dim * world_size
    if grad_output.size(-1) != expected_last_dim:
        raise RuntimeError(
            "True-on-policy replicated-loss gather backward expected the full padded "
            f"vocab dimension to be {expected_last_dim}, got {grad_output.size(-1)}."
        )
    start = rank * local_last_dim
    return grad_output[..., start : start + local_last_dim].contiguous()


class _ReplicatedLossAllGatherLastDim(torch.autograd.Function):
    """All-gather vocab shards for a loss replicated on every TP rank.

    Megatron's standard all-gather autograd uses reduce-scatter in backward,
    which is correct when each rank contributes a distinct output gradient. In
    the true-on-policy logprob path every TP rank computes the same scalar loss
    from the gathered full vocabulary, so reduce-scatter would sum identical
    gradients and scale the local logits gradient by TP size.
    """

    @staticmethod
    def forward(ctx, input_: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        world_size = group.size()
        ctx.group = group
        ctx.local_last_dim = input_.shape[-1]
        ctx.world_size = world_size

        if world_size == 1:
            return input_.contiguous()

        from megatron.core.tensor_parallel.mappings import dist_all_gather_func

        gather_shape = list(input_.shape)
        gather_shape[0] *= world_size
        gathered = torch.empty(gather_shape, dtype=input_.dtype, device=input_.device)
        dist_all_gather_func(gathered, input_.contiguous(), group=group)
        return torch.cat(gathered.chunk(world_size, dim=0), dim=-1).contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (
            _split_replicated_loss_gather_grad(
                grad_output,
                rank=ctx.group.rank(),
                world_size=ctx.world_size,
                local_last_dim=ctx.local_last_dim,
            ),
            None,
        )


def _gather_true_on_policy_full_logits(
    logits: torch.Tensor,
    process_group: dist.ProcessGroup | None,
    *,
    vocab_size: int | None = None,
) -> torch.Tensor:
    if process_group is None or process_group.size() <= 1:
        return _prepare_true_on_policy_full_logits(logits, vocab_size=vocab_size)

    full_logits = _ReplicatedLossAllGatherLastDim.apply(logits.contiguous(), process_group)
    return _prepare_true_on_policy_full_logits(full_logits, vocab_size=vocab_size)


__all__ = [
    "_ReplicatedLossAllGatherLastDim",
    "_gather_true_on_policy_full_logits",
    "_prepare_true_on_policy_full_logits",
    "_split_replicated_loss_gather_grad",
]
