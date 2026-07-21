from argparse import Namespace
from collections.abc import Callable

import torch

from miles.backends.training_utils.cp_utils import (
    get_local_response_loss_masks,
    slice_response_tensor_with_cp,
)
from miles.backends.training_utils.parallel import get_parallel_state
from miles.utils.types import RolloutBatch

from .logit_processors import get_responses
from .math_utils import compute_log_probs


_SPARSE_TARGET_KEYS = (
    "teacher_topk_token_ids",
    "teacher_topk_log_probs",
    "teacher_topk_valid_mask",
)


def _require_sparse_targets(batch: RolloutBatch, batch_size: int) -> tuple[list[torch.Tensor], ...]:
    values = tuple(batch.get(key) for key in _SPARSE_TARGET_KEYS)
    if not all(value is not None for value in values):
        missing = [key for key, value in zip(_SPARSE_TARGET_KEYS, values, strict=True) if value is None]
        raise ValueError(f"Top-K DAgger loss requires sparse teacher targets; missing {missing}.")
    if any(len(value) != batch_size for value in values):
        raise ValueError(
            "Top-K DAgger target batch mismatch: "
            + ", ".join(f"{key}={len(value)}" for key, value in zip(_SPARSE_TARGET_KEYS, values, strict=True))
            + f", response_lengths={batch_size}."
        )
    return values


def explicit_topk_cross_entropy_per_token(
    logits: torch.Tensor,
    teacher_token_ids: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    teacher_valid_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    tp_group,
    true_on_policy_mode: bool,
    vocab_size: int | None,
) -> dict[str, torch.Tensor]:
    """Evaluate the unnormalized explicit teacher Top-K term on local rows.

    Each candidate column reuses Miles' existing one-target vocab-parallel
    log-probability primitive. Masked entries are selected out before exponent
    or multiplication, so the ``-inf`` padding sentinel never enters arithmetic.
    """
    if logits.ndim != 2 or logits.size(-1) == 0:
        raise ValueError(f"Top-K DAgger logits must have shape [T, V_local], got {tuple(logits.shape)}.")

    teacher_token_ids = torch.as_tensor(teacher_token_ids, device=logits.device, dtype=torch.long).detach()
    teacher_log_probs = torch.as_tensor(teacher_log_probs, device=logits.device, dtype=torch.float32).detach()
    teacher_valid_mask = torch.as_tensor(teacher_valid_mask, device=logits.device, dtype=torch.bool).detach()
    loss_mask = torch.as_tensor(loss_mask, device=logits.device, dtype=torch.bool)

    target_shape = teacher_token_ids.shape
    if (
        teacher_token_ids.ndim != 2
        or target_shape != teacher_log_probs.shape
        or target_shape != teacher_valid_mask.shape
    ):
        raise ValueError(
            "Top-K DAgger target shape mismatch: "
            f"ids={tuple(teacher_token_ids.shape)}, log_probs={tuple(teacher_log_probs.shape)}, "
            f"valid_mask={tuple(teacher_valid_mask.shape)}."
        )
    if target_shape[0] != logits.size(0) or loss_mask.shape != (logits.size(0),):
        raise ValueError(
            "Top-K DAgger response alignment mismatch: "
            f"logits={tuple(logits.shape)}, targets={target_shape}, loss_mask={tuple(loss_mask.shape)}."
        )
    if target_shape[1] == 0:
        raise ValueError("Top-K DAgger targets require K > 0.")

    candidate_mask = teacher_valid_mask & loss_mask.unsqueeze(-1)
    # Upstream parsing validates finite active values and 0/-inf sentinels. Use
    # where before exp so padded -inf never participates in arithmetic, while
    # keeping fixed [T] columns avoids CUDA-synchronizing nonzero/index gathers.
    safe_teacher_log_probs = torch.where(candidate_mask, teacher_log_probs, teacher_log_probs.new_zeros(()))
    teacher_probs = torch.where(candidate_mask, safe_teacher_log_probs.exp(), teacher_log_probs.new_zeros(()))
    teacher_topk_mass = teacher_probs.sum(dim=-1)

    # Preserve a zero-valued path to logits for empty/masked rows and ranks.
    per_token_loss = logits[:, 0].float() * 0.0
    if logits.size(0) == 0:
        valid_candidates = candidate_mask.sum(dim=-1).float()
        return {
            "per_token_loss": per_token_loss,
            "teacher_topk_mass": teacher_topk_mass,
            "valid_candidates": valid_candidates,
            "valid_positions": valid_candidates.gt(0).float(),
        }

    for candidate_rank in range(target_shape[1]):
        student_log_probs = compute_log_probs(
            logits.clone(),
            teacher_token_ids[:, candidate_rank],
            tp_group,
            true_on_policy_mode=true_on_policy_mode,
            vocab_size=vocab_size,
        ).reshape(-1)
        candidate_loss = -teacher_probs[:, candidate_rank] * student_log_probs.float()
        per_token_loss = per_token_loss + torch.where(
            candidate_mask[:, candidate_rank],
            candidate_loss,
            candidate_loss.new_zeros(()),
        )

    valid_candidates = candidate_mask.sum(dim=-1).float()
    return {
        "per_token_loss": per_token_loss,
        "teacher_topk_mass": teacher_topk_mass,
        "valid_candidates": valid_candidates,
        "valid_positions": valid_candidates.gt(0).float(),
    }


def compute_explicit_dagger_loss(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    reduce_response_values: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute milestone-01 Top-K explicit CE from the current trainer forward."""
    parallel_state = get_parallel_state()
    if parallel_state.cp.size > 1 and getattr(args, "allgather_cp", False):
        raise NotImplementedError(
            "Top-K DAgger does not yet support --allgather-cp with context_parallel_size > 1; "
            "the 01 path supports CP=1 and the existing zigzag CP layout."
        )

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)
    batch_size = len(response_lengths)
    teacher_ids, teacher_log_probs, teacher_valid_masks = _require_sparse_targets(batch, batch_size)

    response_logits = [
        logits_chunk
        for logits_chunk, _ in get_responses(
            logits,
            args=args,
            unconcat_tokens=batch["unconcat_tokens"],
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )
    ]
    local_loss_masks = get_local_response_loss_masks(
        total_lengths,
        response_lengths,
        batch["loss_masks"],
        args.qkv_format,
        max_seq_lens,
    )
    if len(response_logits) != batch_size or len(local_loss_masks) != batch_size:
        raise ValueError(
            f"Top-K DAgger response batch mismatch: logits={len(response_logits)}, "
            f"loss_masks={len(local_loss_masks)}, samples={batch_size}."
        )

    outputs: dict[str, list[torch.Tensor]] = {
        "per_token_loss": [],
        "teacher_topk_mass": [],
        "valid_candidates": [],
        "valid_positions": [],
    }
    expected_top_k = int(getattr(args, "opd_dagger_top_k", 0) or 0)
    for sample_index, logits_chunk in enumerate(response_logits):
        max_seq_len = max_seq_lens[sample_index] if max_seq_lens is not None else None
        local_ids = slice_response_tensor_with_cp(
            teacher_ids[sample_index],
            total_lengths[sample_index],
            response_lengths[sample_index],
            args.qkv_format,
            max_seq_len,
        )
        local_teacher_log_probs = slice_response_tensor_with_cp(
            teacher_log_probs[sample_index],
            total_lengths[sample_index],
            response_lengths[sample_index],
            args.qkv_format,
            max_seq_len,
        )
        local_valid_mask = slice_response_tensor_with_cp(
            teacher_valid_masks[sample_index],
            total_lengths[sample_index],
            response_lengths[sample_index],
            args.qkv_format,
            max_seq_len,
        )
        if expected_top_k > 0 and local_ids.shape[1] != expected_top_k:
            raise ValueError(
                f"Top-K DAgger width mismatch at sample {sample_index}: "
                f"targets K={local_ids.shape[1]}, configured K={expected_top_k}."
            )

        sample_outputs = explicit_topk_cross_entropy_per_token(
            logits_chunk,
            local_ids,
            local_teacher_log_probs,
            local_valid_mask,
            local_loss_masks[sample_index],
            tp_group=parallel_state.tp.group,
            true_on_policy_mode=bool(getattr(args, "true_on_policy_mode", False)),
            vocab_size=getattr(args, "vocab_size", None),
        )
        for key, value in sample_outputs.items():
            outputs[key].append(value)

    reduced = {key: reduce_response_values(torch.cat(values, dim=0)) for key, values in outputs.items()}
    return reduced["per_token_loss"], {
        "explicit_ce": reduced["per_token_loss"].detach(),
        "teacher_topk_mass": reduced["teacher_topk_mass"].detach(),
        "valid_candidates_mean": reduced["valid_candidates"].detach(),
        "valid_position_ratio": reduced["valid_positions"].detach(),
    }
