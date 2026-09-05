import math
from argparse import Namespace
from collections.abc import Callable

import torch

from orbit.backends.training_utils.cp_utils import (
    get_local_response_loss_masks,
    slice_response_tensor_with_cp,
)
from orbit.backends.training_utils.parallel import get_parallel_state
from orbit.utils.types import RolloutBatch

from .logit_processors import get_responses
from .math_utils import compute_log_probs, vocab_parallel_topk_rest_cross_entropy


_SPARSE_TARGET_KEYS = (
    "teacher_topk_token_ids",
    "teacher_topk_log_probs",
    "teacher_topk_valid_mask",
)
_TEACHER_TOPK_MASS_TOLERANCE = 1e-5


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

    Each candidate column reuses Orbit' existing one-target vocab-parallel
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


def topk_rest_cross_entropy_per_token(
    logits: torch.Tensor,
    teacher_token_ids: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    teacher_valid_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    tp_group,
    vocab_size: int,
) -> dict[str, torch.Tensor]:
    """Evaluate complete coarse teacher Top-K + Rest CE on local response rows."""
    if logits.ndim != 2 or logits.size(-1) == 0:
        raise ValueError(f"Top-K + Rest logits must have shape [T,V_local], got {tuple(logits.shape)}.")

    teacher_token_ids = torch.as_tensor(teacher_token_ids, device=logits.device, dtype=torch.long).detach()
    teacher_log_probs = torch.as_tensor(teacher_log_probs, device=logits.device, dtype=torch.float32).detach()
    teacher_valid_mask = torch.as_tensor(teacher_valid_mask, device=logits.device, dtype=torch.bool).detach()
    loss_mask = torch.as_tensor(loss_mask, device=logits.device, dtype=torch.bool)

    target_shape = teacher_token_ids.shape
    if (
        teacher_token_ids.ndim != 2
        or target_shape != teacher_log_probs.shape
        or target_shape != teacher_valid_mask.shape
        or target_shape[0] != logits.size(0)
        or loss_mask.shape != (logits.size(0),)
    ):
        raise ValueError(
            "Top-K + Rest target shape mismatch: "
            f"logits={tuple(logits.shape)}, ids={target_shape}, log_probs={tuple(teacher_log_probs.shape)}, "
            f"valid_mask={tuple(teacher_valid_mask.shape)}, loss_mask={tuple(loss_mask.shape)}."
        )
    if target_shape[1] == 0:
        raise ValueError("Top-K + Rest targets require K > 0.")

    candidate_mask = teacher_valid_mask & loss_mask.unsqueeze(-1)
    active_log_probs = teacher_log_probs[candidate_mask]
    active_ids = teacher_token_ids[candidate_mask]
    if active_log_probs.numel() and not torch.isfinite(active_log_probs).all().item():
        raise ValueError("Top-K + Rest teacher log-probs must be finite on valid positions.")
    if active_ids.numel() and ((active_ids < 0) | (active_ids >= vocab_size)).any().item():
        raise ValueError(f"Top-K + Rest teacher token IDs must be in [0, {vocab_size}).")

    duplicate_rows = torch.zeros(logits.size(0), device=logits.device, dtype=torch.bool)
    for left in range(target_shape[1]):
        for right in range(left + 1, target_shape[1]):
            duplicate_rows |= (
                candidate_mask[:, left]
                & candidate_mask[:, right]
                & (teacher_token_ids[:, left] == teacher_token_ids[:, right])
            )
    if duplicate_rows.any().item():
        row = int(duplicate_rows.nonzero(as_tuple=False)[0].item())
        raise ValueError(f"Top-K + Rest teacher row {row} contains duplicate token IDs.")

    masked_log_probs = torch.where(
        candidate_mask,
        teacher_log_probs,
        teacher_log_probs.new_full((), -torch.inf),
    )
    log_topk_mass = torch.logsumexp(masked_log_probs, dim=-1)
    log_mass_limit = math.log1p(_TEACHER_TOPK_MASS_TOLERANCE)
    if (log_topk_mass > log_mass_limit).any().item():
        max_log_mass = log_topk_mass.max().item()
        raise ValueError(
            "Top-K + Rest teacher probability mass exceeds 1: "
            f"max log-mass={max_log_mass:.9g}, tolerance={_TEACHER_TOPK_MASS_TOLERANCE}."
        )

    safe_log_probs = torch.where(candidate_mask, teacher_log_probs, teacher_log_probs.new_zeros(()))
    teacher_probs = torch.where(candidate_mask, safe_log_probs.exp(), teacher_log_probs.new_zeros(()))
    # Reject material mass violations above, then map only tolerated positive
    # log-mass roundoff to p_R=0. This is a domain guard, not an objective clamp:
    # every valid log_topk_mass <= 0 passes through unchanged.
    teacher_rest_mass = -torch.expm1(torch.clamp_max(log_topk_mass, 0.0))

    outputs = vocab_parallel_topk_rest_cross_entropy(
        logits,
        teacher_token_ids,
        teacher_probs,
        teacher_rest_mass,
        candidate_mask,
        tp_group,
        vocab_size=vocab_size,
    )
    effective_teacher_rest_mass = outputs["teacher_rest_mass"]
    teacher_entropy = -torch.xlogy(teacher_probs, teacher_probs).sum(dim=-1) - torch.xlogy(
        effective_teacher_rest_mass, effective_teacher_rest_mass
    )
    outputs["teacher_entropy"] = teacher_entropy
    # CE includes the fixed teacher entropy. Report the mismatch separately so
    # coefficient selection is not based on incomparable absolute CE scales.
    outputs["coarse_kl"] = outputs["per_token_loss"].detach() - teacher_entropy
    # This clamp is telemetry-only. The loss and gradient use logZ_R-logZ
    # directly and never use a clamped student probability.
    outputs["student_topk_mass"] = (1.0 - outputs["student_rest_mass"]).clamp(0.0, 1.0)
    outputs["rest_mass_abs_error"] = (outputs["student_rest_mass"] - outputs["teacher_rest_mass"]).abs()
    valid_candidates = candidate_mask.sum(dim=-1).float()
    outputs["valid_candidates"] = valid_candidates
    outputs["valid_positions"] = valid_candidates.gt(0).float()
    return outputs


def _prepare_local_dagger_inputs(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    parallel_state = get_parallel_state()
    if parallel_state.cp.size > 1 and getattr(args, "allgather_cp", False):
        raise NotImplementedError(
            "Top-K DAgger does not yet support --allgather-cp with context_parallel_size > 1; "
            "the current path supports CP=1 and the existing zigzag CP layout."
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

    expected_top_k = int(getattr(args, "opd_dagger_top_k", 0) or 0)
    prepared = []
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
        prepared.append(
            (
                logits_chunk,
                local_ids,
                local_teacher_log_probs,
                local_valid_mask,
                local_loss_masks[sample_index],
            )
        )
    return prepared


def compute_explicit_dagger_loss(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    reduce_response_values: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute milestone-01 Top-K explicit CE from the current trainer forward."""
    parallel_state = get_parallel_state()
    outputs: dict[str, list[torch.Tensor]] = {
        "per_token_loss": [],
        "teacher_topk_mass": [],
        "valid_candidates": [],
        "valid_positions": [],
    }
    for (
        logits_chunk,
        local_ids,
        local_teacher_log_probs,
        local_valid_mask,
        local_loss_mask,
    ) in _prepare_local_dagger_inputs(args, batch, logits):
        sample_outputs = explicit_topk_cross_entropy_per_token(
            logits_chunk,
            local_ids,
            local_teacher_log_probs,
            local_valid_mask,
            local_loss_mask,
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


def compute_topk_rest_dagger_loss(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    reduce_response_values: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute milestone-02 complete Top-K + Rest CE from current trainer logits."""
    parallel_state = get_parallel_state()
    vocab_size = getattr(args, "vocab_size", None)
    if vocab_size is None:
        raise ValueError("Top-K + Rest DAgger requires args.vocab_size to exclude padded vocabulary logits.")

    output_keys = (
        "per_token_loss",
        "explicit_ce",
        "rest_ce",
        "teacher_entropy",
        "coarse_kl",
        "teacher_topk_mass",
        "teacher_rest_mass",
        "student_topk_mass",
        "student_rest_mass",
        "rest_mass_abs_error",
        "valid_candidates",
        "valid_positions",
    )
    outputs: dict[str, list[torch.Tensor]] = {key: [] for key in output_keys}
    for (
        logits_chunk,
        local_ids,
        local_teacher_log_probs,
        local_valid_mask,
        local_loss_mask,
    ) in _prepare_local_dagger_inputs(args, batch, logits):
        sample_outputs = topk_rest_cross_entropy_per_token(
            logits_chunk,
            local_ids,
            local_teacher_log_probs,
            local_valid_mask,
            local_loss_mask,
            tp_group=parallel_state.tp.group,
            vocab_size=int(vocab_size),
        )
        for key in output_keys:
            outputs[key].append(sample_outputs[key])

    reduced = {key: reduce_response_values(torch.cat(values, dim=0)) for key, values in outputs.items()}
    return reduced["per_token_loss"], {
        "cross_entropy": reduced["per_token_loss"].detach(),
        "explicit_ce": reduced["explicit_ce"].detach(),
        "rest_ce": reduced["rest_ce"].detach(),
        "teacher_entropy": reduced["teacher_entropy"].detach(),
        "coarse_kl": reduced["coarse_kl"].detach(),
        "teacher_topk_mass": reduced["teacher_topk_mass"].detach(),
        "teacher_rest_mass": reduced["teacher_rest_mass"].detach(),
        "student_topk_mass": reduced["student_topk_mass"].detach(),
        "student_rest_mass": reduced["student_rest_mass"].detach(),
        "rest_mass_abs_error": reduced["rest_mass_abs_error"].detach(),
        "valid_candidates_mean": reduced["valid_candidates"].detach(),
        "valid_position_ratio": reduced["valid_positions"].detach(),
    }
