import torch


def dense_topk_rest_oracle(
    logits: torch.Tensor,
    teacher_token_ids: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    teacher_valid_mask: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    vocab_size: int,
    mass_tolerance: float = 1e-6,
) -> dict[str, torch.Tensor]:
    """Independent FP64 reference for the coarse teacher Top-K + Rest CE.

    This intentionally uses a dense probability-space construction rather
    than the production Stable TP form. It is test-only and must never be used
    by the training path.
    """
    if logits.ndim != 2 or logits.size(-1) < vocab_size or vocab_size <= 0:
        raise ValueError(f"Oracle logits/vocab mismatch: logits={tuple(logits.shape)}, vocab_size={vocab_size}.")

    logits_fp64 = logits[:, :vocab_size].to(torch.float64)
    teacher_token_ids = torch.as_tensor(teacher_token_ids, device=logits.device, dtype=torch.long).detach()
    teacher_log_probs = torch.as_tensor(teacher_log_probs, device=logits.device, dtype=torch.float64).detach()
    teacher_valid_mask = torch.as_tensor(teacher_valid_mask, device=logits.device, dtype=torch.bool).detach()
    response_mask = torch.as_tensor(response_mask, device=logits.device, dtype=torch.bool)

    target_shape = teacher_token_ids.shape
    if (
        teacher_token_ids.ndim != 2
        or teacher_log_probs.shape != target_shape
        or teacher_valid_mask.shape != target_shape
        or response_mask.shape != (logits.size(0),)
        or target_shape[0] != logits.size(0)
    ):
        raise ValueError(
            "Oracle target shape mismatch: "
            f"logits={tuple(logits.shape)}, ids={target_shape}, "
            f"log_probs={tuple(teacher_log_probs.shape)}, valid_mask={tuple(teacher_valid_mask.shape)}, "
            f"response_mask={tuple(response_mask.shape)}."
        )

    candidate_mask = teacher_valid_mask & response_mask.unsqueeze(-1)
    active_log_probs = teacher_log_probs[candidate_mask]
    active_ids = teacher_token_ids[candidate_mask]
    if active_log_probs.numel() and not torch.isfinite(active_log_probs).all().item():
        raise ValueError("Oracle teacher Top-K log-probs must be finite on valid positions.")
    if active_ids.numel() and ((active_ids < 0) | (active_ids >= vocab_size)).any().item():
        raise ValueError(f"Oracle teacher token IDs must be in [0, {vocab_size}).")

    for row in range(teacher_token_ids.size(0)):
        row_ids = teacher_token_ids[row][candidate_mask[row]]
        if row_ids.numel() != torch.unique(row_ids).numel():
            raise ValueError(f"Oracle teacher Top-K row {row} contains duplicate token IDs.")

    safe_log_probs = torch.where(candidate_mask, teacher_log_probs, teacher_log_probs.new_zeros(()))
    teacher_probs = torch.where(candidate_mask, safe_log_probs.exp(), teacher_log_probs.new_zeros(()))
    teacher_topk_mass = teacher_probs.sum(dim=-1)
    if (teacher_topk_mass > 1.0 + mass_tolerance).any().item():
        raise ValueError("Oracle teacher Top-K probability mass exceeds 1.")
    teacher_rest_mass = (1.0 - teacher_topk_mass).clamp_min(0.0)

    student_probs = torch.softmax(logits_fp64, dim=-1)
    safe_ids = torch.where(candidate_mask, teacher_token_ids, teacher_token_ids.new_zeros(()))
    student_candidate_probs = student_probs.gather(dim=-1, index=safe_ids)

    selected_mask = torch.zeros_like(student_probs, dtype=torch.bool)
    for candidate_rank in range(teacher_token_ids.size(1)):
        valid_rows = candidate_mask[:, candidate_rank]
        selected_mask[valid_rows, teacher_token_ids[valid_rows, candidate_rank]] = True
    student_rest_mass = student_probs.masked_fill(selected_mask, 0.0).sum(dim=-1)

    explicit_ce = -torch.xlogy(teacher_probs, student_candidate_probs).sum(dim=-1)
    rest_ce = -torch.xlogy(teacher_rest_mass, student_rest_mass)
    per_token_loss = torch.where(
        response_mask,
        explicit_ce + rest_ce,
        logits_fp64[:, 0] * 0.0,
    )

    return {
        "per_token_loss": per_token_loss,
        "explicit_ce": torch.where(response_mask, explicit_ce, explicit_ce.new_zeros(())),
        "rest_ce": torch.where(response_mask, rest_ce, rest_ce.new_zeros(())),
        "teacher_topk_mass": torch.where(response_mask, teacher_topk_mass, teacher_topk_mass.new_zeros(())),
        "teacher_rest_mass": torch.where(response_mask, teacher_rest_mass, teacher_rest_mass.new_zeros(())),
        "student_rest_mass": torch.where(response_mask, student_rest_mass, student_rest_mass.new_zeros(())),
    }
