"""On-policy-distillation advantage shaping (orbit-only).

Lifted verbatim out of ``miles/utils/ppo_utils.py`` (Phase-3 slice 3e, P1
lift-out): the MOPD advantage, the blend-form reverse-KL penalty, the shared
ICE-POP hard gate and the OPD gate that consumes it.
``miles.utils.ppo_utils`` re-exports all four names behind a stamped seam, so
existing importers (``miles/backends/training_utils/loss.py``, the OPD tests)
are unchanged.

No module-level ``miles.*`` imports: this module depends only on torch.
"""

import torch


def opd_mopd_advantages(
    rollout_data: dict,
    student_log_probs: list[torch.Tensor],
    response_lengths: list[int],
) -> list[torch.Tensor]:
    """Pure on-policy distillation (MOPD) advantage: teacher_logp - student_logp.

    Args:
        rollout_data: Rollout batch dict; must contain `teacher_log_probs`
            (list[torch.Tensor], one per sample, aligned to `response_lengths`).
        student_log_probs: Current policy log-probs per sample.
        response_lengths: Response length per sample.

    Returns:
        list[torch.Tensor]: `teacher_i[-L_i:] - student_i` per sample.
    """
    precomputed_reverse_kls = rollout_data.get("opd_reverse_kl")
    if precomputed_reverse_kls is not None:
        # Top-k OPD: rollout-side scoring already computed the per-token
        # weighted reverse KL; the MOPD advantage is simply its negation.
        if len(precomputed_reverse_kls) != len(student_log_probs):
            raise ValueError(
                f"OPD length mismatch: opd_reverse_kl={len(precomputed_reverse_kls)} "
                f"student={len(student_log_probs)}"
            )
        device = student_log_probs[0].device
        out = []
        for i, reverse_kl in enumerate(precomputed_reverse_kls):
            if not torch.is_tensor(reverse_kl):
                reverse_kl = torch.tensor(reverse_kl, dtype=torch.float32)
            reverse_kl = reverse_kl.to(device=device)
            if reverse_kl.shape != student_log_probs[i].shape:
                raise ValueError(
                    f"OPD shape mismatch at {i}: opd_reverse_kl={tuple(reverse_kl.shape)} "
                    f"student={tuple(student_log_probs[i].shape)}"
                )
            out.append(-reverse_kl)
        return out

    teacher_log_probs = rollout_data.get("teacher_log_probs")
    if teacher_log_probs is None:
        raise ValueError(
            "advantage_estimator='on_policy_distillation' needs teacher_log_probs. "
            "Enable a teacher producer with --opd-type {megatron,sglang}."
        )
    if len(teacher_log_probs) != len(student_log_probs):
        raise ValueError(f"OPD length mismatch: teacher={len(teacher_log_probs)} student={len(student_log_probs)}")
    device = student_log_probs[0].device
    out = []
    for i, (t, s, response_length) in enumerate(
        zip(teacher_log_probs, student_log_probs, response_lengths, strict=False)
    ):
        t = t.to(device=device)[-response_length:]
        if t.shape != s.shape:
            raise ValueError(f"OPD shape mismatch at {i}: teacher={tuple(t.shape)} student={tuple(s.shape)}")
        out.append(t - s)
    return out


def apply_opd_kl_to_advantages(
    opd_kl_coef: float,
    rollout_data: dict,
    advantages: list[torch.Tensor],
    student_log_probs: list[torch.Tensor],
) -> None:
    """Blend an on-policy-distillation reverse-KL penalty into base advantages, in place.

    `adv_i -= opd_kl_coef * (student_i - teacher_i)`, i.e. the blend form of OPD
    (slime/miles): any base estimator's advantage combined with distillation
    toward the teacher. Also stores the per-sample reverse KL in
    `rollout_data["opd_reverse_kl"]`.

    Args:
        opd_kl_coef: Coefficient for the reverse-KL penalty.
        rollout_data: Rollout batch dict; must contain `teacher_log_probs`
            (list[torch.Tensor], one per sample, aligned to `student_log_probs`).
        advantages: Base-estimator advantages per sample; mutated in place.
        student_log_probs: Current policy log-probs per sample, or None (e.g. on
            the critic with KL off) — the blend is then a silent no-op.
    """
    if student_log_probs is None:
        return

    precomputed_reverse_kls = rollout_data.get("opd_reverse_kl")
    if precomputed_reverse_kls is not None:
        # Top-k OPD: consume the rollout-side precomputed per-token reverse KL.
        if len(advantages) != len(precomputed_reverse_kls):
            raise ValueError(
                f"OPD length mismatch: advantages={len(advantages)}, "
                f"opd_reverse_kl={len(precomputed_reverse_kls)}."
            )
        reverse_kls = []
        for i, adv in enumerate(advantages):
            reverse_kl = precomputed_reverse_kls[i]
            if not torch.is_tensor(reverse_kl):
                reverse_kl = torch.tensor(reverse_kl, dtype=torch.float32)
            reverse_kl = reverse_kl.to(device=adv.device)
            if adv.shape != reverse_kl.shape:
                raise ValueError(
                    f"OPD shape mismatch at sample {i}: advantages={tuple(adv.shape)}, "
                    f"opd_reverse_kl={tuple(reverse_kl.shape)}."
                )
            advantages[i] = adv - opd_kl_coef * reverse_kl
            reverse_kls.append(reverse_kl)
        rollout_data["opd_reverse_kl"] = reverse_kls
        return

    teacher_log_probs = rollout_data.get("teacher_log_probs")
    if teacher_log_probs is None:
        raise ValueError("--use-opd requires teacher_log_probs; enable a teacher producer (--opd-type).")
    if not (len(advantages) == len(student_log_probs) == len(teacher_log_probs)):
        raise ValueError(
            f"OPD length mismatch: advantages={len(advantages)}, "
            f"student_log_probs={len(student_log_probs)}, teacher_log_probs={len(teacher_log_probs)}."
        )
    device = student_log_probs[0].device
    reverse_kls = []
    for i, adv in enumerate(advantages):
        t = teacher_log_probs[i].to(device=device)
        s = student_log_probs[i]
        if t.shape != s.shape:
            raise ValueError(f"OPD shape mismatch at {i}: teacher={tuple(t.shape)} student={tuple(s.shape)}")
        reverse_kl = s - t
        advantages[i] = adv - opd_kl_coef * reverse_kl
        reverse_kls.append(reverse_kl)
    rollout_data["opd_reverse_kl"] = reverse_kls


def icepop_gate(ratio: torch.Tensor, clip_low: float, clip_high: float) -> torch.Tensor:
    """ICE-POP hard gate: pass the importance ratio through inside the band, zero outside.

    Shared masking core used by both the policy-gradient ICE-POP path
    (``icepop_function`` in loss.py) and the OPD advantage gate
    (``apply_opd_icepop_gate``). Tokens whose ``ratio`` lies in
    ``[clip_low, clip_high]`` keep ``ratio`` (importance weight); tokens outside
    the band are zeroed (hard-gated).

    Args:
        ratio: Per-token importance ratio ``exp(train_logp - rollout_logp)``.
        clip_low: Lower band edge (``args.tis_clip_low``).
        clip_high: Upper band edge (``args.tis_clip``).

    Returns:
        Per-token gate weight, same shape as ``ratio``.
    """
    return torch.where(
        (ratio >= clip_low) & (ratio <= clip_high), ratio, torch.zeros_like(ratio)
    )


def apply_opd_icepop_gate(
    rollout_data: dict,
    advantages: list[torch.Tensor],
    clip_low: float,
    clip_high: float,
) -> None:
    """Hard-gate the OPD advantage by the per-token train/rollout ratio, in place.

    For async / off-policy rollouts the acting (rollout) policy drifts from the
    current student, biasing the OPD advantage. Following NeMo-RL MOPD, gate each
    token by the ICE-POP importance ratio ``exp(train_logp - rollout_logp)``:
    in-band tokens are reweighted by the ratio, out-of-band tokens are zeroed —
    reusing the exact ratio and gate as orbit's policy-gradient path
    (``icepop_function``).

    Args:
        rollout_data: Rollout batch dict; must contain per-sample ``log_probs``
            (train-recomputed student log-probs) and ``rollout_log_probs``
            (acting-policy log-probs), aligned to ``advantages``.
        advantages: OPD advantages per sample; mutated in place.
        clip_low: Lower band edge (``args.tis_clip_low``).
        clip_high: Upper band edge (``args.tis_clip``).
    """
    train_log_probs = rollout_data.get("log_probs")
    rollout_log_probs = rollout_data.get("rollout_log_probs")
    if train_log_probs is None or rollout_log_probs is None:
        raise ValueError(
            "--opd-icepop needs train log_probs and rollout_log_probs to form the "
            "train/rollout importance ratio, but one is missing from rollout_data. "
            "Ensure --use-rollout-logprobs is off (so student log-probs are recomputed) "
            "and rollout log-probs are collected."
        )
    for i in range(len(advantages)):
        ratio = torch.exp(train_log_probs[i] - rollout_log_probs[i])
        advantages[i] = advantages[i] * icepop_gate(ratio, clip_low, clip_high)


__all__ = [
    "apply_opd_icepop_gate",
    "apply_opd_kl_to_advantages",
    "icepop_gate",
    "opd_mopd_advantages",
]
