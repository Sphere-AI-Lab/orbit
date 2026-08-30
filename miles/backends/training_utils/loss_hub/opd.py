from argparse import Namespace

import torch

# ORBIT-SEAM: the OPD reverse-KL advantage blend has a single implementation, orbit's home
# `orbit.opd.advantages.apply_opd_kl_to_advantages` (re-exported by loss_hub.math_utils and
# imported from there by the orbit OPD tests). Upstream's own copy of the same algorithm used
# to live in this module; it is gone. What remains here is the thin adapter that keeps
# upstream's call signature (`args`-keyed, used by loss.py and tests/fast/backends/.../loss)
# working on top of the home's `opd_kl_coef`-keyed one, plus upstream's defensive
# consumer-boundary detach for callers that bypass compute_advantages_and_returns' own
# persistent-data detach.
from miles.backends.training_utils.loss_hub.math_utils import (
    apply_opd_kl_to_advantages as _apply_opd_kl_to_advantages_home,
)
from miles.utils.types import RolloutBatch


def apply_opd_kl_to_advantages(
    args: Namespace,
    rollout_data: RolloutBatch,
    advantages: list[torch.Tensor],
    student_log_probs: list[torch.Tensor] | None,
) -> None:
    """Apply on-policy distillation KL penalty to advantages.

    Computes reverse KL (student_logp - teacher_logp) and adds weighted penalty
    to advantages in-place. This is orthogonal to the base advantage estimator.

    Args:
        args: Configuration containing `opd_kl_coef` (and `opd_type`, used only
            for the missing-teacher error message).
        rollout_data: Dict containing "teacher_log_probs" (or a precomputed
            "opd_reverse_kl").
        advantages: List of advantage tensors to modify in-place.
        student_log_probs: List of old-student log-probability tensors. OPD
            treats these as fixed scoring inputs.

    References:
        https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/distillation/train_on_policy.py
    """
    if student_log_probs is None:
        return

    # Defensive consumer boundary for direct callers that bypass
    # compute_advantages_and_returns' persistent-data detach. No-op on the
    # in-tree path, where those lists are already detached.
    precomputed_reverse_kls = rollout_data.get("opd_reverse_kl")
    if precomputed_reverse_kls is not None:
        rollout_data["opd_reverse_kl"] = [
            reverse_kl.detach() if torch.is_tensor(reverse_kl) else torch.tensor(reverse_kl, dtype=torch.float32)
            for reverse_kl in precomputed_reverse_kls
        ]
    else:
        teacher_log_probs = rollout_data.get("teacher_log_probs")
        if teacher_log_probs is None:
            raise ValueError(f"OPD with opd_type='{args.opd_type}' requires teacher_log_probs, but it is missing.")
        rollout_data["teacher_log_probs"] = [t.detach() for t in teacher_log_probs]
        # Base guard the home does not carry: the home only compares teacher against student,
        # so a broadcastable per-sample scalar advantage would silently expand here.
        for i, adv in enumerate(advantages[: len(student_log_probs)]):
            if adv.shape != student_log_probs[i].shape:
                raise ValueError(
                    f"OPD shape mismatch at sample {i}: advantages={tuple(adv.shape)}, "
                    f"student_log_probs={tuple(student_log_probs[i].shape)}. "
                    "OPD expects per-token advantages; broadcast scalar advantages must be expanded "
                    "before this call."
                )

    _apply_opd_kl_to_advantages_home(
        args.opd_kl_coef,
        rollout_data,
        advantages,
        [log_prob.detach() for log_prob in student_log_probs],
    )
