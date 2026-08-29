"""Critic explained-variance statistics (orbit-only).

Lifted verbatim out of ``miles/utils/ppo_utils.py`` (Phase-3 slice 3e, P1
lift-out). ``miles.utils.ppo_utils`` re-exports every name below behind a
stamped seam, so existing importers (``miles/backends/training_utils/loss.py``,
``miles/backends/training_utils/log_utils.py``, the fast tests) are unchanged.
"""

import math

# Critic explained-variance metric plumbing.
#
# EV = 1 - Var(returns - values) / Var(returns), over trainable (unmasked)
# tokens of the whole optimizer step. Per-micro-batch EV values cannot simply
# be averaged (micro-batches differ in token count and mean), so
# value_loss_function emits the masked token-level sufficient statistics below
# per micro-batch; aggregate_train_losses SUM-reduces them across micro-batches
# and DP/CP ranks (applying one normalization constant shared by all metrics,
# which cancels in the ratios) and folds them into VALUE_EV_METRIC_KEY via
# compute_value_explained_var.
VALUE_EV_STAT_KEYS: tuple[str, ...] = (
    "value_ev/token_count",
    "value_ev/return_sum",
    "value_ev/return_sumsq",
    "value_ev/err_sum",
    "value_ev/err_sumsq",
)
VALUE_EV_METRIC_KEY = "value_explained_var"
_VALUE_EV_MIN_RETURN_VAR = 1e-8


def compute_value_explained_var(
    token_count: float,
    return_sum: float,
    return_sumsq: float,
    err_sum: float,
    err_sumsq: float,
) -> float:
    """Compute EV = 1 - Var(returns - values) / Var(returns) from token sums.

    The five inputs are masked token-level sums (population statistics). They
    may all carry one common positive scale factor — e.g. the
    `cp_size / num_samples_or_tokens` normalization that aggregate_train_losses
    applies to every metric — since it cancels in every ratio below.

    Degenerate cases return 0.0 by convention so logs never carry NaN/inf:
    no trainable tokens, (near-)constant returns, or non-finite statistics.
    """
    if not all(map(math.isfinite, (token_count, return_sum, return_sumsq, err_sum, err_sumsq))):
        return 0.0
    if token_count <= 0.0:
        return 0.0
    return_mean = return_sum / token_count
    return_var = return_sumsq / token_count - return_mean**2
    if return_var <= _VALUE_EV_MIN_RETURN_VAR:
        return 0.0
    err_mean = err_sum / token_count
    err_var = max(err_sumsq / token_count - err_mean**2, 0.0)
    return 1.0 - err_var / return_var


__all__ = [
    "VALUE_EV_METRIC_KEY",
    "VALUE_EV_STAT_KEYS",
    "compute_value_explained_var",
]
