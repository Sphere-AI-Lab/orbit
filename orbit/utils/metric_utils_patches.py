"""Orbit's two additions to ``miles.utils.metric_utils.compute_pass_rate``.

Both used to be edits inside the vendored function; expressing them from here is
what makes that file byte-pristine again.

* ``scale`` -- PEFT-Arena reports pass@k as a percentage alongside ``acc``, so
  the whole result is multiplied. Pure post-processing on upstream's dict.

* ``k_values`` -- an explicit list of k, because eval configs ask for k the
  upstream set does not contain (``--eval-pass-k-values 1 4 8`` is common, and
  nothing stops a 3). Upstream derives its k from ``group_size`` alone:
  1, 2, 4, ... up to the group. That is not a list a caller can steer, so orbit
  adds the case rather than replacing the function -- upstream still computes
  every power-of-two k, orbit fills in only the k upstream cannot produce, and
  the estimator itself stays upstream's (``_estimate_pass_at_k``). Ask for
  ``[1, 4]`` and every returned number came out of upstream's body.

The signature therefore GROWS, which no other orbit patch does, and a static
reader of a call site sees kwargs the vendored ``def`` has no parameters for.
tools/check_call_signatures.py resolves patched targets through the patch
registry for exactly this reason; without that it would (correctly, on the
vendored text alone) call every ``k_values=`` call site a TypeError.

Nothing here imports numpy or miles at module scope: ``import orbit`` executes
this module and must stay cheap (see orbit/patch/runtime.py).
"""

from __future__ import annotations

from orbit.patch import original, patch_function

_METRIC_UTILS = "miles.utils.metric_utils"

_REASON = (
    "eval reporting needs pass@k at caller-chosen k (and as a percentage); "
    "upstream derives its k set from group_size alone, so a k that is not a "
    "power of two has no upstream spelling"
)


def _fill_missing_k(flat_rewards, group_size, num_groups, ks):
    """pass@k for the k upstream did not compute, with upstream's estimator.

    Only the k set is orbit's. ``_estimate_pass_at_k`` is looked up on the
    module at call time rather than imported at the top, so a rename upstream
    fails loudly here instead of at ``import orbit``.
    """
    import numpy as np

    from miles.utils import metric_utils

    if num_groups is None:
        num_groups = len(flat_rewards) // group_size
    rewards_of_group = np.array(flat_rewards).reshape(num_groups, group_size)
    num_correct = np.sum(rewards_of_group == 1, axis=1)
    num_samples = np.full(num_groups, group_size)
    return {
        k: np.mean(metric_utils._estimate_pass_at_k(num_samples, num_correct, k))
        for k in ks
    }


def _at_requested_k(log_dict, flat_rewards, group_size, num_groups, k_values):
    # Upstream returns nothing at all for a group of one; asking for explicit k
    # does not change that there is no group to estimate over.
    if not log_dict:
        return {}
    wanted = sorted({int(k) for k in k_values if 1 <= int(k) <= group_size})
    if not wanted:
        return {}
    missing = [k for k in wanted if f"pass@{k}" not in log_dict]
    filled = _fill_missing_k(flat_rewards, group_size, num_groups, missing) if missing else {}
    return {f"pass@{k}": log_dict.get(f"pass@{k}", filled.get(k)) for k in wanted}


@patch_function(
    "miles.utils.metric_utils",
    "compute_pass_rate",
    upstream_sha="91ab64968ba701e2330990bea1413a70bbf13e27d3472e185bfb1b01a48a9a40",
    reason=_REASON,
)
def compute_pass_rate(flat_rewards, group_size, num_groups=None, k_values=None, scale=1.0):
    log_dict = original(_METRIC_UTILS, "compute_pass_rate")(
        flat_rewards, group_size, num_groups
    )
    if k_values is not None:
        log_dict = _at_requested_k(log_dict, flat_rewards, group_size, num_groups, k_values)
    if scale != 1.0:
        log_dict = {name: value * scale for name, value in log_dict.items()}
    return log_dict
