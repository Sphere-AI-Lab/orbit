"""Per-call sampling overrides on ``compute_sampling_params``.

Upstream builds the rollout sampling params from ``args`` alone, so every
generation in a run shares one ``stop`` / ``stop_token_ids`` set and can never
ask for a minimum response length. Orbit needs both to vary per call:

* eval runs configure ``stop``, ``stop_token_ids`` and ``min_new_tokens`` PER
  EVAL DATASET (miles/rollout/inference_rollout/inference_rollout_eval.py reads
  them off the dataset config), and upstream's signature simply drops them;
* OPD scoring asks for a floor on generated tokens.

Mechanism: a DELEGATING patch that GROWS the signature. Upstream still decides
what a sampling-params dict contains and what the args-derived defaults are;
orbit adds three optional keyword arguments and overrides only the keys it was
actually given. ``None`` means "no override", which is why the defaults cannot
simply be read from ``args`` here -- ``args.rollout_stop`` may legitimately be
``None``, and upstream is the one that knows that.

The signature growing is the same shape as
orbit/utils/metric_utils_patches.py's ``k_values``: a static reader of the
vendored ``def`` sees kwargs it has no parameters for, and
tools/check_call_signatures.py resolves patched targets through the patch
registry for exactly that reason.

Key ORDER is preserved as well as key content: ``stop`` and ``stop_token_ids``
already exist in upstream's dict, so assigning to them leaves them where
upstream put them, and ``min_new_tokens`` lands last -- byte-identical to what
the vendored edit produced.

Nothing here imports miles at module scope: ``import orbit`` executes this module
and must stay cheap (see orbit/patch/runtime.py).
"""

from __future__ import annotations

from orbit.patch import original, patch_function

_INFERENCE_ROLLOUT_COMMON = "miles.rollout.inference_rollout.inference_rollout_common"

_REASON = (
    "per-eval-dataset sampling (and OPD scoring) needs stop / stop_token_ids / "
    "min_new_tokens chosen per call; upstream derives all three from args alone "
    "and has no parameter to override them"
)


@patch_function(
    "miles.rollout.inference_rollout.inference_rollout_common",
    "compute_sampling_params",
    upstream_sha="ea44890cceb50ce6da6a84ec684a6f09ca8189474458f3281cff035b9d66da11",
    reason=_REASON,
)
def compute_sampling_params(
    args,
    *,
    temperature,
    top_p,
    top_k,
    max_new_tokens,
    stop=None,
    stop_token_ids=None,
    min_new_tokens=None,
):
    """Upstream's dict, with the caller's overrides applied on top."""
    sampling_params = original(_INFERENCE_ROLLOUT_COMMON, "compute_sampling_params")(
        args,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
    )
    if stop is not None:
        sampling_params["stop"] = stop
    if stop_token_ids is not None:
        sampling_params["stop_token_ids"] = stop_token_ids
    if min_new_tokens is not None:
        sampling_params["min_new_tokens"] = min_new_tokens
    return sampling_params
