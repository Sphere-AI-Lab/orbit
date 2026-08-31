"""Orbit's tool-response update over ``miles.rollout.generate_utils``.

Upstream's ``update_sample_with_tool_responses`` extends five parallel fields on
a sample by the length of the tool observation, and the last of them is
``rollout_log_probs += [0.0] * n``. Orbit legitimately leaves that field ``None``
when it did not ask the engine for logprobs in this phase (evaluation without
``--eval-return-rollout-logprobs``; see ``should_request_rollout_logprobs`` in
generate_utils/generate_endpoint_utils.py), so upstream's unconditional ``+=``
raises ``TypeError: unsupported operand type(s) for +=: 'NoneType' and 'list'``
mid-rollout. Upstream always requests logprobs, so it never sees this.

Expressed as a DELEGATING patch rather than a copy, which needs one trick: the
function returns nothing and does its work by mutation, so orbit cannot fix up a
return value. Instead it lends upstream an empty list to append to and takes it
back afterwards -- upstream still owns all five updates and the tokenisation, and
orbit owns only the decision that a sample tracking no logprobs keeps tracking
none. Copying the body would have been five lines of upstream's arithmetic that
silently stop matching the day upstream adds a sixth field.

Nothing here imports torch or miles at module scope: ``import orbit`` executes
this module and must stay cheap (see orbit/patch/runtime.py).
"""

from __future__ import annotations

from orbit.patch import original, patch_function

_TOOL_CALL_UTILS = "miles.rollout.generate_utils.tool_call_utils"

_REASON = (
    "orbit skips return_logprob for phases that do not need it, so "
    "sample.rollout_log_probs is legitimately None; upstream appends filler to "
    "it unconditionally and raises TypeError on such a sample"
)


@patch_function(
    "miles.rollout.generate_utils.tool_call_utils",
    "update_sample_with_tool_responses",
    upstream_sha="c1e40495a1c209d037971e4cc02abe26974d32cfc22373ca42211c71f91d7b1b",
    reason=_REASON,
)
def update_sample_with_tool_responses(sample, tool_messages, tokenizer):
    tracking_logprobs = sample.rollout_log_probs is not None
    if not tracking_logprobs:
        # Lend upstream something to append to; the filler is discarded below.
        sample.rollout_log_probs = []
    try:
        return original(_TOOL_CALL_UTILS, "update_sample_with_tool_responses")(
            sample, tool_messages, tokenizer
        )
    finally:
        if not tracking_logprobs:
            # Restore the state the caller had: a sample that tracks no logprobs
            # must not come back holding fabricated ones. In a `finally` so an
            # exception from upstream cannot leave the borrowed list behind.
            sample.rollout_log_probs = None
