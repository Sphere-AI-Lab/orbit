"""Gemma-4 math reward, moved out of ``miles/rollout/rm_hub/deepscaler.py``.

A LIFT: upstream has no gemma reward at all. What kept the vendored file dirty
was not this function but the refactor around it -- orbit split upstream's
``get_deepscaler_rule_based_reward`` into a ``_grade_boxed_solution`` tail plus a
thin front end so the gemma reward could reuse the tail. That split changed no
behaviour whatsoever (the front end still selects on ``</think>`` / ``###Response``
and returns 0 otherwise, then runs the identical grading body), so the vendored
function is simply restored to upstream's and nothing patches it.

Reuse without the split is what this module does instead: upstream's grader is
``<select the solution text><grade the boxed answer in it>`` with no entry point
for the second half on its own, so orbit hands it a string whose selection is a
no-op and lets upstream's ~25 lines of boxed-answer grading be the code that
actually runs. Copying that body into orbit would have been the alternative, and
it would have started drifting the day upstream touched it.
"""

from __future__ import annotations

# Gemma-4 closes its thinking block with this instead of `</think>`.
GEMMA_THINKING_END = "<channel|>"

# The marker upstream's grader splits the response on. Prefixing the selected
# solution with it makes upstream's `split(...)[-1]` a no-op, which is how the
# solution reaches upstream's grading half unchanged.
#
# It is only a no-op if the solution carries no `</think>` of its own: upstream
# splits on the LAST one, and `extract_answer` reads the LAST `\boxed`, so a
# solution like `\boxed{42} </think> no answer here` would grade the empty tail
# and score 0 where orbit-main scored 1. In the Gemma-4 format `</think>` is not
# a marker at all -- it is ordinary response text -- so it is stripped before the
# round trip rather than left to act as one.
_DEEPSCALER_THINKING_END = "</think>"


def get_gemma_math_reward(response, label):
    from miles.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward

    # Gemma-4 closes thinking with <channel|>; grade text after the last one.
    if GEMMA_THINKING_END in response:
        response = response.split(GEMMA_THINKING_END)[-1]
    # Neutralise any `</think>` in the selection so the prefix below is the only
    # one upstream can split on. `</think>` is brace-free, so removing it cannot
    # disturb the `\boxed{...}` brace matching that grading depends on.
    response = response.replace(_DEEPSCALER_THINKING_END, "")
    return get_deepscaler_rule_based_reward(_DEEPSCALER_THINKING_END + response, label)
