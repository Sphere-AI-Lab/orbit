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

# The marker upstream's grader splits the response on. `split(...)[-1]` on a
# string that starts with it and contains no other returns the rest verbatim,
# which is how the already-selected solution reaches upstream's grading half
# unchanged. The one input that would not survive the round trip is a solution
# containing `</think>` itself -- the Gemma-4 format cannot produce one, since
# that is the whole reason it has its own reward.
_DEEPSCALER_THINKING_END = "</think>"


def get_gemma_math_reward(response, label):
    from miles.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward

    # Gemma-4 closes thinking with <channel|>; grade text after the last one.
    if GEMMA_THINKING_END in response:
        response = response.split(GEMMA_THINKING_END)[-1]
    return get_deepscaler_rule_based_reward(_DEEPSCALER_THINKING_END + response, label)
