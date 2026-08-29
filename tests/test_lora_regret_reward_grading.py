"""The reward the RL campaign actually earns, exercised through `async_rm`.

Every other assertion about E4 is a string check against the launcher. This
file is different on purpose: it runs the reward function, because the failure
it exists to prevent is invisible to a string check. `--rm-type boxed_math`
reads like the obviously right choice for a boxed-answer task, is spelled
correctly, dispatches to the intended pair of functions -- and returns 0 for a
perfectly correct response, always, because both halves extract the box:
`async_rm` strips `\\boxed{...}` down to the bare answer, and then
`grade_answer_verl` calls `extract_answer` on what is left, which returns None
for any string without a `\\boxed` in it.

An all-zero reward is not a loud failure in RL. Advantages are rewards minus
their group mean, so an all-zero group has zero advantage and contributes no
gradient: every arm trains on nothing, every learning rate produces the same
flat line, and the sweep reports a tidy null result. Three rollouts of the E4
probe on 2026-07-31 logged exactly that -- `rollout/rewards: 0.0`,
`passrate/pass@32: 0.0` -- and it read as a base-model or prompt problem.
"""

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

from miles.rollout.rm_hub import async_rm
from miles.utils.types import Sample

RL_LAUNCHER = Path(__file__).resolve().parents[1] / "examples" / "high_precision" / "run-llama3_1-8b-bf16-rl-math-gsm8k.sh"

# A response of the shape the campaign's prompt asks for: reasoning, then the
# final answer inside \boxed{}.
CORRECT_RESPONSE = "He climbs 11*10 + 6*7 = 110 + 42 = 152 steps.\n\nThe final answer is \\boxed{152}."
WRONG_RESPONSE = "He climbs 11 + 6 = 17 steps.\n\nThe final answer is \\boxed{17}."
UNBOXED_RESPONSE = "He climbs 11*10 + 6*7 = 152 steps. The answer is 152."
LABEL = "152"


def _reward(rm_type: str, response: str, label: str = LABEL) -> float:
    args = SimpleNamespace(custom_rm_path=None, rm_type=rm_type, rm_url=None)
    return asyncio.run(async_rm(args, Sample(prompt="ignored", response=response, label=label)))


def _launcher_rm_type() -> str:
    """The RM_TYPE the launcher defaults to, read out of the script itself."""
    match = re.search(r'--rm-type "\$\{RM_TYPE:-([a-z_]+)\}"', RL_LAUNCHER.read_text(encoding="utf-8"))
    assert match, "the RL launcher no longer sets --rm-type with an RM_TYPE default"
    return match.group(1)


def test_the_launchers_configured_reward_can_actually_return_one():
    """The property the whole campaign rests on: under the reward function the
    launcher is *configured with*, a correct answer scores 1.

    Read out of the launcher rather than hardcoded, so that a change back to
    any reward function with an empty positive range fails here in five
    seconds instead of in a 500-node-hour sweep that reports a flat line.
    """
    assert _reward(_launcher_rm_type(), CORRECT_RESPONSE) == 1


def test_a_correct_boxed_response_earns_reward_one():
    assert _reward("math", CORRECT_RESPONSE) == 1


def test_boxed_math_double_extracts_and_can_never_earn_reward():
    """Pins the trap itself, so nobody re-adopts `boxed_` on a grader that
    already extracts. `math` and `dapo` both extract; prefixing either with
    `boxed_` yields a reward function whose range is {0}."""
    assert _reward("boxed_math", CORRECT_RESPONSE) == 0


def test_a_wrong_boxed_response_earns_reward_zero():
    assert _reward("math", WRONG_RESPONSE) == 0


def test_an_unboxed_response_earns_reward_zero():
    """`grade_answer_verl` requires a `\\boxed{...}` in the response, so the
    prompt has to elicit one. This is why the campaign's prompt carries
    ANSWER_INSTRUCTION and why the renderer has to be one a *base* model can
    follow -- a correct answer in prose still scores 0."""
    assert _reward("math", UNBOXED_RESPONSE) == 0


def test_symbolic_answers_grade_by_equivalence_not_string_equality():
    """MATH labels are LaTeX, and the grader has to accept a differently
    spelled equivalent."""
    assert _reward("math", "So the volume is \\boxed{18\\pi}.", label="18\\pi") == 1
    assert _reward("math", "So the volume is \\boxed{\\frac{1}{2}}.", label="0.5") == 1
