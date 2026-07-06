"""LLM-judge reward hooks: grade rollouts with an external judge model.

Any instruct model served by sglang (or any OpenAI-compatible endpoint) acts as
the judge; each sample is graded per-request through orbit's custom-reward
hook::

    --custom-rm-path orbit.rollout.llm_judge.reward_func
    --judge-base-url http://<judge-host>:<port>
    --judge-mode equivalence   # or: score

Modes (mirroring the per-sample judge agents in NeMo-RL's Nemotron recipes):

- ``equivalence``: the judge decides whether the response's final answer is
  equivalent to the reference ``sample.label`` (useful where exact string /
  boxed matching fails on freeform short answers). Reward 1.0 / 0.0.
- ``score``: pointwise 0-10 quality grade against the question (and the
  reference answer when a label is present), normalized to [0, 1] — a
  GenRM-lite pointwise signal.

An unparseable judge reply yields reward 0.0 with a warning rather than
aborting the rollout (judges occasionally break format; one flaky reply should
not kill a training step). Group-wise *pairwise* GenRM comparison is not
implemented here — it needs cross-sample orchestration and a different hook
shape.

Judging is deterministic (temperature 0). The judge sees the *last user turn*
of the prompt as the question.
"""

import logging
import re
from argparse import Namespace

from orbit.rollout.scoring_client import post_chat_completions
from orbit.utils.types import Sample

logger = logging.getLogger(__name__)

JUDGE_MODES = {"equivalence", "score"}

_EQUIVALENCE_SYSTEM = (
    "You are a strict grader. Compare a candidate response's final answer to a "
    "reference answer. Judge only whether the final answers are mathematically or "
    "semantically equivalent — ignore formatting, phrasing, and working."
)
_SCORE_SYSTEM = (
    "You are a strict grader. Rate how well a candidate response answers the "
    "question: correctness first, then completeness and clarity."
)

_VERDICT_RE = re.compile(r"VERDICT:\s*(EQUIVALENT|DIFFERENT)", re.IGNORECASE)
_SCORE_RE = re.compile(r"SCORE:\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def _extract_question(prompt) -> str:
    """The question shown to the judge: the last user turn of a chat prompt,
    or the prompt itself when it is a plain string."""
    if isinstance(prompt, str):
        return prompt
    for message in reversed(prompt):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def _build_judge_messages(mode: str, question: str, response: str, label: str | None) -> list[dict[str, str]]:
    if mode == "equivalence":
        if label is None:
            raise ValueError("--judge-mode equivalence requires a reference answer (sample.label is None).")
        return [
            {"role": "system", "content": _EQUIVALENCE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"Reference answer:\n{label}\n\n"
                    f"Candidate response:\n{response}\n\n"
                    "Is the candidate response's final answer equivalent to the reference answer? "
                    "Reason briefly, then reply on the final line with exactly "
                    "`VERDICT: EQUIVALENT` or `VERDICT: DIFFERENT`."
                ),
            },
        ]
    if mode == "score":
        reference = f"Reference answer:\n{label}\n\n" if label is not None else ""
        return [
            {"role": "system", "content": _SCORE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\n"
                    f"{reference}"
                    f"Candidate response:\n{response}\n\n"
                    "Grade the candidate response. Reason briefly, then reply on the final "
                    "line with exactly `SCORE: <integer 0-10>`."
                ),
            },
        ]
    raise ValueError(f"Unknown judge mode: {mode}")


def _parse_equivalence(text: str) -> float | None:
    matches = _VERDICT_RE.findall(text or "")
    if not matches:
        return None
    return 1.0 if matches[-1].upper() == "EQUIVALENT" else 0.0


def _parse_score(text: str) -> float | None:
    matches = _SCORE_RE.findall(text or "")
    if not matches:
        return None
    score = min(max(float(matches[-1]), 0.0), 10.0)
    return score / 10.0


async def reward_func(args: Namespace, sample: Sample, **kwargs) -> float:
    """``--custom-rm-path`` hook: grade one sample with the external judge."""
    mode = getattr(args, "judge_mode", "equivalence")
    if mode not in JUDGE_MODES:
        raise ValueError(f"Unknown judge mode: {mode}")
    base_url = getattr(args, "judge_base_url", None)
    if not base_url:
        raise ValueError("orbit.rollout.llm_judge.reward_func requires --judge-base-url.")

    messages = _build_judge_messages(mode, _extract_question(sample.prompt), sample.response, sample.label)
    reply = await post_chat_completions(
        base_url,
        messages,
        model=getattr(args, "judge_model", "default") or "default",
        temperature=0.0,
        max_tokens=int(getattr(args, "judge_max_tokens", 1024) or 1024),
        timeout_secs=getattr(args, "judge_timeout_secs", None),
    )

    parsed = _parse_equivalence(reply) if mode == "equivalence" else _parse_score(reply)
    if parsed is None:
        logger.warning(
            "LLM judge reply had no parseable %s for sample %s; treating as reward 0.0. Reply tail: %r",
            "verdict" if mode == "equivalence" else "score",
            sample.index,
            (reply or "")[-120:],
        )
        return 0.0
    return parsed
