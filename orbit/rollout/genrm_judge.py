"""Group-wise pairwise GenRM rewards: rank a rollout group with a judge model.

The batch-mode counterpart of ``orbit.rollout.llm_judge`` (which grades one
sample at a time): here the judge compares the *whole n-samples-per-prompt
group* pairwise and each response's reward is its win-rate. This is the hook
shape NeMo-RL's ``genrm_simple_agent`` uses for the Nemotron rlhf/ifbench
blends, whose rows carry the grading rubric in ``metadata["principle"]``::

    --custom-rm-path orbit.rollout.genrm_judge.reward_func
    --group-rm
    --judge-base-url http://<judge-host>:<port>

Mechanics:

- ``--group-rm`` makes ``generate_and_rm_group`` hand the full group to
  ``batched_async_rm``, which calls this ``reward_func(args, samples)``.
- Pairs are judged round-robin in a single order (K*(K-1)/2 calls, fired
  concurrently), deterministically (temperature 0). A win scores 1 point, a
  tie or an unparseable/failed judge reply scores 0.5 for each side —
  fail-soft, one flaky reply should not kill a training step.
- reward_i = wins_i / (K_valid - 1) in [0, 1]. Relative rewards like these
  only make sense within a group; combine with a group-baselined advantage
  estimator (GRPO).
- Empty responses are excluded from judging and score 0.0; a group with fewer
  than two valid responses is neutral (0.5 for the valid one) — zero
  advantage, no gradient.

Evaluation caveat: with ``--n-samples-per-eval-prompt 1`` every eval group is
a singleton, so eval rewards are a constant 0.5 — use ``llm_judge`` score
mode for judge-scored eval instead.
"""

import asyncio
import logging
import re
from argparse import Namespace

from orbit.rollout.llm_judge import _extract_question
from orbit.rollout.scoring_client import post_chat_completions
from orbit.utils.types import Sample

logger = logging.getLogger(__name__)

_WINNER_RE = re.compile(r"WINNER:\s*(A|B|TIE)", re.IGNORECASE)

_PAIRWISE_SYSTEM = (
    "You are a strict pairwise judge. Compare two candidate responses to the same "
    "question and decide which one better satisfies the grading rubric. Judge only "
    "the content of the responses; ignore their order, length, and formatting."
)
_DEFAULT_RUBRIC = "Prefer the response that is more correct, more helpful, and clearer."


def _parse_winner(text: str) -> str | None:
    matches = _WINNER_RE.findall(text or "")
    if not matches:
        return None
    return matches[-1].upper()


def _build_pair_messages(
    rubric: str | None, question: str, response_a: str, response_b: str
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _PAIRWISE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Grading rubric:\n{rubric or _DEFAULT_RUBRIC}\n\n"
                f"Question:\n{question}\n\n"
                f"Response A:\n{response_a}\n\n"
                f"Response B:\n{response_b}\n\n"
                "Which response better satisfies the rubric? Reason briefly, then reply "
                "on the final line with exactly `WINNER: A`, `WINNER: B`, or `WINNER: TIE`."
            ),
        },
    ]


async def _judge_pair(
    args: Namespace, rubric: str | None, question: str, response_a: str, response_b: str
) -> tuple[float, float]:
    """One pairwise comparison -> (points_a, points_b). Fail-soft to a tie."""
    messages = _build_pair_messages(rubric, question, response_a, response_b)
    try:
        reply = await post_chat_completions(
            args.judge_base_url,
            messages,
            model=getattr(args, "judge_model", "default") or "default",
            temperature=0.0,
            max_tokens=int(getattr(args, "judge_max_tokens", 1024) or 1024),
            timeout_secs=getattr(args, "judge_timeout_secs", None),
        )
    except Exception as exc:
        logger.warning("GenRM judge call failed (%s); scoring the pair as a tie.", exc)
        return 0.5, 0.5

    winner = _parse_winner(reply)
    if winner == "A":
        return 1.0, 0.0
    if winner == "B":
        return 0.0, 1.0
    if winner is None:
        logger.warning(
            "GenRM judge reply had no parseable winner; scoring the pair as a tie. Reply tail: %r",
            (reply or "")[-120:],
        )
    return 0.5, 0.5


async def reward_func(args: Namespace, samples: list[Sample], **kwargs) -> list[float]:
    """``--custom-rm-path`` hook (batch mode, requires ``--group-rm``)."""
    if not samples:
        return []
    if not getattr(args, "judge_base_url", None):
        raise ValueError("orbit.rollout.genrm_judge.reward_func requires --judge-base-url.")

    rewards = [0.0] * len(samples)
    valid = [i for i, s in enumerate(samples) if (s.response or "").strip()]
    if len(valid) == 1:
        rewards[valid[0]] = 0.5
        return rewards
    if not valid:
        return rewards

    question = _extract_question(samples[0].prompt)
    metadata = samples[0].metadata if isinstance(samples[0].metadata, dict) else {}
    rubric = metadata.get("principle") or None

    pairs = [(i, j) for pos, i in enumerate(valid) for j in valid[pos + 1 :]]
    results = await asyncio.gather(
        *(_judge_pair(args, rubric, question, samples[i].response, samples[j].response) for i, j in pairs)
    )

    wins = {i: 0.0 for i in valid}
    for (i, j), (points_a, points_b) in zip(pairs, results, strict=True):
        wins[i] += points_a
        wins[j] += points_b

    denom = float(len(valid) - 1)
    for i in valid:
        rewards[i] = wins[i] / denom
    return rewards
