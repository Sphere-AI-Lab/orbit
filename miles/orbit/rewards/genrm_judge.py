"""Group-wise pairwise GenRM rewards: rank a rollout group with a judge model.

The batch-mode counterpart of ``miles.orbit.rewards.llm_judge`` (which grades one
sample at a time): here the judge compares the *whole n-samples-per-prompt
group* pairwise and each response's reward is its win-rate. This is the hook
shape NeMo-RL's ``genrm_simple_agent`` uses for the Nemotron rlhf/ifbench
blends, whose rows carry the grading rubric in ``metadata["principle"]``::

    --custom-rm-path miles.orbit.rewards.genrm_judge.reward_func
    --group-rm
    --judge-base-url http://<judge-host>:<port>

Mechanics:

- ``--group-rm`` makes ``generate_and_rm_group`` hand the full group to
  ``batched_async_rm``, which calls this ``reward_func(args, samples)``.
- Pairs are judged round-robin in a single order (K*(K-1)/2 calls, fired
  concurrently), deterministically (temperature 0). A win scores 1 point and
  an explicit tie scores 0.5 for each side. The judge is constrained to a
  strict JSON verdict; service and protocol failures are surfaced as typed
  grader infrastructure errors.
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
import re
from argparse import Namespace

from miles.orbit.rewards.grader_errors import GraderInfrastructureError, InfrastructureErrorCode
from miles.orbit.rewards.llm_judge import _extract_question
from miles.orbit.rewards.scoring_client import ScoringProtocolError, post_chat_completions
from miles.orbit.ultra.strict_json import loads_strict
from miles.utils.types import Sample

_WINNER_RE = re.compile(r"WINNER: (A|B|TIE)")
_WINNER_JSON_MAX_BYTES = 1024
_WINNER_JSON_MAX_DEPTH = 4
_WINNERS = {"A", "B", "TIE"}

_PAIRWISE_SYSTEM = (
    "You are a strict pairwise judge. Compare two candidate responses to the same "
    "question and decide which one better satisfies the grading rubric. Judge only "
    "the content of the responses; ignore their order, length, and formatting."
)
_DEFAULT_RUBRIC = "Prefer the response that is more correct, more helpful, and clearer."


def _winner_response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "pairwise_winner",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "winner": {"type": "string", "enum": ["A", "B", "TIE"]}
                },
                "required": ["winner"],
                "additionalProperties": False,
            },
        },
    }


def _parse_winner(text: str) -> str | None:
    if not isinstance(text, str) or not text.strip():
        return None
    stripped = text.strip()
    try:
        payload = loads_strict(
            stripped.encode("utf-8"),
            max_bytes=_WINNER_JSON_MAX_BYTES,
            max_depth=_WINNER_JSON_MAX_DEPTH,
        )
    except (UnicodeEncodeError, TypeError, ValueError):
        pass
    else:
        if (
            type(payload) is dict
            and set(payload) == {"winner"}
            and type(payload["winner"]) is str
            and payload["winner"] in _WINNERS
        ):
            return payload["winner"]
        return None

    # Accept the original exact final-line contract for older judge services.
    match = _WINNER_RE.fullmatch(stripped.splitlines()[-1].strip())
    return match.group(1) if match is not None else None


def _build_pair_messages(rubric: str | None, question: str, response_a: str, response_b: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": _PAIRWISE_SYSTEM},
        {
            "role": "user",
            "content": (
                f"Grading rubric:\n{rubric or _DEFAULT_RUBRIC}\n\n"
                f"Question:\n{question}\n\n"
                f"Response A:\n{response_a}\n\n"
                f"Response B:\n{response_b}\n\n"
                "Which response better satisfies the rubric? Return only one JSON object: "
                '{"winner":"A"}, {"winner":"B"}, or {"winner":"TIE"}.'
            ),
        },
    ]


async def _judge_pair(
    args: Namespace, rubric: str | None, question: str, response_a: str, response_b: str
) -> tuple[float, float]:
    """One pairwise comparison -> ``(points_a, points_b)``."""
    messages = _build_pair_messages(rubric, question, response_a, response_b)
    try:
        reply = await post_chat_completions(
            args.judge_base_url,
            messages,
            model=getattr(args, "judge_model", "default") or "default",
            temperature=0.0,
            max_tokens=int(getattr(args, "judge_max_tokens", 1024) or 1024),
            timeout_secs=getattr(args, "judge_timeout_secs", None),
            max_retries=0,
            response_format=_winner_response_format(),
        )
    except GraderInfrastructureError:
        raise
    except ScoringProtocolError as exc:
        raise GraderInfrastructureError(
            InfrastructureErrorCode.PROTOCOL_ERROR,
            grader="genrm",
            stage="judge_response",
            retryable=False,
            safe_detail="GenRM judge returned an invalid response schema",
        ) from exc
    except Exception as exc:
        raise GraderInfrastructureError(
            InfrastructureErrorCode.TRANSPORT_ERROR,
            grader="genrm",
            stage="judge_request",
            retryable=True,
            safe_detail="GenRM judge request failed",
        ) from exc

    winner = _parse_winner(reply)
    if winner == "A":
        return 1.0, 0.0
    if winner == "B":
        return 0.0, 1.0
    if winner is None:
        raise GraderInfrastructureError(
            InfrastructureErrorCode.PROTOCOL_ERROR,
            grader="genrm",
            stage="judge_response",
            retryable=False,
            safe_detail="GenRM judge reply is missing the required winner verdict",
        )
    return 0.5, 0.5


async def reward_func(args: Namespace, samples: list[Sample], **kwargs) -> list[float]:
    """``--custom-rm-path`` hook (batch mode, requires ``--group-rm``)."""
    if not samples:
        return []
    if not getattr(args, "judge_base_url", None):
        raise GraderInfrastructureError(
            InfrastructureErrorCode.CONFIGURATION,
            grader="genrm",
            stage="configuration",
            retryable=False,
            safe_detail="GenRM judge URL is not configured",
        )

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
    tasks = [
        asyncio.create_task(_judge_pair(args, rubric, question, samples[i].response, samples[j].response))
        for i, j in pairs
    ]
    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    wins = {i: 0.0 for i in valid}
    for (i, j), (points_a, points_b) in zip(pairs, results, strict=True):
        wins[i] += points_a
        wins[j] += points_b

    denom = float(len(valid) - 1)
    for i in valid:
        rewards[i] = wins[i] / denom
    return rewards
