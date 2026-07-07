"""Agentic SWE episode loop: orbit custom-generate function (rung 2b).

Design doc: docs/plans/2026-07-07-swe-rung2b-agentic-loop.md. Pattern
adapted from slime-agentic (AgentFlow/MemAgent/ToolOrchestra) — the same
``--custom-generate-function-path`` seam, with one growing token stream per
episode instead of per-turn concatenation:

- model turns append generated ids (loss_mask 1, real rollout logprobs);
- tool-result turns are rendered via chat-template suffix delta and appended
  (loss_mask 0, logprob 0.0);
- the reward (SWE-bench verification of the final session repo state) is set
  in-episode, so no RM hook is needed.

Wire-up (dedicated swe runs)::

    --custom-generate-function-path orbit.rollout.swe_agent.episode.generate
    --swe-rm-sif-cache /path/to/sif_cache
    [--swe-agent-max-turns 12] [--swe-agent-cmd-timeout-secs 30]
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from argparse import Namespace
from typing import Any

from orbit.rollout.sglang_rollout import GenerateState
from orbit.rollout.swe_agent.container_session import ContainerSession, sif_for_instance
from orbit.utils.http_utils import post
from orbit.utils.types import Sample

logger = logging.getLogger(__name__)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

# Per-episode wall clock (ToolOrchestra pattern): a stuck episode must not
# stall the rollout batch.
_EPISODE_TIMEOUT_SECS = 900.0

_SYSTEM_PROMPT = """You are an expert software engineer fixing a GitHub issue inside the repository's own environment.

You interact through tool calls. Available tools:

1. run_shell — run one shell command in the repository root and see its output.
2. submit — declare the fix complete (your edits to the working tree will be tested).

Reply with EXACTLY ONE tool call per turn, formatted as:
<tool_call>
{"name": "run_shell", "arguments": {"command": "<shell command>"}}
</tool_call>
or
<tool_call>
{"name": "submit", "arguments": {}}
</tool_call>

Explore the code (grep, cat, sed), reproduce the problem if useful, EDIT files
(e.g. with sed -i, or cat > file << 'EOF' ... EOF), verify, then submit.
Do not modify the test suite. Keep commands short; output is truncated."""


def _parse_action(text: str) -> dict | None:
    matches = _TOOL_CALL_RE.findall(text or "")
    if not matches:
        return None
    try:
        call = json.loads(matches[-1])
    except json.JSONDecodeError:
        return None
    return call if isinstance(call, dict) else None


def _template_ids(tokenizer, messages: list[dict], add_generation_prompt: bool) -> list[int]:
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=add_generation_prompt
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


async def _model_turn(args: Namespace, token_ids: list[int], sampling_params: dict) -> dict:
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    payload = {
        "input_ids": token_ids,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    return await post(url, payload)


async def _run_episode(args: Namespace, sample: Sample, sampling_params: dict) -> None:
    state = GenerateState(args)
    tokenizer = state.tokenizer

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    swe = metadata.get("swe") or {}
    if not swe.get("image_name"):
        raise ValueError(f"swe_agent: sample {sample.index} has no metadata['swe']['image_name'].")
    cache_dir = getattr(args, "swe_rm_sif_cache", None)
    if not cache_dir:
        raise ValueError("swe_agent requires --swe-rm-sif-cache.")
    sif = sif_for_instance(cache_dir, swe["image_name"])

    max_turns = int(getattr(args, "swe_agent_max_turns", 12) or 12)
    cmd_timeout = float(getattr(args, "swe_agent_cmd_timeout_secs", 30) or 30)
    verify_timeout = float(getattr(args, "swe_rm_timeout_secs", 300) or 300)
    response_budget = int(args.rollout_max_response_len)

    # Conversation both as messages (for template deltas) and as one token stream.
    if isinstance(sample.prompt, list):
        user_messages = [m for m in sample.prompt if m.get("role") == "user"]
        issue_text = user_messages[-1]["content"] if user_messages else str(sample.prompt)
    else:
        issue_text = str(sample.prompt)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": issue_text},
    ]
    stream = _template_ids(tokenizer, messages, add_generation_prompt=True)
    prompt_len = len(stream)
    loss_mask: list[int] = []
    log_probs: list[float] = []

    session = ContainerSession(sif, cmd_timeout_secs=cmd_timeout)
    sample.reward = 0.0
    truncated = False
    try:
        if not await session.start():
            sample.status = Sample.Status.FAILED
            return

        for _turn in range(max_turns):
            remaining = response_budget - (len(stream) - prompt_len)
            if remaining <= 0:
                truncated = True
                break
            turn_params = dict(sampling_params)
            turn_params["max_new_tokens"] = min(
                int(turn_params.get("max_new_tokens") or remaining), remaining
            )
            out = await _model_turn(args, stream, turn_params)
            meta = out["meta_info"]
            turn_ids = [t[1] for t in meta["output_token_logprobs"]]
            turn_lps = [t[0] for t in meta["output_token_logprobs"]]
            turn_text = out["text"]

            stream += turn_ids
            loss_mask += [1] * len(turn_ids)
            log_probs += turn_lps
            messages.append({"role": "assistant", "content": turn_text})

            if meta["finish_reason"]["type"] == "length":
                truncated = True
                break

            action = _parse_action(turn_text)
            if action is None or action.get("name") == "submit":
                break
            if action.get("name") != "run_shell":
                tool_out = f"Unknown tool {action.get('name')!r}. Use run_shell or submit."
            else:
                command = (action.get("arguments") or {}).get("command") or ""
                if not command.strip():
                    tool_out = "Empty command."
                else:
                    rc, output = await session.run(command)
                    tool_out = f"exit_code: {rc}\n{output}"

            # Tool turn: template-suffix delta appended with mask 0. The
            # stream keeps the assistant ids AS GENERATED (re-rendering can
            # differ token-wise), so the delta is computed relative to the
            # canonical render *up to and including* the assistant message
            # and spliced onto the generated stream. (The canonical render
            # ends "...<|im_end|>\n" while the generated stream ends with the
            # stop token only — one boundary-newline token of drift, harmless.)
            prev_ids = _template_ids(tokenizer, messages, add_generation_prompt=False)
            messages.append({"role": "tool", "content": tool_out})
            with_tool_ids = _template_ids(tokenizer, messages, add_generation_prompt=True)
            if with_tool_ids[: len(prev_ids)] != prev_ids:
                # Qwen templates are append-only; violation means the token
                # stream would not match what the engine saw — abort loudly.
                raise RuntimeError("chat template is not append-only; cannot maintain token stream")
            tool_delta = with_tool_ids[len(prev_ids) :]
            stream += tool_delta
            loss_mask += [0] * len(tool_delta)
            log_probs += [0.0] * len(tool_delta)

        # Grade the final repo state (test_patch conflict with model-edited
        # tests => fail).
        passed = await session.verify(swe, timeout_secs=verify_timeout)
        sample.reward = 1.0 if passed else 0.0
    finally:
        await session.stop()

    sample.tokens = stream
    sample.response_length = len(stream) - prompt_len
    sample.response = tokenizer.decode(stream[prompt_len:])
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = log_probs
    sample.status = Sample.Status.TRUNCATED if truncated else Sample.Status.COMPLETED


async def generate(
    args: Namespace, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False
) -> Sample:
    """orbit ``--custom-generate-function-path`` entry point."""
    try:
        await asyncio.wait_for(_run_episode(args, sample, dict(sampling_params)), timeout=_EPISODE_TIMEOUT_SECS)
    except asyncio.TimeoutError:
        logger.warning("swe_agent: episode timed out for sample %s", sample.index)
        sample.reward = 0.0
        sample.status = Sample.Status.FAILED
    except Exception:
        logger.exception("swe_agent: episode crashed for sample %s", sample.index)
        sample.reward = 0.0
        sample.status = Sample.Status.FAILED
    return sample
