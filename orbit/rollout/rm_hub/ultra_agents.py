"""Rule-based graders for Nemotron-RL-Ultra single-turn agents.

NVIDIA decomposed most "agentic" Ultra training into single-turn rows whose
grading needs only a parser and a comparator — no environment:

- ``*single_step_tool_use_with_argument_comparison_agent``: the row carries
  ``expected_action``. For ``function_call`` actions the model's emitted tool
  call (Qwen ``<tool_call>{...}</tool_call>`` format) must match by name and
  deep-equal parsed arguments. For ``message`` actions the model is rewarded
  for NOT calling a tool (a non-empty text reply) — the call-vs-respond
  decision, deliberately not grading text similarity (that would need a
  judge).
- ``mcqa_simple_agent``: extract the answer letter with the row's
  ``template_metadata.output_regex`` (last match wins) and compare
  case-insensitively with ``expected_answer``.
- ``structured_outputs_simple_agent``: parse the response's JSON (last
  ```json fence, else the outermost braces) and validate against the row's
  ``schema_str`` with jsonschema.
- ``instruction_following_simple_agent``: the blend's instruction ids
  (keywords/detectable_format/first_word/...) come from allenai
  open-instruct's IFEvalG registry — NOT allenai/IFBench, whose registry is
  disjoint. Verified strict per the standard IFEval loop.

All graders are pure functions returning 1.0/0.0; the blend reward router
dispatches to them by ``metadata["agent"]``.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL)
_DEFAULT_MCQA_RE = r"<final_answer>\s*([A-Za-z])\s*</final_answer>"


def _normalize(value):
    """Deep-normalize for comparison: ints/floats of equal value compare equal."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        return [_normalize(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize(v) for k, v in value.items()}
    return value


def _extract_tool_call(text: str) -> dict | None:
    matches = _TOOL_CALL_RE.findall(text or "")
    if not matches:
        return None
    try:
        call = json.loads(matches[-1])
    except json.JSONDecodeError:
        return None
    return call if isinstance(call, dict) else None


def grade_tool_call(response: str, expected_action: dict | None) -> float:
    expected_action = expected_action or {}
    action_type = expected_action.get("type")
    call = _extract_tool_call(response or "")

    if action_type == "message":
        # Correct behavior = answer in text, not a tool call.
        return 1.0 if call is None and (response or "").strip() else 0.0

    if action_type != "function_call":
        logger.warning("ultra_agents: unknown expected_action type %r; reward 0.", action_type)
        return 0.0

    if call is None or call.get("name") != expected_action.get("name"):
        return 0.0

    expected_args = expected_action.get("arguments")
    if isinstance(expected_args, str):
        try:
            expected_args = json.loads(expected_args)
        except json.JSONDecodeError:
            logger.warning("ultra_agents: unparseable expected arguments; reward 0.")
            return 0.0
    actual_args = call.get("arguments")
    if isinstance(actual_args, str):
        try:
            actual_args = json.loads(actual_args)
        except json.JSONDecodeError:
            return 0.0

    return 1.0 if _normalize(actual_args) == _normalize(expected_args) else 0.0


def grade_mcqa(response: str, expected_answer: str, output_regex: str | None) -> float:
    pattern = output_regex or _DEFAULT_MCQA_RE
    try:
        matches = re.findall(pattern, response or "")
    except re.error:
        logger.warning("ultra_agents: bad mcqa output_regex %r; using default.", output_regex)
        matches = re.findall(_DEFAULT_MCQA_RE, response or "")
    if not matches:
        return 0.0
    answer = matches[-1] if isinstance(matches[-1], str) else matches[-1][0]
    return 1.0 if answer.strip().lower() == str(expected_answer or "").strip().lower() else 0.0


def _extract_json_candidate(text: str) -> str | None:
    fences = _JSON_FENCE_RE.findall(text or "")
    if fences:
        return fences[-1].strip()
    text = text or ""
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start, end = text.find(open_ch), text.rfind(close_ch)
        if start != -1 and end > start:
            return text[start : end + 1]
    return None


def grade_structured_output(response: str, schema_str: str, schema_type: str | None) -> float:
    schema_type = schema_type or "json"
    if schema_type not in ("json", "yaml"):
        logger.warning("ultra_agents: unsupported schema_type %r; reward 0.", schema_type)
        return 0.0
    try:
        schema = json.loads(schema_str)
    except json.JSONDecodeError:
        return 0.0
    if schema_type == "yaml":
        import yaml

        fences = re.findall(r"```(?:yaml|yml)?\s*\n(.*?)```", response or "", re.DOTALL)
        candidate = fences[-1].strip() if fences else (response or "").strip()
        try:
            payload = yaml.safe_load(candidate)
        except yaml.YAMLError:
            return 0.0
        if payload is None:
            return 0.0
    else:
        candidate = _extract_json_candidate(response)
        if candidate is None:
            return 0.0
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            return 0.0

    import jsonschema

    try:
        jsonschema.validate(payload, schema)
    except jsonschema.ValidationError:
        return 0.0
    except jsonschema.SchemaError:
        logger.warning("ultra_agents: invalid schema in row; reward 0.")
        return 0.0
    return 1.0


# ---------------------------------------------------------------------------
# Instruction following (open-instruct IFEvalG registry)
# ---------------------------------------------------------------------------

_OPEN_INSTRUCT_REPO = Path(
    os.environ.get(
        "ORBIT_OPEN_INSTRUCT_REPO",
        str(Path(__file__).resolve().parents[4] / "open-instruct"),
    )
)


@functools.cache
def _ifeval_registry():
    if not _OPEN_INSTRUCT_REPO.exists():
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", "https://github.com/allenai/open-instruct.git", str(_OPEN_INSTRUCT_REPO)],
                check=True,
                capture_output=True,
            )
        except Exception as exc:
            raise ImportError(
                f"open-instruct repo not found at {_OPEN_INSTRUCT_REPO} and auto-clone failed; "
                "set ORBIT_OPEN_INSTRUCT_REPO or clone allenai/open-instruct."
            ) from exc
    repo = str(_OPEN_INSTRUCT_REPO)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from open_instruct.IFEvalG import instructions_registry

    return instructions_registry.INSTRUCTION_DICT


def grade_instruction_following(
    response: str,
    instruction_id_list: list[str],
    kwargs_list: list[dict] | None,
    prompt_text: str = "",
) -> float:
    """Strict IFEval check: 1.0 iff every instruction is followed."""
    # Rollout decoding intentionally preserves special tokens for on-policy
    # training.  The terminal Qwen chat delimiter is protocol framing, not
    # visible assistant text, and otherwise breaks strict end/last-word rules.
    response = response or ""
    without_trailing_space = response.rstrip()
    if without_trailing_space.endswith("<|im_end|>"):
        response = without_trailing_space.removesuffix("<|im_end|>")
    if not instruction_id_list:
        return 0.0
    if not response.strip():
        return 0.0
    registry = _ifeval_registry()
    kwargs_list = kwargs_list or [{}] * len(instruction_id_list)
    for iid, kw in zip(instruction_id_list, kwargs_list, strict=False):
        cls = registry.get(iid)
        if cls is None:
            logger.warning("ultra_agents: unknown instruction id %r; reward 0.", iid)
            return 0.0
        try:
            instruction = cls(iid)
            instruction.build_description(**{k: v for k, v in (kw or {}).items() if v is not None})
            inst_args = instruction.get_instruction_args()
            if inst_args and "prompt" in inst_args:
                instruction.build_description(prompt=prompt_text)
            if not instruction.check_following(response):
                return 0.0
        except Exception:
            logger.exception("ultra_agents: instruction %r check crashed; reward 0.", iid)
            return 0.0
    return 1.0
