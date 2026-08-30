"""Blend reward router: dispatch each rollout group to its grader by agent name.

The Nemotron-RL-Ultra blends are heterogeneous — every row names its grader in
``agent_ref.name`` (NeMo Gym's per-row agent binding; see NeMo-RL
``experience/rollouts.py``). Orbit binds one reward mechanism per run, so this
router is the missing translation: conversion carries ``agent_ref.name`` into
``metadata["agent"]``, and the router maps agent names onto orbit's verified
graders::

    --custom-rm-path miles.orbit.rewards.reward_router.reward_func
    --group-rm
    [--judge-base-url ...]        # required for judge/genrm-routed rows
    [--reward-router-unmapped zero|error]

Routing table (v1 — the graders orbit has GPU-verified):

- ``math_with_judge_simple_agent``, ``equivalence_llm_judge_simple_agent``
  -> per-sample LLM-judge equivalence (``llm_judge``, needs ``label``)
- ``genrm_simple_agent``, ``genrm_simple_agent_reasoning_off``
  -> group-wise pairwise GenRM (``genrm_judge``, rubric in metadata)
- ``code_gen_simple_agent``
  -> sandboxed code execution (``sandbox.code_rm``, unit_tests in metadata)

Groups are per-prompt (n samples of one row), so the agent is uniform within a
group — the router validates that and dispatches whole groups. Rows whose
agent has no orbit grader yet (swe, tool-use, ...) are zero-rewarded with a
warning by default (train on the covered subset, honestly) or rejected with
``--reward-router-unmapped error``. Per-grader failures fail soft to 0.0.
"""

from __future__ import annotations

import logging
from argparse import Namespace

from miles.orbit.rewards.genrm_judge import reward_func as _genrm_reward
from miles.orbit.rewards.llm_judge import reward_func as _judge_reward
from miles.orbit.rewards.sandbox.code_rm import reward_func as _code_reward
from miles.orbit.rewards.sandbox.swe_rm import reward_func as _swe_reward
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_AGENT_ROUTES: dict[str, str] = {
    "math_with_judge_simple_agent": "judge",
    "equivalence_llm_judge_simple_agent": "judge",
    "genrm_simple_agent": "genrm",
    "genrm_simple_agent_reasoning_off": "genrm",
    "code_gen_simple_agent": "code",
    "swe_agents_train": "swe",
    # single-turn rule-based Ultra agents (rm_hub/ultra_agents.py)
    "single_step_tool_use_with_argument_comparison_agent": "tool_call",
    "swe_pivot_single_step_tool_use_with_argument_comparison_agent": "tool_call",
    "toolcall_schema_single_step_tool_use_with_argument_comparison_agent": "tool_call",
    "mcqa_simple_agent": "mcqa",
    "structured_outputs_simple_agent": "structured",
    "structured_outputs_v3_simple_agent": "structured",
    "instruction_following_simple_agent": "if",
    # long-tail agents (rm_hub/ultra_longtail.py)
    "ns_tools_simple_agent": "judge",  # verifier_type math_with_judge in-row
    "abstention_simple_agent": "judge",
    "rdkit_chemistry_agent": "boxed",
    "reasoning_gym_simple_agent": "boxed",
    "nvarc_transductive_simple_agent": "nvarc_t",
    "nvarc_inductive_simple_agent": "nvarc_i",
    "citation_format_simple_agent": "verifier_spec",
    "freeform_formatting_simple_agent": "verifier_spec",
    "calendar_simple_agent": "calendar",
    "multichallenge_simple_agent": "rubric_judge",
    "jailbreak_refusal_with_explanation": "policy_judge",
    "jailbreak_hard_refusal_with_helplines": "policy_judge",
    "jailbreak_engagement_with_disclaimer": "policy_judge",
    "jailbreak_hard_refusal_no_redirection": "policy_judge",
    "math_formal_lean_refinement_agent": "lean",  # needs --lean-server-url
}


def _route_for_agent(agent: str | None) -> str | None:
    return _AGENT_ROUTES.get(agent or "")


async def reward_func(args: Namespace, samples: list[Sample], **kwargs) -> list[float]:
    """``--custom-rm-path`` hook (batch mode, requires ``--group-rm``)."""
    if not samples:
        return []

    agents = {
        (s.metadata or {}).get("agent") if isinstance(s.metadata, dict) else None for s in samples
    }
    if len(agents) != 1:
        raise ValueError(
            f"reward_router expects a uniform agent per group (one prompt's samples), got {sorted(str(a) for a in agents)}. "
            "Groups mixing rows indicate the conversion or grouping is broken."
        )
    (agent,) = agents
    route = _route_for_agent(agent)

    if route is None:
        if getattr(args, "reward_router_unmapped", "zero") == "error":
            raise ValueError(f"reward_router: unmapped agent {agent!r} and --reward-router-unmapped=error.")
        logger.warning(
            "reward_router: unmapped agent %r (%d samples) — zero reward. Add a grader or filter these rows.",
            agent,
            len(samples),
        )
        return [0.0] * len(samples)

    try:
        if route == "genrm":
            return await _genrm_reward(args, samples, **kwargs)
        rewards = []
        for sample in samples:
            metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
            if route == "judge":
                rewards.append(float(await _judge_reward(args, sample, **kwargs)))
            elif route == "code":
                rewards.append(float(await _code_reward(args, sample, **kwargs)))
            elif route == "swe":
                rewards.append(float(await _swe_reward(args, sample, **kwargs)))
            elif route == "tool_call":
                from miles.orbit.rewards.ultra_agents import grade_tool_call

                rewards.append(grade_tool_call(sample.response, metadata.get("expected_action")))
            elif route == "mcqa":
                from miles.orbit.rewards.ultra_agents import grade_mcqa

                rewards.append(
                    grade_mcqa(sample.response, sample.label or metadata.get("expected_answer"), metadata.get("output_regex"))
                )
            elif route == "structured":
                from miles.orbit.rewards.ultra_agents import grade_structured_output

                rewards.append(
                    grade_structured_output(sample.response, metadata.get("schema_str") or "", metadata.get("schema_type"))
                )
            elif route == "boxed":
                from miles.orbit.rewards.ultra_longtail import grade_boxed_answer

                rewards.append(grade_boxed_answer(sample.response, sample.label or metadata.get("expected_answer")))
            elif route == "nvarc_t":
                from miles.orbit.rewards.ultra_longtail import grade_nvarc_transductive

                rewards.append(grade_nvarc_transductive(sample.response, metadata.get("expected_output")))
            elif route == "nvarc_i":
                from miles.orbit.rewards.ultra_longtail import grade_nvarc_inductive

                rewards.append(
                    float(await grade_nvarc_inductive(sample.response, metadata.get("test_input"), metadata.get("expected_output")))
                )
            elif route == "verifier_spec":
                from miles.orbit.rewards.ultra_longtail import grade_verifier_spec

                rewards.append(grade_verifier_spec(sample.response, metadata.get("verifier")))
            elif route == "calendar":
                from miles.orbit.rewards.ultra_longtail import grade_calendar

                rewards.append(grade_calendar(sample.response, metadata.get("exp_cal_state")))
            elif route == "lean":
                from miles.orbit.rewards.lean_rm import grade_lean_proof

                rewards.append(
                    float(
                        await grade_lean_proof(
                            args, sample.response, metadata.get("header") or "", metadata.get("formal_statement") or ""
                        )
                    )
                )
            elif route == "rubric_judge":
                from miles.orbit.rewards.ultra_longtail import grade_rubric_judge

                rewards.append(
                    float(await grade_rubric_judge(args, metadata.get("context") or "", sample.response, metadata.get("rubric") or []))
                )
            elif route == "policy_judge":
                from miles.orbit.rewards.ultra_longtail import grade_policy_judge

                rewards.append(
                    float(
                        await grade_policy_judge(
                            args, metadata.get("adversarial_prompt") or "", sample.response, metadata.get("response_policy") or ""
                        )
                    )
                )
            elif route == "if":
                # lazy: first use may clone allenai/open-instruct (IFEvalG registry)
                from miles.orbit.rewards.ultra_agents import grade_instruction_following

                rewards.append(
                    grade_instruction_following(
                        sample.response,
                        metadata.get("instruction_id_list") or [],
                        metadata.get("kwargs"),
                        prompt_text=metadata.get("prompt_text") or "",
                    )
                )
            else:  # pragma: no cover — routes and handlers are defined together
                raise AssertionError(f"unhandled route {route!r}")
        return rewards
    except Exception:
        logger.exception(
            "reward_router: grader %r failed for agent %r; zero-rewarding the group (fail-soft).", route, agent
        )
        return [0.0] * len(samples)
