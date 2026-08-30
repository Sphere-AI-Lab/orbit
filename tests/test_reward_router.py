"""Unit tests for the blend reward router (miles/orbit/rewards/reward_router.py).

Routes each rollout group to a grader by ``metadata["agent"]`` (the NeMo Gym
``agent_ref.name`` carried through conversion). Groups are per-prompt, so the
agent is uniform within a group; the router dispatches whole groups.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

import miles.orbit.rewards.reward_router as router
from miles.utils.types import Sample


def _args(**overrides):
    values = {
        "group_rm": True,
        "judge_base_url": "http://judge:30801",
        "reward_router_unmapped": "zero",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _group(agent: str, n: int = 2, **metadata) -> list[Sample]:
    md = {"agent": agent, **metadata}
    return [
        Sample(prompt=[{"role": "user", "content": "q"}], response=f"r{i}", label="ref", metadata=dict(md))
        for i in range(n)
    ]


def _run(coro):
    return asyncio.run(coro)


def test_known_agents_route_to_their_graders(monkeypatch):
    calls = {}

    async def fake_judge(args, sample, **kwargs):
        calls.setdefault("judge", 0)
        calls["judge"] += 1
        return 1.0

    async def fake_genrm(args, samples, **kwargs):
        calls["genrm"] = len(samples)
        return [0.5] * len(samples)

    async def fake_code(args, sample, **kwargs):
        calls.setdefault("code", 0)
        calls["code"] += 1
        return 0.0

    monkeypatch.setattr(router, "_judge_reward", fake_judge)
    monkeypatch.setattr(router, "_genrm_reward", fake_genrm)
    monkeypatch.setattr(router, "_code_reward", fake_code)

    assert _run(router.reward_func(_args(), _group("equivalence_llm_judge_simple_agent"))) == [1.0, 1.0]
    assert calls["judge"] == 2
    assert _run(router.reward_func(_args(), _group("genrm_simple_agent", n=3))) == [0.5, 0.5, 0.5]
    assert calls["genrm"] == 3
    assert _run(router.reward_func(_args(), _group("code_gen_simple_agent"))) == [0.0, 0.0]
    assert calls["code"] == 2


@pytest.mark.parametrize(
    ("agent", "target"),
    [
        ("math_with_judge_simple_agent", "judge"),
        ("equivalence_llm_judge_simple_agent", "judge"),
        ("genrm_simple_agent", "genrm"),
        ("genrm_simple_agent_reasoning_off", "genrm"),
        ("code_gen_simple_agent", "code"),
    ],
)
def test_default_agent_map(agent, target):
    assert router._route_for_agent(agent) == target


def test_unmapped_agent_zero_rewards_loudly(monkeypatch, caplog):
    group = _group("definitely_not_a_real_agent")
    with caplog.at_level("WARNING"):
        rewards = _run(router.reward_func(_args(), group))
    assert rewards == [0.0, 0.0]
    assert any("unmapped" in r.message.lower() for r in caplog.records)


def test_unmapped_agent_can_error_instead():
    group = _group("definitely_not_a_real_agent")
    with pytest.raises(ValueError, match="unmapped"):
        _run(router.reward_func(_args(reward_router_unmapped="error"), group))


def test_missing_agent_metadata_counts_as_unmapped():
    samples = [Sample(prompt="q", response="r", metadata={})]
    assert _run(router.reward_func(_args(), samples)) == [0.0]


def test_mixed_agents_within_group_rejected():
    group = _group("code_gen_simple_agent") + _group("genrm_simple_agent")
    with pytest.raises(ValueError, match="uniform"):
        _run(router.reward_func(_args(), group))


def test_empty_group_returns_empty():
    assert _run(router.reward_func(_args(), [])) == []


def test_judge_failure_fails_soft_to_zero(monkeypatch):
    async def broken_judge(args, sample, **kwargs):
        raise RuntimeError("judge down")

    monkeypatch.setattr(router, "_judge_reward", broken_judge)
    rewards = _run(router.reward_func(_args(), _group("math_with_judge_simple_agent")))
    assert rewards == [0.0, 0.0]
