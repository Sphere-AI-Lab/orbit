"""Unit tests for the agentic SWE episode loop (orbit/rollout/swe_agent/).

The loop is exercised with a scripted fake engine + fake container session:
what's under test is the pure episode logic — action parsing, token-stream
accounting (masks/logprobs aligned; tool tokens masked), turn/budget
termination, in-episode reward setting, fail-soft. The real-container path
is covered by the golden-episode oracle (tools/swe_agent_oracle.py).
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

import orbit.rollout.swe_agent.episode as gen_mod
from orbit.utils.types import Sample


class FakeTokenizer:
    """Character-code tokenizer with an append-only 'template'."""

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
        text = "".join(f"[{m['role']}]{m['content']}[/]" for m in messages)
        if add_generation_prompt:
            text += "[assistant]"
        return text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 251 for c in text]}

    def decode(self, ids):
        return f"<decoded {len(ids)} tokens>"


class FakeSession:
    def __init__(self, *a, **k):
        self.commands: list[str] = []
        self.stopped = False
        self.verify_result = False

    async def start(self, *a, **k):
        return True

    async def run(self, command, timeout_secs=None):
        self.commands.append(command)
        return 0, f"ran: {command}"

    async def verify(self, swe, timeout_secs=300.0):
        return self.verify_result

    async def stop(self):
        self.stopped = True


def _tool_call(name, **arguments):
    return f'<tool_call>{json.dumps({"name": name, "arguments": arguments})}</tool_call>'


def _scripted_engine(turns):
    """Fake /generate: pops scripted turn texts; token ids = char codes."""
    queue = list(turns)
    calls = []

    async def fake_post(url, payload):
        # snapshot at call time: the loop mutates the input_ids list in place
        # (real HTTP serializes at send time, so only the fake sees aliasing)
        calls.append({"input_ids_len": len(payload.get("input_ids") or [])})
        text = queue.pop(0)
        ids = [ord(c) % 251 for c in text]
        return {
            "text": text,
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [(-0.5, i) for i in ids],
            },
        }

    fake_post.calls = calls
    return fake_post


def _args(**overrides):
    values = {
        "sglang_router_ip": "127.0.0.1",
        "sglang_router_port": 1,
        "swe_rm_sif_cache": "/cache",
        "swe_rm_timeout_secs": 300,
        "swe_agent_max_turns": 12,
        "swe_agent_cmd_timeout_secs": 30,
        "rollout_max_response_len": 4096,
        "hf_checkpoint": "unused",
        "chat_template_path": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _sample():
    return Sample(
        prompt=[{"role": "user", "content": "fix the bug in foo()"}],
        metadata={"swe": {"image_name": "docker.io/x/y:z", "fail_to_pass": ["t::a"], "pass_to_pass": []}},
    )


def _run_episode(monkeypatch, turns, verify_result=True, **arg_overrides):
    session = FakeSession()
    session.verify_result = verify_result
    engine = _scripted_engine(turns)

    monkeypatch.setattr(gen_mod, "ContainerSession", lambda *a, **k: session)
    monkeypatch.setattr(gen_mod, "sif_for_instance", lambda cache, image: "/cache/fake.sif")
    monkeypatch.setattr(gen_mod, "post", engine)

    class FakeState:
        def __init__(self, args):
            self.tokenizer = FakeTokenizer()

    monkeypatch.setattr(gen_mod, "GenerateState", FakeState)

    sample = _sample()
    result = asyncio.run(gen_mod.generate(_args(**arg_overrides), sample, {"max_new_tokens": 512}))
    return result, session, engine


# ---------------------------------------------------------------------------
# Action parsing
# ---------------------------------------------------------------------------


def test_parse_action_variants():
    assert gen_mod._parse_action(_tool_call("run_shell", command="ls"))["name"] == "run_shell"
    assert gen_mod._parse_action("thinking...\n" + _tool_call("submit"))["name"] == "submit"
    assert gen_mod._parse_action("no call") is None
    assert gen_mod._parse_action("<tool_call>{bad json}</tool_call>") is None


def test_parse_action_bare_json_fallback():
    bare = '{"name": "run_shell", "arguments": {"command": "pip show click"}}'
    assert gen_mod._parse_action(bare)["name"] == "run_shell"
    assert gen_mod._parse_action("prefix text\n" + bare)["arguments"]["command"] == "pip show click"


# ---------------------------------------------------------------------------
# Episode flow
# ---------------------------------------------------------------------------


def test_episode_runs_commands_then_submits_and_grades(monkeypatch):
    result, session, engine = _run_episode(
        monkeypatch,
        turns=[
            _tool_call("run_shell", command="grep -rn bug foo.py"),
            _tool_call("run_shell", command="sed -i s/bug/fix/ foo.py"),
            _tool_call("submit"),
        ],
        verify_result=True,
    )
    assert session.commands == ["grep -rn bug foo.py", "sed -i s/bug/fix/ foo.py"]
    assert result.reward == 1.0
    assert result.status == Sample.Status.COMPLETED
    assert session.stopped
    assert len(engine.calls) == 3


def test_failed_verification_scores_zero(monkeypatch):
    result, _, _ = _run_episode(monkeypatch, turns=[_tool_call("submit")], verify_result=False)
    assert result.reward == 0.0
    assert result.status == Sample.Status.COMPLETED


def test_token_stream_masks_tool_turns(monkeypatch):
    result, _, engine = _run_episode(
        monkeypatch,
        turns=[_tool_call("run_shell", command="ls"), _tool_call("submit")],
    )
    assert result.response_length == len(result.loss_mask) == len(result.rollout_log_probs)
    assert result.response_length > 0
    # model tokens are mask 1 with real logprobs; tool tokens mask 0 / 0.0
    assert set(result.loss_mask) == {0, 1}
    for m, lp in zip(result.loss_mask, result.rollout_log_probs):
        if m == 0:
            assert lp == 0.0
        else:
            assert lp == -0.5
    # the second engine call saw the stream extended by turn 1 + tool tokens
    assert engine.calls[1]["input_ids_len"] > engine.calls[0]["input_ids_len"]


def test_max_turns_terminates_and_still_grades(monkeypatch):
    result, session, engine = _run_episode(
        monkeypatch,
        turns=[_tool_call("run_shell", command=f"cmd{i}") for i in range(5)],
        verify_result=False,
        swe_agent_max_turns=3,
    )
    assert len(engine.calls) == 3
    assert len(session.commands) == 3
    assert result.status == Sample.Status.COMPLETED
    assert result.reward == 0.0


def test_response_budget_truncates(monkeypatch):
    long_cmd = _tool_call("run_shell", command="x" * 400)
    result, _, _ = _run_episode(
        monkeypatch,
        turns=[long_cmd, long_cmd, _tool_call("submit")],
        rollout_max_response_len=600,
    )
    assert result.status == Sample.Status.TRUNCATED
    assert result.response_length <= 600 + 512  # budget + one turn's overshoot bound


def test_no_tool_call_ends_episode(monkeypatch):
    result, session, engine = _run_episode(
        monkeypatch, turns=["I think the fix is to change foo."], verify_result=False
    )
    assert len(engine.calls) == 1
    assert session.commands == []
    assert result.status == Sample.Status.COMPLETED


def test_session_start_failure_fails_sample(monkeypatch):
    class DeadSession(FakeSession):
        async def start(self, *a, **k):
            return False

    monkeypatch.setattr(gen_mod, "ContainerSession", lambda *a, **k: DeadSession())
    monkeypatch.setattr(gen_mod, "sif_for_instance", lambda cache, image: "/cache/fake.sif")

    class FakeState:
        def __init__(self, args):
            self.tokenizer = FakeTokenizer()

    monkeypatch.setattr(gen_mod, "GenerateState", FakeState)
    sample = _sample()
    result = asyncio.run(gen_mod.generate(_args(), sample, {}))
    assert result.status == Sample.Status.FAILED
    assert result.reward == 0.0


def test_crash_fails_soft(monkeypatch):
    async def boom(url, payload):
        raise RuntimeError("engine down")

    session = FakeSession()
    monkeypatch.setattr(gen_mod, "ContainerSession", lambda *a, **k: session)
    monkeypatch.setattr(gen_mod, "sif_for_instance", lambda cache, image: "/cache/fake.sif")
    monkeypatch.setattr(gen_mod, "post", boom)

    class FakeState:
        def __init__(self, args):
            self.tokenizer = FakeTokenizer()

    monkeypatch.setattr(gen_mod, "GenerateState", FakeState)
    result = asyncio.run(gen_mod.generate(_args(), _sample(), {}))
    assert result.status == Sample.Status.FAILED
    assert result.reward == 0.0
    assert session.stopped  # session cleaned up even on crash
