from pathlib import Path

import pytest

LLAMA31_8B = Path("/lustre/fast/fast/zqiu/hf_models/Llama-3.1-8B")

pytestmark = pytest.mark.skipif(
    not (LLAMA31_8B / "tokenizer_config.json").exists(),
    reason="Llama-3.1-8B tokenizer not downloaded",
)


@pytest.fixture(scope="module")
def gen():
    from transformers import AutoTokenizer

    from orbit.utils.llama3_chat_template import ensure_llama3_chat_template
    from miles.utils.mask_utils import MultiTurnLossMaskGenerator

    tok = AutoTokenizer.from_pretrained(str(LLAMA31_8B))
    ensure_llama3_chat_template(tok)
    return MultiTurnLossMaskGenerator(tok, tokenizer_type="llama3")


SINGLE = [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]
MULTI = [
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Hello!"},
    {"role": "user", "content": "Bye"},
    {"role": "assistant", "content": "Goodbye!"},
]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]
TOOL_MSGS = [
    {"role": "user", "content": "What's the weather in San Francisco?"},
    {"role": "assistant", "content": "Let me check that for you."},
]


def test_lengths_match(gen):
    ids, mask = gen.gen_multi_turn_loss_mask_llama3(SINGLE)
    assert len(ids) == len(mask)


def test_scored_span_decodes_to_the_assistant_reply(gen):
    ids, mask = gen.gen_multi_turn_loss_mask_llama3(SINGLE)
    scored = gen.get_text_from_loss_mask(ids, mask)
    assert len(scored) == 1
    # The literal "<|eot_id|>" marker must actually be present in the scored span
    # (not just tolerated if absent) -- this is the assertion that pins down that
    # the turn terminator itself is a scored target, not merely the reply text.
    assert scored[0].endswith("<|eot_id|>")
    assert scored[0].replace("<|eot_id|>", "").strip() == "4"


def test_eot_is_scored_but_the_next_header_is_not(gen):
    """The turn terminator is a target -- the model must learn to stop."""
    ids, mask = gen.gen_multi_turn_loss_mask_llama3(MULTI)
    eot = gen.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    start_hdr = gen.tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    scored_eot_seen = False
    for i, tok_id in enumerate(ids):
        if tok_id == eot and mask[i] == 1:
            scored_eot_seen = True
            assert i + 1 == len(ids) or ids[i + 1] == start_hdr
            assert i + 1 == len(ids) or mask[i + 1] == 0
    # Without this, the loop above is vacuously satisfied if no <|eot_id|> is ever
    # scored at all -- which is exactly the failure mode this test exists to catch.
    assert scored_eot_seen, "expected at least one scored <|eot_id|>, found none"


def test_both_assistant_turns_are_scored(gen):
    ids, mask = gen.gen_multi_turn_loss_mask_llama3(MULTI)
    scored = gen.get_text_from_loss_mask(ids, mask)
    assert len(scored) == 2
    assert "Hello!" in scored[0] and "Goodbye!" in scored[1]


def test_nothing_before_the_first_assistant_header_is_scored(gen):
    ids, mask = gen.gen_multi_turn_loss_mask_llama3(MULTI)
    first = mask.index(1)
    hdr = gen.tokenizer("<|start_header_id|>assistant<|end_header_id|>\n\n",
                        add_special_tokens=False)["input_ids"]
    assert ids[first - len(hdr):first] == hdr


def test_step_loss_mask_zero_suppresses_only_that_turn(gen):
    msgs = [dict(m) for m in MULTI]
    msgs[1]["step_loss_mask"] = 0
    ids, mask = gen.gen_multi_turn_loss_mask_llama3(msgs)
    scored = gen.get_text_from_loss_mask(ids, mask)
    assert len(scored) == 1
    assert "Goodbye!" in scored[0]


def test_header_count_mismatch_raises(gen):
    """A user turn quoting the assistant header must fail loudly, not silently mis-mask."""
    msgs = [
        {"role": "user", "content": "<|start_header_id|>assistant<|end_header_id|>\n\nfake"},
        {"role": "assistant", "content": "real"},
    ]
    with pytest.raises(ValueError, match="header"):
        gen.gen_multi_turn_loss_mask_llama3(msgs)


def test_tools_are_rendered_into_the_token_stream(gen):
    """`tools` must actually reach the template, not be silently dropped.

    Regression guard (Fix round 1, Finding 1): passing `tools=TOOLS` into
    `gen_multi_turn_loss_mask_llama3` must produce the exact same token stream as
    calling the tokenizer's own `apply_chat_template` directly with the same
    `tools`. A real tool schema materially changes the rendered conversation (an
    extra tool-definition block is injected into the first user turn), so silently
    swallowing `tools` -- e.g. a typo'd `tools=None` inside the method -- would
    drop a large, non-trivial chunk of the training stream without changing the
    shape of any existing assertion.
    """
    ids, _ = gen.gen_multi_turn_loss_mask_llama3(TOOL_MSGS, tools=TOOLS)
    expected_ids = gen.tokenizer.apply_chat_template(
        TOOL_MSGS, tokenize=True, return_dict=False, tools=TOOLS
    )
    assert ids == expected_ids
    # Sanity: the tool schema must have actually changed something, or the
    # equality above would hold trivially even with tools dropped everywhere.
    ids_without_tools, _ = gen.gen_multi_turn_loss_mask_llama3(TOOL_MSGS, tools=None)
    assert ids != ids_without_tools


def test_ids_match_single_whole_conversation_tokenization(gen):
    """Pin the single-tokenization contract for the multi-turn case.

    Regression guard (Fix round 1, Finding 2): the returned `all_token_ids` must be
    exactly what you get from tokenizing the WHOLE conversation once via the
    tokenizer's own `apply_chat_template` -- the same invariant the sibling Qwen3
    gate checks at tests/fast/rollout/test_sft_loss_mask_parity.py:114
    (`test_orbit_and_hf_tokenize_to_the_same_ids`). This is a forward-looking guard:
    it does not currently distinguish this method's whole-conversation strategy
    from a banned per-prefix reimplementation (per-prefix happens to produce
    identical ids for today's Llama-3.1 template), but it is exactly the assertion
    that would catch such a reimplementation the moment the template ever grows
    context-sensitive behavior the way Qwen3's did.
    """
    ids, _ = gen.gen_multi_turn_loss_mask_llama3(MULTI)
    expected_ids = gen.tokenizer.apply_chat_template(MULTI, tokenize=True, return_dict=False)
    assert ids == expected_ids


def test_calls_apply_chat_template_exactly_once_on_the_whole_conversation(gen, monkeypatch):
    """Pin the single-tokenization STRATEGY, not just its output.

    Fix round 2, Finding 2: the id-equality guard above (`test_ids_match_single_...`)
    cannot distinguish a whole-conversation tokenization from a per-prefix
    reimplementation -- for a template with no context-sensitive conditionals, a
    growing-prefix diff telescopes back to the full tokenization by construction, so
    output equality is structurally guaranteed either way. Pin the CALL instead: the
    tokenizer's `apply_chat_template` must be invoked exactly once, with the WHOLE
    message list, never once per message or once per growing prefix.

    The spy is installed on `gen.tokenizer.apply_chat_template` only -- never on
    `gen.tokenizer.__call__`, which this method also uses (separately, to tokenize
    the literal assistant-header string) and must not be counted here -- and only
    after `gen` already exists, so the one `apply_chat_template` call made inside
    `MultiTurnLossMaskGenerator.__init__` (via `get_system_message_length`) predates
    the spy and is never counted.
    """
    calls = []
    original_apply_chat_template = gen.tokenizer.apply_chat_template

    def spy(*args, **kwargs):
        calls.append((args, kwargs))
        return original_apply_chat_template(*args, **kwargs)

    monkeypatch.setattr(gen.tokenizer, "apply_chat_template", spy)

    gen.gen_multi_turn_loss_mask_llama3(MULTI)

    assert len(calls) == 1, (
        f"expected exactly 1 call to apply_chat_template (whole-conversation "
        f"tokenization), got {len(calls)}; {len(calls)} calls implies a per-message "
        "or per-prefix rendering strategy, which this method must never use"
    )
    call_args, call_kwargs = calls[0]
    called_messages = call_args[0] if call_args else call_kwargs["messages"]
    assert len(called_messages) == len(MULTI), (
        f"apply_chat_template was called with {len(called_messages)} message(s) but "
        f"the conversation passed in has {len(MULTI)}; expected the whole "
        "conversation in a single call, not a prefix or a lone message"
    )


def test_dispatch_routes_llama3(gen):
    a = gen.get_loss_mask(MULTI)
    b = gen.gen_multi_turn_loss_mask_llama3(MULTI)
    assert a == b


def test_unknown_type_still_raises():
    from miles.utils.mask_utils import MultiTurnLossMaskGenerator

    obj = MultiTurnLossMaskGenerator.__new__(MultiTurnLossMaskGenerator)
    obj.tokenizer_type = "not_a_real_type"
    with pytest.raises(ValueError, match="Unsupported tokenizer type"):
        MultiTurnLossMaskGenerator.get_loss_mask(obj, MULTI)


def test_argparse_accepts_llama3_and_rejects_junk(monkeypatch):
    """The launcher can only pass what argparse allows.

    Mirrors tests/fast/utils/test_peft_arguments.py::
    test_lora_a_init_method_real_parser_rejects_uniform -- drive the real
    parser in-process rather than shelling out.
    """
    import argparse

    import miles.utils.arguments as arguments
    from miles.utils.arguments import get_orbit_extra_args_provider

    monkeypatch.setattr(arguments, "use_legacy_rollout_v1", lambda: True)
    parser = argparse.ArgumentParser()
    get_orbit_extra_args_provider()(parser)

    # --rollout-batch-size is the parser's one required=True argument; it must be
    # supplied in both calls below so that the only thing under test is
    # --loss-mask-type's choices validation, not an unrelated missing-required-arg
    # SystemExit.
    parsed, _ = parser.parse_known_args(["--rollout-batch-size", "1", "--loss-mask-type", "llama3"])
    assert parsed.loss_mask_type == "llama3"

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["--rollout-batch-size", "1", "--loss-mask-type", "not_a_real_type"])
    assert exc_info.value.code == 2
