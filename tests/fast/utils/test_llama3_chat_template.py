"""The Llama-3.1 base tokenizer ships no chat template; we pin one."""

import json
from pathlib import Path

import pytest

import miles.orbit.utils.chat_template_utils as _orbit_templates

LLAMA31_8B = Path("/lustre/fast/fast/zqiu/hf_models/Llama-3.1-8B")
LLAMA31_8B_INSTRUCT = Path("/lustre/fast/fast/zqiu/hf_models/Llama-3.1-8B-Instruct")
BUNDLED_LLAMA3_JINJA = Path(_orbit_templates.__file__).parent / "templates" / "llama3.1_pinned.jinja"

pytestmark = pytest.mark.skipif(
    not (LLAMA31_8B / "tokenizer_config.json").exists(),
    reason="Llama-3.1-8B tokenizer not downloaded",
)


@pytest.fixture()
def base_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(LLAMA31_8B))


def test_base_tokenizer_really_has_no_template(base_tokenizer):
    """If this ever fails, upstream added a template and Task 1's premise changed."""
    assert base_tokenizer.chat_template is None


def test_ensure_sets_the_template(base_tokenizer):
    from miles.orbit.utils.llama3_chat_template import (
        LLAMA3_CHAT_TEMPLATE,
        ensure_llama3_chat_template,
    )

    ensure_llama3_chat_template(base_tokenizer)
    assert base_tokenizer.chat_template == LLAMA3_CHAT_TEMPLATE


def test_ensure_is_idempotent_and_never_overwrites(base_tokenizer):
    from miles.orbit.utils.llama3_chat_template import ensure_llama3_chat_template

    base_tokenizer.chat_template = "SENTINEL"
    ensure_llama3_chat_template(base_tokenizer)
    assert base_tokenizer.chat_template == "SENTINEL"


def test_template_date_is_a_literal_not_a_clock():
    """A strftime_now date would make every run tokenize differently by day."""
    from miles.orbit.utils.llama3_chat_template import LLAMA3_CHAT_TEMPLATE

    assert "strftime_now" not in LLAMA3_CHAT_TEMPLATE
    assert '"26 Jul 2024"' in LLAMA3_CHAT_TEMPLATE


def test_rendered_conversation_has_expected_markers(base_tokenizer):
    from miles.orbit.utils.llama3_chat_template import ensure_llama3_chat_template

    ensure_llama3_chat_template(base_tokenizer)
    text = base_tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        tokenize=False,
    )
    assert "<|start_header_id|>assistant<|end_header_id|>\n\n" in text
    assert text.rstrip().endswith("<|eot_id|>")


def test_exactly_one_bos_and_it_is_first(base_tokenizer):
    """apply_chat_template must not double-add bos on top of the template's own."""
    from miles.orbit.utils.llama3_chat_template import ensure_llama3_chat_template

    ensure_llama3_chat_template(base_tokenizer)
    ids = base_tokenizer.apply_chat_template(
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        tokenize=True,
        return_dict=False,
    )
    bos = base_tokenizer.convert_tokens_to_ids("<|begin_of_text|>")
    assert ids[0] == bos
    assert ids.count(bos) == 1


@pytest.mark.skipif(
    not (LLAMA31_8B_INSTRUCT / "tokenizer_config.json").exists(),
    reason="Llama-3.1-8B-Instruct checkpoint not available",
)
def test_pinned_template_matches_source():
    """Validate that LLAMA3_CHAT_TEMPLATE matches the source tokenizer_config.json.

    This test catches accidental drift from careless re-copy or merge conflicts.
    A single changed whitespace silently corrupts all downstream loss masks.
    """
    from miles.orbit.utils.llama3_chat_template import LLAMA3_CHAT_TEMPLATE

    # Load the reference template from the Instruct checkpoint
    with open(LLAMA31_8B_INSTRUCT / "tokenizer_config.json") as f:
        source_config = json.load(f)
    source_template = source_config["chat_template"]

    # Assert byte-identity
    assert (
        LLAMA3_CHAT_TEMPLATE == source_template
    ), (
        "LLAMA3_CHAT_TEMPLATE does not match source. "
        "Re-copy from /lustre/fast/fast/zqiu/hf_models/Llama-3.1-8B-Instruct/tokenizer_config.json "
        "by running: miles/orbit/utils/llama3_chat_template.py (extract and embed with raw string r\"\"\"...\"\"\")."
    )


def test_bundled_jinja_matches_the_pinned_python_constant():
    """The .jinja file production actually loads must equal LLAMA3_CHAT_TEMPLATE byte-for-byte.

    `ensure_llama3_chat_template` (and the Python constant it sets) has no production
    caller: `load_tokenizer` (miles/utils/processing_utils.py) only ever sets
    `tokenizer.chat_template` from a `--chat-template-path` FILE, never from this
    module. So the drift test above -- which only guards
    `LLAMA3_CHAT_TEMPLATE` against the Instruct checkpoint -- guards nothing an actual
    training run reads. This test is the missing link: it ties the bundled
    `miles/orbit/utils/chat_template_utils/templates/llama3.1_pinned.jinja` (the file a
    Llama-3 SFT run must point `--chat-template-path` at) to the same pinned string,
    so the two cannot silently diverge.
    """
    from miles.orbit.utils.llama3_chat_template import LLAMA3_CHAT_TEMPLATE

    assert BUNDLED_LLAMA3_JINJA.is_file(), f"bundled template missing: {BUNDLED_LLAMA3_JINJA}"
    on_disk = BUNDLED_LLAMA3_JINJA.read_text()
    assert on_disk == LLAMA3_CHAT_TEMPLATE, (
        f"{BUNDLED_LLAMA3_JINJA} has drifted from LLAMA3_CHAT_TEMPLATE. This .jinja file, "
        "not the Python constant, is what a real Llama-3 SFT run loads via "
        "--chat-template-path; a single byte of drift here changes what tokens the run "
        "scores without any other test in this repo noticing."
    )
