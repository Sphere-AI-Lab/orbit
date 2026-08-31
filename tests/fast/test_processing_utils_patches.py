"""``load_tokenizer`` routes its result through orbit's DSV4 wrapper.

The property that matters is not just "DSV4 checkpoints come back wrapped": it
is that orbit added ONLY that, and everything else ``load_tokenizer`` does --
building the tokenizer, applying ``--chat-template-path``, asserting when the
template file is missing -- still runs UPSTREAM's body. If that ever stops being
true because orbit copied the body instead, upstream's fixes stop reaching us.
"""

import pytest

pytest.importorskip("transformers")

import orbit  # noqa: F401  -- importing orbit installs the patches
from miles.utils import processing_utils as pu


class _DummyTokenizer:
    def __init__(self, name_or_path):
        self.name_or_path = name_or_path
        self.chat_template = None


class _DummyAutoTokenizer:
    """Stands in for transformers' loader so the test needs no checkpoint."""

    @staticmethod
    def from_pretrained(name_or_path, **kwargs):
        return _DummyTokenizer(name_or_path)


@pytest.fixture
def loader(monkeypatch):
    monkeypatch.setattr(pu, "AutoTokenizer", _DummyAutoTokenizer)
    monkeypatch.delenv("DSV4_CHAT_ENCODING", raising=False)
    monkeypatch.delenv("DSV4_ENCODING_PATH", raising=False)
    return pu.load_tokenizer


def _dsv4_checkpoint(tmp_path):
    """A directory that looks like a DeepSeek-V4 checkpoint, encoder included."""
    ckpt = tmp_path / "DeepSeek-V4-Pro"
    (ckpt / "encoding").mkdir(parents=True)
    (ckpt / "encoding" / "encoding_dsv4.py").write_text(
        "def encode_messages(messages, **kwargs):\n    return ''\n"
    )
    return ckpt


def test_the_patch_is_actually_installed():
    assert pu.load_tokenizer.__module__ == "orbit.utils.processing_utils_patches"
    assert hasattr(pu, "_orbit_unpatched_load_tokenizer"), (
        "the pristine upstream function must be kept so the patch can delegate"
    )


def test_a_dsv4_checkpoint_comes_back_wrapped(loader, tmp_path):
    from orbit.utils.chat_template_utils.deepseek_v4 import DeepSeekV4ChatTemplateTokenizer

    ckpt = str(_dsv4_checkpoint(tmp_path))
    assert isinstance(loader(ckpt), DeepSeekV4ChatTemplateTokenizer)

    # ...and prove the patch is what did it: upstream alone cannot.
    assert isinstance(pu._orbit_unpatched_load_tokenizer(ckpt), _DummyTokenizer)


def test_any_other_checkpoint_is_handed_straight_back(loader, tmp_path):
    """The wrapper is a no-op off the DSV4 path, so nothing else changes shape."""
    tokenizer = loader(str(tmp_path / "Qwen3-8B"))
    assert isinstance(tokenizer, _DummyTokenizer)


def test_the_chat_template_path_still_runs_upstreams_body(loader, tmp_path):
    """The delegation property: orbit added a wrap, not a reimplementation."""
    template = tmp_path / "template.jinja"
    template.write_text("{{ 'hello' }}")
    name_or_path = str(tmp_path / "Qwen3-8B")

    patched = loader(name_or_path, chat_template_path=str(template))
    upstream = pu._orbit_unpatched_load_tokenizer(name_or_path, str(template))
    assert patched.chat_template == "{{ 'hello' }}"
    assert patched.chat_template == upstream.chat_template


def test_a_missing_template_file_still_raises_upstreams_assert(loader, tmp_path):
    """Upstream's diagnostics must survive the wrapper, not be swallowed by it."""
    with pytest.raises(AssertionError, match="chat_template_path not found"):
        loader(str(tmp_path), chat_template_path=str(tmp_path / "absent.jinja"))
