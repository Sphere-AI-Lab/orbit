from __future__ import annotations

from miles.utils import data as data_module
from miles.utils.data import Sample, filter_long_prompt


def _sample(prompt, multimodal_inputs=None) -> Sample:
    return Sample(prompt=prompt, multimodal_inputs=multimodal_inputs)


def test_filter_returns_samples_when_max_length_unset() -> None:
    samples = [_sample("short prompt")]
    assert filter_long_prompt(samples, tokenizer=None, processor=None, max_length=None) is samples


def test_filter_returns_samples_for_list_prompts() -> None:
    samples = [_sample([{"role": "user", "content": "hi"}])]
    assert filter_long_prompt(samples, tokenizer=None, processor=object(), max_length=8) is samples


def test_filter_reuses_stored_multimodal_inputs_for_templated_prompts(monkeypatch) -> None:
    """Chat-template flows store vision inputs at Sample construction; the
    templated string prompt must never be re-fed to qwen's message parser
    (job 27846/27847 crash: string indices must be integers)."""
    stored = {"images": ["sentinel-image"], "videos": None}
    samples = [
        _sample("<templated chat string with image pad>", multimodal_inputs=stored),
        _sample("x" * 4096, multimodal_inputs=stored),
    ]

    def _must_not_reextract(prompt, processor):  # pragma: no cover - failure path
        raise AssertionError("process_vision_info must not run on a templated string prompt")

    seen = []

    def _fake_call_processor(processor, text, multimodal_inputs):
        seen.append(multimodal_inputs)
        return {"input_ids": [list(range(10 if len(text) < 100 else 100))]}

    monkeypatch.setattr("miles.utils.processing_utils.process_vision_info", _must_not_reextract)
    monkeypatch.setattr(data_module, "call_processor", _fake_call_processor)

    kept = filter_long_prompt(samples, tokenizer=None, processor=object(), max_length=50)

    assert [s.prompt for s in kept] == ["<templated chat string with image pad>"]
    assert seen == [stored, stored]


def test_filter_reextracts_only_when_inputs_missing(monkeypatch) -> None:
    samples = [_sample("plain multimodal prompt", multimodal_inputs=None)]
    extracted = {"images": ["fresh"], "videos": None}
    calls = []

    def _fake_extract(prompt, processor):
        calls.append(prompt)
        return extracted

    def _fake_call_processor(processor, text, multimodal_inputs):
        assert multimodal_inputs is extracted
        return {"input_ids": [[1, 2, 3]]}

    monkeypatch.setattr("miles.utils.processing_utils.process_vision_info", _fake_extract)
    monkeypatch.setattr(data_module, "call_processor", _fake_call_processor)

    kept = filter_long_prompt(samples, tokenizer=None, processor=object(), max_length=8)
    assert len(kept) == 1
    assert calls == ["plain multimodal prompt"]
