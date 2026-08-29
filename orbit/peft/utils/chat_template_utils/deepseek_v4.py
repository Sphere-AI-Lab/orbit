from __future__ import annotations

import copy
import importlib.util
import logging
import os
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DSV4_CHAT_ENCODING_LEGACY = "legacy_jinja"
DSV4_CHAT_ENCODING_OFFICIAL_CHAT = "official_chat"
DSV4_CHAT_ENCODING_OFFICIAL_THINKING = "official_thinking"
DSV4_CHAT_ENCODING_CHOICES = {
    DSV4_CHAT_ENCODING_LEGACY,
    DSV4_CHAT_ENCODING_OFFICIAL_CHAT,
    DSV4_CHAT_ENCODING_OFFICIAL_THINKING,
}

_ASSISTANT_CHAT_SUFFIX = "<｜Assistant｜></think>"
_ASSISTANT_THINKING_SUFFIX = "<｜Assistant｜><think>"
_ENCODER_RELATIVE_PATH = Path("encoding") / "encoding_dsv4.py"


class DeepSeekV4ChatTemplateTokenizer:
    def __init__(
        self,
        tokenizer: Any,
        *,
        encode_messages: Callable[..., str],
        default_thinking_mode: str = "chat",
        encoder_path: Path | None = None,
    ):
        self._tokenizer = tokenizer
        self._encode_messages = encode_messages
        self._default_thinking_mode = default_thinking_mode
        self._encoder_path = encoder_path

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)

    def __call__(self, *args, **kwargs):
        return self._tokenizer(*args, **kwargs)

    def __len__(self) -> int:
        return len(self._tokenizer)

    def apply_chat_template(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: list[dict] | None = None,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        return_dict: bool = False,
        thinking_mode: str | None = None,
        drop_thinking: bool = True,
        reasoning_effort: str | None = None,
        **tokenizer_kwargs,
    ):
        mode = thinking_mode or self._default_thinking_mode
        if mode not in {"chat", "thinking"}:
            raise ValueError("DeepSeek-V4 thinking_mode must be 'chat' or 'thinking'.")

        prompt = self._encode_messages(
            _prepare_messages(messages, tools),
            thinking_mode=mode,
            drop_thinking=drop_thinking,
            reasoning_effort=reasoning_effort,
        )
        if not add_generation_prompt:
            prompt = _strip_generation_prompt(prompt, mode)

        if not tokenize:
            return prompt

        add_special_tokens = tokenizer_kwargs.pop("add_special_tokens", False)
        encoded = self._tokenizer(prompt, add_special_tokens=add_special_tokens, **tokenizer_kwargs)
        if return_dict:
            return encoded
        return encoded["input_ids"]


def maybe_wrap_deepseek_v4_tokenizer(
    tokenizer: Any,
    name_or_path: str | os.PathLike[str],
    *,
    chat_encoding: str | None = None,
    encoding_path: str | os.PathLike[str] | None = None,
) -> Any:
    if isinstance(tokenizer, DeepSeekV4ChatTemplateTokenizer):
        if chat_encoding is None and encoding_path is None:
            return tokenizer
        tokenizer = tokenizer._tokenizer

    explicit_chat_encoding = chat_encoding is not None
    chat_encoding = chat_encoding if explicit_chat_encoding else os.environ.get("DSV4_CHAT_ENCODING", DSV4_CHAT_ENCODING_LEGACY)
    if explicit_chat_encoding and chat_encoding not in DSV4_CHAT_ENCODING_CHOICES:
        raise ValueError(
            f"Unsupported DSV4_CHAT_ENCODING={chat_encoding!r}; expected one of {sorted(DSV4_CHAT_ENCODING_CHOICES)}."
        )

    checkpoint_path = Path(name_or_path)
    is_dsv4_checkpoint = bool(_derive_model_names(checkpoint_path.name))
    if not explicit_chat_encoding and not is_dsv4_checkpoint:
        return tokenizer

    has_chat_template = bool(getattr(tokenizer, "chat_template", None))
    if chat_encoding == DSV4_CHAT_ENCODING_LEGACY and has_chat_template:
        return tokenizer

    resolved_encoder_path = find_deepseek_v4_encoder(
        checkpoint_path,
        encoding_path=encoding_path,
    )
    if resolved_encoder_path is None:
        if chat_encoding not in DSV4_CHAT_ENCODING_CHOICES and is_dsv4_checkpoint:
            raise ValueError(
                f"Unsupported DSV4_CHAT_ENCODING={chat_encoding!r}; expected one of {sorted(DSV4_CHAT_ENCODING_CHOICES)}."
            )
        if explicit_chat_encoding or is_dsv4_checkpoint:
            raise FileNotFoundError(
                "DSV4_CHAT_ENCODING requests the official DeepSeek-V4 encoder, but encoding_dsv4.py was not found. "
                "Set DSV4_ENCODING_PATH or place encoding/encoding_dsv4.py beside the HF checkpoint."
            )
        return tokenizer
    if chat_encoding not in DSV4_CHAT_ENCODING_CHOICES:
        raise ValueError(
            f"Unsupported DSV4_CHAT_ENCODING={chat_encoding!r}; expected one of {sorted(DSV4_CHAT_ENCODING_CHOICES)}."
        )

    default_thinking_mode = _chat_encoding_to_thinking_mode(chat_encoding)
    module = _load_encoder_module(resolved_encoder_path)
    logger.info("Using DeepSeek-V4 official chat encoder from %s", resolved_encoder_path)
    return DeepSeekV4ChatTemplateTokenizer(
        tokenizer,
        encode_messages=module.encode_messages,
        default_thinking_mode=default_thinking_mode,
        encoder_path=resolved_encoder_path,
    )


def find_deepseek_v4_encoder(
    name_or_path: str | os.PathLike[str],
    *,
    encoding_path: str | os.PathLike[str] | None = None,
) -> Path | None:
    candidates: list[Path] = []

    explicit_path = encoding_path or os.environ.get("DSV4_ENCODING_PATH")
    if explicit_path:
        candidates.extend(_encoding_path_candidates(Path(explicit_path)))

    checkpoint_path = Path(name_or_path)
    candidates.append(checkpoint_path / _ENCODER_RELATIVE_PATH)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _chat_encoding_to_thinking_mode(chat_encoding: str) -> str:
    if chat_encoding == DSV4_CHAT_ENCODING_OFFICIAL_THINKING:
        return "thinking"
    return "chat"


def _encoding_path_candidates(path: Path) -> list[Path]:
    return [
        path,
        path / "encoding_dsv4.py",
        path / _ENCODER_RELATIVE_PATH,
    ]


def _derive_model_names(path_name: str) -> list[str]:
    match = re.search(r"(DeepSeek-V4-(?:Pro|Flash))", path_name)
    if match is None:
        return []
    return [match.group(1)]


def _load_encoder_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"orbit_dsv4_encoding_{abs(hash(path))}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load DeepSeek-V4 encoder from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "encode_messages"):
        raise ImportError(f"DeepSeek-V4 encoder at {path} does not define encode_messages")
    return module


def _prepare_messages(messages: Sequence[Mapping[str, Any]], tools: list[dict] | None) -> list[dict[str, Any]]:
    prepared = [copy.deepcopy(dict(message)) for message in messages]
    for message in prepared:
        message["content"] = _content_to_text(message.get("content"))

    if tools:
        for message in prepared:
            if message.get("role") in {"system", "developer"}:
                message["tools"] = tools
                break
        else:
            prepared.insert(0, {"role": "system", "content": "", "tools": tools})

    return prepared


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, Mapping) or block.get("type") != "text":
                raise ValueError("DeepSeek-V4 official encoder only supports text content blocks.")
            parts.append(str(block.get("text", "")))
        return "".join(parts)
    return str(content)


def _strip_generation_prompt(prompt: str, thinking_mode: str) -> str:
    suffix = _ASSISTANT_THINKING_SUFFIX if thinking_mode == "thinking" else _ASSISTANT_CHAT_SUFFIX
    if prompt.endswith(suffix):
        return prompt[: -len(suffix)]
    return prompt
