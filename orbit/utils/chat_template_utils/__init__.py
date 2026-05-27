"""Chat template utilities for agentic-workflow token consistency."""

from orbit.utils.chat_template_utils.autofix import TEMPLATE_DIR, try_get_fixed_chat_template
from orbit.utils.chat_template_utils.template import (
    apply_chat_template,
    apply_chat_template_from_str,
    assert_messages_append_only_with_allowed_role,
    extract_tool_dicts,
    load_hf_chat_template,
    message_matches,
)
from orbit.utils.chat_template_utils.tito_tokenizer import TITOTokenizer, TITOTokenizerType, get_tito_tokenizer
from orbit.utils.chat_template_utils.token_seq_comparator import Mismatch, MismatchType, TokenSeqComparator

__all__ = [
    "TITOTokenizer",
    "TITOTokenizerType",
    "get_tito_tokenizer",
    "TEMPLATE_DIR",
    "try_get_fixed_chat_template",
    "load_hf_chat_template",
    "apply_chat_template",
    "apply_chat_template_from_str",
    "assert_messages_append_only_with_allowed_role",
    "message_matches",
    "extract_tool_dicts",
    "Mismatch",
    "TokenSeqComparator",
    "MismatchType",
]
