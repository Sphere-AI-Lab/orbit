"""Orbit's addition to ``miles.utils.processing_utils.load_tokenizer``.

DeepSeek-V4 checkpoints do not ship a jinja ``chat_template``; their chat
encoding is a python module inside the checkpoint. The tokenizer
``AutoTokenizer.from_pretrained`` hands back therefore cannot render a
conversation at all, and every caller that does ``tokenizer.apply_chat_template``
breaks. Orbit wraps it (orbit/utils/chat_template_utils/deepseek_v4.py); for any
other checkpoint the wrapper returns the tokenizer unchanged.

This used to be two edits in the vendored file -- a module-level
``from orbit...import`` (which also cost that module the leaf property patched
modules must have) and a changed ``return``. It is a DELEGATING patch now:
upstream builds the tokenizer and applies ``--chat-template-path`` exactly as
before, and orbit only post-processes what came back. The vendored file is
byte-pristine again.

Not moved: upstream's ``process_vision_info`` also carried an orbit edit, but it
was a log level (``logger.info`` -> ``logger.debug``) in the middle of upstream's
body, which a wrapper cannot express and which changes no behaviour. It was
dropped rather than copied; see the batch report.

Nothing here imports transformers or miles at module scope: ``import orbit``
executes this module and must stay cheap (see orbit/patch/runtime.py).
"""

from __future__ import annotations

from orbit.patch import original, patch_function

_PROCESSING_UTILS = "miles.utils.processing_utils"

_REASON = (
    "DSV4 checkpoints carry their chat encoding as a python module instead of a "
    "jinja chat_template, so the tokenizer upstream returns cannot render a "
    "conversation; orbit wraps it. Upstream has no spelling for that, and the "
    "wrapper is a no-op for every other checkpoint."
)


@patch_function(
    "miles.utils.processing_utils",
    "load_tokenizer",
    upstream_sha="15ee8e57b00d57f26ccb0b3eec82570df4cfa9a8eb2551344e3ef821624987a1",
    reason=_REASON,
)
def load_tokenizer(name_or_path, chat_template_path=None, **kwargs):
    """Upstream's tokenizer, wrapped when the checkpoint is DSV4.

    ``chat_template_path`` is forwarded positionally because upstream declares it
    positionally: a caller that passed it either way keeps working, and orbit
    never has to know which spelling was used.
    """
    from orbit.utils.chat_template_utils.deepseek_v4 import maybe_wrap_deepseek_v4_tokenizer

    tokenizer = original(_PROCESSING_UTILS, "load_tokenizer")(
        name_or_path, chat_template_path, **kwargs
    )
    return maybe_wrap_deepseek_v4_tokenizer(tokenizer, name_or_path)
