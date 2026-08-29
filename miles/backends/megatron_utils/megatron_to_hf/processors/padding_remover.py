import torch

from miles.backends.megatron_utils.misc_utils import strip_param_name_prefix


# ORBIT-SEAM: accept HF param names too — newer megatron-bridge drops megatron names (upstream candidate)
# Megatron-native names (for legacy 3-tuple code paths).
_MEGATRON_VOCAB_LAYER_NAMES = {
    "embedding.word_embeddings.weight",
    "output_layer.weight",
}
# HF-native names (used by newer megatron-bridge HFWeightTuple which doesn't
# expose the megatron name anymore).
_HF_VOCAB_LAYER_NAMES = {
    "model.embed_tokens.weight",
    "lm_head.weight",
}


def remove_padding(name: str, param: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """
    Remove vocab padding: param[:vocab_size] for embedding/output layers, else unchanged.

    Accepts either the Megatron parameter name (for legacy callers) or the
    HF parameter name (for the newer megatron-bridge HFWeightTuple API where
    the megatron name is no longer yielded).
    """
    # ORBIT-SEAM: HF-name compat, see the name sets above
    stripped = strip_param_name_prefix(name)
    if stripped in _MEGATRON_VOCAB_LAYER_NAMES or stripped in _HF_VOCAB_LAYER_NAMES:
        return param[:vocab_size]
    return param
