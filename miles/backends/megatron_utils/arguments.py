import logging
# ORBIT-SEAM: math import backs the inlined _vocab_size_with_padding below (upstream Megatron
# dropped the module it used to live in).
import math
import os

from megatron.training.arguments import parse_args, validate_args


# Inlined from megatron.training.tokenizer.tokenizer. That module was removed
# upstream in commit 325709393 ("fully remove legacy tokenizer system"), but
# orbit only needs this one standalone helper. Keeping it local lets us run
# against the same Megatron-LM commit verl uses.
def _vocab_size_with_padding(orig_vocab_size, args, logging_enabled=True):
    """Pad vocab size so it is divisible by model parallel size and still
    having GPU friendly size."""
    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * args.tensor_model_parallel_size
    after = int(math.ceil(after / multiple) * multiple)
    if args.rank == 0 and logging_enabled:
        print(
            " > padded vocab (size: {}) with {} dummy tokens "
            "(new size: {})".format(orig_vocab_size, after - orig_vocab_size, after),
            flush=True,
        )
    return after

__all__ = ["validate_args", "parse_args", "set_default_megatron_args"]

logger = logging.getLogger(__name__)


# ORBIT-SEAM: Muon/Pion optimizers own their own sharding and reject Megatron's distributed
# optimizer, so set_default_megatron_args below now conditions use_distributed_optimizer on the
# selected optimizer instead of always forcing it on.
def _is_muon_optimizer(optimizer: str | None) -> bool:
    return optimizer is not None and "muon" in optimizer.lower()


def _is_pion_optimizer(optimizer: str | None) -> bool:
    return optimizer is not None and "pion" in optimizer.lower()


# ORBIT-SEAM: use_distributed_optimizer now conditioned on _is_muon_optimizer/_is_pion_optimizer
# above instead of base's unconditional True; comment style pass (TODO -> Follow-up) below
def set_default_megatron_args(args):
    # Muon and Pion each own their own sharding path and raise on Megatron's
    # distributed optimizer; Adam/SGD keep the historical ZeRO default.
    _opt = getattr(args, "optimizer", None)
    args.use_distributed_optimizer = not (_is_muon_optimizer(_opt) or _is_pion_optimizer(_opt))
    # Follow-up: maybe change this after megatron has good fp8 support
    args.bf16 = not args.fp16
    # placeholders
    if args.seq_length is None:
        args.seq_length = 4096
    args.max_position_embeddings = args.seq_length
    # Notice(Jiajun): new megatron has removed this argument and use dp_reshardable instead of fully_shard
    if os.getenv("DEPRECATED_MEGATRON_COMPATIBLE", "0") == "1":
        args.dist_ckpt_save_pre_mcore_014 = True
    # compatible for megatron
    if hasattr(args, "rope_type") and args.rope_type is None:
        args.rope_type = "yarn" if args.multi_latent_attention else "rope"

    if args.vocab_size and not args.padded_vocab_size:
        args.padded_vocab_size = _vocab_size_with_padding(args.vocab_size, args)

    if not args.tokenizer_model and not args.tokenizer_type:
        logger.info("--tokenizer-model not set, use --hf-checkpoint as tokenizer model.")
        args.tokenizer_model = args.hf_checkpoint
        args.tokenizer_type = "HuggingFaceTokenizer"

    return args
