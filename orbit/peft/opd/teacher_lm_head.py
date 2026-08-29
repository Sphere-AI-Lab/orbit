"""Loads a frozen OPD teacher's LM head for full-vocab KL reconstruction.

`--teacher-score-mode full_vocab` ships only the teacher's last-layer hidden state per
response position (see `compute_teacher_log_probs` in
orbit/rollout/generate_utils/generate_endpoint_utils.py), not its full vocab logits --
transmitting a vocab-sized vector per token over HTTP would be far more expensive. The
training side reconstructs the teacher's full logits itself by multiplying that hidden
state through the teacher's own LM head (`hidden_state @ lm_head.weight.T`), which this
module loads once per process and caches -- sharded along the vocabulary to match Megatron's
partition of the student's output layer when tensor parallelism is on.
"""

import json
import logging
import os
from argparse import Namespace

import torch
from safetensors import safe_open

from orbit.backends.training_utils.parallel import get_parallel_state
from orbit.peft.opd.vocab_parallel import vocab_shard_start

logger = logging.getLogger(__name__)

_TEACHER_LM_HEAD_CACHE: dict[str, torch.Tensor] = {}
# Checkpoints whose cached weight has already been narrowed to this rank's vocabulary shard.
_SHARDED: set[str] = set()


def _find_weight_key(checkpoint_path: str) -> str:
    config_path = os.path.join(checkpoint_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    return "model.embed_tokens.weight" if config.get("tie_word_embeddings", False) else "lm_head.weight"


def _load_weight_from_safetensors(checkpoint_path: str, weight_key: str) -> torch.Tensor:
    index_path = os.path.join(checkpoint_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_file = index["weight_map"][weight_key]
    else:
        shard_file = "model.safetensors"

    shard_path = os.path.join(checkpoint_path, shard_file)
    with safe_open(shard_path, framework="pt") as f:
        return f.get_tensor(weight_key)


def load_teacher_lm_head(args: Namespace, local_vocab_size: int | None = None) -> torch.Tensor:
    """Load (or return the cached) teacher LM head weight for this rank.

    `local_vocab_size` is this rank's logit width. Passing it returns the vocabulary shard
    whose rows line up column-for-column with the student's logits -- fewer rows, or none,
    on ranks whose shard runs past the teacher's real vocabulary. Omitting it returns the
    whole `[vocab_size, hidden_size]` weight, which is what the eager prefetch in
    `actor.init()` wants: it only exists to get the safetensors read off the critical path.
    The first call that does pass it shards the cached tensor in place, so from then on the
    cache -- and the sleep()/wake_up() moves -- carry only this rank's rows.

    Loaded once per process, onto CPU, regardless of its current device -- callers that
    need it on GPU should go through `onload_teacher_lm_head` (see
    orbit/backends/megatron_utils/actor.py's sleep()/wake_up() hooks).
    """
    checkpoint_path = args.teacher_hf_checkpoint
    if checkpoint_path not in _TEACHER_LM_HEAD_CACHE:
        weight_key = _find_weight_key(checkpoint_path)
        weight = _load_weight_from_safetensors(checkpoint_path, weight_key)
        logger.info(
            "Loaded teacher LM head %s (%s) from %s for full-vocab OPD",
            weight_key,
            tuple(weight.shape),
            checkpoint_path,
        )
        _TEACHER_LM_HEAD_CACHE[checkpoint_path] = weight

    parallel_state = get_parallel_state()
    if local_vocab_size is not None and parallel_state.tp.size > 1 and checkpoint_path not in _SHARDED:
        weight = _TEACHER_LM_HEAD_CACHE[checkpoint_path]
        vocab_size = weight.size(0)
        start = vocab_shard_start(local_vocab_size)
        stop = max(min(start + local_vocab_size, vocab_size), start)
        # .clone() rather than keeping the slice as a view, so the rows this rank does not own
        # are actually freed -- the point of sharding is to not carry the whole head (~1.2 GiB
        # at 152k x 2048 in fp32) per rank, GPU-resident between wake_up() and sleep().
        _TEACHER_LM_HEAD_CACHE[checkpoint_path] = weight[start:stop].clone()
        _SHARDED.add(checkpoint_path)
        logger.info(
            "Sharded teacher LM head to vocab [%d, %d) of %d for TP rank %d/%d",
            start,
            stop,
            vocab_size,
            parallel_state.tp.rank,
            parallel_state.tp.size,
        )
    return _TEACHER_LM_HEAD_CACHE[checkpoint_path]


def offload_teacher_lm_head(checkpoint_path: str) -> None:
    """Move the cached teacher LM head to CPU in place, if it's been loaded."""
    if checkpoint_path in _TEACHER_LM_HEAD_CACHE:
        _TEACHER_LM_HEAD_CACHE[checkpoint_path] = _TEACHER_LM_HEAD_CACHE[checkpoint_path].to("cpu")


def onload_teacher_lm_head(checkpoint_path: str, device: torch.device | str) -> None:
    """Move the cached teacher LM head to `device` in place, if it's been loaded."""
    if checkpoint_path in _TEACHER_LM_HEAD_CACHE:
        _TEACHER_LM_HEAD_CACHE[checkpoint_path] = _TEACHER_LM_HEAD_CACHE[checkpoint_path].to(device)
