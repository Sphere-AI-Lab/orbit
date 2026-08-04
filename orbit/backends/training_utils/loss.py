import math
import warnings
from argparse import Namespace
from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from orbit.utils.distributed_utils import distributed_masked_whiten
from orbit.utils.misc import load_function
from orbit.utils.ppo_utils import (
    _safe_clamp_log_ratio,
    _safe_exp_neg_ppo_kl,
    apply_opd_icepop_gate,
    apply_opd_kl_to_advantages,
    calculate_log_probs_and_entropy,
    compute_approx_kl,
    compute_gspo_kl,
    compute_opsm_mask,
    compute_policy_loss,
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
    icepop_gate,
    opd_mopd_advantages,
)
from orbit.utils.types import RolloutBatch

from .cp_utils import (
    _allgather_cp_redistribute,
    all_gather_with_cp,
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
)
from .parallel import get_parallel_state
from .teacher_lm_head import load_teacher_lm_head
from .vocab_parallel import (
    compute_vocab_parallel_topk_log_probs,
    vocab_parallel_log_softmax,
    vocab_parallel_sum,
    vocab_parallel_topk_indices,
    vocab_shard_start,
)

def _response_masked_max(
    x: torch.Tensor,
    *,
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    qkv_format: str = "thd",
    max_seq_lens: list[int] | None = None,
) -> torch.Tensor:
    parallel_state = get_parallel_state()
    cp_size = parallel_state.cp.size

    if cp_size == 1:
        chunk_lengths = response_lengths
        chunked_loss_masks = loss_masks
    else:
        chunk_lengths = []
        chunked_loss_masks = []
        for i, (total_length, response_length, loss_mask) in enumerate(
            zip(total_lengths, response_lengths, loss_masks, strict=False)
        ):
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
            prompt_length = total_length - response_length
            _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )
            loss_mask_0 = loss_mask[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
            loss_mask_1 = loss_mask[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
            chunked_loss_mask = torch.cat([loss_mask_0, loss_mask_1], dim=0)
            chunked_loss_masks.append(chunked_loss_mask)
            chunk_lengths.append(chunked_loss_mask.size(0))

    max_values = []
    for x_i, loss_mask_i in zip(x.split(chunk_lengths, dim=0), chunked_loss_masks, strict=False):
        valid_mask = loss_mask_i.to(device=x_i.device, dtype=torch.bool)
        if x_i.numel() == 0:
            max_values.append(torch.zeros((), dtype=x.dtype, device=x.device))
        else:
            max_value = x_i.masked_fill(~valid_mask, -torch.inf).max()
            max_values.append(torch.where(torch.isneginf(max_value), torch.zeros_like(max_value), max_value))

    if not max_values:
        return torch.zeros((), dtype=x.dtype, device=x.device)
    return torch.stack(max_values).max()


def _response_masked_min(
    x: torch.Tensor,
    *,
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    qkv_format: str = "thd",
    max_seq_lens: list[int] | None = None,
) -> torch.Tensor:
    """Minimum of `x` over loss-mask-valid response positions -- the `_response_masked_max`
    sibling for diagnostics that want a worst-case floor (e.g. `opd_topk/teacher_mass_min`).

    Not implemented as `-_response_masked_max(-x, ...)`: `_response_masked_max`'s fallback
    of `0` for an empty/all-masked sample is a safe *identity* only for a max of a
    non-negative quantity (0 is a lower bound, so it can never win a real max). Negated
    into a min, that same `0` becomes the *supremum* of `-x` for `x` in `[0, 1]` (like
    `teacher_mass`) and would silently dominate every real value -- reported min ends up
    `-0.` regardless of the real data (caught by review; see the regression test). Samples
    with nothing valid are therefore skipped entirely here rather than injected as a fake
    reading; if literally no sample in the microbatch has a valid position, there is no
    worst case to report, so this returns `1.0` (this metric's natural upper bound, i.e.
    "no evidence of a problem") rather than fabricate one.
    """
    parallel_state = get_parallel_state()
    cp_size = parallel_state.cp.size

    if cp_size == 1:
        chunk_lengths = response_lengths
        chunked_loss_masks = loss_masks
    else:
        chunk_lengths = []
        chunked_loss_masks = []
        for i, (total_length, response_length, loss_mask) in enumerate(
            zip(total_lengths, response_lengths, loss_masks, strict=False)
        ):
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
            prompt_length = total_length - response_length
            _, _, _, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )
            loss_mask_0 = loss_mask[tokens_offset[0][0] - prompt_length : tokens_offset[0][1] - prompt_length]
            loss_mask_1 = loss_mask[tokens_offset[1][0] - prompt_length : tokens_offset[1][1] - prompt_length]
            chunked_loss_mask = torch.cat([loss_mask_0, loss_mask_1], dim=0)
            chunked_loss_masks.append(chunked_loss_mask)
            chunk_lengths.append(chunked_loss_mask.size(0))

    min_values = []
    for x_i, loss_mask_i in zip(x.split(chunk_lengths, dim=0), chunked_loss_masks, strict=False):
        valid_mask = loss_mask_i.to(device=x_i.device, dtype=torch.bool)
        if x_i.numel() == 0 or not bool(valid_mask.any()):
            continue
        min_values.append(x_i.masked_fill(~valid_mask, torch.inf).min())

    if not min_values:
        return torch.ones((), dtype=x.dtype, device=x.device)
    return torch.stack(min_values).min()


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned `(logits_chunk, tokens_chunk)` pairs per sample.

    After squeezing batch dimension and applying temperature scaling, this
    function extracts the logits and tokens corresponding to response segments
    for each sample. When context parallelism is disabled, it slices directly
    from the concatenated sequence. With context parallelism enabled, it
    handles split sequences across ranks.

    Args:
        logits: Model outputs with shape `[1, T, V]` (policy) or `[1, T, 1]`
            (value). Must be float32.
        args: Configuration containing `rollout_temperature` for scaling.
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.

    Yields:
        Tuple of `(logits_chunk, tokens_chunk)` where `logits_chunk` is shape
        `[R, V]` (policy) or `[R, 1]` (value) and `tokens_chunk` is shape `[R]`
        (1D int64), both aligned to response tokens for one sample.
    """
    parallel_state = get_parallel_state()
    qkv_format = args.qkv_format

    if not args.true_on_policy_mode:
        assert logits.dtype == torch.float32, f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1))

    rollout_temperature = float(args.rollout_temperature)
    # Skip the out-of-place div when temperature is 1.0: the result is identical
    # to the input but allocates a fresh full-vocab tensor (~6.5 GiB at
    # MAX_TOKENS_PER_GPU=16384 with fp32 logits) that immediately drives the
    # CUDA allocator near OOM on the GRPO loss path.
    # Scale only vocab-shaped logits: value logits [*, 1] are not a distribution
    # (miles cc93d97c4). Non-positive temperatures are rejected at arg validation;
    # the > 0 check here keeps the guard total if a caller bypasses validation.
    if logits.size(-1) > 1 and rollout_temperature > 0 and rollout_temperature != 1.0:
        logits = logits.div(rollout_temperature)
    if args.true_on_policy_mode:
        # Parity contract: SGLang computes log_softmax over bf16 logits, so the
        # training side must feed the same dtype (design doc §2.2 invariant 6).
        if getattr(args, "bf16", False):
            logits = logits.to(torch.bfloat16)
        elif getattr(args, "fp16", False):
            logits = logits.to(torch.float16)

    cp_size = parallel_state.cp.size
    end = 0
    seq_start = 0
    for i, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
    ):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

        if cp_size == 1:
            if qkv_format == "bshd":
                end = max_seq_len * i + total_length
                start = end - response_length
            else:
                end += total_length
                start = end - response_length
            logits_chunk = logits[start - 1 : end - 1]
            tokens_chunk = tokens[-response_length:]
        elif args.allgather_cp:
            # DSA: global concat then contiguous CP split. Each rank owns logits for
            # global positions [chunk_start, chunk_end).
            logits_local_len = logits.size(0)
            cp_rank = parallel_state.cp.rank
            chunk_start = cp_rank * logits_local_len
            chunk_end = chunk_start + logits_local_len

            prompt_length = total_length - response_length
            resp_token_start = seq_start + prompt_length
            resp_token_end = seq_start + total_length
            logit_global_start = resp_token_start - 1
            logit_global_end = resp_token_end - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e <= s:
                logits_chunk = logits[0:0]
                tokens_chunk = tokens[0:0]
            else:
                logits_chunk = logits[s - chunk_start : e - chunk_start]
                tokens_chunk = tokens[(s + 1) - seq_start : (e + 1) - seq_start]
            assert logits_chunk.size(0) == tokens_chunk.size(0), f"{logits_chunk.size(0)} vs {tokens_chunk.size(0)}"
        else:
            # Map response logits and labels into the two local context-parallel chunks.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )

            logits_0, logits_1 = logits[end : end + chunk_size], logits[end + chunk_size : end + 2 * chunk_size]
            end += 2 * chunk_size

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            assert logits_0.size(0) == tokens_0.size(0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
            assert logits_1.size(0) == tokens_1.size(0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"

            logits_chunk = torch.cat([logits_0, logits_1], dim=0)
            tokens_chunk = torch.cat([tokens_0, tokens_1], dim=0)

        seq_start += total_length

        yield logits_chunk, tokens_chunk


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    entropy_no_grad: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
    teacher_topk_ids: list[torch.Tensor] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Compute per-token log-probabilities (and optionally entropy) on responses.

    For each sample, extracts response-aligned logits and tokens, then computes
    log-probabilities via softmax across the tensor-parallel group. Log-probs
    are squeezed from `[R, 1]` to `[R]`. Entropy values are always appended
    (even when `with_entropy=False`), but only included in the result dict
    when requested.

    Args:
        logits: Policy logits with shape `[1, T, V]`.
        args: Configuration (temperature applied in `get_responses`).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: If True, include "entropy" key in result.
        non_loss_data: Unused; kept for API compatibility.
        teacher_topk_ids: For on_policy_distillation's "topk" mode, a list of
            `[R, K]` token-id tensors (one per sample, from TeacherManager) to
            additionally gather student log-probs for at each response
            position. When None (the default), no extra gather is done.

    Returns:
        Dict with key "log_probs" mapping to a list of `[R]` tensors per
        sample. If `with_entropy` is True, also includes "entropy" key with
        a list of `[R]` tensors. If `teacher_topk_ids` is given, also
        includes "student_topk_log_probs" mapping to a list of `[R, K]`
        tensors.
    """
    parallel_state = get_parallel_state()
    assert non_loss_data

    if teacher_topk_ids is not None and args.allgather_cp:
        raise NotImplementedError(
            "on_policy_distillation opd_loss_type='topk' does not support --allgather-cp: "
            "the CP redistribution helper only handles 1D per-token tensors, not the "
            "[R, K] student_topk_log_probs tensor."
        )

    # dev's opd_jsd pattern: only pay for the TP collective path when TP is actually
    # on, rather than czy's unconditional parallel_state.tp.group.
    tp_group = parallel_state.tp.group if parallel_state.tp.size > 1 else None

    log_probs_list = []
    entropy_list = []
    topk_log_probs_list = [] if teacher_topk_ids is not None else None
    topk_ids_iter = teacher_topk_ids if teacher_topk_ids is not None else [None] * len(unconcat_tokens)
    for (logits_chunk, tokens_chunk), sample_topk_ids in zip(
        get_responses(
            logits,
            args=args,
            unconcat_tokens=unconcat_tokens,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        ),
        topk_ids_iter,
        strict=True,
    ):
        log_prob, entropy = calculate_log_probs_and_entropy(
            logits_chunk,
            tokens_chunk,
            parallel_state.tp.group,
            with_entropy=with_entropy,
            entropy_no_grad=entropy_no_grad,
            chunk_size=args.log_probs_chunk_size,
            true_on_policy=args.true_on_policy_mode,
            vocab_size=getattr(args, "vocab_size", None),
        )

        log_probs_list.append(log_prob.squeeze(-1))
        entropy_list.append(entropy)

        if sample_topk_ids is not None:
            # Deliberately not calculate_log_probs_and_entropy/fused_vocab_parallel_
            # cross_entropy here: that path is wrapped in @jit_fuser (torch.compile),
            # which recompiles/re-autotunes per new input shape. Calling it many times
            # (once per top-k slot) from inside a pipeline-parallel forward_step has
            # been observed to crash with "CUDA driver error: invalid argument" during
            # Triton autotuning. compute_vocab_parallel_topk_log_probs uses only plain
            # eager ops and computes the log-normalizer once for all k.
            topk_log_probs_list.append(
                compute_vocab_parallel_topk_log_probs(
                    logits_chunk,
                    sample_topk_ids,
                    tp_group,
                )
            )

    res = {
        "log_probs": log_probs_list,
    }
    if with_entropy:
        res["entropy"] = entropy_list
    if topk_log_probs_list is not None:
        res["student_topk_log_probs"] = topk_log_probs_list

    # we need to turn the all gather kv into zigzag ring attn kv
    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits=logits,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return res


def get_values(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Extract per-token value predictions over response tokens.

    For each sample, extracts response-aligned chunks from the value head
    output and squeezes the final dimension from `[R, 1]` to `[R]`.

    Args:
        logits: Value head output with shape `[1, T, 1]`.
        args: Configuration (passed to `get_responses` which uses
            `rollout_temperature` even though values don't need temperature).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: Unused; kept for signature compatibility.
        non_loss_data: Unused; kept for signature compatibility.

    Returns:
        Dict with key "values" mapping to a list of `[R]` value tensors
        per sample.
    """
    value_list = []
    for logits_chunk, _ in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        assert logits_chunk.size(-1) == 1, f"{logits_chunk.shape}"
        value_list.append(logits_chunk.squeeze(-1))

    res = {
        "values": value_list,
    }

    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits=logits,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return res


def compute_advantages_and_returns(args: Namespace, rollout_data: RolloutBatch, role: str = "actor") -> None:
    """Compute advantages and returns in-place based on `args.advantage_estimator`.

    This function extracts rewards, log-probs, values, and masks from
    `rollout_data`, computes KL divergences, then applies the chosen advantage
    estimator. Supported methods: "grpo", "gspo", "ppo", "reinforce_plus_plus",
    and "reinforce_plus_plus_baseline". When `args.normalize_advantages` is
    True, advantages are whitened across the data-parallel group using masked
    statistics.

    Early returns if both `log_probs` and `values` are None (intermediate
    pipeline stages).

    Args:
        args: Configuration specifying estimator type, KL coefficient,
            normalization settings, and other hyperparameters.
        rollout_data: Dict containing input lists ("log_probs", "ref_log_probs",
            "rewards", "values", "response_lengths", "loss_masks",
            "total_lengths"). Modified in-place to add "advantages" and
            "returns" keys, each mapping to lists of tensors per sample.
        role: "actor" or "critic". The critic never receives teacher_log_probs
            (sync_actor_critic_data does not broadcast them) and its value loss
            consumes `returns`, which the OPD blend does not touch — so OPD
            advantage adjustments are skipped for role="critic".
    """
    parallel_state = get_parallel_state()
    log_probs: list[torch.Tensor] = rollout_data.get("rollout_log_probs" if args.use_rollout_logprobs else "log_probs")
    ref_log_probs: list[torch.Tensor] = rollout_data.get("ref_log_probs")
    rewards: list[float] = rollout_data.get("rewards")
    values: None | list[torch.Tensor] = rollout_data.get("values")
    response_lengths: list[int] = rollout_data.get("response_lengths")
    loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks")
    total_lengths: list[int] = rollout_data.get("total_lengths")
    max_seq_lens: list[int] | None = rollout_data.get("max_seq_lens", None)

    # return when not the last pp stage.
    if log_probs is None and values is None:
        return

    if args.kl_coef == 0 or not log_probs:
        # when kl_coef is 0, we won't compute ref_log_prob
        xs = log_probs if log_probs is not None else values
        kl = [torch.zeros_like(x, dtype=torch.float32, device=x.device) for x in xs]
    else:
        kl = [
            compute_approx_kl(
                log_probs[i],
                ref_log_probs[i],
                kl_loss_type=args.kl_loss_type,
            )
            for i in range(len(log_probs))
        ]

    if args.advantage_estimator in ["grpo", "gspo"]:
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_grpo_returns(rewards, kl)
        # Follow-up: is the copy necessary?
        advantages = [r for r in returns]

    elif args.advantage_estimator == "ppo":
        terminal_rewards = rewards
        token_rewards = []
        kl_coef = -args.kl_coef
        for k in kl:
            k *= kl_coef
            token_rewards.append(k)
        advantages, returns = get_advantages_and_returns_batch(
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            values_list=values,
            rewards_list=token_rewards,
            terminal_rewards=terminal_rewards,
            qkv_format=args.qkv_format,
            max_seq_lens=max_seq_lens,
            loss_masks=loss_masks,
            gamma=args.gamma,
            lambd=args.lambd,
        )

    elif args.advantage_estimator == "reinforce_plus_plus":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_reinforce_plus_plus_returns(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            response_lengths=response_lengths,
            total_lengths=total_lengths,
            kl_coef=args.kl_coef,
            gamma=args.gamma,
        )
        advantages = [r for r in returns]

    elif args.advantage_estimator == "reinforce_plus_plus_baseline":
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        advantages = get_reinforce_plus_plus_baseline_advantages(
            rewards=rewards,
            kl=kl,
            loss_masks=loss_masks,
            kl_coef=args.kl_coef,
        )
        returns = advantages

    elif args.advantage_estimator == "on_policy_distillation":
        advantages = opd_mopd_advantages(rollout_data, log_probs, rollout_data.get("response_lengths"))
        returns = advantages

    else:
        raise NotImplementedError(f"advantage_estimator {args.advantage_estimator} is not supported. ")

    if role == "actor" and getattr(args, "use_opd", False):
        apply_opd_kl_to_advantages(args.opd_kl_coef, rollout_data, advantages, log_probs)

    # Optional async/off-policy ICE-POP correction for the OPD advantage (pure-MOPD
    # or blend): hard-gate tokens whose train/rollout importance ratio leaves the band.
    if role == "actor" and getattr(args, "opd_icepop", False):
        apply_opd_icepop_gate(rollout_data, advantages, args.tis_clip_low, args.tis_clip)

    # Follow-up: OpenRLHF always does advantages normalization but veRL doesn't seem to do it.
    if args.normalize_advantages:
        all_advs = torch.cat(advantages)
        cp_size = parallel_state.cp.size
        if cp_size == 1:
            all_masks = torch.cat(loss_masks)
        else:
            mask_chunks = []
            for i in range(len(advantages)):
                total_len = total_lengths[i]
                response_len = response_lengths[i]
                prompt_len = total_len - response_len
                max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

                _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
                    total_len, response_len, args.qkv_format, max_seq_len
                )

                # Convert global offsets to response-space offsets
                s0, e0 = token_offsets[0]
                s1, e1 = token_offsets[1]
                res_s0, res_e0 = max(0, s0 - prompt_len), max(0, e0 - prompt_len)
                res_s1, res_e1 = max(0, s1 - prompt_len), max(0, e1 - prompt_len)

                local_mask_parts = []
                full_mask = loss_masks[i]
                if res_e0 > res_s0:
                    local_mask_parts.append(full_mask[res_s0:res_e0])
                if res_e1 > res_s1:
                    local_mask_parts.append(full_mask[res_s1:res_e1])

                # Concatenate the parts to form the final mask chunk for this rank and this sequence
                local_mask_chunk = (
                    torch.cat(local_mask_parts)
                    if local_mask_parts
                    else torch.tensor([], device=all_advs.device, dtype=full_mask.dtype)
                )
                mask_chunks.append(local_mask_chunk)

            all_masks = torch.cat(mask_chunks)

        if all_masks.numel() > 0:
            assert (
                all_advs.size() == all_masks.size()
            ), f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"
            dp_group = parallel_state.intra_dp.group

            whitened_advs_flat = distributed_masked_whiten(
                all_advs,
                all_masks,
                process_group=dp_group,
                shift_mean=True,
            )
            chunk_lengths = [chunk.size(0) for chunk in advantages]
            advantages = list(torch.split(whitened_advs_flat, chunk_lengths))

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def vanilla_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(old_log_probs - rollout_log_probs)
    tis_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()
    metrics = {
        "tis": tis.clone().detach(),
        "tis_clipfrac": tis_clipfrac.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
    }
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics


def icepop_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    ice_ratio = torch.exp(old_log_probs - rollout_log_probs)
    ice_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    ice_weight = icepop_gate(ice_ratio, args.tis_clip_low, args.tis_clip)
    ice_clipfrac = (ice_weight != ice_ratio).float()
    metrics = {
        "tis": ice_ratio.clone().detach(),
        "tis_clipfrac": ice_clipfrac.clone().detach(),
        "tis_abs": ice_abs.clone().detach(),
    }
    pg_loss = pg_loss * ice_weight
    return pg_loss, loss_masks, metrics


def policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute policy loss (PPO/GSPO) and metrics.

    Computes current log-probabilities and entropy from model logits, then
    calculates PPO-style clipped policy gradient loss. For GSPO, gathers
    full sequences via context-parallel all-gather before computing per-sample
    KL. Optionally applies TIS (Truncated Importance Sampling) correction and
    adds KL loss term if configured.

    Args:
        args: Configuration controlling advantage estimator, clipping thresholds,
            entropy/KL coefficients, and TIS settings.
        batch: Mini-batch containing "advantages", "log_probs" (old policy),
            "unconcat_tokens", "response_lengths", "total_lengths", "loss_masks",
            and optionally "ref_log_probs" and "rollout_log_probs".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and `metrics`
        is a dict containing detached scalars: "loss", "pg_loss",
        "entropy_loss", "pg_clipfrac", "ppo_kl". Additional keys "kl_loss",
        "tis", "ois", "tis_clipfrac" are included when the respective features
        are enabled.
    """
    parallel_state = get_parallel_state()
    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
        entropy_no_grad=args.entropy_coef == 0.0,
        max_seq_lens=max_seq_lens,
    )

    log_probs = log_probs_and_entropy["log_probs"]

    # Pre-gather log probs if needed by OPSM or GSPO to avoid duplicate gathering
    need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"

    full_log_probs = None
    full_old_log_probs = None
    if need_full_log_probs:
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                log_probs, total_lengths, response_lengths, strict=False
            )
        ]
        full_old_log_probs = [
            all_gather_with_cp(old_log_prob, total_length, response_length)
            for old_log_prob, total_length, response_length in zip(
                old_log_probs, total_lengths, response_lengths, strict=False
            )
        ]

    # Compute OPSM mask if enabled
    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=batch["advantages"],
            loss_masks=batch["loss_masks"],
        )

    # Compute KL divergence (GSPO uses sequence-level KL, others use per-token KL)
    if args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_probs,
            loss_masks=batch["loss_masks"],
        )
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
    else:
        old_log_probs = torch.cat(old_log_probs, dim=0)
        log_probs = torch.cat(log_probs, dim=0)
        if getattr(args, "force_on_policy_ratio", False):
            # Ratio pinned to exactly 1.0 with the gradient preserved: the surrogate
            # degenerates to REINFORCE, the exact objective of pure sampled-token MOPD.
            # Independent behaviour correction may still be applied with TIS.
            ppo_kl = log_probs.detach() - log_probs
        else:
            ppo_kl = old_log_probs - log_probs

    pg_loss, pg_clipfrac = compute_policy_loss(
        ppo_kl,
        advantages,
        args.eps_clip,
        args.eps_clip_high,
        args.eps_clip_c,
    )

    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    # Apply off-policy correction using importance sampling if enabled
    if args.get_mismatch_metrics or args.use_tis:
        # NOTE:
        # `tis_func` may apply rejection-sampling style masking (RS) and return `modified_response_masks`.
        # We rebuild `sum_of_sample_mean` with those masks to correct denominators for loss/backprop.
        #
        # However, mismatch/TIS/RS metrics (e.g., "truncate_fraction") are often defined over the
        # *pre-RS* valid tokens. If we aggregate metrics with `modified_response_masks`, the rejected
        # tokens are excluded from the denominator and the metric can be artificially driven to 0.
        # Keep a copy of the original reducer (based on `batch["loss_masks"]`) for metric aggregation.
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean

        assert "rollout_log_probs" in batch, "rollout_log_probs must be provided for TIS"

        ois = _safe_exp_neg_ppo_kl(ppo_kl)
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss,
            "train_log_probs": batch["log_probs"],
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
            "parallel_state": parallel_state,
            "max_seq_lens": max_seq_lens,
        }

        if args.custom_tis_function_path is not None:
            tis_func = load_function(args.custom_tis_function_path)
        else:
            tis_func = vanilla_tis_function
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)

        # [decouple IS and rejection] Rebuild sum_of_sample_mean with modified_response_masks for denominator correction
        # modified_response_masks will be sliced with cp in get_sum_of_sample_mean
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            args.calculate_per_token_loss,
            args.qkv_format,
            max_seq_lens,
        )

    # Determine pg_loss reducer: use custom if specified, otherwise default
    if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
        custom_pg_loss_reducer_func = load_function(args.custom_pg_loss_reducer_function_path)
        # Determine which loss_masks to use for pg_loss reducer
        pg_loss_masks = modified_response_masks if (args.get_mismatch_metrics or args.use_tis) else batch["loss_masks"]
        pg_loss_reducer = custom_pg_loss_reducer_func(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
    else:
        pg_loss_reducer = sum_of_sample_mean

    pg_loss = pg_loss_reducer(pg_loss)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    # entropy loss
    entropy = log_probs_and_entropy["entropy"]
    entropy = torch.cat(entropy, dim=0)
    entropy_loss = sum_of_sample_mean(entropy)

    loss = pg_loss - args.entropy_coef * entropy_loss

    if args.use_kl_loss:
        ref_log_probs = batch["ref_log_probs"]
        ref_log_probs = torch.cat(ref_log_probs, dim=0)
        importance_ratio = None
        if args.use_unbiased_kl:
            # Route the exponent through the same safe clamp as every other
            # ratio path: async/off-policy drift can push |log_probs -
            # old_log_probs| past exp overflow. Differentiable inside the band.
            importance_ratio = _safe_clamp_log_ratio(log_probs - old_log_probs).exp()
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl)

        loss = loss + args.kl_loss_coef * kl_loss

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    train_rollout_logprob_abs_diff = None
    train_rollout_logprob_abs_diff_max = None
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        train_rollout_logprob_token_abs_diff = (old_log_probs - rollout_log_probs).abs()
        train_rollout_logprob_abs_diff = sum_of_sample_mean(train_rollout_logprob_token_abs_diff)
        train_rollout_logprob_abs_diff_max = _response_masked_max(
            train_rollout_logprob_token_abs_diff,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            loss_masks=batch["loss_masks"],
            qkv_format=getattr(args, "qkv_format", "thd"),
            max_seq_lens=max_seq_lens,
        )

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
    }

    if train_rollout_logprob_abs_diff is not None:
        reported_loss["train_rollout_logprob_abs_diff"] = train_rollout_logprob_abs_diff.clone().detach()
        reported_loss["train_rollout_logprob_abs_diff_max"] = train_rollout_logprob_abs_diff_max.clone().detach()

    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()

    if args.get_mismatch_metrics or args.use_tis:
        # Aggregate mismatch/TIS/RS related metrics with the *pre-RS* masks.
        # See comment above where `sum_of_sample_mean_for_mismatch_metrics` is defined.
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        # Assume all metrics are already cloned and detached
        for metric_key, metric_value in tis_metrics.items():
            key_name = f"{metric_key}"
            reported_loss[key_name] = sum_of_sample_mean_for_mismatch_metrics(metric_value)

    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac

    return loss, reported_loss


def value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute clipped value loss and metrics.

    Extracts current value predictions from `logits`, compares them against
    stored old values with clipping, and computes the maximum of clipped and
    unclipped squared errors (PPO-style value clipping).

    Args:
        args: Configuration containing `value_clip` threshold.
        batch: Mini-batch with "values" (old predictions), "returns",
            "unconcat_tokens", "total_lengths", and "response_lengths".
        logits: Value head output with shape `[1, T, 1]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and
        `metrics` contains detached scalars "value_loss" and "value_clipfrac".
    """
    old_values = torch.cat(batch["values"], dim=0)

    values = get_values(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    values = torch.cat([value.flatten() for value in values["values"]], dim=0)

    returns = torch.cat(batch["returns"], dim=0)

    values_clipfrac = torch.abs(values - old_values) > args.value_clip
    values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2
    loss = torch.max(surr1, surr2)

    loss = sum_of_sample_mean(loss)
    values_clipfrac = sum_of_sample_mean(values_clipfrac.float())

    # make sure the gradient could backprop correctly.
    if values.numel() == 0:
        loss += 0 * values.sum()

    reported_loss = {
        "value_loss": loss.clone().detach(),
        "value_clipfrac": values_clipfrac.clone().detach(),
    }

    return loss, reported_loss


def sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute supervised fine-tuning loss over response tokens.

    Computes log-probabilities of the ground-truth tokens in the response
    segments and returns the negative log-likelihood as the loss.

    Args:
        args: Configuration (passed through to helpers).
        batch: Mini-batch with "unconcat_tokens", "response_lengths", and
            "total_lengths".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `metrics` contains a single detached
        scalar "loss".
    """
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )


def _clip_pointwise_kl(kl_elem: torch.Tensor, clip: float | None) -> torch.Tensor:
    """Cap each individual (response-position, vocab-token) divergence summand before it is
    summed over the vocabulary dimension.

    Borrowed from OPSD's (github.com/siyan-zhao/OPSD) `--jsd_token_clip`: they found stylistic
    tokens (e.g. "wait", "think") can carry 6-15x higher per-vocab-entry divergence than
    content/math tokens and dominate the training signal if left unclipped.
    """
    if clip is None:
        return kl_elem
    return kl_elem.clamp(max=clip)


def opd_jsd_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute exact full-vocabulary generalized JSD against a frozen teacher.

    Follows Eq. (1) of the GKD paper. The teacher distribution is reconstructed locally from
    `batch["teacher_hidden_states"]` and the teacher's LM head rather than shipped over the
    wire. `--opd-jsd-beta` (`b`) interpolates between forward `KL(teacher||student)` at `b=0`
    and reverse `KL(student||teacher)` at `b=1`, over the mixture `M = (1-b)*student + b*teacher`:

        jsd(b) = b * KL(teacher || M) + (1-b) * KL(student || M)

    `batch["teacher_hidden_states"]` holds one CPU fp32 tensor per sample, already CP-sliced
    row-for-row with this rank's response logits by `get_rollout_data` (the same
    `slice_log_prob_with_cp` treatment `teacher_log_probs` gets), so each chunk only needs a
    device move here.

    Returns `(loss, metrics)`; `metrics` holds detached "loss" and "entropy", plus "kl_loss"
    under --use-kl-loss and "topk_overlap_k{k}" under --opd-log-topk-overlap.
    """
    parallel_state = get_parallel_state()
    assert not args.allgather_cp, (
        "opd_jsd_loss does not support --allgather-cp: teacher_hidden_states are CP-sliced by "
        "slice_log_prob_with_cp (get_logits_and_tokens_offset_with_cp chunks), not by the DSA "
        "split get_responses takes under that flag."
    )
    beta = args.opd_jsd_beta
    assert 0.0 <= beta <= 1.0, f"--opd-jsd-beta must be in [0, 1], got {beta}"
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    tp_group = parallel_state.tp.group if parallel_state.tp.size > 1 else None
    # The student's logits are the authority on how the vocabulary is split -- they carry
    # exactly this rank's shard, whichever global vocab size the model was actually built with.
    local_vocab_size = logits.size(-1)
    vocab_start = vocab_shard_start(local_vocab_size) if tp_group is not None else 0
    teacher_lm_head = load_teacher_lm_head(args, local_vocab_size=local_vocab_size).to(
        logits.device, torch.float32
    )
    # How many of this rank's vocab columns are real rather than divisibility padding.
    # Clamped from above too: a bigger same-tokenizer teacher can carry MORE padded rows
    # than the student (Qwen2.5-7B pads to 152064 vs 151936 below 3B); rows past the
    # student's width are padding the student cannot emit, so dropping them conditions
    # the teacher on the shared vocabulary. The TP shard path already slices this way.
    teacher_vocab_size = min(teacher_lm_head.size(0), local_vocab_size)

    kl_per_sample = []
    entropy_per_sample = []

    ref_kl_sampled_log_probs = [] if args.use_kl_loss else None
    topk_ks = tuple(args.opd_topk_overlap_ks) if args.opd_log_topk_overlap else ()
    topk_overlap_per_sample: dict[int, list[torch.Tensor]] = {k: [] for k in topk_ks}
    max_seq_lens = batch.get("max_seq_lens", None)
    responses = get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    )
    for i, (logits_chunk, tokens_chunk) in enumerate(responses):
        vocab_size = logits_chunk.size(-1)
        # Columns past the teacher's real vocab stay at this fill. A large finite negative
        # rather than -inf, which would go NaN (0 * -inf) on stray student mass.
        teacher_log_probs_full = logits_chunk.new_full((logits_chunk.size(0), vocab_size), -1e4)
        if logits_chunk.size(0) > 0:
            with torch.no_grad():
                teacher_hidden_states = batch["teacher_hidden_states"][i].to(
                    dtype=torch.float32, device=logits_chunk.device
                )
                assert teacher_hidden_states.size(0) == logits_chunk.size(0), (
                    f"sample {i}: {teacher_hidden_states.size(0)} teacher hidden-state rows vs "
                    f"{logits_chunk.size(0)} response logits -- get_rollout_data's CP slicing "
                    "has drifted from get_responses()."
                )
                teacher_logits = teacher_hidden_states @ teacher_lm_head[:teacher_vocab_size].T

                rollout_temperature = float(args.rollout_temperature)
                if rollout_temperature != 1.0:
                    teacher_logits.div_(rollout_temperature)
                # The clamp bounds forward KL, which weights by the fixed teacher probs.
                teacher_log_probs_full[:, :teacher_vocab_size] = vocab_parallel_log_softmax(
                    teacher_logits, tp_group
                ).clamp_(min=args.opd_log_prob_min_clamp)

        student_log_probs_full = vocab_parallel_log_softmax(logits_chunk.float(), tp_group).clamp(
            min=args.opd_log_prob_min_clamp
        )
        student_probs_full = student_log_probs_full.exp()
        teacher_probs_full = teacher_log_probs_full.exp()

        if topk_ks:
            max_k = max(topk_ks)
            student_topk_idx = vocab_parallel_topk_indices(student_log_probs_full, max_k, vocab_start, tp_group)
            teacher_topk_idx = vocab_parallel_topk_indices(teacher_log_probs_full, max_k, vocab_start, tp_group)
            topk_match = student_topk_idx.unsqueeze(-1) == teacher_topk_idx.unsqueeze(-2)  # [R, max_k, max_k]
            for k in topk_ks:
                overlap_count = topk_match[:, :k, :k].any(dim=-1).sum(dim=-1)  # [R]
                topk_overlap_per_sample[k].append(overlap_count.float() / k)

        if beta == 0.0:
            kl_elem = teacher_probs_full * (teacher_log_probs_full - student_log_probs_full)
        elif beta == 1.0:
            kl_elem = student_probs_full * (student_log_probs_full - teacher_log_probs_full)
        else:
            mixture_log_probs = torch.logsumexp(
                torch.stack(
                    [student_log_probs_full + math.log1p(-beta), teacher_log_probs_full + math.log(beta)]
                ),
                dim=0,
            )
            kl_teacher_elem = teacher_probs_full * (teacher_log_probs_full - mixture_log_probs)
            kl_student_elem = student_probs_full * (student_log_probs_full - mixture_log_probs)

            kl_elem = beta * kl_teacher_elem + (1 - beta) * kl_student_elem

        kl_elem = _clip_pointwise_kl(kl_elem, args.opd_jsd_pointwise_clip)
        # The vocab sum crosses TP shards, so it must complete before the per-position clamp.
        kl = vocab_parallel_sum(kl_elem, tp_group).clamp(max=args.opd_loss_max_clamp)
        kl_per_sample.append(kl)
        entropy_per_sample.append(vocab_parallel_sum(-(student_probs_full * student_log_probs_full), tp_group))
        if ref_kl_sampled_log_probs is not None:
            student_log_prob, _ = calculate_log_probs_and_entropy(
                logits_chunk,
                tokens_chunk,
                parallel_state.tp.group,
                chunk_size=args.log_probs_chunk_size,
                true_on_policy=args.true_on_policy_mode,
                vocab_size=getattr(args, "vocab_size", None),
            )
            ref_kl_sampled_log_probs.append(student_log_prob.squeeze(-1))

    kl_per_sample = torch.cat(kl_per_sample, dim=0)
    loss = sum_of_sample_mean(kl_per_sample)

    # compute_ref_log_probs() populates batch["ref_log_probs"] for any loss_type.
    ref_kl_loss = None
    if args.use_kl_loss:
        student_sampled_log_probs = torch.cat(ref_kl_sampled_log_probs, dim=0)
        ref_log_probs = torch.cat(batch["ref_log_probs"], dim=0)
        ref_kl = compute_approx_kl(student_sampled_log_probs, ref_log_probs, kl_loss_type=args.kl_loss_type)
        ref_kl_loss = sum_of_sample_mean(ref_kl)
        loss = loss + args.kl_loss_coef * ref_kl_loss

    # make sure the gradient could backprop correctly.
    if kl_per_sample.numel() == 0:
        loss = loss + 0 * logits.sum()

    # Per-token quantities, so the same reduction as loss keeps them on a comparable scale.
    entropy_concat = torch.cat(entropy_per_sample, dim=0)
    entropy_metric = sum_of_sample_mean(entropy_concat)

    topk_overlap_concat = {k: torch.cat(topk_overlap_per_sample[k], dim=0) for k in topk_ks}
    topk_overlap_metric = {k: sum_of_sample_mean(topk_overlap_concat[k]) for k in topk_ks}

    metrics = {
        "loss": loss.clone().detach(),
        "entropy": entropy_metric.clone().detach(),
    }
    if ref_kl_loss is not None:
        metrics["kl_loss"] = ref_kl_loss.clone().detach()
    for k in topk_ks:
        metrics[f"topk_overlap_k{k}"] = topk_overlap_metric[k].clone().detach()

    return (loss, metrics)


_TOPK_LOG_INF = -100.0
_TOPK_KL_TYPES = ("forward", "reverse", "mixed")


def _topk_kl_terms(
    teacher_topk_logprobs: torch.Tensor,
    student_topk_logprobs: torch.Tensor,
    entropy: torch.Tensor | None,
    kl_type: str,
    mixed_weight: float,
    zero_outside: bool,
) -> torch.Tensor:
    """Per-token top-k KL between the frozen teacher and the student, truncated to the
    teacher's own top-k support (plus, for the reverse direction, an optional correction
    for the student mass that falls outside that support).

    Padded slots (see `orbit.rollout.opd_sglang._TOPK_PAD_LOGPROB`) carry a teacher
    log-prob of -1e4, so `teacher_topk_logprobs.exp()` underflows to exactly 0.0 in
    float32 -- used below as an exact (not approximate) validity mask over the K
    dimension. Both the forward and the uncorrected-reverse sums only ever touch valid
    slots, so a padded column changes nothing (czy's `_topk_forward_kl`, generalized to
    all three directions; their `renormalize` branch is dropped per spec).

    The same float32 underflow floor sits at log-prob ~-103.97 (ln(2**-149), the
    smallest denormal): a genuine (non-pad) teacher entry that far below the peak also
    reads as invalid and is dropped from the support the same way a pad slot is, with
    the reverse-direction correction re-flooring it at `_TOPK_LOG_INF` (-100.0) instead
    of its true value -- unreachable at any realistic `k` (the teacher's own top-k
    entries are never that improbable), but reachable once `k` approaches the full
    vocabulary.

    Forward (teacher-weighted, `--opd-kl-type forward`):
        `sum_K valid * teacher_prob * (teacher_log_prob - student_log_prob)`
    `zero_outside` is structurally inert here -- the sum never leaves the teacher's own
    reported support -- so passing it true is a caller mistake we warn about once
    (Python's default warning filter already dedupes by message+location) rather than
    silently ignore.

    Reverse (student-weighted, `--opd-kl-type reverse`):
        `sum_K valid * student_prob * (student_log_prob - teacher_log_prob)`
    truncated the same way, but `student_prob`/`student_log_prob` keep gradients (the
    teacher side is always detached -- it is frozen). Without `zero_outside`, this
    silently drops all of the student's probability mass that falls *outside* the
    teacher's reported top-k, which lets the optimizer push probability there for free.
    `zero_outside=True` adds a correction that makes the result exactly equal to the
    full-vocabulary reverse KL against a teacher extended with a `log_inf=-100.0`
    log-prob at every out-of-support token id (see the closed-form test for the
    from-scratch full-vocab derivation this mirrors):

        correction = (H_all - sum_K valid * student_prob * student_log_prob)
                     - log_inf * (1 - sum_K valid * student_prob)

    where `H_all = sum_v student_prob(v) * student_log_prob(v)` is the student's own
    full-vocabulary self-term (note: negative). `calculate_log_probs_and_entropy`'s
    "entropy" output was verified (see the closed-form correction test, which pins this
    sign) to already be the *standard* positive entropy `-sum_v p_v log p_v`, so
    `H_all = -entropy` here, not `entropy` directly.

    Mixed (`--opd-kl-type mixed`, `--opd-mixed-kl-weight` on the forward term, NeMo's
    convention): `w * forward + (1 - w) * reverse`, where `reverse` already includes its
    own correction when requested -- so the correction is implicitly scaled by `(1 - w)`
    too, matching NeMo's DistillationLossFn.

    Args:
        teacher_topk_logprobs: `[R, K]` teacher log-probs at its own top-k token ids.
            Treated as a constant; detached here regardless of what the caller passes.
        student_topk_logprobs: `[R, K]` student log-probs at those same ids,
            differentiable w.r.t. the student's parameters.
        entropy: `[R]` student full-vocabulary entropy (standard positive convention),
            or `None`. Required only when `zero_outside` and `kl_type != "forward"`.
        kl_type: One of "forward", "reverse", "mixed".
        mixed_weight: Weight on the forward term when `kl_type == "mixed"`, in `[0, 1]`.
        zero_outside: Whether to add the reverse-direction out-of-support correction.

    Returns:
        `[R]` tensor of per-token KL values (the loss to minimize).
    """
    if kl_type not in _TOPK_KL_TYPES:
        raise ValueError(f"Unknown top-k KL type: {kl_type!r}")

    # Teacher is frozen: its log-probs arrive as plain (non-autograd) tensors from Ray
    # anyway, but detach explicitly so the intent -- no gradient into the teacher side --
    # is unambiguous regardless of caller.
    teacher_topk_logprobs = teacher_topk_logprobs.detach()
    teacher_weights = teacher_topk_logprobs.exp()
    valid = teacher_weights > 0  # exact float32 underflow at padded slots, see above
    masked_teacher_weights = torch.where(valid, teacher_weights, torch.zeros_like(teacher_weights))

    if kl_type == "forward":
        if zero_outside:
            warnings.warn(
                "--opd-topk-zero-outside has no effect with --opd-kl-type forward: the "
                "forward top-k KL only ever sums over the teacher's own reported support.",
                stacklevel=2,
            )
        return (masked_teacher_weights * (teacher_topk_logprobs - student_topk_logprobs)).sum(dim=-1)

    student_weights = student_topk_logprobs.exp()
    masked_student_weights = torch.where(valid, student_weights, torch.zeros_like(student_weights))
    reverse = (masked_student_weights * (student_topk_logprobs - teacher_topk_logprobs)).sum(dim=-1)

    if zero_outside:
        if entropy is None:
            raise ValueError("`entropy` is required when `zero_outside` is set for the reverse-direction term.")
        h_all = -entropy  # see docstring: the machinery's "entropy" is the standard +H convention
        sum_k_student_weight = masked_student_weights.sum(dim=-1)
        sum_k_student_weighted_logprob = (masked_student_weights * student_topk_logprobs).sum(dim=-1)
        correction = (h_all - sum_k_student_weighted_logprob) - _TOPK_LOG_INF * (1 - sum_k_student_weight)
        reverse = reverse + correction

    if kl_type == "reverse":
        return reverse

    forward = (masked_teacher_weights * (teacher_topk_logprobs - student_topk_logprobs)).sum(dim=-1)
    return mixed_weight * forward + (1 - mixed_weight) * reverse


def _resolve_opd_topk_kl_type(args: Namespace) -> tuple[str, float]:
    """Local counterpart to `orbit.rollout.opd_sglang._get_kl_type` -- kept independent
    (not imported) so this training-side loss module doesn't reach into rollout code for
    a two-line resolution. Mirrors NeMo-RL's DistillationLossFn `kl_type`/`mixed_kl_weight`
    convention: `reverse` (default), `forward`, or `mixed` with `--opd-mixed-kl-weight` on
    the forward term.
    """
    kl_type = getattr(args, "opd_kl_type", "reverse") or "reverse"
    if kl_type not in _TOPK_KL_TYPES:
        raise ValueError(f"Unknown OPD KL type: {kl_type!r}")
    mixed_weight = float(getattr(args, "opd_mixed_kl_weight", 0.5))
    if not (0.0 <= mixed_weight <= 1.0):
        raise ValueError(f"--opd-mixed-kl-weight must be in [0, 1], got {mixed_weight}.")
    return kl_type, mixed_weight


def opd_topk_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Direct (non-policy-gradient) top-k KL on_policy_distillation loss.

    Unlike `--opd-loss-type sampled_token` (which treats `teacher_log_prob(a_t) -
    student_log_prob(a_t)` as a REINFORCE advantage on the token the student happened to
    sample, routed through `compute_policy_loss`'s PPO ratio/clip), this backpropagates
    directly through the student's log-probs at all `--opd-topk-k` of the teacher's top-k
    token ids for every response position -- mirroring verl's `forward_kl_topk` (see
    https://verl.readthedocs.io/en/latest/algo/opd.html, "PG OPD" section). There is no
    importance-sampling ratio here (and hence no PPO clip, no old/rollout log-probs
    needed): the loss is computed directly against the current parameters in the same
    forward pass, so there is no train/rollout policy mismatch to correct for.

    `--opd-kl-type` (`reverse` default, `forward`, or `mixed`) and `--opd-mixed-kl-weight`
    select the KL direction via `_topk_kl_terms`; `--opd-topk-zero-outside` (Task 4 wires
    the arg) controls the reverse-direction out-of-support correction -- until then this
    resolves `None` to `kl_type != "forward"` (correct the reverse direction's blind spot
    by default; inert for forward either way).

    Returns `(loss, metrics)`. In addition to "loss", `metrics` carries diagnostics that
    do not affect the loss itself (Task 4 wires the args gating whether these get
    logged): "opd_topk/teacher_mass" (+"_min") -- how much of the teacher's own
    distribution its reported top-k actually covers, "opd_topk/student_mass" -- how much
    of the *student's* distribution currently sits on the teacher's top-k ids, and
    "opd_topk/overlap_ratio" -- the fraction of the student's own local top-k ids that
    coincide with the teacher's. Every reduction here is a masked sum over a
    clamped->=1 denominator (mirroring `sum_of_sample_mean`'s own convention), never a
    bare `.mean()` over a selection that can be empty.

    Args:
        args: Configuration; uses `opd_kl_type`, `opd_mixed_kl_weight`, and (once Task 4
            lands) `opd_topk_zero_outside`.
        batch: Mini-batch with "teacher_topk_ids" (list of `[R, K]` token ids per sample),
            "teacher_topk_logprobs" (list of `[R, K]` teacher log-probs per sample),
            "unconcat_tokens", "total_lengths", "response_lengths", "loss_masks".
        logits: Policy logits with shape `[1, T, V]`, from the current (grad-enabled)
            forward pass.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)`.
    """
    parallel_state = get_parallel_state()
    device = logits.device
    teacher_topk_ids = [t.to(device=device) for t in batch["teacher_topk_ids"]]
    teacher_topk_logprobs = [t.to(device=device) for t in batch["teacher_topk_logprobs"]]
    # The real transport (get_rollout_data's torch.tensor(...) over the raw per-sample
    # list[list[int]] payload) collapses an empty response's row list (`[]`, not
    # `[[], ...]`) to a 1-D `[0]` tensor rather than `[0, K]`. Normalize to 2-D here --
    # R=0 either way, K is unknowable from an empty sample and irrelevant since there
    # are no rows -- so the per-sample `.sum(dim=-1)` diagnostics below reduce the K
    # axis, not the (already-empty) R axis, and concatenate cleanly with real samples'
    # `[R]`-shaped output instead of collapsing to a 0-d scalar.
    teacher_topk_ids = [t if t.dim() > 1 else t.reshape(0, 0) for t in teacher_topk_ids]
    teacher_topk_logprobs = [t if t.dim() > 1 else t.reshape(0, 0) for t in teacher_topk_logprobs]

    # For the overlap_ratio diagnostic's *student* top-k: dev's opd_jsd pattern (mirrors
    # `vocab_parallel_topk_indices`'s two call sites in opd_jsd_loss_function above) --
    # the student's own local shard is the authority on the vocab split, and vocab_start
    # is only meaningful once TP is actually on.
    tp_group = parallel_state.tp.group if parallel_state.tp.size > 1 else None
    local_vocab_size = logits.size(-1)
    vocab_start = vocab_shard_start(local_vocab_size) if tp_group is not None else 0

    # A bigger-config-vocab teacher (e.g. Qwen2.5-7B pads to 152064 vs a <3B student's
    # 151936) can report top-k ids past the student's own vocabulary. Left alone these
    # break compute_vocab_parallel_topk_log_probs's gather: at TP=1 they index-error; at
    # TP>1 every rank's ownership mask is False for them, so the gather silently returns
    # a fake `0 - log_normalizer` log-prob instead. Mask them to a pad slot before the
    # gather -- id -> 0, logprob -> -1e4 -- exactly like the transport's own padding
    # (orbit.rollout.opd_sglang._TOPK_PAD_TOKEN_ID/_TOPK_PAD_LOGPROB): the -1e4 underflows
    # to exact 0 mass under _topk_kl_terms's `valid` mask.
    global_student_vocab = local_vocab_size * parallel_state.tp.size
    for i, (t_ids, t_lp) in enumerate(zip(teacher_topk_ids, teacher_topk_logprobs, strict=True)):
        overhang = t_ids >= global_student_vocab
        teacher_topk_ids[i] = torch.where(overhang, torch.zeros_like(t_ids), t_ids)
        teacher_topk_logprobs[i] = torch.where(overhang, torch.full_like(t_lp, -1e4), t_lp)

    kl_type, mixed_weight = _resolve_opd_topk_kl_type(args)
    zero_outside = getattr(args, "opd_topk_zero_outside", None)
    if zero_outside is None:
        # Task 4 moves this default into arg validation; until then, correct the reverse
        # direction's out-of-support blind spot by default (inert for forward either way).
        zero_outside = kl_type != "forward"
    needs_correction = zero_outside and kl_type != "forward"

    total_lengths = batch["total_lengths"]
    response_lengths = batch["response_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=needs_correction,
        max_seq_lens=max_seq_lens,
        teacher_topk_ids=teacher_topk_ids,
    )
    student_topk_log_probs = log_probs_and_entropy["student_topk_log_probs"]
    entropy_per_sample = log_probs_and_entropy["entropy"] if needs_correction else [None] * len(teacher_topk_ids)

    responses = get_responses(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    )

    topk_kl_per_sample = []
    teacher_mass_per_sample = []
    student_mass_per_sample = []
    overlap_ratio_per_sample = []
    for (logits_chunk, _), t_ids, t_lp, s_lp, entropy_i in zip(
        responses, teacher_topk_ids, teacher_topk_logprobs, student_topk_log_probs, entropy_per_sample, strict=True
    ):
        topk_kl_per_sample.append(_topk_kl_terms(t_lp, s_lp, entropy_i, kl_type, mixed_weight, zero_outside))

        # Diagnostics only -- detached, no gradient needed.
        valid = t_lp.exp() > 0
        masked_teacher_weight = torch.where(valid, t_lp.exp(), torch.zeros_like(t_lp))
        masked_student_weight = torch.where(valid, s_lp.exp().detach(), torch.zeros_like(s_lp))
        teacher_mass_per_sample.append(masked_teacher_weight.sum(dim=-1))
        student_mass_per_sample.append(masked_student_weight.sum(dim=-1))

        k = t_ids.size(-1)
        # vocab_parallel_topk_indices returns *global* ids (shard-local candidates offset
        # by vocab_start, then all-gathered/re-ranked across TP -- see its docstring):
        # a plain local torch.topk on logits_chunk would instead be shard-local ids in
        # [0, V_local), only coincidentally comparable to the teacher's global ids at
        # tp.size == 1. Raw logits (not log-probs) are fine here: log_softmax only
        # shifts each row by a per-row constant, so it never changes the top-k ordering,
        # and this is diagnostic-only (no_grad inside the helper).
        student_topk_ids = vocab_parallel_topk_indices(logits_chunk, k, vocab_start, tp_group)
        teacher_ids_for_match = torch.where(valid, t_ids, torch.full_like(t_ids, -1))
        overlap_match = student_topk_ids.unsqueeze(-1) == teacher_ids_for_match.unsqueeze(-2)
        overlap_ratio_per_sample.append(overlap_match.any(dim=-1).sum(dim=-1).float() / max(k, 1))

    topk_kl = torch.cat(topk_kl_per_sample, dim=0)
    loss = sum_of_sample_mean(topk_kl)

    # make sure the gradient could backprop correctly.
    if topk_kl.numel() == 0:
        loss = loss + 0 * logits.sum()

    teacher_mass = torch.cat(teacher_mass_per_sample, dim=0)
    student_mass = torch.cat(student_mass_per_sample, dim=0)
    overlap_ratio = torch.cat(overlap_ratio_per_sample, dim=0)

    teacher_mass_min = _response_masked_min(
        teacher_mass,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        loss_masks=batch["loss_masks"],
        qkv_format=getattr(args, "qkv_format", "thd"),
        max_seq_lens=max_seq_lens,
    )

    metrics = {
        "loss": loss.clone().detach(),
        "opd_topk/teacher_mass": sum_of_sample_mean(teacher_mass).clone().detach(),
        "opd_topk/teacher_mass_min": teacher_mass_min.clone().detach(),
        "opd_topk/student_mass": sum_of_sample_mean(student_mass).clone().detach(),
        "opd_topk/overlap_ratio": sum_of_sample_mean(overlap_ratio).clone().detach(),
    }

    return loss, metrics


def loss_function(
    args: Namespace,
    batch: RolloutBatch,
    num_microbatches: int,
    logits: torch.Tensor,
    apply_megatron_loss_scaling: bool = False,
) -> tuple[torch.Tensor, int | torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    """Dispatch to the configured loss and rescale for Megatron integration.

    Selects one of "policy_loss", "value_loss", "sft_loss", "opd_jsd_loss", "opd_topk_loss",
    or a custom loss function based on `args.loss_type`, computes the loss and metrics, then
    rescales the loss by micro-batch and parallelism factors to integrate with
    Megatron's gradient accumulation.

    Args:
        args: Configuration specifying `loss_type`, `calculate_per_token_loss`,
            `global_batch_size`, and optionally `custom_loss_function_path`.
        batch: Mini-batch with "loss_masks", "response_lengths", and other
            keys required by the selected loss function.
        num_microbatches: Number of gradient accumulation steps.
        logits: Model outputs (policy or value head).

    Returns:
        Tuple of `(scaled_loss, normalizer, logging_dict)` where:
        - `scaled_loss` is the loss tensor (scalar) rescaled for Megatron.
        - `normalizer` is `num_tokens` (scalar tensor) if
          `args.calculate_per_token_loss` is True, else `1` (int).
        - `logging_dict` has keys "keys" (list of str metric names) and
          "values" (1D tensor: [count, metric1, metric2, ...]).
    """
    parallel_state = get_parallel_state()
    num_tokens = sum([torch.clamp_min(loss_mask.sum(), 1) for loss_mask in batch["loss_masks"]])
    num_samples = len(batch["response_lengths"])

    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens", None),
    )

    match args.loss_type:
        case "policy_loss":
            func = policy_loss_function
        case "value_loss":
            func = value_loss_function
        case "sft_loss":
            func = sft_loss_function
        case "opd_jsd_loss":
            func = opd_jsd_loss_function
        case "opd_topk_loss":
            func = opd_topk_loss_function
        case "custom_loss":
            func = load_function(args.custom_loss_function_path)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")

    if args.recompute_loss_function:
        loss, log = checkpoint(
            func,
            args,
            batch,
            logits,
            sum_of_sample_mean,
        )
    else:
        loss, log = func(args, batch, logits, sum_of_sample_mean)

    # With allgather-CP, some CP ranks may have no loss-contributing tokens (e.g., all
    # padding). Without this, gradient doesn't flow through their attention path, so
    # the CP gather's backward (reduce-scatter) is not called, deadlocking other CP
    # ranks that call it. Adding this zero loss forces autograd to traverse the full
    # graph on every rank without changing gradient values.
    if parallel_state.cp.size > 1 and args.allgather_cp:
        loss = loss + 0 * logits.sum()

    # Here we need to divide by cp_size because to cancel the multiply in Megatron.
    assert args.use_dynamic_global_batch_size == ("dynamic_global_batch_size" in batch)
    global_batch_size = batch.get("dynamic_global_batch_size", args.global_batch_size)
    if not args.calculate_per_token_loss:
        if apply_megatron_loss_scaling:
            loss = loss * num_microbatches / global_batch_size * parallel_state.intra_dp_cp.size
        else:
            loss = loss / global_batch_size * parallel_state.intra_dp.size
    else:
        if apply_megatron_loss_scaling:
            loss = loss * parallel_state.cp.size

    return (
        loss,
        torch.tensor(num_tokens if args.calculate_per_token_loss else 1, device=logits.device),
        {
            "keys": list(log.keys()),
            "values": torch.tensor(
                [
                    num_samples if not args.calculate_per_token_loss else num_tokens,
                ]
                + list(log.values()),
                device=logits.device,
            ),
        },
    )
