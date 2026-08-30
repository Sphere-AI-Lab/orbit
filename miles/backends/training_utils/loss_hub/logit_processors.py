from argparse import Namespace
from collections.abc import Iterator, Sequence

import torch

from miles.backends.training_utils.cp_utils import allgather_cp_redistribute, get_logits_and_tokens_offset_with_cp
from miles.backends.training_utils.loss_hub.math_utils import calculate_log_probs_and_entropy
from miles.backends.training_utils.parallel import get_parallel_state
from miles.backends.training_utils.sampling_mask import build_local_sampling_mask
from miles.utils.sampling_mask import RolloutSamplingMask

# ORBIT-SEAM: orbit's OPD top-k student scoring (vocab-parallel or true-on-policy full-vocab
# gather) lives in orbit.opd.losses; imported here for the `teacher_topk_ids` branch of
# get_log_probs_and_entropy below, which is orbit's only addition to base's loop.
from orbit.opd.losses import opd_topk_sample_log_probs


def _iter_response_chunks(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
    include_response_indices: bool,
) -> Iterator[tuple[torch.Tensor, torch.Tensor, Sequence[int]]]:
    """Yield response logits, tokens, and original response indices per sample.

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
        Tuple of `(logits_chunk, tokens_chunk, response_indices)`, where
        `logits_chunk` is shape `[R, V]` (policy) or `[R, 1]` (value), and
        `tokens_chunk` is shape `[R]` (1D int64). `response_indices` maps every
        local row back to the full response. The mapping is empty when
        `include_response_indices` is false.
    """
    qkv_format = args.qkv_format

    if not args.true_on_policy_mode:
        # FSDP hands native bf16 here (no full-vocab fp32 buffer); chunks are upcast to fp32 downstream
        assert logits.dtype in (torch.float32, torch.bfloat16), f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1))

    if logits.size(-1) > 1 and args.rollout_temperature > 0 and args.rollout_temperature != 1.0:
        logits = logits.div(args.rollout_temperature)
    if args.true_on_policy_mode:
        if getattr(args, "bf16", False):
            logits = logits.to(torch.bfloat16)
        elif getattr(args, "fp16", False):
            logits = logits.to(torch.float16)

    parallel_state = get_parallel_state()
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
                logits_chunk = logits[start - 1 : end - 1]
            else:
                end += total_length
                start = end - response_length
                logits_chunk = logits[start - 1 : end - 1]
            tokens_chunk = tokens[-response_length:] if response_length else tokens[0:0]
            response_indices = range(response_length) if include_response_indices else ()
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
                response_indices = ()
            else:
                logits_chunk = logits[s - chunk_start : e - chunk_start]
                tokens_chunk = tokens[(s + 1) - seq_start : (e + 1) - seq_start]
                response_indices = (
                    range(
                        s - logit_global_start,
                        e - logit_global_start,
                    )
                    if include_response_indices
                    else ()
                )
            assert logits_chunk.size(0) == tokens_chunk.size(0), f"{logits_chunk.size(0)} vs {tokens_chunk.size(0)}"
        else:
            # ORBIT-SEAM: repo-wide comment-style pass (base's "TODO: this is super ugly..."
            # replaced by a description), no functional change
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
            if include_response_indices:
                prompt_length = total_length - response_length
                response_indices = [
                    *range(
                        tokens_offset[0][0] - prompt_length,
                        tokens_offset[0][1] - prompt_length,
                    ),
                    *range(
                        tokens_offset[1][0] - prompt_length,
                        tokens_offset[1][1] - prompt_length,
                    ),
                ]
            else:
                response_indices = ()

        seq_start += total_length

        if include_response_indices:
            assert len(response_indices) == tokens_chunk.size(0)
        yield logits_chunk, tokens_chunk, response_indices


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned `(logits_chunk, tokens_chunk)` pairs per sample."""
    for logits_chunk, tokens_chunk, _ in _iter_response_chunks(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        include_response_indices=False,
    ):
        yield logits_chunk, tokens_chunk


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    entropy_requires_grad: bool = True,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
    rollout_sampling_mask: Sequence[RolloutSamplingMask] | None = None,
    # ORBIT-SEAM: two orbit-added keyword-only options on the base signature (and their
    # docstring entries below) - teacher_topk_ids/with_log_probs turn on the OPD top-k branch
    # in the loop below
    teacher_topk_ids: list[torch.Tensor] | None = None,
    with_log_probs: bool = True,
) -> dict[str, list[torch.Tensor]]:
    """Compute per-token log-probabilities (and optionally entropy) on responses.

    For each sample, extracts response-aligned logits and tokens, then computes
    log-probabilities via softmax across the tensor-parallel group. Log-probs
    are squeezed from `[R, 1]` to `[R]`. Entropy is computed and returned only
    when requested.

    Args:
        logits: Policy logits with shape `[1, T, V]`.
        args: Configuration (temperature applied in `get_responses`).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: If True, include "entropy" key in result.
        entropy_requires_grad: If False, compute entropy as an observed metric
            without attaching it to the autograd graph.
        non_loss_data: Unused; kept for API compatibility.
        rollout_sampling_mask: One ``RolloutSamplingMask`` per sample,
            covering every response token.
        teacher_topk_ids: For on_policy_distillation's "topk" mode, a list of
            `[R, K]` token-id tensors (one per sample, from TeacherManager) to
            additionally gather student log-probs for at each response
            position. When None (the default), no extra gather is done.
        with_log_probs: Compute sampled-token log-probs. The direct top-k OPD
            loss disables this because it only consumes the supplied-id scores.

    Returns:
        Dict with key "log_probs" mapping to a list of `[R]` tensors per
        sample. If `with_entropy` is True, also includes "entropy" key with
        a list of `[R]` tensors. If `teacher_topk_ids` is given, also
        includes "student_topk_log_probs" mapping to a list of `[R, K]`
        tensors.
    """
    assert non_loss_data
    # ORBIT-SEAM: the OPD top-k branch below cannot ride the CP redistribution helper, nor the
    # sampling-mask path, so reject the unsupported combinations before any work is done
    if teacher_topk_ids is not None:
        if args.allgather_cp:
            raise NotImplementedError(
                "on_policy_distillation opd_loss_type='topk' does not support --allgather-cp: "
                "the CP redistribution helper only handles 1D per-token tensors, not the "
                "[R, K] student_topk_log_probs tensor."
            )
        if rollout_sampling_mask is not None:
            raise NotImplementedError(
                "on_policy_distillation opd_loss_type='topk' does not support a rollout sampling "
                "mask: the top-k scorer gathers supplied ids, not a masked vocabulary slice."
            )
    if rollout_sampling_mask is not None:
        for sample_index, (sampling_mask, response_length) in enumerate(
            zip(rollout_sampling_mask, response_lengths, strict=True)
        ):
            if len(sampling_mask) != response_length:
                raise ValueError(
                    f"sampling-mask length {len(sampling_mask)} != response length "
                    f"{response_length} for sample {sample_index}"
                )
    parallel_state = get_parallel_state()
    # ORBIT-SEAM: TP group for the OPD top-k branch, hoisted out of the loop so the home
    # helper is handed the same object on every sample. Resolved only on that branch: base
    # callers must not have to supply `tp.size`.
    # dev's opd_jsd pattern: only pay for the TP collective path when TP is actually
    # on, rather than czy's unconditional parallel_state.tp.group.
    topk_tp_group = None
    if teacher_topk_ids is not None:
        topk_tp_group = parallel_state.tp.group if parallel_state.tp.size > 1 else None
    log_probs_list = []
    entropy_list = []
    # ORBIT-SEAM: base iterates the response chunks directly; orbit zips them (strict) against
    # the per-sample teacher top-k ids so the OPD branch can score supplied ids alongside the
    # sampled token, and collects those scores under "student_topk_log_probs" below
    topk_log_probs_list = [] if teacher_topk_ids is not None else None
    topk_ids_iter = teacher_topk_ids if teacher_topk_ids is not None else [None] * len(unconcat_tokens)
    response_chunks = _iter_response_chunks(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
        include_response_indices=rollout_sampling_mask is not None,
    )
    for sample_index, ((logits_chunk, tokens_chunk, response_indices), sample_topk_ids) in enumerate(
        zip(response_chunks, topk_ids_iter, strict=True)
    ):
        if sample_topk_ids is not None:
            # ORBIT-SEAM: orbit's OPD top-k scoring for this sample lives in orbit.opd.losses
            topk_log_prob, log_prob, entropy = opd_topk_sample_log_probs(
                logits_chunk,
                tokens_chunk,
                sample_topk_ids,
                args=args,
                parallel_state=parallel_state,
                tp_group=topk_tp_group,
                with_entropy=with_entropy,
                # orbit's home helper keeps the original entropy_no_grad sense; upstream
                # renamed this module's public kwarg to entropy_requires_grad (inverted)
                entropy_no_grad=not entropy_requires_grad,
                with_log_probs=with_log_probs,
            )
            topk_log_probs_list.append(topk_log_prob)
        else:
            # ORBIT-SEAM: base's path, plus the guard for the OPD-only with_log_probs=False
            # combination
            if not with_log_probs:
                raise ValueError("with_log_probs=False requires teacher_topk_ids.")
            sampling_mask = None
            if rollout_sampling_mask is not None:
                sampling_mask = build_local_sampling_mask(
                    logits_chunk,
                    rollout_sampling_mask[sample_index],
                    response_indices,
                    tp_rank=parallel_state.tp.rank,
                )
            log_prob, entropy = calculate_log_probs_and_entropy(
                logits_chunk,
                tokens_chunk,
                parallel_state.tp.group,
                with_entropy=with_entropy,
                entropy_requires_grad=entropy_requires_grad,
                chunk_size=args.log_probs_chunk_size,
                true_on_policy=args.true_on_policy_mode,
                vocab_size=getattr(args, "vocab_size", None),
                sampling_mask=sampling_mask,
            )

        # ORBIT-SEAM: base always appends log_prob.squeeze(-1); orbit skips the append when the
        # OPD top-k loss asked for scores only, and reshapes instead of squeezing.
        # Standard Megatron CE returns [R, 1], whereas the true-on-policy full-vocab path
        # returns [R]. Preserve the public per-token [R] shape even for one-token responses
        # in both cases.
        if with_log_probs:
            log_probs_list.append(log_prob.reshape(-1))
        if with_entropy:
            entropy_list.append(entropy)

    res = {
        "log_probs": log_probs_list,
    }
    if with_entropy:
        res["entropy"] = entropy_list
    # ORBIT-SEAM: extra result key consumed by orbit.opd.losses.opd_topk_loss_function
    if topk_log_probs_list is not None:
        res["student_topk_log_probs"] = topk_log_probs_list

    # we need to turn the all gather kv into zigzag ring attn kv
    if args.allgather_cp:
        allgather_cp_redistribute(
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
        # upcast (no-op for fp32) so value-head outputs stay fp32 even when logits arrive bf16
        value_list.append(logits_chunk.squeeze(-1).float())

    res = {
        "values": value_list,
    }

    if args.allgather_cp:
        allgather_cp_redistribute(
            res,
            logits=logits,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return res
