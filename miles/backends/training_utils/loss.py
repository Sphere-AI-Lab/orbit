from argparse import Namespace
from collections.abc import Callable, Iterator
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from miles.utils.distributed_utils import distributed_masked_whiten
from miles.utils.misc import load_function
# ORBIT-SEAM: orbit-added ppo_utils helpers - critic explained-variance stat keys, the
# true-on-policy full-logits gather, the overflow-safe ratio helpers, the OPD advantage
# shaping and the icepop gate. _gather_true_on_policy_full_logits is imported here purely
# as a re-export: orbit.opd.losses reads it back off this module at call time.
from miles.utils.ppo_utils import (  # noqa: F401
    VALUE_EV_STAT_KEYS,
    _gather_true_on_policy_full_logits,
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
from miles.utils.types import RolloutBatch

from .cp_utils import (
    _allgather_cp_redistribute,
    all_gather_with_cp,
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
)
from .parallel import get_parallel_state
# ORBIT-SEAM: orbit's OPD losses (full-vocab JSD, direct top-k KL) and the masked
# response-reduction helpers added with them live in orbit.opd.losses; bound here so the
# loss_type arms below, policy_loss_function's max diagnostic and existing importers that
# rebind these on this module (tests/fast/test_opd_topk_loss.py, test_unbiased_kl_numerics.py)
# keep resolving them exactly as they did before the move
from orbit.opd.losses import (  # noqa: F401
    _TOPK_LOG_INF,
    _response_masked_max,
    _response_masked_min,
    _topk_kl_terms,
    _topk_overlap_membership,
    opd_jsd_loss_function,
    opd_topk_loss_function,
    opd_topk_sample_log_probs,
)


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

    # ORBIT-SEAM: base asserts fp32 unconditionally; true-on-policy mode deliberately feeds
    # the model's native bf16/fp16 logits (parity with SGLang's log_softmax dtype)
    if not args.true_on_policy_mode:
        assert logits.dtype == torch.float32, f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1))

    # ORBIT-SEAM: base does an unconditional `logits.div(args.rollout_temperature)`; orbit
    # guards the allocation and adds the true-on-policy dtype cast (rationale inline below)
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
    # ORBIT-SEAM: three orbit-added keyword-only options on the base signature (and their
    # docstring entries below) - entropy_no_grad skips the entropy graph when entropy_coef
    # is 0; teacher_topk_ids/with_log_probs turn on the OPD top-k branch in the loop below
    entropy_no_grad: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
    teacher_topk_ids: list[torch.Tensor] | None = None,
    with_log_probs: bool = True,
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
        with_log_probs: Compute sampled-token log-probs. The direct top-k OPD
            loss disables this because it only consumes the supplied-id scores.

    Returns:
        Dict with key "log_probs" mapping to a list of `[R]` tensors per
        sample. If `with_entropy` is True, also includes "entropy" key with
        a list of `[R]` tensors. If `teacher_topk_ids` is given, also
        includes "student_topk_log_probs" mapping to a list of `[R, K]`
        tensors.
    """
    parallel_state = get_parallel_state()
    assert non_loss_data

    # ORBIT-SEAM: the OPD top-k branch below cannot ride the CP redistribution helper, so
    # reject the unsupported combination before any work is done
    if teacher_topk_ids is not None and args.allgather_cp:
        raise NotImplementedError(
            "on_policy_distillation opd_loss_type='topk' does not support --allgather-cp: "
            "the CP redistribution helper only handles 1D per-token tensors, not the "
            "[R, K] student_topk_log_probs tensor."
        )

    # ORBIT-SEAM: TP group for the OPD top-k branch, hoisted out of the loop so the home
    # helper is handed the same object on every sample
    # dev's opd_jsd pattern: only pay for the TP collective path when TP is actually
    # on, rather than czy's unconditional parallel_state.tp.group.
    tp_group = parallel_state.tp.group if parallel_state.tp.size > 1 else None

    log_probs_list = []
    entropy_list = []
    # ORBIT-SEAM: base iterates get_responses() directly; orbit zips it (strict) against the
    # per-sample teacher top-k ids so the OPD branch can score supplied ids alongside the
    # sampled token, and collects those scores under "student_topk_log_probs" below
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
        if sample_topk_ids is not None:
            # ORBIT-SEAM: orbit's OPD top-k scoring for this sample (vocab-parallel or
            # true-on-policy full-vocab gather) lives in orbit.opd.losses
            topk_log_prob, log_prob, entropy = opd_topk_sample_log_probs(
                logits_chunk,
                tokens_chunk,
                sample_topk_ids,
                args=args,
                parallel_state=parallel_state,
                tp_group=tp_group,
                with_entropy=with_entropy,
                entropy_no_grad=entropy_no_grad,
                with_log_probs=with_log_probs,
            )
            topk_log_probs_list.append(topk_log_prob)
        else:
            # ORBIT-SEAM: base's path, plus the guard for the OPD-only with_log_probs=False
            # combination and the orbit-added entropy_no_grad / vocab_size kwargs
            if not with_log_probs:
                raise ValueError("with_log_probs=False requires teacher_topk_ids.")
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

        # ORBIT-SEAM: base always appends log_prob.squeeze(-1); orbit skips the append when
        # the OPD top-k loss asked for scores only, and reshapes instead of squeezing
        if with_log_probs:
            # Standard Megatron CE returns [R, 1], whereas the true-on-policy
            # full-vocab path returns [R]. Preserve the public per-token [R]
            # shape even for one-token responses in both cases.
            log_probs_list.append(log_prob.reshape(-1))
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


# ORBIT-SEAM: orbit adds the `role` argument (defaulted, so base callers are unaffected) so
# the critic can skip the actor-only OPD advantage adjustments below; documented in the
# docstring's Args block
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
        # ORBIT-SEAM: repo-wide comment-style pass (TODO -> Follow-up), no functional change
        # Follow-up: is the copy necessary?
        advantages = [r for r in returns]

    elif args.advantage_estimator == "ppo":
        # ORBIT-SEAM: base folds the terminal reward onto each sample's last local token on
        # cp_rank 0 before calling GAE; orbit passes the terminal rewards through instead, so
        # get_advantages_and_returns_batch can place them on the last *trainable* token under
        # CP (hence the extra qkv_format/max_seq_lens/loss_masks arguments it now takes)
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
        # ORBIT-SEAM: base computes the teacher-minus-student advantage inline; orbit's
        # version (device move, CP-aware response slicing, MOPD variants) lives in
        # miles.utils.ppo_utils.opd_mopd_advantages
        advantages = opd_mopd_advantages(rollout_data, log_probs, rollout_data.get("response_lengths"))
        returns = advantages

    else:
        raise NotImplementedError(f"advantage_estimator {args.advantage_estimator} is not supported. ")

    # ORBIT-SEAM: orbit-only post-estimator advantage shaping, actor-side only - blend the OPD
    # teacher KL into whichever advantage the estimator above produced, then optionally
    # hard-gate off-policy tokens. Both transforms live in miles.utils.ppo_utils.
    if role == "actor" and getattr(args, "use_opd", False):
        apply_opd_kl_to_advantages(args.opd_kl_coef, rollout_data, advantages, log_probs)

    # Optional async/off-policy ICE-POP correction for the OPD advantage (pure-MOPD
    # or blend): hard-gate tokens whose train/rollout importance ratio leaves the band.
    if role == "actor" and getattr(args, "opd_icepop", False):
        apply_opd_icepop_gate(rollout_data, advantages, args.tis_clip_low, args.tis_clip)

    # ORBIT-SEAM: repo-wide comment-style pass (TODO -> Follow-up), no functional change
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

        # ORBIT-SEAM: base skips whitening entirely when this rank's mask is empty and whitens
        # over parallel_state.intra_dp.group; orbit always enters the collective and uses the
        # combined intra-DP+CP group (rationale inline below) - a genuine behavioural change
        # to base, kept in place because it is one call, not a liftable block
        assert (
            all_advs.size() == all_masks.size()
        ), f"Shape mismatch before whitening: advantages {all_advs.size()}, masks {all_masks.size()}"

        # CP ranks own disjoint response-token slices, so whitening over the
        # DP-only group would normalize each CP shard independently.  Use the
        # combined DP+CP group and have empty local shards enter the collective
        # as well; otherwise an uneven response layout can strand its peers.
        whitened_advs_flat = distributed_masked_whiten(
            all_advs,
            all_masks,
            process_group=parallel_state.intra_dp_cp.group,
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
    # ORBIT-SEAM: base inlines the band torch.where; orbit shares the gate with the OPD
    # advantage path via miles.utils.ppo_utils.icepop_gate (same expression)
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
        # ORBIT-SEAM: entropy is a metric only when entropy_coef is 0, so drop its graph
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
        # ORBIT-SEAM: base always uses `old_log_probs - log_probs`; --force-on-policy-ratio is
        # orbit's pure-MOPD switch (rationale inline below)
        if getattr(args, "force_on_policy_ratio", False):
            # Ratio pinned to exactly 1.0 with the gradient preserved: the surrogate
            # degenerates to REINFORCE, the exact objective of pure sampled-token MOPD.
            # Independent behaviour correction may still be applied with TIS.
            ppo_kl = log_probs.detach() - log_probs
        else:
            ppo_kl = old_log_probs - log_probs

    # ORBIT-SEAM: orbit passes the dual-clip bound `eps_clip_c` through to compute_policy_loss
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

        # ORBIT-SEAM: base computes `(-ppo_kl).exp()`; orbit routes it through the shared
        # overflow-safe clamp (identical in range, no inf/NaN under async drift)
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
            # ORBIT-SEAM: base uses `torch.exp(log_probs - old_log_probs)`; orbit clamps the
            # exponent and switches the denominator to the rollout behavior policy under TIS
            # Route the exponent through the same safe clamp as every other
            # ratio path: async/off-policy drift can push the log-ratio past exp
            # overflow. TIS bridges the trainer snapshot to the rollout behavior
            # policy for the PG term; the sampled KL needs that behavior policy as
            # its denominator directly to remain unbiased.
            behavior_log_probs = old_log_probs
            if args.use_tis:
                behavior_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
            importance_ratio = _safe_clamp_log_ratio(log_probs - behavior_log_probs).exp()
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
    # ORBIT-SEAM: base reports only the mean train-vs-rollout log-prob gap; orbit adds the
    # worst-token gap through orbit.opd.losses._response_masked_max (rebindable on this
    # module, which tests/fast/test_unbiased_kl_numerics.py and tests/test_opd_advantage.py do)
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
        # ORBIT-SEAM: companion metric for the max diagnostic computed above
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

    # ORBIT-SEAM: orbit-added critic explained-variance sufficient statistics. Left in place
    # rather than lifted: the block is five calls to this module's own sum_of_sample_mean
    # reducer, and its natural home (orbit/critic/value_stats.py, which owns
    # VALUE_EV_STAT_KEYS and compute_value_explained_var) belongs to the ppo_utils slice.
    # Sufficient statistics for the critic explained-variance metric,
    # EV = 1 - Var(returns - values) / Var(returns) over trainable tokens.
    # Averaging per-micro-batch EV would be biased when micro-batches differ in
    # token count or mean, so emit masked token-level sums instead: the metric
    # pipeline (aggregate_train_losses) SUM-reduces every non-extrema metric
    # across micro-batches and DP/CP ranks and divides by one count shared by
    # all keys, which cancels in the ratios taken by compute_value_explained_var
    # at aggregation time. The token-sum reducer below is the CP-aware masked
    # sum (`calculate_per_token_loss=True` selects sum-of-token semantics)
    # regardless of the reduction mode used for the loss itself.
    sum_of_token = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        calculate_per_token_loss=True,
        qkv_format=args.qkv_format,
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    detached_returns = returns.detach().float()
    detached_err = detached_returns - values.detach().float()

    ev_stats = dict(
        zip(
            VALUE_EV_STAT_KEYS,
            (
                sum_of_token(torch.ones_like(detached_returns)),
                sum_of_token(detached_returns),
                sum_of_token(detached_returns**2),
                sum_of_token(detached_err),
                sum_of_token(detached_err**2),
            ),
            strict=True,
        )
    )

    reported_loss = {
        "value_loss": loss.clone().detach(),
        "value_clipfrac": values_clipfrac.clone().detach(),
        # ORBIT-SEAM: the explained-variance statistics computed above ride the metric dict
        **ev_stats,
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


# ORBIT-SEAM: base's dispatcher, unchanged in shape; the docstring below and the match arms
# further down name the two orbit loss types (opd_jsd_loss, opd_topk_loss) added to it
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
    # ORBIT-SEAM: base clamps each sample's token count to a minimum of 1; orbit keeps the
    # count exact (rationale inline below)
    # Megatron sums this normalizer across micro-batches and DP/CP ranks before
    # scaling gradients, and already leaves gradients unscaled when that global
    # count is zero. Keep the local count exact: a rejected/all-masked sample has
    # a zero loss numerator and must not add a phantom token to the denominator.
    num_tokens = sum(loss_mask.sum() for loss_mask in batch["loss_masks"])
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
        # ORBIT-SEAM: two orbit loss types added to base's match; both names are bound at the
        # top of this module to the home implementations in orbit.opd.losses. Deliberately not
        # routed through --custom-loss-function-path: `--loss-type opd_jsd_loss` /
        # `opd_topk_loss` is the CLI contract orbit's recipes and tests already use.
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

    # ORBIT-SEAM: base builds the normalizer with torch.tensor(num_tokens, ...), which would
    # now drag the (unclamped, still-attached) token-count graph into Megatron; detach and
    # move the existing tensor instead
    normalizer = (
        num_tokens.detach().to(device=logits.device)
        if args.calculate_per_token_loss
        else torch.tensor(1, device=logits.device)
    )
    return (
        loss,
        normalizer,
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
