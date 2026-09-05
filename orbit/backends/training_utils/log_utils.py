import logging
from argparse import Namespace
from math import isclose

import numpy as np
import psutil
import torch
import torch.distributed as dist

from orbit.utils import train_metric_utils
from orbit.utils.flops_utils import fwd_tflops_per_gpu
from orbit.utils.ft_utils.process_group_utils import MultiPGUtil
from orbit.utils.metric_utils import compute_rollout_step
from orbit.utils.tracking_utils.structured_log import log_structured
from orbit.utils.types import RolloutBatch

from ...utils.tracking_utils import tracking
from .cp_utils import get_local_response_loss_masks, get_sum_of_sample_mean
from .data import DataIterator
from .parallel import get_parallel_state

logger = logging.getLogger(__name__)

_MULTI_TURN_REDUCTION_BY_KEY = {
    "raw_response_length/response_length_max": "max",
    "raw_response_length/response_length_min": "min",
    "wo_obs_response_length/response_length_max": "max",
    "wo_obs_response_length/response_length_min": "min",
    "multi_turn_metric/round_number_max": "max",
    "multi_turn_metric/round_number_min": "min",
}


def reduce_gathered_log_dict(
    gathered: list[dict],
    dp_size: int,
    reduction_by_key: dict[str, str] | None = None,
) -> dict[str, float]:
    """Reduce already-gathered per-rank metrics without adding another collective.

    ``(sum, count)`` tuples reduce as ``Σsum / Σcount``. Scalars reduce by the
    reduction named in ``reduction_by_key`` ("mean", "min" or "max");
    unspecified keys reduce by mean. Metric names do not implicitly determine
    their reduction semantics. Rank-local extrema must be reduced as extrema:
    averaging per-rank maxima under-reports the global maximum (and
    over-reports the global minimum).
    """
    if not gathered:
        return {}

    expected_keys = gathered[0].keys()
    if reduction_by_key is not None:
        for rank, rank_metrics in enumerate(gathered[1:], start=1):
            if rank_metrics.keys() != expected_keys:
                raise ValueError(
                    f"Metric keys differ across ranks: rank 0={list(expected_keys)}, "
                    f"rank {rank}={list(rank_metrics.keys())}."
                )
    reduction_by_key = reduction_by_key or {}

    reduced: dict[str, float] = {}
    for key in expected_keys:
        values = [d[key] for d in gathered]
        first = values[0]
        reduction = reduction_by_key.get(key, "mean")
        if reduction not in ("mean", "min", "max"):
            raise ValueError(f"Unsupported metric reduction {reduction!r} for {key!r}.")
        if isinstance(first, tuple) and len(first) == 2:
            total_sum = sum(v[0] for v in values)
            total_count = sum(v[1] for v in values)
            reduced[key] = total_sum / total_count if total_count else 0.0
        elif reduction == "mean":
            reduced[key] = sum(values) / dp_size
        elif reduction == "min":
            reduced[key] = min(values)
        else:
            reduced[key] = max(values)
    return reduced


def gather_log_data(
    metric_name: str,
    args: Namespace,
    rollout_id: int,
    log_dict: dict[str, "float | tuple[float, float]"],
    reduction_by_key: dict[str, str] | None = None,
) -> dict[str, float] | None:
    """
    Gather per-rank metrics, reduce on the DP source rank, and log.

    ``(sum, count)`` tuple values reduce as ``Σsum / Σcount``; scalar keys
    reduce by mean unless `reduction_by_key` explicitly selects "min" or
    "max" for them. Returns the reduced dict on the DP source rank; returns
    None on others.
    """

    parallel_state = get_parallel_state()

    pg = parallel_state.effective_dp_cp
    log_structured(logger.info, op="cross_cell", phase="start", kind="log_gather", rank=pg.rank)
    try:
        gathered_log_dict = MultiPGUtil.gather_object(
            obj=log_dict,
            groups_inner_to_outer=pg.gloo_groups_inner_to_outer,
        )
        log_structured(logger.info, op="cross_cell", phase="end", kind="log_gather", rank=pg.rank, success=True)
    except RuntimeError:
        log_structured(
            logger.warning,
            op="cross_cell",
            phase="end",
            kind="log_gather",
            rank=pg.rank,
            success=False,
            degraded=True,
            exc_info=True,
        )
        return None

    if pg.rank == 0:
        reduced = reduce_gathered_log_dict(gathered_log_dict, pg.size, reduction_by_key)
        reduced_log_dict = {f"{metric_name}/{key}": value for key, value in reduced.items()}
        logger.info(f"{metric_name} {rollout_id}: {reduced_log_dict}")

        # Calculate step once to avoid duplication
        step = compute_rollout_step(args, rollout_id)
        reduced_log_dict["rollout/step"] = step
        tracking.log(args, reduced_log_dict, step_key="rollout/step")

        return reduced_log_dict
    else:
        return None


def aggregate_forward_results(
    forward_data_store: list[dict[str, list]],
    data_iterator: DataIterator,
    args: Namespace,
    store_prefix: str = "",
) -> dict[str, list]:
    rollout_data = {}
    if not forward_data_store:
        return rollout_data

    keys = forward_data_store[0].keys()
    for key in keys:
        values = []
        for batch_result in forward_data_store:
            assert isinstance(batch_result[key], list), f"Expected list for key {key}, got {type(batch_result[key])}"
            values += batch_result[key]

        # Handle dynamic batch size: restore original order
        if args.use_dynamic_batch_size and hasattr(data_iterator, "micro_batch_indices"):
            origin_values = [None] * len(values)
            origin_indices = sum(data_iterator.micro_batch_indices, [])
            for value, origin_index in zip(values, origin_indices, strict=False):
                origin_values[origin_index] = value
            values = origin_values

        rollout_data[key] = values

    return rollout_data


def _get_rollout_kl_log_ratio_groups(
    args: Namespace,
    rollout_data: RolloutBatch,
) -> dict[str, list[torch.Tensor]]:
    """Return sampled log-ratios grouped by their two distributions.

    ``kl`` retains the existing policy/reference namespace used by ordinary RL
    and follows the same trainer-vs-rollout policy-logprob selection as advantage
    computation. ``opd_kl`` is reserved for student/teacher sampled OPD. Keeping
    these groups separate prevents two different targets from sharing one W&B
    time series.
    """
    groups: dict[str, list[torch.Tensor]] = {}
    policy_log_probs_key = "rollout_log_probs" if getattr(args, "use_rollout_logprobs", False) else "log_probs"
    policy_log_probs = rollout_data.get(policy_log_probs_key)
    ref_log_probs = rollout_data.get("ref_log_probs")
    if policy_log_probs and ref_log_probs:
        if len(policy_log_probs) != len(ref_log_probs):
            raise ValueError(
                f"Rollout KL batch mismatch: {policy_log_probs_key}={len(policy_log_probs)}, "
                f"ref_log_probs={len(ref_log_probs)}."
            )

        policy_ref_log_ratios = []
        for sample_index, (policy, reference) in enumerate(zip(policy_log_probs, ref_log_probs, strict=True)):
            policy = torch.as_tensor(policy)
            reference = torch.as_tensor(reference, device=policy.device)
            if policy.shape != reference.shape:
                raise ValueError(
                    f"Rollout KL shape mismatch at sample {sample_index}: "
                    f"{policy_log_probs_key}={tuple(policy.shape)}, reference={tuple(reference.shape)}."
                )
            policy_ref_log_ratios.append(policy - reference)
        groups["kl"] = policy_ref_log_ratios

    sampled_opd_log_ratios = rollout_data.get("opd_reverse_kl")
    if sampled_opd_log_ratios and int(getattr(args, "opd_log_prob_top_k", 0) or 0) == 0:
        groups["opd_kl"] = sampled_opd_log_ratios

    # ``rollout_train_kl`` is the inference/training mismatch: q = the rollout
    # engine that sampled the tokens, p = the trainer's recompute of the same
    # tokens. Unlike ``kl`` it needs no reference model, so it is the group
    # ordinary RL runs actually get. Direction matches the train-panel k3
    # diagnostics (KL(rollout || train)), so the two can cross-validate.
    #
    # PRECONDITION: p comes from the store_prefix="" log-prob forward, which
    # actor.py skips under --use-rollout-logprobs unless --get-mismatch-metrics
    # is also set. On such runs rollout_data["log_probs"] is absent and this
    # group is simply not emitted -- there is no trainer recompute to compare
    # against, so the metric would have nothing to mean.
    rollout_log_probs = rollout_data.get("rollout_log_probs")
    trainer_log_probs = rollout_data.get("log_probs")
    if rollout_log_probs and trainer_log_probs:
        if len(rollout_log_probs) != len(trainer_log_probs):
            raise ValueError(
                f"Rollout KL batch mismatch: rollout_log_probs={len(rollout_log_probs)}, "
                f"log_probs={len(trainer_log_probs)}."
            )
        mismatch_log_ratios = []
        for sample_index, (sampled, recomputed) in enumerate(zip(rollout_log_probs, trainer_log_probs, strict=True)):
            sampled = torch.as_tensor(sampled)
            recomputed = torch.as_tensor(recomputed, device=sampled.device)
            if sampled.shape != recomputed.shape:
                raise ValueError(
                    f"Rollout KL shape mismatch at sample {sample_index}: "
                    f"rollout_log_probs={tuple(sampled.shape)}, log_probs={tuple(recomputed.shape)}."
                )
            mismatch_log_ratios.append(sampled - recomputed)
        groups["rollout_train_kl"] = mismatch_log_ratios

    return groups


def _compute_rollout_kl_statistics(
    args: Namespace,
    rollout_data: RolloutBatch,
    cp_size: int,
    rollout_count_share: float | None = None,
) -> tuple[dict[str, float | tuple[float, float]], dict[str, str]]:
    """Compute sampled KL estimators over active response tokens.

    For d = log q(a|h) - log p(a|h), sampled under q:
      k1 = d
      k2 = 0.5 * d^2
      k3 = exp(-d) - 1 + d

    Policy/ref log-probs supply q=policy and p=reference under ``kl/*``.
    Sampled OPD supplies q=student and p=teacher under ``opd_kl/*``.
    ``rollout_train_kl/*`` is q=rollout engine and p=trainer recompute — the
    training/inference mismatch, available on ordinary RL runs without a
    reference model. Every available group is emitted. Legacy top-k detached
    scalars are excluded from the sampled OPD group.
    """
    log_ratio_groups = _get_rollout_kl_log_ratio_groups(args, rollout_data)
    if not log_ratio_groups:
        return {}, {}

    response_lengths = rollout_data["response_lengths"]
    total_lengths = rollout_data["total_lengths"]
    loss_masks = rollout_data["loss_masks"]
    max_seq_lens = rollout_data.get("max_seq_lens", None)
    local_masks = get_local_response_loss_masks(
        total_lengths,
        response_lengths,
        loss_masks,
        args.qkv_format,
        max_seq_lens,
    )
    rollout_denominators = rollout_data.get("rollout_mask_sums", None)
    if rollout_denominators is not None and rollout_count_share is None:
        raise ValueError(
            "Rollout-normalized KL metrics require rollout_count_share; "
            "compute it from the batch-global rollout IDs before DP splitting."
        )
    sample_mean = get_sum_of_sample_mean(
        total_lengths,
        response_lengths,
        loss_masks,
        qkv_format=args.qkv_format,
        max_seq_lens=max_seq_lens,
        denominators=rollout_denominators,
    )
    metrics: dict[str, float | tuple[float, float]] = {}
    reduction_by_key: dict[str, str] = {}
    rollout_count = rollout_count_share if rollout_count_share is not None else len(loss_masks)
    for metric_group, log_ratios in log_ratio_groups.items():
        if len(log_ratios) != len(local_masks):
            raise ValueError(
                f"Rollout KL metric batch mismatch for {metric_group}: "
                f"log_ratios={len(log_ratios)}, local_masks={len(local_masks)}."
            )

        estimator_chunks: dict[str, list[torch.Tensor]] = {"k1": [], "k2": [], "k3": []}
        bool_masks = []
        for sample_index, (log_ratio, local_mask) in enumerate(zip(log_ratios, local_masks, strict=True)):
            d = torch.as_tensor(log_ratio).detach().to(dtype=torch.float64)
            mask = torch.as_tensor(local_mask, device=d.device, dtype=torch.bool)
            if d.ndim != 1 or d.shape != mask.shape:
                raise ValueError(
                    f"Rollout KL metric shape mismatch for {metric_group} at sample {sample_index}: "
                    f"log_ratio={tuple(d.shape)}, local_mask={tuple(mask.shape)}."
                )
            if not torch.isfinite(d[mask]).all():
                raise ValueError(
                    f"Rollout KL metric received a non-finite active {metric_group} log-ratio "
                    f"at sample {sample_index}."
                )

            # Mask before exp so observation/sentinel rows cannot create infinities.
            d = torch.where(mask, d, d.new_zeros(()))
            estimator_chunks["k1"].append(d)
            estimator_chunks["k2"].append(0.5 * d.square())
            estimator_chunks["k3"].append(torch.expm1(-d) + d)
            bool_masks.append(mask)

        for estimator, chunks in estimator_chunks.items():
            values = torch.cat(chunks, dim=0)
            active_values = torch.cat(
                [chunk[mask] for chunk, mask in zip(chunks, bool_masks, strict=True)],
                dim=0,
            )
            local_sum = cp_size * sample_mean(values)
            prefix = f"{metric_group}/{estimator}"
            # Preserve the numerator and rollout count until DP/CP gather. This
            # keeps compacted siblings token-weighted within one rollout and
            # avoids equal-weighting ranks that hold different rollout counts.
            metrics[f"{prefix}/mean"] = (local_sum.item(), rollout_count)
            if active_values.numel() == 0:
                metrics[f"{prefix}/min"] = float("inf")
                metrics[f"{prefix}/max"] = float("-inf")
            else:
                metrics[f"{prefix}/min"] = active_values.min().item()
                metrics[f"{prefix}/max"] = active_values.max().item()
            reduction_by_key[f"{prefix}/min"] = "min"
            reduction_by_key[f"{prefix}/max"] = "max"

    return metrics, reduction_by_key


def log_rollout_data(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """
    Summarize rollout fields and log reduced metrics on PP last stage, TP rank 0.

    - Tensor-valued lists are concatenated and averaged. For token-level metrics
      like log-probs/returns/advantages/values, computes a CP-correct sample mean
      using `loss_masks` and total/response lengths.
    - Non-tensor lists are averaged elementwise.
    - Scalars are converted to Python numbers.
    """
    parallel_state = get_parallel_state()
    if parallel_state.tp.rank == 0 and parallel_state.is_pp_last_stage:
        cp_size = parallel_state.cp.size
        log_dict = {}
        response_lengths = rollout_data["response_lengths"]
        loss_masks = rollout_data["loss_masks"]
        total_lengths = rollout_data["total_lengths"]
        max_seq_lens = rollout_data.get("max_seq_lens", None)
        # Per-rollout mean count share: raw splits carry it directly; scheduled
        # splits derive it from their per-step global rollout counts.
        rollout_count_share = rollout_data.get("rollout_count_share")
        if (num_rollouts := rollout_data.get("num_rollouts")) is not None:
            rollout_count_share = sum(num_rollouts) / parallel_state.intra_dp.size

        for key, val in rollout_data.items():
            if key in [
                "tokens",
                "multimodal_train_inputs",
                "loss_masks",
                "sample_indices",
                "rollout_ids",
                "rollout_mask_sums",
                "rollout_routed_experts",
                "rollout_indexer_topk",
                "teacher_topk_token_ids",
                "teacher_topk_log_probs",
                "teacher_topk_valid_mask",
                "max_seq_lens",
                "dynamic_global_batch_size",
                "witness_ids",
                "weight_versions",
                "metadata",
                "num_microbatches",
                "micro_batch_indices",
                "num_rollouts",
                "rollout_count_share",
                "n_adapters",
                "adapter_slots",
                "step_slots",
                "step_adapter_names",
                "step_adapter_batch_sizes",
                "prompt_group_sizes",
            ]:
                continue
            if isinstance(val, (list, tuple)):
                if isinstance(val[0], torch.Tensor):
                    count = len(val)
                    # NOTE: Here we have to do the clone().detach(), otherwise the tensor will be
                    # modified in place and will cause problem for the next rollout.
                    tensor = torch.cat(val).clone().detach()
                    if tensor.device != loss_masks[0].device:
                        tensor = tensor.to(loss_masks[0].device)
                    if key in [
                        "log_probs",
                        "ref_log_probs",
                        "rollout_log_probs",
                        "returns",
                        "advantages",
                        "values",
                        "teacher_log_probs",
                        "opd_reverse_kl",
                        "entropy",
                    ]:
                        sum_of_sample_mean = get_sum_of_sample_mean(
                            total_lengths,
                            response_lengths,
                            loss_masks,
                            qkv_format=args.qkv_format,
                            max_seq_lens=max_seq_lens,
                            denominators=rollout_data.get("rollout_mask_sums", None),
                        )
                        per_rank_sum = cp_size * sum_of_sample_mean(tensor)
                        if rollout_count_share is not None:
                            count = rollout_count_share
                    else:
                        per_rank_sum = tensor.mean() * cp_size * count
                    log_dict[key] = (per_rank_sum.item(), count)
                else:
                    # Flatten nested lists (e.g. list of lists from async rollout)
                    flat = val
                    if isinstance(val[0], (list, tuple)):
                        flat = [x for sublist in val for x in sublist]
                    # Skip non-numeric values (e.g. strings from async rollout metadata)
                    if flat and not isinstance(flat[0], (int, float)):
                        continue
                    log_dict[key] = (sum(flat), len(flat))
            elif isinstance(val, torch.Tensor):
                log_dict[key] = (val.float().mean().item(), 1)
            else:
                raise ValueError(f"Unsupported type: {type(val)} for key: {key}")

        kl_metrics, reduction_by_key = _compute_rollout_kl_statistics(
            args,
            rollout_data,
            cp_size,
            rollout_count_share=rollout_count_share,
        )
        log_dict.update(kl_metrics)
        reduced_log_dict = gather_log_data(
            "rollout",
            args,
            rollout_id,
            log_dict,
            reduction_by_key=reduction_by_key,
        )
        if args.ci_test and not args.ci_disable_logprobs_checker and reduced_log_dict is not None:
            if (
                rollout_id == 0
                and "rollout/log_probs" in reduced_log_dict
                and "rollout/ref_log_probs" in reduced_log_dict
            ):
                # When R3 (rollout routing replay) is enabled, ref model does not use R3
                # so log_probs and ref_log_probs may diverge; use a relaxed tolerance.
                # When --sglang-config deploys multiple models, the heavier offload/onload
                # cycle can amplify flash-attention non-determinism; use 1e-8.
                # The default branch also covers larger TP/CP/EP variants (e.g. stage-c-long
                # test_qwen2.5_0.5B_gsm8k.py on 8 GPUs hit ~3.7e-9 diff in CI), so use 1e-8
                # rather than the previous 3e-9 to absorb BF16 reduction noise across configs.
                if args.use_rollout_routing_replay:
                    # lop diff w/ w/o r3 is very big
                    abs_tol = 5e-3
                elif getattr(args, "sglang_config", None) is not None:
                    abs_tol = 1e-8
                else:
                    abs_tol = 1e-8
                assert isclose(
                    reduced_log_dict["rollout/log_probs"], reduced_log_dict["rollout/ref_log_probs"], abs_tol=abs_tol
                ), f"CI check failed: log_probs ({reduced_log_dict['rollout/log_probs']}) != ref_log_probs ({reduced_log_dict['rollout/ref_log_probs']})"
            if "rollout/log_probs" in reduced_log_dict and "rollout/rollout_log_probs" in reduced_log_dict:
                assert isclose(
                    reduced_log_dict["rollout/log_probs"], reduced_log_dict["rollout/rollout_log_probs"], abs_tol=0.03
                ), f"CI check failed: log_probs ({reduced_log_dict['rollout/log_probs']}) != rollout_log_probs ({reduced_log_dict['rollout/rollout_log_probs']})"
            if "rollout/entropy" in reduced_log_dict:
                assert 0 < reduced_log_dict["rollout/entropy"] < 0.7

        if args.ci_test and args.true_on_policy_mode and not args.ci_disable_logprobs_checker:
            assert log_dict["log_probs"] == log_dict["rollout_log_probs"], (
                f"CI check failed: true_on_policy_mode is enabled, but log_probs "
                f"({log_dict['log_probs']}) != rollout_log_probs "
                f"({log_dict['rollout_log_probs']})"
            )

    if args.log_multi_turn:
        log_multi_turn_data(rollout_id, args, rollout_data)

    if args.log_correct_samples:
        if parallel_state.tp.rank == 0 and parallel_state.is_pp_last_stage:
            cp_size = parallel_state.cp.size
            log_dict = {}
            response_lengths = rollout_data["response_lengths"]
            loss_masks = rollout_data["loss_masks"]
            total_lengths = rollout_data["total_lengths"]

            def quantile(total_value, n_quantiles, data) -> dict:
                import math

                assert n_quantiles > 1, f"n_quantiles({n_quantiles}) must be greater than 1."

                quantiles = [((i + 1) / n_quantiles) for i in range(n_quantiles)]
                cut_points = [total_value * q for q in quantiles]
                cut_points[-1] = total_value

                count = [0] * n_quantiles
                for d in data:
                    for i, point in enumerate(cut_points):
                        if d <= point:
                            count[i] += 1
                            break

                total = sum(count) + 1e-9
                percentile = [c / total for c in count]

                percentile = {f"p{min(math.ceil(q*100),100)}": p for q, p in zip(quantiles, percentile, strict=True)}
                return percentile

            raw_rewards = rollout_data["raw_reward"]
            # Additional metrics for correct cases are calculated separately below.
            correct_response_lengths = []
            correct_total_lengths = []
            correct_loss_masks = []
            correct_entropy = []
            for i, raw_reward in enumerate(raw_rewards):
                if raw_reward == 1:
                    correct_response_lengths.append(response_lengths[i])
                    correct_total_lengths.append(total_lengths[i])
                    correct_loss_masks.append(loss_masks[i])
                    correct_entropy.append(-rollout_data["log_probs"][i])
            num_correct_responses = len(correct_total_lengths)
            rollout_data["correct_response_lengths"] = correct_response_lengths
            correct_response_length_percentile = quantile(
                args.rollout_max_response_len, 4, rollout_data["correct_response_lengths"]
            )
            for p, val in correct_response_length_percentile.items():
                rollout_data[f"correct_length/{p}"] = [val] * num_correct_responses
            if len(correct_entropy) > 0:
                # per-sample mean over the correct subset, not per-rollout
                sum_of_sample_mean = get_sum_of_sample_mean(
                    correct_total_lengths, correct_response_lengths, correct_loss_masks, denominators=None
                )
                correct_entropy = sum_of_sample_mean(torch.cat(correct_entropy, dim=0))
                rollout_data["correct_entropy"] = [correct_entropy.item()] * num_correct_responses
            else:
                rollout_data["correct_entropy"] = [0] * num_correct_responses


def log_multi_turn_data(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """
    Log interaction structure without duplicating generic rollout-length metrics.

    Operates only on PP last stage and TP rank 0. Uses GPU tensors when available
    to compute statistics without host transfers.
    """
    parallel_state = get_parallel_state()
    if parallel_state.tp.rank == 0 and parallel_state.is_pp_last_stage:
        log_dict = {}
        reduction_by_key = {}
        for key, val in rollout_data.items():
            if key == "loss_masks":
                if val:
                    device = val[0].device
                    raw_response_lengths = torch.tensor([v.shape[0] for v in val], dtype=torch.float32, device=device)
                    active_response_lengths = torch.stack([v.float().sum() for v in val])
                    observation_lengths = raw_response_lengths - active_response_lengths

                    log_dict["raw_tokens/max"] = raw_response_lengths.max().item()
                    reduction_by_key["raw_tokens/max"] = "max"
                    log_dict["length_cap_ratio"] = (
                        (raw_response_lengths >= args.rollout_max_response_len).float().mean().item()
                    )
                    log_dict["observation_tokens/mean"] = observation_lengths.mean().item()
                    log_dict["observation_token_ratio"] = (
                        (observation_lengths / raw_response_lengths.clamp_min(1)).mean().item()
                    )
            if key == "round_number":
                round_number_array = np.array(val)
                log_dict["rounds/mean"] = np.mean(round_number_array)
                log_dict["rounds/max"] = np.max(round_number_array)
                log_dict["rounds/min"] = np.min(round_number_array)
                reduction_by_key["rounds/max"] = "max"
                reduction_by_key["rounds/min"] = "min"
        gather_log_data(
            "interaction",
            args,
            rollout_id,
            log_dict,
            reduction_by_key=reduction_by_key,
        )


def log_perf_data(rollout_id: int, args: Namespace, extra_metrics: dict | None = None) -> None:
    parallel_state = get_parallel_state()
    train_metric_utils.log_perf_data_raw(
        rollout_id=rollout_id,
        args=args,
        is_primary_rank=(
            parallel_state.tp.rank == 0
            and parallel_state.is_pp_last_stage
            and parallel_state.effective_dp_cp.rank == 0
        ),
        compute_total_fwd_flops=lambda seq_lens: fwd_tflops_per_gpu(seq_lens, args, dist.get_world_size()),
        extra_metrics=extra_metrics,
    )


def log_cpu_memory(rollout_id: int, args: Namespace, label: str) -> None:
    """Log current system CPU memory usage to wandb/tensorboard.

    Caller is responsible for ensuring this runs on a single rank only.
    """

    cpu_mem_gb = psutil.virtual_memory().used / 1e9
    step = compute_rollout_step(args, rollout_id)
    logger.info(f"[CPU memory] {label}: {cpu_mem_gb:.2f} GB (rollout_id={rollout_id}, step={step})")
    tracking.log(
        args,
        {f"perf/cpu_memory_{label}_gb": cpu_mem_gb, "rollout/step": step},
        step_key="rollout/step",
    )


def aggregate_train_losses(
    losses_reduced: list[dict[str, list[str] | torch.Tensor]],
    num_rollouts: int | None = None,
) -> dict[str, float]:
    """Aggregate loss metrics across micro-batches.

    Sums loss values across all micro-batches, performs all-reduce across
    the data-parallel group, and computes per-sample/token averages.

    Args:
        losses_reduced: List of log_dict from each micro-batch.
            Each log_dict has format: {"keys": list[str], "values": torch.Tensor}
        num_rollouts: report per-rollout means — divide every metric by this
            step's rollout count (no CP factor; CP-chunked numerators reconstruct
            exactly once under the DP*CP all-reduce). None keeps the legacy
            reduction: divide by the all-reduced ``values[0]`` count, cancelled
            by ``cp_size``.

    Returns:
        Dictionary mapping metric names to averaged values.
    """
    parallel_state = get_parallel_state()
    if not losses_reduced:
        return {}

    keys = losses_reduced[0]["keys"]

    values = None
    for log_dict in losses_reduced:
        if values is None:
            values = log_dict["values"].clone()
        else:
            values += log_dict["values"]

    assert len(keys) + 1 == values.numel(), f"Expected {len(keys) + 1} values, got {values.numel()}"

    for group in parallel_state.effective_dp_cp.groups_inner_to_outer:
        MultiPGUtil.all_reduce(values, [group], op=dist.ReduceOp.SUM)

    loss_reduced = {}
    values = values.tolist()
    if num_rollouts is not None:
        num_samples_or_tokens = num_rollouts
        cp_factor = 1
    else:
        num_samples_or_tokens = values[0]
        cp_factor = parallel_state.cp.size

    for key, value in zip(keys, values[1:], strict=False):
        loss_reduced[key] = value * cp_factor / num_samples_or_tokens

    return loss_reduced


def log_train_step(
    args: Namespace,
    loss_dict: dict[str, float],
    grad_norm: float,
    rollout_id: int,
    step_id: int,
    num_steps_per_rollout: int,
    role: str = "actor",
    extra_metrics: dict[str, float] | None = None,
    should_log: bool | None = None,
) -> dict[str, float]:
    """Log training metrics for one step.

    Formats loss metrics, gradient norm, and extra metrics (e.g., learning rates, MTP loss) for tracking.

    Args:
        args: Configuration.
        loss_dict: Dictionary of loss metrics from aggregate_train_losses.
        grad_norm: Global gradient L2 norm before clipping.
        rollout_id: Rollout ID.
        step_id: Step ID within the rollout.
        num_steps_per_rollout: Total number of steps per rollout.
        role: Role name (e.g., "actor", "critic").
        extra_metrics: Optional extra metrics to log (e.g., learning rates, MTP loss).
        should_log: Optional override for logging condition. If None, uses rank == 0.

    Returns:
        The formatted log_dict (for CI tests or other uses).
    """
    accumulated_step_id = rollout_id * num_steps_per_rollout + step_id
    role_tag = "" if role == "actor" else f"{role}-"

    log_dict_out = {}
    for key, val in loss_dict.items():
        output_key = key if key.startswith("opd_dagger/") else f"train/{role_tag}{key}"
        log_dict_out[output_key] = val.mean().item() if isinstance(val, torch.Tensor) else val
    log_dict_out[f"train/{role_tag}grad_norm"] = float(grad_norm)

    if extra_metrics:
        for key, val in extra_metrics.items():
            log_dict_out[f"train/{role_tag}{key}"] = val

    log_dict_out["train/step"] = accumulated_step_id

    if should_log is None:
        should_log = dist.get_rank() == 0

    if should_log:
        tracking.log(args, log_dict_out, step_key="train/step")
        logger.info(f"{role_tag}step {accumulated_step_id}: {log_dict_out}")

    return log_dict_out
