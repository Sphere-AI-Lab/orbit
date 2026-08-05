import logging
from argparse import Namespace
from math import isclose

import numpy as np
import psutil
import torch
import torch.distributed as dist

from orbit.utils import train_metric_utils
from orbit.utils.flops_utils import calculate_fwd_flops
from orbit.utils.metric_utils import compute_pass_rate, compute_rollout_step
from orbit.utils.types import RolloutBatch

from ...utils import tracking_utils
from .cp_utils import get_logits_and_tokens_offset_with_cp, get_sum_of_sample_mean
from .data import DataIterator
from .parallel import get_parallel_state

logger = logging.getLogger(__name__)


def _local_response_loss_masks(
    *,
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    qkv_format: str,
    max_seq_lens: list[int] | None,
) -> list[torch.Tensor]:
    """Return loss masks in the same response/CP layout as stored log-probs."""
    if get_parallel_state().cp.size == 1:
        return loss_masks

    local_masks = []
    for i, (total_length, response_length, loss_mask) in enumerate(
        zip(total_lengths, response_lengths, loss_masks, strict=True)
    ):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
        prompt_length = total_length - response_length
        _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
            total_length,
            response_length,
            qkv_format,
            max_seq_len,
        )
        first = loss_mask[token_offsets[0][0] - prompt_length : token_offsets[0][1] - prompt_length]
        second = loss_mask[token_offsets[1][0] - prompt_length : token_offsets[1][1] - prompt_length]
        local_masks.append(torch.cat((first, second), dim=0))
    return local_masks


def _assert_true_on_policy_logprob_parity(
    args: Namespace,
    rollout_data: RolloutBatch,
    *,
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    max_seq_lens: list[int] | None,
) -> None:
    """Require exact per-token train/rollout parity at valid response positions."""
    train_log_probs = rollout_data.get("log_probs")
    rollout_log_probs = rollout_data.get("rollout_log_probs")
    assert train_log_probs is not None and rollout_log_probs is not None, (
        "CI check failed: the Phase-5 true-on-policy parity gate requires both " "log_probs and rollout_log_probs."
    )
    assert len(train_log_probs) == len(rollout_log_probs) == len(loss_masks), (
        "CI check failed: true-on-policy log-prob and loss-mask sample counts differ: "
        f"{len(train_log_probs)}, {len(rollout_log_probs)}, and {len(loss_masks)}."
    )

    local_masks = _local_response_loss_masks(
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        loss_masks=loss_masks,
        qkv_format=args.qkv_format,
        max_seq_lens=max_seq_lens,
    )
    for sample_index, (train, rollout, loss_mask) in enumerate(
        zip(train_log_probs, rollout_log_probs, local_masks, strict=True)
    ):
        assert isinstance(train, torch.Tensor) and isinstance(
            rollout, torch.Tensor
        ), "CI check failed: true-on-policy log-probs must be tensors."
        assert train.shape == rollout.shape, (
            f"CI check failed: true-on-policy sample {sample_index} shapes differ: "
            f"{tuple(train.shape)} != {tuple(rollout.shape)}."
        )
        assert train.dtype == rollout.dtype, (
            f"CI check failed: true-on-policy sample {sample_index} dtypes differ: "
            f"{train.dtype} != {rollout.dtype}."
        )
        assert train.numel() == loss_mask.numel(), (
            f"CI check failed: true-on-policy sample {sample_index} has {train.numel()} "
            f"log-probs but {loss_mask.numel()} local response mask entries."
        )

        valid = loss_mask.to(device=train.device, dtype=torch.bool).reshape(train.shape)
        rollout = rollout.to(device=train.device)
        train_valid = train[valid]
        rollout_valid = rollout[valid]
        if not torch.equal(train_valid, rollout_valid):
            max_abs_diff = (
                (train_valid.float() - rollout_valid.float()).abs().max().item() if train_valid.numel() else 0.0
            )
            raise AssertionError(
                "CI check failed: true_on_policy_mode is enabled with the Phase-5 "
                "SGLang-in-Megatron backend active, but masked per-token log_probs "
                f"and rollout_log_probs differ for sample {sample_index} "
                f"(max_abs_diff={max_abs_diff})."
            )


def _assert_true_on_policy_logprob_parity_synchronized(
    args: Namespace,
    rollout_data: RolloutBatch,
    *,
    total_lengths: list[int],
    response_lengths: list[int],
    loss_masks: list[torch.Tensor],
    max_seq_lens: list[int] | None,
) -> None:
    """Run the parity check without stranding peers in the next collective."""
    local_error = None
    try:
        _assert_true_on_policy_logprob_parity(
            args,
            rollout_data,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            loss_masks=loss_masks,
            max_seq_lens=max_seq_lens,
        )
    except Exception as exc:  # synchronize structural errors as well as value mismatches
        local_error = f"{type(exc).__name__}: {exc}"

    group_info = get_parallel_state().intra_dp_cp
    if group_info.size > 1:
        errors: list[str | None] = [None] * group_info.size
        dist.all_gather_object(errors, local_error, group=group_info.gloo_group)
    else:
        errors = [local_error]

    failures = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error is not None]
    if failures:
        raise AssertionError("Synchronized true-on-policy parity failure; " + "; ".join(failures))


def gather_log_data(
    metric_name: str,
    args: Namespace,
    rollout_id: int,
    log_dict: dict[str, float],
) -> dict[str, float] | None:
    """
    Gather per-rank metrics, reduce by mean on the DP source rank, and log.

    Expects `log_dict` to contain plain scalars. The DP source rank prints and
    optionally logs to WandB/TensorBoard with a step derived from `rollout_id` and
    batch sizes. Returns the reduced dict on the DP source rank; returns None on others.
    """

    parallel_state = get_parallel_state()

    pg = parallel_state.intra_dp_cp
    dp_size = pg.size
    gathered_log_dict = [None] * dp_size
    # Not sure if this will be a performance bottleneck.
    dist.gather_object(
        log_dict,
        gathered_log_dict if pg.rank == 0 else None,
        dst=dist.get_global_rank(pg.gloo_group, 0),
        group=pg.gloo_group,
    )

    if pg.rank == 0:
        reduced_log_dict = {
            f"{metric_name}/{key}": sum([d[key] for d in gathered_log_dict]) / dp_size for key in log_dict
        }
        logger.info(f"{metric_name} {rollout_id}: {reduced_log_dict}")

        # Calculate step once to avoid duplication
        step = compute_rollout_step(args, rollout_id)
        reduced_log_dict["rollout/step"] = step
        tracking_utils.log(args, reduced_log_dict, step_key="rollout/step")

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

        if (
            getattr(args, "ci_test", False)
            and not getattr(args, "ci_disable_logprobs_checker", False)
            and getattr(args, "true_on_policy_mode", False)
            and getattr(args, "true_on_policy_megatron_uses_sglang_backend", False)
        ):
            # A reduced scalar mean can hide equal-and-opposite token errors.
            # Gate the contract on the raw, loss-mask-valid response values.
            _assert_true_on_policy_logprob_parity_synchronized(
                args,
                rollout_data,
                total_lengths=total_lengths,
                response_lengths=response_lengths,
                loss_masks=loss_masks,
                max_seq_lens=max_seq_lens,
            )

        for key, val in rollout_data.items():
            if key in [
                "tokens",
                "multimodal_train_inputs",
                "loss_masks",
                "sample_indices",
                "rollout_routed_experts",
                "teacher_topk_ids",
                "teacher_topk_logprobs",
                "max_seq_lens",
                "dynamic_global_batch_size",
                "weight_versions",
                "metadata",
            ]:
                continue
            # Upload per sample mean for each rollout value
            # There are the following assumptions:
            # - Each dp rank has the same number of samples
            if isinstance(val, (list, tuple)):
                if isinstance(val[0], torch.Tensor):
                    # NOTE: Here we have to do the clone().detach(), otherwise the tensor will be
                    # modified in place and will cause problem for the next rollout.
                    val = torch.cat(val).clone().detach()
                    if val.device != loss_masks[0].device:
                        val = val.to(loss_masks[0].device)
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
                        )
                        val = cp_size * sum_of_sample_mean(val) / len(loss_masks)
                    else:
                        val = val.mean() * cp_size
                else:
                    # Flatten nested lists (e.g. list of lists from async rollout)
                    flat = val
                    if isinstance(val[0], (list, tuple)):
                        flat = [x for sublist in val for x in sublist]
                    # Skip non-numeric values (e.g. strings from async rollout metadata)
                    if flat and not isinstance(flat[0], (int, float)):
                        continue
                    val = sum(flat) / len(flat)
            elif isinstance(val, torch.Tensor):
                val = val.float().mean()
            else:
                raise ValueError(f"Unsupported type: {type(val)} for key: {key}")
            log_dict[key] = val.item() if isinstance(val, torch.Tensor) else val

        reduced_log_dict = gather_log_data("rollout", args, rollout_id, log_dict)
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
                if args.use_rollout_routing_replay:
                    abs_tol = 1e-5
                elif getattr(args, "sglang_config", None) is not None:
                    abs_tol = 1e-8
                else:
                    abs_tol = 1e-9
                assert isclose(
                    reduced_log_dict["rollout/log_probs"], reduced_log_dict["rollout/ref_log_probs"], abs_tol=abs_tol
                ), f"CI check failed: log_probs ({reduced_log_dict['rollout/log_probs']}) != ref_log_probs ({reduced_log_dict['rollout/ref_log_probs']})"
            if "rollout/log_probs" in reduced_log_dict and "rollout/rollout_log_probs" in reduced_log_dict:
                assert isclose(
                    reduced_log_dict["rollout/log_probs"], reduced_log_dict["rollout/rollout_log_probs"], abs_tol=0.03
                ), f"CI check failed: log_probs ({reduced_log_dict['rollout/log_probs']}) != rollout_log_probs ({reduced_log_dict['rollout/rollout_log_probs']})"
            if "rollout/entropy" in reduced_log_dict:
                assert 0 < reduced_log_dict["rollout/entropy"] < 0.7

    if args.log_multi_turn:
        log_multi_turn_data(rollout_id, args, rollout_data)
    if args.log_passrate:
        log_passrate(rollout_id, args, rollout_data)

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
                sum_of_sample_mean = get_sum_of_sample_mean(
                    correct_total_lengths, correct_response_lengths, correct_loss_masks
                )
                correct_entropy = sum_of_sample_mean(torch.cat(correct_entropy, dim=0))
                rollout_data["correct_entropy"] = [correct_entropy.item()] * num_correct_responses
            else:
                rollout_data["correct_entropy"] = [0] * num_correct_responses


def log_multi_turn_data(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """
    Log multi-turn auxiliary metrics such as raw/observed response lengths and rounds.

    Operates only on PP last stage and TP rank 0. Uses GPU tensors when available
    to compute statistics without host transfers.
    """
    parallel_state = get_parallel_state()
    if parallel_state.tp.rank == 0 and parallel_state.is_pp_last_stage:
        log_dict = {}
        for key, val in rollout_data.items():
            if key == "loss_masks":
                if val:  # Check if val is not empty
                    device = val[0].device  # Get device from first tensor

                    # Vectorized length calculation using torch
                    raw_response_lengths = torch.tensor([v.shape[0] for v in val], dtype=torch.float32, device=device)
                    log_dict["raw_response_length/response_length_mean"] = raw_response_lengths.mean().item()
                    log_dict["raw_response_length/response_length_max"] = raw_response_lengths.max().item()
                    log_dict["raw_response_length/response_length_min"] = raw_response_lengths.min().item()
                    log_dict["raw_response_length/response_length_clip_ratio"] = (
                        (raw_response_lengths >= args.rollout_max_response_len).float().mean().item()
                    )

                    # Vectorized sum calculation using torch - stay on GPU
                    wo_obs_response_lengths = torch.tensor(
                        [v.sum().item() for v in val], dtype=torch.float32, device=device
                    )
                    log_dict["wo_obs_response_length/response_length_mean"] = wo_obs_response_lengths.mean().item()
                    log_dict["wo_obs_response_length/response_length_max"] = wo_obs_response_lengths.max().item()
                    log_dict["wo_obs_response_length/response_length_min"] = wo_obs_response_lengths.min().item()
            if key == "round_number":
                # Use numpy for vectorized round number statistics
                round_number_array = np.array(val)
                log_dict["multi_turn_metric/round_number_mean"] = np.mean(round_number_array)
                log_dict["multi_turn_metric/round_number_max"] = np.max(round_number_array)
                log_dict["multi_turn_metric/round_number_min"] = np.min(round_number_array)
        gather_log_data("multi_turn", args, rollout_id, log_dict)


def log_passrate(rollout_id: int, args: Namespace, rollout_data: RolloutBatch) -> None:
    """
    Compute pass@k metrics from `raw_reward` groups and log the results.

    `raw_reward` is reshaped to `[group_number, group_size]`, then pass@k is
    estimated per problem and averaged.
    """
    parallel_state = get_parallel_state()
    if parallel_state.tp.rank == 0 and parallel_state.is_pp_last_stage:
        log_dict = {}
        for key, val in rollout_data.items():
            if key != "raw_reward":
                continue

            log_dict |= compute_pass_rate(
                flat_rewards=val,
                group_size=args.n_samples_per_prompt,
                num_groups=args.rollout_batch_size,
            )

        gather_log_data("passrate", args, rollout_id, log_dict)


def log_perf_data(rollout_id: int, args: Namespace) -> None:
    parallel_state = get_parallel_state()
    train_metric_utils.log_perf_data_raw(
        rollout_id=rollout_id,
        args=args,
        is_primary_rank=(
            parallel_state.tp.rank == 0 and parallel_state.is_pp_last_stage and parallel_state.intra_dp_cp.rank == 0
        ),
        compute_total_fwd_flops=lambda seq_lens: calculate_fwd_flops(seqlens=seq_lens, args=args)
        / dist.get_world_size()
        / 1e12,
    )


def log_cpu_memory(rollout_id: int, args: Namespace, label: str) -> None:
    """Log current system CPU memory usage to wandb/tensorboard.

    Caller is responsible for ensuring this runs on a single rank only.
    """

    cpu_mem_gb = psutil.virtual_memory().used / 1e9
    step = compute_rollout_step(args, rollout_id)
    logger.info(f"[CPU memory] {label}: {cpu_mem_gb:.2f} GB (rollout_id={rollout_id}, step={step})")
    tracking_utils.log(
        args,
        {f"perf/cpu_memory_{label}_gb": cpu_mem_gb, "rollout/step": step},
        step_key="rollout/step",
    )


def aggregate_train_losses(
    losses_reduced: list[dict[str, list[str] | torch.Tensor]],
) -> dict[str, float]:
    """Aggregate loss metrics across micro-batches.

    Sums loss values across all micro-batches, performs all-reduce across
    the data-parallel group, and computes per-sample/token averages.

    Args:
        losses_reduced: List of log_dict from each micro-batch.
            Each log_dict has format: {"keys": list[str], "values": torch.Tensor}
        parallel_state: Parallel state containing dp_group and cp_size.

    Returns:
        Dictionary mapping metric names to averaged values.
    """
    parallel_state = get_parallel_state()
    if not losses_reduced:
        return {}

    keys = losses_reduced[0]["keys"]

    max_metric_indices = {i + 1 for i, key in enumerate(keys) if key.endswith("_max")}
    min_metric_indices = {i + 1 for i, key in enumerate(keys) if key.endswith("_min")}
    values = None
    max_values = None
    min_values = None
    for log_dict in losses_reduced:
        log_values = log_dict["values"]
        if values is None:
            values = torch.zeros_like(log_values)
            max_values = torch.full_like(log_values, -torch.inf)
            min_values = torch.full_like(log_values, torch.inf)

        values[0] += log_values[0]
        for i in range(1, log_values.numel()):
            if i in max_metric_indices:
                max_values[i] = torch.maximum(max_values[i], log_values[i])
            elif i in min_metric_indices:
                min_values[i] = torch.minimum(min_values[i], log_values[i])
            else:
                values[i] += log_values[i]

    assert len(keys) + 1 == values.numel(), f"Expected {len(keys) + 1} values, got {values.numel()}"

    dist.all_reduce(values, op=dist.ReduceOp.SUM, group=parallel_state.intra_dp_cp.group)
    if max_metric_indices:
        dist.all_reduce(max_values, op=dist.ReduceOp.MAX, group=parallel_state.intra_dp_cp.group)
        for i in max_metric_indices:
            values[i] = max_values[i]
    if min_metric_indices:
        dist.all_reduce(min_values, op=dist.ReduceOp.MIN, group=parallel_state.intra_dp_cp.group)
        for i in min_metric_indices:
            values[i] = min_values[i]

    loss_reduced = {}
    values = values.tolist()
    num_samples_or_tokens = values[0]

    for key, value in zip(keys, values[1:], strict=False):
        if key.endswith(("_max", "_min")):
            loss_reduced[key] = value
        else:
            loss_reduced[key] = value * parallel_state.cp.size / num_samples_or_tokens

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
        grad_norm: Gradient norm after clipping.
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

    log_dict_out = {
        f"train/{role_tag}{key}": val.mean().item() if isinstance(val, torch.Tensor) else val
        for key, val in loss_dict.items()
    }
    log_dict_out[f"train/{role_tag}grad_norm"] = float(grad_norm)

    if extra_metrics:
        for key, val in extra_metrics.items():
            log_dict_out[f"train/{role_tag}{key}"] = val

    log_dict_out["train/step"] = accumulated_step_id

    if should_log is None:
        should_log = dist.get_rank() == 0

    if should_log:
        tracking_utils.log(args, log_dict_out, step_key="train/step")
        logger.info(f"{role_tag}step {accumulated_step_id}: {log_dict_out}")

    return log_dict_out
