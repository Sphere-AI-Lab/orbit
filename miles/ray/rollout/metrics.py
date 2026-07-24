import logging
from typing import Any

import numpy as np

from miles.utils.iter_utils import group_by
from miles.utils.metric_utils import (
    compute_pass_rate,
    compute_rollout_step,
    compute_statistics,
    dict_add_prefix,
    has_repetition,
)
from miles.utils.misc import load_function
from miles.utils.tracking_utils import tracking
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


def log_eval_rollout_data(rollout_id, args, data, extra_metrics: dict[str, Any] | None = None):
    extra_metrics = extra_metrics or {}
    if (x := args.custom_eval_rollout_log_function_path) is not None:
        custom_log_func = load_function(x)
        if custom_log_func(rollout_id, args, data, extra_metrics):
            return

    log_dict = extra_metrics
    for key in data.keys():
        rewards = data[key]["rewards"]
        num_none = sum(1 for r in rewards if r is None)
        log_dict[f"eval/{key}-none_reward_ratio"] = num_none / len(rewards) if len(rewards) > 0 else 0.0
        if num_none:
            logger.warning(
                f"eval/{key}: {num_none}/{len(rewards)} samples have reward=None "
                "(likely errored/aborted trials); treating as 0.0 for metrics."
            )
            rewards = [0.0 if r is None else r for r in rewards]
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards) if len(rewards) > 0 else 0.0
        if (samples := data[key].get("samples")) is not None:
            log_dict |= dict_add_prefix(_compute_metrics_from_samples(args, samples), f"eval/{key}/")
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)
        if args.log_passrate:
            log_dict |= dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=rewards,
                    group_size=args.n_samples_per_eval_prompt,
                ),
                f"eval/{key}-",
            )

    logger.info(f"eval {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    tracking.log(args, log_dict, step_key="eval/step")

    return log_dict


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    if (x := args.custom_rollout_log_function_path) is not None:
        custom_log_func = load_function(x)
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            return

    if args.load_debug_rollout_data:
        return

    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= dict_add_prefix(_compute_metrics_from_samples(args, samples), "rollout/")
    log_dict |= dict_add_prefix(_compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    log_dict |= _compute_distillation_rpc_metrics(samples)
    if args.log_passrate:
        log_dict |= dict_add_prefix(
            _compute_passrate_from_samples(args, samples),
            "passrate/",
        )
    logger.info(f"perf {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    tracking.log(args, log_dict, step_key="rollout/step")


def _compute_metrics_from_samples(args, samples):
    response_lengths = [sample.effective_response_length for sample in samples]

    log_dict = {}
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= _compute_zero_std_metrics(args, samples)
    log_dict |= _compute_spec_metrics(args, samples)
    log_dict |= _compute_prefix_cache_metrics(args, samples)
    log_dict |= _compute_reward_cat_metrics(args, samples)
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in samples]).item()
    log_dict["truncated_ratio"] = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item()

    oldest_versions = [s.oldest_weight_version for s in samples if s.oldest_weight_version is not None]
    if oldest_versions:
        log_dict |= dict_add_prefix(compute_statistics(oldest_versions), "weight_version/")
        mixed = sum(1 for s in samples if len(set(s.weight_versions)) > 1)
        log_dict["weight_version/mixed_version_ratio"] = mixed / len(samples)

    tito_vals = [s.metadata.get("tito_session_mismatch") for s in samples]
    tito_vals = [v for v in tito_vals if v is not None]
    if tito_vals:
        log_dict["tito_session_mismatch_rate"] = np.mean([len(v) > 0 for v in tito_vals]).item()
        for mtype in ("special_token_count", "special_token_type", "non_assistant_text", "assistant_text"):
            log_dict[f"tito_session_mismatch_rate/{mtype}"] = np.mean(
                [any(m.get("type") == mtype for m in v) for v in tito_vals]
            ).item()
        if args.ci_test:
            for strict_type in ("special_token_count", "special_token_type", "non_assistant_text"):
                rate = log_dict.get(f"tito_session_mismatch_rate/{strict_type}", 0)
                assert rate == 0, (
                    f"tito_session_mismatch_rate/{strict_type}={rate:.4f} must be 0 — "
                    "this indicates a bug in the TITO algorithm or chat template. "
                    "Please check your tito model and chat template."
                )
            # assistant_text mismatch is non-critical: assistant tokens are inherited
            # from the pretokenized prefix and may differ from canonical tokenization.

    return log_dict


def _compute_distillation_rpc_metrics(all_samples: list[Sample]) -> dict[str, float | int]:
    """Publish a compact health summary for distillation scoring RPCs.

    The per-request metadata remains intentionally detailed for debug dumps and
    local inspection. W&B only receives the signals needed to detect extra
    student rescoring, retries, queue pressure, connection churn, and payload
    growth.
    """
    calls = []
    sample_count = 0
    for sample in all_samples:
        entries = sample.metadata.get("opd_scoring_telemetry") if sample.metadata else None
        if not isinstance(entries, list):
            continue
        valid_entries = [entry for entry in entries if isinstance(entry, dict)]
        if valid_entries:
            sample_count += 1
            calls.extend(valid_entries)

    if not calls:
        return {}

    # Matrix-shape proxy: native top-k is N*K; arbitrary-ID rescoring is N*U.
    enriched_calls = []
    for call in calls:
        enriched = dict(call)
        returned_positions = call.get("returned_positions")
        requested_token_ids = call.get("requested_token_ids")
        top_k = call.get("top_k")
        if isinstance(returned_positions, (int, float)):
            requested_count = requested_token_ids if isinstance(requested_token_ids, (int, float)) else 0
            top_k_count = top_k if isinstance(top_k, (int, float)) else 0
            enriched["candidate_logprob_cells"] = returned_positions * (requested_count + top_k_count)
        enriched_calls.append(enriched)
    calls = enriched_calls

    metrics: dict[str, float | int] = {
        "distillation_rpc/teacher/requests_per_sample": (
            sum(call.get("target") == "teacher" for call in calls) / sample_count
        ),
        "distillation_rpc/student/requests_per_sample": (
            sum(call.get("target") == "student" for call in calls) / sample_count
        ),
        "distillation_rpc/retry_rate": np.mean(
            [
                int(
                    int(call.get("attempts", 1)) > 1
                    or int(call.get("transport_attempts", 1)) > 1
                    or int(call.get("stale_connection_retry_count", 0)) > 0
                )
                for call in calls
            ]
        ).item(),
    }

    connection_reuse = [call["connection_reused"] for call in calls if isinstance(call.get("connection_reused"), bool)]
    if connection_reuse:
        metrics["distillation_rpc/connection_reuse_rate"] = sum(connection_reuse) / len(connection_reuse)

    percentile_fields = {
        "e2e_latency_s": "distillation_rpc/e2e_latency_s/p95",
        "semaphore_wait_s": "distillation_rpc/semaphore_wait_s/p95",
        "response_body_bytes": "distillation_rpc/payload/response_body_bytes_p95",
    }
    for field, key in percentile_fields.items():
        values = [float(call[field]) for call in calls if isinstance(call.get(field), (int, float))]
        if values:
            metrics[key] = np.percentile(np.asarray(values, dtype=np.float64), 95).item()

    candidate_cells = [
        float(call["candidate_logprob_cells"])
        for call in calls
        if isinstance(call.get("candidate_logprob_cells"), (int, float))
    ]
    if candidate_cells:
        metrics["distillation_rpc/payload/candidate_logprob_cells_max"] = max(candidate_cells)

    return metrics


def _compute_perf_metrics_from_samples(args, samples, rollout_time):
    non_generation_time = [sample.non_generation_time for sample in samples]

    log_dict = {}
    log_dict["rollout_time"] = rollout_time
    if max(non_generation_time) > 0:
        log_dict |= dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    def token_perf(response_lengths, non_generation_time, key=""):
        max_response_length = max(response_lengths)
        if args.rollout_num_gpus:
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        if max(non_generation_time) == 0:
            return

        non_generation_time = [
            t for t, length in zip(non_generation_time, response_lengths, strict=True) if length == max_response_length
        ]
        mean_non_generation_time = sum(non_generation_time) / len(non_generation_time)

        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (
            rollout_time - mean_non_generation_time
        )

    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")

    return log_dict


def _compute_zero_std_metrics(args, all_samples: list[Sample]):
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    if args.advantage_estimator == "ppo":
        return {}

    def _reward_value(sample: Sample):
        if getattr(args, "use_opd", False) and getattr(args, "opd_log_task_reward", False):
            metadata = sample.metadata or {}
            if "raw_reward" not in metadata:
                raise ValueError("OPD task-reward logging is enabled, but sample.metadata['raw_reward'] is missing.")
            return metadata["raw_reward"]
        return sample.get_reward_value(args)

    def _is_zero_std(samples: list[Sample]):
        rewards = [_reward_value(sample) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    interesting_sample_groups = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    interesting_rewards = [round(float(_reward_value(g[0])), 1) for g in interesting_sample_groups]

    counts = {reward: len(items) for reward, items in group_by(interesting_rewards).items()}
    log_dict = {f"zero_std/count_{reward:g}": count for reward, count in counts.items()}

    # Percentages over total groups, so "too hard" (all-0) and "too easy"
    # (all-1) rates are comparable across runs without needing to know the
    # rollout batch size.
    total_groups = len(all_sample_groups)
    if total_groups > 0:
        log_dict["zero_std/all_zero_percentage"] = counts.get(0.0, 0) / total_groups
        log_dict["zero_std/all_one_percentage"] = counts.get(1.0, 0) / total_groups

    return log_dict


def _compute_spec_metrics(args, all_samples: list[Sample]):
    if args.sglang_speculative_algorithm is None:
        return {}
    num_samples = len(all_samples)
    metrics = {}
    metrics["spec_accept_rate"] = sum(sample.spec_info.spec_accept_rate for sample in all_samples) / num_samples
    metrics["spec_accept_length"] = sum(sample.spec_info.spec_accept_length for sample in all_samples) / num_samples
    return metrics


def _compute_prefix_cache_metrics(args, all_samples: list[Sample]):
    num_samples = len(all_samples)
    metrics = {}
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in all_samples)
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in all_samples)

    metrics["prefix_cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    metrics["avg_cached_tokens_per_sample"] = total_cached_tokens / num_samples
    return metrics


def _compute_reward_cat_metrics(args, all_samples: list[Sample]):
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}


def _compute_passrate_from_samples(args, all_samples: list[Sample]) -> dict[str, float]:
    """Compute pass@k metrics from samples using group_index for correct grouping.

    Unlike the trainer-side log_passrate (which assumed a flat reward array with
    contiguous groups of n_samples_per_prompt), this groups samples by their
    group_index field and computes pass@k over complete groups only. This is
    robust to filtering that may remove individual samples from a group —
    incomplete groups are excluded from the estimate rather than skewing it
    or crashing the reshape.

    Called on the rollout side (before convert_samples_to_train_data), so
    normally all samples are present and every group is complete.
    """
    group_size = args.n_samples_per_prompt
    if group_size <= 1:
        return {}

    groups = group_by(all_samples, lambda s: s.group_index)
    completed_groups = [g for g in groups.values() if len(g) == group_size]
    if len(completed_groups) < len(groups):
        logger.warning(
            f"pass@k: excluding {len(groups) - len(completed_groups)}/{len(groups)} incomplete "
            f"groups (fewer than n_samples_per_prompt={group_size} samples)."
        )
    if not completed_groups:
        return {}

    flat_rewards = [sample.get_reward_value(args) for group in completed_groups for sample in group]

    return compute_pass_rate(
        flat_rewards=flat_rewards,
        group_size=group_size,
    )
