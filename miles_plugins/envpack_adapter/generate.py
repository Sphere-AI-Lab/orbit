"""Custom generate entrypoint for envpack-backed rollouts.

This is the IMP-Miles compatibility path:

- Miles still owns tokenization, processor packing, SGLang calls, logprobs,
  response-length accounting, sticky routing, and training Sample fields.
- Envpack owns environment reset/step/finalize, episode state, parser, rubric,
  reward, credit metadata, and real multimodal observation bytes.

The structure intentionally mirrors Miles' existing custom-generate contract;
the environment calls are swapped for the envpack session client.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import uuid
from typing import Any

from miles.backends.megatron_utils.lora_utils import LORA_ADAPTER_NAME, is_lora_enabled
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.utils.http_utils import post
from miles.utils.processing_utils import encode_image_for_rollout_engine
from miles.utils.types import Sample

from miles_plugins.envpack_adapter.config import (
    EnvpackConfigError,
    load_envpack_config,
    validate_pool_env_config_overrides,
    validate_runtime_args,
)
from miles_plugins.envpack_adapter.renderer import RenderedObservation, observation_to_chat_message
from miles_plugins.envpack_adapter.runtime import get_client_bundle

logger = logging.getLogger(__name__)

_PLACEHOLDER_SYSTEM = {"role": "system", "content": "placeholder"}
_system_prompt_prefix_cache: dict[tuple[int, str], list[int]] = {}


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    config = load_envpack_config(input.args)
    validate_runtime_args(input.args, config)
    return await _generate_with_refill(input, config)


async def _generate_with_refill(input: GenerateFnInput, config) -> GenerateFnOutput:
    max_attempts = int(config.refill.max_attempts)
    if max_attempts == 1:
        return await _generate_once(
            input,
            config,
            episode_id_override=None,
            refill_attempt_index=0,
        )

    original = copy.deepcopy(input.sample)
    failed_attempts: list[dict[str, Any]] = []
    for attempt_index in range(max_attempts):
        if attempt_index > 0 and config.refill.backoff_s:
            await asyncio.sleep(float(config.refill.backoff_s) * (2 ** (attempt_index - 1)))
        _restore_sample(input.sample, original)
        episode_id_override = _episode_id_for_attempt(original, attempt_index)
        try:
            output = await _generate_once(
                input,
                config,
                episode_id_override=episode_id_override,
                refill_attempt_index=attempt_index,
            )
            _record_refill_success(output.samples, failed_attempts, attempt_index, max_attempts)
            return output
        except Exception as exc:
            retryable = _is_refillable_error(exc)
            failed_attempts.append(
                {
                    "attempt_index": attempt_index,
                    "episode_id": input.sample.session_id,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "retryable": retryable,
                }
            )
            if not retryable or attempt_index + 1 >= max_attempts:
                _restore_sample(input.sample, original)
                input.sample.metadata.setdefault("envpack", {})["refill"] = {
                    "failed_attempts": failed_attempts,
                    "max_attempts": max_attempts,
                    "exhausted": retryable,
                }
                raise
            logger.warning(
                "envpack trajectory attempt failed; refilling from reset "
                "(attempt %s/%s, sample_index=%s, group_index=%s, episode_id=%s): %s",
                attempt_index + 1,
                max_attempts,
                getattr(input.sample, "index", None),
                getattr(input.sample, "group_index", None),
                input.sample.session_id,
                exc,
                exc_info=True,
            )

    raise RuntimeError("unreachable envpack refill state")


async def _generate_once(
    input: GenerateFnInput,
    config,
    *,
    episode_id_override: str | None,
    refill_attempt_index: int,
) -> GenerateFnOutput:

    try:
        from envpack.core import ActorOutput, EpisodeCreateRequest, EpisodeStepRequest
    except Exception as exc:
        raise EnvpackConfigError(
            "envpack is not importable. Run `pip install -e thirdparty/envpack` "
            "or add the envpack repo to PYTHONPATH on every Miles worker."
        ) from exc

    state = input.state
    sample = input.sample
    args = state.args
    tokenizer = state.tokenizer
    processor = state.processor
    sampling_params = dict(input.sampling_params)
    apply_kwargs = getattr(args, "apply_chat_template_kwargs", {}) or {}

    meta = _envpack_meta(sample)
    bundle = get_client_bundle(config, args=args)
    pool_id = str(meta.get("pool_id") or config.pool_for_env(meta["env_name"]).resolved_pool_id)
    pool_config = config.pool_by_id(pool_id)
    env_config = bundle.env_config(pool_id)
    env_config.update(dict(meta.get("env_config") or {}))
    # Treat adapter pool.env_config as an explicit launch-time override so
    # render-only changes (e.g. sprite -> tiny) can reuse the same puzzle rows.
    env_config.update(
        validate_pool_env_config_overrides(
            meta["env_name"], pool_config.env_config, context="samples.jsonl generate path"
        )
    )

    episode_id = str(
        episode_id_override or meta.get("episode_id") or sample.session_id or f"envpack-{uuid.uuid4().hex}"
    )
    sample.session_id = episode_id
    create_request_id = f"create:{episode_id}"
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    created = None
    finalized = False
    try:
        created = await bundle.client.create_episode(
            EpisodeCreateRequest(
                env_name=meta["env_name"],
                request_id=create_request_id,
                episode_id=episode_id,
                example_id=meta.get("example_id"),
                env_config=env_config,
                seed=int(meta["seed"]),
                renderer_spec=_renderer_spec(args),
                metadata={
                    "pool_id": pool_id,
                    "source": "imp_miles_adapter",
                    "api": config.api,
                    "refill_attempt_index": refill_attempt_index,
                },
            )
        )
        rendered = observation_to_chat_message(created.observation)
        _check_env_uuid(meta, rendered, created.observation)
        _reject_unsupported_videos(rendered)
        system_msg = _system_message(created.prompt_bundle, content_blocks=processor is not None)
        init_messages = [system_msg, rendered.message] if system_msg is not None else [rendered.message]
        prompt_ids, init_mm_train, prompt_text = _encode_messages(
            processor=processor,
            tokenizer=tokenizer,
            messages=init_messages,
            images=rendered.images,
            apply_kwargs=apply_kwargs,
        )

        sample.prompt = prompt_text
        sample.tokens = list(prompt_ids)
        sample.multimodal_inputs = {"images": list(rendered.images)} if rendered.images else {}
        sample.loss_mask = []
        sample.rollout_log_probs = []
        response_tokens: list[int] = []
        image_data = [encode_image_for_rollout_engine(image) for image in rendered.images]
        mm_train_buf: list[dict | None] = [init_mm_train] if init_mm_train else []
        per_turn: list[dict[str, Any]] = []
        media_hashes: list[str] = list(rendered.media_hashes)
        artifacts = list(rendered.artifacts)
        model_generated_token_count = 0
        env_suffix_token_count = 0
        pending_obs_pil: Any = rendered.images[0] if rendered.images else None
        per_turn_obs_pils: list[Any] = []
        final_obs_pil: Any = None
        traj_success = False

        sys_prefix = _compute_system_prompt_prefix(processor or tokenizer, apply_kwargs)
        sys_prefix_len = len(sys_prefix)
        budget = _compute_budget(args, sampling_params, sample)
        max_turns, per_turn_cap = _resolve_rollout_budget(config, meta)
        _seed_legacy_debug_metadata(
            sample,
            meta=meta,
            env_config=env_config,
            max_turns=max_turns,
            response_length_per_turn=per_turn_cap,
        )

        for turn_idx in range(max_turns):
            cur_sampling_params = _turn_sampling_params(sampling_params, per_turn_cap, budget)
            max_new = cur_sampling_params.get("max_new_tokens")
            if max_new is not None and max_new <= 0:
                sample.status = Sample.Status.TRUNCATED
                break

            payload = {
                "input_ids": sample.tokens,
                "sampling_params": cur_sampling_params,
                "return_logprob": True,
            }
            if image_data:
                payload["image_data"] = image_data
            if is_lora_enabled(args):
                payload["lora_path"] = LORA_ADAPTER_NAME
            headers = _routing_headers(args, sample)
            output = await post(url, payload, headers=headers)
            new_tokens, new_log_probs = _tokens_and_log_probs(output)
            response_text_for_env = tokenizer.decode(new_tokens, skip_special_tokens=True)
            finish_type = output["meta_info"]["finish_reason"]["type"]

            _append_tokens(sample, response_tokens, new_tokens, new_log_probs, loss_mask_val=1)
            model_generated_token_count += len(new_tokens)
            budget = _update_budget(budget, len(new_tokens))
            if finish_type == "abort":
                sample.status = Sample.Status.ABORTED
                break

            step = await bundle.client.step_episode(
                EpisodeStepRequest(
                    request_id=f"step:{episode_id}:{created.turn_id + turn_idx}",
                    episode_id=episode_id,
                    expected_turn_id=created.turn_id + turn_idx,
                    actor_output=ActorOutput(
                        text=response_text_for_env,
                        metadata={
                            "source": "miles_adapter",
                            "finish_reason": output["meta_info"].get("finish_reason"),
                            "weight_version": output["meta_info"].get("weight_version"),
                            "turn_idx": turn_idx,
                        },
                    ),
                    action_text=response_text_for_env,
                )
            )
            _update_rollout_engine_metadata(args, sample, output["meta_info"])

            turn_info = dict(step.turn_trace.info if step.turn_trace is not None else {})
            turn_success = bool(turn_info.get("success", False))
            traj_success = bool(traj_success or turn_success)
            obs_rendered: RenderedObservation | None = None
            post_step_obs_pil = None
            if step.observation is not None:
                obs_rendered = observation_to_chat_message(step.observation)
                _reject_unsupported_videos(obs_rendered)
                post_step_obs_pil = obs_rendered.images[0] if obs_rendered.images else None
            per_turn_obs_pils.append(pending_obs_pil)
            per_turn.append(
                {
                    "turn": turn_idx,
                    "turn_id": step.turn_id,
                    "reward": float(step.reward_delta),
                    "reward_delta": float(step.reward_delta),
                    "done": bool(step.done),
                    "truncated": bool(step.truncated),
                    "status": step.status.value,
                    "success": turn_success,
                    "format_correct": bool(turn_info.get("format_correct", False)),
                    "n_actions_parsed": len(turn_info.get("actions") or []),
                    "action_is_valid": bool(
                        ((turn_info.get("metrics") or {}).get("turn_metrics") or {}).get("action_is_valid", False)
                    ),
                    "action_is_effective": bool(
                        ((turn_info.get("metrics") or {}).get("turn_metrics") or {}).get("action_is_effective", False)
                    ),
                    "turn_effect": dict(turn_info.get("turn_effect") or {}),
                }
            )

            is_last_turn = turn_idx + 1 >= max_turns
            if step.done or is_last_turn:
                if sample.status != Sample.Status.TRUNCATED:
                    sample.status = Sample.Status.COMPLETED
                final_obs_pil = post_step_obs_pil
                break
            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                final_obs_pil = post_step_obs_pil
                break

            if obs_rendered is None:
                sample.status = Sample.Status.FAILED
                break

            obs_ids, obs_mm_train, _ = _encode_messages(
                processor=processor,
                tokenizer=tokenizer,
                messages=[_PLACEHOLDER_SYSTEM, obs_rendered.message],
                images=obs_rendered.images,
                apply_kwargs=apply_kwargs,
            )
            obs_ids = obs_ids[sys_prefix_len:]
            _append_tokens(sample, response_tokens, obs_ids, [0.0] * len(obs_ids), loss_mask_val=0)
            env_suffix_token_count += len(obs_ids)
            budget = _update_budget(budget, len(obs_ids))

            if obs_rendered.images:
                image_data.extend(encode_image_for_rollout_engine(image) for image in obs_rendered.images)
                sample.multimodal_inputs.setdefault("images", []).extend(obs_rendered.images)
            if obs_mm_train:
                mm_train_buf.append(obs_mm_train)
            media_hashes.extend(obs_rendered.media_hashes)
            artifacts.extend(obs_rendered.artifacts)

            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                final_obs_pil = post_step_obs_pil
                break
            pending_obs_pil = post_step_obs_pil

        if sample.status == Sample.Status.PENDING:
            sample.status = Sample.Status.COMPLETED

        final = await bundle.client.finalize_episode(episode_id)
        finalized = True
        sample.reward = float(final.reward_report.reward)
        traj_success = bool(traj_success or _reward_report_success(final.reward_report))
        sample.response = tokenizer.decode(response_tokens, skip_special_tokens=False)
        sample.response_length = len(response_tokens)
        sample.multimodal_train_inputs = _merge_mm_train(mm_train_buf)
        sample.metadata.setdefault("envpack", {}).update(
            {
                "episode_id": episode_id,
                "owner": {
                    "env_name": created.owner.env_name,
                    "pool_id": created.owner.pool_id,
                    "instance_id": created.owner.instance_id,
                    "lease_id": created.owner.lease_id,
                },
                "prompt_bundle_hash": created.prompt_bundle.prompt_bundle_hash,
                "env_config": env_config,
                "final_status": final.status.value,
                "reward_report": _plain_reward_report(final.reward_report),
                "credit": _plain_credit(final.credit),
                "success": traj_success,
                "traj_success": traj_success,
                "trace_summary": dict(final.trace_summary),
                "per_turn": per_turn,
                "media_hashes": media_hashes,
                "artifacts": [_plain_artifact(artifact) for artifact in artifacts],
                "model_generated_token_count": model_generated_token_count,
                "env_suffix_token_count": env_suffix_token_count,
                "trainable_token_count": sum(sample.loss_mask or []),
                "refill_attempt_index": refill_attempt_index,
                "adapter": "miles_plugins.envpack_adapter.generate",
                "api": config.api,
                "solver_metrics": copy.deepcopy(meta.get("solver_metrics") or {}),
                "bucket_name": meta.get("bucket_name") or (meta.get("solver_metrics") or {}).get("bucket_name"),
            }
        )
        sample.metadata.setdefault("vagen", {}).update(
            {
                "env_name": meta["env_name"],
                "seed": int(meta["seed"]),
                "config": env_config,
                "max_turns": max_turns,
                "response_length_per_turn": per_turn_cap,
                "source_format": meta.get("source_format"),
                "drift_check_required": bool(meta.get("env_uuid")),
                "env_uuid": meta.get("env_uuid"),
                "env_uuid_kind": meta.get("env_uuid_kind"),
                "solver_metrics": copy.deepcopy(meta.get("solver_metrics") or {}),
                "bucket_name": meta.get("bucket_name") or (meta.get("solver_metrics") or {}).get("bucket_name"),
                "env_reward": sample.reward,
                "num_turns": len(per_turn),
                "traj_success": traj_success,
                "per_turn": per_turn,
                "_per_turn_obs_pils": per_turn_obs_pils,
                "_final_obs_pil": final_obs_pil,
                "adapter": "envpack",
            }
        )
        sample.validate()
        return GenerateFnOutput(samples=sample)
    except Exception:
        if sample.status == Sample.Status.PENDING:
            sample.status = Sample.Status.FAILED
            sample.reward = 0.0
        raise
    finally:
        if created is not None and not finalized:
            try:
                await bundle.client.cancel_episode(episode_id, reason="miles_adapter_cleanup")
            except Exception as exc:
                logger.warning("envpack cleanup failed for episode %s: %s", episode_id, exc)


def _envpack_meta(sample: Sample) -> dict[str, Any]:
    metadata = sample.metadata or {}
    meta = metadata.get("envpack")
    if not isinstance(meta, dict):
        raise EnvpackConfigError("sample.metadata['envpack'] is required for envpack generate")
    if not meta.get("env_name"):
        raise EnvpackConfigError("sample.metadata['envpack'].env_name is required")
    if meta.get("seed") is None:
        raise EnvpackConfigError("sample.metadata['envpack'].seed is required")
    return meta


def _restore_sample(sample: Sample, snapshot: Sample) -> None:
    sample.__dict__.clear()
    sample.__dict__.update(copy.deepcopy(snapshot.__dict__))


def _episode_id_for_attempt(sample: Sample, attempt_index: int) -> str | None:
    if attempt_index == 0:
        return None
    meta = _envpack_meta(sample)
    base = str(meta.get("episode_id") or sample.session_id or f"envpack-{sample.group_index}-{sample.index}")
    return f"{base}-refill-{attempt_index}-{uuid.uuid4().hex[:12]}"


def _record_refill_success(
    samples: Sample | list[Sample],
    failed_attempts: list[dict[str, Any]],
    attempt_index: int,
    max_attempts: int,
) -> None:
    targets = samples if isinstance(samples, list) else [samples]
    for sample in targets:
        if failed_attempts or attempt_index > 0:
            sample.metadata.setdefault("envpack", {})["refill"] = {
                "attempt_index": attempt_index,
                "max_attempts": max_attempts,
                "failed_attempts": failed_attempts,
            }


def _is_refillable_error(exc: Exception) -> bool:
    if isinstance(exc, EnvpackConfigError):
        return False
    if isinstance(
        exc, (AssertionError, AttributeError, ImportError, KeyError, ModuleNotFoundError, TypeError, ValueError)
    ):
        return False

    error = getattr(exc, "error", None)
    retryable = getattr(error, "retryable", None)
    if retryable is not None:
        return bool(retryable)
    status_code = getattr(exc, "status_code", None)
    if status_code in {0, 429, 500, 502, 503, 504}:
        return True

    exc_name = type(exc).__name__
    if exc_name in {"CapacityError", "EpisodeNotFoundError"}:
        return True

    return False


def _renderer_spec(args) -> dict[str, Any]:
    return {
        "owner": "miles",
        "hf_checkpoint": getattr(args, "hf_checkpoint", None),
        "chat_template_path": getattr(args, "chat_template_path", None),
        "apply_chat_template_kwargs": getattr(args, "apply_chat_template_kwargs", {}) or {},
    }


def _seed_legacy_debug_metadata(
    sample: Sample,
    *,
    meta: dict[str, Any],
    env_config: dict[str, Any],
    max_turns: int,
    response_length_per_turn,
) -> None:
    """Populate legacy debug metadata for Miles' current all-samples dumper."""

    sample.metadata.setdefault("vagen", {}).update(
        {
            "env_name": meta["env_name"],
            "seed": int(meta["seed"]),
            "config": env_config,
            "max_turns": max_turns,
            "response_length_per_turn": response_length_per_turn,
            "source_format": meta.get("source_format"),
            "drift_check_required": bool(meta.get("env_uuid")),
            "env_uuid": meta.get("env_uuid"),
            "env_uuid_kind": meta.get("env_uuid_kind"),
            "solver_metrics": copy.deepcopy(meta.get("solver_metrics") or {}),
            "bucket_name": meta.get("bucket_name") or (meta.get("solver_metrics") or {}).get("bucket_name"),
            "adapter": "envpack",
        }
    )


def _reward_report_success(report) -> bool:
    components = dict(getattr(report, "components", {}) or {})
    raw_reward = getattr(report, "raw_reward", None)
    verifier_outputs = dict(getattr(report, "verifier_outputs", {}) or {})
    candidates = (
        components.get("success"),
        components.get("success_reward"),
        verifier_outputs.get("success"),
        raw_reward.get("success") if isinstance(raw_reward, dict) else None,
    )
    return any(bool(value) for value in candidates)


def _system_message(prompt_bundle, *, content_blocks: bool) -> dict[str, Any] | None:
    if not prompt_bundle.system:
        return None
    content: str | list[dict[str, str]]
    if content_blocks:
        # Qwen-VL processors expect multimodal chat messages to keep system
        # prompts in the same content-block shape as observations.
        content = [{"type": "text", "text": prompt_bundle.system}]
    else:
        content = prompt_bundle.system
    return {"role": "system", "content": content}


def _encode_messages(
    *,
    processor,
    tokenizer,
    messages: list[dict[str, Any]],
    images: list[Any],
    apply_kwargs: dict[str, Any],
) -> tuple[list[int], dict | None, str]:
    renderer = processor or tokenizer
    if renderer is None:
        raise EnvpackConfigError("Miles state has neither processor nor tokenizer")
    text = renderer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        **apply_kwargs,
    )
    if processor is not None:
        processor_output = processor(text=[text], images=(images or None), return_tensors="pt")
        ids = processor_output.pop("input_ids").squeeze(0).tolist()
        mm_train = {
            key: value for key, value in processor_output.items() if key not in ("attention_mask", "mm_token_type_ids")
        } or None
        return ids, mm_train, text
    if images:
        raise EnvpackConfigError("envpack observation contains images but Miles state.processor is None")
    return tokenizer.encode(text, add_special_tokens=False), None, text


def _compute_system_prompt_prefix(renderer, apply_kwargs: dict[str, Any]) -> list[int]:
    cache_key = (id(renderer), repr(sorted(apply_kwargs.items())))
    cached = _system_prompt_prefix_cache.get(cache_key)
    if cached is not None:
        return cached
    text = renderer.apply_chat_template(
        [_PLACEHOLDER_SYSTEM],
        add_generation_prompt=False,
        tokenize=False,
        **apply_kwargs,
    )
    if hasattr(renderer, "tokenizer") and callable(renderer):
        ids = renderer(text=[text], return_tensors="pt")["input_ids"].squeeze(0).tolist()
    else:
        ids = renderer.encode(text, add_special_tokens=False)
    _system_prompt_prefix_cache[cache_key] = ids
    return ids


def _compute_budget(args, sampling_params: dict[str, Any], sample: Sample) -> int | None:
    if getattr(args, "rollout_max_context_len", None) is not None:
        return args.rollout_max_context_len - len(sample.tokens)
    if sampling_params.get("max_new_tokens") is not None:
        return sampling_params["max_new_tokens"] - len(sample.tokens)
    return None


def _resolve_rollout_budget(config, meta: dict[str, Any]) -> tuple[int, int]:
    raw_max_turns = config.rollout.max_turns
    if raw_max_turns is None:
        raw_max_turns = meta.get("max_turns")
    if raw_max_turns is None:
        raise EnvpackConfigError(
            "envpack_adapter.rollout.max_turns is required for envpack custom generate; "
            "server_train recipes set this through INTERACTION_BUDGET_ARGS"
        )
    max_turns = int(raw_max_turns)
    if max_turns < 1:
        raise EnvpackConfigError("envpack_adapter.rollout.max_turns must be >= 1")

    raw_per_turn_cap = config.rollout.response_length_per_turn
    if raw_per_turn_cap is None:
        raw_per_turn_cap = meta.get("response_length_per_turn")
    if raw_per_turn_cap is None:
        raise EnvpackConfigError(
            "envpack_adapter.rollout.response_length_per_turn is required for envpack custom generate; "
            "server_train recipes set this through INTERACTION_BUDGET_ARGS"
        )
    per_turn_cap = int(raw_per_turn_cap)
    if per_turn_cap < 1:
        raise EnvpackConfigError("envpack_adapter.rollout.response_length_per_turn must be >= 1")
    return max_turns, per_turn_cap


def _turn_sampling_params(sampling_params: dict[str, Any], per_turn_cap: int, budget: int | None) -> dict[str, Any]:
    cur = dict(sampling_params)
    max_new = cur.get("max_new_tokens")
    max_new = min(max_new, per_turn_cap) if max_new is not None else per_turn_cap
    if budget is not None:
        max_new = min(max_new, budget) if max_new is not None else budget
    if max_new is not None:
        cur["max_new_tokens"] = int(max_new)
    return cur


def _routing_headers(args, sample: Sample) -> dict[str, str] | None:
    if getattr(args, "sglang_router_policy", None) == "consistent_hashing" and sample.session_id:
        return {"X-SMG-Routing-Key": str(sample.session_id)}
    return None


def _tokens_and_log_probs(output: dict[str, Any]) -> tuple[list[int], list[float]]:
    token_log_probs = output["meta_info"].get("output_token_logprobs", []) or []
    return [item[1] for item in token_log_probs], [item[0] for item in token_log_probs]


def _update_rollout_engine_metadata(args, sample: Sample, meta_info: dict[str, Any]) -> None:
    if getattr(args, "sglang_speculative_algorithm", None):
        sample.spec_info.add(meta_info=meta_info)
    sample.prefix_cache_info.add(meta_info=meta_info)
    if "weight_version" in meta_info:
        sample.weight_versions.append(meta_info["weight_version"])


def _append_tokens(
    sample: Sample,
    response_tokens: list[int],
    tokens: list[int],
    log_probs: list[float],
    *,
    loss_mask_val: int,
) -> None:
    if len(tokens) != len(log_probs):
        raise RuntimeError(f"token/logprob length mismatch: {len(tokens)} != {len(log_probs)}")
    sample.tokens.extend(tokens)
    response_tokens.extend(tokens)
    sample.loss_mask.extend([loss_mask_val] * len(tokens))
    sample.rollout_log_probs.extend(log_probs)
    sample.response_length = len(response_tokens)


def _update_budget(budget: int | None, consumed: int) -> int | None:
    return None if budget is None else budget - consumed


def _merge_mm_train(buf: list[dict | None]) -> dict | None:
    if not buf:
        return None
    import torch

    values_by_key: dict[str, list[Any]] = {}
    for chunk in buf:
        if not chunk:
            continue
        for key, value in chunk.items():
            if value is not None:
                values_by_key.setdefault(key, []).append(value)
    merged: dict[str, Any] = {}
    for key, values in values_by_key.items():
        if all(isinstance(value, torch.Tensor) for value in values):
            merged[key] = torch.cat(values, dim=0)
    return merged or None


def _reject_unsupported_videos(rendered: RenderedObservation) -> None:
    if rendered.videos:
        raise EnvpackConfigError(
            "envpack produced video bytes, but IMP-Miles SGLang generate currently exposes image_data only"
        )


def _check_env_uuid(meta: dict[str, Any], rendered: RenderedObservation, observation) -> None:
    expected = meta.get("env_uuid")
    if not expected:
        return
    live_uuid = _live_observation_uuid(meta, rendered, observation)
    if live_uuid != str(expected):
        raise EnvpackConfigError(
            "envpack drift check failed for "
            f"env={meta.get('env_name')} seed={meta.get('seed')}: "
            f"live={live_uuid} != baked={expected}. Rebuild samples.jsonl "
            "or use the same envpack code/config that produced the dataset."
        )


def _live_observation_uuid(meta: dict[str, Any], rendered: RenderedObservation, observation) -> str:
    if meta.get("env_uuid_kind") == "sokoban_state":
        state_payload = getattr(observation, "state", None) or {}
        state_uuid = state_payload.get("env_uuid")
        if state_uuid:
            return str(state_uuid)
        try:
            from envpack.envs.sokoban.solver import sokoban_env_uuid
        except Exception as exc:
            raise EnvpackConfigError("sokoban_state env_uuid check requires envpack Sokoban helpers") from exc
        return sokoban_env_uuid(
            room_fixed=state_payload["room_fixed"],
            room_state=state_payload["room_state"],
            player_position=state_payload["player_position"],
            num_boxes=int(state_payload["num_boxes"]),
        )
    if rendered.media_hashes:
        return str(rendered.media_hashes[0])
    state_payload = getattr(observation, "state", None) or {}
    return hashlib.sha256(json.dumps(state_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _plain_reward_report(report) -> dict[str, Any]:
    return {
        "reward": float(report.reward),
        "raw_reward": report.raw_reward,
        "components": dict(report.components),
        "verifier_outputs": dict(report.verifier_outputs),
        "signals": [_plain_reward_signal(signal) for signal in getattr(report, "signals", [])],
    }


def _plain_credit(credit) -> dict[str, Any]:
    span_hints = getattr(credit, "span_hints", getattr(credit, "token_spans", []))
    return {
        "episode_reward": float(credit.episode_reward),
        "components": dict(credit.components),
        "per_turn": list(credit.per_turn),
        "span_hints": list(span_hints),
        "signals": [_plain_reward_signal(signal) for signal in getattr(credit, "signals", [])],
        "mode": credit.mode,
    }


def _plain_reward_signal(signal) -> dict[str, Any]:
    if isinstance(signal, dict):
        return dict(signal)
    return {
        "name": getattr(signal, "name", None),
        "value": getattr(signal, "value", None),
        "scope": getattr(signal, "scope", None),
        "kind": getattr(signal, "kind", None),
        "turn_id": getattr(signal, "turn_id", None),
        "target": dict(getattr(signal, "target", {}) or {}),
        "metadata": dict(getattr(signal, "metadata", {}) or {}),
    }


def _plain_artifact(artifact) -> dict[str, Any]:
    return {
        "uri": artifact.uri,
        "kind": artifact.kind,
        "mime_type": artifact.mime_type,
        "logical_role": artifact.logical_role,
        "sha256": artifact.sha256,
        "width": artifact.width,
        "height": artifact.height,
        "duration_s": artifact.duration_s,
        "metadata": dict(artifact.metadata),
    }
