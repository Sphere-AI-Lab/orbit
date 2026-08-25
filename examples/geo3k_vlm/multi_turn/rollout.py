from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch
from examples.geo3k_vlm.multi_turn.base_env import BaseInteractionEnv
from examples.model_response_trace_viewer.response_log import RESPONSE_TURNS_KEY

# When executed as a module: python -m examples.geo3k_vlm.multi_turn.rollout
from miles.rollout.sglang_rollout import GenerateState
from miles.utils.http_utils import post
from miles.utils.processing_utils import encode_image_for_rollout_engine
from miles.utils.types import Sample

DEFAULT_ENV_MODULE = "examples.geo3k_vlm.multi_turn.env_geo3k"

# Dummy messages used for calculating trim length in chat template encoding
DUMMY_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


def _load_env_module(env_path: str | None):
    """Load the interaction environment module from a module path or a file path."""
    target = env_path or DEFAULT_ENV_MODULE
    module_path = Path(target)
    if module_path.suffix == ".py" and module_path.exists():
        spec = importlib.util.spec_from_file_location(f"rollout_env_{module_path.stem}", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import environment module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(target)


def _build_env(env_module, sample: Sample, args: Any):
    """Instantiate the interaction environment using the provided module."""
    build_fn = env_module.build_env
    if not callable(build_fn):
        raise ValueError("Environment module must expose a callable `build_env(sample, args)`.")
    try:
        return build_fn(sample=sample, args=args)
    except TypeError:
        # Fallback to positional signature
        return build_fn(sample, args)


def _encode_observation_for_generation(
    tokenizer,
    processor,
    message: dict,
    metadata: dict | None,
    apply_chat_template: bool,
    apply_chat_template_kwargs: dict | None,
):
    """
    Encode a single observation turn that may include images/videos in the content list.
    Trim out the system/tool preamble added by the chat template so only the observation tokens remain.
    """
    tools = metadata.get("tools") if metadata else None
    apply_kwargs = apply_chat_template_kwargs or {}

    trim_length = 0

    if apply_chat_template:
        dummy_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES,
            tools=tools,
            tokenize=False,
            add_generation_prompt=False,
            **apply_kwargs,
        )
        formatted_prompt = tokenizer.apply_chat_template(
            DUMMY_MESSAGES + [message],
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **apply_kwargs,
        )
        trim_length = len(tokenizer.encode(dummy_prompt, add_special_tokens=False))
    else:
        formatted_prompt = [message]

    compact_prompt_ids = None
    if isinstance(formatted_prompt, str):
        compact_prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)

    multimodal_inputs = None
    multimodal_train_inputs = None
    if processor:
        # Convert content-embedded images/videos into multimodal inputs for the processor.
        from qwen_vl_utils import process_vision_info

        images, videos = process_vision_info([message])
        multimodal_inputs = {"images": images, "videos": videos}
        processor_output = processor(text=formatted_prompt, **multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]
        multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = tokenizer.encode(formatted_prompt, add_special_tokens=False)
        compact_prompt_ids = list(prompt_ids)

    if trim_length:
        prompt_ids = prompt_ids[trim_length:]
        if compact_prompt_ids is not None:
            compact_prompt_ids = compact_prompt_ids[trim_length:]

    image_data = []
    if multimodal_inputs and multimodal_inputs.get("images"):
        image_data = [encode_image_for_rollout_engine(img) for img in multimodal_inputs["images"]]
    if compact_prompt_ids is None:
        if image_data:
            raise ValueError(
                "Multimodal observation turns require a rendered string prompt so "
                "SGLang can expand compact media placeholders without retokenizing "
                "prior sampled actions."
            )
        compact_prompt_ids = list(prompt_ids)
    return prompt_ids, compact_prompt_ids, image_data, multimodal_inputs, multimodal_train_inputs


def _merge_multimodal_train_inputs(chunks: list[dict | None]) -> dict | None:
    """
    Merge per-turn multimodal_train_inputs with a single concat per key.

    Note: Only torch.Tensor values are merged; non-tensor fields are ignored by design.
    """
    if not chunks:
        return None

    values_by_key = {}
    for chunk in chunks:
        if not chunk:
            continue
        for key, val in chunk.items():
            if val is None:
                continue
            values_by_key.setdefault(key, []).append(val)

    merged = {}
    for key, values in values_by_key.items():
        if all(isinstance(v, torch.Tensor) for v in values):
            merged[key] = torch.cat(values, dim=0)

    return merged


def _initialize_resources(args: Any, sample: Sample):
    env_module = _load_env_module(args.rollout_interaction_env_path)
    max_turns = args.max_turns
    if max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path in the custom config file.")
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
    sample.metadata = sample.metadata or {}
    env = _build_env(env_module, sample, args)
    config = {"max_turns": max_turns}
    return env, env_module, config, state, url


def _prepare_initial_inputs(sample: Sample, processor, tokenizer):
    if processor:
        if not isinstance(sample.prompt, str) or not sample.prompt:
            raise ValueError(
                "Multimodal multi-turn generation requires a non-empty rendered string prompt "
                "to preserve sampled action IDs across turns."
            )
        compact_prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
        processor_output = processor(text=sample.prompt, **(sample.multimodal_inputs or {}))
        prompt_ids = processor_output["input_ids"][0]
        sample.multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = tokenizer.encode(sample.prompt, add_special_tokens=False)
        compact_prompt_ids = list(prompt_ids)

    image_data = []
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = [encode_image_for_rollout_engine(img) for img in sample.multimodal_inputs["images"]]
    return prompt_ids, compact_prompt_ids, image_data, sample.multimodal_train_inputs


def _prepare_start_state(sample: Sample, state, args: Any, sampling_params: dict):
    prompt_ids, compact_prompt_ids, image_data, init_mm_train = _prepare_initial_inputs(
        sample, state.processor, state.tokenizer
    )
    current_image_data = image_data
    multimodal_train_inputs_buffer: list[dict | None] = []
    if init_mm_train:
        multimodal_train_inputs_buffer.append(init_mm_train)

    if not sample.tokens:
        sample.tokens = list(prompt_ids)
    response_tokens: list[int] = sample.tokens[len(prompt_ids) :] if len(sample.tokens) >= len(prompt_ids) else []
    generation_tokens = [*compact_prompt_ids, *response_tokens]
    sample.loss_mask = sample.loss_mask or []
    sample.rollout_log_probs = sample.rollout_log_probs or []
    sample.response_length = len(response_tokens)

    remaining_budgets = []
    if sampling_params.get("max_new_tokens") is not None:
        remaining_budgets.append(sampling_params["max_new_tokens"] - len(response_tokens))
    if args.rollout_max_context_len is not None:
        remaining_budgets.append(args.rollout_max_context_len - len(sample.tokens))
    budget = min(remaining_budgets) if remaining_budgets else None
    return current_image_data, response_tokens, budget, multimodal_train_inputs_buffer, generation_tokens


async def _run_inference_step(url: str, tokens: list[int], sampling_params: dict, image_data, tokenizer):
    payload = {
        "input_ids": tokens,
        "sampling_params": sampling_params,
        "return_logprob": True,
    }
    if image_data:
        payload["image_data"] = image_data

    output = await post(url, payload)
    response_text = output["text"]
    if "output_token_logprobs" in output["meta_info"]:
        new_tokens = [item[1] for item in output["meta_info"]["output_token_logprobs"]]
        new_log_probs = [item[0] for item in output["meta_info"]["output_token_logprobs"]]
    else:
        new_tokens, new_log_probs = [], []
    finish_type = output["meta_info"]["finish_reason"]["type"]
    return response_text, new_tokens, new_log_probs, finish_type, output["meta_info"]


def _validate_multimodal_generation_alignment(meta_info: dict, expected_prompt_tokens: int, image_data) -> None:
    """Require SGLang's rebuilt multimodal prefix to match the trainer token stream."""
    if not image_data:
        return

    actual_prompt_tokens = meta_info.get("prompt_tokens")
    if actual_prompt_tokens is None:
        raise RuntimeError(
            "Multimodal multi-turn generation requires SGLang to return meta_info.prompt_tokens "
            "for exact-action alignment validation."
        )
    if int(actual_prompt_tokens) != expected_prompt_tokens:
        raise RuntimeError(
            "Multimodal multi-turn generation token alignment mismatch: "
            f"trainer_expected={expected_prompt_tokens}, sglang_prompt_tokens={actual_prompt_tokens}. "
            "Send compact media placeholders with exact sampled suffix IDs and keep "
            "SGLANG_MM_AVOID_RETOKENIZE=1."
        )


def _record_response_turn(
    args: Any,
    sample: Sample,
    entry: dict[str, Any],
) -> None:
    """Append a turn to the viewer's record, if any trace output is enabled.

    Turns live on ``Sample.metadata`` so this stays a customization-layer
    feature: nothing in ``miles/`` has to know the viewer exists.
    """
    if (
        getattr(args, "save_model_response_log", None) is None
        and getattr(args, "save_model_response_trace_dir", None) is None
    ):
        return
    sample.metadata.setdefault(RESPONSE_TURNS_KEY, []).append(entry)


def _process_env_step(env: BaseInteractionEnv, response_text: str, tokenizer, processor, args, sample_metadata):
    observation, done, _ = env.step(response_text)
    if done:
        return None, None, None, None, None, None, True

    next_user_message = env.format_observation(observation)
    (
        obs_prompt_ids,
        compact_obs_prompt_ids,
        obs_image_data,
        obs_multimodal_inputs,
        obs_multimodal_train_inputs,
    ) = _encode_observation_for_generation(
        tokenizer,
        processor,
        next_user_message,
        sample_metadata,
        args.apply_chat_template,
        args.apply_chat_template_kwargs,
    )

    bos_id = tokenizer.bos_token_id
    if bos_id is not None and obs_prompt_ids and obs_prompt_ids[0] == bos_id:
        obs_prompt_ids = obs_prompt_ids[1:]
    if bos_id is not None and compact_obs_prompt_ids and compact_obs_prompt_ids[0] == bos_id:
        compact_obs_prompt_ids = compact_obs_prompt_ids[1:]

    environment_turn = {
        "role": "environment",
        "model_input_role": next_user_message.get("role", "user"),
        "content": next_user_message.get("content", ""),
    }

    return (
        obs_prompt_ids,
        compact_obs_prompt_ids,
        obs_image_data,
        obs_multimodal_inputs,
        obs_multimodal_train_inputs,
        environment_turn,
        False,
    )


def _append_to_sample(
    sample: Sample,
    response_tokens: list[int],
    tokens_to_add: list[int],
    logprobs: list[float],
    loss_mask_val: int,
) -> None:
    sample.tokens.extend(tokens_to_add)
    response_tokens.extend(tokens_to_add)
    sample.loss_mask.extend([loss_mask_val] * len(tokens_to_add))
    sample.rollout_log_probs.extend(logprobs)
    sample.response_length = len(response_tokens)


def _update_multimodal_state(
    sample: Sample,
    current_image_data,
    obs_image_data,
    obs_multimodal_inputs,
    obs_multimodal_train_inputs,
    multimodal_train_inputs_buffer: list[dict | None],
):
    if obs_image_data:
        current_image_data = (current_image_data or []) + obs_image_data

    if obs_multimodal_inputs:
        if not sample.multimodal_inputs:
            sample.multimodal_inputs = obs_multimodal_inputs
        elif isinstance(sample.multimodal_inputs, dict) and isinstance(obs_multimodal_inputs, dict):
            for key, val in obs_multimodal_inputs.items():
                if val is None:
                    continue
                if (
                    key in sample.multimodal_inputs
                    and isinstance(sample.multimodal_inputs[key], list)
                    and isinstance(val, list)
                ):
                    sample.multimodal_inputs[key].extend(val)
        else:
            sample.multimodal_inputs = obs_multimodal_inputs

    if obs_multimodal_train_inputs:
        multimodal_train_inputs_buffer.append(obs_multimodal_train_inputs)

    return current_image_data


def _should_stop_on_finish(sample: Sample, finish_type: str) -> bool:
    match finish_type:
        case "length":
            sample.status = Sample.Status.TRUNCATED
            return True
        case "abort":
            sample.status = Sample.Status.ABORTED
            return True
    return False


def _update_budget(budget, consumed: int):
    if budget is None:
        return None
    return budget - consumed


def _finalize_sample(sample: Sample, tokenizer, response_tokens, multimodal_train_inputs_buffer):
    sample.multimodal_train_inputs = _merge_multimodal_train_inputs(multimodal_train_inputs_buffer)
    sample.response = tokenizer.decode(response_tokens, skip_special_tokens=False)
    sample.response_length = len(response_tokens)
    if sample.status is None:
        sample.status = Sample.Status.COMPLETED
    return sample


async def generate(args: Any, sample: Sample, sampling_params) -> Sample:
    """Custom multi-turn rollout that interacts with a pluggable environment."""
    assert not args.partial_rollout, "Partial rollout is not supported for interaction rollouts."

    sample.capture_multimodal_inputs_for_retry()
    sample.metadata = sample.metadata or {}
    # metadata survives reset_for_retry, so drop any turns the aborted attempt left.
    sample.metadata.pop(RESPONSE_TURNS_KEY, None)
    env, env_module, config, state, url = _initialize_resources(args, sample)
    sampling_params = sampling_params.copy()
    (
        current_image_data,
        response_tokens,
        budget,
        multimodal_train_inputs_buffer,
        generation_tokens,
    ) = _prepare_start_state(sample, state, args, sampling_params)
    try:
        env.reset()
        if budget is not None and budget <= 0:
            sample.status = Sample.Status.TRUNCATED
            return sample

        cur_sampling_params = sampling_params
        for turn_idx in range(config["max_turns"]):
            if budget is not None:
                cur_sampling_params["max_new_tokens"] = budget

            expected_prompt_tokens = len(sample.tokens)
            response_text, new_response_tokens, new_response_log_probs, finish_type, meta_info = (
                await _run_inference_step(
                    url,
                    generation_tokens,
                    cur_sampling_params,
                    current_image_data,
                    state.tokenizer,
                )
            )
            _validate_multimodal_generation_alignment(
                meta_info,
                expected_prompt_tokens,
                current_image_data,
            )
            _record_response_turn(
                args,
                sample,
                {
                    "turn": turn_idx + 1,
                    "role": "assistant",
                    "content": response_text,
                    "finish_reason": finish_type,
                    "weight_version": meta_info.get("weight_version"),
                },
            )
            # Miles' multi-turn logger consumes this top-level field after
            # rollout conversion. It is telemetry only and does not affect
            # rewards, masks, scoring, or the trainer objective.
            sample.metadata["round_number"] = turn_idx + 1
            # Record this generation call's engine weight version. SGLang issues one call
            # per turn, so we append each one (weight_versions is a list); the min across
            # turns (Sample.oldest_weight_version) is what the fully-async staleness filter
            # reads (miles/rollout/fully_async_data_buffer.py). Without this the list
            # stays empty -> oldest_weight_version is None -> the staleness filter silently
            # never fires. We intentionally do NOT call the full Sample.update_from_meta_info
            # here: its finish_reason->status mapping (incl. "stop" -> COMPLETED) is
            # single-turn semantics and would fight this rollout's own multi-turn status
            # state machine (_should_stop_on_finish / _finalize_sample).
            if "weight_version" in meta_info:
                sample.weight_versions.append(meta_info["weight_version"])
            _append_to_sample(sample, response_tokens, new_response_tokens, new_response_log_probs, loss_mask_val=1)
            generation_tokens.extend(new_response_tokens)
            budget = _update_budget(budget, len(new_response_tokens))

            if _should_stop_on_finish(sample, finish_type):
                break
            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                break

            (
                obs_prompt_ids,
                compact_obs_prompt_ids,
                obs_image_data,
                obs_multimodal_inputs,
                obs_multimodal_train_inputs,
                environment_turn,
                done,
            ) = _process_env_step(env, response_text, state.tokenizer, state.processor, args, sample.metadata)
            if done:
                sample.status = Sample.Status.COMPLETED
                break

            if budget is not None and len(obs_prompt_ids) > budget:
                # Keep environment observations atomic instead of exceeding the
                # response/context cap or appending a partial tool result.
                sample.status = Sample.Status.TRUNCATED
                break

            obs_log_probs = [0.0] * len(obs_prompt_ids)
            _append_to_sample(sample, response_tokens, obs_prompt_ids, obs_log_probs, loss_mask_val=0)
            generation_tokens.extend(compact_obs_prompt_ids)
            budget = _update_budget(budget, len(obs_prompt_ids))

            current_image_data = _update_multimodal_state(
                sample,
                current_image_data,
                obs_image_data,
                obs_multimodal_inputs,
                obs_multimodal_train_inputs,
                multimodal_train_inputs_buffer,
            )

            # Only record the observation the model will actually be shown: if this
            # is the last turn, it never reaches the model and would be a phantom turn.
            will_generate_another_turn = turn_idx + 1 < config["max_turns"] and (budget is None or budget > 0)
            if will_generate_another_turn and environment_turn is not None:
                _record_response_turn(args, sample, {"turn": turn_idx + 1, **environment_turn})

            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                break
            if turn_idx + 1 >= config["max_turns"]:
                sample.status = Sample.Status.COMPLETED
                break

        return _finalize_sample(sample, state.tokenizer, response_tokens, multimodal_train_inputs_buffer)
    finally:
        try:
            env.close()
        except Exception:
            pass
