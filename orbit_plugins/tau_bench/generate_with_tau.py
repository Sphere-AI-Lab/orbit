"""Tau-bench raw-generate compatibility rollout for Orbit PPO."""

import argparse
import inspect
import logging
import os
from copy import deepcopy
from typing import Any

from orbit_plugins.tau_bench.openai_tool_adapter import create_openai_adapter
from orbit.utils.types import Sample

logger = logging.getLogger(__name__)

TOOL_INSTRUCTION = (
    " At each turn, you are allowed to call one or no function to assist "
    "with task execution using <tools></tools> XML tags.\n"
    "YOU MUST EXECUTE TOOLS TO MAKE ANY MODIFICATIONS OR CANCELLATIONS. "
    "Each tool call leads to a message returned by the system.\n"
    "NEVER confirm execution to the user without seeing confirmation "
    "from the tool system.\n"
)

_PROVIDER_KEY_ENV = {
    "gemini": "GEMINI_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "mock": None,
}


def _reformulate_tool_call(text: str) -> str:
    return text.replace("You may call one or more functions to assist with the user query.", TOOL_INSTRUCTION)


def _model_text(response_text: str) -> str:
    if response_text.endswith("<|im_end|>"):
        return response_text[: -len("<|im_end|>")]
    return response_text


def _model_dump(value) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    return {}


def _task_index_from_sample(sample: Sample) -> int:
    if isinstance(sample.metadata, dict) and "index" in sample.metadata:
        return int(sample.metadata["index"])
    return int(sample.prompt)


def _validate_user_model_key(args) -> None:
    provider = args.tau_bench_user_model_provider
    key_env = _PROVIDER_KEY_ENV.get(provider, f"{provider.upper()}_API_KEY")
    if key_env and not os.environ.get(key_env):
        raise RuntimeError(
            f"Tau-bench user simulator requires {key_env} for provider {provider!r}. "
            "Export it before launching the run."
        )


def _load_env(args, task_index: int):
    from tau_bench.envs import get_env

    _validate_user_model_key(args)
    return get_env(
        env_name=args.tau_bench_env,
        user_strategy=args.tau_bench_user_strategy,
        user_model=args.tau_bench_user_model,
        user_provider=args.tau_bench_user_model_provider,
        task_split=args.tau_bench_task_split,
        task_index=task_index,
    )


def _is_respond_action(action) -> bool:
    from tau_bench.agents.tool_calling_agent import RESPOND_ACTION_NAME

    return action.name == RESPOND_ACTION_NAME


async def _step_env(env, action):
    response = env.step(action)
    if inspect.isawaitable(response):
        return await response
    return response


def _reset_env(env, task_index: int) -> tuple[str, dict[str, Any]]:
    reset_res = env.reset(task_index=task_index)
    return reset_res.observation, _model_dump(reset_res.info)


def _render_messages(tokenizer, messages: list[dict[str, Any]], tools_info: list[dict[str, Any]]) -> tuple[str, list[int]]:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools_info)
    text = _reformulate_tool_call(text)
    return text, tokenizer.encode(text, add_special_tokens=False)


def append_environment_delta(
    sample: Sample,
    next_input_ids: list[int],
    tokenizer,
    *,
    has_rollout_logprobs: bool,
) -> bool:
    """Append non-trainable env/tool/user tokens needed for the next model call.

    Returns False if the newly rendered prompt is not an append-only extension
    of the current sample tokens.
    """
    if next_input_ids[: len(sample.tokens)] != sample.tokens:
        sample.metadata["tau_bench_token_mismatch"] = {
            "current_tokens": len(sample.tokens),
            "next_input_tokens": len(next_input_ids),
        }
        return False

    env_token_ids = next_input_ids[len(sample.tokens) :]
    if not env_token_ids:
        return True

    sample.tokens += env_token_ids
    sample.response_length += len(env_token_ids)
    if sample.loss_mask is None:
        sample.loss_mask = []
    sample.loss_mask += [0] * len(env_token_ids)
    if has_rollout_logprobs:
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs += [0.0] * len(env_token_ids)
    if hasattr(tokenizer, "decode"):
        sample.response += tokenizer.decode(env_token_ids)
    sample.validate()
    return True


def build_generation_payload(args, input_ids: list[int], sampling_params: dict, *, evaluation: bool = False):
    from orbit.rollout.generate_utils.generate_endpoint_utils import (
        compute_request_payload,
        should_request_rollout_logprobs,
    )

    return compute_request_payload(
        args,
        input_ids,
        sampling_params,
        return_logprob=should_request_rollout_logprobs(args, evaluation),
    )


async def generate(args, sample: Sample, sampling_params: dict, evaluation: bool = False) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for Tau-bench."

    from orbit.rollout.generate_utils.generate_endpoint_utils import (
        should_request_rollout_logprobs,
        update_sample_from_response,
    )
    from orbit.rollout.sglang_rollout import GenerateState
    from orbit.utils.http_utils import post

    task_index = _task_index_from_sample(sample)
    env = _load_env(args, task_index)
    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    observation, info = _reset_env(env, task_index)
    messages = [{"role": "system", "content": env.wiki}, {"role": "user", "content": observation}]
    prompt_text, prompt_token_ids = _render_messages(tokenizer, messages, env.tools_info)

    result = deepcopy(sample)
    result.index = task_index
    result.prompt = prompt_text
    result.tokens = list(prompt_token_ids)
    result.response = ""
    result.response_length = 0
    result.loss_mask = []
    result.rollout_log_probs = [] if should_request_rollout_logprobs(args, evaluation) else None
    result.reward = 0.0
    result.metadata.update(
        {
            "tau_bench_env": args.tau_bench_env,
            "tau_bench_task_split": args.tau_bench_task_split,
            "tau_bench_task_index": task_index,
            **info,
        }
    )

    adapter = create_openai_adapter(env.tools_info, parser_type=args.tau_bench_tool_parser)
    total_reward = 0.0

    for step_idx in range(args.tau_bench_agent_max_steps):
        _, input_ids = _render_messages(tokenizer, messages, env.tools_info)
        if input_ids != result.tokens:
            result.metadata["tau_bench_token_mismatch"] = {
                "current_tokens": len(result.tokens),
                "input_tokens": len(input_ids),
            }
            result.status = Sample.Status.ABORTED
            break

        payload, halt_status = build_generation_payload(args, input_ids, sampling_params, evaluation=evaluation)
        if payload is None:
            result.status = halt_status
            break

        output = await post(url, payload)
        if payload.get("return_logprob") and "output_token_logprobs" not in output.get("meta_info", {}):
            raise RuntimeError("output_token_logprobs missing from SGLang response despite return_logprob=True")
        if not payload.get("return_logprob"):
            output = deepcopy(output)
            output["output_ids"] = tokenizer.encode(output["text"], add_special_tokens=False)

        await update_sample_from_response(args, result, payload=payload, output=output, update_loss_mask=True)
        result.metadata["tau_bench_steps"] = step_idx + 1

        finish_type = output["meta_info"]["finish_reason"]["type"]
        if finish_type == "abort":
            result.status = Sample.Status.ABORTED
            break
        if finish_type == "length":
            result.status = Sample.Status.TRUNCATED
            break

        response_text = _model_text(output["text"])
        parsed = adapter.parse_response_to_openai_format(response_text)
        if not parsed["success"]:
            result.metadata["tau_bench_parse_error"] = parsed.get("error")
            result.status = Sample.Status.ABORTED
            break

        messages.append({"role": "assistant", "content": response_text})
        action = adapter.call_to_action(parsed["parsed_result"]["calls"], parsed["parsed_result"]["normal_text"])

        try:
            env_response = await _step_env(env, action)
        except Exception as exc:
            logger.warning("Tau-bench environment step failed: %s", exc)
            result.metadata["tau_bench_env_error"] = str(exc)
            result.status = Sample.Status.ABORTED
            break

        total_reward = float(env_response.reward)
        result.reward = total_reward
        result.metadata.update(_model_dump(env_response.info))

        if _is_respond_action(action):
            messages.append({"role": "user", "content": env_response.observation})
        else:
            messages.append({"role": "tool", "name": action.name, "content": env_response.observation})

        _, next_input_ids = _render_messages(tokenizer, messages, env.tools_info)
        if not append_environment_delta(
            result,
            next_input_ids,
            tokenizer,
            has_rollout_logprobs=payload.get("return_logprob", False),
        ):
            result.status = Sample.Status.ABORTED
            break

        if env_response.done:
            result.status = Sample.Status.COMPLETED
            break
    else:
        result.status = Sample.Status.TRUNCATED

    result.reward = total_reward
    result.validate()
    return result


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--tau-bench-env", default="retail")
    parser.add_argument("--tau-bench-task-split", default="train")
    parser.add_argument("--tau-bench-user-strategy", default="llm")
    parser.add_argument("--tau-bench-user-model-provider", default=os.environ.get("TAU_USER_MODEL_PROVIDER", "gemini"))
    parser.add_argument("--tau-bench-user-model", default=os.environ.get("TAU_USER_MODEL", "gemini-2.5-flash-lite"))
    parser.add_argument("--tau-bench-agent-max-steps", type=int, default=30)
    parser.add_argument("--tau-bench-tool-parser", default="qwen25")


generate.add_arguments = _add_arguments
