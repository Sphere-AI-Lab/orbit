"""Search-R1 rollout for Orbit PPO.

The model alternates between generated assistant spans and retrieval
observations. Assistant spans are trainable; retrieval observations are part of
the next prompt but are masked out of the policy loss.
"""

import argparse
import asyncio
import logging
import re
from copy import deepcopy
from typing import Any

from miles_plugins.search_r1.qa_em_format import compute_score_em
from miles.utils.types import Sample

logger = logging.getLogger(__name__)

_SEMAPHORES: dict[int, asyncio.Semaphore] = {}


def _get_semaphore(concurrency: int) -> asyncio.Semaphore:
    if concurrency not in _SEMAPHORES:
        _SEMAPHORES[concurrency] = asyncio.Semaphore(concurrency)
    return _SEMAPHORES[concurrency]


def passages_to_string(retrieval_result: list[dict[str, Any]]) -> str:
    references = []
    for idx, doc_item in enumerate(retrieval_result):
        content = doc_item.get("document", {}).get("contents", "")
        title, _, text = content.partition("\n")
        references.append(f"Doc {idx + 1}(Title: {title}) {text}")
    return "\n".join(references) + ("\n" if references else "")


async def search(args, query: str) -> str:
    backend = args.search_r1_backend
    if backend == "local":
        from miles_plugins.search_r1.local_search_server import local_search

        result = await local_search(
            args.search_r1_local_url,
            query,
            args.search_r1_topk,
            timeout=args.search_r1_timeout,
            proxy=getattr(args, "search_r1_proxy", None),
        )
    elif backend == "google":
        raise NotImplementedError(
            "Search-R1 google backend is not bundled with Orbit yet. "
            "Use --search-r1-backend local or provide a local retrieval adapter."
        )
    else:
        raise ValueError(f"Unknown Search-R1 backend: {backend!r}")

    return passages_to_string(result)


def postprocess_responses(resp: str) -> str:
    """Trim to a complete Search-R1 action when rollout logprobs are disabled."""
    if "</search>" in resp:
        return resp.split("</search>")[0] + "</search>"
    if "</answer>" in resp:
        return resp.split("</answer>")[0] + "</answer>"
    return resp


def postprocess_predictions(prediction: str) -> tuple[str | None, str]:
    match = re.search(r"<(search|answer)>(.*?)</\1>", prediction, re.DOTALL)
    if not match:
        return None, ""
    return match.group(1), match.group(2).strip()


async def execute_prediction(args, prediction: str) -> tuple[str, bool]:
    action, content = postprocess_predictions(prediction)

    if action == "search":
        async with _get_semaphore(args.search_r1_concurrency):
            search_results = await search(args, content)
        return f"\n\n<information>{search_results.strip()}</information>\n\n", False

    if action == "answer":
        return "", True

    return (
        "\nMy previous action is invalid. If I want to search, I should put the query between "
        "<search> and </search>. If I want to give the final answer, I should put the answer "
        "between <answer> and </answer>. Let me try again.\n",
        False,
    )


def append_environment_observation(sample: Sample, observation: str, tokenizer, *, has_rollout_logprobs: bool) -> None:
    if not observation:
        return

    obs_token_ids = tokenizer.encode(observation, add_special_tokens=False)
    sample.response += observation
    sample.tokens += obs_token_ids
    sample.response_length += len(obs_token_ids)

    if sample.loss_mask is None:
        sample.loss_mask = []
    sample.loss_mask += [0] * len(obs_token_ids)

    if has_rollout_logprobs:
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs += [0.0] * len(obs_token_ids)

    sample.validate()


def build_generation_payload(args, input_ids: list[int], sampling_params: dict, *, evaluation: bool = False):
    from miles.rollout.generate_utils.generate_endpoint_utils import (
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
    assert not args.partial_rollout, "Partial rollout is not supported for Search-R1."

    from miles.rollout.generate_utils.generate_endpoint_utils import (
        compute_prompt_ids_from_sample,
        should_request_rollout_logprobs,
        update_sample_from_response,
    )
    from miles.rollout.sglang_rollout import GenerateState
    from miles.utils.http_utils import post

    state = GenerateState(args)
    tokenizer = state.tokenizer
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    result = deepcopy(sample)
    prompt_token_ids = compute_prompt_ids_from_sample(state, result)
    result.tokens = list(prompt_token_ids)
    result.response = ""
    result.response_length = 0
    result.loss_mask = []
    result.rollout_log_probs = [] if should_request_rollout_logprobs(args, evaluation) else None
    result.metadata.setdefault("search_r1_backend", args.search_r1_backend)

    for turn_idx in range(args.search_r1_max_turns):
        payload, halt_status = build_generation_payload(args, result.tokens, sampling_params, evaluation=evaluation)
        if payload is None:
            result.status = halt_status
            break

        output = await post(url, payload)
        if payload.get("return_logprob") and "output_token_logprobs" not in output.get("meta_info", {}):
            raise RuntimeError("output_token_logprobs missing from SGLang response despite return_logprob=True")

        if not payload.get("return_logprob"):
            output = deepcopy(output)
            output["text"] = postprocess_responses(output["text"])
            output["output_ids"] = tokenizer.encode(output["text"], add_special_tokens=False)

        await update_sample_from_response(args, result, payload=payload, output=output, update_loss_mask=True)
        result.metadata["search_r1_turns"] = turn_idx + 1
        result.validate()

        finish_type = output["meta_info"]["finish_reason"]["type"]
        if finish_type in ("abort", "length"):
            break

        observation, done = await execute_prediction(args, output["text"])
        if done:
            break

        append_environment_observation(
            result,
            observation,
            tokenizer,
            has_rollout_logprobs=payload.get("return_logprob", False),
        )

    result.validate()
    return result


def _ground_truth_from_label(label: Any) -> dict:
    if isinstance(label, dict) and "ground_truth" in label:
        return label["ground_truth"]
    if isinstance(label, dict) and "target" in label:
        return label
    raise ValueError(f"Search-R1 labels must contain ground_truth.target or target, got: {label!r}")


def _score_sample(args, sample: Sample) -> float:
    if not isinstance(sample, Sample):
        raise TypeError("sample must be an miles.utils.types.Sample")

    return compute_score_em(
        solution_str=sample.prompt + sample.response,
        ground_truth=_ground_truth_from_label(sample.label),
        format_score=args.search_r1_format_score,
    )


async def reward_func(args, sample: Sample | list[Sample], **kwargs) -> float | list[float]:
    if isinstance(sample, list):
        return [_score_sample(args, item) for item in sample]
    return _score_sample(args, sample)


def _add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--search-r1-backend", choices=["local", "google"], default="local")
    parser.add_argument("--search-r1-local-url", default="http://127.0.0.1:8000/retrieve")
    parser.add_argument("--search-r1-proxy", default=None)
    parser.add_argument("--search-r1-timeout", type=int, default=60)
    parser.add_argument("--search-r1-topk", type=int, default=3)
    parser.add_argument("--search-r1-max-turns", type=int, default=2)
    parser.add_argument("--search-r1-concurrency", type=int, default=256)
    parser.add_argument("--search-r1-format-score", type=float, default=0.2)


generate.add_arguments = _add_arguments
