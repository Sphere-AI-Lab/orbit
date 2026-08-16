import asyncio
import copy
import inspect
import json
import logging
import os
import time
import uuid
from argparse import Namespace
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import numpy as np
import pybase64
import sglang_router
from packaging.version import parse
from tqdm import tqdm

from orbit.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from orbit.rollout.generate_utils.prefill_logprobs import recompute_samples_rollout_logprobs_via_prefill
from orbit.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from orbit.rollout.generate_utils.generate_endpoint_utils import (
    attach_peft_request_payload,
    should_request_rollout_logprobs,
)
from orbit.utils import dumper_utils
from orbit.utils.async_utils import run
from orbit.utils.data import Dataset
from orbit.utils.eval_config import EvalDatasetConfig
from orbit.utils.http_utils import get, post
from orbit.utils.misc import SingletonMeta, load_function
from orbit.utils.processing_utils import (
    build_processor_kwargs,
    encode_image_for_rollout_engine,
    load_processor,
    load_tokenizer,
)
from orbit.utils.types import Sample

from .opd_scoring import local_scoring_enabled, opd_score_sample
from .rm_hub import async_rm, batched_async_rm

__all__ = ["generate_rollout", "get_model_url"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lightweight timing instrumentation. All counters are process-local; the
# eval/rollout coroutines run on a single rank so this is safe.
# ---------------------------------------------------------------------------


class _PhaseStats:
    __slots__ = (
        "name",
        "tokenize_s",
        "http_post_s",
        "server_e2e_s",
        "n_completed",
        "completion_tokens",
        "prompt_tokens",
        "cached_tokens",
        "n_started",
        "first_completion_t",
        "started_t",
    )

    def __init__(self, name: str) -> None:
        self.name = name
        self.tokenize_s = 0.0
        self.http_post_s = 0.0
        self.server_e2e_s = 0.0
        self.n_completed = 0
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.n_started = 0
        self.first_completion_t: float | None = None
        self.started_t: float | None = None

    def reset(self) -> None:
        self.tokenize_s = 0.0
        self.http_post_s = 0.0
        self.server_e2e_s = 0.0
        self.n_completed = 0
        self.completion_tokens = 0
        self.prompt_tokens = 0
        self.cached_tokens = 0
        self.n_started = 0
        self.first_completion_t = None
        self.started_t = None


_EVAL_PHASE = _PhaseStats("eval")
_TRAIN_PHASE = _PhaseStats("train")
_ACTIVE_PHASE: _PhaseStats | None = None


def _set_active_phase(phase: _PhaseStats | None) -> None:
    global _ACTIVE_PHASE
    _ACTIVE_PHASE = phase
    if phase is not None:
        phase.reset()
        phase.started_t = time.perf_counter()


def _phase_record_request(
    tokenize_s: float,
    http_post_s: float,
    meta_info: dict | None,
) -> None:
    phase = _ACTIVE_PHASE
    if phase is None:
        return
    phase.n_completed += 1
    phase.tokenize_s += tokenize_s
    phase.http_post_s += http_post_s
    if phase.first_completion_t is None:
        phase.first_completion_t = time.perf_counter()
    if meta_info is None:
        return
    phase.completion_tokens += int(meta_info.get("completion_tokens", 0) or 0)
    phase.prompt_tokens += int(meta_info.get("prompt_tokens", 0) or 0)
    phase.cached_tokens += int(meta_info.get("cached_tokens", 0) or 0)
    e2e = meta_info.get("e2e_latency")
    if e2e is None:
        e2e = meta_info.get("finish_time")
    if isinstance(e2e, (int, float)):
        phase.server_e2e_s += float(e2e)


def _phase_log_progress(phase: _PhaseStats, total: int, prefix: str) -> None:
    # [eval-prof] per-checkpoint progress line disabled — the SUMMARY at end of phase
    # carries the same numbers, and the tqdm bar already shows live progress.
    # Re-enable by removing the early return below.
    return
    if phase.n_completed == 0 or phase.started_t is None:
        return
    elapsed = time.perf_counter() - phase.started_t
    if phase.first_completion_t is not None:
        ttft = phase.first_completion_t - phase.started_t
    else:
        ttft = float("nan")
    rate = phase.n_completed / elapsed if elapsed > 0 else 0.0
    tps = phase.completion_tokens / elapsed if elapsed > 0 else 0.0
    cache_hit = (
        phase.cached_tokens / phase.prompt_tokens if phase.prompt_tokens > 0 else 0.0
    )
    avg_tokenize_ms = 1000 * phase.tokenize_s / max(1, phase.n_completed)
    avg_http_ms = 1000 * phase.http_post_s / max(1, phase.n_completed)
    avg_server_ms = 1000 * phase.server_e2e_s / max(1, phase.n_completed)
    logger.info(
        "%s progress: completed=%d/%d (%.1f%%) elapsed=%.1fs ttft=%.1fs "
        "throughput=%.1f req/s decode=%.0f tok/s avg_tokenize=%.1fms "
        "avg_http=%.0fms avg_server_e2e=%.0fms cache_hit=%.2f%% "
        "completion_tokens=%d prompt_tokens=%d",
        prefix,
        phase.n_completed,
        total,
        100.0 * phase.n_completed / max(1, total),
        elapsed,
        ttft,
        rate,
        tps,
        avg_tokenize_ms,
        avg_http_ms,
        avg_server_ms,
        100.0 * cache_hit,
        phase.completion_tokens,
        phase.prompt_tokens,
    )


def _phase_log_summary(phase: _PhaseStats, total: int, prefix: str) -> None:
    if phase.started_t is None:
        return
    elapsed = time.perf_counter() - phase.started_t
    rate = phase.n_completed / elapsed if elapsed > 0 else 0.0
    tps = phase.completion_tokens / elapsed if elapsed > 0 else 0.0
    cache_hit = (
        phase.cached_tokens / phase.prompt_tokens if phase.prompt_tokens > 0 else 0.0
    )
    ttft = (
        (phase.first_completion_t - phase.started_t)
        if phase.first_completion_t is not None
        else float("nan")
    )
    logger.info(
        "%s SUMMARY: total=%d completed=%d wall=%.1fs ttft=%.1fs "
        "throughput=%.2f req/s decode=%.1f tok/s tokenize_total=%.2fs "
        "http_total=%.2fs server_e2e_total=%.2fs (sum across reqs) "
        "cache_hit=%.2f%% completion_tokens=%d prompt_tokens=%d",
        prefix,
        total,
        phase.n_completed,
        elapsed,
        ttft,
        rate,
        tps,
        phase.tokenize_s,
        phase.http_post_s,
        phase.server_e2e_s,
        100.0 * cache_hit,
        phase.completion_tokens,
        phase.prompt_tokens,
    )


_PROGRESS_LOG_EVERY = int(os.environ.get("ORBIT_ROLLOUT_PROGRESS_EVERY", "100") or "0")
_SERVER_POLL_INTERVAL = float(
    os.environ.get("ORBIT_ROLLOUT_SERVER_POLL_S", "5") or "0"
)


async def _server_info_poller(args: Namespace, prefix: str, stop_event: asyncio.Event) -> None:
    """Periodically log sglang scheduler stats (running batch, queue, KV usage)."""
    if _SERVER_POLL_INTERVAL <= 0:
        return
    base = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
    try:
        workers_resp = await get(f"{base}/workers")
        worker_urls = [w["url"] for w in workers_resp.get("workers", [])]
    except Exception:
        try:
            workers_resp = await get(f"{base}/list_workers")
            worker_urls = workers_resp.get("urls", [])
        except Exception as e:
            logger.warning("[server-poll] cannot list workers: %r", e)
            return
    if not worker_urls:
        logger.warning("[server-poll] no worker urls found, disabling polling")
        return

    poll_t0 = time.perf_counter()
    while not stop_event.is_set():
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=_SERVER_POLL_INTERVAL)
            break
        except asyncio.TimeoutError:
            pass
        for url in worker_urls:
            try:
                info = await get(f"{url}/server_info")
            except Exception as e:
                logger.warning("[server-poll] %s failed: %r", url, e)
                continue
            states = info.get("internal_states") or []
            for i, st in enumerate(states):
                gen_tps = st.get("last_gen_throughput") or st.get("gen_throughput")
                max_running = st.get(
                    "effective_max_running_requests_per_dp"
                ) or st.get("max_running_requests")
                memu = st.get("memory_usage") or {}
                logger.info(
                    "[server-poll] %s t=%.0fs worker=%s dp=%d "
                    "last_gen_throughput=%s max_running_per_dp=%s "
                    "kv_token_capacity=%s kvcache_mem=%s graph_mem=%s",
                    prefix,
                    time.perf_counter() - poll_t0,
                    url.split("//")[-1],
                    i,
                    gen_tps,
                    max_running,
                    memu.get("token_capacity"),
                    memu.get("kvcache"),
                    memu.get("graph"),
                )


def get_model_url(args: Namespace, model_name: str, endpoint: str = "/generate") -> str:
    """Return the router URL for a named model.

    Use this in custom rollout functions to route requests to a specific
    model when multiple models are deployed via ``--sglang-config``::

        url = get_model_url(args, "ref", "/generate")
        resp = await post(url, json=payload)

    Falls back to the default router if *model_name* is not found or
    ``sglang_model_routers`` is not set.
    """
    routers = getattr(args, "sglang_model_routers", None)
    if routers and model_name in routers:
        ip, port = routers[model_name]
        return f"http://{ip}:{port}{endpoint}"
    return f"http://{args.sglang_router_ip}:{args.sglang_router_port}{endpoint}"


class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args: Namespace) -> None:
        # persistent state for the generation process
        self.args = args
        self.tokenizer = load_tokenizer(
            args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
        )
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        self.semaphore = asyncio.Semaphore(
            args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        # dp rank balancing
        self.dp_counts = [0] * (args.sglang_dp_size or 1)
        self.dp_rank = 0

        self.reset()

    @contextmanager
    def dp_rank_context(self):
        candidates = [i for i, count in enumerate(self.dp_counts) if count == min(self.dp_counts)]
        dp_rank = int(np.random.choice(candidates))
        self.dp_counts[dp_rank] += 1
        self.dp_rank = dp_rank
        try:
            yield dp_rank
        finally:
            self.dp_counts[dp_rank] -= 1
            assert self.dp_counts[dp_rank] >= 0

    def reset(self) -> None:
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False

    def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
        for group in samples:
            self.pendings.add(
                asyncio.create_task(
                    # submit a group of samples as a single task.
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(samples)


async def generate(
    args: Namespace, sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False
) -> Sample:
    """Generate using traditional SGLang router with token-based workflow"""
    if args.ci_test:
        assert isinstance(sample.prompt, str)

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    _t_tok0 = time.perf_counter()
    if state.processor and sample.multimodal_inputs and any(v is not None for v in sample.multimodal_inputs.values()):
        processor_kwargs = build_processor_kwargs(sample.multimodal_inputs)
        processor_output = state.processor(text=sample.prompt, **processor_kwargs)
        prompt_ids = processor_output["input_ids"][0]
        sample.multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None
    else:
        prompt_ids = state.tokenizer.encode(sample.prompt, add_special_tokens=False)
    _tokenize_s = time.perf_counter() - _t_tok0

    if len(sample.response) > 0:
        sampling_params["max_new_tokens"] -= len(sample.tokens) - len(prompt_ids)

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    # Prepare payload for sglang server
    payload = {
        "sampling_params": sampling_params,
        "return_logprob": should_request_rollout_logprobs(args, evaluation),
    }

    # Top-k OPD (sglang teacher): collect the student's own top-k logprobs during
    # generation; post_process cross-scores them against the teacher's top-k.
    _opd_top_k = getattr(args, "opd_log_prob_top_k", 0) or 0
    _opd_wants_student_top = (
        not evaluation
        and _opd_top_k > 0
        and getattr(args, "opd_type", None) == "sglang"
        and getattr(args, "opd_top_k_strategy", "only-student") != "only-teacher"
    )
    if _opd_wants_student_top:
        payload["top_logprobs_num"] = _opd_top_k
        payload["return_logprob"] = True  # sglang returns output_top_logprobs only with logprobs on

    attach_peft_request_payload(args, payload)

    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True

    if sample.multimodal_inputs and sample.multimodal_inputs["images"]:
        image_data = sample.multimodal_inputs["images"]
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    # Use existing tokens for multi-turn or tokenize the new prompt
    if len(sample.response) > 0:
        payload["input_ids"] = sample.tokens
    else:
        payload["input_ids"] = prompt_ids
        if not sample.tokens:  # Initialize sample.tokens for the first turn
            sample.tokens = prompt_ids

    # Use session_id for consistent hashing routing if router uses consistent_hashing policy
    headers = None
    if args.sglang_router_policy == "consistent_hashing" and sample.session_id:
        headers = {"X-SMG-Routing-Key": sample.session_id}

    _t_http0 = time.perf_counter()
    output = await post(url, payload, headers=headers)
    if _opd_wants_student_top:
        _output_top_logprobs = output.get("meta_info", {}).get("output_top_logprobs")
        if _output_top_logprobs is not None:
            sample.metadata.setdefault("opd_student_top_logprobs", [])
            sample.metadata["opd_student_top_logprobs"].extend(_output_top_logprobs)
    if os.environ.get("ORBIT_DSV4_RESPONSE_DEBUG", "0") == "1":
        dump_dir = os.environ.get("ORBIT_DSV4_RESPONSE_DEBUG_DIR", "/tmp")
        os.makedirs(dump_dir, exist_ok=True)
        request_id = output.get("meta_info", {}).get("id", "unknown") if isinstance(output, dict) else "unknown"
        dump_path = os.path.join(dump_dir, f"orbit_sglang_response_{int(time.time() * 1000)}_{request_id}.json")
        with open(dump_path, "w") as f:
            json.dump(
                {
                    "url": url,
                    "headers": headers,
                    "payload": payload,
                    "prompt_ids": prompt_ids,
                    "sample_tokens_before": sample.tokens,
                    "output": output,
                },
                f,
                default=str,
            )
        logger.warning("[ORBIT_DSV4_RESPONSE_DEBUG] wrote %s", dump_path)
    _http_post_s = time.perf_counter() - _t_http0
    _phase_record_request(
        tokenize_s=_tokenize_s,
        http_post_s=_http_post_s,
        meta_info=output.get("meta_info") if isinstance(output, dict) else None,
    )

    if args.use_orbit_router and "RadixTreeMiddleware" in args.orbit_router_middleware_paths:
        from orbit.router.middleware_hub.radix_tree_middleware import postprocess_sample_with_radix_tree

        sample = await postprocess_sample_with_radix_tree(args, sample, output)
    else:
        if "output_token_logprobs" in output["meta_info"]:
            output_token_logprobs = output["meta_info"]["output_token_logprobs"]
            new_response_log_probs = [item[0] for item in output_token_logprobs]
        else:
            output_token_logprobs = None
            new_response_log_probs = None

        if output.get("output_ids") is not None:
            new_response_tokens = output["output_ids"]
        elif output_token_logprobs is not None:
            new_response_tokens = [item[1] for item in output_token_logprobs]
        else:
            new_response_tokens = []

        # Update sample with tokens directly - avoiding re-tokenization
        sample.tokens = sample.tokens + new_response_tokens
        sample.response_length += len(new_response_tokens)
        sample.response += output["text"]

        # When partial rollout and masking off policy is enabled, update the loss mask
        if sample.loss_mask is not None:
            assert args.partial_rollout and args.mask_offpolicy_in_partial_rollout
            sample.loss_mask += [1] * len(new_response_tokens)

        if new_response_log_probs is not None:
            if sample.rollout_log_probs is None:
                sample.rollout_log_probs = []
            sample.rollout_log_probs += new_response_log_probs

    if "routed_experts" in output["meta_info"]:
        sample.rollout_routed_experts = np.frombuffer(
            pybase64.b64decode(output["meta_info"]["routed_experts"].encode("ascii")),
            dtype=np.int32,
        ).reshape(
            len(sample.tokens) - 1,
            args.num_layers,
            args.moe_router_topk,
        )

    sample.update_from_meta_info(args, output["meta_info"])

    return sample


async def generate_and_rm(
    args: Namespace,
    sample: Sample | list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    # mask previous off-policy generation for partial rollout
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        with state.dp_rank_context() as _:
            # Check sample.generate_function_path for per-sample custom_generate_function_path (e.g., from eval dataset config)
            custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path

            if custom_func_path is not None:
                custom_generate_func = load_function(custom_func_path)
                # if signature has evaluation, pass evaluation
                if "evaluation" in inspect.signature(custom_generate_func).parameters:
                    sample = await custom_generate_func(args, sample, sampling_params, evaluation=evaluation)
                else:
                    sample = await custom_generate_func(args, sample, sampling_params)
            else:
                sample = await generate(args, sample, sampling_params, evaluation=evaluation)

    # score against the local same-engine teacher (adapter-slot or base), once
    # per generated sample, before any reward computation below; dormant
    # unless local_scoring_enabled(args) (same-base --opd-teacher, no external
    # --opd-teacher-url/-urls).
    if local_scoring_enabled(args):
        for scored_sample in sample if isinstance(sample, list) else [sample]:
            if scored_sample.status != Sample.Status.ABORTED:
                await opd_score_sample(args, scored_sample)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    # multi samples
    if isinstance(sample, list):
        samples = sample
        if any([sample.status == Sample.Status.ABORTED for sample in samples]):
            return samples

        # for multi agent system, the reward of some sample is calculated during generation.
        samples_need_reward = [sample for sample in samples if sample.reward is None]
        rewards = await batched_async_rm(args, samples_need_reward)
        for sample, reward in zip(samples_need_reward, rewards, strict=False):
            sample.reward = reward
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        # for multi-turn environment, a reward could be assigned to the agent.
        if sample.reward is None:
            # Custom rms receive the evaluation flag so reward-slot transports
            # (OPD teacher scoring) can hand eval samples to the real task RM.
            sample.reward = await async_rm(args, sample, evaluation=evaluation)

    return sample


async def generate_and_rm_group(
    args: Namespace, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample]:
    state = GenerateState(args)

    if state.aborted:
        return group

    # Generate a unique session_id for each sample in the group (consistent hashing only)
    if args.sglang_router_policy == "consistent_hashing":
        for sample in group:
            if sample.session_id is None:
                sample.session_id = str(uuid.uuid4())

    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            seed = state.group_sampling_seeds[idx]
            current_sampling_params["sampling_seed"] = seed
        tasks.append(
            asyncio.create_task(generate_and_rm(args, sample, current_sampling_params, evaluation=evaluation))
        )

    group = await asyncio.gather(*tasks)

    # for the rm that need the whole group, we will do the rm here
    if not state.aborted and args.group_rm:
        rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward

    return group


async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    aborted_samples = []

    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True

    if parse(sglang_router.__version__) <= parse("0.2.1") or args.use_orbit_router:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/list_workers")
        urls = response["urls"]
    else:
        response = await get(f"http://{args.sglang_router_ip}:{args.sglang_router_port}/workers")
        urls = [worker["url"] for worker in response["workers"]]

    logger.info(f"Abort request for {urls}")
    abort_tasks = [post(f"{url}/abort_request", {"abort_all": True}) for url in urls]
    abort_results = await asyncio.gather(*abort_tasks, return_exceptions=True)
    for url, result in zip(urls, abort_results, strict=False):
        if isinstance(result, Exception):
            logger.warning(f"Failed to abort worker at {url}: {result}")

    # make sure all the pending tasks are finished
    count = 0
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)

        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)

    if args.partial_rollout:
        logger.info(f"Collected {count} partial samples into the data buffer")

    return aborted_samples


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        tuple[RolloutFnTrainOutput, list[list[Sample]]]:
            - data: a list of groups of samples generated by the rollout, length equals `rollout_batch_size`
            - aborted_samples: any partial groups collected during abort when partial_rollout is enabled
    """
    assert args.rollout_global_dataset

    await dumper_utils.configure_sglang(args)

    state = GenerateState(args)

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    metric_gatherer = MetricGatherer()

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size

    data = []
    all_data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    _set_active_phase(_TRAIN_PHASE)
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    _phase_log_summary(
        _TRAIN_PHASE,
        target_data_size * args.n_samples_per_prompt,
        f"[rollout-prof] rollout_id={rollout_id}",
    )
    _set_active_phase(None)
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
    )

    # there are still some unfinished requests, abort them
    aborted_samples = await abort(args, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)
    all_samples = sorted(
        all_data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
    )

    # True-on-policy Phase 1: replace decode-time rollout_log_probs with one
    # clean prefill re-scoring pass (before state.reset(), which clears
    # state.sampling_params). Groups may nest one level (multi-turn).
    flat_samples = [
        sample
        for group in data
        for item in group
        for sample in (item if isinstance(item, list) else [item])
    ]
    await recompute_samples_rollout_logprobs_via_prefill(
        args,
        flat_samples,
        url=get_model_url(args, "default"),
        sampling_params=state.sampling_params,
    )

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    # There can be circumstances where users want to process all samples including filtered ones.
    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, all_samples, data_source)

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args: Namespace, rollout_id: int) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:
    # --group-rm eval is supported via singleton-group grading in
    # _generate_and_rm_eval (each eval sample becomes its own group).

    coros = []
    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
        coros.append(eval_rollout_single_dataset(args, rollout_id, dataset_cfg))
    results_list = await asyncio.gather(*coros)
    results = {}
    for r in results_list:
        results.update(r)
    return RolloutFnEvalOutput(data=results), []


async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig
) -> dict[str, dict[str, list[Any]]]:
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        dataset_cfg: configuration of the dataset
    """
    # --group-rm eval is supported via singleton-group grading in
    # _generate_and_rm_eval (each eval sample becomes its own group).

    global EVAL_PROMPT_DATASET

    _eval_t0 = time.perf_counter()
    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template, args.chat_template_path)
    if args.apply_chat_template_kwargs:
        cache_key += (json.dumps(args.apply_chat_template_kwargs, sort_keys=True),)
    _ds_built = False
    if cache_key not in EVAL_PROMPT_DATASET:
        _ds_t0 = time.perf_counter()
        tokenizer = load_tokenizer(
            args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
        )
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )
        _ds_built = True
        logger.info(
            "[eval-prof] dataset_build name=%s path=%s elapsed=%.2fs num_samples=%d",
            dataset_cfg.name,
            dataset_cfg.path,
            time.perf_counter() - _ds_t0,
            len(EVAL_PROMPT_DATASET[cache_key].samples),
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]
    if not _ds_built:
        logger.info(
            "[eval-prof] dataset_cached name=%s num_samples=%d",
            dataset_cfg.name,
            len(dataset.samples),
        )

    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    eval_generate_max_concurrency = getattr(args, "eval_generate_max_concurrency", None)
    eval_generate_semaphore = (
        asyncio.Semaphore(eval_generate_max_concurrency)
        if eval_generate_max_concurrency is not None and eval_generate_max_concurrency > 0
        else None
    )

    async def _generate_and_rm_eval(sample, sampling_params):
        sample = await generate_and_rm(
            args,
            sample,
            sampling_params=sampling_params,
            evaluation=True,
        )
        # With --group-rm, generate_and_rm defers rewards to the group hook;
        # eval samples have no training group, so grade each as a singleton
        # group (batch hooks handle len-1 groups: router/code/judge stay
        # meaningful, genrm degenerates to a documented neutral 0.5).
        if args.group_rm and sample.reward is None and sample.status != Sample.Status.ABORTED:
            sample.reward = (await batched_async_rm(args, [sample]))[0]
        return sample

    async def generate_eval_sample(sample, sampling_params):
        if eval_generate_semaphore is None:
            return await _generate_and_rm_eval(sample, sampling_params)

        async with eval_generate_semaphore:
            return await _generate_and_rm_eval(sample, sampling_params)

    _setup_t0 = time.perf_counter()
    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for _i, prompt_sample in enumerate(dataset.samples):
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            tasks.append(
                asyncio.create_task(
                    generate_eval_sample(sample, sampling_params)
                )
            )
    logger.info(
        "[eval-prof] task_setup name=%s n_tasks=%d elapsed=%.2fs concurrency=%s",
        dataset_cfg.name,
        len(tasks),
        time.perf_counter() - _setup_t0,
        eval_generate_max_concurrency,
    )

    _set_active_phase(_EVAL_PHASE)
    poll_stop = asyncio.Event()
    poll_task = asyncio.create_task(
        _server_info_poller(args, f"eval/{dataset_cfg.name}", poll_stop)
    )
    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
    last_log_n = 0
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print:
            logger.info(
                "eval_rollout_single_dataset example data: "
                f"{[str(sample.prompt) + sample.response]} "
                f"reward={sample.reward}"
            )
            do_print = False
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)
        pbar.update(1)
        if (
            _PROGRESS_LOG_EVERY > 0
            and _EVAL_PHASE.n_completed - last_log_n >= _PROGRESS_LOG_EVERY
        ):
            last_log_n = _EVAL_PHASE.n_completed
            _phase_log_progress(
                _EVAL_PHASE, len(tasks), f"[eval-prof] {dataset_cfg.name}"
            )
    pbar.close()
    poll_stop.set()
    try:
        await asyncio.wait_for(poll_task, timeout=2.0)
    except (asyncio.TimeoutError, Exception):
        poll_task.cancel()
    _phase_log_summary(_EVAL_PHASE, len(tasks), f"[eval-prof] {dataset_cfg.name}")
    logger.info(
        "[eval-prof] dataset_total name=%s wall=%.2fs",
        dataset_cfg.name,
        time.perf_counter() - _eval_t0,
    )
    _set_active_phase(None)

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_cfg.name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_buffer: the data buffer to store the generated samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        list[list[Sample]]: a list of list of samples generated by the rollout
    """
    assert args.rollout_global_dataset
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    data_source.add_samples(aborted_samples)
    return output
