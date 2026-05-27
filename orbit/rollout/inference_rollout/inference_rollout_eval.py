import asyncio
import copy
import json
import logging
from contextlib import suppress
from typing import Any

from tqdm import tqdm

from orbit.rollout.inference_rollout.eval_logging import (
    _EvalTaskProgress,
    _log_pending_eval_tasks,
    _update_eval_task_progress,
)
from orbit.rollout.inference_rollout.inference_rollout_common import (
    GenerateState,
    compute_sampling_params,
    generate_and_rm,
)
from orbit.utils.data import Dataset
from orbit.utils.eval_config import EvalDatasetConfig
from orbit.utils.misc import as_completed_async
from orbit.utils.processing_utils import load_processor, load_tokenizer
from orbit.utils.types import Sample

logger = logging.getLogger(__name__)


async def _generate_and_rm_for_eval_with_progress(
    state: GenerateState,
    sample: Sample,
    sampling_params: dict[str, Any],
    progress_by_index: dict[int, _EvalTaskProgress],
) -> Sample | list[Sample]:
    _update_eval_task_progress(progress_by_index, "generate_started", sample)
    try:
        result = await generate_and_rm(
            state,
            sample,
            sampling_params=sampling_params,
            evaluation=True,
        )
    except Exception:
        _update_eval_task_progress(progress_by_index, "failed", sample)
        raise

    completed_sample = result[0] if isinstance(result, list) and result else result
    if isinstance(completed_sample, Sample):
        _update_eval_task_progress(progress_by_index, "completed", completed_sample)
    return result


async def eval_rollout_single_dataset(
    state: GenerateState,
    dataset_cfg: EvalDatasetConfig,
    prompt_dataset_cache: dict[Any, Dataset],
) -> dict[str, dict[str, list[Any]]]:
    args = state.args
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template, args.chat_template_path)
    if args.apply_chat_template_kwargs:
        cache_key += (json.dumps(args.apply_chat_template_kwargs, sort_keys=True),)
    if cache_key not in prompt_dataset_cache:
        tokenizer = load_tokenizer(
            args.hf_checkpoint, chat_template_path=args.chat_template_path, trust_remote_code=True
        )
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        prompt_dataset_cache[cache_key] = Dataset(
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
    dataset = prompt_dataset_cache[cache_key]

    base_sampling_params = compute_sampling_params(
        args,
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        stop=dataset_cfg.stop,
        stop_token_ids=dataset_cfg.stop_token_ids,
        min_new_tokens=dataset_cfg.min_new_tokens,
    )

    tasks_by_index: dict[int, asyncio.Task[Sample | list[Sample]]] = {}
    progress_by_index: dict[int, _EvalTaskProgress] = {}
    # do multiple samples for eval prompts
    sample_index = 0
    for _i, prompt_sample in enumerate(dataset.samples):
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["sampling_seed"] = args.rollout_seed + j
            tasks_by_index[sample.index] = asyncio.create_task(
                _generate_and_rm_for_eval_with_progress(
                    state,
                    sample,
                    sampling_params=sampling_params,
                    progress_by_index=progress_by_index,
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks_by_index), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
    pending_logger = asyncio.create_task(
        _log_pending_eval_tasks(
            dataset_name=dataset_cfg.name,
            tasks_by_index=tasks_by_index,
            progress_by_index=progress_by_index,
        )
    )
    try:
        async for sample in as_completed_async(tasks_by_index.values()):
            if do_print:
                # Improve this after enhancing samples' type
                s = (sample[0] if len(sample) > 0 else None) if isinstance(sample, list) else sample
                if s is not None:
                    logger.info(
                        "eval_rollout_single_dataset example data: "
                        f"{[str(s.prompt) + s.response]} "
                        f"reward={s.reward}"
                    )
                do_print = False
            if isinstance(sample, list):
                data.extend(sample)
            else:
                data.append(sample)
            pbar.update(1)
    finally:
        pending_logger.cancel()
        with suppress(asyncio.CancelledError):
            await pending_logger
        pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_cfg.name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }
