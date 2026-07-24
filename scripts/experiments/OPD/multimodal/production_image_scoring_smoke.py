"""Exercise multimodal OPD through the production scoring and parsing path.

The probe holds response token IDs fixed while swapping same-geometry Geo3K
images. It runs both sampled RKLD and native teacher Top-K DAgger through
``reward_func`` plus ``post_process_rewards``, including one masked response
row that represents an inter-turn environment observation.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import statistics
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageOps
from teacher_prefill_smoke import _flush_cache, _normalize_base_url

from miles.rollout.on_policy_distillation import close_scoring_transport, post_process_rewards, reward_func
from miles.utils.data import Dataset
from miles.utils.hf_config import load_hf_config
from miles.utils.processing_utils import call_processor, load_processor, load_tokenizer
from miles.utils.types import Sample


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-url", required=True, help="SGLang base URL, without /generate")
    parser.add_argument("--student-model-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input-key", default="problem")
    parser.add_argument("--label-key", default="answer")
    parser.add_argument("--row-a", type=int, default=0)
    parser.add_argument("--row-b", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--image-sensitivity-tolerance", type=float, default=1e-6)
    parser.add_argument("--cache-consistency-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--sglang-mm-exact-scoring-suffix",
        action="store_true",
        help="Enable the opt-in SGLang scoring_suffix_ids contract.",
    )
    parser.add_argument("--output-json")
    parser.add_argument("--wandb-run-id", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.row_a < 0 or args.row_b < 0 or args.row_a == args.row_b:
        parser.error("--row-a and --row-b must be distinct non-negative indices")
    if args.image_size <= 0 or args.top_k <= 0 or args.timeout <= 0:
        parser.error("--image-size, --top-k, and --timeout must be positive")
    if args.image_sensitivity_tolerance < 0 or args.cache_consistency_tolerance < 0:
        parser.error("tolerances must be non-negative")
    return args


def _as_token_ids(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or not value or not all(isinstance(token_id, int) for token_id in value):
        raise ValueError(f"Processor returned malformed input_ids: {type(value).__name__}")
    return value


def _resize_image(image: Image.Image, size: int) -> Image.Image:
    return ImageOps.fit(image.convert("RGB"), (size, size), method=Image.Resampling.LANCZOS)


def _config_vocab_size(config: Any) -> int:
    vocab_size = getattr(config, "vocab_size", None)
    if vocab_size is None and getattr(config, "text_config", None) is not None:
        vocab_size = getattr(config.text_config, "vocab_size", None)
    if vocab_size is None:
        raise ValueError(f"Could not determine text vocabulary size from {type(config).__name__}")
    return int(vocab_size)


def _load_probe_inputs(args: argparse.Namespace):
    tokenizer = load_tokenizer(args.student_model_dir, trust_remote_code=True)
    processor = load_processor(args.student_model_dir, trust_remote_code=True)
    if processor is None:
        raise RuntimeError(f"Could not load a multimodal processor from {args.student_model_dir}")

    start = min(args.row_a, args.row_b)
    stop = max(args.row_a, args.row_b) + 1
    dataset = Dataset(
        f"{args.dataset}@[{start}:{stop}]",
        tokenizer,
        processor,
        max_length=None,
        prompt_key=args.input_key,
        multimodal_keys={"image": "images"},
        label_key=args.label_key,
        apply_chat_template=True,
    )
    sample_a = dataset[args.row_a - start]
    sample_b = dataset[args.row_b - start]
    images_a = (sample_a.multimodal_inputs or {}).get("images") or []
    images_b = (sample_b.multimodal_inputs or {}).get("images") or []
    if len(images_a) != 1 or len(images_b) != 1:
        raise ValueError(
            "01 smoke requires exactly one image in each selected Geo3K row: "
            f"row_a={len(images_a)}, row_b={len(images_b)}"
        )

    image_a = _resize_image(images_a[0], args.image_size)
    image_b = _resize_image(images_b[0], args.image_size)
    prompt_ids_a = _as_token_ids(call_processor(processor, sample_a.prompt, {"images": [image_a]})["input_ids"])
    prompt_ids_b = _as_token_ids(call_processor(processor, sample_a.prompt, {"images": [image_b]})["input_ids"])
    if prompt_ids_a != prompt_ids_b:
        raise ValueError(
            "Same-size image substitution changed student-expanded input IDs; "
            f"len(A)={len(prompt_ids_a)}, len(B)={len(prompt_ids_b)}"
        )

    response_text = f"Answer: \\boxed{{{sample_a.label}}}"
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    if len(response_ids) < 2:
        raise ValueError("01 smoke requires at least two response tokens to exercise a masked row")
    masked_position = len(response_ids) // 2
    loss_mask = [1] * len(response_ids)
    loss_mask[masked_position] = 0

    if not isinstance(sample_a.prompt, str) or not sample_a.prompt:
        raise ValueError("01 exact-suffix smoke requires a non-empty rendered string prompt")

    config = load_hf_config(args.student_model_dir, trust_remote_code=True)
    return (
        sample_a.prompt,
        prompt_ids_a,
        response_ids,
        loss_mask,
        masked_position,
        image_a,
        image_b,
        response_text,
        _config_vocab_size(config),
    )


def _opd_args(args: argparse.Namespace, base_url: str, *, dagger_top_k: int) -> Namespace:
    return Namespace(
        rm_url=f"{base_url}/generate",
        reward_key=None,
        vocab_size=args.vocab_size,
        opd_log_task_reward=False,
        opd_log_prob_top_k=0,
        opd_dagger_top_k=dagger_top_k,
        opd_scoring_timeout=args.timeout,
        opd_scoring_max_inflight=1,
        opd_scoring_retries=0,
        opd_scoring_persistent_session=True,
        sglang_mm_exact_scoring_suffix=args.sglang_mm_exact_scoring_suffix,
    )


def _new_sample(
    prompt: str,
    prompt_ids: list[int],
    response_ids: list[int],
    loss_mask: list[int],
    image: Image.Image,
) -> Sample:
    return Sample(
        index=0,
        group_index=0,
        prompt=prompt,
        tokens=[*prompt_ids, *response_ids],
        response_length=len(response_ids),
        loss_mask=list(loss_mask),
        multimodal_inputs={"images": [image]},
    )


async def _score_once(
    opd_args: Namespace,
    prompt: str,
    prompt_ids: list[int],
    response_ids: list[int],
    loss_mask: list[int],
    masked_position: int,
    image: Image.Image,
) -> dict[str, Any]:
    sample = _new_sample(prompt, prompt_ids, response_ids, loss_mask, image)
    sample.reward = await reward_func(opd_args, sample)
    post_process_rewards(opd_args, [sample])
    sample.validate()

    sampled_log_probs = sample.teacher_log_probs.tolist()
    if sampled_log_probs[masked_position] != 0.0:
        raise AssertionError(f"Masked sampled teacher logprob must be zero, got {sampled_log_probs[masked_position]}")

    result: dict[str, Any] = {
        "sampled_log_probs": sampled_log_probs,
        "telemetry": sample.metadata["opd_scoring_telemetry"],
    }
    if opd_args.opd_dagger_top_k > 0:
        if sample.teacher_topk_token_ids[masked_position].ne(0).any():
            raise AssertionError("Masked teacher Top-K IDs must use the zero sentinel")
        if sample.teacher_topk_valid_mask[masked_position].any():
            raise AssertionError("Masked teacher Top-K entries must all be invalid")
        if not sample.teacher_topk_log_probs[masked_position].isneginf().all():
            raise AssertionError("Masked teacher Top-K logprobs must use the -inf sentinel")
        result.update(
            topk_shape=list(sample.teacher_topk_token_ids.shape),
            active_topk_rows=int(sample.teacher_topk_valid_mask.any(dim=-1).sum().item()),
        )
    return result


def _active_max_abs_diff(left: list[float], right: list[float], loss_mask: list[int]) -> float:
    values = [abs(a - b) for a, b, active in zip(left, right, loss_mask, strict=True) if active]
    return max(values, default=0.0)


async def _run_mode(
    args: argparse.Namespace,
    base_url: str,
    prompt: str,
    prompt_ids: list[int],
    response_ids: list[int],
    loss_mask: list[int],
    masked_position: int,
    image_a: Image.Image,
    image_b: Image.Image,
    *,
    dagger_top_k: int,
) -> dict[str, Any]:
    opd_args = _opd_args(args, base_url, dagger_top_k=dagger_top_k)
    _flush_cache(base_url, args.timeout)
    a_clean = await _score_once(opd_args, prompt, prompt_ids, response_ids, loss_mask, masked_position, image_a)
    b_after_a = await _score_once(opd_args, prompt, prompt_ids, response_ids, loss_mask, masked_position, image_b)
    _flush_cache(base_url, args.timeout)
    b_clean = await _score_once(opd_args, prompt, prompt_ids, response_ids, loss_mask, masked_position, image_b)

    image_delta = _active_max_abs_diff(a_clean["sampled_log_probs"], b_clean["sampled_log_probs"], loss_mask)
    cache_delta = _active_max_abs_diff(b_after_a["sampled_log_probs"], b_clean["sampled_log_probs"], loss_mask)
    session_reused = bool(b_after_a["telemetry"][0]["client_session_reused"])
    if image_delta <= args.image_sensitivity_tolerance:
        raise AssertionError(
            "Production teacher scores are not image-sensitive: "
            f"dagger_top_k={dagger_top_k}, max_abs_delta={image_delta:.9g}"
        )
    if cache_delta > args.cache_consistency_tolerance:
        raise AssertionError(
            "Production scores depend on preceding same-token/different-image cache state: "
            f"dagger_top_k={dagger_top_k}, max_abs_delta={cache_delta:.9g}"
        )
    if not session_reused:
        raise AssertionError(f"Production scoring did not reuse its persistent HTTP session for top_k={dagger_top_k}")

    active_a = [value for value, active in zip(a_clean["sampled_log_probs"], loss_mask, strict=True) if active]
    telemetry = a_clean["telemetry"]
    result = {
        "mode": "dagger" if dagger_top_k else "sampled_rkld",
        "top_k": dagger_top_k,
        "image_max_abs_logprob_delta": image_delta,
        "cache_max_abs_logprob_delta": cache_delta,
        "active_sampled_logprob_mean_a": statistics.fmean(active_a),
        "request_body_bytes": telemetry[0].get("request_body_bytes"),
        "response_body_bytes": telemetry[0].get("response_body_bytes"),
        "client_session_reused_after_first_request": session_reused,
    }
    if dagger_top_k:
        result["topk_shape"] = a_clean["topk_shape"]
        result["active_topk_rows"] = a_clean["active_topk_rows"]
    return result


def _output_path(args: argparse.Namespace) -> Path | None:
    if args.output_json:
        return Path(args.output_json)
    run_dir = os.environ.get("MILES_RUN_DIR")
    return Path(run_dir) / "production_image_scoring_smoke.json" if run_dir else None


async def _main(args: argparse.Namespace) -> dict[str, Any]:
    base_url = _normalize_base_url(args.teacher_url)
    (
        prompt,
        prompt_ids,
        response_ids,
        loss_mask,
        masked_position,
        image_a,
        image_b,
        response_text,
        vocab_size,
    ) = _load_probe_inputs(args)
    args.vocab_size = vocab_size
    try:
        modes = [
            await _run_mode(
                args,
                base_url,
                prompt,
                prompt_ids,
                response_ids,
                loss_mask,
                masked_position,
                image_a,
                image_b,
                dagger_top_k=0,
            ),
            await _run_mode(
                args,
                base_url,
                prompt,
                prompt_ids,
                response_ids,
                loss_mask,
                masked_position,
                image_a,
                image_b,
                dagger_top_k=args.top_k,
            ),
        ]
    finally:
        await close_scoring_transport()

    return {
        "status": "PASS",
        "teacher_url": base_url,
        "student_model_dir": args.student_model_dir,
        "sglang_mm_exact_scoring_suffix": args.sglang_mm_exact_scoring_suffix,
        "dataset": args.dataset,
        "rows": [args.row_a, args.row_b],
        "image_size": [args.image_size, args.image_size],
        "image_a_pixel_sha256": hashlib.sha256(image_a.tobytes()).hexdigest(),
        "image_b_pixel_sha256": hashlib.sha256(image_b.tobytes()).hexdigest(),
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(response_ids),
        "masked_response_position": masked_position,
        "response_text": response_text,
        "student_vocab_size": vocab_size,
        "modes": modes,
    }


def main() -> None:
    args = _parse_args()
    summary = asyncio.run(_main(args))
    output_path = _output_path(args)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
