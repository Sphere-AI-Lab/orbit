"""Probe the exact multimodal SGLang prefill contract needed by OPD.

The probe deliberately uses student-expanded input IDs with two different,
same-geometry images. It validates sampled-token scoring, native teacher Top-K
targets, token alignment, image sensitivity, and image-aware prefix caching.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageOps

from orbit.rollout.on_policy_distillation import _teacher_sampled_log_probs, _teacher_topk_targets
from orbit.utils.data import Dataset
from orbit.utils.hf_config import load_hf_config
from orbit.utils.processing_utils import (
    call_processor,
    encode_image_for_rollout_engine,
    load_processor,
    load_tokenizer,
)
from orbit.utils.types import Sample


@dataclass(frozen=True)
class ScoringResult:
    sampled_log_probs: list[float]
    request_body_bytes: int
    response_body_bytes: int
    topk_shape: tuple[int, int] | None
    topk_mass_min: float | None
    topk_mass_max: float | None


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
    parser.add_argument("--output-json")
    parser.add_argument("--wandb-run-id", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.row_a < 0 or args.row_b < 0 or args.row_a == args.row_b:
        parser.error("--row-a and --row-b must be distinct non-negative indices")
    if args.image_size <= 0:
        parser.error("--image-size must be positive")
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.image_sensitivity_tolerance < 0 or args.cache_consistency_tolerance < 0:
        parser.error("tolerances must be non-negative")
    return args


def _post(url: str, payload: dict[str, Any], timeout: float) -> tuple[Any, int, int]:
    request_body = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} returned HTTP {exc.code}: {body[:2000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"POST {url} failed: {exc}") from exc

    decoded = response_body.decode("utf-8")
    try:
        output = json.loads(decoded)
    except json.JSONDecodeError:
        output = decoded
    return output, len(request_body), len(response_body)


def _flush_cache(base_url: str, timeout: float) -> None:
    _post(f"{base_url}/flush_cache", {}, timeout)


def _normalize_base_url(url: str) -> str:
    base_url = url.rstrip("/")
    if base_url.endswith("/generate"):
        base_url = base_url[: -len("/generate")]
    return base_url


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
            "00 smoke requires exactly one image in each selected Geo3K row: "
            f"row_a={len(images_a)}, row_b={len(images_b)}"
        )

    image_a = _resize_image(images_a[0], args.image_size)
    image_b = _resize_image(images_b[0], args.image_size)
    encoded_a = encode_image_for_rollout_engine(image_a)
    encoded_b = encode_image_for_rollout_engine(image_b)
    if encoded_a == encoded_b:
        raise ValueError("Selected Geo3K rows produced identical normalized images")

    prompt_ids_a = _as_token_ids(call_processor(processor, sample_a.prompt, {"images": [image_a]})["input_ids"])
    prompt_ids_b = _as_token_ids(call_processor(processor, sample_a.prompt, {"images": [image_b]})["input_ids"])
    if prompt_ids_a != prompt_ids_b:
        raise ValueError(
            "Same-size image substitution changed student-expanded input IDs; "
            f"len(A)={len(prompt_ids_a)}, len(B)={len(prompt_ids_b)}"
        )

    response_text = f"Answer: \\boxed{{{sample_a.label}}}"
    response_ids = tokenizer.encode(response_text, add_special_tokens=False)
    if not response_ids:
        raise ValueError(f"Response text tokenized to an empty sequence: {response_text!r}")

    config = load_hf_config(args.student_model_dir, trust_remote_code=True)
    vocab_size = _config_vocab_size(config)
    return prompt_ids_a, response_ids, encoded_a, encoded_b, response_text, vocab_size


def _topk_mass_range(log_probs, valid_mask) -> tuple[float, float]:
    masses = []
    for row_log_probs, row_mask in zip(log_probs.tolist(), valid_mask.tolist(), strict=True):
        masses.append(
            math.fsum(math.exp(value) for value, valid in zip(row_log_probs, row_mask, strict=True) if valid)
        )
    return min(masses), max(masses)


def _score(
    base_url: str,
    input_ids: list[int],
    response_ids: list[int],
    image_data: str,
    *,
    top_k: int,
    timeout: float,
    vocab_size: int,
) -> ScoringResult:
    prompt_length = len(input_ids) - len(response_ids)
    payload: dict[str, Any] = {
        "input_ids": input_ids,
        "sampling_params": {
            "max_new_tokens": 0,
            "temperature": 0,
            "skip_special_tokens": False,
        },
        "return_logprob": True,
        "logprob_start_len": prompt_length - 1,
        "image_data": [image_data],
    }
    if top_k > 0:
        payload["top_logprobs_num"] = top_k

    response, request_bytes, response_bytes = _post(f"{base_url}/generate", payload, timeout)
    if not isinstance(response, dict):
        raise ValueError(f"SGLang /generate returned {type(response).__name__}, expected an object")

    sample = Sample(
        index=0,
        group_index=0,
        tokens=input_ids,
        response_length=len(response_ids),
        loss_mask=[1] * len(response_ids),
    )
    sampled_log_probs = _teacher_sampled_log_probs(response, sample).tolist()

    topk_shape = None
    topk_mass_min = None
    topk_mass_max = None
    if top_k > 0:
        token_ids, topk_log_probs, valid_mask = _teacher_topk_targets(
            response,
            sample,
            top_k,
            vocab_size=vocab_size,
        )
        topk_shape = tuple(token_ids.shape)
        topk_mass_min, topk_mass_max = _topk_mass_range(topk_log_probs, valid_mask)

    return ScoringResult(
        sampled_log_probs=sampled_log_probs,
        request_body_bytes=request_bytes,
        response_body_bytes=response_bytes,
        topk_shape=topk_shape,
        topk_mass_min=topk_mass_min,
        topk_mass_max=topk_mass_max,
    )


def _max_abs_diff(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"Cannot compare logprob vectors with lengths {len(left)} and {len(right)}")
    return max((abs(a - b) for a, b in zip(left, right, strict=True)), default=0.0)


def _run_mode(
    args: argparse.Namespace,
    base_url: str,
    input_ids: list[int],
    response_ids: list[int],
    image_a: str,
    image_b: str,
    vocab_size: int,
    top_k: int,
) -> dict[str, Any]:
    _flush_cache(base_url, args.timeout)
    a_clean = _score(
        base_url,
        input_ids,
        response_ids,
        image_a,
        top_k=top_k,
        timeout=args.timeout,
        vocab_size=vocab_size,
    )

    # Same token IDs, different image, no flush: this catches a prefix cache key
    # that incorrectly ignores image content.
    b_after_a = _score(
        base_url,
        input_ids,
        response_ids,
        image_b,
        top_k=top_k,
        timeout=args.timeout,
        vocab_size=vocab_size,
    )

    _flush_cache(base_url, args.timeout)
    b_clean = _score(
        base_url,
        input_ids,
        response_ids,
        image_b,
        top_k=top_k,
        timeout=args.timeout,
        vocab_size=vocab_size,
    )

    image_delta = _max_abs_diff(a_clean.sampled_log_probs, b_clean.sampled_log_probs)
    cache_delta = _max_abs_diff(b_after_a.sampled_log_probs, b_clean.sampled_log_probs)
    if image_delta <= args.image_sensitivity_tolerance:
        raise AssertionError(
            "Teacher response-token logprobs are not image-sensitive: "
            f"top_k={top_k}, max_abs_delta={image_delta:.9g}, "
            f"required>{args.image_sensitivity_tolerance:.9g}"
        )
    if cache_delta > args.cache_consistency_tolerance:
        raise AssertionError(
            "Changing image content with identical input IDs produced cache-dependent scores: "
            f"top_k={top_k}, max_abs_delta={cache_delta:.9g}, "
            f"allowed<={args.cache_consistency_tolerance:.9g}"
        )

    return {
        "top_k": top_k,
        "image_max_abs_logprob_delta": image_delta,
        "cache_max_abs_logprob_delta": cache_delta,
        "sampled_logprob_mean_a": statistics.fmean(a_clean.sampled_log_probs),
        "sampled_logprob_mean_b": statistics.fmean(b_clean.sampled_log_probs),
        "request_body_bytes": a_clean.request_body_bytes,
        "response_body_bytes": a_clean.response_body_bytes,
        "topk_shape": list(a_clean.topk_shape) if a_clean.topk_shape else None,
        "topk_mass_min": a_clean.topk_mass_min,
        "topk_mass_max": a_clean.topk_mass_max,
    }


def _output_path(args: argparse.Namespace) -> Path | None:
    if args.output_json:
        return Path(args.output_json)
    run_dir = os.environ.get("ORBIT_RUN_DIR")
    return Path(run_dir) / "teacher_prefill_smoke.json" if run_dir else None


def main() -> None:
    args = _parse_args()
    base_url = _normalize_base_url(args.teacher_url)
    prompt_ids, response_ids, image_a, image_b, response_text, vocab_size = _load_probe_inputs(args)
    input_ids = [*prompt_ids, *response_ids]

    modes = [
        _run_mode(args, base_url, input_ids, response_ids, image_a, image_b, vocab_size, top_k=0),
        _run_mode(args, base_url, input_ids, response_ids, image_a, image_b, vocab_size, top_k=args.top_k),
    ]
    summary = {
        "status": "PASS",
        "teacher_url": base_url,
        "student_model_dir": args.student_model_dir,
        "dataset": args.dataset,
        "rows": [args.row_a, args.row_b],
        "image_size": [args.image_size, args.image_size],
        "image_a_sha256": hashlib.sha256(image_a.encode("utf-8")).hexdigest(),
        "image_b_sha256": hashlib.sha256(image_b.encode("utf-8")).hexdigest(),
        "prompt_tokens": len(prompt_ids),
        "response_tokens": len(response_ids),
        "response_text": response_text,
        "student_vocab_size": vocab_size,
        "modes": modes,
    }

    output_path = _output_path(args)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
