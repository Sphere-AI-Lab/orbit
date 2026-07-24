"""Verify that a VLM generation turn preserves exact prior sampled token IDs."""

import base64
import io
import os

import pytest
import requests
from PIL import Image
from tests.ci.ci_register import register_cuda_ci
from tests.e2e.sglang.utils.sglang_server import start_sglang_server
from transformers import AutoProcessor

register_cuda_ci(est_time=300, suite="stage-c-4-gpu-h200", labels=["sglang"])

MODEL_PATH = os.environ.get(
    "SGLANG_VLM_E2E_MODEL_PATH",
    "Qwen/Qwen2.5-VL-3B-Instruct",
)
IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"


@pytest.fixture(scope="module")
def sglang_server():
    previous_avoid_retokenize = os.environ.get("SGLANG_MM_AVOID_RETOKENIZE")
    os.environ["SGLANG_MM_AVOID_RETOKENIZE"] = "1"
    server = None
    try:
        server = start_sglang_server(
            model_path=MODEL_PATH,
            enable_deterministic_inference=False,
            extra_args=["--mem-fraction-static", "0.7"],
        )
        yield server
    finally:
        if server is not None:
            server.stop()
        if previous_avoid_retokenize is None:
            os.environ.pop("SGLANG_MM_AVOID_RETOKENIZE", None)
        else:
            os.environ["SGLANG_MM_AVOID_RETOKENIZE"] = previous_avoid_retokenize


def _image_and_data_uri() -> tuple[Image.Image, str]:
    image = Image.new("RGB", (64, 64), (128, 128, 128))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return image, f"data:image/png;base64,{encoded}"


def _find_noncanonical_ids(tokenizer) -> list[int]:
    for left, right in (("J", "K"), ("D", "escribe"), ("1", "2")):
        token_ids = [
            *tokenizer.encode(left, add_special_tokens=False),
            *tokenizer.encode(right, add_special_tokens=False),
        ]
        canonical_ids = tokenizer.encode(
            tokenizer.decode(token_ids, skip_special_tokens=False),
            add_special_tokens=False,
        )
        if token_ids != canonical_ids:
            return token_ids
    raise AssertionError("Could not construct a noncanonical token sequence")


def test_compact_vlm_prefix_preserves_noncanonical_prior_turn_ids(sglang_server):
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=True,
    )
    tokenizer = processor.tokenizer
    image, image_data = _image_and_data_uri()

    rendered_prompt = f"{IMAGE_TOKEN}\nDescribe the geometry, then call the tool."
    compact_prefix_ids = tokenizer.encode(rendered_prompt, add_special_tokens=False)
    expanded_prefix_ids = (
        processor(
            text=rendered_prompt,
            images=[image],
            return_tensors="pt",
        )
        .input_ids[0]
        .tolist()
    )
    prior_action_ids = _find_noncanonical_ids(tokenizer)
    observation_ids = tokenizer.encode("\nTool result: 0", add_special_tokens=False)
    exact_history_ids = [*prior_action_ids, *observation_ids]

    response = requests.post(
        f"{sglang_server.base_url}/generate",
        json={
            "input_ids": [*compact_prefix_ids, *exact_history_ids],
            "image_data": [image_data],
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 1},
            "return_logprob": True,
            "logprob_start_len": len(expanded_prefix_ids) - 1,
        },
        timeout=300,
    )
    response.raise_for_status()
    meta_info = response.json()["meta_info"]

    expected_prompt_tokens = len(expanded_prefix_ids) + len(exact_history_ids)
    assert meta_info["prompt_tokens"] == expected_prompt_tokens

    scored_input_ids = [item[1] for item in meta_info["input_token_logprobs"] if item is not None]
    assert scored_input_ids[-len(exact_history_ids) :] == exact_history_ids
