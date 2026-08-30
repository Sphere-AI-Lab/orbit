"""
Utils to integrate SGLang's `/generate` endpoint with RL things like Sample.
"""

# ORBIT-SEAM: logging/os support the PEFT-request debug logging in attach_peft_request_payload below
import logging
import os
from copy import deepcopy
from typing import Any

import numpy as np
import pybase64

# ORBIT-SEAM: OFT adapter slot name / PEFT-method lookup, consumed by attach_peft_request_payload below
# (miles.utils.lora's LORA_ADAPTER_NAME/lora_rollout_enabled are deliberately NOT imported -- see
# attach_peft_request_payload below for why orbit never puts lora_path on a generate request)
from miles.orbit.megatron.oft_utils import OFT_ADAPTER_NAME
from miles.orbit.megatron.peft_utils import get_peft_method
from miles.utils.processing_utils import encode_image_for_rollout_engine, extract_multimodal_train_inputs
from miles.utils.types import Sample

# ORBIT-SEAM: logger + rate-limit counter for the ORBIT_DEBUG_PEFT_REQUEST diagnostic in
# attach_peft_request_payload below
logger = logging.getLogger(__name__)
_debug_peft_request_count = 0


# Make this an isolated function because users may want to compute their own
def compute_prompt_ids_from_sample(state, sample, tools=None):
    prompt = sample.prompt

    if state.processor and sample.multimodal_inputs and any(v is not None for v in sample.multimodal_inputs.values()):
        processor_output = state.processor(text=prompt, **sample.multimodal_inputs)
        prompt_ids = processor_output["input_ids"][0]

        # ORBIT-SEAM: TODO wording normalized (repo-wide comment style pass, no functional change)
        # Follow-up shall we move it to other places? then can make this function immutable
        sample.multimodal_train_inputs = extract_multimodal_train_inputs(processor_output)

        return prompt_ids
    else:
        if not isinstance(prompt, str):
            prompt = state.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True, tools=tools
            )

        return state.tokenizer.encode(prompt, add_special_tokens=False)


def policy_uses_routing_key(args) -> bool:
    return args.sglang_router_policy in ("consistent_hashing", "manual")


def compute_routing_headers(args, sample: Sample) -> dict[str, str] | None:
    if policy_uses_routing_key(args) and not sample.routing_key:
        raise ValueError(
            f"router policy {args.sglang_router_policy} routes by X-SMG-Routing-Key, "
            f"but sample (index={sample.index}) has no routing_key set"
        )
    if sample.routing_key:
        return {"X-SMG-Routing-Key": sample.routing_key}
    return None


def compute_request_payload(
    args,
    input_ids: list[int],
    sampling_params: dict,
    multimodal_inputs: dict | None = None,
    # ORBIT-SEAM: caller-controllable return_logprob (see should_request_rollout_logprobs below)
    return_logprob: bool = True,
) -> tuple[dict[str, Any] | None, Sample.Status | None]:
    sampling_params = deepcopy(sampling_params)
    max_new_tokens = sampling_params.pop("max_new_tokens", args.rollout_max_response_len)
    if x := args.rollout_max_context_len:
        max_new_tokens = min(max_new_tokens, x - len(input_ids))
    if max_new_tokens <= 0:
        return None, Sample.Status.TRUNCATED

    payload = {
        "input_ids": input_ids,
        "sampling_params": {**sampling_params, "max_new_tokens": max_new_tokens},
        # ORBIT-SEAM: honor the caller-supplied return_logprob instead of hardcoding True
        "return_logprob": return_logprob,
        "return_routed_experts": args.use_rollout_routing_replay,
        "return_indexer_topk": args.use_rollout_indexer_replay,
    }
    # ORBIT-SEAM: upstream's `if lora_rollout_enabled(args): payload["lora_path"] = LORA_ADAPTER_NAME`
    # is dropped here -- attach_peft_request_payload below is orbit's LoRA+OFT generalization of it,
    # and orbit's single-active fork 400s on a named lora_path (see the rationale there).
    if image_data := (multimodal_inputs or {}).get("images"):
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    # ORBIT-SEAM: attach the LoRA/OFT adapter selection to the generate payload (see below)
    attach_peft_request_payload(args, payload)

    return payload, None


# ORBIT-SEAM: LoRA (single-active adapter, index 0, no adapter_path needed) vs OFT (multi-slot,
# selects its trained slot via adapter_path=OFT_ADAPTER_NAME) request-payload wiring, plus an opt-in
# rate-limited debug log (ORBIT_DEBUG_PEFT_REQUEST); should_request_rollout_logprobs below decides
# whether the RadixTreeMiddleware router needs logprobs even outside eval
def attach_peft_request_payload(args, payload: dict[str, Any]) -> dict[str, Any]:
    global _debug_peft_request_count

    peft_method = get_peft_method(args)
    # LoRA is routed through the fork's SINGLE-ACTIVE peft/lora (peft_method="lora",
    # see sglang_engine.py) -- NOT upstream's multi-tenant LoRAManager. The
    # single-active path applies the index-0 adapter unconditionally, so the
    # generate request must NOT name an adapter (sending lora_path 400s in
    # upstream's _validate_and_resolve_lora when enable_lora is unset).
    # OFT runs multi-slot (base slot 0 + adapter slot 1) and selects its trained
    # slot via the fork's adapter_* wire key (v0.5.16 rename of oft_path).
    if peft_method == "oft" and not os.environ.get("ORBIT_DSV4_DISABLE_OFT_REQUEST"):
        payload["adapter_path"] = OFT_ADAPTER_NAME

    if os.environ.get("ORBIT_DEBUG_PEFT_REQUEST"):
        limit = int(os.environ.get("ORBIT_DEBUG_PEFT_REQUEST_LIMIT", "16"))
        if _debug_peft_request_count < limit:
            logger.info(
                "peft_request_payload peft_method=%s has_adapter_path=%s "
                "disable_oft_request=%s return_logprob=%s sampling_keys=%s",
                peft_method,
                "adapter_path" in payload,
                bool(os.environ.get("ORBIT_DSV4_DISABLE_OFT_REQUEST")),
                payload.get("return_logprob"),
                sorted(payload.get("sampling_params", {}).keys()),
            )
            _debug_peft_request_count += 1
    return payload


def should_request_rollout_logprobs(args, evaluation: bool = False) -> bool:
    # ORBIT-SEAM: the RadixTreeMiddleware clause is inert after the dbbab156 merge (upstream deleted
    # miles/router/middleware_hub/); the getattr guards keep it a no-op, and it stays so the contract
    # is restored for free if orbit re-lands the middleware.
    if getattr(args, "use_orbit_router", False) and "RadixTreeMiddleware" in getattr(
        args, "miles_router_middleware_paths", []
    ):
        return True
    if evaluation:
        return bool(getattr(args, "eval_return_rollout_logprobs", False))
    return True


async def update_sample_from_response(
    args, sample: Sample, payload: dict, output: dict, update_loss_mask: bool = False
):
    # Initialize sample.tokens for the first turn
    if (len(sample.response) == 0) and not sample.tokens:
        sample.tokens = payload["input_ids"]

    # ORBIT-SEAM: prefer output_ids directly when the engine returns them (avoids depending on
    # output_token_logprobs for tokens when logprobs weren't requested); new_response_log_probs
    # is left None rather than [] so the rollout_log_probs append below can be skipped cleanly
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
        # ORBIT-SEAM: new_response_log_probs is set (or left None) above, not reset here
        new_response_tokens = []

    # Update sample with tokens directly - avoiding re-tokenization
    sample.tokens = sample.tokens + new_response_tokens
    sample.response_length += len(new_response_tokens)
    sample.response += output["text"]

    # ORBIT-SEAM: only touch rollout_log_probs when logprobs were actually returned (see above)
    if new_response_log_probs is not None:
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        sample.rollout_log_probs += new_response_log_probs

    if update_loss_mask:
        if sample.loss_mask is None:
            sample.loss_mask = []
        sample.loss_mask += [1] * len(new_response_tokens)

    # ORBIT-SEAM: TODO wording normalized (repo-wide comment style pass, no functional change; both
    # comments below)
    # Follow-up handle multi-turn cases (may need concat instead of assignment)
    sample.rollout_routed_experts = get_routed_experts_from_response(args, output, len(sample.tokens) - 1)
    sample.rollout_indexer_topk = get_indexer_topk_from_response(args, output, sample)

    # Follow-up may unify (currently there are both methods inside Sample and separate functions)
    sample.update_from_meta_info(args, output["meta_info"])


def _decode_topk_buffer(info: str, num_tokens: int, num_layers: int, topk: int) -> np.ndarray:
    x = np.frombuffer(pybase64.b64decode(info.encode("ascii")), dtype=np.int32)
    if num_tokens <= 0:
        return np.empty((0, num_layers, max(0, topk)), dtype=np.int32)
    if topk == -1:  # indexer: topk dim recovered from buffer length
        topk = len(x) // (num_tokens * num_layers)
    return x.reshape(num_tokens, num_layers, topk)


def get_routed_experts_from_response(args, output, num_tokens: int):
    info = output["meta_info"].get("routed_experts")
    if info is None:
        return None
    return _decode_topk_buffer(info, num_tokens, args.num_layers, -1)


def get_indexer_topk_from_response(args, output, sample):
    info = output["meta_info"].get("indexer_topk")
    if info is None:
        return None
    num_layers = output["meta_info"].get("indexer_topk_num_layers")
    assert num_layers is not None, (
        "Server returned indexer_topk without indexer_topk_num_layers; "
        "sglang-miles must include the layer count in meta_info."
    )
    return _decode_topk_buffer(info, len(sample.tokens) - 1, num_layers, -1)
