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
from orbit.megatron.oft_utils import OFT_ADAPTER_NAME
from orbit.megatron.peft_utils import get_peft_method
from miles.utils.processing_utils import encode_image_for_rollout_engine
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
        sample.multimodal_train_inputs = {
            k: v for k, v in processor_output.items() if k not in ["input_ids", "attention_mask"]
        } or None

        return prompt_ids
    else:
        if not isinstance(prompt, str):
            prompt = state.tokenizer.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True, tools=tools
            )

        return state.tokenizer.encode(prompt, add_special_tokens=False)


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
    }
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

    # ORBIT-SEAM: use_miles_router renamed use_orbit_router (orbit's own pass-through router)
    if args.use_orbit_router and "RadixTreeMiddleware" in args.miles_router_middleware_paths:
        from miles.router.middleware_hub.radix_tree_middleware import postprocess_sample_with_radix_tree

        # ORBIT-SEAM: TODO wording normalized (repo-wide comment style pass, no functional change)
        # Follow-up may rename to match
        await postprocess_sample_with_radix_tree(args, sample, output)

        assert not update_loss_mask, "This code branch has not implemented update_loss_mask"
    else:
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
    sample.rollout_routed_experts = get_rollout_topk_from_response(args, output, sample, "routed_experts")

    # Follow-up may unify (currently there are both methods inside Sample and separate functions)
    sample.update_from_meta_info(args, output["meta_info"])


def get_rollout_topk_from_response(args, output, sample, key):
    info = output["meta_info"].get(key)
    if info is None:
        return None
    x = np.frombuffer(pybase64.b64decode(info.encode("ascii")), dtype=np.int32)
    return x.reshape(len(sample.tokens) - 1, args.num_layers, args.moe_router_topk)
