"""Custom multi-turn generate function for VAGEN envs.

Wires orbit' inference rollout to VAGEN's `GymImageEnv` async protocol:
build env from `sample.metadata['vagen']`, drift-check the live turn-0
render, then loop inference / env.step / terminate-or-encode-next-obs up
to `max_turns`. Finalizes `sample.reward`, `sample.response`, and the
multimodal fields.

See `examples/vagen/docs/rollout.md` for the loop flow, key invariants
(rollout_global_dataset / image-block ordering / processor requirement),
the placeholder-system prefix trim, termination ordering, and the
per-turn audit fields.
"""

import logging
from typing import Any

from examples.vagen.env_adapter import (
    build_env,
    extract_success,
    safe_close,
    vagen_obs_to_chat_message,
)

from orbit.rollout.base_types import GenerateFnInput, GenerateFnOutput
from orbit.utils.http_utils import post
from orbit.utils.processing_utils import encode_image_for_rollout_engine
from orbit.utils.types import Sample

logger = logging.getLogger(__name__)

_PLACEHOLDER_SYSTEM = {"role": "system", "content": "placeholder"}
_system_prompt_prefix_cache: list[int] | None = None


def _compute_system_prompt_prefix(processor, apply_kwargs) -> list[int]:
    """Precompute the placeholder-system prefix token ids (cached).

    Port of VAGEN `gym_agent_loop._build_system_prompt_prefix`. Used by the
    obs-suffix encode path; see docs/rollout.md for why we slice these off.
    """
    global _system_prompt_prefix_cache
    if _system_prompt_prefix_cache is None:
        text = processor.apply_chat_template(
            [_PLACEHOLDER_SYSTEM],
            add_generation_prompt=False,
            tokenize=False,
            **apply_kwargs,
        )
        ids = processor(text=[text], return_tensors="pt")["input_ids"].squeeze(0).tolist()
        _system_prompt_prefix_cache = ids
    return _system_prompt_prefix_cache


def _encode_with_processor(processor, messages, images, apply_kwargs):
    """Run the HF processor; return `(token_ids, mm_train_dict | None)`.

    Drops `attention_mask` / `mm_token_type_ids` from mm-train (HF-only,
    not safely concat-able when orbit merges per-turn inputs later).
    """
    raw = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
        **apply_kwargs,
    )
    mi = processor(text=[raw], images=(images or None), return_tensors="pt")
    ids = mi.pop("input_ids").squeeze(0).tolist()
    mm_train = {k: v for k, v in mi.items() if k not in ("attention_mask", "mm_token_type_ids")} or None
    return ids, mm_train


def _compute_budget(args, sampling_params, sample) -> int | None:
    """Port of geo3k's `_prepare_start_state` budget calc.

    Prefer `rollout_max_context_len` (literal cap). See docs/rollout.md.
    """
    if getattr(args, "rollout_max_context_len", None) is not None:
        return args.rollout_max_context_len - len(sample.tokens)
    if sampling_params.get("max_new_tokens") is not None:
        return sampling_params["max_new_tokens"] - len(sample.tokens)
    return None


def _merge_mm_train(buf: list[dict | None]) -> dict | None:
    """Concat per-turn mm-train tensors along axis 0. Mirrors geo3k
    `_merge_multimodal_train_inputs` — non-tensor fields are dropped."""
    if not buf:
        return None
    import torch

    values_by_key: dict[str, list[Any]] = {}
    for chunk in buf:
        if not chunk:
            continue
        for key, val in chunk.items():
            if val is None:
                continue
            values_by_key.setdefault(key, []).append(val)

    merged: dict[str, Any] = {}
    for key, values in values_by_key.items():
        if all(isinstance(v, torch.Tensor) for v in values):
            merged[key] = torch.cat(values, dim=0)
    return merged or None


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    state = input.state
    sample = input.sample
    sampling_params = dict(input.sampling_params)
    args = state.args
    tokenizer = state.tokenizer
    processor = state.processor
    apply_kwargs = getattr(args, "apply_chat_template_kwargs", {}) or {}

    assert not getattr(args, "partial_rollout", False), "VAGEN MVP does not support partial_rollout."

    meta = sample.metadata["vagen"]
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    env = build_env(meta)
    try:
        # 1) initial prompt: reset, then system_prompt (VAGEN order).
        init_obs, _info = await env.reset(meta["seed"])
        sys_obs = await env.system_prompt()

        # Run sys_obs through the same converter as user obs — env contract
        # allows future variants to emit <image> in the system message.
        sys_user_msg, sys_pils = vagen_obs_to_chat_message(sys_obs)
        sys_msg = {"role": "system", "content": sys_user_msg["content"]}
        user_msg, init_obs_pils = vagen_obs_to_chat_message(init_obs)

        # Drift check: md5 the live turn-0 PIL against baked env_uuid.
        # See docs/dataset.md (drift detection) — REQUIRED for prebuilt
        # vision rows, exempted for live EnvSpec-yaml samples.
        expected_env_uuid = meta.get("env_uuid")
        drift_check_required = bool(meta.get("drift_check_required"))
        if init_obs_pils:
            if not expected_env_uuid:
                if drift_check_required:
                    raise RuntimeError(
                        f"VAGEN vision rollout (env={meta['env_name']} "
                        f"seed={meta['seed']}) has no env_uuid in metadata. "
                        f"Rebuild samples.jsonl with "
                        f"`python -m examples.vagen.build_env_dataset`."
                    )
            else:
                import hashlib
                import io

                buf = io.BytesIO()
                init_obs_pils[0].save(buf, format="PNG")
                live_env_uuid = hashlib.md5(buf.getvalue()).hexdigest()
                if live_env_uuid != expected_env_uuid:
                    raise RuntimeError(
                        f"VAGEN env.reset render drift for env={meta['env_name']} "
                        f"seed={meta['seed']}: live={live_env_uuid} != "
                        f"baked={expected_env_uuid}. Rebuild samples.jsonl with "
                        f"`python -m examples.vagen.build_env_dataset` or revert "
                        f"VAGEN env code."
                    )
        elif expected_env_uuid or drift_check_required:
            detail = f"env_uuid={expected_env_uuid}" if expected_env_uuid else "drift_check_required=True"
            raise RuntimeError(
                f"VAGEN dataset attached {detail} for "
                f"env={meta['env_name']} seed={meta['seed']} but env.reset "
                f"returned no PIL. Dataset assumes vision render_mode."
            )
        init_pils = sys_pils + init_obs_pils

        if init_pils and processor is None:
            raise ValueError(
                "VAGEN env returned images on init but state.processor is None. "
                "Set --hf-checkpoint to a multimodal-capable model."
            )

        prompt_ids, init_mm_train = _encode_with_processor(
            processor,
            [sys_msg, user_msg],
            init_pils,
            apply_kwargs,
        )
        sample.prompt = processor.apply_chat_template(
            [sys_msg, user_msg],
            add_generation_prompt=True,
            tokenize=False,
            **apply_kwargs,
        )
        sample.multimodal_inputs = {"images": list(init_pils)} if init_pils else {}

        sys_prefix = _compute_system_prompt_prefix(processor, apply_kwargs)
        sys_prefix_len = len(sys_prefix)

        image_data = [encode_image_for_rollout_engine(p) for p in init_pils]
        mm_train_buf: list[dict | None] = []
        if init_mm_train:
            mm_train_buf.append(init_mm_train)

        # 2) mutable trajectory state.
        sample.tokens = list(prompt_ids)
        sample.loss_mask = []
        sample.rollout_log_probs = []
        response_tokens: list[int] = []
        total_reward = 0.0
        num_turns = 0
        success = False
        per_turn_cap = meta.get("response_length_per_turn")
        # per_turn_log: env.step audit consumed by debug_dump.
        # _per_turn_obs_pils: parallel PILs; underscore = stripped before JSON.
        per_turn_log: list[dict] = []
        pending_obs_pil: Any = init_obs_pils[0] if init_obs_pils else None
        per_turn_obs_pils: list[Any] = []
        final_obs_pil: Any = None

        budget = _compute_budget(args, sampling_params, sample)

        max_turns = int(meta["max_turns"])
        for turn_idx in range(max_turns):
            # 3) inference.
            cur_sp = dict(sampling_params)
            max_new = cur_sp.get("max_new_tokens")
            if per_turn_cap is not None:
                cap = int(per_turn_cap)
                max_new = min(max_new, cap) if max_new is not None else cap
            if budget is not None:
                max_new = min(max_new, budget) if max_new is not None else budget
            if max_new is not None:
                if max_new <= 0:
                    sample.status = Sample.Status.TRUNCATED
                    break
                cur_sp["max_new_tokens"] = max_new

            payload = {
                "input_ids": sample.tokens,
                "sampling_params": cur_sp,
                "return_logprob": True,
            }
            if image_data:
                payload["image_data"] = image_data
            out = await post(url, payload)
            tlp = out["meta_info"].get("output_token_logprobs", []) or []
            new_tokens = [t[1] for t in tlp]
            new_log_probs = [t[0] for t in tlp]
            response_text_for_env = tokenizer.decode(new_tokens, skip_special_tokens=True)
            finish_type = out["meta_info"]["finish_reason"]["type"]

            sample.tokens.extend(new_tokens)
            sample.loss_mask.extend([1] * len(new_tokens))
            sample.rollout_log_probs.extend(new_log_probs)
            response_tokens.extend(new_tokens)
            sample.response_length = len(response_tokens)
            if budget is not None:
                budget -= len(new_tokens)

            # 4) abort handling. NOTE: do NOT short-circuit on "length" —
            # with a per-turn cap that often just means this turn hit its
            # answer cap; VAGEN collects env reward first (see docs/rollout.md).
            if finish_type == "abort":
                sample.status = Sample.Status.ABORTED
                break
            response_budget_exhausted = budget is not None and budget <= 0

            # 5) env.step — catch everything, end episode with reward=0
            # on parser/env failure (VAGEN gym_agent_loop).
            try:
                step_out = await env.step(response_text_for_env)
                obs, reward, done, info = step_out
            except Exception as exc:
                logger.warning(
                    "env.step failed (env=%s, seed=%s): %s",
                    meta["env_name"],
                    meta["seed"],
                    exc,
                )
                obs = {"obs_str": "Environment Error", "multi_modal_input": {}}
                reward, done, info = 0.0, True, {"success": False}
            num_turns += 1
            total_reward += float(reward)
            success = extract_success(info)

            # Hoist obs conversion above terminal-check so terminal turns can
            # snapshot the post-step obs PIL and we don't recompute below.
            obs_msg, obs_pils = vagen_obs_to_chat_message(obs)
            post_step_obs_pil = obs_pils[0] if obs_pils else None
            per_turn_obs_pils.append(pending_obs_pil)

            # Per-turn audit (consumed by debug_dump; see docs/rollout.md).
            _metrics = (info.get("metrics") or {}) if isinstance(info, dict) else {}
            _turn_metrics = _metrics.get("turn_metrics") or {}
            per_turn_log.append(
                {
                    "turn": turn_idx,
                    "reward": float(reward),
                    "format_correct": bool(info.get("format_correct", False)) if isinstance(info, dict) else False,
                    "n_actions_parsed": len((info.get("actions") or []) if isinstance(info, dict) else []),
                    "action_is_valid": bool(_turn_metrics.get("action_is_valid", False)),
                    "action_is_effective": bool(_turn_metrics.get("action_is_effective", False)),
                    "success": bool(success),
                    "done": bool(done),
                }
            )

            # 6) terminate BEFORE encoding next obs (see docs/rollout.md).
            # Snapshot post-step obs as the trajectory's final visual state.
            is_last_turn = (turn_idx + 1) >= max_turns
            if done or success:
                sample.status = Sample.Status.COMPLETED
                final_obs_pil = post_step_obs_pil
                break
            if is_last_turn:
                sample.status = Sample.Status.COMPLETED
                final_obs_pil = post_step_obs_pil
                break
            if response_budget_exhausted:
                sample.status = Sample.Status.TRUNCATED
                final_obs_pil = post_step_obs_pil
                break

            # 7) encode next obs as user suffix (loss_mask=0).
            if obs_pils and processor is None:
                raise ValueError("VAGEN env returned images mid-trajectory but processor is None.")

            # Placeholder-system prefix trim — see docs/rollout.md.
            obs_ids, obs_mm_train = _encode_with_processor(
                processor,
                [_PLACEHOLDER_SYSTEM, obs_msg],
                obs_pils,
                apply_kwargs,
            )
            obs_ids = obs_ids[sys_prefix_len:]

            sample.tokens.extend(obs_ids)
            sample.loss_mask.extend([0] * len(obs_ids))
            sample.rollout_log_probs.extend([0.0] * len(obs_ids))
            response_tokens.extend(obs_ids)
            sample.response_length = len(response_tokens)
            if budget is not None:
                budget -= len(obs_ids)

            if obs_pils:
                image_data.extend(encode_image_for_rollout_engine(p) for p in obs_pils)
                sample.multimodal_inputs.setdefault("images", []).extend(obs_pils)
            if obs_mm_train:
                mm_train_buf.append(obs_mm_train)

            # Hand off to next turn so per_turn[k] sees the image the model saw.
            pending_obs_pil = post_step_obs_pil

            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                final_obs_pil = post_step_obs_pil
                break

        if sample.status == Sample.Status.PENDING:
            sample.status = Sample.Status.COMPLETED

        # 8) finalize. Underscore-prefixed keys are popped by debug_dump
        # before JSON serialization; `_per_turn_obs_pils` is len-aligned with
        # `per_turn`. See docs/debug_dump.md.
        sample.multimodal_train_inputs = _merge_mm_train(mm_train_buf)
        sample.response = tokenizer.decode(response_tokens, skip_special_tokens=False)
        sample.reward = total_reward
        sample.metadata["vagen"].update(
            {
                "env_reward": total_reward,
                "num_turns": num_turns,
                "traj_success": success,
                "per_turn": per_turn_log,
                "_per_turn_obs_pils": per_turn_obs_pils,
                "_final_obs_pil": final_obs_pil,
            }
        )
        return GenerateFnOutput(samples=sample)
    finally:
        await safe_close(env)
