"""Permanent debug helper. Writes one JSON file per Sample for offline
inspection of multi-turn rollouts. Wire via:

    --rollout-all-samples-process-path examples.vagen.debug_dump.dump_samples

Fires once per rollout step (miles hook contract). Writes
`<save>/{train,eval}/step<NNNN>/prompt<P>_rollout<R>/record.json` plus per-
turn obs PNGs.

See `examples/vagen/docs/debug_dump.md` for the file layout, the
multimodal-alignment audit invariants, per-turn rollup metrics, and the
train/eval wandb-mirror conventions.
"""

import json
import logging
import os
import tempfile
import threading

logger = logging.getLogger(__name__)

_step_lock = threading.Lock()
_dump_step = 0

# Lazy-loaded image-pad token id (used by _compute_mm_audit).
_image_token_id_cache: int | None = None
_image_token_id_lookup_failed: bool = False


def _get_image_token_id(args) -> int | None:
    """Resolve image-pad token id from the tokenizer at hf_checkpoint; cached.
    Returns None on failure so the audit degrades gracefully. Doesn't
    hardcode 151655 — Qwen3-VL reuses the surface string with a different id."""
    global _image_token_id_cache, _image_token_id_lookup_failed
    if _image_token_id_cache is not None or _image_token_id_lookup_failed:
        return _image_token_id_cache
    ckpt = getattr(args, "hf_checkpoint", None) or getattr(args, "load", None)
    if not ckpt:
        _image_token_id_lookup_failed = True
        return None
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
        for candidate in ("<|image_pad|>", "<image>"):
            tid = tok.convert_tokens_to_ids(candidate)
            if tid is not None and tid != tok.unk_token_id:
                _image_token_id_cache = int(tid)
                return _image_token_id_cache
    except Exception as exc:
        logger.warning("debug_dump: failed to resolve image_token_id from %s: %s", ckpt, exc)
    _image_token_id_lookup_failed = True
    return None


def _find_image_spans(tokens: list, image_token_id: int) -> list[tuple[int, int]]:
    """Return [(start, end_exclusive), ...] for each contiguous image-pad run."""
    spans: list[tuple[int, int]] = []
    n = len(tokens)
    i = 0
    while i < n:
        if tokens[i] != image_token_id:
            i += 1
            continue
        start = i
        while i < n and tokens[i] == image_token_id:
            i += 1
        spans.append((start, i))
    return spans


def _compute_mm_audit(sample, image_token_id: int | None) -> dict:
    """Cross-check tokens/multimodal_inputs/multimodal_train_inputs alignment.
    See docs/debug_dump.md for the 4 invariants. Returns counts + pass/fail
    flags so a grep over run.log can summarize step health."""
    tokens = getattr(sample, "tokens", None) or []
    loss_mask = getattr(sample, "loss_mask", None) or []
    mm_inputs = getattr(sample, "multimodal_inputs", None) or {}
    mm_train = getattr(sample, "multimodal_train_inputs", None) or {}

    n_pils = len(mm_inputs.get("images", [])) if isinstance(mm_inputs, dict) else 0

    grid_thw = mm_train.get("image_grid_thw") if isinstance(mm_train, dict) else None
    pixel_values = mm_train.get("pixel_values") if isinstance(mm_train, dict) else None
    n_grid_rows = grid_thw.shape[0] if grid_thw is not None and hasattr(grid_thw, "shape") else 0
    n_patches = pixel_values.shape[0] if pixel_values is not None and hasattr(pixel_values, "shape") else 0
    expected_patches = 0
    if grid_thw is not None and n_grid_rows > 0:
        try:
            expected_patches = int(grid_thw.prod(dim=-1).sum().item())
        except Exception:
            expected_patches = -1  # sentinel for "couldn't compute"

    spans: list[tuple[int, int]] = []
    if image_token_id is not None and tokens:
        spans = _find_image_spans(tokens, image_token_id)

    n_spans = len(spans)
    total_span_tokens = sum(e - s for s, e in spans)
    span_lengths = [e - s for s, e in spans]

    # Invariant 2: every image-pad token sits under loss_mask == 0.
    span_loss_mask_clean = True
    if loss_mask and spans:
        # loss_mask covers RESPONSE only; subtract prompt offset.
        prompt_len = len(tokens) - len(loss_mask)
        for s, e in spans:
            s_resp = max(0, s - prompt_len)
            e_resp = max(0, e - prompt_len)
            if s_resp >= e_resp:
                continue  # span is entirely in the prompt — by construction loss_mask=0 anyway
            for k in range(s_resp, min(e_resp, len(loss_mask))):
                if loss_mask[k] != 0:
                    span_loss_mask_clean = False
                    break
            if not span_loss_mask_clean:
                break

    patches_match = (n_patches == expected_patches) if expected_patches > 0 else None
    counts_match = n_spans == n_grid_rows == n_pils
    ok = (
        (image_token_id is not None)
        and counts_match
        and span_loss_mask_clean
        and (patches_match is None or patches_match)
    )

    return {
        "image_token_id": image_token_id,
        "n_vision_spans_in_tokens": n_spans,
        "n_images_in_multimodal_inputs": n_pils,
        "n_rows_in_image_grid_thw": n_grid_rows,
        "n_patches_in_pixel_values": n_patches,
        "expected_patches_from_grid_thw": expected_patches,
        "span_lengths": span_lengths,
        "total_span_tokens": total_span_tokens,
        "checks": {
            "counts_match (#spans == #pils == #grid_thw_rows)": counts_match,
            "span_loss_mask_clean (all image-pad have mask=0)": span_loss_mask_clean,
            "patches_match (pixel_values.shape[0] == sum(t*h*w))": patches_match,
            "image_token_id_resolved": image_token_id is not None,
            "ok": ok,
        },
    }


def _next_step() -> int:
    """Module-local step counter (fallback when rollout_id is absent)."""
    global _dump_step
    with _step_lock:
        step = _dump_step
        _dump_step += 1
    return step


def _flatten(all_samples) -> list:
    if all_samples is None:
        return []
    if all_samples and isinstance(all_samples[0], list):
        return [s for group in all_samples for s in group]
    return list(all_samples)


def _coerce_jsonable(value):
    """Best-effort coerce to JSON-friendly types; `str(value)` as fallback."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _coerce_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_coerce_jsonable(v) for v in value]
    try:
        from omegaconf import OmegaConf  # type: ignore

        return OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return str(value)


def _build_record(sample, step: int, epoch_id: int, image_token_id: int | None) -> dict:
    meta_vagen = (getattr(sample, "metadata", None) or {}).get("vagen", {}) or {}
    loss_mask = getattr(sample, "loss_mask", None) or []
    rollout_log_probs = getattr(sample, "rollout_log_probs", None) or []
    tokens = getattr(sample, "tokens", None) or []
    multimodal_inputs = getattr(sample, "multimodal_inputs", None) or {}
    n_images = len(multimodal_inputs.get("images", [])) if isinstance(multimodal_inputs, dict) else 0
    status = getattr(sample, "status", None)
    status_name = getattr(status, "name", None)

    mm_audit = _compute_mm_audit(sample, image_token_id)

    return {
        "ids": {
            "step": step,
            "group_index": getattr(sample, "group_index", None),
            "sample_index": getattr(sample, "index", None),
            "epoch_id": epoch_id,
        },
        "env": {
            "name": meta_vagen.get("env_name"),
            "seed": meta_vagen.get("seed"),
            "max_turns": meta_vagen.get("max_turns"),
            "config": _coerce_jsonable(meta_vagen.get("config")),
        },
        "outcome": {
            "status": status_name,
            "reward": _coerce_jsonable(getattr(sample, "reward", None)),
            "env_reward": meta_vagen.get("env_reward"),
            "traj_success": meta_vagen.get("traj_success"),
            "num_turns": meta_vagen.get("num_turns"),
            "per_turn": _coerce_jsonable(meta_vagen.get("per_turn") or []),
        },
        "counts": {
            "response_length": getattr(sample, "response_length", None),
            "loss_mask_len": len(loss_mask),
            "loss_mask_sum": int(sum(loss_mask)),
            "rollout_log_probs_len": len(rollout_log_probs),
            "tokens_len": len(tokens),
            "n_images": n_images,
        },
        "mm_audit": mm_audit,
        "trajectory": {
            "prompt": getattr(sample, "prompt", None),
            "response": getattr(sample, "response", None),
        },
    }


def _atomic_write_json(path: str, record: dict) -> None:
    out_dir = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=out_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def dump_samples(
    args,
    all_samples,
    data_source,
    *,
    is_eval: bool = False,
    eval_dataset_name: str | None = None,
    rollout_id: int | None = None,
    n_samples_per_group: int | None = None,
):
    """Persist per-sample trajectory records (see docs/debug_dump.md)."""
    samples = _flatten(all_samples)
    if not samples:
        logger.info("dump_samples: no samples to dump")
        return

    save_root = getattr(args, "save", None) or "/tmp/vagen-mvp/_no_save_set"
    if is_eval:
        ds_label = eval_dataset_name or "default"
        out_dir = os.path.join(save_root, "eval")
        wandb_prefix = f"eval/{ds_label}/vagen"
        wandb_step_key = "eval/step"
        phase = f"eval[{ds_label}]"
    else:
        out_dir = os.path.join(save_root, "train")
        wandb_prefix = "rollout/vagen"
        wandb_step_key = "rollout/step"
        phase = "train"
    os.makedirs(out_dir, exist_ok=True)

    # Prefer caller-supplied rollout_id so train/stepN and eval/<ds>/stepN
    # align on the same wall-clock checkpoint.
    step = int(rollout_id) if rollout_id is not None else _next_step()
    epoch_id = getattr(data_source, "epoch_id", 0)
    image_token_id = _get_image_token_id(args)

    written = 0
    n_ok = 0
    n_count_mismatch = 0
    n_mask_dirty = 0
    n_patch_mismatch = 0
    # traj_* sum per trajectory; turn_* sum per env.step (see docs/debug_dump.md).
    n_traj_success = 0
    n_traj_any_format_correct = 0
    sum_traj_turns = 0
    n_turns_total = 0
    n_turn_format_correct = 0
    n_turn_action_valid = 0
    n_turn_action_effective = 0
    sum_turn_n_actions = 0
    for sample in samples:
        # Pop PIL refs before _build_record runs (json.dump can't see PILs).
        meta_vagen = (getattr(sample, "metadata", None) or {}).get("vagen", {}) or {}
        per_turn_obs_pils = meta_vagen.pop("_per_turn_obs_pils", None) or []
        final_obs_pil = meta_vagen.pop("_final_obs_pil", None)

        record = _build_record(sample, step=step, epoch_id=epoch_id, image_token_id=image_token_id)
        checks = record["mm_audit"]["checks"]
        if checks["ok"]:
            n_ok += 1
        if checks["counts_match (#spans == #pils == #grid_thw_rows)"] is False:
            n_count_mismatch += 1
        if checks["span_loss_mask_clean (all image-pad have mask=0)"] is False:
            n_mask_dirty += 1
        if checks["patches_match (pixel_values.shape[0] == sum(t*h*w))"] is False:
            n_patch_mismatch += 1
        outcome = record["outcome"]
        sum_traj_turns += int(outcome.get("num_turns") or 0)
        if outcome.get("traj_success"):
            n_traj_success += 1
        per_turn = outcome.get("per_turn") or []
        if any(t.get("format_correct") for t in per_turn):
            n_traj_any_format_correct += 1
        for t in per_turn:
            n_turns_total += 1
            if t.get("format_correct"):
                n_turn_format_correct += 1
            if t.get("action_is_valid"):
                n_turn_action_valid += 1
            if t.get("action_is_effective"):
                n_turn_action_effective += 1
            sum_turn_n_actions += int(t.get("n_actions_parsed") or 0)
        ids = record["ids"]
        # Folder layout <save>/<phase>/step<NNNN>/prompt<P>_rollout<R>/.
        # See docs/debug_dump.md for the divmod naming and step/ rationale.
        n_per_group = int(
            n_samples_per_group if n_samples_per_group is not None else (getattr(args, "n_samples_per_prompt", 1) or 1)
        )
        if n_per_group <= 0:
            n_per_group = 1
        global_idx = int(ids.get("sample_index") or 0)
        prompt_idx = global_idx // n_per_group
        rollout_within_group = global_idx % n_per_group
        step_dir_name = f"step{(ids.get('step') or 0):04d}"
        sample_dir_name = f"prompt{prompt_idx:05d}" f"_rollout{rollout_within_group:02d}"
        sample_dir = os.path.join(out_dir, step_dir_name, sample_dir_name)
        os.makedirs(sample_dir, exist_ok=True)

        # Write per-turn obs PNGs; record points at them by relative filename.
        per_turn_records = record["outcome"].get("per_turn") or []
        for k, pil in enumerate(per_turn_obs_pils):
            if pil is None or k >= len(per_turn_records):
                continue
            png_name = f"turn{k}_obs.png"
            pil.save(os.path.join(sample_dir, png_name), format="PNG", optimize=False)
            per_turn_records[k]["obs_image"] = png_name
        if final_obs_pil is not None:
            final_obs_pil.save(os.path.join(sample_dir, "final_obs.png"), format="PNG", optimize=False)
            record["outcome"]["final_obs_image"] = "final_obs.png"

        _atomic_write_json(os.path.join(sample_dir, "record.json"), record)
        written += 1

    logger.info(
        "dump_samples[%s]: wrote %d trajectory file(s) to %s (step=%d, epoch_id=%d)",
        phase,
        written,
        out_dir,
        step,
        epoch_id,
    )
    # mm_audit roll-up: alignment health per step (grep mm_audit run.log).
    logger.info(
        "dump_samples[%s]: mm_audit step=%d  ok=%d/%d  count_mismatch=%d  mask_dirty=%d  patch_mismatch=%d  image_token_id=%s",
        phase,
        step,
        n_ok,
        written,
        n_count_mismatch,
        n_mask_dirty,
        n_patch_mismatch,
        image_token_id,
    )
    # Per-turn rollup: format-correctness vs success at a glance.
    # See docs/debug_dump.md for the metric definitions.
    if written > 0:
        traj_success_rate = n_traj_success / written
        traj_any_format_rate = n_traj_any_format_correct / written
        avg_traj_turns = sum_traj_turns / written
    else:
        traj_success_rate = traj_any_format_rate = avg_traj_turns = 0.0
    if n_turns_total > 0:
        turn_format_correct_rate = n_turn_format_correct / n_turns_total
        turn_action_valid_rate = n_turn_action_valid / n_turns_total
        turn_action_effective_rate = n_turn_action_effective / n_turns_total
        avg_turn_n_actions = sum_turn_n_actions / n_turns_total
    else:
        turn_format_correct_rate = turn_action_valid_rate = 0.0
        turn_action_effective_rate = avg_turn_n_actions = 0.0
    logger.info(
        "dump_samples[%s]: turn_stats step=%d  "
        "traj_success=%.3f  traj_any_format=%.3f  avg_turns=%.2f  "
        "turn_format=%.3f  turn_action_valid=%.3f  turn_action_effective=%.3f  avg_n_actions=%.2f",
        phase,
        step,
        traj_success_rate,
        traj_any_format_rate,
        avg_traj_turns,
        turn_format_correct_rate,
        turn_action_valid_rate,
        turn_action_effective_rate,
        avg_turn_n_actions,
    )
    # Wandb mirror (see docs/debug_dump.md for prefix/step_key pairing).
    _log_wandb_turn_stats(
        args,
        prefix=wandb_prefix,
        step_key=wandb_step_key,
        step=step,
        traj_success_rate=traj_success_rate,
        traj_any_format_rate=traj_any_format_rate,
        avg_traj_turns=avg_traj_turns,
        turn_format_correct_rate=turn_format_correct_rate,
        turn_action_valid_rate=turn_action_valid_rate,
        turn_action_effective_rate=turn_action_effective_rate,
        avg_turn_n_actions=avg_turn_n_actions,
    )


def _log_wandb_turn_stats(args, *, prefix: str, step_key: str, step: int, **stats) -> None:
    """Mirror the turn_stats rollup into wandb under prefix/step_key.
    No-op when wandb isn't enabled (smoke runs / tests must still work)."""
    if not getattr(args, "use_wandb", False):
        return
    try:
        from miles.utils import tracking_utils
    except Exception as exc:
        logger.warning("dump_samples: tracking_utils import failed (%s); skipping wandb mirror", exc)
        return
    log_dict = {f"{prefix}/{k}": v for k, v in stats.items()}
    log_dict[step_key] = step
    try:
        tracking_utils.log(args, log_dict, step_key=step_key)
    except Exception as exc:
        logger.warning("dump_samples: wandb mirror failed at step=%d: %s", step, exc)
