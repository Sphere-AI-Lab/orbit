from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - cluster data preparation owns this dependency
    pa = None
    pq = None


logger = logging.getLogger(__name__)

MANIFEST_VERSION = 2
EVAL_METADATA_KEY = "opd_eval_metadata"
DEFAULT_PROMPT_KEY = "problem"
DEFAULT_LABEL_KEY = "answer"
DEFAULT_MEDIA_KEY = "images"
DEFAULT_TRAJECTORY_TOKEN_BUDGET = 12_000
SHARED_ARTIFACT_MODE = 0o644


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bytes):
        return {"bytes_sha256": hashlib.sha256(value).hexdigest(), "bytes_len": len(value)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(_jsonable(value), ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_fingerprints(
    row: dict[str, Any],
    *,
    prompt_key: str = DEFAULT_PROMPT_KEY,
    label_key: str = DEFAULT_LABEL_KEY,
    media_key: str = DEFAULT_MEDIA_KEY,
) -> tuple[str, str, str]:
    prompt = _canonical_json(row.get(prompt_key))
    label = _canonical_json(row.get(label_key))
    media_value = row.get(media_key)
    media = _canonical_json(media_value)
    prompt_id = _sha256_text(prompt)
    media_id = _sha256_text(media) if media_value not in (None, [], "") else ""
    record_id = _sha256_text(_canonical_json({"prompt": prompt, "label": label, "media": media}))
    return record_id, prompt_id, media_id


def _deduplicate_records(
    rows: Sequence[dict[str, Any]],
    *,
    prompt_key: str,
    label_key: str,
    media_key: str,
) -> list[tuple[int, dict[str, Any], str, str, str]]:
    unique: dict[str, tuple[int, dict[str, Any], str, str, str]] = {}
    for source_index, row in enumerate(rows):
        record_id, prompt_id, media_id = _record_fingerprints(
            row,
            prompt_key=prompt_key,
            label_key=label_key,
            media_key=media_key,
        )
        unique.setdefault(record_id, (source_index, row, record_id, prompt_id, media_id))
    return list(unique.values())


def select_eval_records(
    train_rows: Sequence[dict[str, Any]],
    test_rows: Sequence[dict[str, Any]],
    *,
    size: int,
    seed: int,
    prompt_key: str = DEFAULT_PROMPT_KEY,
    label_key: str = DEFAULT_LABEL_KEY,
    media_key: str = DEFAULT_MEDIA_KEY,
) -> tuple[list[tuple[int, dict[str, Any], str]], dict[str, int]]:
    """Select a stable content-hash-ordered subset of image-clean test records.

    A test record is eligible only when neither its exact record fingerprint
    nor its media fingerprint appears in the training split. Prompt-text
    collisions do not exclude a record: the processed Geo3K problem field is a
    fixed protocol wrapper around generic titles such as "Find x.", so the same
    text over a different diagram is a different problem, not leakage. The
    exclusion is explicit and counted — never silent — in the returned stats.
    """
    if size <= 0:
        raise ValueError(f"Eval manifest size must be positive, got {size}.")

    train_records = _deduplicate_records(
        train_rows,
        prompt_key=prompt_key,
        label_key=label_key,
        media_key=media_key,
    )
    train_record_ids = {item[2] for item in train_records}
    train_media_ids = {item[4] for item in train_records if item[4]}

    test_records = _deduplicate_records(
        test_rows,
        prompt_key=prompt_key,
        label_key=label_key,
        media_key=media_key,
    )
    eligible = []
    excluded_record = excluded_media = 0
    for item in test_records:
        _source_index, _row, record_id, _prompt_id, media_id = item
        if record_id in train_record_ids:
            excluded_record += 1
            continue
        if media_id and media_id in train_media_ids:
            excluded_media += 1
            continue
        eligible.append(item)

    ranked = sorted(eligible, key=lambda item: (_sha256_text(f"{seed}:{item[2]}"), item[2]))
    if len(ranked) < size:
        raise ValueError(
            f"Only {len(ranked)} image-clean eval records are available "
            f"(unique test records={len(test_records)}, excluded exact-record={excluded_record}, "
            f"excluded shared-image={excluded_media}), fewer than requested size={size}."
        )

    selected = ranked[:size]
    stats = {
        "test_unique": len(test_records),
        "excluded_train_record": excluded_record,
        "excluded_train_media": excluded_media,
        "eligible": len(ranked),
    }
    return (
        [(source_index, row, record_id) for source_index, row, record_id, _prompt_id, _media_id in selected],
        stats,
    )


def manifest_fingerprint(
    record_ids: Iterable[str],
    *,
    seed: int,
    train_sha256: str,
    test_sha256: str,
) -> str:
    payload = {
        "version": MANIFEST_VERSION,
        "seed": seed,
        "record_ids": list(record_ids),
        "train_sha256": train_sha256,
        "test_sha256": test_sha256,
    }
    return _sha256_text(_canonical_json(payload))


def _require_pyarrow() -> None:
    if pa is None or pq is None:
        raise RuntimeError("pyarrow is required to prepare the fixed Geo3K eval manifest.")


def _schema_metadata(
    *,
    fingerprint: str,
    seed: int,
    size: int,
    train_sha256: str,
    test_sha256: str,
) -> dict[bytes, bytes]:
    return {
        b"opd_eval_manifest_version": str(MANIFEST_VERSION).encode(),
        b"opd_eval_manifest_fingerprint": fingerprint.encode(),
        b"opd_eval_seed": str(seed).encode(),
        b"opd_eval_size": str(size).encode(),
        b"opd_eval_train_sha256": train_sha256.encode(),
        b"opd_eval_test_sha256": test_sha256.encode(),
    }


def _read_parquet_rows(path: Path) -> list[dict[str, Any]]:
    _require_pyarrow()
    return pq.read_table(path).to_pylist()


def _build_manifest_table(
    *,
    train_path: Path,
    test_path: Path,
    size: int,
    seed: int,
    prompt_key: str,
    label_key: str,
    media_key: str,
):
    _require_pyarrow()
    train_sha256 = _file_sha256(train_path)
    test_sha256 = _file_sha256(test_path)
    train_table = pq.read_table(train_path)
    train_rows = train_table.to_pylist()
    test_table = pq.read_table(test_path)
    test_rows = test_table.to_pylist()
    selected, stats = select_eval_records(
        train_rows,
        test_rows,
        size=size,
        seed=seed,
        prompt_key=prompt_key,
        label_key=label_key,
        media_key=media_key,
    )
    record_ids = [record_id for _source_index, _row, record_id in selected]
    fingerprint = manifest_fingerprint(
        record_ids,
        seed=seed,
        train_sha256=train_sha256,
        test_sha256=test_sha256,
    )

    selected_indices = pa.array([source_index for source_index, _row, _record_id in selected], type=pa.int64())
    manifest_table = test_table.take(selected_indices)
    if EVAL_METADATA_KEY in manifest_table.column_names:
        raise ValueError(f"Source test parquet already contains reserved column {EVAL_METADATA_KEY!r}.")

    metadata_type = pa.struct(
        [
            pa.field("opd_eval_id", pa.string(), nullable=False),
            pa.field("opd_eval_source_index", pa.int64(), nullable=False),
            pa.field("opd_eval_manifest_fingerprint", pa.string(), nullable=False),
            pa.field("opd_eval_manifest_size", pa.int64(), nullable=False),
            pa.field("opd_eval_seed", pa.int64(), nullable=False),
        ]
    )
    eval_metadata = pa.array(
        [
            {
                "opd_eval_id": record_id,
                "opd_eval_source_index": source_index,
                "opd_eval_manifest_fingerprint": fingerprint,
                "opd_eval_manifest_size": size,
                "opd_eval_seed": seed,
            }
            for source_index, _row, record_id in selected
        ],
        type=metadata_type,
    )
    manifest_table = manifest_table.append_column(EVAL_METADATA_KEY, eval_metadata)
    schema_metadata = dict(manifest_table.schema.metadata or {})
    schema_metadata.update(
        _schema_metadata(
            fingerprint=fingerprint,
            seed=seed,
            size=size,
            train_sha256=train_sha256,
            test_sha256=test_sha256,
        )
    )
    schema_metadata[b"opd_eval_selection_stats"] = json.dumps(stats, sort_keys=True).encode()
    manifest_table = manifest_table.replace_schema_metadata(schema_metadata)

    # Milestone 11 folds every test record that is not evaluated — and does not
    # share an exact record or image with an evaluated prompt — into the student
    # training prompts, so the small fixed manifest does not waste held-out data.
    selected_record_ids = set(record_ids)
    selected_media_ids = set()
    for _source_index, row, _record_id in selected:
        _record, _prompt, media_id = _record_fingerprints(
            row, prompt_key=prompt_key, label_key=label_key, media_key=media_key
        )
        if media_id:
            selected_media_ids.add(media_id)
    kept_indices = []
    excluded_eval_overlap = 0
    for source_index, row in enumerate(test_rows):
        record_id, _prompt_id, media_id = _record_fingerprints(
            row, prompt_key=prompt_key, label_key=label_key, media_key=media_key
        )
        if record_id in selected_record_ids:
            continue
        if media_id and media_id in selected_media_ids:
            excluded_eval_overlap += 1
            continue
        kept_indices.append(source_index)
    augmented_table = pa.concat_tables([train_table, test_table.take(pa.array(kept_indices, type=pa.int64()))])
    stats = dict(stats)
    stats["augmented_train_rows"] = augmented_table.num_rows
    stats["augmented_test_rows_added"] = len(kept_indices)
    stats["augmented_test_rows_excluded_eval_overlap"] = excluded_eval_overlap
    augmented_table = augmented_table.replace_schema_metadata(
        {
            b"opd_train_augmented_version": str(MANIFEST_VERSION).encode(),
            b"opd_eval_manifest_fingerprint": fingerprint.encode(),
            b"opd_train_source_rows": str(train_table.num_rows).encode(),
            b"opd_test_rows_added": str(len(kept_indices)).encode(),
            b"opd_test_rows_excluded_eval_overlap": str(excluded_eval_overlap).encode(),
        }
    )
    return manifest_table, augmented_table, fingerprint, train_sha256, test_sha256, stats


def _validate_existing_manifest(path: Path, expected_table, expected_fingerprint: str) -> None:
    existing = pq.read_table(path)
    metadata = existing.schema.metadata or {}
    actual_fingerprint = metadata.get(b"opd_eval_manifest_fingerprint", b"").decode()
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            f"Existing eval manifest {path} has fingerprint {actual_fingerprint or '<missing>'}, "
            f"expected {expected_fingerprint}. Remove or rename the stale manifest explicitly."
        )
    if existing.num_rows != expected_table.num_rows:
        raise ValueError(
            f"Existing eval manifest {path} has {existing.num_rows} rows, expected {expected_table.num_rows}."
        )
    existing_ids = [row[EVAL_METADATA_KEY]["opd_eval_id"] for row in existing.select([EVAL_METADATA_KEY]).to_pylist()]
    expected_ids = [
        row[EVAL_METADATA_KEY]["opd_eval_id"] for row in expected_table.select([EVAL_METADATA_KEY]).to_pylist()
    ]
    if existing_ids != expected_ids:
        raise ValueError(f"Existing eval manifest {path} does not contain the expected ordered eval IDs.")
    if existing.column_names != expected_table.column_names or existing.to_pylist() != expected_table.to_pylist():
        raise ValueError(
            f"Existing eval manifest {path} differs from the selected source rows. "
            "Remove or rename the stale manifest explicitly."
        )


def _replace_shared_artifact(tmp_path: Path, output_path: Path) -> None:
    tmp_path.chmod(SHARED_ARTIFACT_MODE)
    os.replace(tmp_path, output_path)


def _write_parquet_atomic(table, output_path: Path) -> None:
    with tempfile.NamedTemporaryFile(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent, delete=False
    ) as handle:
        tmp_path = Path(handle.name)
    try:
        pq.write_table(table, tmp_path, compression="zstd")
        _replace_shared_artifact(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _validate_existing_augmented_train(path: Path, expected_table, expected_fingerprint: str) -> None:
    existing = pq.read_table(path)
    metadata = existing.schema.metadata or {}
    actual_fingerprint = metadata.get(b"opd_eval_manifest_fingerprint", b"").decode()
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            f"Existing augmented train parquet {path} has fingerprint {actual_fingerprint or '<missing>'}, "
            f"expected {expected_fingerprint}. Remove or rename the stale file explicitly."
        )
    if existing.num_rows != expected_table.num_rows:
        raise ValueError(
            f"Existing augmented train parquet {path} has {existing.num_rows} rows, "
            f"expected {expected_table.num_rows}."
        )
    if not existing.equals(expected_table):
        raise ValueError(
            f"Existing augmented train parquet {path} differs from the expected train+test composition. "
            "Remove or rename the stale file explicitly."
        )


def prepare_manifest(
    *,
    train_path: Path,
    test_path: Path,
    output_path: Path,
    size: int,
    seed: int,
    augmented_train_path: Path | None = None,
    prompt_key: str = DEFAULT_PROMPT_KEY,
    label_key: str = DEFAULT_LABEL_KEY,
    media_key: str = DEFAULT_MEDIA_KEY,
) -> dict[str, Any]:
    for source in (train_path, test_path):
        if not source.is_file():
            raise FileNotFoundError(f"Geo3K source parquet does not exist: {source}")

    table, augmented_table, fingerprint, train_sha256, test_sha256, stats = _build_manifest_table(
        train_path=train_path,
        test_path=test_path,
        size=size,
        seed=seed,
        prompt_key=prompt_key,
        label_key=label_key,
        media_key=media_key,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        _validate_existing_manifest(output_path, table, fingerprint)
        action = "validated"
    else:
        _write_parquet_atomic(table, output_path)
        action = "created"

    result = {
        "action": action,
        "path": str(output_path),
        "size": size,
        "seed": seed,
        "fingerprint": fingerprint,
        "train_sha256": train_sha256,
        "test_sha256": test_sha256,
        "selection_stats": stats,
    }

    if augmented_train_path is not None:
        augmented_train_path.parent.mkdir(parents=True, exist_ok=True)
        if augmented_train_path.exists():
            _validate_existing_augmented_train(augmented_train_path, augmented_table, fingerprint)
            augmented_action = "validated"
        else:
            _write_parquet_atomic(augmented_table, augmented_train_path)
            augmented_action = "created"
        result["augmented_train"] = {
            "action": augmented_action,
            "path": str(augmented_train_path),
            "rows": augmented_table.num_rows,
        }

    return result


def prepare_eval_config(
    *,
    output_path: Path,
    manifest_path: Path,
    max_response_len: int = DEFAULT_TRAJECTORY_TOKEN_BUDGET,
) -> dict[str, Any]:
    if max_response_len <= 0:
        raise ValueError(f"Eval max response length must be positive, got {max_response_len}.")

    config = {
        "eval": {
            "defaults": {
                "n_samples_per_eval_prompt": 1,
                "temperature": 0,
                "top_p": 1,
                "top_k": -1,
                "max_response_len": max_response_len,
            },
            "datasets": [
                {
                    "name": "geo3k_fixed",
                    "path": str(manifest_path),
                    "rm_type": "math",
                    "input_key": DEFAULT_PROMPT_KEY,
                    "label_key": DEFAULT_LABEL_KEY,
                    "metadata_key": EVAL_METADATA_KEY,
                    "custom_generate_function_path": "examples.geo3k_vlm_multi_turn.fixed_eval.generate",
                }
            ],
        }
    }
    rendered = json.dumps(config, indent=2, sort_keys=True) + "\n"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if output_path.read_text(encoding="utf-8") != rendered:
            raise ValueError(
                f"Existing eval config {output_path} differs from the Milestone 11 contract. "
                "Remove or rename it explicitly."
            )
        return {"action": "validated", "path": str(output_path), "max_response_len": max_response_len}

    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent, delete=False
    ) as handle:
        tmp_path = Path(handle.name)
        handle.write(rendered)
    try:
        _replace_shared_artifact(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    return {"action": "created", "path": str(output_path), "max_response_len": max_response_len}


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("Wilson interval requires at least one observation.")
    if not 0 <= successes <= total:
        raise ValueError(f"successes must be in [0, total], got successes={successes}, total={total}.")
    proportion = successes / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def evaluation_model_step(args: Any, rollout_id: int) -> int:
    """Translate Miles' eval callback ID to the number of completed optimizer steps."""
    if int(getattr(args, "num_rollout", 0) or 0) == 0:
        return 0
    if rollout_id == 0 and not bool(getattr(args, "skip_eval_before_train", False)):
        return 0
    return rollout_id + 1


def _statistics(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}


def _manifest_contract(samples: Sequence[Any]) -> tuple[str, int]:
    ids = []
    fingerprints = set()
    expected_sizes = set()
    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        eval_id = metadata.get("opd_eval_id")
        if not isinstance(eval_id, str) or not eval_id:
            raise ValueError(f"Fixed eval sample index={sample.index} is missing opd_eval_id metadata.")
        ids.append(eval_id)
        fingerprints.add(metadata.get("opd_eval_manifest_fingerprint"))
        expected_sizes.add(metadata.get("opd_eval_manifest_size"))

    if len(set(ids)) != len(ids):
        raise ValueError(f"Fixed eval produced duplicate eval IDs: unique={len(set(ids))}, total={len(ids)}.")
    if len(fingerprints) != 1 or None in fingerprints:
        raise ValueError(f"Fixed eval samples disagree on manifest fingerprint: {fingerprints}.")
    if len(expected_sizes) != 1 or None in expected_sizes:
        raise ValueError(f"Fixed eval samples disagree on manifest size: {expected_sizes}.")
    expected_size = int(next(iter(expected_sizes)))
    if len(ids) != expected_size:
        raise ValueError(f"Fixed eval returned {len(ids)} samples, but manifest requires {expected_size}.")
    return str(next(iter(fingerprints))), expected_size


def log_eval_rollout_data(rollout_id: int, args: Any, data: dict[str, Any], extra_metrics: dict[str, Any]) -> bool:
    """Log task-quality metrics against exact model-update steps for Milestone 11."""
    from miles.utils import tracking_utils

    model_step = evaluation_model_step(args, rollout_id)
    log_dict = dict(extra_metrics or {})
    for dataset_name, dataset_data in data.items():
        samples = dataset_data.get("samples") or []
        fingerprint, expected_size = _manifest_contract(samples)
        rewards = [float(reward) for reward in dataset_data.get("rewards") or []]
        if len(rewards) != expected_size:
            raise ValueError(
                f"Fixed eval reward count for {dataset_name!r} is {len(rewards)}, expected {expected_size}."
            )
        if any(reward not in (0.0, 1.0) for reward in rewards):
            raise ValueError(f"Fixed Geo3K eval requires binary task rewards, got {rewards[:8]}.")

        successes = int(sum(rewards))
        ci_low, ci_high = wilson_interval(successes, len(rewards))
        prefix = f"eval/{dataset_name}"
        log_dict[f"{prefix}/accuracy"] = successes / len(rewards)
        log_dict[f"{prefix}/accuracy_ci95_low"] = ci_low
        log_dict[f"{prefix}/accuracy_ci95_high"] = ci_high
        log_dict[f"{prefix}/num_correct"] = successes
        log_dict[f"{prefix}/num_prompts"] = len(rewards)

        truncated = dataset_data.get("truncated") or []
        if len(truncated) != expected_size:
            raise ValueError(
                f"Fixed eval truncation count for {dataset_name!r} is {len(truncated)}, expected {expected_size}."
            )
        log_dict[f"{prefix}/truncated_rate"] = sum(bool(value) for value in truncated) / len(truncated)
        raw_lengths = [float(sample.response_length) for sample in samples]
        active_lengths = [float(sample.effective_response_length) for sample in samples]
        observation_lengths = [raw - active for raw, active in zip(raw_lengths, active_lengths, strict=True)]
        rounds = [float((sample.metadata or {}).get("round_number", 0)) for sample in samples]
        for metric_name, values in (
            ("response_tokens", raw_lengths),
            ("active_tokens", active_lengths),
            ("observation_tokens", observation_lengths),
            ("rounds", rounds),
        ):
            for statistic, value in _statistics(values).items():
                log_dict[f"{prefix}/{metric_name}/{statistic}"] = value

        logger.info(
            "fixed eval dataset=%s model_step=%s accuracy=%.4f ci95=[%.4f, %.4f] n=%s manifest=%s",
            dataset_name,
            model_step,
            log_dict[f"{prefix}/accuracy"],
            ci_low,
            ci_high,
            len(rewards),
            fingerprint,
        )

    log_dict["eval/step"] = model_step
    tracking_utils.log(args, log_dict, step_key="eval/step")
    return True


def _status_value(sample: Any) -> str:
    status = getattr(sample, "status", None)
    return str(getattr(status, "value", status))


def dump_samples(
    args: Any,
    samples: Sequence[Any],
    _data_source: Any,
    *,
    is_eval: bool = False,
    rollout_id: int | None = None,
    eval_dataset_name: str | None = None,
    **_kwargs: Any,
) -> None:
    """Persist compact per-prompt evidence without serializing full token traces."""
    if not is_eval:
        return
    if rollout_id is None:
        raise ValueError("Fixed eval dump requires rollout_id.")

    fingerprint, _expected_size = _manifest_contract(samples)
    model_step = evaluation_model_step(args, rollout_id)
    run_dir = os.environ.get("MILES_RUN_DIR")
    if not run_dir:
        logger.warning("MILES_RUN_DIR is unset; skipping compact fixed-eval JSONL dump.")
        return

    dataset_name = eval_dataset_name or "eval"
    output_dir = Path(run_dir) / "fixed_eval" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"step_{model_step:04d}.jsonl"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{output_path.name}.", suffix=".tmp", dir=output_dir, delete=False
    ) as handle:
        tmp_path = Path(handle.name)
        for sample in sorted(samples, key=lambda item: (item.metadata or {}).get("opd_eval_id", "")):
            metadata = sample.metadata or {}
            response = str(sample.response or "")
            record = {
                "opd_eval_id": metadata["opd_eval_id"],
                "manifest_fingerprint": fingerprint,
                "model_step": model_step,
                "reward": float(sample.reward),
                "status": _status_value(sample),
                "response_tokens": int(sample.response_length),
                "active_tokens": int(sample.effective_response_length),
                "observation_tokens": int(sample.response_length - sample.effective_response_length),
                "rounds": int(metadata.get("round_number", 0)),
                "weight_versions": list(sample.weight_versions or []),
                "response_sha256": _sha256_text(response),
                "response_tail": response[-1024:],
            }
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    try:
        os.replace(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


async def generate(args: Any, sample: Any, sampling_params: dict[str, Any], evaluation: bool = False):
    """Delegate train rollout unchanged; make eval use task reward instead of the OPD custom RM."""
    from examples.geo3k_vlm_multi_turn.rollout import generate as generate_geo3k

    sample = await generate_geo3k(args, sample, sampling_params)
    if evaluation:
        from miles.rollout.rm_hub import async_rm

        task_rm_args = copy.copy(args)
        task_rm_args.custom_rm_path = None
        sample.reward = await async_rm(task_rm_args, sample)
    return sample


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare and validate a fixed held-out Geo3K evaluation manifest.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--train", type=Path, required=True)
    prepare.add_argument("--test", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--config-output", type=Path, required=True)
    prepare.add_argument("--augmented-train-output", type=Path, default=None)
    prepare.add_argument("--size", type=int, required=True)
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument("--max-response-len", type=int, default=DEFAULT_TRAJECTORY_TOKEN_BUDGET)
    prepare.add_argument("--prompt-key", default=DEFAULT_PROMPT_KEY)
    prepare.add_argument("--label-key", default=DEFAULT_LABEL_KEY)
    prepare.add_argument("--media-key", default=DEFAULT_MEDIA_KEY)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.command != "prepare":  # pragma: no cover - argparse enforces this
        raise ValueError(f"Unsupported command: {args.command}")
    result = prepare_manifest(
        train_path=args.train,
        test_path=args.test,
        output_path=args.output,
        size=args.size,
        seed=args.seed,
        augmented_train_path=args.augmented_train_output,
        prompt_key=args.prompt_key,
        label_key=args.label_key,
        media_key=args.media_key,
    )
    result["eval_config"] = prepare_eval_config(
        output_path=args.config_output,
        manifest_path=args.output,
        max_response_len=args.max_response_len,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
