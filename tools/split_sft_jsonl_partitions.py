"""Split an Orbit SFT JSONL file into deterministic stratified partitions."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
import random
import shutil
from typing import Any


@dataclass(frozen=True)
class PartitionConfig:
    partitions: int
    seed: int
    stratify_key: str


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fin:
        for line_number, line in enumerate(fin, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            records.append(record)
    return records


def _jsonl_write(path: Path, records: Iterable[dict[str, Any]], *, force: bool) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {path}; pass --force")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def _path_get(record: dict[str, Any], dotted_key: str) -> Any:
    value: Any = record
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _stratum_key(value: Any) -> str:
    return "UNKNOWN" if value in (None, "") else str(value)


def _stable_sort_key(record: dict[str, Any], original_index: int) -> tuple[str, int]:
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        index = metadata.get("index")
        raw_index = metadata.get("raw_index")
        if index is not None:
            return (str(index), original_index)
        if raw_index is not None:
            return (str(raw_index), original_index)
    return (str(original_index), original_index)


def _with_partition_metadata(record: dict[str, Any], partition_name: str, partition_count: int) -> dict[str, Any]:
    copied = copy.deepcopy(record)
    metadata = copied.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("record metadata must be an object when present")
    metadata["partition"] = partition_name
    metadata["partition_count"] = partition_count
    return copied


def split_records(records: Sequence[dict[str, Any]], config: PartitionConfig) -> list[list[dict[str, Any]]]:
    if config.partitions < 2:
        raise ValueError("partitions must be at least 2")
    if not records:
        raise ValueError("input JSONL contains no records")

    rng = random.Random(config.seed)
    strata: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for original_index, record in enumerate(records):
        stratum = _path_get(record, config.stratify_key)
        strata[_stratum_key(stratum)].append((original_index, record))

    partitions: list[list[dict[str, Any]]] = [[] for _ in range(config.partitions)]
    partition_cursor = 0
    for stratum_key in sorted(strata):
        bucket = sorted(strata[stratum_key], key=lambda item: _stable_sort_key(item[1], item[0]))
        rng.shuffle(bucket)
        for _original_index, record in bucket:
            partition_index = partition_cursor % config.partitions
            partition_name = f"P{partition_index + 1}"
            partitions[partition_index].append(
                _with_partition_metadata(record, partition_name, config.partitions)
            )
            partition_cursor += 1

    for partition_index, partition in enumerate(partitions):
        partition.sort(key=lambda record: json.dumps(record.get("metadata", {}), sort_keys=True))
        if not partition:
            raise ValueError(f"P{partition_index + 1} is empty; reduce partition count")

    return partitions


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fin:
        for chunk in iter(lambda: fin.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value_counts(partition: Sequence[dict[str, Any]], dotted_key: str) -> dict[str, int]:
    counts = Counter(_stratum_key(_path_get(record, dotted_key)) for record in partition)
    return dict(sorted(counts.items()))


def _is_partition_entry(path: Path) -> bool:
    return path.name.startswith("P") and path.name[1:].isdigit()


def _validate_output_layout(
    root: Path,
    manifest_path: Path,
    partition_targets: Sequence[tuple[str, Path]],
    *,
    force: bool,
) -> None:
    if root.exists() and not root.is_dir():
        raise FileExistsError(f"refusing to overwrite {root}; expected output directory")
    if manifest_path.is_dir():
        raise FileExistsError(f"refusing to overwrite {manifest_path}; expected manifest file")

    for _name, output_path in partition_targets:
        partition_dir = output_path.parent
        if partition_dir.exists() and not partition_dir.is_dir():
            raise FileExistsError(f"refusing to overwrite {partition_dir}; expected partition directory")
        if output_path.is_dir():
            raise FileExistsError(f"refusing to overwrite {output_path}; expected train JSONL file")

    if not force and root.exists():
        for child in root.iterdir():
            if _is_partition_entry(child):
                raise FileExistsError(f"refusing to overwrite {child}; pass --force")


def _remove_stale_partition_dirs(root: Path, partition_names: set[str]) -> None:
    if not root.exists():
        return
    for child in root.iterdir():
        if child.is_dir() and _is_partition_entry(child) and child.name not in partition_names:
            shutil.rmtree(child)


def write_partitions(
    partitions: Sequence[Sequence[dict[str, Any]]],
    output_dir: str | Path,
    config: PartitionConfig,
    *,
    input_path: str,
    input_sha256: str,
    force: bool,
) -> dict[str, Any]:
    if len(partitions) != config.partitions:
        raise ValueError("partition list length must match config.partitions")

    root = Path(output_dir)
    manifest_path = root / "manifest.json"
    partition_targets = [
        (f"P{index}", root / f"P{index}" / "train.jsonl")
        for index in range(1, len(partitions) + 1)
    ]
    partition_names = {name for name, _output_path in partition_targets}

    _validate_output_layout(root, manifest_path, partition_targets, force=force)

    if force:
        _remove_stale_partition_dirs(root, partition_names)

    if not force:
        target_paths = [manifest_path, *(output_path for _name, output_path in partition_targets)]
        for target_path in target_paths:
            if target_path.exists():
                raise FileExistsError(f"refusing to overwrite {target_path}; pass --force")

    partition_rows: dict[str, int] = {}
    stratify_counts: dict[str, dict[str, int]] = {}
    output_sha256: dict[str, str] = {}

    for index, (name, output_path) in enumerate(partition_targets):
        partition = partitions[index]
        _jsonl_write(output_path, partition, force=force)
        partition_rows[name] = len(partition)
        stratify_counts[name] = _value_counts(partition, config.stratify_key)
        output_sha256[str(output_path.relative_to(root))] = sha256_file(output_path)

    manifest = {
        "input_path": input_path,
        "input_sha256": input_sha256,
        "partitions": config.partitions,
        "seed": config.seed,
        "stratify_key": config.stratify_key,
        "total_rows": sum(partition_rows.values()),
        "partition_rows": partition_rows,
        "stratify_counts": stratify_counts,
        "output_sha256": output_sha256,
    }

    if manifest_path.exists() and not force:
        raise FileExistsError(f"refusing to overwrite {manifest_path}; pass --force")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input Orbit SFT JSONL file.")
    parser.add_argument("--output-dir", required=True, help="Directory to receive P*/train.jsonl files.")
    parser.add_argument("--partitions", type=int, default=4, help="Number of partitions to write.")
    parser.add_argument("--seed", type=int, default=20260615, help="Deterministic split seed.")
    parser.add_argument("--stratify-key", default="metadata.dataset", help="Dotted row key for stratification.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input)
    records = read_jsonl(input_path)
    config = PartitionConfig(partitions=args.partitions, seed=args.seed, stratify_key=args.stratify_key)
    partitions = split_records(records, config)
    manifest = write_partitions(
        partitions,
        args.output_dir,
        config,
        input_path=str(input_path),
        input_sha256=sha256_file(input_path),
        force=args.force,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
