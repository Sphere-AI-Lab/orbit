import json
from collections import Counter

import pytest

from tools.split_sft_jsonl_partitions import (
    PartitionConfig,
    main,
    read_jsonl,
    sha256_file,
    split_records,
    write_partitions,
)


def _row(index: int, dataset: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": f"problem {index}"},
            {"role": "assistant", "content": f"solution {index}"},
        ],
        "metadata": {
            "dataset": dataset,
            "index": index,
        },
    }


def test_split_records_is_deterministic_and_stratifies_by_metadata_dataset():
    records = [_row(i, "math") for i in range(8)] + [_row(100 + i, "code") for i in range(8)]
    config = PartitionConfig(partitions=4, seed=1234, stratify_key="metadata.dataset")

    first = split_records(records, config)
    second = split_records(records, config)

    assert first == second
    assert [len(partition) for partition in first] == [4, 4, 4, 4]
    assert [Counter(row["metadata"]["dataset"] for row in partition) for partition in first] == [
        Counter({"math": 2, "code": 2}),
        Counter({"math": 2, "code": 2}),
        Counter({"math": 2, "code": 2}),
        Counter({"math": 2, "code": 2}),
    ]
    assert all(row["metadata"]["partition"] in {"P1", "P2", "P3", "P4"} for partition in first for row in partition)
    assert all(row["metadata"]["partition_count"] == 4 for partition in first for row in partition)


def test_split_records_rotates_tiny_strata_across_partitions():
    records = [_row(index, dataset) for index, dataset in enumerate(["a", "b", "c", "d"])]
    config = PartitionConfig(partitions=4, seed=1234, stratify_key="metadata.dataset")

    partitions = split_records(records, config)

    assert [len(partition) for partition in partitions] == [1, 1, 1, 1]
    assert [partition[0]["metadata"]["partition"] for partition in partitions] == ["P1", "P2", "P3", "P4"]


def test_split_records_rejects_invalid_partition_count():
    with pytest.raises(ValueError, match="partitions must be at least 2"):
        split_records([_row(1, "math")], PartitionConfig(partitions=1, seed=0, stratify_key="metadata.dataset"))


def test_write_partitions_writes_train_jsonl_and_manifest(tmp_path):
    records = [_row(i, "math") for i in range(4)] + [_row(100 + i, "code") for i in range(4)]
    config = PartitionConfig(partitions=4, seed=7, stratify_key="metadata.dataset")
    partitions = split_records(records, config)

    manifest = write_partitions(
        partitions,
        tmp_path,
        config,
        input_path="input/train.jsonl",
        input_sha256="abc123",
        force=False,
    )

    assert sorted(path.name for path in tmp_path.iterdir()) == ["P1", "P2", "P3", "P4", "manifest.json"]
    assert read_jsonl(tmp_path / "P1" / "train.jsonl")
    assert manifest["input_path"] == "input/train.jsonl"
    assert manifest["input_sha256"] == "abc123"
    assert manifest["partitions"] == 4
    assert manifest["total_rows"] == 8
    assert manifest["partition_rows"] == {"P1": 2, "P2": 2, "P3": 2, "P4": 2}
    assert manifest["stratify_counts"]["P1"] == {"code": 1, "math": 1}
    assert (tmp_path / "manifest.json").exists()


def test_write_partitions_records_counts_for_configured_stratify_key(tmp_path):
    records = [_row(index, "math") for index in range(4)]
    for index, record in enumerate(records):
        record["metadata"]["difficulty"] = "easy" if index < 2 else "hard"
    config = PartitionConfig(partitions=2, seed=7, stratify_key="metadata.difficulty")
    partitions = split_records(records, config)

    manifest = write_partitions(
        partitions,
        tmp_path,
        config,
        input_path="input/train.jsonl",
        input_sha256="abc123",
        force=False,
    )

    assert manifest["stratify_key"] == "metadata.difficulty"
    assert manifest["stratify_counts"] == {
        "P1": {"easy": 1, "hard": 1},
        "P2": {"easy": 1, "hard": 1},
    }


def test_write_partitions_preserves_falsey_stratify_values_in_counts(tmp_path):
    records = [_row(index, "math") for index in range(4)]
    for record, bucket in zip(records, [0, False, None, ""]):
        record["metadata"]["bucket"] = bucket
    config = PartitionConfig(partitions=2, seed=7, stratify_key="metadata.bucket")
    partitions = split_records(records, config)

    manifest = write_partitions(
        partitions,
        tmp_path,
        config,
        input_path="input/train.jsonl",
        input_sha256="abc123",
        force=False,
    )

    combined_counts = Counter()
    for counts in manifest["stratify_counts"].values():
        combined_counts.update(counts)

    assert combined_counts == Counter({"0": 1, "False": 1, "UNKNOWN": 2})


def test_write_partitions_refuses_to_overwrite_without_force(tmp_path):
    records = [_row(i, "math") for i in range(4)]
    config = PartitionConfig(partitions=2, seed=1, stratify_key="metadata.dataset")
    partitions = split_records(records, config)

    write_partitions(
        partitions,
        tmp_path,
        config,
        input_path="input/train.jsonl",
        input_sha256="abc123",
        force=False,
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_partitions(
            partitions,
            tmp_path,
            config,
            input_path="input/train.jsonl",
            input_sha256="abc123",
            force=False,
        )


def test_write_partitions_preflights_all_targets_before_writing(tmp_path):
    config = PartitionConfig(partitions=2, seed=1, stratify_key="metadata.dataset")
    partitions = [[_row(1, "math")], [_row(2, "math")]]
    existing_path = tmp_path / "P2" / "train.jsonl"
    existing_path.parent.mkdir(parents=True)
    existing_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_partitions(
            partitions,
            tmp_path,
            config,
            input_path="input/train.jsonl",
            input_sha256="abc123",
            force=False,
        )

    assert not (tmp_path / "P1" / "train.jsonl").exists()
    assert existing_path.read_text(encoding="utf-8") == "existing\n"
    assert not (tmp_path / "manifest.json").exists()


def test_write_partitions_non_force_rejects_stale_generated_partitions(tmp_path):
    config = PartitionConfig(partitions=2, seed=1, stratify_key="metadata.dataset")
    partitions = [[_row(1, "math")], [_row(2, "math")]]
    stale_path = tmp_path / "P3" / "train.jsonl"
    stale_path.parent.mkdir(parents=True)
    stale_path.write_text("stale\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_partitions(
            partitions,
            tmp_path,
            config,
            input_path="input/train.jsonl",
            input_sha256="abc123",
            force=False,
        )

    assert not (tmp_path / "P1" / "train.jsonl").exists()
    assert stale_path.read_text(encoding="utf-8") == "stale\n"
    assert not (tmp_path / "manifest.json").exists()


def test_write_partitions_preflights_partition_parent_conflicts(tmp_path):
    config = PartitionConfig(partitions=2, seed=1, stratify_key="metadata.dataset")
    partitions = [[_row(1, "math")], [_row(2, "math")]]
    (tmp_path / "P2").write_text("not a directory\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_partitions(
            partitions,
            tmp_path,
            config,
            input_path="input/train.jsonl",
            input_sha256="abc123",
            force=False,
        )

    assert not (tmp_path / "P1" / "train.jsonl").exists()
    assert (tmp_path / "P2").read_text(encoding="utf-8") == "not a directory\n"
    assert not (tmp_path / "manifest.json").exists()


def test_write_partitions_force_preflights_train_target_directories(tmp_path):
    config = PartitionConfig(partitions=2, seed=1, stratify_key="metadata.dataset")
    partitions = [[_row(1, "math")], [_row(2, "math")]]
    p1_path = tmp_path / "P1" / "train.jsonl"
    p1_path.parent.mkdir(parents=True)
    p1_path.write_text("old\n", encoding="utf-8")
    (tmp_path / "P2" / "train.jsonl").mkdir(parents=True)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_partitions(
            partitions,
            tmp_path,
            config,
            input_path="input/train.jsonl",
            input_sha256="abc123",
            force=True,
        )

    assert p1_path.read_text(encoding="utf-8") == "old\n"
    assert not (tmp_path / "manifest.json").exists()


def test_write_partitions_force_preflights_manifest_directory(tmp_path):
    config = PartitionConfig(partitions=2, seed=1, stratify_key="metadata.dataset")
    partitions = [[_row(1, "math")], [_row(2, "math")]]
    p1_path = tmp_path / "P1" / "train.jsonl"
    p1_path.parent.mkdir(parents=True)
    p1_path.write_text("old\n", encoding="utf-8")
    (tmp_path / "manifest.json").mkdir()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_partitions(
            partitions,
            tmp_path,
            config,
            input_path="input/train.jsonl",
            input_sha256="abc123",
            force=True,
        )

    assert p1_path.read_text(encoding="utf-8") == "old\n"
    assert (tmp_path / "manifest.json").is_dir()


def test_write_partitions_force_removes_stale_partition_directories(tmp_path):
    first_config = PartitionConfig(partitions=4, seed=1, stratify_key="metadata.dataset")
    first_partitions = split_records([_row(index, "math") for index in range(4)], first_config)
    write_partitions(
        first_partitions,
        tmp_path,
        first_config,
        input_path="input/train.jsonl",
        input_sha256="abc123",
        force=False,
    )

    second_config = PartitionConfig(partitions=2, seed=1, stratify_key="metadata.dataset")
    second_partitions = split_records([_row(index, "math") for index in range(2)], second_config)
    write_partitions(
        second_partitions,
        tmp_path,
        second_config,
        input_path="input/train.jsonl",
        input_sha256="def456",
        force=True,
    )

    assert sorted(path.name for path in tmp_path.iterdir()) == ["P1", "P2", "manifest.json"]


def test_write_partitions_rejects_partition_count_mismatch(tmp_path):
    config = PartitionConfig(partitions=2, seed=1, stratify_key="metadata.dataset")

    with pytest.raises(ValueError, match="partition list length must match config.partitions"):
        write_partitions(
            [[_row(1, "math")]],
            tmp_path,
            config,
            input_path="input/train.jsonl",
            input_sha256="abc123",
            force=False,
        )

    assert list(tmp_path.iterdir()) == []


def test_main_writes_partitions_manifest_and_stdout(tmp_path, capsys):
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    records = [_row(index, "math") for index in range(2)] + [_row(10 + index, "code") for index in range(2)]
    input_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--partitions",
            "2",
            "--seed",
            "7",
            "--stratify-key",
            "metadata.dataset",
        ]
    )

    stdout_manifest = json.loads(capsys.readouterr().out)
    written_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    assert read_jsonl(output_dir / "P1" / "train.jsonl")
    assert read_jsonl(output_dir / "P2" / "train.jsonl")
    assert written_manifest["input_sha256"] == sha256_file(input_path)
    assert stdout_manifest == written_manifest
