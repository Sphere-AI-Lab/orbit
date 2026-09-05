from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.geo3k_vlm.multi_turn import fixed_eval


REPO_ROOT = Path(__file__).resolve().parents[3]


def _row(index: int) -> dict:
    return {
        "problem": f"problem-{index}",
        "answer": str(index),
        "images": [f"https://example.test/{index}.png"],
    }


def test_fixed_eval_selection_is_content_deterministic() -> None:
    train_rows = [_row(index) for index in range(3)]
    test_rows = [_row(index) for index in range(10, 20)]

    selected, stats = fixed_eval.select_eval_records(train_rows, test_rows, size=6, seed=20260720)
    reordered, _ = fixed_eval.select_eval_records(train_rows[::-1], test_rows[::-1], size=6, seed=20260720)

    assert [record_id for _index, _row_data, record_id in selected] == [
        record_id for _index, _row_data, record_id in reordered
    ]
    assert stats == {
        "test_unique": 10,
        "excluded_train_record": 0,
        "excluded_train_media": 0,
        "eligible": 10,
    }


@pytest.mark.parametrize("leak_field", ["record", "media"])
def test_fixed_eval_selection_excludes_exact_record_and_shared_image(leak_field: str) -> None:
    train = _row(1)
    leaked = _row(2)
    clean = _row(3)
    if leak_field == "record":
        leaked = dict(train)
    else:
        leaked["images"] = train["images"]

    selected, stats = fixed_eval.select_eval_records([train], [leaked, clean], size=1, seed=7)
    assert len(selected) == 1
    assert selected[0][1]["problem"] == clean["problem"]
    excluded_key = "excluded_train_record" if leak_field == "record" else "excluded_train_media"
    assert stats[excluded_key] == 1
    assert stats["eligible"] == 1

    with pytest.raises(ValueError, match="image-clean eval records"):
        fixed_eval.select_eval_records([train], [leaked, clean], size=2, seed=7)


def test_fixed_eval_selection_allows_boilerplate_prompt_text_collision() -> None:
    train = _row(1)
    test = _row(2)
    test["problem"] = train["problem"]

    selected, stats = fixed_eval.select_eval_records([train], [test], size=1, seed=7)
    assert len(selected) == 1
    assert stats == {
        "test_unique": 1,
        "excluded_train_record": 0,
        "excluded_train_media": 0,
        "eligible": 1,
    }


def test_manifest_fingerprint_changes_with_contract() -> None:
    base = fixed_eval.manifest_fingerprint(
        ["a", "b"],
        seed=7,
        train_sha256="train",
        test_sha256="test",
    )
    assert base == fixed_eval.manifest_fingerprint(
        ["a", "b"],
        seed=7,
        train_sha256="train",
        test_sha256="test",
    )
    assert base != fixed_eval.manifest_fingerprint(
        ["b", "a"],
        seed=7,
        train_sha256="train",
        test_sha256="test",
    )
    assert base != fixed_eval.manifest_fingerprint(
        ["a", "b"],
        seed=8,
        train_sha256="train",
        test_sha256="test",
    )


def test_prepare_manifest_is_idempotent_and_embeds_contract(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    output_path = tmp_path / "eval.parquet"
    augmented_path = tmp_path / "train_augmented.parquet"
    pq.write_table(pa.Table.from_pylist([_row(index) for index in range(3)]), train_path)
    pq.write_table(pa.Table.from_pylist([_row(index) for index in range(10, 20)]), test_path)

    first = fixed_eval.prepare_manifest(
        train_path=train_path,
        test_path=test_path,
        output_path=output_path,
        size=6,
        seed=20260720,
        augmented_train_path=augmented_path,
    )
    second = fixed_eval.prepare_manifest(
        train_path=train_path,
        test_path=test_path,
        output_path=output_path,
        size=6,
        seed=20260720,
        augmented_train_path=augmented_path,
    )

    assert first["action"] == "created"
    assert second["action"] == "validated"
    assert first["fingerprint"] == second["fingerprint"]
    assert first["selection_stats"]["eligible"] == 10
    assert first["augmented_train"]["action"] == "created"
    assert second["augmented_train"]["action"] == "validated"
    assert output_path.stat().st_mode & 0o777 == fixed_eval.SHARED_ARTIFACT_MODE
    assert augmented_path.stat().st_mode & 0o777 == fixed_eval.SHARED_ARTIFACT_MODE
    manifest = pq.read_table(output_path)
    assert manifest.num_rows == 6
    assert fixed_eval.EVAL_METADATA_KEY in manifest.column_names
    metadata = manifest.schema.metadata
    assert metadata[b"opd_eval_manifest_fingerprint"].decode() == first["fingerprint"]

    # train(3) + the 4 non-selected test rows; no image is shared with the manifest.
    augmented = pq.read_table(augmented_path)
    assert augmented.num_rows == 7
    assert first["augmented_train"]["rows"] == 7
    manifest_images = {tuple(row["images"]) for row in manifest.to_pylist()}
    augmented_images = {tuple(row["images"]) for row in augmented.to_pylist()}
    assert not manifest_images & augmented_images
    assert augmented.schema.metadata[b"opd_eval_manifest_fingerprint"].decode() == first["fingerprint"]


def test_prepare_manifest_rejects_tampered_augmented_train(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    output_path = tmp_path / "eval.parquet"
    augmented_path = tmp_path / "train_augmented.parquet"
    pq.write_table(pa.Table.from_pylist([_row(index) for index in range(3)]), train_path)
    pq.write_table(pa.Table.from_pylist([_row(index) for index in range(10, 20)]), test_path)
    fixed_eval.prepare_manifest(
        train_path=train_path,
        test_path=test_path,
        output_path=output_path,
        size=6,
        seed=20260720,
        augmented_train_path=augmented_path,
    )

    augmented = pq.read_table(augmented_path)
    tampered_rows = augmented.to_pylist()
    tampered_rows[0]["answer"] = "tampered"
    pq.write_table(
        pa.Table.from_pylist(tampered_rows, schema=augmented.schema).replace_schema_metadata(
            augmented.schema.metadata
        ),
        augmented_path,
    )

    with pytest.raises(ValueError, match="train\\+test composition"):
        fixed_eval.prepare_manifest(
            train_path=train_path,
            test_path=test_path,
            output_path=output_path,
            size=6,
            seed=20260720,
            augmented_train_path=augmented_path,
        )


def test_prepare_manifest_rejects_tampered_existing_rows(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    train_path = tmp_path / "train.parquet"
    test_path = tmp_path / "test.parquet"
    output_path = tmp_path / "eval.parquet"
    pq.write_table(pa.Table.from_pylist([_row(index) for index in range(3)]), train_path)
    pq.write_table(pa.Table.from_pylist([_row(index) for index in range(10, 20)]), test_path)
    fixed_eval.prepare_manifest(
        train_path=train_path,
        test_path=test_path,
        output_path=output_path,
        size=6,
        seed=20260720,
    )

    manifest = pq.read_table(output_path)
    tampered_rows = manifest.to_pylist()
    tampered_rows[0]["answer"] = "tampered"
    pq.write_table(pa.Table.from_pylist(tampered_rows, schema=manifest.schema), output_path)

    with pytest.raises(ValueError, match="differs from the selected source rows"):
        fixed_eval.prepare_manifest(
            train_path=train_path,
            test_path=test_path,
            output_path=output_path,
            size=6,
            seed=20260720,
        )


def test_eval_config_scopes_metadata_and_task_reward_wrapper_to_eval(tmp_path) -> None:
    config_path = tmp_path / "eval.json"
    manifest_path = tmp_path / "eval.parquet"

    first = fixed_eval.prepare_eval_config(output_path=config_path, manifest_path=manifest_path)
    second = fixed_eval.prepare_eval_config(output_path=config_path, manifest_path=manifest_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    defaults = config["eval"]["defaults"]
    dataset = config["eval"]["datasets"][0]

    assert first["action"] == "created"
    assert second["action"] == "validated"
    assert first["max_response_len"] == 12000
    assert second["max_response_len"] == 12000
    assert config_path.stat().st_mode & 0o777 == fixed_eval.SHARED_ARTIFACT_MODE
    assert defaults == {
        "max_response_len": 12000,
        "n_samples_per_eval_prompt": 1,
        "temperature": 0,
        "top_k": -1,
        "top_p": 1,
    }
    assert dataset["path"] == str(manifest_path)
    assert dataset["metadata_key"] == fixed_eval.EVAL_METADATA_KEY
    assert dataset["custom_generate_function_path"] == "examples.geo3k_vlm.multi_turn.fixed_eval.generate"


def test_eval_config_rejects_nonpositive_token_budget(tmp_path) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        fixed_eval.prepare_eval_config(
            output_path=tmp_path / "eval.json",
            manifest_path=tmp_path / "eval.parquet",
            max_response_len=0,
        )


def test_eval_config_rejects_reusing_path_with_different_token_budget(tmp_path) -> None:
    output_path = tmp_path / "eval.json"
    manifest_path = tmp_path / "eval.parquet"
    fixed_eval.prepare_eval_config(
        output_path=output_path,
        manifest_path=manifest_path,
        max_response_len=12000,
    )

    with pytest.raises(ValueError, match="differs from the Milestone 11 contract"):
        fixed_eval.prepare_eval_config(
            output_path=output_path,
            manifest_path=manifest_path,
            max_response_len=4096,
        )


def test_wilson_interval_contains_observed_accuracy() -> None:
    low, high = fixed_eval.wilson_interval(128, 256)
    assert low < 0.5 < high
    assert high - low < 0.13


@pytest.mark.parametrize(
    ("num_rollout", "rollout_id", "expected"),
    [(0, 0, 0), (200, 0, 0), (200, 4, 5), (200, 9, 10), (200, 199, 200)],
)
def test_eval_callback_uses_completed_optimizer_steps(num_rollout: int, rollout_id: int, expected: int) -> None:
    args = SimpleNamespace(num_rollout=num_rollout, skip_eval_before_train=False)
    assert fixed_eval.evaluation_model_step(args, rollout_id) == expected


@dataclass
class _EvalSample:
    index: int
    reward: float
    response_length: int = 12
    effective_response_length: int = 10
    response: str = "Answer: \\boxed{1}"
    status: object = field(default_factory=lambda: SimpleNamespace(value="completed"))
    weight_versions: list[str] = field(default_factory=lambda: ["5"])
    metadata: dict = field(default_factory=dict)


def _eval_samples(size: int = 4) -> list[_EvalSample]:
    fingerprint = "manifest-fingerprint"
    return [
        _EvalSample(
            index=index,
            reward=float(index % 2 == 0),
            metadata={
                "opd_eval_id": f"id-{index}",
                "opd_eval_manifest_fingerprint": fingerprint,
                "opd_eval_manifest_size": size,
                "round_number": 2,
            },
        )
        for index in range(size)
    ]


def test_eval_logger_records_accuracy_ci_and_exact_step(monkeypatch) -> None:
    from orbit.utils.tracking_utils import tracking

    captured = {}
    monkeypatch.setattr(
        tracking,
        "log",
        lambda _args, metrics, step_key: captured.update(metrics=metrics, step_key=step_key),
    )
    samples = _eval_samples()
    data = {
        "geo3k_fixed": {
            "rewards": [sample.reward for sample in samples],
            "truncated": [False] * len(samples),
            "samples": samples,
        }
    }

    handled = fixed_eval.log_eval_rollout_data(
        4,
        SimpleNamespace(num_rollout=200, skip_eval_before_train=False),
        data,
        {},
    )

    assert handled is True
    assert captured["step_key"] == "eval/step"
    assert captured["metrics"]["eval/step"] == 5
    assert captured["metrics"]["eval/geo3k_fixed/accuracy"] == 0.5
    assert captured["metrics"]["eval/geo3k_fixed/accuracy_ci95_low"] < 0.5
    assert captured["metrics"]["eval/geo3k_fixed/accuracy_ci95_high"] > 0.5


def test_manifest_contract_rejects_duplicate_ids() -> None:
    samples = _eval_samples()
    samples[1].metadata["opd_eval_id"] = samples[0].metadata["opd_eval_id"]
    with pytest.raises(ValueError, match="duplicate eval IDs"):
        fixed_eval._manifest_contract(samples)


def test_eval_logger_rejects_missing_truncation_rows(monkeypatch) -> None:
    from orbit.utils.tracking_utils import tracking

    monkeypatch.setattr(tracking, "log", lambda *_args, **_kwargs: None)
    samples = _eval_samples()
    data = {
        "geo3k_fixed": {
            "rewards": [sample.reward for sample in samples],
            "truncated": [False] * (len(samples) - 1),
            "samples": samples,
        }
    }

    with pytest.raises(ValueError, match="truncation count"):
        fixed_eval.log_eval_rollout_data(
            4,
            SimpleNamespace(num_rollout=200, skip_eval_before_train=False),
            data,
            {},
        )


@pytest.mark.asyncio
async def test_eval_generate_assigns_task_reward_before_custom_opd_rm(monkeypatch) -> None:
    from examples.geo3k_vlm.multi_turn import rollout
    from orbit.rollout import rm_hub

    calls = []

    async def fake_generate(_args, sample, _sampling_params):
        calls.append("generate")
        return sample

    async def fake_task_rm(args, _sample):
        calls.append(("rm", args.custom_rm_path))
        return 1

    monkeypatch.setattr(rollout, "generate", fake_generate)
    monkeypatch.setattr(rm_hub, "async_rm", fake_task_rm)
    args = SimpleNamespace(custom_rm_path="orbit.rollout.on_policy_distillation.reward_func")
    sample = SimpleNamespace(reward=None)

    result = await fixed_eval.generate(args, sample, {}, evaluation=True)

    assert result.reward == 1
    assert calls == ["generate", ("rm", None)]
    assert args.custom_rm_path == "orbit.rollout.on_policy_distillation.reward_func"


def _source_recipe(
    tmp_path: Path,
    recipe_name: str,
    *,
    env_overrides: dict[str, str] | None = None,
) -> tuple[list[str], dict[str, str]]:
    for model_name in (
        "Qwen3-VL-8B-Instruct",
        "Qwen3-VL-8B-Thinking",
        "Qwen3-VL-30B-A3B-Thinking",
    ):
        model_dir = tmp_path / "models" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").touch()

    recipe = REPO_ROOT / "scripts" / "experiments" / "OPD" / "multimodal" / recipe_name
    shell = f"""
set -euo pipefail
source {shlex.quote(str(recipe))}
printf '__M11_META__%s|%s|%s|%s|%s|%s\n' "$EXPERIMENT_NODES" "${{ORBIT_TRAIN_ENTRY:-train.py}}" "$HF_MODEL_REPO" "$OPD_EVAL_INTERVAL" "${{OPD_TEACHER_MODEL_DIR:-}}" "$WANDB_RUN_NAME"
printf '__M11_ARGS_BEGIN__\n'
printf '%s\n' "${{ORBIT_ARGS[@]}}"
"""
    env = os.environ.copy()
    env.update(
        {
            "HEAD_IP": "127.0.0.1",
            "HF_CACHE_DIR": str(tmp_path),
            "ORBIT_REPO": str(REPO_ROOT),
            "OPD_NUM_ROLLOUT": "5",
        }
    )
    env.update(env_overrides or {})
    result = subprocess.run(["bash", "-c", shell], check=True, capture_output=True, text=True, env=env)
    meta_line = next(line for line in result.stdout.splitlines() if line.startswith("__M11_META__"))
    metadata = dict(
        zip(
            ("nodes", "entry", "model_repo", "eval_interval", "teacher_model_dir", "run_name"),
            meta_line.removeprefix("__M11_META__").split("|"),
            strict=True,
        )
    )
    args = result.stdout.split("__M11_ARGS_BEGIN__\n", maxsplit=1)[1].splitlines()
    return args, metadata


def _arg_value(args: list[str], flag: str) -> str:
    assert args.count(flag) == 1, f"expected exactly one {flag}, got {args.count(flag)}"
    return args[args.index(flag) + 1]


@pytest.mark.parametrize(
    ("recipe_name", "teacher_model", "expected_mem_fraction"),
    [
        ("11c-geo3k-multiturn-hybrid-sync-eval-teacher8b-200step.sh", "Qwen3-VL-8B-Thinking", "0.85"),
        ("11d-geo3k-multiturn-hybrid-sync-eval-teacher30b-200step.sh", "Qwen3-VL-30B-A3B-Thinking", "0.80"),
    ],
)
def test_student_recipe_keeps_sync_hybrid_training_contract(
    tmp_path, recipe_name: str, teacher_model: str, expected_mem_fraction: str
) -> None:
    args, metadata = _source_recipe(tmp_path, recipe_name)

    assert metadata["nodes"] == "3"
    assert metadata["entry"] == "train.py"
    assert metadata["eval_interval"] == "5"
    assert _arg_value(args, "--num-rollout") == "5"
    assert _arg_value(args, "--opd-kl-coef") == "1"
    assert _arg_value(args, "--opd-log-prob-top-k") == "0"
    assert _arg_value(args, "--opd-dagger-top-k") == "2"
    assert _arg_value(args, "--opd-dagger-coef") == "0.5"
    assert _arg_value(args, "--custom-rm-path") == "orbit.rollout.on_policy_distillation.reward_func"
    assert _arg_value(args, "--custom-generate-function-path") == "examples.geo3k_vlm.multi_turn.rollout.generate"
    assert _arg_value(args, "--rollout-max-response-len") == "12000"
    assert _arg_value(args, "--rollout-max-context-len") == "12000"
    assert _arg_value(args, "--eval-interval") == "5"
    assert _arg_value(args, "--eval-config").endswith("opd_eval_seed20260720_n30.ctx12000.eval.json")
    assert _arg_value(args, "--prompt-data").endswith("opd_eval_seed20260720_n30.train_augmented.parquet")
    assert (
        _arg_value(args, "--custom-eval-rollout-log-function-path")
        == "examples.geo3k_vlm.multi_turn.fixed_eval.log_eval_rollout_data"
    )
    assert _arg_value(args, "--rollout-all-samples-process-path") == (
        "examples.geo3k_vlm.multi_turn.fixed_eval.dump_samples"
    )
    assert _arg_value(args, "--sglang-mem-fraction-static") == expected_mem_fraction
    assert Path(metadata["teacher_model_dir"]).name == teacher_model
    assert "--opd-log-task-reward" in args
    assert "--sglang-mm-exact-scoring-suffix" in args
    assert not any(arg.startswith("--fully-async-") for arg in args)
    assert "examples.fully_async.fully_async_rollout.generate_rollout_fully_async" not in args
    assert "--opd-optimize-task-reward" not in args
    assert "--use-rollout-logprobs" not in args
    assert not any(arg == "--save" or arg.startswith("--save-") or arg == "--async-save" for arg in args)


def test_student_recipe_allows_explicit_eval_cost_overrides(tmp_path) -> None:
    args, metadata = _source_recipe(
        tmp_path,
        "11c-geo3k-multiturn-hybrid-sync-eval-teacher8b-200step.sh",
        env_overrides={
            "OPD_EVAL_INTERVAL": "10",
            "OPD_EVAL_NUM_PROMPTS": "64",
            "OPD_EVAL_MAX_CONTEXT_LEN": "16000",
        },
    )

    assert metadata["eval_interval"] == "10"
    assert metadata["run_name"] == "opd-mm-11c-sync-hybrid-teacher8b-eval10-n64-ctx16000-5step"
    assert _arg_value(args, "--rollout-max-response-len") == "16000"
    assert _arg_value(args, "--rollout-max-context-len") == "16000"
    assert _arg_value(args, "--eval-interval") == "10"
    assert _arg_value(args, "--eval-config").endswith("opd_eval_seed20260720_n64.ctx16000.eval.json")
    assert _arg_value(args, "--prompt-data").endswith("opd_eval_seed20260720_n64.train_augmented.parquet")


def test_student_recipe_rejects_interval_one_step_collision(tmp_path) -> None:
    with pytest.raises(subprocess.CalledProcessError):
        _source_recipe(
            tmp_path,
            "11c-geo3k-multiturn-hybrid-sync-eval-teacher8b-200step.sh",
            env_overrides={"OPD_EVAL_INTERVAL": "1"},
        )


@pytest.mark.parametrize(
    ("recipe_name", "model_repo", "run_name"),
    [
        (
            "11a-geo3k-fixed-eval-teacher8b-reference.sh",
            "Qwen/Qwen3-VL-8B-Thinking",
            "opd-mm-11a-teacher8b-fixed-eval-n30-ctx12000",
        ),
        (
            "11b-geo3k-fixed-eval-teacher30b-reference.sh",
            "Qwen/Qwen3-VL-30B-A3B-Thinking",
            "opd-mm-11b-teacher30b-fixed-eval-n30-ctx12000",
        ),
    ],
)
def test_teacher_reference_recipe_is_eval_only(tmp_path, recipe_name: str, model_repo: str, run_name: str) -> None:
    args, metadata = _source_recipe(tmp_path, recipe_name)

    assert metadata == {
        "nodes": "2",
        "entry": "train.py",
        "model_repo": model_repo,
        "eval_interval": "1",
        "teacher_model_dir": "",
        "run_name": run_name,
    }
    assert _arg_value(args, "--num-rollout") == "0"
    assert _arg_value(args, "--actor-num-gpus-per-node") == "8"
    assert _arg_value(args, "--rollout-num-gpus") == "8"
    assert _arg_value(args, "--rollout-num-gpus-per-engine") == "8"
    assert _arg_value(args, "--tensor-model-parallel-size") == "8"
    assert _arg_value(args, "--rollout-max-response-len") == "12000"
    assert _arg_value(args, "--rollout-max-context-len") == "12000"
    assert _arg_value(args, "--eval-interval") == "1"
    assert _arg_value(args, "--eval-config").endswith("opd_eval_seed20260720_n30.ctx12000.eval.json")
    assert _arg_value(args, "--prompt-data").endswith("opd_eval_seed20260720_n30.parquet")
    assert (
        _arg_value(args, "--custom-eval-rollout-log-function-path")
        == "examples.geo3k_vlm.multi_turn.fixed_eval.log_eval_rollout_data"
    )
    assert _arg_value(args, "--rollout-all-samples-process-path") == (
        "examples.geo3k_vlm.multi_turn.fixed_eval.dump_samples"
    )
    assert "--use-opd" not in args
    assert "--custom-rm-path" not in args
    assert "--opd-log-task-reward" not in args
    assert not any(arg == "--save" or arg.startswith("--save-") or arg == "--async-save" for arg in args)
