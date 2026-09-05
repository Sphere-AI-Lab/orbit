"""Build JSONL datasets for envpack-backed Orbit rollouts.

This is intentionally lighter than the VAGEN builder: rows contain only the
semantic envpack rollout spec. The live env state and images are produced by
envpack during rollout/eval.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import logging
import os
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from orbit_plugins.envpack_adapter.config import EnvpackAdapterConfig, EnvpackPoolConfig, EnvpackRolloutConfig
from orbit_plugins.envpack_adapter.runtime import build_in_process_client

logger = logging.getLogger(__name__)

MAX_INT32 = 2**31 - 1
SAMPLES_NAME = "samples.jsonl"


@dataclass(slots=True)
class EnvSpec:
    name: str
    n_envs: int
    config: dict[str, Any] = field(default_factory=dict)
    seed: list[int] = field(default_factory=lambda: [0])
    seed_list: list[int] | None = None
    profile: str | None = None
    pool_id: str | None = None
    bucket_prefix: str | None = None
    sampling: dict[str, Any] | None = None


async def build_dataset(
    *,
    yaml_path: str,
    output_dir: str,
    split: str,
    base_seed: int,
    exclude_data: str | None,
    dedup_within: bool,
    target_kept: int | None,
    eval_output_dir: str | None = None,
    force: bool,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    samples_path = os.path.join(output_dir, SAMPLES_NAME)
    meta_path = os.path.join(output_dir, "dataset_meta.json")
    raw = _load_yaml_or_json(yaml_path)
    sampling = _parse_sampling(raw.get("sampling"))
    build_meta = {
        "yaml_sha256": _file_sha256(yaml_path),
        "base_seed": int(base_seed),
        "split": split,
        "exclude_data_sha256": _file_sha256(exclude_data) if exclude_data else None,
        "dedup_within": bool(dedup_within),
        "target_kept": target_kept,
        "mode": "balanced_sokoban" if sampling is not None else "legacy",
    }
    specs = _load_specs_from_raw(raw, yaml_path)
    if sampling is not None:
        if split != "train":
            raise RuntimeError("balanced sampling must be invoked once with --split train")
        if eval_output_dir is None:
            raise RuntimeError("balanced sampling requires --eval-output-dir")
        await _build_balanced_sokoban_dataset(
            yaml_path=yaml_path,
            output_dir=output_dir,
            eval_output_dir=eval_output_dir,
            base_seed=base_seed,
            specs=specs,
            sampling=sampling,
            build_meta=build_meta,
            force=force,
        )
        return

    if not force and _is_current(meta_path, samples_path, build_meta):
        logger.info("dataset already current: %s", samples_path)
        return

    excluded = _load_excluded_env_uuids(exclude_data)
    tmp_path = samples_path + ".tmp"
    try:
        os.unlink(tmp_path)
    except FileNotFoundError:
        pass

    rows_written = 0
    n_excluded = 0
    n_intra_dup = 0
    kept_uuids: set[str] = set()
    with open(tmp_path, "w", encoding="utf-8") as out_f:
        for spec_idx, spec in enumerate(specs):
            env_name = _normalize_env_name(spec.name)
            profile = spec.profile or "vision_free_think_local"
            pool_id = spec.pool_id or f"{env_name}:{profile}"
            seeds = _generate_seeds_for_spec(spec, base_seed, spec_idx)
            uuid_resolver = _EnvUuidResolver(env_name=env_name, profile=profile, pool_id=pool_id)
            for seed in seeds:
                if target_kept is not None and rows_written >= target_kept:
                    break
                env_config = copy.deepcopy(spec.config)
                env_uuid = await uuid_resolver.resolve(env_config=env_config, seed=int(seed))
                if env_uuid in excluded:
                    n_excluded += 1
                    continue
                if dedup_within and env_uuid in kept_uuids:
                    n_intra_dup += 1
                    continue
                kept_uuids.add(env_uuid)
                row = {
                    "input": "envpack_placeholder",
                    "images": [],
                    "metadata": {
                        "envpack": {
                            "env_name": env_name,
                            "seed": int(seed),
                            "env_config": env_config,
                            "profile": profile,
                            "pool_id": pool_id,
                            "env_uuid": env_uuid,
                            "split": split,
                            "heldout": bool(excluded),
                            "source_format": "samples_jsonl",
                        }
                    },
                }
                out_f.write(json.dumps(row, sort_keys=True) + "\n")
                rows_written += 1
            if target_kept is not None and rows_written >= target_kept:
                break

    if target_kept is not None and rows_written < target_kept:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise RuntimeError(f"target_kept={target_kept} but only kept {rows_written} rows")

    os.replace(tmp_path, samples_path)
    build_meta.update(
        {
            "rows_written": rows_written,
            "n_excluded": n_excluded,
            "n_intra_dup": n_intra_dup,
            "n_unique_env_uuids": len(kept_uuids),
        }
    )
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(build_meta, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info(
        "wrote %d rows (%d unique, %d excluded, %d intra-dup) -> %s",
        rows_written,
        len(kept_uuids),
        n_excluded,
        n_intra_dup,
        samples_path,
    )


class _EnvUuidResolver:
    def __init__(self, *, env_name: str, profile: str, pool_id: str) -> None:
        self.env_name = env_name
        self.profile = profile
        self.pool_id = pool_id
        self._bundle = None

    async def resolve(self, *, env_config: dict[str, Any], seed: int) -> str:
        try:
            from envpack.core import EpisodeCreateRequest
        except Exception as exc:
            raise RuntimeError(
                "envpack must be importable to build envpack datasets. "
                "Install it with `pip install -e thirdparty/envpack`."
            ) from exc
        if self._bundle is None:
            config = EnvpackAdapterConfig(
                api="in_process",
                pools=(
                    EnvpackPoolConfig(
                        env=self.env_name,
                        profile=self.profile,
                        pool_id=self.pool_id,
                        runtime_config={"num_instances": 1, "max_active_episodes_per_instance": 1},
                    ),
                ),
                rollout=EnvpackRolloutConfig(),
            )
            self._bundle = build_in_process_client(config)
        resolved_env_config = copy.deepcopy(self._bundle.env_config(self.pool_id))
        resolved_env_config.update(copy.deepcopy(env_config))
        episode_id = f"dataset-{self.env_name}-{seed}"
        created = await self._bundle.client.create_episode(
            EpisodeCreateRequest(
                env_name=self.env_name,
                request_id=f"create:{episode_id}",
                episode_id=episode_id,
                env_config=resolved_env_config,
                seed=seed,
                metadata={"pool_id": self.pool_id, "source": "envpack_dataset_builder"},
            )
        )
        try:
            media_hashes = [media.sha256 for media in created.observation.media if media.sha256]
            if media_hashes:
                return media_hashes[0]
            state_payload = created.observation.state or {}
            return hashlib.sha256(
                json.dumps(state_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        finally:
            await self._bundle.client.cancel_episode(episode_id, reason="dataset_builder_done")


async def _build_balanced_sokoban_dataset(
    *,
    yaml_path: str,
    output_dir: str,
    eval_output_dir: str,
    base_seed: int,
    specs: list[EnvSpec],
    sampling,
    build_meta: dict[str, Any],
    force: bool,
) -> None:
    try:
        from envpack.envs.sokoban.dataset import (
            SokobanDatasetSpec,
            build_balanced_sokoban_dataset,
            parse_sampling_spec,
        )
        from envpack.envs.sokoban.dataset import sampling_meta as sokoban_sampling_meta
    except Exception as exc:
        raise RuntimeError("balanced Sokoban build requires envpack with Sokoban dataset helpers installed") from exc

    for spec in specs:
        env_name = _normalize_env_name(spec.name)
        if env_name != "sokoban":
            raise RuntimeError("balanced sampling is currently implemented only for Sokoban")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(eval_output_dir, exist_ok=True)
    train_samples_path = os.path.join(output_dir, SAMPLES_NAME)
    eval_samples_path = os.path.join(eval_output_dir, SAMPLES_NAME)
    train_meta_path = os.path.join(output_dir, "dataset_meta.json")
    eval_meta_path = os.path.join(eval_output_dir, "dataset_meta.json")
    capacity_report_path = os.path.join(os.path.dirname(os.path.abspath(output_dir)), "capacity_report.json")
    sampling_payload = sokoban_sampling_meta(sampling)
    balanced_meta = dict(build_meta)
    balanced_meta.update(
        {
            "sampling": sampling_payload,
            "eval_output_dir": os.path.abspath(eval_output_dir),
            "num_env_specs": len(specs),
        }
    )
    if (
        not force
        and _is_current(train_meta_path, train_samples_path, {**balanced_meta, "split": "train"})
        and _is_current(eval_meta_path, eval_samples_path, {**balanced_meta, "split": "eval"})
    ):
        logger.info("balanced dataset already current: %s and %s", train_samples_path, eval_samples_path)
        return

    all_train_rows: list[dict[str, Any]] = []
    all_eval_rows: list[dict[str, Any]] = []
    family_reports: list[dict[str, Any]] = []
    for spec_idx, spec in enumerate(specs):
        env_name = _normalize_env_name(spec.name)
        profile = spec.profile or "vision_free_think_local"
        pool_id = spec.pool_id or f"{env_name}:{profile}"
        config = EnvpackAdapterConfig(
            api="in_process",
            pools=(
                EnvpackPoolConfig(
                    env=env_name,
                    profile=profile,
                    pool_id=pool_id,
                    runtime_config={"num_instances": 1, "max_active_episodes_per_instance": 1},
                ),
            ),
            rollout=EnvpackRolloutConfig(),
        )
        bundle = build_in_process_client(config)
        env_config = copy.deepcopy(bundle.env_config(pool_id))
        env_config.update(copy.deepcopy(spec.config))
        seeds = _generate_seeds_for_spec(spec, base_seed, spec_idx)
        spec_sampling = sampling
        spec_sampling_payload = sampling_payload
        if spec.sampling:
            spec_sampling = parse_sampling_spec({**sampling_payload, **copy.deepcopy(spec.sampling)})
            spec_sampling_payload = sokoban_sampling_meta(spec_sampling)
        result = await build_balanced_sokoban_dataset(
            spec=SokobanDatasetSpec(
                n_envs=spec.n_envs,
                seeds=seeds,
                env_config=env_config,
                profile=profile,
                pool_id=pool_id,
                env_name=env_name,
                bucket_prefix=spec.bucket_prefix,
            ),
            sampling=spec_sampling,
        )
        _annotate_family(result.train_rows, spec_idx=spec_idx, pool_id=pool_id)
        _annotate_family(result.eval_rows, spec_idx=spec_idx, pool_id=pool_id)
        all_train_rows.extend(result.train_rows)
        all_eval_rows.extend(result.eval_rows)
        family_reports.append(
            {
                "spec_idx": spec_idx,
                "env_name": env_name,
                "profile": profile,
                "pool_id": pool_id,
                "bucket_prefix": spec.bucket_prefix,
                "sampling": spec_sampling_payload,
                "env_config": env_config,
                "n_train_rows": len(result.train_rows),
                "n_eval_rows": len(result.eval_rows),
                "capacity_report": result.capacity_report,
            }
        )

    _validate_no_env_uuid_overlap(all_train_rows, all_eval_rows)
    capacity_report = _merge_sokoban_capacity_reports(
        family_reports=family_reports,
        sampling_payload=sampling_payload,
    )

    _write_rows(train_samples_path, all_train_rows)
    _write_rows(eval_samples_path, all_eval_rows)

    train_meta = {**balanced_meta, "split": "train", "rows_written": len(all_train_rows)}
    eval_meta = {**balanced_meta, "split": "eval", "rows_written": len(all_eval_rows)}
    _write_json(train_meta_path, train_meta)
    _write_json(eval_meta_path, eval_meta)

    _write_json(capacity_report_path, capacity_report)
    logger.info(
        "wrote balanced Sokoban train=%d eval=%d report=%s",
        len(all_train_rows),
        len(all_eval_rows),
        capacity_report_path,
    )


def _annotate_family(rows: list[dict[str, Any]], *, spec_idx: int, pool_id: str) -> None:
    for row in rows:
        metadata = row.setdefault("metadata", {})
        meta = metadata.setdefault("envpack", {})
        meta["dataset_family_index"] = spec_idx
        meta["dataset_family"] = pool_id


def _validate_no_env_uuid_overlap(train_rows: list[dict[str, Any]], eval_rows: list[dict[str, Any]]) -> None:
    train_seen: dict[str, int] = {}
    eval_seen: dict[str, int] = {}
    for idx, row in enumerate(train_rows):
        env_uuid = _row_env_uuid(row)
        if env_uuid in train_seen:
            raise RuntimeError(f"duplicate train env_uuid {env_uuid!r} at rows {train_seen[env_uuid]} and {idx}")
        train_seen[env_uuid] = idx
    for idx, row in enumerate(eval_rows):
        env_uuid = _row_env_uuid(row)
        if env_uuid in eval_seen:
            raise RuntimeError(f"duplicate eval env_uuid {env_uuid!r} at rows {eval_seen[env_uuid]} and {idx}")
        eval_seen[env_uuid] = idx
    overlap = sorted(set(train_seen).intersection(eval_seen))
    if overlap:
        raise RuntimeError(f"train/eval env_uuid overlap after concat: {overlap[:5]}")


def _row_env_uuid(row: dict[str, Any]) -> str:
    env_uuid = ((row.get("metadata") or {}).get("envpack") or {}).get("env_uuid")
    if not env_uuid:
        raise RuntimeError("balanced Sokoban row is missing metadata.envpack.env_uuid")
    return str(env_uuid)


def _merge_sokoban_capacity_reports(
    *,
    family_reports: list[dict[str, Any]],
    sampling_payload: dict[str, Any],
) -> dict[str, Any]:
    numeric_sum_keys = (
        "candidate_seeds",
        "probed_candidates",
        "unique_env_uuid",
        "duplicate_env_uuid_count",
        "out_of_range_count",
        "generation_failed_count",
        "accepted_candidates",
        "selected_train",
        "selected_eval",
    )
    aggregate: dict[str, Any] = {
        "mode": "balanced_sokoban_concat",
        "num_env_specs": len(family_reports),
        "sampling": sampling_payload,
        "families": family_reports,
    }
    for key in numeric_sum_keys:
        aggregate[key] = sum(int(report["capacity_report"].get(key, 0)) for report in family_reports)

    candidate_seeds = int(aggregate["candidate_seeds"])
    duplicate_count = int(aggregate["duplicate_env_uuid_count"])
    aggregate["duplicate_env_uuid_rate"] = duplicate_count / candidate_seeds if candidate_seeds else 0.0

    for key in (
        "solver_status_counts",
        "min_solve_steps_histogram",
        "critical_steps_histogram",
        "bucket_available",
        "bucket_selected",
        "bucket_train_counts",
        "bucket_eval_counts",
    ):
        aggregate[key] = _sum_int_maps(report["capacity_report"].get(key, {}) for report in family_reports)
    return aggregate


def _sum_int_maps(items: Iterable[dict[str, Any]]) -> dict[str, int]:
    merged: dict[str, int] = {}
    for item in items:
        for key, value in dict(item).items():
            merged[str(key)] = merged.get(str(key), 0) + int(value)
    return dict(sorted(merged.items()))


def _write_rows(path: str, rows: list[dict[str, Any]]) -> None:
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as out_f:
        for row in rows:
            out_f.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp_path, path)


def _load_specs_from_raw(raw: dict[str, Any], path: str) -> list[EnvSpec]:
    specs = [_parse_spec(item, idx) for idx, item in enumerate(raw.get("envs") or [])]
    if not specs:
        raise RuntimeError(f"no envs entries in {path!r}")
    return specs


def _parse_spec(raw: dict[str, Any], idx: int) -> EnvSpec:
    if not isinstance(raw, dict):
        raise RuntimeError(f"envs[{idx}] must be a dict")
    return EnvSpec(
        name=str(raw["name"]),
        n_envs=int(raw["n_envs"]),
        config=dict(raw.get("config") or {}),
        seed=_normalize_seed_directive(raw.get("seed")),
        seed_list=None if raw.get("seed_list") is None else [int(value) for value in raw["seed_list"]],
        profile=None if raw.get("profile") is None else str(raw["profile"]),
        pool_id=None if raw.get("pool_id") is None else str(raw["pool_id"]),
        bucket_prefix=None if raw.get("bucket_prefix") is None else str(raw["bucket_prefix"]),
        sampling=None if raw.get("sampling") is None else dict(raw["sampling"]),
    )


def _parse_sampling(raw: Any):
    if raw is None:
        return None
    try:
        from envpack.envs.sokoban.dataset import parse_sampling_spec
    except Exception as exc:
        raise RuntimeError("balanced sampling requires envpack Sokoban dataset helpers") from exc
    return parse_sampling_spec(raw)


def _generate_seeds_for_spec(spec: EnvSpec, base_seed: int, spec_idx: int) -> list[int]:
    if spec.seed_list is not None:
        if len(spec.seed_list) < spec.n_envs:
            raise ValueError(f"seed_list for env {spec.name!r} has fewer than n_envs values")
        return spec.seed_list[: spec.n_envs]

    directive = _normalize_seed_directive(spec.seed)
    rng = random.Random(_make_rng_seed(base_seed, spec, spec_idx, f"seed-{directive}"))
    if len(directive) == 1:
        return [rng.randrange(0, MAX_INT32 + 1) for _ in range(spec.n_envs)]
    if len(directive) == 2:
        minimum, maximum = directive
        if maximum < minimum:
            raise ValueError("seed[1] must be >= seed[0] when len(seed) == 2")
        return [rng.randrange(minimum, maximum + 1) for _ in range(spec.n_envs)]
    if len(directive) == 3:
        minimum, maximum, limit = directive
        if maximum < minimum:
            raise ValueError("seed[1] must be >= seed[0] when len(seed) == 3")
        if limit <= 0:
            raise ValueError("seed[2] must be positive when len(seed) == 3")
        if (maximum - minimum + 1) * limit < spec.n_envs:
            raise ValueError("seed range with given limit cannot supply enough unique seeds")
        if limit == 1:
            return rng.sample(range(minimum, maximum + 1), spec.n_envs)
        counts: dict[int, int] = {}
        seeds: list[int] = []
        while len(seeds) < spec.n_envs:
            candidate = rng.randint(minimum, maximum)
            if counts.get(candidate, 0) >= limit:
                continue
            counts[candidate] = counts.get(candidate, 0) + 1
            seeds.append(candidate)
        return seeds
    raise ValueError("seed directive must be length 1, 2, or 3")


def _make_rng_seed(base_seed: int, spec: EnvSpec, spec_idx: int, hint: str) -> int:
    payload = f"{base_seed}|{spec_idx}|{spec.name}|{hint}"
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _normalize_seed_directive(seed_field) -> list[int]:
    if seed_field is None:
        return [0]
    if isinstance(seed_field, (int, float)):
        return [int(seed_field)]
    if isinstance(seed_field, Sequence) and not isinstance(seed_field, (str, bytes)):
        coerced = [int(value) for value in seed_field]
        return coerced if coerced else [0]
    raise TypeError("seed must be an integer or a sequence of integers")


def _normalize_env_name(name: str) -> str:
    normalized = str(name).strip().lower()
    aliases = {"sokoban": "sokoban", "frozenlake": "frozenlake", "frozen_lake": "frozenlake"}
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise RuntimeError(f"unsupported envpack env {name!r}") from exc


def _load_excluded_env_uuids(path: str | None) -> set[str]:
    if not path:
        return set()
    excluded: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            env_uuid = ((json.loads(line).get("metadata") or {}).get("envpack") or {}).get("env_uuid")
            if env_uuid:
                excluded.add(str(env_uuid))
    return excluded


def _is_current(meta_path: str, samples_path: str, expected: dict[str, Any]) -> bool:
    if not os.path.exists(meta_path) or not os.path.exists(samples_path):
        return False
    try:
        with open(meta_path, encoding="utf-8") as f:
            current = json.load(f)
    except Exception:
        return False
    return all(current.get(key) == value for key, value in expected.items())


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def _file_sha256(path: str | None) -> str | None:
    if path is None:
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_yaml_or_json(path: str) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8")
    try:
        import yaml
    except Exception:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("PyYAML is required to load non-JSON EnvSpec files") from exc
    return yaml.safe_load(text) or {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build envpack samples.jsonl from EnvSpec yaml")
    parser.add_argument("--yaml", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--exclude-data")
    parser.add_argument("--dedup-within", action="store_true")
    parser.add_argument("--target-kept", type=int)
    parser.add_argument("--eval-output-dir")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s envpack_build_dataset: %(message)s")
    asyncio.run(
        build_dataset(
            yaml_path=args.yaml,
            output_dir=args.output_dir,
            split=args.split,
            base_seed=args.base_seed,
            exclude_data=args.exclude_data,
            dedup_within=args.dedup_within,
            target_kept=args.target_kept,
            eval_output_dir=args.eval_output_dir,
            force=args.force,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
