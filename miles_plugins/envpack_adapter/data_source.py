"""Data source for envpack-backed rollouts.

This mirrors the VAGEN EnvSpec/jsonl source shape, but emits
`sample.metadata["envpack"]` and does not import VAGEN.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from miles.rollout.data_source import RolloutDataSource
from miles.utils.types import Sample
from miles_plugins.envpack_adapter.config import (
    load_envpack_config,
    validate_pool_env_config_overrides,
    validate_runtime_args,
)

logger = logging.getLogger(__name__)

MAX_INT32 = 2**31 - 1


@dataclass(slots=True)
class EnvpackEnvSpec:
    name: str
    n_envs: int
    config: dict[str, Any] = field(default_factory=dict)
    seed: list[int] = field(default_factory=lambda: [0])
    seed_list: list[int] | None = None
    profile: str | None = None
    pool_id: str | None = None


class EnvpackDataSource(RolloutDataSource):
    def __init__(self, args):
        self.args = args
        self.config = load_envpack_config(args)
        validate_runtime_args(args, self.config)
        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        self.metadata: dict[str, Any] = {}
        self.dataset = None

        self._origin_samples = _load_samples(args.prompt_data, args, self.config)
        if not self._origin_samples:
            raise RuntimeError(f"EnvpackDataSource: no samples loaded from {args.prompt_data!r}")
        self._prompt_samples = list(self._origin_samples)
        if getattr(args, "rollout_shuffle", False):
            self._shuffle_for_epoch(self.epoch_id)
        logger.info(
            "EnvpackDataSource: loaded %d samples from %s (rollout_shuffle=%s)",
            len(self._origin_samples),
            args.prompt_data,
            getattr(args, "rollout_shuffle", False),
        )

    def _shuffle_for_epoch(self, epoch_id: int) -> None:
        seed = int(getattr(self.args, "rollout_seed", 0) or 0) + int(epoch_id)
        rng = random.Random(seed)
        indices = list(range(len(self._origin_samples)))
        rng.shuffle(indices)
        self._prompt_samples = [self._origin_samples[index] for index in indices]

    def get_samples(self, num_samples) -> list[list[Sample]]:
        sample_count = len(self._prompt_samples)
        if sample_count == 0:
            raise RuntimeError("EnvpackDataSource: empty prompt-sample pool")

        prompt_samples: list[Sample] = []
        while len(prompt_samples) < num_samples:
            need = num_samples - len(prompt_samples)
            if self.sample_offset + need <= sample_count:
                prompt_samples += self._prompt_samples[self.sample_offset : self.sample_offset + need]
                self.sample_offset += need
                if self.sample_offset == sample_count:
                    self.sample_offset = 0
                    self.epoch_id += 1
                    if getattr(self.args, "rollout_shuffle", False):
                        self._shuffle_for_epoch(self.epoch_id)
            else:
                prompt_samples += self._prompt_samples[self.sample_offset :]
                self.sample_offset = 0
                self.epoch_id += 1
                if getattr(self.args, "rollout_shuffle", False):
                    self._shuffle_for_epoch(self.epoch_id)

        groups: list[list[Sample]] = []
        for prompt_sample in prompt_samples:
            group: list[Sample] = []
            for _ in range(self.args.n_samples_per_prompt):
                sample = copy.deepcopy(prompt_sample)
                sample.group_index = self.sample_group_index
                sample.index = self.sample_index
                self.sample_index += 1
                group.append(sample)
            self.sample_group_index += 1
            groups.append(group)
        return groups

    def add_samples(self, samples):
        if not samples:
            return
        raise RuntimeError(
            f"{type(self).__name__}: partial_rollout is not supported in envpack MVP; "
            f"received {len(samples)} aborted sample groups."
        )

    def save(self, rollout_id):
        state = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
        }
        path = os.path.join(self.args.save, f"rollout/envpack_data_source_state_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)

    def load(self, rollout_id=None):
        if self.args.load is None:
            return
        path = os.path.join(self.args.load, f"rollout/envpack_data_source_state_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info("EnvpackDataSource: no checkpoint at %s", path)
            return
        state = torch.load(path)
        self.sample_offset = state["sample_offset"]
        self.epoch_id = state["epoch_id"]
        self.sample_group_index = state["sample_group_index"]
        self.sample_index = state["sample_index"]
        if getattr(self.args, "rollout_shuffle", False):
            self._shuffle_for_epoch(self.epoch_id)


def _load_samples(path: str, args, adapter_config) -> list[Sample]:
    if str(path).endswith(".jsonl"):
        return _load_jsonl(path)
    return _materialize_envspec_yaml(path, args, adapter_config)


def _load_jsonl(path: str) -> list[Sample]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"EnvpackDataSource: samples jsonl not found at {path!r}")
    anchor = Path(path).resolve().parent
    samples: list[Sample] = []
    with open(path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
            metadata = row.get("metadata") or {}
            envpack = metadata.get("envpack") or row.get("envpack") or {}
            if not envpack.get("env_name") or envpack.get("seed") is None:
                raise RuntimeError(f"{path}:{line_no}: missing envpack.env_name or envpack.seed")
            render_mode = (envpack.get("env_config") or {}).get("render_mode")
            if render_mode == "vision" and not envpack.get("env_uuid"):
                raise RuntimeError(
                    f"{path}:{line_no}: vision row missing envpack.env_uuid. "
                    "Rebuild the dataset with `python -m miles_plugins.envpack_adapter.build_env_dataset`."
                )
            image_path = envpack.get("image_path")
            if image_path:
                envpack["image_path"] = str(anchor / image_path)
            envpack["source_format"] = envpack.get("source_format") or "samples_jsonl"
            samples.append(Sample(prompt="", metadata={"envpack": envpack}))
    return samples


def _materialize_envspec_yaml(path: str, args, adapter_config) -> list[Sample]:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"EnvpackDataSource: EnvSpec yaml not found at {path!r}")
    raw = _load_yaml_or_json(path)
    env_specs = [_parse_spec(item, idx) for idx, item in enumerate(raw.get("envs") or [])]
    if not env_specs:
        raise RuntimeError(f"EnvpackDataSource: no envs entries in {path!r}")

    base_seed = int(getattr(args, "seed", 0) or 0)
    samples: list[Sample] = []
    for spec_idx, spec in enumerate(env_specs):
        env_name = _normalize_env_name(spec.name)
        pool = (
            adapter_config.pool_by_id(spec.pool_id)
            if spec.pool_id is not None
            else adapter_config.pool_for_env(env_name)
        )
        seeds = _generate_seeds_for_spec(spec, base_seed, spec_idx)
        pool_env_config = _validate_envspec_yaml_pool_env_config(env_name, pool.env_config)
        env_config = copy.deepcopy(spec.config)
        env_config.update(pool_env_config)
        for env_seed in seeds:
            samples.append(
                Sample(
                    prompt="",
                    metadata={
                        "envpack": {
                            "env_name": env_name,
                            "seed": int(env_seed),
                            "env_config": copy.deepcopy(env_config),
                            "profile": spec.profile or pool.profile,
                            "pool_id": spec.pool_id or pool.resolved_pool_id,
                            "source_format": "envspec_yaml",
                        }
                    },
                )
            )
    logger.info("EnvpackDataSource: materialized %d samples from EnvSpec yaml %s", len(samples), path)
    return samples


def _validate_envspec_yaml_pool_env_config(env_name: str, pool_env_config: dict[str, Any]) -> dict[str, Any]:
    return validate_pool_env_config_overrides(env_name, pool_env_config, context="EnvSpec YAML data-source path")


def _parse_spec(raw: dict[str, Any], idx: int) -> EnvpackEnvSpec:
    if not isinstance(raw, dict):
        raise RuntimeError(f"envs[{idx}] must be a dict")
    return EnvpackEnvSpec(
        name=str(raw["name"]),
        n_envs=int(raw["n_envs"]),
        config=dict(raw.get("config") or {}),
        seed=_normalize_seed_directive(raw.get("seed")),
        seed_list=None if raw.get("seed_list") is None else [int(value) for value in raw["seed_list"]],
        profile=None if raw.get("profile") is None else str(raw["profile"]),
        pool_id=None if raw.get("pool_id") is None else str(raw["pool_id"]),
    )


def _normalize_env_name(name: str) -> str:
    value = str(name).strip().lower()
    aliases = {"sokoban": "sokoban", "frozenlake": "frozenlake", "frozen_lake": "frozenlake"}
    try:
        return aliases[value]
    except KeyError as exc:
        raise RuntimeError(f"unsupported envpack env {name!r}") from exc


def _make_rng_seed(base_seed: int, spec: EnvpackEnvSpec, spec_idx: int, hint: str) -> int:
    payload = f"{base_seed}|{spec_idx}|{spec.name}|{hint}"
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def _coerce_to_int_list(values: Sequence | None) -> list[int] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise TypeError("seed_list must be a sequence of integers, not string")
    return [int(value) for value in values]


def _normalize_seed_directive(seed_field) -> list[int]:
    if seed_field is None:
        return [0]
    if isinstance(seed_field, (int, float)):
        return [int(seed_field)]
    if isinstance(seed_field, Sequence) and not isinstance(seed_field, (str, bytes)):
        coerced = [int(value) for value in seed_field]
        return coerced if coerced else [0]
    raise TypeError("seed must be an integer or a sequence of integers")


def _generate_seeds_for_spec(spec: EnvpackEnvSpec, base_seed: int, spec_idx: int) -> list[int]:
    explicit_list = _coerce_to_int_list(spec.seed_list)
    if explicit_list is not None:
        if len(explicit_list) < spec.n_envs:
            raise ValueError(f"seed_list for env {spec.name!r} must contain at least n_envs values")
        return explicit_list[: spec.n_envs]

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
            raise ValueError("seed[2] must be a positive integer when len(seed) == 3")
        if (maximum - minimum + 1) * limit < spec.n_envs:
            raise ValueError("seed range with given limit cannot supply enough unique seeds for n_envs")
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
    raise ValueError("seed directive must be of length 1, 2, or 3")


def _load_yaml_or_json(path: str) -> dict[str, Any]:
    with open(path) as f:
        text = f.read()
    try:
        import yaml
    except Exception:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                "PyYAML is required to load non-JSON envpack EnvSpec files. "
                "Install IMP-Miles dependencies or provide JSON-compatible EnvSpec for tests."
            ) from exc
    return yaml.safe_load(text) or {}
