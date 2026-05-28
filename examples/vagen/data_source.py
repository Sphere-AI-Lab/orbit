"""In-memory data source for VAGEN MVP.

Reads either an EnvSpec yaml (original VAGEN semantics — materialize seeds
in-process) or a prebuilt `samples.jsonl` (from `build_env_dataset`).
Bypasses `RolloutDataSource`'s HF-Dataset path because neither input is the
parquet prompt table the base class expects.

See `examples/vagen/docs/dataset.md` for the row schema, the two input
modes, GRPO-group deepcopy semantics, and the `rollout_global_dataset`
constraint.
"""

import copy
import json
import logging
import os
import random
from typing import Any

import torch

from miles.rollout.data_source import RolloutDataSource
from miles.utils.types import Sample

logger = logging.getLogger(__name__)


class VagenEnvSpecDataSource(RolloutDataSource):
    def __init__(self, args):
        # Skip super().__init__() — it would treat prompt_data as a parquet path.
        self.args = args
        self.epoch_id = 0
        self.sample_group_index = 0
        self.sample_index = 0
        self.sample_offset = 0
        self.metadata: dict = {}
        self.dataset = None

        # _origin_samples is the canonical load order; per-epoch shuffles
        # re-derive from it. `.jsonl` → prebuilt; else → EnvSpec yaml.
        self._origin_samples: list[Sample] = _load_samples(args.prompt_data, args)
        if not self._origin_samples:
            raise RuntimeError(
                f"VagenEnvSpecDataSource: no samples loaded from {args.prompt_data!r}. "
                f"For jsonl, build via `python -m examples.vagen.build_env_dataset`; "
                f"for yaml, check the `envs:` list and `n_envs` fields."
            )
        self._prompt_samples: list[Sample] = list(self._origin_samples)
        if getattr(args, "rollout_shuffle", False):
            self._shuffle_for_epoch(self.epoch_id)
        logger.info(
            "VagenEnvSpecDataSource: loaded %d samples from %s (rollout_shuffle=%s)",
            len(self._origin_samples),
            args.prompt_data,
            getattr(args, "rollout_shuffle", False),
        )

    def _shuffle_for_epoch(self, epoch_id: int) -> None:
        seed = int(getattr(self.args, "rollout_seed", 0) or 0) + int(epoch_id)
        rng = random.Random(seed)
        permutation = list(range(len(self._origin_samples)))
        rng.shuffle(permutation)
        self._prompt_samples = [self._origin_samples[i] for i in permutation]

    def get_samples(self, num_samples) -> list[list[Sample]]:
        # Walk sample_offset / epoch wraparound ourselves (no HF Dataset).
        # Multi-wrap needed: num_samples can exceed len(_prompt_samples).
        N = len(self._prompt_samples)
        if N == 0:
            raise RuntimeError("VagenEnvSpecDataSource: empty prompt-sample pool")

        prompt_samples: list[Sample] = []
        while len(prompt_samples) < num_samples:
            need = num_samples - len(prompt_samples)
            if self.sample_offset + need <= N:
                prompt_samples += self._prompt_samples[self.sample_offset : self.sample_offset + need]
                self.sample_offset += need
                if self.sample_offset == N:
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
        for ps in prompt_samples:
            group: list[Sample] = []
            for _ in range(self.args.n_samples_per_prompt):
                s = copy.deepcopy(ps)
                s.group_index = self.sample_group_index
                s.index = self.sample_index
                self.sample_index += 1
                group.append(s)
            self.sample_group_index += 1
            groups.append(group)
        return groups

    def add_samples(self, samples):
        # miles' rollout path calls this unconditionally with aborted_samples
        # (sglang_rollout.py, inference_rollout_common.py). Empty is the
        # no-partial_rollout case; raise on non-empty to catch scope creep.
        if not samples:
            return
        raise RuntimeError(
            f"{type(self).__name__}: partial_rollout is not supported in MVP; "
            f"received {len(samples)} aborted samples."
        )

    def save(self, rollout_id):
        # Skip super().save() — base would try to dump an HF Dataset.
        state = {
            "sample_offset": self.sample_offset,
            "epoch_id": self.epoch_id,
            "sample_group_index": self.sample_group_index,
            "sample_index": self.sample_index,
        }
        path = os.path.join(self.args.save, f"rollout/vagen_data_source_state_{rollout_id}.pt")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(state, path)

    def load(self, rollout_id=None):
        if self.args.load is None:
            return
        path = os.path.join(self.args.load, f"rollout/vagen_data_source_state_{rollout_id}.pt")
        if not os.path.exists(path):
            logger.info("VagenEnvSpecDataSource: no checkpoint at %s", path)
            return
        st = torch.load(path)
        self.sample_offset = st["sample_offset"]
        self.epoch_id = st["epoch_id"]
        self.sample_group_index = st["sample_group_index"]
        self.sample_index = st["sample_index"]


def _load_samples(path: str, args) -> list[Sample]:
    if str(path).endswith(".jsonl"):
        return _load_samples_jsonl(path)
    return _materialize_envspec_yaml(path, args)


def _materialize_envspec_yaml(yaml_path: str, args) -> list[Sample]:
    """Expand a VAGEN EnvSpec yaml into samples (live-yaml path).

    No `env_uuid` is attached, so rollout skips drift detection for these rows.
    """
    from vagen.gym_agent_dataset import _generate_seeds_for_spec, load_envspecs

    env_specs = load_envspecs(yaml_path).specs
    base_seed = int(getattr(args, "seed", 0) or 0)

    samples: list[Sample] = []
    for spec_idx, spec in enumerate(env_specs):
        seeds = _generate_seeds_for_spec(spec, base_seed, spec_idx)
        spec_config = _to_plain_dict(spec.config)
        for env_seed in seeds:
            meta: dict[str, Any] = {
                "vagen": {
                    "env_name": spec.name,
                    "seed": int(env_seed),
                    "config": copy.deepcopy(spec_config),
                    "max_turns": int(spec.max_turns),
                    "response_length_per_turn": (
                        None if spec.response_length_per_turn is None else int(spec.response_length_per_turn)
                    ),
                    "source_format": "envspec_yaml",
                    "drift_check_required": False,
                },
            }
            # `prompt` is filled in inside generate() after env.reset.
            samples.append(Sample(prompt="", metadata=meta))
    logger.info("VagenEnvSpecDataSource: materialized %d samples from EnvSpec yaml %s", len(samples), yaml_path)
    return samples


def _to_plain_dict(value: Any) -> dict:
    """Coerce an OmegaConf DictConfig (or anything dict-like) to a plain dict."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(value, resolve=True)  # type: ignore[return-value]
    except Exception:
        return dict(value)


def _load_samples_jsonl(jsonl_path: str) -> list[Sample]:
    """Load a samples.jsonl produced by `build_env_dataset`.

    Resolves `image_path` against the jsonl's own dir (consumers don't have
    to re-resolve against cwd). `prompt` stays empty — rollout fills it.
    """
    if not jsonl_path or not os.path.exists(jsonl_path):
        raise FileNotFoundError(
            f"VagenEnvSpecDataSource: samples.jsonl not found at {jsonl_path!r}. "
            f"Build it via `python -m examples.vagen.build_env_dataset`."
        )
    anchor = os.path.dirname(os.path.abspath(jsonl_path))
    out: list[Sample] = []
    with open(jsonl_path) as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{jsonl_path}:{line_no}: invalid JSON ({exc})") from exc
            meta = row.get("metadata") or {}
            vagen = meta.get("vagen") or {}
            if not vagen.get("env_name") or vagen.get("seed") is None:
                raise RuntimeError(f"{jsonl_path}:{line_no}: missing metadata.vagen.env_name or seed")
            # Require env_uuid for vision rows — see docs/dataset.md (drift
            # detection). Text-only envs are exempt (no image to hash).
            render_mode = (vagen.get("config") or {}).get("render_mode")
            if render_mode == "vision" and not vagen.get("env_uuid"):
                raise RuntimeError(
                    f"{jsonl_path}:{line_no}: vision row missing env_uuid. "
                    f"Rebuild the dataset with `python -m examples.vagen.build_env_dataset`."
                )
            rel = vagen.get("image_path")
            if rel:
                vagen["image_path"] = os.path.join(anchor, rel)
            vagen["source_format"] = vagen.get("source_format") or "samples_jsonl"
            if render_mode == "vision":
                vagen["drift_check_required"] = True
            # `prompt` is overridden inside rollout.generate after env.reset.
            out.append(Sample(prompt="", metadata={"vagen": vagen}))
    return out


def _main() -> None:
    """Tiny CLI to peek at a built samples.jsonl. Not used by training."""
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a VAGEN samples.jsonl.")
    parser.add_argument("path", help="Path to samples.jsonl.")
    parser.add_argument("--head", type=int, default=2, help="Print this many rows. Default 2.")
    args = parser.parse_args()
    samples = _load_samples_jsonl(args.path)
    print(f"loaded {len(samples)} rows from {args.path}")
    for i, s in enumerate(samples[: args.head]):
        v = s.metadata.get("vagen", {})
        print(
            f"row {i}: env={v.get('env_name')} seed={v.get('seed')} "
            f"env_uuid={v.get('env_uuid')!r:20s} split={v.get('split')!r} "
            f"heldout={v.get('heldout')}"
        )


if __name__ == "__main__":
    _main()
