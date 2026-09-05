"""Offline builder for a VAGEN env-prompt dataset.

Runs `env.reset(seed)` once per row, writes the turn-0 PIL as a PNG, and
emits one jsonl row carrying `metadata.vagen.{env_name, seed, config,
max_turns, response_length_per_turn, env_uuid, image_path, split, heldout}`.
The jsonl is the single source of truth — both training and orbit eval
load it via `--prompt-data` / `--eval-prompt-data`.

See `examples/vagen/docs/dataset.md` for the row schema, drift-detection
design, heldout strategy (--exclude-data / --target-kept), idempotence,
and determinism prerequisites.
"""

import argparse
import asyncio
import copy
import hashlib
import json
import logging
import os
import sys
from typing import Any

logger = logging.getLogger(__name__)


_IMAGES_SUBDIR = "images"
_SAMPLES_NAME = "samples.jsonl"
_META_NAME = "dataset_meta.json"
_VALID_SPLITS = ("train", "eval", "eval_heldout")


def _spec_token(env_name: str, env_config: dict) -> str:
    """Short discriminator that disambiguates filenames within a multi-spec
    yaml. Same `(env_name, config)` -> same token; different config -> almost
    surely different token (8 hex chars = 32 bits, plenty for our scale)."""
    canonical = json.dumps(env_config or {}, sort_keys=True, default=str)
    return hashlib.md5(f"{env_name}|{canonical}".encode()).hexdigest()[:8]


def _save_png_and_uuid(pil, png_path: str) -> str:
    """Persist `pil` as PNG to `png_path`; return md5(bytes) as env_uuid.
    Hashes the bytes we just wrote out so a later `md5(open(...).read())`
    will match exactly."""
    parent = os.path.dirname(png_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    pil.save(png_path, format="PNG")
    with open(png_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


async def _reset_and_save(cls, env_config: dict, seed: int, png_path: str) -> str | None:
    """Run env.reset once, save its PIL as PNG, return env_uuid. Returns
    None when the env emits no image (text-only render); the row is then
    dropped."""
    env = cls(env_config=env_config)
    try:
        obs, _ = await env.reset(seed)
        pils = (obs.get("multi_modal_input") or {}).get("<image>") or []
        if not pils:
            return None
        return _save_png_and_uuid(pils[0], png_path)
    finally:
        await env.close()


def _load_excluded_env_uuids(exclude_path: str | None) -> set[str]:
    """Collect the env_uuid set from another samples.jsonl (used by
    --exclude-data to derive a map-heldout split)."""
    if not exclude_path:
        return set()
    out: set[str] = set()
    with open(exclude_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uuid = ((row.get("metadata") or {}).get("vagen") or {}).get("env_uuid")
            if uuid:
                out.add(str(uuid))
    return out


def _to_plain_dict(value: Any) -> dict:
    """Coerce an OmegaConf DictConfig to a plain dict so json.dump succeeds."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    try:
        from omegaconf import OmegaConf

        return OmegaConf.to_container(value, resolve=True)  # type: ignore[return-value]
    except Exception:
        return dict(value)


def _yaml_md5(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def _file_md5(path: str | None) -> str | None:
    if not path:
        return None
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


async def _build(
    yaml_path: str,
    base_seed: int,
    output_dir: str,
    split: str,
    excluded_uuids: set[str],
    target_kept: int | None,
    dedup_within: bool,
) -> dict[str, Any]:
    """Walk (env, seed) pairs sequentially (VAGEN's set_seed is global-state),
    write PNGs, and stream rows to `samples.jsonl.tmp`. Caller does the
    atomic rename only after the target_kept floor check passes."""
    from vagen.envs.registry import get_env_cls
    from vagen.gym_agent_dataset import _generate_seeds_for_spec, load_envspecs

    env_specs = load_envspecs(yaml_path).specs
    images_dir = os.path.join(output_dir, _IMAGES_SUBDIR)
    os.makedirs(images_dir, exist_ok=True)
    tmp_samples_path = os.path.join(output_dir, _SAMPLES_NAME + ".tmp")
    # Unlink a stale .tmp from a previously-killed build (would pollute rows).
    try:
        os.unlink(tmp_samples_path)
    except FileNotFoundError:
        pass

    rows_written = 0
    n_excluded = 0
    n_no_image = 0
    n_intra_dup = 0
    # dedup_within: env_uuids already kept by THIS build count as excluded
    # for subsequent candidates (N rows → N unique maps; see docs/dataset.md).
    kept_uuids: set[str] = set()
    heldout_flag = bool(excluded_uuids)
    limit_reached = False
    with open(tmp_samples_path, "w") as out_f:
        for spec_idx, spec in enumerate(env_specs):
            if limit_reached:
                break
            cls = get_env_cls(spec.name)
            seeds = _generate_seeds_for_spec(spec, base_seed, spec_idx)
            spec_config = _to_plain_dict(spec.config)
            spec_tok = _spec_token(spec.name, spec_config)
            for env_seed in seeds:
                if target_kept is not None and rows_written >= target_kept:
                    limit_reached = True
                    break
                rel_image_path = os.path.join(_IMAGES_SUBDIR, f"seed_{int(env_seed):08d}_{spec_tok}.png")
                abs_image_path = os.path.join(output_dir, rel_image_path)
                uuid = await _reset_and_save(cls, copy.deepcopy(spec_config), int(env_seed), abs_image_path)
                if uuid is None:
                    n_no_image += 1
                    continue
                if uuid in excluded_uuids:
                    # Drop the PNG to keep images/ in sync with samples.jsonl.
                    try:
                        os.unlink(abs_image_path)
                    except OSError:
                        pass
                    n_excluded += 1
                    if (n_excluded + n_intra_dup + rows_written) % 200 == 0:
                        logger.info(
                            "  processed %d seeds (%d kept, %d excluded, %d intra-dup)",
                            n_excluded + n_intra_dup + rows_written,
                            rows_written,
                            n_excluded,
                            n_intra_dup,
                        )
                    continue
                if dedup_within and uuid in kept_uuids:
                    try:
                        os.unlink(abs_image_path)
                    except OSError:
                        pass
                    n_intra_dup += 1
                    if (n_excluded + n_intra_dup + rows_written) % 200 == 0:
                        logger.info(
                            "  processed %d seeds (%d kept, %d excluded, %d intra-dup)",
                            n_excluded + n_intra_dup + rows_written,
                            rows_written,
                            n_excluded,
                            n_intra_dup,
                        )
                    continue
                kept_uuids.add(uuid)
                # source_format + drift_check_required let both consumer paths
                # (train data_source / orbit eval Dataset) skip re-deriving them.
                render_mode = (spec_config or {}).get("render_mode")
                row = {
                    "input": "vagen_placeholder",
                    "images": [],
                    "metadata": {
                        "vagen": {
                            "env_name": spec.name,
                            "seed": int(env_seed),
                            "config": copy.deepcopy(spec_config),
                            "max_turns": int(spec.max_turns),
                            "response_length_per_turn": (
                                None if spec.response_length_per_turn is None else int(spec.response_length_per_turn)
                            ),
                            "env_uuid": uuid,
                            "image_path": rel_image_path,
                            "split": split,
                            "heldout": heldout_flag,
                            "source_format": "samples_jsonl",
                            "drift_check_required": render_mode == "vision",
                        }
                    },
                }
                out_f.write(json.dumps(row) + "\n")
                rows_written += 1
                if rows_written % 200 == 0:
                    logger.info("  built %d rows (%d excluded so far)", rows_written, n_excluded)

    n_unique = _count_unique_uuids(tmp_samples_path)
    return {
        "yaml_path": os.path.abspath(yaml_path),
        "base_seed": base_seed,
        "split": split,
        "rows_written": rows_written,
        "n_excluded": n_excluded,
        "n_intra_dup": n_intra_dup,
        "n_no_image": n_no_image,
        "n_unique_env_uuids": n_unique,
        "dedup_within": dedup_within,
        "limit_reached": limit_reached,
        "_tmp_samples_path": tmp_samples_path,
    }


def _count_unique_uuids(samples_path: str) -> int:
    seen: set[str] = set()
    with open(samples_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            uuid = ((json.loads(line).get("metadata") or {}).get("vagen") or {}).get("env_uuid")
            if uuid:
                seen.add(str(uuid))
    return len(seen)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a per-(env, seed) image dataset + samples.jsonl for a VAGEN EnvSpec yaml."
    )
    parser.add_argument(
        "--yaml",
        required=True,
        dest="yaml_path",
        help="Path to the EnvSpec yaml (train or eval split — same script handles both).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory. samples.jsonl lands at <output-dir>/samples.jsonl; "
        "PNGs at <output-dir>/images/. Convention: data/<dataset>/{train,eval,eval_heldout}/.",
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=_VALID_SPLITS,
        help="Stamps metadata.vagen.split on each row.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="Forwarded to vagen's _generate_seeds_for_spec. VAGEN's default is 0 "
        "(see vagen/gym_agent_dataset.py: config.get('base_seed', 0)); mirror it "
        "here so the baked seeds match VAGEN main's config. Default 0.",
    )
    parser.add_argument(
        "--exclude-data",
        default=None,
        help="Optional path to another samples.jsonl. Any (env, seed) whose env_uuid "
        "appears in that file is dropped from the output. Use this to derive a "
        "map-heldout eval set from (eval_pool_yaml, train_samples.jsonl).",
    )
    parser.add_argument(
        "--target-kept",
        type=int,
        default=None,
        help="'Exactly N rows or fail': short-circuit the build once N rows "
        "survive the exclude filter, and fail if fewer than N survive after "
        "the whole pool is walked. Use with --exclude-data to draw exactly N "
        "heldout seeds from a larger candidate pool.",
    )
    parser.add_argument(
        "--dedup-within",
        action="store_true",
        help="Drop candidates whose env_uuid duplicates one already kept by THIS "
        "build (Sokoban's many-to-one seed->map orbit). With --target-kept this "
        "promotes 'N rows' to 'N unique maps'. Default off; recipes set it for "
        "eval splits where measurement quality matters.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if dataset_meta.json matches the current input "
        "(yaml_md5, base_seed, exclude_data_md5, target_kept, dedup_within).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s build_env_dataset: %(message)s")

    meta_path = os.path.join(args.output_dir, _META_NAME)
    samples_path = os.path.join(args.output_dir, _SAMPLES_NAME)
    yaml_md5 = _yaml_md5(args.yaml_path)
    exclude_md5 = _file_md5(args.exclude_data)
    if os.path.exists(meta_path) and os.path.exists(samples_path) and not args.force:
        try:
            with open(meta_path) as f:
                existing = json.load(f)
            if (
                existing.get("yaml_md5") == yaml_md5
                and existing.get("base_seed") == args.base_seed
                and existing.get("exclude_data_md5") == exclude_md5
                and existing.get("target_kept") == args.target_kept
                and existing.get("split") == args.split
                and existing.get("dedup_within") == args.dedup_within
            ):
                logger.info(
                    "dataset at %s already fresh (yaml_md5 + base_seed + "
                    "exclude_data + target_kept + dedup_within + split match); use --force to rebuild",
                    args.output_dir,
                )
                return 0
        except Exception as exc:
            logger.warning("could not parse existing %s (%s); rebuilding", meta_path, exc)

    os.makedirs(args.output_dir, exist_ok=True)
    if args.force:
        # Wipe stale PNGs so images/ stays in lockstep with the new jsonl.
        images_dir = os.path.join(args.output_dir, _IMAGES_SUBDIR)
        if os.path.isdir(images_dir):
            for name in os.listdir(images_dir):
                if name.endswith(".png"):
                    try:
                        os.unlink(os.path.join(images_dir, name))
                    except OSError:
                        pass
    excluded = _load_excluded_env_uuids(args.exclude_data)
    if excluded:
        logger.info("loaded %d env_uuids to exclude from %s", len(excluded), args.exclude_data)
    payload = asyncio.run(
        _build(
            args.yaml_path,
            args.base_seed,
            args.output_dir,
            args.split,
            excluded,
            args.target_kept,
            args.dedup_within,
        )
    )
    payload["yaml_md5"] = yaml_md5
    payload["exclude_data_md5"] = exclude_md5
    payload["exclude_data_path"] = os.path.abspath(args.exclude_data) if args.exclude_data else None
    payload["target_kept"] = args.target_kept

    tmp_samples_path = payload.pop("_tmp_samples_path")
    if args.target_kept is not None and payload["rows_written"] < args.target_kept:
        # Leave .tmp behind for inspection; do NOT promote it to samples.jsonl
        # (the launcher's `-s samples.jsonl` check would accept a short partial).
        raise RuntimeError(
            f"build_env_dataset: --target-kept={args.target_kept} but only "
            f"{payload['rows_written']} rows survived the exclude filter. "
            f"Widen the candidate yaml or relax --target-kept. "
            f"Inspect partial output at {tmp_samples_path}."
        )

    # Atomic promote: os.replace is atomic on POSIX. A kill between the rename
    # and the meta write triggers a rebuild next run (idempotence needs both).
    os.replace(tmp_samples_path, samples_path)
    with open(meta_path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(
        "wrote %d rows (%d unique env_uuids, %d excluded, %d intra-dup, %d no-image) -> " "samples=%s, images=%s/",
        payload["rows_written"],
        payload["n_unique_env_uuids"],
        payload["n_excluded"],
        payload["n_intra_dup"],
        payload["n_no_image"],
        samples_path,
        os.path.join(args.output_dir, _IMAGES_SUBDIR),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
