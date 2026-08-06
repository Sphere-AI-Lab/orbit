import json
import logging
import os
import re
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist

# Follow-up: may need to copy those 2 functions and do refactoring.
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
from megatron.training.checkpointing import save_checkpoint as _save_checkpoint_megatron
from megatron.training.global_vars import get_args

from orbit.utils import megatron_bridge_utils

from .low_precision_bootstrap import (
    is_distributed_checkpoint,
    is_legacy_megatron_checkpoint,
    load_dist_checkpoint,
    resolve_distributed_checkpoint_dir,
    validate_low_precision_bootstrap_args,
)
from .peft_utils import (
    is_peft_enabled,
    is_peft_model,
    load_peft_adapter,
    preflight_peft_adapter_checkpoint,
    save_peft_checkpoint,
)

try:
    # Here we patch out the `validate_non_overlapping_shards_metadata` in both functions
    # because it is really slow for large models with many shards.
    # Follow-up: find a less hacky way to do this.
    import torch.distributed._shard.sharding_spec as shard_spec
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata
    from torch.distributed._shard.sharded_tensor.shard import Shard
    from torch.distributed._shard.sharded_tensor.utils import _parse_and_validate_remote_device
    from torch.distributed._shard.sharding_spec.api import EnumerableShardingSpec

    def __post_init__(self):
        pass

    EnumerableShardingSpec.__post_init__ = __post_init__

    @classmethod
    def _init_from_local_shards_and_global_metadata(  # type: ignore[override]
        cls,
        local_shards: list[Shard],
        sharded_tensor_metadata: ShardedTensorMetadata,
        process_group=None,
        init_rrefs=False,
        sharding_spec=None,
    ) -> ShardedTensor:
        """
        Initialize a ShardedTensor with local shards and a global
        ShardedTensorMetadata built on each rank.

        Warning: This API is experimental and subject to change. It does
                 not do cross rank validations, and fully rely on the user
                 for the correctness of sharded_tensor_metadata on each rank
        """
        process_group = cls._normalize_pg(process_group)
        current_rank = dist.get_rank()  # intentional to get global rank

        shards_metadata = sharded_tensor_metadata.shards_metadata

        local_shard_metadatas = []

        # collect local shard metadatas from the global sharded_tensor_metadata
        for shard_metadata in shards_metadata:  # type: ignore[attr-defined]
            rank, local_device = _parse_and_validate_remote_device(process_group, shard_metadata.placement)

            if current_rank == rank:
                local_shard_metadatas.append(shard_metadata)

        shards_metadata = sharded_tensor_metadata.shards_metadata
        tensor_properties = sharded_tensor_metadata.tensor_properties

        if sharding_spec is None:
            spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
        else:
            spec = sharding_spec

        sharded_tensor = ShardedTensor.__new__(
            ShardedTensor,
            spec,
            sharded_tensor_metadata.size,
            dtype=tensor_properties.dtype,
            layout=tensor_properties.layout,
            pin_memory=tensor_properties.pin_memory,
            requires_grad=tensor_properties.requires_grad,
        )

        # done validation, add local_shards
        sharded_tensor._local_shards = local_shards
        sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        # run post initialization, i.e. map registration, rpc initialization
        sharded_tensor._post_init()
        return sharded_tensor

    ShardedTensor._init_from_local_shards_and_global_metadata = _init_from_local_shards_and_global_metadata

except ImportError:
    pass

logger = logging.getLogger(__name__)

__all__ = ["save_checkpoint", "save_checkpoint_with_peft", "load_checkpoint"]

_ORBIT_TRAINING_CHECKPOINT_MARKER = ".orbit_training_checkpoint.json"
_ORBIT_TRAINING_CHECKPOINT_FORMAT = "orbit.training_checkpoint"
_ORBIT_TRAINING_CHECKPOINT_VERSION = 1
_CHECKPOINT_ROLES = frozenset({"actor", "critic"})
_ITERATION_DIRECTORY_RE = re.compile(r"iter_(\d+)")
_MEGATRON_TRACKER_FILE = "latest_checkpointed_iteration.txt"


class _ExplicitZeroCheckpointStep(int):
    """Keep explicit step zero truthy for Megatron versions that test it as a bool."""

    def __new__(cls):
        return super().__new__(cls, 0)

    def __bool__(self):
        return True


def _bounded_nonnegative_integer(value) -> bool:
    return type(value) is int and 0 <= value < 2**63


def _checkpoint_iteration_dir(save_root: str | Path, iteration: int, *, release: bool = False) -> Path:
    directory = "release" if release else f"iter_{iteration:07d}"
    save_root = Path(save_root)
    # This also makes the marker helper safe for callers that already resolved
    # a direct per-iteration checkpoint path.
    return save_root if save_root.name == directory else save_root / directory


def _model_checkpoint_role(model) -> str | None:
    if not model:
        return None
    role = getattr(model[0], "role", None)
    return role if role in _CHECKPOINT_ROLES else None


def _write_orbit_training_checkpoint_marker(
    save_root: str | Path,
    iteration: int,
    role: str,
    *,
    optimizer_state_saved: bool,
    scheduler_state_saved: bool,
    release: bool = False,
) -> Path:
    if not _bounded_nonnegative_integer(iteration):
        raise ValueError(f"invalid Orbit checkpoint iteration: {iteration!r}")
    if role not in _CHECKPOINT_ROLES:
        raise ValueError(f"invalid Orbit checkpoint role: {role!r}")

    checkpoint_dir = _checkpoint_iteration_dir(save_root, iteration, release=release)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    marker_path = checkpoint_dir / _ORBIT_TRAINING_CHECKPOINT_MARKER
    marker = {
        "format": _ORBIT_TRAINING_CHECKPOINT_FORMAT,
        "version": _ORBIT_TRAINING_CHECKPOINT_VERSION,
        "iteration": iteration,
        "role": role,
        "optimizer_state_saved": bool(optimizer_state_saved),
        "scheduler_state_saved": bool(scheduler_state_saved),
    }
    temporary_path = marker_path.with_name(f"{marker_path.name}.tmp-{os.getpid()}")
    temporary_path.write_text(json.dumps(marker, sort_keys=True) + "\n")
    os.replace(temporary_path, marker_path)
    return marker_path


def save_checkpoint(iteration, model, optimizer, opt_param_scheduler, *args, **kwargs):
    """Save through Megatron and mark the result as an Orbit training checkpoint.

    The per-iteration marker is written only after Megatron accepts the save. For
    asynchronous saves, Megatron updates the root tracker only after finalization,
    so an early marker in an incomplete iteration directory is never selected as
    the latest resumable checkpoint.
    """
    runtime_args = get_args()
    role = _model_checkpoint_role(model)
    result = _save_checkpoint_megatron(iteration, model, optimizer, opt_param_scheduler, *args, **kwargs)

    is_rank_zero = not dist.is_initialized() or dist.get_rank() == 0
    if is_rank_zero and role is not None and getattr(runtime_args, "save", None) is not None:
        optimizer_state_saved = (
            optimizer is not None
            and not getattr(optimizer, "is_stub_optimizer", False)
            and not getattr(runtime_args, "no_save_optim", False)
        )
        scheduler_state_saved = opt_param_scheduler is not None and not getattr(runtime_args, "no_save_optim", False)
        marker_path = _write_orbit_training_checkpoint_marker(
            runtime_args.save,
            iteration,
            role,
            optimizer_state_saved=optimizer_state_saved,
            scheduler_state_saved=scheduler_state_saved,
            release=bool(kwargs.get("release", False)),
        )
        logger.info("Wrote Orbit training checkpoint marker to %s", marker_path)

    return result


def _resolve_selected_distributed_checkpoint(args) -> Path | None:
    load_path = Path(args.load)
    checkpoint_step = getattr(args, "ckpt_step", None)
    if checkpoint_step is not None:
        if not _bounded_nonnegative_integer(checkpoint_step):
            raise ValueError(f"invalid checkpoint step: {checkpoint_step!r}")
        if (load_path / ".metadata").is_file():
            candidate = load_path.resolve(strict=True)
            direct_iteration = _ITERATION_DIRECTORY_RE.fullmatch(candidate.name)
            if direct_iteration is not None:
                requested_iteration = int(direct_iteration.group(1))
            else:
                marker = _read_orbit_training_checkpoint_marker(candidate)
                if marker is None:
                    raise ValueError(
                        f"cannot validate --ckpt-step={checkpoint_step} against direct checkpoint path {load_path}"
                    )
                requested_iteration = marker["iteration"]
            if requested_iteration != checkpoint_step:
                raise ValueError(
                    f"--ckpt-step={checkpoint_step} does not match direct checkpoint path {load_path.name}"
                )
        else:
            candidate = load_path / f"iter_{checkpoint_step:07d}"
        if not (candidate / ".metadata").is_file():
            raise FileNotFoundError(f"distributed checkpoint iteration {checkpoint_step} not found at {candidate}")
        return candidate.resolve(strict=True)
    checkpoint_dir = resolve_distributed_checkpoint_dir(load_path)
    return checkpoint_dir.resolve(strict=True) if checkpoint_dir is not None else None


def _raise_if_incomplete_direct_distributed_checkpoint(load_path: str | Path) -> None:
    """Reject partial torch_dist directories before legacy detection can claim them."""
    direct_path = Path(load_path)
    try:
        resolved_path = direct_path.resolve(strict=True)
    except OSError:
        return
    if (resolved_path / ".metadata").is_file():
        return

    is_iteration_directory = (
        _ITERATION_DIRECTORY_RE.fullmatch(direct_path.name) is not None
        or _ITERATION_DIRECTORY_RE.fullmatch(resolved_path.name) is not None
    )
    marker_exists = (resolved_path / _ORBIT_TRAINING_CHECKPOINT_MARKER).is_file()
    has_torch_dist_remnants = (resolved_path / "common.pt").is_file() or any(
        resolved_path.glob("*.distcp")
    )
    if (is_iteration_directory or marker_exists) and (marker_exists or has_torch_dist_remnants):
        raise RuntimeError(
            f"incomplete distributed checkpoint at {direct_path}: missing finalized .metadata; "
            "wait for asynchronous checkpoint finalization or choose a completed iteration"
        )


def _read_orbit_training_checkpoint_marker(checkpoint_dir: Path) -> dict | None:
    marker_path = checkpoint_dir / _ORBIT_TRAINING_CHECKPOINT_MARKER
    if not marker_path.is_file():
        return None

    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid Orbit training checkpoint marker at {marker_path}") from exc

    if type(marker) is not dict:
        raise RuntimeError(f"invalid Orbit training checkpoint marker at {marker_path}: expected an object")
    if marker.get("format") != _ORBIT_TRAINING_CHECKPOINT_FORMAT:
        raise RuntimeError(f"invalid Orbit training checkpoint marker format at {marker_path}")
    if marker.get("version") != _ORBIT_TRAINING_CHECKPOINT_VERSION:
        raise RuntimeError(
            f"unsupported Orbit training checkpoint marker version at {marker_path}: {marker.get('version')!r}"
        )
    if marker.get("role") not in _CHECKPOINT_ROLES:
        raise RuntimeError(f"invalid Orbit training checkpoint role at {marker_path}: {marker.get('role')!r}")
    if not _bounded_nonnegative_integer(marker.get("iteration")):
        raise RuntimeError(f"invalid Orbit training checkpoint iteration at {marker_path}")
    for state_field in ("optimizer_state_saved", "scheduler_state_saved"):
        if type(marker.get(state_field)) is not bool:
            raise RuntimeError(f"invalid Orbit training checkpoint {state_field} at {marker_path}")

    iteration_match = _ITERATION_DIRECTORY_RE.fullmatch(checkpoint_dir.name)
    if iteration_match is not None and marker["iteration"] != int(iteration_match.group(1)):
        raise RuntimeError(
            f"Orbit training checkpoint marker iteration mismatch at {marker_path}: marker={marker['iteration']}, directory={checkpoint_dir.name}"
        )
    return marker


def _checkpoint_root(checkpoint_dir: Path) -> Path:
    return checkpoint_dir.parent if _ITERATION_DIRECTORY_RE.fullmatch(checkpoint_dir.name) else checkpoint_dir


def _common_arg(common_state: dict, name: str):
    checkpoint_args = common_state.get("args")
    if isinstance(checkpoint_args, dict):
        return checkpoint_args.get(name)
    return getattr(checkpoint_args, name, None)


def _same_path(left, right: Path) -> bool:
    if left is None:
        return False
    return os.path.abspath(os.path.expanduser(os.fspath(left))) == os.path.abspath(os.fspath(right))


def _infer_legacy_checkpoint_role(common_state: dict, checkpoint_dir: Path) -> str | None:
    checkpoint_root = _checkpoint_root(checkpoint_dir)
    if _same_path(_common_arg(common_state, "critic_save"), checkpoint_root):
        return "critic"
    if _same_path(_common_arg(common_state, "save"), checkpoint_root):
        return "actor"
    return None


def _checkpoint_has_sharded_optimizer_state(checkpoint_dir: Path) -> bool:
    """Inspect torch-dist metadata for optimizer tensors absent from common.pt."""
    try:
        from megatron.core.dist_checkpointing.serialization import load_tensors_metadata

        tensor_metadata = load_tensors_metadata(str(checkpoint_dir))
    except Exception as exc:
        logger.warning("Could not inspect distributed checkpoint tensor metadata at %s: %s", checkpoint_dir, exc)
        return False

    if not isinstance(tensor_metadata, dict):
        return False
    for key in tensor_metadata:
        if not isinstance(key, str):
            continue
        components = key.split(".")
        if any(
            component == "optimizer" and index + 1 < len(components) and components[index + 1] == "state"
            for index, component in enumerate(components)
        ):
            return True
    return False


def _legacy_checkpoint_has_training_state(common_state: dict, checkpoint_dir: Path) -> bool:
    """Recognize only checkpoints capable of restoring both optimizer and schedule.

    Iteration numbers, saved args, and model tensor shapes are deliberately not
    sufficient: converted/base checkpoints and scalar-head critic bootstraps may
    carry all three without being resumable training checkpoints.
    """
    iteration = common_state.get("iteration")
    has_scheduler = common_state.get("opt_param_scheduler") is not None or common_state.get("lr_scheduler") is not None
    if not _bounded_nonnegative_integer(iteration) or not has_scheduler:
        return False
    return common_state.get("optimizer") is not None or _checkpoint_has_sharded_optimizer_state(checkpoint_dir)


def _select_megatron_training_checkpoint(
    args,
    expected_role: str,
    checkpoint_dir: Path,
) -> Path | None:
    if expected_role not in _CHECKPOINT_ROLES:
        raise ValueError(f"invalid expected checkpoint role: {expected_role!r}")

    # Megatron treats release checkpoints as model initialization and resets
    # iteration to zero, so they are never training-resume candidates.
    if checkpoint_dir.name == "release":
        return None

    # These flags request a weights-only warm start. Keep Orbit's model-only
    # loader semantics (iteration zero and fresh optimizer/scheduler) even when
    # the source is an otherwise resumable training checkpoint.
    if getattr(args, "finetune", False) or getattr(args, "no_load_optim", False):
        logger.info(
            "Treating distributed checkpoint at %s as model-only because finetune/no-load-optim is set",
            checkpoint_dir,
        )
        return None

    marker = _read_orbit_training_checkpoint_marker(checkpoint_dir)
    if marker is not None:
        if marker["role"] != expected_role:
            logger.info(
                "Treating Orbit %s checkpoint at %s as model-only for %s bootstrap",
                marker["role"],
                checkpoint_dir,
                expected_role,
            )
            return None
        if not (marker["optimizer_state_saved"] and marker["scheduler_state_saved"]):
            raise RuntimeError(
                f"Orbit {expected_role} checkpoint at {checkpoint_dir} was saved without complete optimizer/scheduler state and cannot resume training. Use --no-load-optim for an explicit model-only warm start, or resume from a checkpoint saved without --no-save-optim."
            )
        return checkpoint_dir

    common_state_path = checkpoint_dir / "common.pt"
    if not common_state_path.is_file():
        return None
    try:
        common_state = torch.load(common_state_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        logger.warning("Could not inspect legacy distributed checkpoint state at %s: %s", common_state_path, exc)
        return None
    if type(common_state) is not dict or not _legacy_checkpoint_has_training_state(common_state, checkpoint_dir):
        return None

    inferred_role = _infer_legacy_checkpoint_role(common_state, checkpoint_dir)
    if inferred_role is not None and inferred_role != expected_role:
        logger.info(
            "Treating legacy Orbit %s checkpoint at %s as model-only for %s bootstrap",
            inferred_role,
            checkpoint_dir,
            expected_role,
        )
        return None
    if inferred_role is None and expected_role == "critic":
        logger.warning(
            "Legacy distributed checkpoint at %s has training state but no defensible role metadata; using model-only critic bootstrap. Re-save it with an Orbit checkpoint marker to enable critic resume.",
            checkpoint_dir,
        )
        return None

    logger.warning(
        "Resuming %s from legacy distributed checkpoint training state at %s; future saves will include an explicit Orbit marker.",
        expected_role,
        checkpoint_dir,
    )
    return checkpoint_dir


def _load_selected_megatron_training_checkpoint(args, checkpoint_dir: Path, **load_kwargs):
    """Pin Megatron's full loader to the checkpoint that was classified.

    Pinning closes a race with async saves updating the root tracker between
    classification and load. It also makes direct ``iter_*`` paths work with
    Megatron's root-oriented loader. The truthy int subclass works around
    upstream's ``if args.ckpt_step`` handling for an explicit step of zero.
    """
    checkpoint_dir = checkpoint_dir.resolve(strict=True)
    iteration_match = _ITERATION_DIRECTORY_RE.fullmatch(checkpoint_dir.name)
    if iteration_match is not None:
        iteration = int(iteration_match.group(1))
    else:
        marker = _read_orbit_training_checkpoint_marker(checkpoint_dir)
        if marker is None:
            raise RuntimeError(
                f"Orbit training checkpoint iteration cannot be derived from directory or marker at {checkpoint_dir}"
            )
        iteration = marker["iteration"]

    original_load = args.load
    missing = object()
    original_checkpoint_step = getattr(args, "ckpt_step", missing)
    with tempfile.TemporaryDirectory(prefix="orbit-megatron-load-") as temporary_root:
        temporary_root_path = Path(temporary_root)
        selected_name = f"iter_{iteration:07d}"
        os.symlink(checkpoint_dir, temporary_root_path / selected_name, target_is_directory=True)
        (temporary_root_path / _MEGATRON_TRACKER_FILE).write_text(f"{iteration}\n")
        args.load = temporary_root
        args.ckpt_step = _ExplicitZeroCheckpointStep() if iteration == 0 else iteration
        try:
            return _load_checkpoint_megatron(**load_kwargs)
        finally:
            args.load = original_load
            if original_checkpoint_step is missing:
                delattr(args, "ckpt_step")
            else:
                args.ckpt_step = original_checkpoint_step


def _optimizer_scheduler_state_was_restored(
    args,
    optimizer,
    opt_param_scheduler,
    *,
    skip_load_to_model_and_opt: bool,
) -> bool:
    return (
        not getattr(args, "finetune", False)
        and not getattr(args, "no_load_optim", False)
        and not skip_load_to_model_and_opt
        and optimizer is not None
        and not getattr(optimizer, "is_stub_optimizer", False)
        and opt_param_scheduler is not None
    )


def _selected_legacy_megatron_checkpoint(args) -> tuple[int | None, Path | None]:
    """Return the explicitly selected numeric checkpoint and its directory.

    Megatron uses the literal tracker value ``release`` for model bootstrap and
    decimal text (including ``0``) for training checkpoints.  Direct iteration
    paths and ``--ckpt-step`` carry the same distinction without consulting the
    root tracker.
    """
    load_path = Path(args.load).expanduser().resolve(strict=True)
    iteration_match = _ITERATION_DIRECTORY_RE.fullmatch(load_path.name)
    checkpoint_step = getattr(args, "ckpt_step", None)

    if iteration_match is not None:
        selected_iteration = int(iteration_match.group(1))
        if checkpoint_step is not None and checkpoint_step != selected_iteration:
            return None, None
        return selected_iteration, load_path

    if load_path.name == "release":
        return None, None

    if checkpoint_step is not None:
        if not _bounded_nonnegative_integer(checkpoint_step):
            return None, None
        return checkpoint_step, _checkpoint_iteration_dir(load_path, checkpoint_step)

    try:
        tracker_value = (load_path / _MEGATRON_TRACKER_FILE).read_text().strip()
    except OSError:
        return None, None
    if tracker_value == "release" or not tracker_value.isdigit():
        return None, None

    selected_iteration = int(tracker_value)
    if not _bounded_nonnegative_integer(selected_iteration):
        return None, None
    return selected_iteration, _checkpoint_iteration_dir(load_path, selected_iteration)


def _legacy_megatron_load_restored_training_iteration(
    args,
    result,
    *,
    load_training_state: bool,
    expected_role: str | None,
) -> bool:
    """Exclude release/finetune/model-only legacy loads from resume orchestration."""
    if not load_training_state or getattr(args, "finetune", False) or getattr(args, "no_load_optim", False):
        return False
    if not isinstance(result, tuple) or not result:
        return False
    iteration = result[0]
    if not _bounded_nonnegative_integer(iteration):
        return False

    selected_iteration, checkpoint_dir = _selected_legacy_megatron_checkpoint(args)
    if selected_iteration != iteration or checkpoint_dir is None:
        return False

    # New Orbit saves carry stronger provenance than the legacy numeric
    # tracker.  When present, require the marker to describe a complete
    # training checkpoint for the model role being restored.
    marker = _read_orbit_training_checkpoint_marker(checkpoint_dir)
    if marker is not None:
        if marker["iteration"] != iteration:
            raise RuntimeError(
                f"Orbit training checkpoint iteration mismatch at {checkpoint_dir}: "
                f"marker={marker['iteration']}, loaded={iteration}"
            )
        if expected_role is not None and marker["role"] != expected_role:
            return False
        if not (marker["optimizer_state_saved"] and marker["scheduler_state_saved"]):
            return False

    return True


def load_checkpoint(
    ddp_model,
    optimizer,
    opt_param_scheduler,
    checkpointing_context,
    skip_load_to_model_and_opt,
    *,
    is_value_model: bool = False,
    load_training_state: bool = False,
):
    """Load a model source or, when explicitly requested and identified, a full training checkpoint.

    ``load_training_state`` is opt-in because this function also loads reference
    policies and base/converted checkpoints. Distributed training checkpoints use
    Megatron's native loader so optimizer, scheduler, RNG, and iteration state are
    restored together; model-only sources retain Orbit's flexible custom loader.
    """
    # ref: how megatron `load_checkpoint` gets directory
    args = get_args()
    if getattr(args, "megatron_to_hf_mode", None) == "bridge":
        validate_low_precision_bootstrap_args(args)
    load_path = args.load
    # Orchestration distinguishes a real training resume from a base-model
    # bootstrap. The second flag is deliberately narrower: initialization uses
    # it to avoid advancing a scheduler Megatron has already restored.
    args._orbit_training_checkpoint_loaded = False
    args._orbit_optimizer_scheduler_state_restored = False

    assert Path(load_path).exists() and _is_dir_nonempty(load_path), (
        f"{args.load=} does not exist or is an empty directory. Did you specify the wrong folder?"
    )
    _raise_if_incomplete_direct_distributed_checkpoint(load_path)

    if is_distributed_checkpoint(load_path):
        selected_checkpoint_dir = _resolve_selected_distributed_checkpoint(args)
        if selected_checkpoint_dir is None:
            raise RuntimeError(f"could not resolve distributed checkpoint at {load_path}")
        expected_role = _model_checkpoint_role(ddp_model) if load_training_state else None
        training_checkpoint_dir = (
            _select_megatron_training_checkpoint(args, expected_role, selected_checkpoint_dir)
            if expected_role is not None
            else None
        )
        if training_checkpoint_dir is not None:
            logger.info(
                "Load Orbit %s training checkpoint through Megatron (path=%s)",
                expected_role,
                training_checkpoint_dir,
            )
            result = _load_selected_megatron_training_checkpoint(
                args,
                training_checkpoint_dir,
                ddp_model=ddp_model,
                optimizer=optimizer,
                opt_param_scheduler=opt_param_scheduler,
                checkpointing_context=checkpointing_context,
                skip_load_to_model_and_opt=skip_load_to_model_and_opt,
            )
            args._orbit_training_checkpoint_loaded = True
            args._orbit_optimizer_scheduler_state_restored = _optimizer_scheduler_state_was_restored(
                args,
                optimizer,
                opt_param_scheduler,
                skip_load_to_model_and_opt=skip_load_to_model_and_opt,
            )
        else:
            result = _load_checkpoint_dist(
                ddp_model=ddp_model,
                optimizer=optimizer,
                args=args,
                # Pass the selected per-iteration directory so model-only
                # loads also honor --ckpt-step and cannot race tracker updates.
                load_path=str(selected_checkpoint_dir),
                is_value_model=is_value_model,
            )
    elif _is_megatron_checkpoint(load_path):
        result = _load_checkpoint_megatron(
            ddp_model=ddp_model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=skip_load_to_model_and_opt,
        )
        if load_training_state and getattr(args, "no_load_optim", False):
            # Match distributed model-only warm starts: do not advance rollout
            # IDs or a fresh scheduler from the checkpoint's saved iteration.
            result = (0, result[1])
        args._orbit_training_checkpoint_loaded = _legacy_megatron_load_restored_training_iteration(
            args,
            result,
            load_training_state=load_training_state,
            expected_role=_model_checkpoint_role(ddp_model),
        )
        if args._orbit_training_checkpoint_loaded:
            args._orbit_optimizer_scheduler_state_restored = _optimizer_scheduler_state_was_restored(
                args,
                optimizer,
                opt_param_scheduler,
                skip_load_to_model_and_opt=skip_load_to_model_and_opt,
            )
    else:
        result = _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )

    # Keep adapter tensor loading distinct from training-state restoration. A
    # native adapter can intentionally be weights-only (for example after
    # --no-save-optim), in which case training starts with a fresh optimizer.
    args._peft_adapter_weights_loaded = False
    args._peft_training_state_found = False
    args._peft_checkpoint_preflight = None

    # Load PEFT adapter weights if available
    if is_peft_enabled(args) and is_peft_model(ddp_model):
        adapter_path = (
            getattr(args, "peft_adapter_path", None)
            or getattr(args, "lora_adapter_path", None)
            or getattr(args, "oft_adapter_path", None)
        )
        if adapter_path is not None:
            checkpoint_preflight = preflight_peft_adapter_checkpoint(adapter_path)
            loaded, iteration = load_peft_adapter(
                ddp_model,
                args,
                adapter_path,
                optimizer=optimizer,
                opt_param_scheduler=opt_param_scheduler,
                checkpoint_preflight=checkpoint_preflight,
            )
            if loaded:
                logger.info(f"Successfully loaded PEFT adapter from {adapter_path}")
                args._peft_adapter_weights_loaded = True
                args._peft_training_state_found = checkpoint_preflight.training_state_present
                args._peft_checkpoint_preflight = checkpoint_preflight
                # Self-teacher sidecars (and future pool bindings) live beside the
                # adapter; the actor's restore hook reads this after teacher init.
                args._peft_resume_adapter_dir = str(adapter_path)
                if not checkpoint_preflight.training_state_present and optimizer is not None and (
                    getattr(args, "fp16", False) or getattr(args, "bf16", False)
                ):
                    # Adapter tensors were copied after mixed-precision optimizer
                    # main parameters were constructed (and, for base checkpoints,
                    # refreshed). Keep those optimizer-owned FP32 parameters in
                    # sync without overwriting a real optimizer-sidecar restore.
                    reload_model_params = getattr(optimizer, "reload_model_params", None)
                    if not callable(reload_model_params):
                        raise RuntimeError(
                            "mixed-precision PEFT optimizer cannot synchronize weights-only adapter parameters: "
                            "reload_model_params() is unavailable"
                        )
                    reload_model_params()
                if iteration is not None:
                    result = (iteration, result[1])
                    args._orbit_training_checkpoint_loaded = True
                    # High-precision PEFT resume restores this state in
                    # load_peft_adapter itself. Low-precision two-phase resume
                    # has no optimizer here and remains false until its second phase.
                    if optimizer is not None and opt_param_scheduler is not None:
                        args._orbit_optimizer_scheduler_state_restored = True
            else:
                logger.warning(
                    f"PEFT is enabled and adapter_path={adapter_path} was specified, but adapter weights could not be loaded. Training will start with freshly initialized adapter weights."
                )

    return result


def save_checkpoint_with_peft(
    iteration,
    model,
    optimizer,
    opt_param_scheduler,
    *,
    self_teacher=None,
):
    """Extended save that handles PEFT adapters separately."""
    args = get_args()

    if is_peft_model(model):
        save_dir = Path(args.save) / f"iter_{iteration:07d}" / "adapter"
        logger.info(f"Saving PEFT checkpoint to {save_dir}")
        save_peft_checkpoint(
            model,
            args,
            str(save_dir),
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            iteration=iteration,
            self_teacher=self_teacher,
        )
    else:
        save_checkpoint(iteration, model, optimizer, opt_param_scheduler)


def _is_megatron_checkpoint(path: str | Path) -> bool:
    return is_legacy_megatron_checkpoint(path)


def _load_checkpoint_dist(ddp_model, optimizer, args, load_path: str, *, is_value_model: bool = False):
    logger.info("Load checkpoint from Megatron distributed checkpoint (path=%s)", load_path)

    load_dist_checkpoint(ddp_model, load_path, is_value_model=is_value_model)

    if (args.fp16 or args.bf16) and optimizer is not None:
        assert not args.load_main_params_from_ckpt
        optimizer.reload_model_params()

    iteration = 0
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far


def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"
    from megatron.bridge import AutoBridge

    logger.info(f"Load checkpoint from HuggingFace model into Megatron (path={load_path})")

    with megatron_bridge_utils.patch_megatron_model(ddp_model):
        bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
        bridge.load_hf_weights(ddp_model)

    # Copied from Megatron-core :: load_checkpoint (with simplifications)
    if (args.fp16 or args.bf16) and optimizer is not None:
        assert not args.load_main_params_from_ckpt
        optimizer.reload_model_params()

    # We can see `successfully loaded checkpoint from ... [ t 1/2, p 1/1 ] at iteration 0`
    # when loading Megatron, thus it is 0
    iteration = 0
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far


def _is_dir_nonempty(path):
    with os.scandir(path) as it:
        return any(it)
