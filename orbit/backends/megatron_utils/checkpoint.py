import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.distributed as dist

# Follow-up: may need to copy those 2 functions and do refactoring.
from megatron.training.checkpointing import get_checkpoint_name as _get_megatron_checkpoint_name
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
from megatron.training.checkpointing import save_checkpoint as _save_checkpoint_megatron
from megatron.training.global_vars import get_args

from orbit.utils import distributed_utils, megatron_bridge_utils

from .lora_utils import is_lora_enabled, is_lora_model, load_lora_adapter, save_lora_checkpoint
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

__all__ = [
    "save_checkpoint",
    "save_checkpoint_with_peft",
    "save_checkpoint_with_lora",
    "load_checkpoint",
    "resolve_start_rollout_id_after_load",
]

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


@dataclass(frozen=True)
class _LegacyMegatronCheckpointPreflight:
    selected_iteration: int | None
    checkpoint_dir: Path | None
    marker: dict | None
    numeric_zero_state_present: bool = False
    requires_numeric_zero_restore_proof: bool = False
    force_model_only: bool = False


class _RestoreMethodObserver:
    """Observe successful state restores without replacing the wrapped object.

    Megatron inspects optimizer attributes and some extensions may inspect its
    exact type. Patching bound methods on the instance preserves both identity
    and type, unlike a forwarding proxy.
    """

    def __init__(self, target, method_names: tuple[str, ...]):
        self.target = target
        self.method_names = method_names
        self.restored_methods: set[str] = set()
        self._original_instance_attributes: dict[str, tuple[bool, object]] = {}

    def install(self) -> bool:
        if self.target is None:
            return False
        instance_attributes = getattr(self.target, "__dict__", None)
        if not isinstance(instance_attributes, dict):
            return False

        try:
            for method_name in self.method_names:
                method = getattr(self.target, method_name, None)
                if not callable(method):
                    continue
                had_instance_attribute = method_name in instance_attributes
                original_instance_attribute = instance_attributes.get(method_name)

                def observed_method(*args, __method=method, __method_name=method_name, **kwargs):
                    result = __method(*args, **kwargs)
                    self.restored_methods.add(__method_name)
                    return result

                setattr(self.target, method_name, observed_method)
                self._original_instance_attributes[method_name] = (
                    had_instance_attribute,
                    original_instance_attribute,
                )
        except Exception:
            self.restore()
            return False
        return bool(self._original_instance_attributes)

    def restore(self) -> None:
        for method_name, (had_instance_attribute, original_instance_attribute) in reversed(
            tuple(self._original_instance_attributes.items())
        ):
            if had_instance_attribute:
                setattr(self.target, method_name, original_instance_attribute)
            else:
                try:
                    delattr(self.target, method_name)
                except AttributeError:
                    pass
        self._original_instance_attributes.clear()

    @property
    def restored(self) -> bool:
        return bool(self.restored_methods)


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
    has_torch_dist_remnants = (resolved_path / "common.pt").is_file() or any(resolved_path.glob("*.distcp"))
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
    """Return the explicitly selected checkpoint iteration and directory.

    Megatron uses the literal tracker value ``release`` for model bootstrap and
    decimal text (including ``0``) for training checkpoints.  Direct iteration
    paths and ``--ckpt-step`` carry the same distinction without consulting
    the root tracker. Release selections return ``(None, release_directory)``
    so their marker can still be validated before Megatron mutates the model.
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
        return None, load_path

    if checkpoint_step is not None:
        if not _bounded_nonnegative_integer(checkpoint_step):
            return None, None
        return checkpoint_step, _checkpoint_iteration_dir(load_path, checkpoint_step)

    try:
        tracker_value = (load_path / _MEGATRON_TRACKER_FILE).read_text().strip()
    except OSError:
        return None, None
    if tracker_value == "release":
        return None, load_path / "release"
    if not tracker_value.isdigit():
        return None, None

    selected_iteration = int(tracker_value)
    if not _bounded_nonnegative_integer(selected_iteration):
        return None, None
    return selected_iteration, _checkpoint_iteration_dir(load_path, selected_iteration)


def _legacy_checkpoint_consensus_group():
    """Use Orbit's world Gloo group, or the default group in Gloo-only tests."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return None
    if distributed_utils.GLOO_GROUP is not None:
        return distributed_utils.GLOO_GROUP
    if str(dist.get_backend()).lower().endswith("gloo"):
        return None
    raise RuntimeError("legacy Megatron checkpoint preflight requires Orbit's world Gloo process group")


def _all_gather_legacy_checkpoint_object(value) -> list:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [value]
    group = _legacy_checkpoint_consensus_group()
    gathered = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, value, group=group)
    return gathered


def _coordinated_legacy_checkpoint_call(label: str, fn):
    value = None
    local_error = None
    try:
        value = fn()
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    errors = _all_gather_legacy_checkpoint_object(local_error)
    failures = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error is not None]
    if failures:
        raise RuntimeError(f"{label} failed on one or more ranks; " + "; ".join(failures))
    return value


def _legacy_megatron_rank_shard(checkpoint_dir: Path, iteration: int) -> Path | None:
    """Resolve the legacy torch shard that Megatron will read on this rank."""
    checkpoint_root = _checkpoint_root(checkpoint_dir)
    if dist.is_initialized():
        candidate = Path(_get_megatron_checkpoint_name(str(checkpoint_root), iteration, release=False))
        return candidate if candidate.is_file() else None

    # Unit tests and single-process inspection run before model-parallel groups
    # exist. There is only one relevant rank shard in that setting.
    candidates = sorted(checkpoint_dir.glob("mp_rank_*/model_optim_rng.pt"))
    return candidates[0] if candidates else None


def _legacy_rank_shard_has_training_state(checkpoint_dir: Path, iteration: int) -> bool:
    """Inspect legacy checkpoint keys without allocating tensor storages."""
    rank_shard = _legacy_megatron_rank_shard(checkpoint_dir, iteration)
    if rank_shard is None:
        logger.warning("Legacy checkpoint %s has no rank shard available for training-state preflight", checkpoint_dir)
        return False

    try:
        from torch._subclasses.fake_tensor import FakeTensorMode

        with FakeTensorMode():
            state_dict = torch.load(rank_shard, map_location="cpu", weights_only=False)
    except Exception as exc:
        logger.warning("Could not inspect legacy checkpoint training state at %s: %s", rank_shard, exc)
        return False

    if type(state_dict) is not dict:
        return False
    checkpoint_iteration = state_dict.get("iteration", state_dict.get("total_iters"))
    scheduler_state = state_dict.get("lr_scheduler", state_dict.get("opt_param_scheduler"))
    return (
        checkpoint_iteration == iteration and state_dict.get("optimizer") is not None and scheduler_state is not None
    )


def _local_legacy_megatron_preflight(
    args,
    *,
    load_training_state: bool,
    expected_role: str | None,
    optimizer,
    opt_param_scheduler,
    skip_load_to_model_and_opt: bool,
) -> _LegacyMegatronCheckpointPreflight:
    selected_iteration, checkpoint_dir = _selected_legacy_megatron_checkpoint(args)
    marker = _read_orbit_training_checkpoint_marker(checkpoint_dir) if checkpoint_dir is not None else None

    if marker is not None and selected_iteration is not None and marker["iteration"] != selected_iteration:
        raise RuntimeError(
            f"Orbit training checkpoint iteration mismatch at {checkpoint_dir}: "
            f"marker={marker['iteration']}, selected={selected_iteration}"
        )

    explicit_model_only = getattr(args, "finetune", False) or getattr(args, "no_load_optim", False)
    if marker is not None and selected_iteration is not None and load_training_state:
        if expected_role is not None and marker["role"] != expected_role:
            raise RuntimeError(
                f"Orbit checkpoint role mismatch at {checkpoint_dir}: "
                f"expected {expected_role}, found {marker['role']}"
            )
        if not explicit_model_only and not (marker["optimizer_state_saved"] and marker["scheduler_state_saved"]):
            raise RuntimeError(
                f"Orbit {marker['role']} checkpoint at {checkpoint_dir} was saved without complete "
                "optimizer/scheduler state and cannot resume training. Use --no-load-optim for an explicit "
                "model-only warm start, or resume from a checkpoint saved without --no-save-optim."
            )

    restore_requested = (
        load_training_state
        and not explicit_model_only
        and not skip_load_to_model_and_opt
        and optimizer is not None
        and not getattr(optimizer, "is_stub_optimizer", False)
        and opt_param_scheduler is not None
    )
    requires_numeric_zero_restore_proof = selected_iteration == 0 and marker is None and load_training_state
    numeric_zero_state_present = False
    if requires_numeric_zero_restore_proof and checkpoint_dir is not None:
        numeric_zero_state_present = _legacy_rank_shard_has_training_state(checkpoint_dir, selected_iteration)

    return _LegacyMegatronCheckpointPreflight(
        selected_iteration=selected_iteration,
        checkpoint_dir=checkpoint_dir,
        marker=marker,
        numeric_zero_state_present=numeric_zero_state_present,
        requires_numeric_zero_restore_proof=requires_numeric_zero_restore_proof,
        force_model_only=requires_numeric_zero_restore_proof
        and (not restore_requested or not numeric_zero_state_present),
    )


def _preflight_legacy_megatron_checkpoint(
    args,
    *,
    load_training_state: bool,
    expected_role: str | None,
    optimizer,
    opt_param_scheduler,
    skip_load_to_model_and_opt: bool,
) -> _LegacyMegatronCheckpointPreflight:
    preflight = _coordinated_legacy_checkpoint_call(
        "legacy Megatron checkpoint preflight",
        lambda: _local_legacy_megatron_preflight(
            args,
            load_training_state=load_training_state,
            expected_role=expected_role,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            skip_load_to_model_and_opt=skip_load_to_model_and_opt,
        ),
    )
    marker_summary = tuple(sorted(preflight.marker.items())) if preflight.marker is not None else None
    selection_summary = (
        preflight.selected_iteration,
        str(preflight.checkpoint_dir) if preflight.checkpoint_dir is not None else None,
        marker_summary,
        preflight.requires_numeric_zero_restore_proof,
    )
    selections = _all_gather_legacy_checkpoint_object(selection_summary)
    if len(set(selections)) != 1:
        raise RuntimeError(f"legacy Megatron checkpoint selection differs across ranks: {selections}")

    state_presence = _all_gather_legacy_checkpoint_object(preflight.numeric_zero_state_present)
    globally_present = all(state_presence)
    force_model_only_by_rank = _all_gather_legacy_checkpoint_object(preflight.force_model_only)
    force_model_only = any(force_model_only_by_rank) or (
        preflight.requires_numeric_zero_restore_proof and not globally_present
    )
    return replace(
        preflight,
        numeric_zero_state_present=globally_present,
        force_model_only=force_model_only,
    )


def _load_preflighted_legacy_megatron_checkpoint(
    args,
    preflight: _LegacyMegatronCheckpointPreflight,
    *,
    ddp_model,
    optimizer,
    opt_param_scheduler,
    checkpointing_context,
    skip_load_to_model_and_opt: bool,
):
    """Load a legacy checkpoint and return whether numeric-zero state was proven."""
    force_model_only = preflight.force_model_only
    optimizer_observer = None
    scheduler_observer = None

    if preflight.requires_numeric_zero_restore_proof and not force_model_only:
        optimizer_observer = _RestoreMethodObserver(
            optimizer,
            ("load_state_dict", "load_state_dict_from_file"),
        )
        scheduler_observer = _RestoreMethodObserver(opt_param_scheduler, ("load_state_dict",))
        local_capability = optimizer_observer.install() and scheduler_observer.install()
        try:
            capabilities = _all_gather_legacy_checkpoint_object(local_capability)
        except Exception:
            optimizer_observer.restore()
            scheduler_observer.restore()
            raise
        if not all(capabilities):
            force_model_only = True
            optimizer_observer.restore()
            scheduler_observer.restore()
            logger.warning(
                "Legacy numeric-zero checkpoint restore methods cannot be observed on every rank; "
                "using a model-only bootstrap"
            )

    missing = object()
    original_no_load_optim = getattr(args, "no_load_optim", missing)
    if force_model_only:
        args.no_load_optim = True
        logger.warning(
            "Legacy numeric-zero checkpoint at %s lacks globally proven optimizer/scheduler state; "
            "using a model-only bootstrap",
            preflight.checkpoint_dir,
        )

    try:
        result = _load_checkpoint_megatron(
            ddp_model=ddp_model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=skip_load_to_model_and_opt,
        )
    finally:
        if force_model_only:
            if original_no_load_optim is missing:
                delattr(args, "no_load_optim")
            else:
                args.no_load_optim = original_no_load_optim
        if optimizer_observer is not None:
            optimizer_observer.restore()
        if scheduler_observer is not None:
            scheduler_observer.restore()

    numeric_zero_restore_proven = False
    if preflight.requires_numeric_zero_restore_proof and not force_model_only:
        local_restore = (optimizer_observer.restored, scheduler_observer.restored)
        restores = _all_gather_legacy_checkpoint_object(local_restore)
        if not all(optimizer_restored and scheduler_restored for optimizer_restored, scheduler_restored in restores):
            raise RuntimeError(
                "legacy numeric-zero checkpoint loader did not prove optimizer and scheduler restoration on every rank"
            )
        numeric_zero_restore_proven = True

    if force_model_only and isinstance(result, tuple) and result:
        result = (0, *result[1:])
    return result, numeric_zero_restore_proven


def _legacy_megatron_load_restored_training_iteration(
    args,
    result,
    *,
    load_training_state: bool,
    preflight: _LegacyMegatronCheckpointPreflight,
    numeric_zero_restore_proven: bool,
) -> bool:
    """Exclude release/finetune/model-only legacy loads from resume orchestration."""
    if not load_training_state or getattr(args, "finetune", False) or getattr(args, "no_load_optim", False):
        return False
    if not isinstance(result, tuple) or not result:
        return False
    iteration = result[0]
    if not _bounded_nonnegative_integer(iteration):
        return False

    if preflight.selected_iteration != iteration or preflight.checkpoint_dir is None:
        return False

    # Markers were validated before Megatron could mutate runtime state. An
    # unmarked zero is ambiguous with a model bootstrap, so it additionally
    # requires observed optimizer and scheduler restore calls on every rank.
    if iteration == 0 and preflight.marker is None:
        return numeric_zero_restore_proven

    return True


def load_checkpoint_orbit(
    ddp_model,
    optimizer,
    opt_param_scheduler,
    checkpointing_context,
    skip_load_to_model_and_opt,
    *,
    is_value_model: bool = False,
    load_training_state: bool = False,
):
    """Orbit loader: distributed/legacy/low-precision aware. Not yet wired
    into orbit' actor (follow-up); orbit' primary loader is load_checkpoint below.

    Load a model source or, when explicitly requested and identified, a full training checkpoint.

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

    has_local_checkpoint_manager = "local_checkpoint_manager" in (checkpointing_context or {})
    if has_local_checkpoint_manager:
        logger.info("Skipping disk path validation: using in-memory checkpoint via local_checkpoint_manager")
    else:
        assert Path(load_path).exists() and _is_dir_nonempty(
            load_path
        ), f"{args.load=} does not exist or is an empty directory. Did you specify the wrong folder?"
        _raise_if_incomplete_direct_distributed_checkpoint(load_path)

    # orbit: a bridge-mode HF or release-checkpoint load is a weights-only
    # bootstrap; start_rollout_id defaults to 0 for it (see end of function).
    is_bridge_bootstrap = (
        getattr(args, "megatron_to_hf_mode", None) == "bridge"
        and not has_local_checkpoint_manager
        and not is_distributed_checkpoint(load_path)
        and (not _is_megatron_checkpoint(load_path) or _is_release_checkpoint(load_path))
    )

    if has_local_checkpoint_manager:
        result = _load_checkpoint_megatron(
            ddp_model=ddp_model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=skip_load_to_model_and_opt,
        )
    elif is_distributed_checkpoint(load_path):
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
        expected_role = _model_checkpoint_role(ddp_model)
        legacy_preflight = _preflight_legacy_megatron_checkpoint(
            args,
            load_training_state=load_training_state,
            expected_role=expected_role,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            skip_load_to_model_and_opt=skip_load_to_model_and_opt,
        )
        result, numeric_zero_restore_proven = _load_preflighted_legacy_megatron_checkpoint(
            args,
            legacy_preflight,
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
            preflight=legacy_preflight,
            numeric_zero_restore_proven=numeric_zero_restore_proven,
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
                if (
                    not checkpoint_preflight.training_state_present
                    and optimizer is not None
                    and (getattr(args, "fp16", False) or getattr(args, "bf16", False))
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

    if (
        getattr(args, "start_rollout_id", None) is None
        and is_bridge_bootstrap
        and not getattr(args, "_peft_training_state_found", False)
    ):
        args.start_rollout_id = 0

    return result


def save_checkpoint_with_peft(
    iteration,
    model,
    optimizer,
    opt_param_scheduler,
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


def _is_release_checkpoint(path: str | Path) -> bool:
    tracker = Path(path) / "latest_checkpointed_iteration.txt"
    try:
        return tracker.read_text().strip() == "release"
    except FileNotFoundError:
        return False


def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt):
    # ref: how megatron `load_checkpoint` gets directory
    args = get_args()

    load_path = args.load

    has_local_checkpoint_manager = "local_checkpoint_manager" in (checkpointing_context or {})
    if has_local_checkpoint_manager:
        logger.info("Skipping disk path validation: using in-memory checkpoint via local_checkpoint_manager")
    else:
        assert Path(load_path).exists() and _is_dir_nonempty(
            load_path
        ), f"{args.load=} does not exist or is an empty directory. Did you specify the wrong folder?"

    is_megatron_checkpoint = not has_local_checkpoint_manager and _is_megatron_checkpoint(load_path)
    is_bridge_bootstrap = (
        getattr(args, "megatron_to_hf_mode", None) == "bridge"
        and not has_local_checkpoint_manager
        and (not is_megatron_checkpoint or _is_release_checkpoint(load_path))
    )

    if has_local_checkpoint_manager or is_megatron_checkpoint:
        result = _load_checkpoint_megatron(
            ddp_model=ddp_model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=skip_load_to_model_and_opt,
        )
    else:
        result = _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )

    # Load LoRA adapter weights if available. Only a restored training-state
    # iteration turns an HF/release bootstrap into a resume.
    adapter_iteration = None
    if is_lora_enabled(args):
        adapter_path = getattr(args, "lora_adapter_path", None)
        if adapter_path is not None:
            loaded, iteration = load_lora_adapter(
                ddp_model,
                adapter_path,
                optimizer=optimizer,
                opt_param_scheduler=opt_param_scheduler,
            )
            if loaded:
                logger.info(f"Successfully loaded LoRA adapter from {adapter_path}")
                if iteration is not None:
                    adapter_iteration = iteration
                    result = (iteration, result[1])
            else:
                logger.warning(
                    f"LoRA is enabled and --lora-adapter-path={adapter_path} was specified, "
                    f"but adapter weights could not be loaded. "
                    f"Training will start with freshly initialized adapter weights."
                )

    # OFT (orbit port): adapters resume through the PEFT loader; LoRA keeps the
    # orbit path above so its contract (and tests patching it) stay intact.
    if (
        getattr(args, "peft_method", "none") == "oft"
        and adapter_iteration is None
        and getattr(args, "oft_adapter_path", None)
    ):
        loaded, iteration = load_peft_adapter(
            ddp_model,
            args.oft_adapter_path,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
        )
        if loaded and iteration is not None:
            result = (iteration, result[1])

    if getattr(args, "start_rollout_id", None) is None and is_bridge_bootstrap and adapter_iteration is None:
        args.start_rollout_id = 0

    return result


def resolve_start_rollout_id_after_load(args, loaded_iteration: int) -> int:
    """Prefer the start ID resolved from the actual load over iteration + 1."""
    start_rollout_id = getattr(args, "start_rollout_id", None)
    return loaded_iteration + 1 if start_rollout_id is None else start_rollout_id


def save_checkpoint_with_lora(iteration, model, optimizer, opt_param_scheduler):
    """Extended save that handles LoRA adapters separately (orbit contract).

    Kept alongside save_checkpoint_with_peft: tests and legacy callers patch
    ``is_lora_model`` / ``save_lora_checkpoint`` on this module and expect this
    exact routing.
    """
    args = get_args()

    if is_lora_model(model):
        save_dir = Path(args.save) / f"iter_{iteration:07d}" / "adapter"
        logger.info(f"Saving LoRA checkpoint to {save_dir}")
        save_lora_checkpoint(
            model,
            args,
            str(save_dir),
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            iteration=iteration,
        )
    else:
        save_checkpoint(iteration, model, optimizer, opt_param_scheduler)
