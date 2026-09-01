import json
import logging
import os
import stat
import tempfile
from argparse import Namespace
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
from megatron.core import mpu
from safetensors.torch import save_file as safetensors_save_file

from miles.backends.megatron_utils.update_weight.common import is_dsv4_grouped_moe_oft_param_name
from orbit.utils.adapter_tensors import AdapterTensorKey, adapter_named_parameters, adapter_tensor_key_digest

logger = logging.getLogger(__name__)


LORA_SYNC_TRANSPORT = "lora_adapter"
OFT_SYNC_TRANSPORT = "oft_adapter"
_OPTIMIZER_PARAMETER_STATE_PREFIX = "optimizer_parameter_state_rank"
_EMBEDDED_DISTRIBUTED_PARAMETER_STATE_KEYS = frozenset(
    {
        # Pinned Megatron's DistributedOptimizer.load_state_dict() treats
        # these as an instruction to enter its parameter-state loaders. Older
        # pinned checkpoints may contain only ``param_state``.
        "param_state",
        "param_state_sharding_type",
    }
)
_MAX_CHECKPOINT_COUNTER = 2**63 - 1

Variant = Literal["standard", "canonical", "mla", "dsv4"]


def _is_bounded_nonnegative_integer(value: object) -> bool:
    return type(value) is int and 0 <= value <= _MAX_CHECKPOINT_COUNTER


def _is_canonical_student_version(value: object) -> bool:
    if (
        type(value) is not str
        or not value.isascii()
        or not value.isdecimal()
        or len(value) > 19
        or (len(value) > 1 and value.startswith("0"))
    ):
        return False
    return int(value) <= _MAX_CHECKPOINT_COUNTER


def _contains_tensor(value: Any) -> bool:
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, torch.Tensor):
            return True
        if type(item) is dict:
            pending.extend(item.values())
        elif type(item) in (list, tuple):
            pending.extend(item)
    return False


def _contains_inline_optimizer_tensor(value: Any) -> bool:
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is dict:
            if "state" in item and _contains_tensor(item["state"]):
                return True
            if "optimizer" in item:
                pending.append(item["optimizer"])
            if "fp32_from_fp16_params" in item and _contains_tensor(item["fp32_from_fp16_params"]):
                return True
        elif type(item) in (list, tuple):
            pending.extend(item)
    return False


def _contains_megatron_optimizer_wrapper(value: Any) -> bool:
    pending = [value]
    while pending:
        item = pending.pop()
        if type(item) is dict:
            if "optimizer" in item:
                return True
            pending.extend(item.values())
        elif type(item) in (list, tuple):
            pending.extend(item)
    return False


def _validate_no_embedded_distributed_parameter_state(optimizer_state: Any) -> None:
    """Reject parameter state that bypasses Orbit's external-state preflight.

    Megatron's distributed optimizer recognizes these keys at any nested
    chained-optimizer leaf and may enter rank-dependent collectives from
    ``load_state_dict``. PEFT checkpoints keep that state in separately
    validated rank-local files, so its presence in the optimizer payload is
    always incompatible.
    """
    pending = [optimizer_state]
    seen: set[int] = set()
    found: set[str] = set()
    while pending:
        item = pending.pop()
        if not isinstance(item, Mapping) and type(item) not in (list, tuple):
            continue
        item_id = id(item)
        if item_id in seen:
            continue
        seen.add(item_id)
        if isinstance(item, Mapping):
            found.update(key for key in item if key in _EMBEDDED_DISTRIBUTED_PARAMETER_STATE_KEYS)
            pending.extend(item.values())
        else:
            pending.extend(item)

    if found:
        keys = ", ".join(sorted(found))
        raise RuntimeError(
            "PEFT checkpoint optimizer payload contains embedded distributed parameter state "
            f"({keys}); expected validated optimizer_parameter_state_rank*.pt sidecars"
        )


def _is_distributed_optimizer_leaf(optimizer) -> bool:
    return (
        not getattr(optimizer, "is_stub_optimizer", False)
        and callable(getattr(optimizer, "get_parameter_state_dp_zero", None))
        and callable(getattr(optimizer, "load_parameter_state_from_dp_zero", None))
        and getattr(optimizer, "data_parallel_group", None) is not None
    )


def _megatron_external_parameter_state_layout(optimizer) -> bool | None:
    """Describe pinned Megatron's filename-based external-state layout.

    ``None`` means an unknown/custom optimizer, for which the serialized-state
    compatibility heuristic remains available.
    """
    children = getattr(optimizer, "chained_optimizers", None)
    if children is None:
        if getattr(optimizer, "is_stub_optimizer", False):
            return False
        return True if _is_distributed_optimizer_leaf(optimizer) else None
    if len(children) == 0:
        return False
    if len(children) == 1:
        return _megatron_external_parameter_state_layout(children[0])

    active_children = [child for child in children if not getattr(child, "is_stub_optimizer", False)]
    if not active_children:
        return False

    child_layouts = [_megatron_external_parameter_state_layout(child) for child in active_children]
    if any(layout is True for layout in child_layouts):
        # Pinned Megatron's multi-child filename loader iterates every child that
        # exposes the distributed-optimizer methods. Stub DistributedOptimizers
        # inherit those methods but do not initialize their process groups, so a
        # distributed+stub chain fails on only a subset of ranks before scatter.
        direct_distributed = [_is_distributed_optimizer_leaf(child) for child in active_children]
        has_stub_children = len(active_children) != len(children)
        if has_stub_children or not all(direct_distributed):
            raise RuntimeError(
                "PEFT checkpointing does not support mixed, nested, or stub distributed children in "
                "Megatron ChainedOptimizer"
            )
        return True

    if any(layout is None for layout in child_layouts):
        return None
    if any(_is_distributed_optimizer_leaf(child) for child in active_children):
        raise RuntimeError(
            "PEFT checkpointing does not support mixed or nested inline/distributed children in "
            "Megatron ChainedOptimizer"
        )
    return False


def _uses_external_parameter_state(optimizer: Any, optimizer_state: Any, transfer_fn: Any) -> bool:
    layout = _megatron_external_parameter_state_layout(optimizer)
    if layout is not None:
        if layout and not callable(transfer_fn):
            raise RuntimeError("distributed optimizer does not expose save_parameter_state()")
        return layout
    detected = (
        not _contains_inline_optimizer_tensor(optimizer_state)
        and _contains_megatron_optimizer_wrapper(optimizer_state)
        and callable(transfer_fn)
    )
    if detected and dist.is_initialized() and dist.get_world_size() > 1:
        raise RuntimeError("custom external optimizer state is unsupported in distributed PEFT checkpointing")
    return detected


def _checkpoint_consensus_group():
    """Use Orbit's world Gloo group, or the default group in Gloo-only tests."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return None

    from miles.utils import distributed_utils

    if distributed_utils.GLOO_GROUP is not None:
        return distributed_utils.GLOO_GROUP
    if str(dist.get_backend()).lower().endswith("gloo"):
        return None
    raise RuntimeError("PEFT checkpoint consensus requires Orbit's world Gloo process group")


def _all_gather_checkpoint_object(value: Any) -> list[Any]:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return [value]
    group = _checkpoint_consensus_group()
    gathered: list[Any] = [None] * dist.get_world_size(group=group)
    dist.all_gather_object(gathered, value, group=group)
    return gathered


def _raise_if_checkpoint_errors(label: str, local_error: str | None) -> None:
    errors = _all_gather_checkpoint_object(local_error)
    failures = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error is not None]
    if failures:
        raise RuntimeError(f"{label} failed on one or more ranks; " + "; ".join(failures))


def _coordinated_checkpoint_call(label: str, fn):
    value = None
    local_error = None
    try:
        value = fn()
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    _raise_if_checkpoint_errors(label, local_error)
    return value


def _training_state_path(adapter_dir: str | Path, rank: int | None = None) -> Path:
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
    return Path(adapter_dir) / f"training_state_rank{rank}.pt"


def _optimizer_parameter_state_path(adapter_dir: str | Path, rank: int | None = None) -> Path:
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
    return Path(adapter_dir) / f"{_OPTIMIZER_PARAMETER_STATE_PREFIX}{rank}.pt"


@dataclass(frozen=True)
class _RegularFileFingerprint:
    """Identity and mutation-sensitive metadata for one regular file."""

    dev: int
    ino: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass(frozen=True)
class _CheckpointFileBinding:
    """An absolute checkpoint path bound to the exact file seen in preflight."""

    path: str
    fingerprint: _RegularFileFingerprint


def _absolute_checkpoint_path(path: str | Path) -> Path:
    """Return an absolute lexical path without resolving its final symlink."""
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _regular_file_fingerprint(stat_result: os.stat_result) -> _RegularFileFingerprint:
    if not stat.S_ISREG(stat_result.st_mode):
        raise RuntimeError("checkpoint path is not a regular file")
    return _RegularFileFingerprint(
        dev=stat_result.st_dev,
        ino=stat_result.st_ino,
        mode=stat_result.st_mode,
        size=stat_result.st_size,
        mtime_ns=stat_result.st_mtime_ns,
        ctime_ns=stat_result.st_ctime_ns,
    )


def _open_checkpoint_file(path: Path):
    """Open without blocking on a concurrently substituted FIFO/device."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0)
    fd = os.open(path, flags)
    try:
        return os.fdopen(fd, "rb")
    except Exception:
        os.close(fd)
        raise


def _capture_checkpoint_file_binding(path: str | Path) -> _CheckpointFileBinding | None:
    """Capture a stable regular-file identity, or ``None`` for a missing path.

    The descriptor and final pathname are compared so a concurrent rename
    cannot make preflight bind a different inode from the one it opened.
    """
    path = _absolute_checkpoint_path(path)
    try:
        checkpoint_file = _open_checkpoint_file(path)
    except FileNotFoundError:
        try:
            path.stat()
        except FileNotFoundError:
            return None
        raise RuntimeError(f"checkpoint path changed while capturing preflight: {path}") from None

    with checkpoint_file:
        before = _regular_file_fingerprint(os.fstat(checkpoint_file.fileno()))
        after = _regular_file_fingerprint(os.fstat(checkpoint_file.fileno()))
        try:
            final = _regular_file_fingerprint(path.stat())
        except FileNotFoundError as exc:
            raise RuntimeError(f"checkpoint file disappeared while capturing preflight: {path}") from exc
    if before != after or before != final:
        raise RuntimeError(f"checkpoint file changed while capturing preflight: {path}")
    return _CheckpointFileBinding(path=str(path), fingerprint=before)


def _verify_checkpoint_file_binding(
    path: str | Path,
    binding: _CheckpointFileBinding | None,
) -> None:
    """Verify that ``path`` still denotes the exact preflight file (or absence)."""
    path = _absolute_checkpoint_path(path)
    if binding is None:
        try:
            path.stat()
        except FileNotFoundError:
            return
        raise RuntimeError(f"checkpoint file appeared after preflight: {path}")

    if str(path) != binding.path:
        raise RuntimeError(f"checkpoint binding was for {binding.path}, not {path}")
    try:
        checkpoint_file = _open_checkpoint_file(path)
    except FileNotFoundError as exc:
        raise RuntimeError(f"checkpoint file disappeared after preflight: {path}") from exc
    with checkpoint_file:
        before = _regular_file_fingerprint(os.fstat(checkpoint_file.fileno()))
        after = _regular_file_fingerprint(os.fstat(checkpoint_file.fileno()))
        try:
            final = _regular_file_fingerprint(path.stat())
        except FileNotFoundError as exc:
            raise RuntimeError(f"checkpoint file disappeared after preflight: {path}") from exc
    if before != binding.fingerprint or after != binding.fingerprint or final != binding.fingerprint:
        raise RuntimeError(f"checkpoint file changed after preflight: {path}")


def _load_bound_torch_checkpoint(
    path: str | Path,
    binding: _CheckpointFileBinding | None,
    *,
    map_location: str | torch.device = "cpu",
    weights_only: bool,
) -> Any:
    """Load from the bound descriptor and reject mutation or path replacement.

    ``torch.load`` never reopens the pathname. Descriptor metadata is checked
    both before and after deserialization, followed by a pathname check, so
    truncation, in-place writes, and rename-based replacement are detected.
    """
    path = _absolute_checkpoint_path(path)
    if binding is None:
        raise RuntimeError(f"checkpoint file was absent during preflight: {path}")
    if str(path) != binding.path:
        raise RuntimeError(f"checkpoint binding was for {binding.path}, not {path}")
    try:
        checkpoint_file = _open_checkpoint_file(path)
    except FileNotFoundError as exc:
        raise RuntimeError(f"checkpoint file disappeared after preflight: {path}") from exc
    with checkpoint_file:
        before = _regular_file_fingerprint(os.fstat(checkpoint_file.fileno()))
        if before != binding.fingerprint:
            raise RuntimeError(f"checkpoint file changed after preflight: {path}")
        payload = torch.load(checkpoint_file, map_location=map_location, weights_only=weights_only)
        after = _regular_file_fingerprint(os.fstat(checkpoint_file.fileno()))
        try:
            final = _regular_file_fingerprint(path.stat())
        except FileNotFoundError as exc:
            raise RuntimeError(f"checkpoint file disappeared after preflight: {path}") from exc
    if after != binding.fingerprint or final != binding.fingerprint:
        raise RuntimeError(f"checkpoint file changed while it was being loaded: {path}")
    return payload


@dataclass(frozen=True)
class PeftCheckpointPreflight:
    adapter_dir: str
    native_shards_present: bool
    training_state_present: bool
    native_shard_binding: _CheckpointFileBinding | None
    training_state_binding: _CheckpointFileBinding | None
    optimizer_parameter_state_binding: _CheckpointFileBinding | None
    native_shard_path: str | None = None


@dataclass(frozen=True)
class _NativeAdapterShardCoordinates:
    tp_rank: int
    tp_size: int
    pp_rank: int
    ep_rank: int
    ep_size: int
    etp_rank: int
    etp_size: int


def _local_native_adapter_shard_coordinates() -> _NativeAdapterShardCoordinates:
    """Resolve the local native-shard identity from Megatron's MPU state."""
    if not dist.is_initialized():
        return _NativeAdapterShardCoordinates(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            ep_rank=0,
            ep_size=1,
            etp_rank=0,
            etp_size=1,
        )
    return _NativeAdapterShardCoordinates(
        tp_rank=mpu.get_tensor_model_parallel_rank(),
        tp_size=mpu.get_tensor_model_parallel_world_size(),
        pp_rank=mpu.get_pipeline_model_parallel_rank(),
        ep_rank=mpu.get_expert_model_parallel_rank(),
        ep_size=mpu.get_expert_model_parallel_world_size(),
        etp_rank=mpu.get_expert_tensor_parallel_rank(),
        etp_size=mpu.get_expert_tensor_parallel_world_size(),
    )


def _native_adapter_shard_name(
    tp_rank: int,
    pp_rank: int,
    ep_rank: int,
    ep_size: int,
    etp_rank: int,
    etp_size: int,
    tp_size: int,
) -> str:
    """Name one native shard, preserving the legacy name for redundant axes."""
    name = f"adapter_megatron_tp{tp_rank}_pp{pp_rank}"
    if ep_size > 1:
        name += f"_ep{ep_rank}"
    if tp_size % etp_size != 0:
        name += f"_etp{etp_rank}"
    return name + ".pt"


def _local_native_adapter_shard_path(adapter_dir: str | Path) -> Path:
    coordinates = _local_native_adapter_shard_coordinates()
    return Path(adapter_dir) / _native_adapter_shard_name(
        coordinates.tp_rank,
        coordinates.pp_rank,
        coordinates.ep_rank,
        coordinates.ep_size,
        coordinates.etp_rank,
        coordinates.etp_size,
        coordinates.tp_size,
    )


def _local_native_adapter_shard_candidates(adapter_dir: str | Path) -> tuple[Path, Path]:
    """Return the current coordinate shard and safe legacy global-rank shard."""
    rank = dist.get_rank() if dist.is_initialized() else 0
    adapter_dir = Path(adapter_dir)
    return (
        _local_native_adapter_shard_path(adapter_dir),
        adapter_dir / f"adapter_megatron_rank{rank}.pt",
    )


def _preflight_native_adapter_paths(
    adapter_dir: str | Path,
    preflight: PeftCheckpointPreflight,
) -> tuple[Path, tuple[Path, Path]]:
    candidates = tuple(
        _absolute_checkpoint_path(path) for path in _local_native_adapter_shard_candidates(adapter_dir)
    )
    selected = preflight.native_shard_path
    if selected is None and preflight.native_shard_binding is not None:
        selected = preflight.native_shard_binding.path
    selected_path = _absolute_checkpoint_path(selected) if selected is not None else candidates[0]
    if selected_path not in candidates:
        raise RuntimeError(f"native adapter preflight path is not valid for this rank: {selected_path}")
    return selected_path, candidates


def preflight_peft_adapter_checkpoint(adapter_path: str | Path) -> PeftCheckpointPreflight:
    """Bind rank-local files and reach consensus on required-file presence."""
    adapter_dir = _coordinated_checkpoint_call(
        "PEFT checkpoint path resolution",
        lambda: Path(adapter_path).expanduser().resolve(strict=False),
    )

    def capture_local_bindings():
        native_paths = tuple(
            _absolute_checkpoint_path(path) for path in _local_native_adapter_shard_candidates(adapter_dir)
        )
        native_bindings = tuple(_capture_checkpoint_file_binding(path) for path in native_paths)
        present = [(path, binding) for path, binding in zip(native_paths, native_bindings, strict=True) if binding]
        if len(present) > 1:
            raise RuntimeError(f"multiple native adapter shards match this rank: {[str(path) for path, _ in present]}")
        native_path, native_binding = present[0] if present else (native_paths[0], None)
        native_layout = None
        if native_binding is not None:
            native_layout = "coordinate" if native_path == native_paths[0] else "global-rank"
        return (
            str(native_path),
            native_binding,
            _capture_checkpoint_file_binding(_training_state_path(adapter_dir)),
            _capture_checkpoint_file_binding(_optimizer_parameter_state_path(adapter_dir)),
            native_layout,
        )

    native_path, native_binding, training_binding, optimizer_parameter_binding, native_layout = (
        _coordinated_checkpoint_call(
            "PEFT checkpoint snapshot capture",
            capture_local_bindings,
        )
    )
    local_presence = (
        str(adapter_dir),
        native_binding is not None,
        training_binding is not None,
        optimizer_parameter_binding is not None,
        native_layout,
    )
    presence_by_rank = _all_gather_checkpoint_object(local_presence)

    adapter_dirs = [presence[0] for presence in presence_by_rank]
    native_presence = [presence[1] for presence in presence_by_rank]
    training_presence = [presence[2] for presence in presence_by_rank]
    optimizer_parameter_presence = [presence[3] for presence in presence_by_rank]
    native_layouts = [presence[4] for presence in presence_by_rank]
    inconsistencies = []
    if len(set(adapter_dirs)) != 1:
        inconsistencies.append(f"adapter paths differ across ranks: {adapter_dirs}")
    if len(set(native_presence)) != 1:
        inconsistencies.append(
            "native adapter shards are present on ranks "
            f"{[rank for rank, present in enumerate(native_presence) if present]} and missing on ranks "
            f"{[rank for rank, present in enumerate(native_presence) if not present]}"
        )
    elif native_presence[0] and len(set(native_layouts)) != 1:
        inconsistencies.append(f"native adapter shard layouts differ across ranks: {native_layouts}")
    if len(set(training_presence)) != 1:
        inconsistencies.append(
            "training-state sidecars are present on ranks "
            f"{[rank for rank, present in enumerate(training_presence) if present]} and missing on ranks "
            f"{[rank for rank, present in enumerate(training_presence) if not present]}"
        )
    elif not training_presence[0] and any(optimizer_parameter_presence):
        inconsistencies.append(
            "optimizer parameter-state sidecars are present without training state on ranks "
            f"{[rank for rank, present in enumerate(optimizer_parameter_presence) if present]}"
        )
    if inconsistencies:
        raise RuntimeError("PEFT checkpoint preflight found inconsistent rank-local files: " + "; ".join(inconsistencies))

    return PeftCheckpointPreflight(
        adapter_dir=str(adapter_dir),
        native_shards_present=native_presence[0],
        training_state_present=training_presence[0],
        native_shard_binding=native_binding,
        training_state_binding=training_binding,
        optimizer_parameter_state_binding=optimizer_parameter_binding,
        native_shard_path=native_path,
    )


def _validate_preflight_adapter_dir(adapter_dir: str | Path, preflight: PeftCheckpointPreflight) -> None:
    normalized_adapter_dir, normalized_preflight_dir = _coordinated_checkpoint_call(
        "PEFT checkpoint preflight directory resolution",
        lambda: (
            str(Path(adapter_dir).expanduser().resolve(strict=False)),
            str(Path(preflight.adapter_dir).expanduser().resolve(strict=False)),
        ),
    )

    def resolve_local_paths():
        native_path, native_candidates = _preflight_native_adapter_paths(normalized_adapter_dir, preflight)
        native_layout = None
        if preflight.native_shard_binding is not None:
            native_layout = "coordinate" if native_path == native_candidates[0] else "global-rank"
        if dist.is_initialized():
            rank = dist.get_rank()
        else:
            rank = 0
        return (
            (
                str(native_path),
                str(_absolute_checkpoint_path(_training_state_path(normalized_adapter_dir, rank))),
                str(_absolute_checkpoint_path(_optimizer_parameter_state_path(normalized_adapter_dir, rank))),
            ),
            native_layout,
        )

    expected_paths, native_layout = _coordinated_checkpoint_call(
        "PEFT checkpoint preflight path resolution",
        resolve_local_paths,
    )
    local_binding = (
        normalized_adapter_dir,
        normalized_preflight_dir,
        preflight.native_shards_present,
        preflight.training_state_present,
        native_layout,
    )
    bindings = _all_gather_checkpoint_object(local_binding)
    local_error = None
    if normalized_adapter_dir != normalized_preflight_dir:
        local_error = f"preflight was for {preflight.adapter_dir}, not {adapter_dir}"
    elif len(set(bindings)) != 1:
        local_error = f"preflight binding differs across ranks: {bindings}"
    elif preflight.native_shards_present != (preflight.native_shard_binding is not None):
        local_error = "native adapter presence does not match its rank-local preflight binding"
    elif preflight.training_state_present != (preflight.training_state_binding is not None):
        local_error = "training-state presence does not match its rank-local preflight binding"
    elif any(
        binding is not None and binding.path != path
        for binding, path in zip(
            (
                preflight.native_shard_binding,
                preflight.training_state_binding,
                preflight.optimizer_parameter_state_binding,
            ),
            expected_paths,
            strict=True,
        )
    ):
        local_error = "rank-local preflight file binding has an unexpected path"
    _raise_if_checkpoint_errors("PEFT checkpoint preflight binding", local_error)


def _validate_peft_checkpoint_snapshot(preflight: PeftCheckpointPreflight) -> None:
    """Validate every rank-local path in a saved PEFT snapshot together."""
    adapter_dir = Path(preflight.adapter_dir)

    def validate_local_snapshot() -> None:
        selected_path, candidate_paths = _preflight_native_adapter_paths(adapter_dir, preflight)
        for candidate_path in candidate_paths:
            binding = preflight.native_shard_binding if candidate_path == selected_path else None
            _verify_checkpoint_file_binding(candidate_path, binding)
        _verify_checkpoint_file_binding(
            _training_state_path(adapter_dir),
            preflight.training_state_binding,
        )
        _verify_checkpoint_file_binding(
            _optimizer_parameter_state_path(adapter_dir),
            preflight.optimizer_parameter_state_binding,
        )

    _coordinated_checkpoint_call("PEFT checkpoint snapshot validation", validate_local_snapshot)


@dataclass(frozen=True)
class PeftSyncSpec:
    method: str
    adapter_name: str
    adapter_config: dict
    sync_transport: str


def get_peft_method(args) -> str:
    return getattr(args, "peft_method", "none")


def is_peft_enabled(args) -> bool:
    return get_peft_method(args) != "none"


from megatron.bridge.orbit.oft.param_names import CANONICAL_OFT_SLICE_NAMES, is_peft_adapter_param_name


def is_adapter_param_name(name: str) -> bool:
    return is_peft_adapter_param_name(name) or is_dsv4_grouped_moe_oft_param_name(name)


def _maybe_legacy_canonical_oft_key(name: str) -> str | None:
    """If ``name`` is a CanonicalOFT split slice (``...adapter_q.oft_r``), return
    the legacy shared-R key (``...adapter.oft_r``) it would have lived under in
    a pre-fix checkpoint. Returns ``None`` for non-split keys."""
    for slice_name in CANONICAL_OFT_SLICE_NAMES:
        token = f".adapter_{slice_name}."
        if token in name:
            return name.replace(token, ".adapter.", 1)
    return None


def is_peft_model(model: Sequence[torch.nn.Module]) -> bool:
    for model_chunk in model:
        for name, _ in model_chunk.named_parameters():
            if is_adapter_param_name(name):
                return True
    return False


def validate_peft_checkpoint_type(adapter_dir: Path, expected_method: str) -> dict:
    config_path = Path(adapter_dir) / "adapter_config.json"
    if not config_path.exists():
        return {}

    with config_path.open() as f:
        config = json.load(f)

    actual_type = config.get("peft_type")
    expected_type = expected_method.upper()
    if actual_type is not None and actual_type.upper() != expected_type:
        raise ValueError(f"PEFT checkpoint at {adapter_dir} has peft_type={actual_type}, expected {expected_type}.")
    return config


def create_peft_instance(args):
    method = get_peft_method(args)
    if method == "oft":
        from orbit.megatron.oft_utils import create_oft_instance

        return create_oft_instance(args)
    if method == "lora":
        from miles.backends.megatron_utils.lora_utils import create_lora_instance

        return create_lora_instance(args)
    return None


def build_peft_sync_spec(args) -> PeftSyncSpec | None:
    method = get_peft_method(args)
    if method == "oft":
        from orbit.megatron.oft_utils import OFT_ADAPTER_NAME, build_oft_sync_config

        return PeftSyncSpec(
            method="oft",
            adapter_name=OFT_ADAPTER_NAME,
            adapter_config=build_oft_sync_config(args),
            sync_transport=OFT_SYNC_TRANSPORT,
        )
    if method == "lora":
        from miles.backends.megatron_utils.lora_utils import LORA_ADAPTER_NAME, build_lora_sync_config

        return PeftSyncSpec(
            method="lora",
            adapter_name=LORA_ADAPTER_NAME,
            adapter_config=build_lora_sync_config(args),
            sync_transport=LORA_SYNC_TRANSPORT,
        )
    return None


def save_peft_checkpoint(
    model,
    args,
    save_dir,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
    iteration: int | None = None,
    active_student_version: str | None = None,
    self_teacher: Any | None = None,
) -> str:
    def build_local_dispatch() -> tuple[str, bool]:
        method = get_peft_method(args)
        if method not in ("lora", "oft"):
            raise ValueError(f"Cannot save PEFT checkpoint when peft_method={method!r}.")
        return method, self_teacher is not None

    local_dispatch = _coordinated_checkpoint_call("PEFT save dispatch validation", build_local_dispatch)
    dispatches = _all_gather_checkpoint_object(local_dispatch)
    if len(set(dispatches)) != 1:
        raise RuntimeError(f"PEFT save dispatch differs across ranks: {dispatches}")
    method = local_dispatch[0]
    if method == "lora":
        from miles.backends.megatron_utils.lora_utils import save_lora_checkpoint

        adapter_dir = save_lora_checkpoint(
            model,
            args,
            save_dir,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            iteration=iteration,
            active_student_version=active_student_version,
        )
    elif method == "oft":
        from orbit.megatron.oft_utils import save_oft_checkpoint

        adapter_dir = save_oft_checkpoint(
            model,
            args,
            save_dir,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            iteration=iteration,
            active_student_version=active_student_version,
        )
    else:  # pragma: no cover - validated by the coordinated dispatch above
        raise AssertionError(f"unreachable PEFT save method: {method!r}")

    if self_teacher is not None:
        from orbit.opd.self_teacher_checkpoint import TeacherCheckpointError, save_self_teacher_sidecar

        rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        local_error = None
        try:
            save_self_teacher_sidecar(
                adapter_dir,
                self_teacher,
                rank=rank,
                world_size=world_size,
            )
        except Exception as exc:  # every rank must leave the collective together
            local_error = f"{type(exc).__name__}: {exc}"

        if world_size > 1:
            from miles.utils.distributed_utils import get_gloo_group

            errors: list[str | None] = [None] * world_size
            dist.all_gather_object(errors, local_error, group=get_gloo_group())
        else:
            errors = [local_error]

        failures = [f"rank {failed_rank}: {error}" for failed_rank, error in enumerate(errors) if error]
        if failures:
            raise TeacherCheckpointError(
                "self-teacher sidecar save failed on one or more ranks; " + "; ".join(failures)
            )
    return adapter_dir


def load_peft_adapter(
    model,
    args,
    adapter_path: str,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
    expected_iteration: int | None = None,
    expected_active_student_version: str | None = None,
    checkpoint_preflight: PeftCheckpointPreflight | None = None,
) -> tuple[bool, int | None]:
    method = get_peft_method(args)
    adapter_dir = Path(adapter_path)
    if checkpoint_preflight is None:
        checkpoint_preflight = preflight_peft_adapter_checkpoint(adapter_dir)
    else:
        _validate_preflight_adapter_dir(adapter_dir, checkpoint_preflight)

    _coordinated_checkpoint_call(
        "PEFT adapter config validation",
        lambda: validate_peft_checkpoint_type(adapter_dir, expected_method=method),
    )

    if method == "lora":
        from miles.backends.megatron_utils.lora_utils import load_lora_adapter

        return load_lora_adapter(
            model,
            adapter_path,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            expected_iteration=expected_iteration,
            expected_active_student_version=expected_active_student_version,
            checkpoint_preflight=checkpoint_preflight,
        )
    if method == "oft":
        from orbit.megatron.oft_utils import load_oft_adapter

        return load_oft_adapter(
            model,
            adapter_path,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            expected_iteration=expected_iteration,
            expected_active_student_version=expected_active_student_version,
            checkpoint_preflight=checkpoint_preflight,
        )
    raise ValueError(f"Cannot load PEFT adapter when peft_method={method!r}.")


# ---------------------------------------------------------------------------
# Shared training-state save/load (used by save/load_{lora,oft}_checkpoint)
# ---------------------------------------------------------------------------


def save_training_state(
    adapter_dir: Path,
    optimizer: Any | None,
    opt_param_scheduler: Any | None,
    iteration: int | None,
    *,
    active_student_version: str | None = None,
    no_save_optim: bool = False,
) -> None:
    def validate_save_request() -> None:
        if iteration is not None and not _is_bounded_nonnegative_integer(iteration):
            raise ValueError("PEFT checkpoint iteration must be a bounded nonnegative integer")
        if active_student_version is not None and not _is_canonical_student_version(active_student_version):
            raise ValueError("active student version must be canonical nonnegative decimal text")

    _coordinated_checkpoint_call("PEFT training-state save request validation", validate_save_request)
    state_path = _training_state_path(adapter_dir)
    parameter_state_path = _optimizer_parameter_state_path(adapter_dir)
    if optimizer is None or no_save_optim:
        # Repeated writes to an existing export directory must not leave a
        # resumable optimizer sidecar behind when --no-save-optim is active.
        _coordinated_checkpoint_call(
            "PEFT stale training-state cleanup",
            lambda: (state_path.unlink(missing_ok=True), parameter_state_path.unlink(missing_ok=True)),
        )
        if no_save_optim:
            logger.info(f"Skipped optimizer/scheduler state for {adapter_dir} (--no-save-optim)")
        return

    _coordinated_checkpoint_call(
        "PEFT distributed optimizer state initialization",
        lambda: prepare_distributed_optimizer_state_for_save(optimizer),
    )
    optimizer_state = _coordinated_checkpoint_call(
        "PEFT optimizer state serialization",
        optimizer.state_dict,
    )
    save_parameter_state = getattr(optimizer, "save_parameter_state", None)
    has_external_parameter_state = _coordinated_checkpoint_call(
        "PEFT external optimizer layout validation",
        lambda: _uses_external_parameter_state(
            optimizer,
            optimizer_state,
            save_parameter_state,
        ),
    )
    external_layouts = _all_gather_checkpoint_object(has_external_parameter_state)
    if len(set(external_layouts)) != 1:
        raise RuntimeError(f"PEFT external optimizer layout differs across ranks: {external_layouts}")
    _coordinated_checkpoint_call(
        "PEFT distributed optimizer source validation",
        lambda: validate_distributed_optimizer_sources_for_save(optimizer)
        if has_external_parameter_state
        else None,
    )
    _coordinated_checkpoint_call(
        "PEFT optimizer parameter-state materialization",
        lambda: save_parameter_state(str(parameter_state_path))
        if has_external_parameter_state
        else parameter_state_path.unlink(missing_ok=True),
    )
    scheduler_state = _coordinated_checkpoint_call(
        "PEFT optimizer scheduler state serialization",
        opt_param_scheduler.state_dict if opt_param_scheduler else lambda: None,
    )
    _coordinated_checkpoint_call(
        "PEFT training-state save",
        lambda: torch.save(
            {
                "iteration": iteration,
                "active_student_version": active_student_version,
                "optimizer": optimizer_state,
                "optimizer_parameter_state": has_external_parameter_state,
                "opt_param_scheduler": scheduler_state,
            },
            state_path,
        ),
    )
    logger.info(f"Saved optimizer/scheduler state to {state_path.parent}")


def peft_training_state_exists(adapter_dir: str | Path) -> bool:
    """Return whether this rank has a resumable PEFT training-state sidecar.

    Callers that will enter optimizer collectives must use
    :func:`preflight_peft_adapter_checkpoint` instead.
    """
    return _training_state_path(adapter_dir).is_file()


def _process_group_rank(group: Any) -> int:
    rank = getattr(group, "rank", None)
    return int(rank()) if callable(rank) else dist.get_rank(group=group)


def _process_group_size(group: Any) -> int:
    size = getattr(group, "size", None)
    return int(size()) if callable(size) else dist.get_world_size(group=group)


def _validate_distributed_optimizer_leaf_topology(optimizer: Any) -> None:
    data_parallel_group = getattr(optimizer, "data_parallel_group", None)
    data_parallel_group_gloo = getattr(optimizer, "data_parallel_group_gloo", None)
    if data_parallel_group is None or data_parallel_group_gloo is None:
        raise RuntimeError("distributed optimizer is missing its NCCL or Gloo data-parallel group")

    group_rank = _process_group_rank(data_parallel_group)
    gloo_rank = _process_group_rank(data_parallel_group_gloo)
    group_size = _process_group_size(data_parallel_group)
    gloo_size = _process_group_size(data_parallel_group_gloo)
    if (group_rank, group_size) != (gloo_rank, gloo_size):
        raise RuntimeError(
            "distributed optimizer NCCL/Gloo data-parallel rank layouts differ: "
            f"NCCL={(group_rank, group_size)}, Gloo={(gloo_rank, gloo_size)}"
        )
    if dist.is_initialized():
        group_ranks = dist.get_process_group_ranks(data_parallel_group)
        gloo_ranks = dist.get_process_group_ranks(data_parallel_group_gloo)
        if group_ranks != gloo_ranks:
            raise RuntimeError(
                "distributed optimizer NCCL/Gloo data-parallel memberships differ: "
                f"NCCL={group_ranks}, Gloo={gloo_ranks}"
            )

    gbuf_ranges = getattr(optimizer, "gbuf_ranges", None)
    buffers = getattr(optimizer, "buffers", None)
    if not isinstance(gbuf_ranges, Sequence) or not isinstance(buffers, Sequence):
        raise RuntimeError("distributed optimizer has no inspectable gradient-buffer layout")
    if len(gbuf_ranges) != len(buffers):
        raise RuntimeError("distributed optimizer gradient-buffer layout length is inconsistent")

    for gbuf_idx, gbuf_range_maps in enumerate(gbuf_ranges):
        if not isinstance(gbuf_range_maps, Mapping) or len(gbuf_range_maps) != 1:
            raise RuntimeError(f"distributed optimizer gbuf {gbuf_idx} must contain exactly one dtype")
        range_maps = next(iter(gbuf_range_maps.values()))
        buckets = getattr(buffers[gbuf_idx], "buckets", None)
        if not isinstance(range_maps, Sequence) or not isinstance(buckets, Sequence):
            raise RuntimeError(f"distributed optimizer gbuf {gbuf_idx} has no inspectable buckets")
        if len(range_maps) != len(buckets):
            raise RuntimeError(f"distributed optimizer gbuf {gbuf_idx} bucket count is inconsistent")
        for bucket_idx, (range_map, bucket) in enumerate(zip(range_maps, buckets, strict=True)):
            padded_numel = int(bucket.grad_data.numel())
            unpadded_numel = int(bucket.numel_unpadded)
            if padded_numel <= 0 or padded_numel % gloo_size != 0:
                raise RuntimeError(
                    f"distributed optimizer gbuf {gbuf_idx} bucket {bucket_idx} padded size is invalid"
                )
            if not 0 < unpadded_numel <= padded_numel:
                raise RuntimeError(
                    f"distributed optimizer gbuf {gbuf_idx} bucket {bucket_idx} unpadded size is invalid"
                )
            local_numel = padded_numel // gloo_size
            param_map = range_map.get("param_map") if isinstance(range_map, Mapping) else None
            if not isinstance(param_map, Mapping):
                raise RuntimeError(
                    f"distributed optimizer gbuf {gbuf_idx} bucket {bucket_idx} has no parameter map"
                )
            for param_range_map in param_map.values():
                local_range = param_range_map.get("gbuf_local") if isinstance(param_range_map, Mapping) else None
                start = getattr(local_range, "start", None)
                end = getattr(local_range, "end", None)
                if type(start) is not int or type(end) is not int or not 0 <= start <= end <= local_numel:
                    raise RuntimeError(
                        f"distributed optimizer gbuf {gbuf_idx} bucket {bucket_idx} has an invalid local range"
                    )


def _validate_and_normalize_external_leaf_state(optimizer: Any, state: Any) -> dict[Any, Any]:
    if not isinstance(state, Mapping):
        raise RuntimeError("distributed optimizer parameter state must be a mapping")
    normalized = dict(state)
    split_state_dict_if_needed = getattr(optimizer, "split_state_dict_if_needed", None)
    if not callable(split_state_dict_if_needed):
        raise RuntimeError("distributed optimizer does not expose checkpoint layout normalization")
    split_state_dict_if_needed(normalized)
    if normalized.get("buckets_coalesced") is not True:
        raise RuntimeError("distributed optimizer parameter state is not in the coalesced format")

    gloo_size = _process_group_size(optimizer.data_parallel_group_gloo)
    for gbuf_idx, gbuf_range_maps in enumerate(optimizer.gbuf_ranges):
        dtype, range_maps = next(iter(gbuf_range_maps.items()))
        buffer = optimizer.buffers[gbuf_idx]
        expected_numel = int(buffer.numel_unpadded)
        bucket_numel = sum(int(bucket.numel_unpadded) for bucket in buffer.buckets)
        if expected_numel != bucket_numel:
            raise RuntimeError(f"distributed optimizer gbuf {gbuf_idx} unpadded size is inconsistent")
        try:
            dtype_state = normalized[gbuf_idx][dtype]
        except (KeyError, TypeError) as exc:
            raise RuntimeError(
                f"distributed optimizer parameter state is missing gbuf {gbuf_idx} dtype {dtype}"
            ) from exc
        if not isinstance(dtype_state, Mapping) or dtype_state.get("numel_unpadded") != expected_numel:
            raise RuntimeError(f"distributed optimizer gbuf {gbuf_idx} checkpoint size is incompatible")
        if len(range_maps) != len(buffer.buckets):
            raise RuntimeError(f"distributed optimizer gbuf {gbuf_idx} checkpoint bucket count is incompatible")
        for key in ("param", "exp_avg", "exp_avg_sq"):
            tensor = dtype_state.get(key)
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.layout != torch.strided
                or tensor.dtype != torch.float32
                or tensor.ndim != 1
                or not tensor.is_contiguous()
                or tensor.numel() != expected_numel
            ):
                raise RuntimeError(
                    f"distributed optimizer gbuf {gbuf_idx} checkpoint tensor {key!r} is incompatible"
                )
        for bucket_idx, bucket in enumerate(buffer.buckets):
            padded_numel = int(bucket.grad_data.numel())
            if padded_numel <= 0 or padded_numel % gloo_size != 0:
                raise RuntimeError(
                    f"distributed optimizer gbuf {gbuf_idx} bucket {bucket_idx} is incompatible"
                )
    return normalized


def _external_parameter_state_leaves(optimizer: Any) -> tuple[tuple[Any, ...], bool]:
    if _megatron_external_parameter_state_layout(optimizer) is not True:
        raise RuntimeError("optimizer does not use a supported distributed external-state layout")
    children = getattr(optimizer, "chained_optimizers", None)
    if children is None:
        return (optimizer,), False
    if len(children) == 1:
        return _external_parameter_state_leaves(children[0])
    # Layout validation above guarantees direct, active distributed leaves.
    return tuple(children), True


def _validate_external_parameter_state_topology(optimizer: Any) -> None:
    if _megatron_external_parameter_state_layout(optimizer) is not True:
        return
    leaves, _ = _external_parameter_state_leaves(optimizer)
    for leaf in leaves:
        _validate_distributed_optimizer_leaf_topology(leaf)


def _distributed_optimizer_source_params(leaf: Any):
    """Yield each model parameter's validated optimizer source and local width."""
    index_map = getattr(leaf, "model_param_group_index_map", None)
    inner_optimizer = getattr(leaf, "optimizer", None)
    param_groups = getattr(inner_optimizer, "param_groups", None)
    if not isinstance(index_map, Mapping):
        raise RuntimeError("distributed optimizer has no model-parameter group index map")
    if not isinstance(param_groups, Sequence):
        raise RuntimeError("distributed optimizer has no inspectable parameter groups")

    for gbuf_idx, gbuf_range_maps in enumerate(leaf.gbuf_ranges):
        for range_maps in gbuf_range_maps.values():
            for bucket_idx, range_map in enumerate(range_maps):
                for model_param, param_range_map in range_map["param_map"].items():
                    context = f"gbuf {gbuf_idx} bucket {bucket_idx}"
                    try:
                        index = index_map[model_param]
                    except (KeyError, TypeError) as exc:
                        raise RuntimeError(
                            f"distributed optimizer {context} model parameter has no group index"
                        ) from exc
                    if type(index) not in (tuple, list) or len(index) != 2:
                        raise RuntimeError(
                            f"distributed optimizer {context} model-parameter group index is invalid"
                        )
                    group_index, group_order = index
                    if type(group_index) is not int or type(group_order) is not int:
                        raise RuntimeError(
                            f"distributed optimizer {context} model-parameter group index is invalid"
                        )
                    if not 0 <= group_index < len(param_groups):
                        raise RuntimeError(
                            f"distributed optimizer {context} parameter-group index is out of range"
                        )
                    param_group = param_groups[group_index]
                    group_params = param_group.get("params") if isinstance(param_group, Mapping) else None
                    if not isinstance(group_params, Sequence) or not 0 <= group_order < len(group_params):
                        raise RuntimeError(
                            f"distributed optimizer {context} parameter order is out of range"
                        )
                    local_range = param_range_map["gbuf_local"]
                    yield model_param, group_params[group_order], local_range.end - local_range.start, context


def _leaf_has_absent_optimizer_state(leaf: Any) -> bool:
    inner_state = getattr(getattr(leaf, "optimizer", None), "state", None)
    if not isinstance(inner_state, Mapping):
        raise RuntimeError("distributed optimizer has no inspectable optimizer state")
    for _, optimizer_param, _, context in _distributed_optimizer_source_params(leaf):
        state = inner_state.get(optimizer_param)
        if state is None:
            return True
        if not isinstance(state, Mapping):
            raise RuntimeError(f"distributed optimizer {context} optimizer state is invalid")
        if len(state) == 0:
            return True
    return False


def prepare_distributed_optimizer_state_for_save(optimizer: Any) -> None:
    """Lazily initialize state for pinned distributed-optimizer save layouts.

    Megatron initializes Adam moments on the first optimizer step. A PEFT
    checkpoint can precede that step during critic-only warmup, so initialize
    only missing leaf state through Megatron's own precision-aware hook.
    """
    if _megatron_external_parameter_state_layout(optimizer) is not True:
        return
    leaves, _ = _external_parameter_state_leaves(optimizer)
    for leaf in leaves:
        _validate_distributed_optimizer_leaf_topology(leaf)
        if not _leaf_has_absent_optimizer_state(leaf):
            continue
        init_state_fn = getattr(leaf, "init_state_fn", None)
        if not callable(init_state_fn):
            raise RuntimeError("distributed optimizer cannot initialize absent optimizer state")
        init_state_fn(leaf.optimizer, leaf.config)


def _validate_distributed_optimizer_source_tensor(
    tensor: Any,
    *,
    key: str,
    expected_numel: int,
    context: str,
) -> None:
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.layout != torch.strided
        or tensor.device.type == "meta"
        or tensor.is_quantized
        or not tensor.is_floating_point()
        or tensor.ndim != 1
        or tensor.numel() != expected_numel
    ):
        raise RuntimeError(
            f"distributed optimizer {context} source {key!r} is incompatible; "
            f"expected a dense, non-meta, non-quantized floating 1-D tensor with {expected_numel} elements"
        )


def validate_distributed_optimizer_sources_for_save(optimizer: Any) -> None:
    """Validate every source dereferenced before Megatron's first save gather."""
    if _megatron_external_parameter_state_layout(optimizer) is not True:
        return
    leaves, _ = _external_parameter_state_leaves(optimizer)
    for leaf in leaves:
        _validate_distributed_optimizer_leaf_topology(leaf)
        get_tensors = getattr(leaf, "_get_main_param_and_optimizer_states", None)
        if not callable(get_tensors):
            raise RuntimeError("distributed optimizer does not expose pinned parameter-state sources")
        for model_param, _, expected_numel, context in _distributed_optimizer_source_params(leaf):
            try:
                tensors = get_tensors(model_param)
            except Exception as exc:
                raise RuntimeError(
                    f"distributed optimizer {context} parameter-state source lookup failed: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if not isinstance(tensors, Mapping):
                raise RuntimeError(f"distributed optimizer {context} parameter-state sources are invalid")
            for key in ("param", "exp_avg", "exp_avg_sq"):
                _validate_distributed_optimizer_source_tensor(
                    tensors.get(key),
                    key=key,
                    expected_numel=expected_numel,
                    context=context,
                )


@dataclass(frozen=True)
class _ExternalParameterStatePlan:
    leaves: tuple[Any, ...]
    cached_states: tuple[dict[Any, Any] | None, ...]
    custom_cached_state: Any | None = None
    is_custom: bool = False


def _build_external_parameter_state_plan(
    optimizer: Any,
    parameter_state_path: Path,
    parameter_state_binding: _CheckpointFileBinding | None,
) -> _ExternalParameterStatePlan | None:
    """Load and fully validate external state before any optimizer collective.

    Unknown filename-based optimizer APIs remain supported only in a
    single-process job, where a rank-local parse failure cannot strand peers.
    """
    layout = _megatron_external_parameter_state_layout(optimizer)
    if layout is not True:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if layout is False:
            raise RuntimeError("checkpoint external optimizer state does not match the current optimizer")
        if world_size != 1:
            raise RuntimeError("custom external optimizer state is unsupported in distributed PEFT resume")
        raw_state = _load_bound_torch_checkpoint(
            parameter_state_path,
            parameter_state_binding,
            map_location="cpu",
            weights_only=False,
        )
        return _ExternalParameterStatePlan(
            leaves=(),
            cached_states=(),
            custom_cached_state=raw_state,
            is_custom=True,
        )

    leaves, is_multi_child = _external_parameter_state_leaves(optimizer)
    for leaf in leaves:
        _validate_distributed_optimizer_leaf_topology(leaf)

    root_flags = tuple(_process_group_rank(leaf.data_parallel_group_gloo) == 0 for leaf in leaves)
    owns_file = any(root_flags)
    if owns_file:
        raw_state = _load_bound_torch_checkpoint(
            parameter_state_path,
            parameter_state_binding,
            map_location="cpu",
            weights_only=False,
        )
    else:
        if parameter_state_binding is not None:
            raise RuntimeError(f"unexpected optimizer parameter-state file on non-owner rank: {parameter_state_path}")
        raw_state = None

    if is_multi_child and owns_file:
        if type(raw_state) is not list or len(raw_state) != len(leaves):
            raise RuntimeError("chained optimizer parameter state must contain one slot per child")
        raw_states = tuple(raw_state)
    elif is_multi_child:
        raw_states = (None,) * len(leaves)
    else:
        raw_states = (raw_state,)

    cached_states: list[dict[Any, Any] | None] = []
    for index, (leaf, is_root, leaf_state) in enumerate(zip(leaves, root_flags, raw_states, strict=True)):
        if is_root:
            if leaf_state is None:
                raise RuntimeError(f"optimizer parameter state is missing child {index} on its DP root")
            cached_states.append(_validate_and_normalize_external_leaf_state(leaf, leaf_state))
        else:
            if owns_file and leaf_state is not None:
                raise RuntimeError(f"optimizer parameter state child {index} must be empty on this non-root rank")
            cached_states.append(None)
    return _ExternalParameterStatePlan(leaves=leaves, cached_states=tuple(cached_states))


def _validate_external_parameter_state_destinations(plan: _ExternalParameterStatePlan) -> None:
    for leaf in plan.leaves:
        for gbuf_idx, gbuf_range_maps in enumerate(leaf.gbuf_ranges):
            for range_maps in gbuf_range_maps.values():
                for range_map in range_maps:
                    for model_param, param_range_map in range_map["param_map"].items():
                        local_range = param_range_map["gbuf_local"]
                        expected_numel = local_range.end - local_range.start
                        group_index, group_order = leaf.model_param_group_index_map[model_param]
                        main_param = leaf.optimizer.param_groups[group_index]["params"][group_order]
                        if main_param.numel() != expected_numel:
                            raise RuntimeError(
                                f"distributed optimizer gbuf {gbuf_idx} main-parameter destination is incompatible"
                            )
                        optimizer_state = leaf.optimizer.state[main_param]
                        for key in ("exp_avg", "exp_avg_sq"):
                            tensor = optimizer_state.get(key)
                            if not isinstance(tensor, torch.Tensor) or tensor.numel() != expected_numel:
                                raise RuntimeError(
                                    f"distributed optimizer gbuf {gbuf_idx} destination {key!r} is incompatible"
                                )


def _dispatch_external_parameter_state(
    plan: _ExternalParameterStatePlan,
    custom_load_parameter_state=None,
) -> None:
    if plan.is_custom:
        if not callable(custom_load_parameter_state):
            raise RuntimeError("custom optimizer does not expose load_parameter_state()")
        # Unknown single-process optimizers only expose a filename API. Feed it
        # a private serialization of the state already read from the bound fd;
        # it must never reopen the mutable checkpoint pathname.
        with tempfile.NamedTemporaryFile(prefix="orbit-peft-optimizer-state-", suffix=".pt") as cached_file:
            torch.save(plan.custom_cached_state, cached_file)
            cached_file.flush()
            custom_load_parameter_state(cached_file.name)
        return
    for leaf, cached_state in zip(plan.leaves, plan.cached_states, strict=True):
        leaf.load_parameter_state_from_dp_zero(cached_state, update_legacy_format=False)


def _validate_training_state_payload(
    state_path: Path,
    state_binding: _CheckpointFileBinding | None,
    optimizer_parameter_state_binding: _CheckpointFileBinding | None,
    *,
    optimizer: Any | None,
    opt_param_scheduler: Any | None,
    expected_iteration: int | None,
    expected_active_student_version: str | None,
) -> dict[str, Any]:
    training_state = _load_bound_torch_checkpoint(
        state_path,
        state_binding,
        map_location="cpu",
        weights_only=False,
    )
    if type(training_state) is not dict:
        raise RuntimeError("PEFT checkpoint training state is invalid")

    iteration = training_state.get("iteration")
    if iteration is not None and not _is_bounded_nonnegative_integer(iteration):
        raise RuntimeError("PEFT checkpoint iteration is invalid")
    if expected_iteration is not None and (
        not _is_bounded_nonnegative_integer(iteration) or iteration != expected_iteration
    ):
        raise RuntimeError("PEFT checkpoint iteration does not match teacher-pool binding")

    active_student_version = training_state.get("active_student_version")
    if active_student_version is not None and not _is_canonical_student_version(active_student_version):
        raise RuntimeError("PEFT checkpoint active student version is invalid")
    if expected_active_student_version is not None and active_student_version != expected_active_student_version:
        raise RuntimeError("PEFT checkpoint active student version does not match teacher-pool binding")

    external_parameter_state = training_state.get("optimizer_parameter_state", False)
    if type(external_parameter_state) is not bool:
        raise RuntimeError("PEFT checkpoint optimizer-parameter-state marker is invalid")
    if not external_parameter_state and optimizer_parameter_state_binding is not None:
        raise RuntimeError(
            "PEFT checkpoint has an optimizer parameter-state file but its training-state marker is false"
        )

    optimizer_state = training_state.get("optimizer")
    if optimizer_state is not None:
        _validate_no_embedded_distributed_parameter_state(optimizer_state)

    if optimizer is not None:
        if optimizer_state is None:
            raise RuntimeError("PEFT checkpoint has no optimizer state; training resume is not possible")
        current_external_layout = _megatron_external_parameter_state_layout(optimizer)
        if current_external_layout is True and not external_parameter_state:
            raise RuntimeError("PEFT checkpoint is missing distributed optimizer parameter state")
        if current_external_layout is False and external_parameter_state:
            raise RuntimeError("PEFT checkpoint external optimizer state does not match the current optimizer")
        if external_parameter_state and not callable(getattr(optimizer, "load_parameter_state", None)):
            raise RuntimeError("PEFT checkpoint requires distributed optimizer parameter state")
        if opt_param_scheduler is not None and training_state.get("opt_param_scheduler") is None:
            raise RuntimeError("PEFT checkpoint has no optimizer scheduler state; training resume is not possible")

    return training_state


def _validate_training_metadata_consensus(training_state: dict[str, Any]) -> None:
    local_metadata = (
        training_state.get("iteration"),
        training_state.get("active_student_version"),
    )
    metadata_by_rank = _all_gather_checkpoint_object(local_metadata)
    if len(set(metadata_by_rank)) != 1:
        raise RuntimeError(f"PEFT checkpoint training metadata differs across ranks: {metadata_by_rank}")


def _validate_expected_training_binding(
    expected_iteration: int | None,
    expected_active_student_version: str | None,
) -> None:
    if expected_iteration is not None and not _is_bounded_nonnegative_integer(expected_iteration):
        raise ValueError("expected PEFT checkpoint iteration must be bounded and nonnegative")
    if expected_active_student_version is not None and not _is_canonical_student_version(
        expected_active_student_version
    ):
        raise ValueError("expected active student version must be canonical decimal text")


@dataclass(frozen=True)
class _PreparedPeftTrainingState:
    training_state: dict[str, Any] | None
    external_parameter_state_plan: _ExternalParameterStatePlan | None


def _prepare_training_state(
    adapter_dir: Path,
    optimizer: Any | None,
    opt_param_scheduler: Any | None,
    *,
    expected_iteration: int | None = None,
    expected_active_student_version: str | None = None,
    checkpoint_preflight: PeftCheckpointPreflight | None = None,
) -> _PreparedPeftTrainingState:
    """Parse and validate all needed state without mutating model/optimizer state."""
    _coordinated_checkpoint_call(
        "PEFT expected checkpoint binding validation",
        lambda: _validate_expected_training_binding(expected_iteration, expected_active_student_version),
    )
    state_path = _training_state_path(adapter_dir)
    if checkpoint_preflight is None:
        checkpoint_preflight = preflight_peft_adapter_checkpoint(adapter_dir)
    else:
        _validate_preflight_adapter_dir(adapter_dir, checkpoint_preflight)
    _validate_peft_checkpoint_snapshot(checkpoint_preflight)
    training_state_present = checkpoint_preflight.training_state_present

    if not training_state_present:
        if expected_iteration is not None or expected_active_student_version is not None:
            raise RuntimeError("PEFT checkpoint training state required by binding is missing")
        return _PreparedPeftTrainingState(
            training_state=None,
            external_parameter_state_plan=None,
        )

    training_state = _coordinated_checkpoint_call(
        "PEFT training-state parse/validation",
        lambda: _validate_training_state_payload(
            state_path,
            checkpoint_preflight.training_state_binding,
            checkpoint_preflight.optimizer_parameter_state_binding,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            expected_iteration=expected_iteration,
            expected_active_student_version=expected_active_student_version,
        ),
    )
    _validate_training_metadata_consensus(training_state)
    parameter_state_path = _optimizer_parameter_state_path(adapter_dir)
    external_parameter_state = training_state.get("optimizer_parameter_state") is True
    external_parameter_state_plan = (
        _coordinated_checkpoint_call(
            "PEFT optimizer parameter-state preflight",
            lambda: _build_external_parameter_state_plan(
                optimizer,
                parameter_state_path,
                checkpoint_preflight.optimizer_parameter_state_binding,
            )
            if external_parameter_state
            else None,
        )
        if optimizer is not None
        else None
    )
    return _PreparedPeftTrainingState(
        training_state=training_state,
        external_parameter_state_plan=external_parameter_state_plan,
    )


def _restore_prepared_training_state(
    prepared: _PreparedPeftTrainingState,
    optimizer: Any | None,
    opt_param_scheduler: Any | None,
) -> int | None:
    training_state = prepared.training_state
    if training_state is None:
        return None

    iteration = training_state.get("iteration")
    if optimizer is None:
        if iteration is not None:
            logger.info(f"Validated PEFT training state at iteration {iteration}")
        return iteration

    optimizer_state = training_state["optimizer"]
    load_parameter_state = getattr(optimizer, "load_parameter_state", None)
    external_parameter_state = training_state.get("optimizer_parameter_state") is True
    external_parameter_state_plan = prepared.external_parameter_state_plan

    _coordinated_checkpoint_call(
        "PEFT optimizer state restore",
        lambda: optimizer.load_state_dict(optimizer_state),
    )
    _coordinated_checkpoint_call(
        "PEFT optimizer parameter-state destination validation",
        lambda: _validate_external_parameter_state_destinations(external_parameter_state_plan)
        if external_parameter_state_plan is not None
        else None,
    )

    def restore_external_parameter_state() -> None:
        if not external_parameter_state:
            return
        if external_parameter_state_plan is None:
            raise RuntimeError("PEFT external optimizer state has no validated restore plan")
        _dispatch_external_parameter_state(
            external_parameter_state_plan,
            custom_load_parameter_state=load_parameter_state,
        )

    _coordinated_checkpoint_call(
        "PEFT optimizer parameter-state restore",
        restore_external_parameter_state,
    )
    logger.info("Restored optimizer state from PEFT checkpoint")

    if opt_param_scheduler is not None:
        _coordinated_checkpoint_call(
            "PEFT optimizer scheduler restore",
            lambda: opt_param_scheduler.load_state_dict(training_state["opt_param_scheduler"]),
        )
        logger.info("Restored LR scheduler state from PEFT checkpoint")

    if iteration is not None:
        logger.info(f"Resuming PEFT training from iteration {iteration}")
    return iteration


def load_training_state(
    adapter_dir: Path,
    optimizer: Any | None,
    opt_param_scheduler: Any | None,
    *,
    expected_iteration: int | None = None,
    expected_active_student_version: str | None = None,
    checkpoint_preflight: PeftCheckpointPreflight | None = None,
) -> int | None:
    prepared = _prepare_training_state(
        adapter_dir,
        optimizer,
        opt_param_scheduler,
        expected_iteration=expected_iteration,
        expected_active_student_version=expected_active_student_version,
        checkpoint_preflight=checkpoint_preflight,
    )
    return _restore_prepared_training_state(prepared, optimizer, opt_param_scheduler)


def restore_peft_training_state_after_optimizer_build(
    args: Namespace,
    optimizer: Any,
    opt_param_scheduler: Any,
    *,
    expected_iteration: int,
) -> bool:
    """Complete the second half of a low-precision PEFT resume.

    Low-precision actors must load base and adapter model tensors before the
    optimizer exists. ``load_training_state(..., optimizer=None)`` discovers
    the saved iteration during that first phase; this helper then restores the
    optimizer, scheduler, and any external distributed-optimizer tensors after
    construction. Re-reading with ``expected_iteration`` catches a sidecar that
    is missing, changed, or inconsistent before optimizer state is mutated.
    """
    adapter_dir = getattr(args, "_peft_resume_adapter_dir", None)
    if adapter_dir is None:
        return False

    training_state_found = getattr(args, "_peft_training_state_found", None)
    checkpoint_preflight = getattr(args, "_peft_checkpoint_preflight", None)
    if checkpoint_preflight is None:
        raise RuntimeError("PEFT second-phase optimizer restore requires the saved checkpoint preflight")
    _validate_preflight_adapter_dir(adapter_dir, checkpoint_preflight)
    # Validate native weights, training state, and rank-local external state as
    # one saved snapshot before mutating the newly constructed optimizer.
    _validate_peft_checkpoint_snapshot(checkpoint_preflight)
    if training_state_found is False:
        # A weights-only adapter intentionally keeps the fresh optimizer, but
        # still re-check the saved preflight so a sidecar that appeared between
        # model load and optimizer construction cannot silently change the
        # resume mode.
        load_training_state(
            Path(adapter_dir),
            None,
            None,
            checkpoint_preflight=checkpoint_preflight,
        )
        return False

    restored_iteration = load_training_state(
        Path(adapter_dir),
        optimizer,
        opt_param_scheduler,
        expected_iteration=expected_iteration,
        checkpoint_preflight=checkpoint_preflight,
    )
    if restored_iteration != expected_iteration:
        raise RuntimeError(
            "PEFT optimizer training-state iteration does not match the model/adapter resume iteration"
        )
    return True


# ---------------------------------------------------------------------------
# Shared HF <-> Megatron module-name mappings (PEFT-neutral)
# ---------------------------------------------------------------------------


# Standard PEFT: merged Q/K/V and merged up/gate (default for LoRA and OFT).
_STANDARD_HF_TO_MEGATRON = {
    "q_proj": "linear_qkv",
    "k_proj": "linear_qkv",
    "v_proj": "linear_qkv",
    "o_proj": "linear_proj",
    "gate_proj": "linear_fc1",
    "up_proj": "linear_fc1",
    "down_proj": "linear_fc2",
    "embed_tokens": "word_embeddings",
    "lm_head": "output_layer",
}

_STANDARD_ALL_MODULES = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]

# CanonicalLoRA: Split Q/K/V and up/gate (LoRA-only variant).
_CANONICAL_HF_TO_MEGATRON = {
    "q_proj": "linear_q",
    "k_proj": "linear_k",
    "v_proj": "linear_v",
    "o_proj": "linear_proj",
    "gate_proj": "linear_fc1_gate",
    "up_proj": "linear_fc1_up",
    "down_proj": "linear_fc2",
    "embed_tokens": "word_embeddings",
    "lm_head": "output_layer",
}

_CANONICAL_ALL_MODULES = [
    "linear_q",
    "linear_k",
    "linear_v",
    "linear_proj",
    "linear_fc1_up",
    "linear_fc1_gate",
    "linear_fc2",
]

# Fused Megatron leaf names accepted under variant="canonical" for backwards
# compat with legacy launchers. CanonicalOFT itself rejects these (it asserts
# in __post_init__), so we expand them to the split forms before the HF map.
_CANONICAL_FUSED_MEGATRON_TO_SPLIT = {
    "linear_qkv": ["linear_q", "linear_k", "linear_v"],
    "linear_fc1": ["linear_fc1_gate", "linear_fc1_up"],
}

# Multi-Latent Attention (DeepSeek v2/v3/v4): split low-rank Q/K/V projections.
# ``q_proj`` is used only when ``q_lora_rank is None`` (DeepSeek-Lite path);
# otherwise the model exposes ``q_a_proj`` / ``q_b_proj``. KV is always factored.
_MLA_HF_TO_MEGATRON = {
    "q_proj": "linear_q_proj",
    "q_a_proj": "linear_q_down_proj",
    "q_b_proj": "linear_q_up_proj",
    "kv_a_proj_with_mqa": "linear_kv_down_proj",
    "kv_b_proj": "linear_kv_up_proj",
    "o_proj": "linear_proj",
    "gate_proj": "linear_fc1",
    "up_proj": "linear_fc1",
    "down_proj": "linear_fc2",
}

_MLA_ALL_MODULES = [
    "linear_q_proj",
    "linear_q_down_proj",
    "linear_q_up_proj",
    "linear_kv_down_proj",
    "linear_kv_up_proj",
    "linear_proj",
    "linear_fc1",
    "linear_fc2",
]

# DeepSeek V4 uses native attention sublayer names. Its grouped attention
# exposes wq_a/wq_b/wkv/wo_a/wo_b directly; wkv must not match nested
# compressor modules, so use full-name globs instead of bare leaf names.
_DSV4_HF_TO_MEGATRON: dict[str, str] = {
    "wq_a": "*.self_attention.wq_a",
    "wq_b": "*.self_attention.wq_b",
    "wkv": "*.self_attention.wkv",
    "wo_a": "*.self_attention.wo_a",
    "wo_b": "*.self_attention.wo_b",
    "w1": "*.experts.*.w1",
    "w2": "*.experts.*.w2",
    "w3": "*.experts.*.w3",
}

_DSV4_ALL_MODULES = [
    "*.self_attention.wq_a",
    "*.self_attention.wq_b",
    "*.self_attention.wkv",
    "*.self_attention.wo_a",
    "*.self_attention.wo_b",
]

DSV4_MOE_HF_TARGET_MODULES = ["w1", "w2", "w3"]
DSV4_MOE_MEGATRON_TARGET_MODULES = [
    "*.experts.*.w1",
    "*.experts.*.w2",
    "*.experts.*.w3",
]

# Megatron -> HF (inverse mapping, one-to-many).
_MEGATRON_TO_HF_MODULES = {
    # Standard (merged layers)
    "linear_qkv": ["q_proj", "k_proj", "v_proj"],
    "linear_proj": ["o_proj"],
    "linear_fc1": ["gate_proj", "up_proj"],
    "linear_fc2": ["down_proj"],
    # Canonical (split layers)
    "linear_q": ["q_proj"],
    "linear_k": ["k_proj"],
    "linear_v": ["v_proj"],
    "linear_fc1_gate": ["gate_proj"],
    "linear_fc1_up": ["up_proj"],
    # MLA
    "linear_q_proj": ["q_proj"],
    "linear_q_down_proj": ["q_a_proj"],
    "linear_q_up_proj": ["q_b_proj"],
    "linear_kv_down_proj": ["kv_a_proj_with_mqa"],
    "linear_kv_up_proj": ["kv_b_proj"],
    # all-mode adds
    "word_embeddings": ["embed_tokens"],
    "output_layer": ["lm_head"],
}

_HF_MODULE_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    # MLA-only HF names
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    # all-mode adds
    "embed_tokens",
    "lm_head",
}

DEFAULT_HF_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

DEFAULT_MLA_HF_TARGET_MODULES = [
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

DEFAULT_DSV4_HF_TARGET_MODULES = [
    "wq_a",
    "wq_b",
    "wkv",
    "wo_a",
    "wo_b",
]


def detect_peft_variant(args: Namespace) -> Variant:
    """Pick the right HF<->Megatron mapping variant from runtime args.

    MLA models advertise themselves via Megatron's ``--multi-latent-attention``
    flag. DeepSeek V4 also sets MLA, but it uses native wq_a/wq_b/wkv/wo_a/wo_b
    sublayer names and must be selected explicitly with ``--peft-variant=dsv4``.
    CanonicalLoRA opts in via ``--peft-variant=canonical`` (LoRA-only).
    Everything else falls back to the standard merged-QKV mapping.
    """
    explicit = getattr(args, "peft_variant", "standard")
    if explicit == "dsv4":
        return "dsv4"
    if getattr(args, "multi_latent_attention", False):
        return "mla"
    if explicit == "canonical":
        return "canonical"
    return "standard"


def convert_target_modules_to_megatron(
    hf_modules: str | list[str],
    variant: Variant = "standard",
) -> list[str]:
    """Convert HuggingFace module names to Megatron module names.

    HF (standard):  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    HF (MLA):       q_proj | q_a_proj/q_b_proj, kv_a_proj_with_mqa, kv_b_proj,
                    o_proj, gate_proj, up_proj, down_proj
    Megatron (standard):  linear_qkv, linear_proj, linear_fc1, linear_fc2
    Megatron (canonical): linear_q, linear_k, linear_v, linear_proj,
                          linear_fc1_up, linear_fc1_gate, linear_fc2
    Megatron (mla):       linear_q_proj | linear_q_down_proj/linear_q_up_proj,
                          linear_kv_down_proj, linear_kv_up_proj, linear_proj,
                          linear_fc1, linear_fc2

    ``variant`` controls LoRA's variant selection and OFT's explicit
    ``--oft-type oft`` compatibility path. Default OFT calls this with
    ``variant="canonical"`` so HF Q/K/V and gate/up targets map to split
    Megatron names for CanonicalOFT; the fused Megatron names ``linear_qkv``
    and ``linear_fc1`` are also accepted under this variant and are expanded
    into ``linear_q/k/v`` and ``linear_fc1_gate/linear_fc1_up`` respectively,
    so legacy launchers continue to work without an explicit migration.
    ``--oft-type oft`` calls this with the detected runtime variant, so
    fused-QKV models map Q/K/V to ``linear_qkv`` and gate/up to ``linear_fc1``
    for the legacy shared-R OFT wrapper.

    Special values: "all", "all-linear", "all_linear" -> all linear modules
    for the selected variant. If input is already in Megatron format, returns
    as-is.
    """
    if variant not in ("standard", "canonical", "mla", "dsv4"):
        raise ValueError(f"variant must be 'standard', 'canonical', 'mla', or 'dsv4', got {variant!r}")

    if variant == "dsv4":
        all_modules = _DSV4_ALL_MODULES
        hf_to_megatron = _DSV4_HF_TO_MEGATRON
    elif variant == "mla":
        all_modules = _MLA_ALL_MODULES
        hf_to_megatron = _MLA_HF_TO_MEGATRON
    elif variant == "canonical":
        all_modules = _CANONICAL_ALL_MODULES
        hf_to_megatron = _CANONICAL_HF_TO_MEGATRON
    else:
        all_modules = _STANDARD_ALL_MODULES
        hf_to_megatron = _STANDARD_HF_TO_MEGATRON

    if isinstance(hf_modules, str):
        if hf_modules in ("all", "all-linear", "all_linear"):
            return list(all_modules)
        hf_modules = [hf_modules]
    elif isinstance(hf_modules, list) and len(hf_modules) == 1:
        if hf_modules[0] in ("all", "all-linear", "all_linear"):
            return list(all_modules)

    if variant == "canonical":
        # Expand legacy fused names (linear_qkv, linear_fc1) into canonical split forms; see docstring.
        expanded_modules: list[str] = []
        for module in hf_modules:
            split_modules = _CANONICAL_FUSED_MEGATRON_TO_SPLIT.get(module, [module])
            for split_module in split_modules:
                if split_module not in expanded_modules:
                    expanded_modules.append(split_module)
        hf_modules = expanded_modules

    known_hf_names = set(hf_to_megatron)
    if all(m not in known_hf_names for m in hf_modules if "*" not in m):
        return hf_modules

    megatron_modules: list[str] = []
    for module in hf_modules:
        megatron_name = hf_to_megatron.get(module, module)
        if megatron_name not in megatron_modules:
            megatron_modules.append(megatron_name)
    return megatron_modules


def convert_target_modules_to_hf(megatron_modules: list[str]) -> list[str]:
    """Convert Megatron module names to HuggingFace module names.

    Supports both standard and canonical Megatron names.
    """
    hf_modules: list[str] = []
    for module in megatron_modules:
        if module in _MEGATRON_TO_HF_MODULES:
            hf_modules.extend(_MEGATRON_TO_HF_MODULES[module])
        else:
            hf_modules.append(module)
    return hf_modules


def parse_exclude_modules(args: Namespace, variant: Variant = "standard") -> list[str]:
    """Parse and convert the ``--exclude-modules`` argument to Megatron names."""
    exclude_modules: list[str] = []
    raw = getattr(args, "exclude_modules", None)
    if raw:
        if isinstance(raw, str):
            exclude_modules = [m.strip() for m in raw.split(",")]
        else:
            exclude_modules = list(raw)
        exclude_modules = convert_target_modules_to_megatron(exclude_modules, variant=variant)
    return exclude_modules


def resolve_target_modules_hf(args: Namespace) -> list[str]:
    """HF-format target modules from ``args``; falls back to the full all-linear set."""
    modules = getattr(args, "target_modules", None)
    variant = detect_peft_variant(args)
    if not modules:
        if variant == "dsv4":
            return list(DEFAULT_DSV4_HF_TARGET_MODULES)
        if variant == "mla":
            return list(DEFAULT_MLA_HF_TARGET_MODULES)
        return list(DEFAULT_HF_TARGET_MODULES)
    if variant == "dsv4":
        return list(modules) if isinstance(modules, list) else [m.strip() for m in modules.split(",")]
    return convert_target_modules_to_hf(list(modules))


# ---------------------------------------------------------------------------
# Shared PEFT checkpoint save/load (consumed by lora_utils and oft_utils)
# ---------------------------------------------------------------------------


def native_adapter_state(
    model: Sequence[torch.nn.Module],
) -> dict[AdapterTensorKey, torch.Tensor]:
    """Snapshot every local adapter tensor with its VPP chunk identity."""

    return {
        key: parameter.detach().cpu().clone()
        for key, parameter in adapter_named_parameters(model, is_adapter_param_name).items()
    }


def _mapping_difference_message(
    expected: set[AdapterTensorKey],
    actual: set[AdapterTensorKey],
) -> str:
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    return f"native adapter state keys do not match model; missing={missing[:5]!r}, unknown={extra[:5]!r}"


def resolve_native_adapter_state(
    model: Sequence[torch.nn.Module],
    state: Mapping[object, object],
) -> dict[AdapterTensorKey, torch.Tensor]:
    """Validate tuple-key native state or unambiguous legacy plain-name state.

    The returned mapping is complete and shape-checked. Callers can therefore
    prepare all device conversions before mutating any live parameter.
    """

    params = adapter_named_parameters(model, is_adapter_param_name)
    if type(state) is not dict or not state:
        raise ValueError("native adapter state must be a nonempty exact dict")

    raw_keys = list(state)
    tuple_format = all(type(key) is tuple for key in raw_keys)
    legacy_format = all(type(key) is str and bool(key) for key in raw_keys)
    if not tuple_format and not legacy_format:
        raise ValueError("native adapter state key format is invalid or mixed")

    resolved: dict[AdapterTensorKey, object]
    if tuple_format:
        adapter_tensor_key_digest(raw_keys)
        actual_keys = set(raw_keys)
        expected_keys = set(params)
        if actual_keys != expected_keys:
            raise ValueError(_mapping_difference_message(expected_keys, actual_keys))
        resolved = {key: state[key] for key in params}
    else:
        resolved = {}
        for legacy_name in raw_keys:
            matches = [
                key
                for key in params
                if key[1] == legacy_name or _maybe_legacy_canonical_oft_key(key[1]) == legacy_name
            ]
            if len(matches) > 1:
                raise ValueError(f"legacy native adapter name {legacy_name!r} is ambiguous across model chunks")
            if not matches:
                raise ValueError(f"legacy native adapter state has unknown key {legacy_name!r}")
            key = matches[0]
            if key in resolved:
                raise ValueError(f"legacy native adapter state maps multiple names to {key!r}")
            resolved[key] = state[legacy_name]
        if set(resolved) != set(params):
            raise ValueError(_mapping_difference_message(set(params), set(resolved)))

    validated: dict[AdapterTensorKey, torch.Tensor] = {}
    for key, parameter in params.items():
        tensor = resolved[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"native adapter state value for {key!r} is not a tensor")
        if tensor.shape != parameter.shape:
            raise ValueError(
                f"native adapter tensor {key!r} shape {tuple(tensor.shape)} "
                f"does not match model shape {tuple(parameter.shape)}"
            )
        validated[key] = tensor
    return validated


def _to_peft_canonical_key(name: str) -> str:
    """Wrap a megatron-bridge adapter weight name into peft on-disk form.

    megatron-bridge emits LoRA/OFT adapter keys in two forms depending on
    the bridge version:

    * Suffix-only (older / unit-test inputs):
      ``model.layers.X.<mod>.lora_A`` / ``lora_B`` / ``oft_R``
    * With trailing ``.weight`` (bridge export_adapter_weights output):
      ``model.layers.X.<mod>.lora_A.weight`` etc.

    HF PEFT 0.19.x writes adapter safetensors with keys shaped
    ``base_model.model.<orig>.<adapter_suffix>.weight`` (the adapter name
    ``default`` is *not* in the on-disk key — peft's loader injects it
    during ``set_peft_model_state_dict`` via
    ``_insert_adapter_name_into_state_dict``).  Saving with ``.default.``
    already in the key causes peft to produce ``.default.default.weight``
    on load and miss every adapter tensor.

    This helper:
    * Prefixes with ``base_model.model.``
    * Validates the recognised adapter suffix (raises ``ValueError`` if
      none matches, so the save aborts rather than silently producing a
      non-loadable adapter)
    * Ensures the result ends in ``.weight`` (OFT bridge keys lack it,
      LoRA bridge keys already have it).
    """
    suffixes = ("oft_R", "lora_A", "lora_B")
    stripped = name[: -len(".weight")] if name.endswith(".weight") else name
    if not any(stripped.endswith(f".{suffix}") for suffix in suffixes):
        raise ValueError(
            f"cannot wrap adapter weight '{name}' to peft canonical form: " f"expected suffix in {suffixes}"
        )
    return f"base_model.model.{stripped}.weight"


def _save_peft_hf_artifacts(
    save_path: Path,
    state_dict: dict[str, torch.Tensor],
    config: dict[str, Any],
    *,
    base_model_name_or_path: str,
) -> None:
    """Write ``adapter_model.safetensors`` + ``adapter_config.json``.

    The on-disk format is HF-PEFT canonical so ``peft.PeftModel.from_pretrained``
    can load it directly: tensor keys are wrapped via
    ``_to_peft_canonical_key`` and the config carries
    ``base_model_name_or_path``.

    Tensors are cloned to guarantee distinct safetensors storage; safetensors
    forbids storage sharing, and the Megatron-Bridge LoRA exporter emits the
    fused-QKV / fused-gate-up A-side as a single tensor aliased under three HF
    names. ``.clone()`` also yields a contiguous tensor.
    """
    if not base_model_name_or_path:
        raise ValueError(
            "_save_peft_hf_artifacts: base_model_name_or_path must be a non-empty "
            "string (typically args.hf_checkpoint)"
        )

    serializable = {_to_peft_canonical_key(name): tensor.detach().clone() for name, tensor in state_dict.items()}
    safetensors_save_file(serializable, str(save_path / "adapter_model.safetensors"))

    enriched_config = dict(config)
    enriched_config["base_model_name_or_path"] = base_model_name_or_path
    with open(save_path / "adapter_config.json", "w") as f:
        json.dump(enriched_config, f, indent=2)

    os.sync()
    logger.info(f"Saved HF PEFT adapter to {save_path} with {len(serializable)} tensors")


@dataclass(frozen=True)
class _PeftSaveRankRoles:
    native_writer: bool
    hf_writer: bool
    tp_rank: int
    tp_size: int
    pp_rank: int
    ep_rank: int
    ep_size: int
    etp_rank: int
    etp_size: int


def _validate_peft_save_request(
    save_dir: str,
    *,
    method: str,
    args: Namespace,
    optimizer: Any | None,
    opt_param_scheduler: Any | None,
    iteration: int | None,
    active_student_version: str | None,
) -> tuple[Path, bool]:
    """Validate the branch-defining save request before any rank performs I/O."""

    def build_local_request() -> tuple[Any, ...]:
        no_save_optim = getattr(args, "no_save_optim", False)
        if method not in ("lora", "oft"):
            raise ValueError(f"unsupported PEFT save method: {method!r}")
        if type(no_save_optim) is not bool:
            raise TypeError("no_save_optim must be a boolean")
        if iteration is not None and not _is_bounded_nonnegative_integer(iteration):
            raise ValueError("PEFT checkpoint iteration must be a bounded nonnegative integer")
        if active_student_version is not None and not _is_canonical_student_version(active_student_version):
            raise ValueError("active student version must be canonical nonnegative decimal text")
        optimizer_present = optimizer is not None
        optimizer_stub = bool(getattr(optimizer, "is_stub_optimizer", False)) if optimizer_present else False
        optimizer_layout = _megatron_external_parameter_state_layout(optimizer) if optimizer_present else None
        return (
            str(Path(save_dir).expanduser().resolve(strict=False)),
            method,
            iteration,
            active_student_version,
            no_save_optim,
            optimizer_present,
            optimizer_stub,
            optimizer_layout,
            opt_param_scheduler is not None,
            str(getattr(args, "hf_checkpoint", "")),
        )

    local_request = _coordinated_checkpoint_call("PEFT save request validation", build_local_request)
    requests = _all_gather_checkpoint_object(local_request)
    if len(set(requests)) != 1:
        raise RuntimeError(f"PEFT save request differs across ranks: {requests}")
    return Path(local_request[0]), local_request[4]


def _resolve_peft_save_rank_roles() -> _PeftSaveRankRoles:
    """Resolve write ownership without excluding ranks from save collectives.

    EP and ETP ranks can own different expert adapter tensors, so native shard
    identity includes TP, PP, EP, and ETP. Exactly the lowest global rank among
    replicas of each realized coordinate writes that shard. The HF exporter
    produces a complete state on every participant and therefore has one global
    writer.
    """
    coordinates = _local_native_adapter_shard_coordinates()
    coordinate_key = (
        coordinates.tp_rank,
        coordinates.pp_rank,
        coordinates.ep_rank,
        coordinates.etp_rank,
    )
    global_rank = dist.get_rank() if dist.is_initialized() else 0
    gathered = _all_gather_checkpoint_object((coordinate_key, global_rank))
    native_writer = global_rank == min(
        rank for rank_coordinates, rank in gathered if rank_coordinates == coordinate_key
    )
    return _PeftSaveRankRoles(
        native_writer=native_writer,
        hf_writer=not dist.is_initialized() or dist.get_rank() == 0,
        tp_rank=coordinates.tp_rank,
        tp_size=coordinates.tp_size,
        pp_rank=coordinates.pp_rank,
        ep_rank=coordinates.ep_rank,
        ep_size=coordinates.ep_size,
        etp_rank=coordinates.etp_rank,
        etp_size=coordinates.etp_size,
    )


def _save_native_adapter_shard(
    model: Sequence[torch.nn.Module],
    save_path: Path,
    roles: _PeftSaveRankRoles,
) -> tuple[int, Path] | None:
    if not roles.native_writer:
        return None
    adapter_state = native_adapter_state(model)
    native_path = save_path / _native_adapter_shard_name(
        roles.tp_rank,
        roles.pp_rank,
        roles.ep_rank,
        roles.ep_size,
        roles.etp_rank,
        roles.etp_size,
        roles.tp_size,
    )
    torch.save(adapter_state, native_path)
    return len(adapter_state), native_path


def _export_peft_hf_state(
    model: Sequence[torch.nn.Module],
    exporter,
    patch_megatron_model,
) -> dict[str, torch.Tensor]:
    """Consume the distributed exporter on every rank and collect local output.

    The caller coordinates failures before and after this call.  The exporter's
    own TP/PP/EP collectives still require every participating rank to enter and
    make matching progress; no outer error gather can repair divergence inside
    those collectives.
    """
    state_dict: dict[str, torch.Tensor] = {}
    with patch_megatron_model(model):
        # megatron-bridge >=0.5 yields a 2-tuple (hf_name, tensor); older
        # versions yielded 3-tuples. Positional access handles both.
        for item in exporter(model, cpu=True, show_progress=False):
            hf_name, weight = item[0], item[1]
            state_dict[hf_name] = weight
    return state_dict


def save_peft_adapter_checkpoint(
    model: Sequence[torch.nn.Module],
    args: Namespace,
    save_dir: str,
    *,
    method: Literal["lora", "oft"],
    build_config: Any,  # callable() -> dict
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
    iteration: int | None = None,
    active_student_version: str | None = None,
) -> str:
    """Save a PEFT adapter checkpoint (native per-rank shards + HF artifacts).

    Both LoRA and OFT use this helper; the only method-specific pieces are the
    bridge exporter and the ``adapter_config.json`` contents.
    """
    save_path, no_save_optim = _validate_peft_save_request(
        save_dir,
        method=method,
        args=args,
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
        iteration=iteration,
        active_student_version=active_student_version,
    )
    import functools

    from megatron.bridge import AutoBridge

    from miles.utils import megatron_bridge_utils

    roles = _coordinated_checkpoint_call("PEFT save ownership resolution", _resolve_peft_save_rank_roles)

    _coordinated_checkpoint_call(
        "PEFT checkpoint directory creation",
        lambda: save_path.mkdir(parents=True, exist_ok=True)
        if roles.native_writer or roles.hf_writer
        else None,
    )

    # Megatron-native format (per realized TP/PP/EP coordinate, fast resume)
    native_result = _coordinated_checkpoint_call(
        "PEFT native adapter shard save",
        lambda: _save_native_adapter_shard(model, save_path, roles),
    )
    if native_result is not None:
        native_count, native_path = native_result
        logger.info(f"Saved {native_count} adapter tensors (native) to {native_path}")

    # HF PEFT format — bridge export is TP-collective, so every rank calls it.
    bridge = _coordinated_checkpoint_call(
        "PEFT bridge initialization",
        lambda: AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True),
    )
    def _select_exporter():
        if method != "oft":
            return bridge.export_adapter_weights
        # export_oft_adapter_weights is a free function (megatron.bridge.orbit
        # namespace, post-reattach), not a bridge method like
        # export_adapter_weights -- bind it so callers keep a uniform
        # exporter(model, ...) call shape.
        from megatron.bridge.orbit.conversion.oft_export import export_oft_adapter_weights

        return functools.partial(export_oft_adapter_weights, bridge)

    exporter = _coordinated_checkpoint_call(
        "PEFT bridge exporter selection",
        _select_exporter,
    )
    state_dict = _coordinated_checkpoint_call(
        "PEFT HF adapter export",
        lambda: _export_peft_hf_state(model, exporter, megatron_bridge_utils.patch_megatron_model),
    )

    _coordinated_checkpoint_call(
        "PEFT HF adapter artifact save",
        lambda: _save_peft_hf_artifacts(
            save_path,
            state_dict,
            config=build_config(),
            base_model_name_or_path=args.hf_checkpoint,
        )
        if roles.hf_writer
        else None,
    )

    # Every rank must participate: distributed optimizers gather parameter
    # state inside this call before their DP roots write rank-local files.
    save_training_state(
        save_path,
        optimizer,
        opt_param_scheduler,
        iteration,
        active_student_version=active_student_version,
        no_save_optim=no_save_optim,
    )

    return str(save_path)


def _load_and_convert_native_adapter_state(
    model: Sequence[torch.nn.Module],
    native_path: Path,
    native_binding: _CheckpointFileBinding | None,
) -> tuple[dict[AdapterTensorKey, torch.Tensor], dict[AdapterTensorKey, torch.nn.Parameter]]:
    state_dict = _load_bound_torch_checkpoint(
        native_path,
        native_binding,
        map_location="cpu",
        weights_only=True,
    )
    resolved = resolve_native_adapter_state(model, state_dict)
    params = adapter_named_parameters(model, is_adapter_param_name)
    converted = {
        key: resolved[key].to(device=parameter.device, dtype=parameter.dtype) for key, parameter in params.items()
    }
    return converted, params


def load_peft_adapter_checkpoint(
    model: Sequence[torch.nn.Module],
    adapter_path: str,
    *,
    label: str,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
    expected_iteration: int | None = None,
    expected_active_student_version: str | None = None,
    checkpoint_preflight: PeftCheckpointPreflight | None = None,
) -> tuple[bool, int | None]:
    """Load a PEFT adapter checkpoint from Megatron-native shards.

    ``label`` is the user-visible method name (``"LoRA"`` / ``"OFT"``) used in
    log messages. If only an HF PEFT artifact exists (no native shards), warns
    and returns ``(False, None)``.
    """
    adapter_dir = Path(adapter_path)
    if checkpoint_preflight is None:
        checkpoint_preflight = preflight_peft_adapter_checkpoint(adapter_dir)
    else:
        _validate_preflight_adapter_dir(adapter_dir, checkpoint_preflight)
    _validate_peft_checkpoint_snapshot(checkpoint_preflight)

    if checkpoint_preflight.native_shards_present:
        native_binding = checkpoint_preflight.native_shard_binding
        if native_binding is None:  # guarded by _validate_preflight_adapter_dir
            raise RuntimeError("native adapter preflight binding is missing")
        native_path = Path(native_binding.path)
        # Parse training/external payloads before copying adapter parameters so
        # every rank leaves checkpoint validation together without partial
        # model or optimizer mutation.
        prepared_training_state = _prepare_training_state(
            adapter_dir,
            optimizer,
            opt_param_scheduler,
            expected_iteration=expected_iteration,
            expected_active_student_version=expected_active_student_version,
            checkpoint_preflight=checkpoint_preflight,
        )
        converted, params = _coordinated_checkpoint_call(
            "PEFT native adapter shard parse/validation",
            lambda: _load_and_convert_native_adapter_state(
                model,
                native_path,
                native_binding,
            ),
        )
        with torch.no_grad():
            for key, parameter in params.items():
                parameter.copy_(converted[key])
        loaded = len(params)
        logger.info(f"Loaded {loaded} adapter tensors from Megatron-native checkpoint: {native_path}")
        iteration = _restore_prepared_training_state(
            prepared_training_state,
            optimizer,
            opt_param_scheduler,
        )
        return True, iteration

    if not adapter_dir.exists():
        logger.warning(f"{label} adapter path does not exist: {adapter_dir}")
        return False, None

    hf_safetensors = adapter_dir / "adapter_model.safetensors"
    hf_bin = adapter_dir / "adapter_model.bin"
    if hf_safetensors.exists() or hf_bin.exists():
        found = hf_safetensors if hf_safetensors.exists() else hf_bin
        logger.warning(
            f"Found HF PEFT adapter at {found} but direct HF PEFT loading into "
            f"Megatron is not yet supported. Please save using Megatron-native format "
            f"(adapter_megatron_tp*_pp*[_ep*].pt files) for checkpoint resume."
        )
        return False, None

    logger.warning(f"No adapter checkpoint found at {adapter_dir}")
    return False, None


def load_adapter_tensors_for_teacher(
    model: Sequence[torch.nn.Module],
    adapter_path: str,
) -> dict[AdapterTensorKey, torch.Tensor]:
    """Load a frozen teacher adapter as a chunk-aware tensor dict.

    Requires the rank-local Megatron-native shard written by
    save_peft_checkpoint; HF-only artifacts are rejected — the engine side
    consumes those, while the trainer side needs native names and shapes.
    """
    adapter_dir = Path(adapter_path)
    checkpoint_preflight = preflight_peft_adapter_checkpoint(adapter_dir)
    _validate_peft_checkpoint_snapshot(checkpoint_preflight)
    if not checkpoint_preflight.native_shards_present:
        native_path = _local_native_adapter_shard_path(adapter_dir)
        raise FileNotFoundError(
            f"OPD teacher adapter needs Megatron-native shards, missing {native_path}. "
            "Save the teacher with orbit's save_peft_checkpoint (HF-only artifacts are not "
            "loadable trainer-side)."
        )
    native_binding = checkpoint_preflight.native_shard_binding
    if native_binding is None:  # guarded by the preflight presence invariant
        raise RuntimeError("native adapter preflight binding is missing")
    native_path = Path(native_binding.path)
    converted, params = _coordinated_checkpoint_call(
        "OPD teacher native adapter shard parse/validation",
        lambda: _load_and_convert_native_adapter_state(model, native_path, native_binding),
    )
    return {key: converted[key].detach().clone() for key in params}
