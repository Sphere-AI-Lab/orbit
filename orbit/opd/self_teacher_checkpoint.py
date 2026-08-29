"""Self-teacher checkpoint sidecars: fp32 EMA/lag state that survives resume.

Extracted from the ultra teacher-pool tier's teacher_checkpoint.py, keeping only
the per-rank sidecar save/load core (pool bindings, preflight reports, and the
hardened nofollow reader stay with the ultra program -- the sidecar sits beside
a checkpoint this same job wrote). Written next to the PEFT adapter checkpoint
so a resumed run continues the self-teacher from its exact prior state instead
of re-seeding from the resumed student.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from pathlib import Path

import torch

from orbit.ultra.strict_json import loads_strict
from orbit.opd.self_teacher import SELF_TEACHER_STATE_SCHEMA_VERSION, SelfTeacherBuffer


def has_self_teacher_sidecar(adapter_dir, *, rank: int) -> bool:
    """Whether a sidecar for this rank exists beside the adapter checkpoint.

    Resuming from a checkpoint written before sidecars existed is legitimate;
    callers gate the strict load on this instead of treating absence as
    corruption.
    """
    return os.path.lexists(Path(adapter_dir) / _metadata_filename(rank))


_SIDECAR_SCHEMA_VERSION = 1


_MAX_JSON_BYTES = 1024 * 1024


_MAX_TENSOR_BYTES = 4 * 1024 * 1024 * 1024


_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class TeacherCheckpointError(RuntimeError):
    pass


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, OverflowError):
        raise TeacherCheckpointError("checkpoint metadata is not canonical JSON") from None


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _exact_digest(value: object, *, name: str) -> str:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise TeacherCheckpointError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _exact_nonnegative(value: object, *, name: str) -> int:
    if type(value) is not int or value < 0 or value > 2**63 - 1:
        raise TeacherCheckpointError(f"{name} must be a bounded nonnegative integer")
    return value


def _absolute_nofollow_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _adapter_directory(adapter_dir: str | Path, *, create: bool) -> Path:
    if not isinstance(adapter_dir, (str, os.PathLike)):
        raise TeacherCheckpointError("adapter checkpoint directory path is invalid")
    path = _absolute_nofollow_path(Path(adapter_dir))
    try:
        if create:
            path.mkdir(parents=True, exist_ok=True)
        status = path.lstat()
    except OSError:
        raise TeacherCheckpointError("adapter checkpoint directory is not accessible") from None
    if not stat.S_ISDIR(status.st_mode) or path.is_symlink():
        raise TeacherCheckpointError("adapter checkpoint directory must be a real directory")
    return path


def _reject_symlink_or_nonregular(path: Path, *, missing_ok: bool) -> bool:
    try:
        status = path.lstat()
    except FileNotFoundError:
        if missing_ok:
            return False
        raise TeacherCheckpointError(f"checkpoint sidecar is missing: {path.name}") from None
    except OSError:
        raise TeacherCheckpointError(f"checkpoint sidecar is not accessible: {path.name}") from None
    if not stat.S_ISREG(status.st_mode) or path.is_symlink():
        raise TeacherCheckpointError(f"checkpoint sidecar must be a regular no-symlink file: {path.name}")
    return True


def _read_bytes(path: Path, *, max_bytes: int) -> bytes:
    _reject_symlink_or_nonregular(path, missing_ok=False)
    try:
        with open(path, "rb") as handle:
            encoded = handle.read(max_bytes + 1)
    except OSError:
        raise TeacherCheckpointError(f"checkpoint sidecar read failed: {path.name}") from None
    if len(encoded) > max_bytes:
        raise TeacherCheckpointError(f"checkpoint sidecar exceeds its byte limit: {path.name}")
    return encoded


def _read_json(path: Path) -> object:
    encoded = _read_bytes(path, max_bytes=_MAX_JSON_BYTES)
    try:
        return loads_strict(encoded, max_bytes=_MAX_JSON_BYTES, max_depth=8)
    except (TypeError, ValueError):
        raise TeacherCheckpointError(f"checkpoint metadata is corrupt: {path.name}") from None


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, encoded: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if _reject_symlink_or_nonregular(path, missing_ok=True):
            pass
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _atomic_write_json(path: Path, payload: Mapping[str, object]) -> None:
    _atomic_write_bytes(path, _canonical_json_bytes(dict(payload)) + b"\n")


def _validate_rank_world(rank: object, world_size: object) -> tuple[int, int]:
    rank = _exact_nonnegative(rank, name="distributed rank")
    if type(world_size) is not int or world_size <= 0 or world_size > 2**31 - 1:
        raise TeacherCheckpointError("distributed world size must be a positive integer")
    if rank >= world_size:
        raise TeacherCheckpointError("distributed rank must be smaller than world size")
    return rank, world_size


def _tensor_filename(rank: int) -> str:
    return f"self_teacher_rank{rank}.pt"


def _metadata_filename(rank: int) -> str:
    return f"self_teacher_rank{rank}.json"


def _atomic_save_tensors(path: Path, tensors: dict[object, torch.Tensor]) -> str:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(tensors, handle)
            handle.flush()
            os.fsync(handle.fileno())
        encoded = _read_bytes(temporary, max_bytes=_MAX_TENSOR_BYTES)
        digest = hashlib.sha256(encoded).hexdigest()
        if _reject_symlink_or_nonregular(path, missing_ok=True):
            pass
        os.replace(temporary, path)
        _fsync_directory(path.parent)
        return digest
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def save_self_teacher_sidecar(
    adapter_dir,
    buffer,
    *,
    rank: int,
    world_size: int,
) -> None:
    rank, world_size = _validate_rank_world(rank, world_size)
    if type(buffer) is not SelfTeacherBuffer:
        raise TeacherCheckpointError("self-teacher sidecar requires an exact buffer")
    directory = _adapter_directory(adapter_dir, create=True)
    state = buffer.state_dict()
    tensor_path = directory / _tensor_filename(rank)
    metadata_path = directory / _metadata_filename(rank)
    try:
        tensor_digest = _atomic_save_tensors(tensor_path, state["tensors"])
        identity: dict[str, object] = {
            "schema_version": _SIDECAR_SCHEMA_VERSION,
            "state_schema_version": state["schema_version"],
            "rank": rank,
            "world_size": world_size,
            "mode": state["mode"],
            "decay": state["decay"],
            "interval": state["interval"],
            "step": state["step"],
            "key_digest": state["key_digest"],
            "tensor_file": tensor_path.name,
            "tensor_file_sha256": tensor_digest,
        }
        metadata = {**identity, "sidecar_sha256": _canonical_digest(identity)}
        _atomic_write_json(metadata_path, metadata)
    except TeacherCheckpointError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise TeacherCheckpointError("self-teacher sidecar save failed") from None


_SIDECAR_FIELDS = {
    "schema_version",
    "state_schema_version",
    "rank",
    "world_size",
    "mode",
    "decay",
    "interval",
    "step",
    "key_digest",
    "tensor_file",
    "tensor_file_sha256",
    "sidecar_sha256",
}


def _validate_sidecar_metadata(
    value: object,
    *,
    rank: int,
    world_size: int,
) -> dict[str, object]:
    if type(value) is not dict or set(value) != _SIDECAR_FIELDS:
        raise TeacherCheckpointError("self-teacher sidecar metadata fields are invalid")
    if type(value["schema_version"]) is not int or value["schema_version"] != _SIDECAR_SCHEMA_VERSION:
        raise TeacherCheckpointError("self-teacher sidecar schema is invalid")
    if (
        type(value["state_schema_version"]) is not int
        or value["state_schema_version"] != SELF_TEACHER_STATE_SCHEMA_VERSION
    ):
        raise TeacherCheckpointError("self-teacher state schema is invalid")
    if value["rank"] != rank or type(value["rank"]) is not int:
        raise TeacherCheckpointError("self-teacher sidecar rank does not match")
    if value["world_size"] != world_size or type(value["world_size"]) is not int:
        raise TeacherCheckpointError("self-teacher sidecar world size does not match")
    if value["tensor_file"] != _tensor_filename(rank) or type(value["tensor_file"]) is not str:
        raise TeacherCheckpointError("self-teacher sidecar tensor file does not match rank")
    _exact_digest(value["key_digest"], name="self-teacher key")
    _exact_digest(value["tensor_file_sha256"], name="self-teacher tensor file")
    _exact_digest(value["sidecar_sha256"], name="self-teacher sidecar")
    identity = {key: item for key, item in value.items() if key != "sidecar_sha256"}
    if value["sidecar_sha256"] != _canonical_digest(identity):
        raise TeacherCheckpointError("self-teacher sidecar metadata digest does not match")
    if type(value["mode"]) is not str or value["mode"] not in {"ema", "lag"}:
        raise TeacherCheckpointError("self-teacher sidecar mode is invalid")
    if (
        type(value["decay"]) is not float
        or not math.isfinite(value["decay"])
        or not 0.0 < value["decay"] < 1.0
    ):
        raise TeacherCheckpointError("self-teacher sidecar decay is invalid")
    if type(value["interval"]) is not int or value["interval"] < 1:
        raise TeacherCheckpointError("self-teacher sidecar interval is invalid")
    _exact_nonnegative(value["step"], name="self-teacher step")
    return value


def load_self_teacher_sidecar(
    adapter_dir,
    buffer,
    *,
    rank: int,
    world_size: int,
) -> None:
    rank, world_size = _validate_rank_world(rank, world_size)
    if type(buffer) is not SelfTeacherBuffer:
        raise TeacherCheckpointError("self-teacher sidecar requires an exact buffer")
    directory = _adapter_directory(adapter_dir, create=False)
    metadata_path = directory / _metadata_filename(rank)
    tensor_path = directory / _tensor_filename(rank)
    metadata = _validate_sidecar_metadata(
        _read_json(metadata_path),
        rank=rank,
        world_size=world_size,
    )
    encoded = _read_bytes(tensor_path, max_bytes=_MAX_TENSOR_BYTES)
    if hashlib.sha256(encoded).hexdigest() != metadata["tensor_file_sha256"]:
        raise TeacherCheckpointError("self-teacher tensor file digest does not match")
    try:
        tensors = torch.load(io.BytesIO(encoded), map_location="cpu", weights_only=True)
    except (EOFError, OSError, RuntimeError, TypeError, ValueError):
        raise TeacherCheckpointError("self-teacher tensor file is corrupt") from None
    state = {
        "schema_version": metadata["state_schema_version"],
        "mode": metadata["mode"],
        "decay": metadata["decay"],
        "interval": metadata["interval"],
        "step": metadata["step"],
        "key_digest": metadata["key_digest"],
        "tensors": tensors,
    }
    try:
        buffer.load_state_dict(state)
    except (TypeError, ValueError, RuntimeError):
        raise TeacherCheckpointError(
            "self-teacher sidecar state does not match configured buffer"
        ) from None
