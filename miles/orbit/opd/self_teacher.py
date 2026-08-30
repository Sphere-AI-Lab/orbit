"""FP32 EMA/lag snapshots of the student adapter used as OPD teachers.

Adapter tensors are MB-scale, so keeping a full-precision master is cheap and
avoids accumulating BF16/FP16 rounding error across post-training updates.
This module remains torch-only so its numerical and checkpoint contract stays
CPU unit-testable.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from miles.orbit.utils.adapter_tensors import AdapterTensorKey, adapter_tensor_key_digest


SELF_TEACHER_STATE_SCHEMA_VERSION = 1
_STATE_FIELDS = {
    "schema_version",
    "mode",
    "decay",
    "interval",
    "step",
    "key_digest",
    "tensors",
}


def _validate_config(mode: object, decay: object, interval: object) -> tuple[str, float, int]:
    if type(mode) is not str or mode not in {"ema", "lag"}:
        raise ValueError(f"SelfTeacherBuffer mode must be 'ema' or 'lag', got {mode!r}.")
    if (
        type(decay) is not float
        or not math.isfinite(decay)
        or decay <= 0.0
        or decay >= 1.0
    ):
        raise ValueError("SelfTeacherBuffer decay must be a finite float in (0, 1).")
    if type(interval) is not int or interval < 1:
        raise ValueError("SelfTeacherBuffer interval must be a positive integer.")
    return mode, decay, interval


def _validate_tensor_mapping(
    named_tensors: object,
    *,
    require_fp32: bool,
    require_finite: bool,
    context: str,
) -> dict[AdapterTensorKey, torch.Tensor]:
    if type(named_tensors) is not dict or not named_tensors:
        raise ValueError(f"{context} tensors must be a nonempty exact dict")
    adapter_tensor_key_digest(named_tensors)
    validated: dict[AdapterTensorKey, torch.Tensor] = {}
    for key, tensor in named_tensors.items():
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise ValueError(f"{context} tensor {key!r} is invalid")
        if require_fp32 and tensor.dtype is not torch.float32:
            raise ValueError(f"{context} tensor {key!r} must be FP32")
        if require_finite and not torch.isfinite(tensor).all().item():
            raise ValueError(f"{context} tensor {key!r} must be finite")
        validated[key] = tensor
    return validated


class SelfTeacherBuffer:
    def __init__(
        self,
        named_tensors: dict[AdapterTensorKey, torch.Tensor],
        mode: str,
        decay: float = 0.999,
        interval: int = 1,
    ) -> None:
        mode, decay, interval = _validate_config(mode, decay, interval)
        validated = _validate_tensor_mapping(
            named_tensors,
            require_fp32=False,
            require_finite=True,
            context="self-teacher",
        )
        self.mode = mode
        self.decay = decay
        self.interval = interval
        self._step = 0
        # Step-0 init distills toward the starting adapter. These are detached,
        # contiguous FP32 masters on the same device as each live parameter.
        self.tensors = {
            key: tensor.detach().to(dtype=torch.float32).clone().contiguous()
            for key, tensor in validated.items()
        }

    def _prepare_live_tensors(
        self,
        named_tensors: object,
        *,
        convert: bool,
    ) -> dict[AdapterTensorKey, torch.Tensor]:
        if type(named_tensors) is not dict or set(named_tensors) != set(self.tensors):
            raise ValueError("SelfTeacherBuffer.update: adapter param keys changed since init.")
        validated = _validate_tensor_mapping(
            named_tensors,
            require_fp32=False,
            require_finite=False,
            context="live adapter",
        )
        prepared: dict[AdapterTensorKey, torch.Tensor] = {}
        for key, master in self.tensors.items():
            live = validated[key]
            if live.shape != master.shape:
                raise ValueError(
                    f"live adapter tensor {key!r} shape {tuple(live.shape)} "
                    f"does not match master shape {tuple(master.shape)}"
                )
            if convert:
                prepared[key] = live.detach().to(device=master.device, dtype=torch.float32)
        return prepared

    @torch.no_grad()
    def update(self, named_tensors: dict[AdapterTensorKey, torch.Tensor]) -> None:
        next_step = self._step + 1
        refresh = self.mode == "ema" or next_step % self.interval == 0
        prepared = self._prepare_live_tensors(named_tensors, convert=refresh)
        self._step = next_step
        if self.mode == "ema":
            for key, live in prepared.items():
                self.tensors[key].mul_(self.decay).add_(live, alpha=1.0 - self.decay)
        elif self._step % self.interval == 0:
            for key, live in prepared.items():
                self.tensors[key].copy_(live)

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": SELF_TEACHER_STATE_SCHEMA_VERSION,
            "mode": self.mode,
            "decay": self.decay,
            "interval": self.interval,
            "step": self._step,
            "key_digest": adapter_tensor_key_digest(self.tensors),
            "tensors": {
                key: tensor.detach().cpu().clone().contiguous()
                for key, tensor in self.tensors.items()
            },
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> SelfTeacherBuffer:
        if type(state) is not dict or set(state) != _STATE_FIELDS:
            raise ValueError("self-teacher state fields do not match schema")
        schema_version = state["schema_version"]
        if (
            type(schema_version) is not int
            or schema_version != SELF_TEACHER_STATE_SCHEMA_VERSION
        ):
            raise ValueError("unsupported self-teacher state schema")
        step = state["step"]
        if type(step) is not int or step < 0:
            raise ValueError("self-teacher step must be nonnegative")
        tensors = _validate_tensor_mapping(
            state["tensors"],
            require_fp32=True,
            require_finite=True,
            context="self-teacher state",
        )
        key_digest = state["key_digest"]
        if type(key_digest) is not str or key_digest != adapter_tensor_key_digest(tensors):
            raise ValueError("self-teacher key digest mismatch")
        result = cls(
            tensors,
            mode=state["mode"],
            decay=state["decay"],
            interval=state["interval"],
        )
        result._step = step
        return result

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        candidate = type(self).from_state_dict(state)
        if (
            candidate.mode != self.mode
            or candidate.decay != self.decay
            or candidate.interval != self.interval
            or set(candidate.tensors) != set(self.tensors)
            or adapter_tensor_key_digest(candidate.tensors)
            != adapter_tensor_key_digest(self.tensors)
            or any(
                candidate.tensors[key].shape != self.tensors[key].shape
                for key in self.tensors
            )
        ):
            raise ValueError("self-teacher state does not match configured buffer")

        # Convert every candidate first. Only replace live state after the full
        # candidate has validated and every allocation has succeeded.
        replacement = {
            key: candidate.tensors[key]
            .to(device=self.tensors[key].device, dtype=torch.float32)
            .clone()
            .contiguous()
            for key in self.tensors
        }
        self._step = candidate._step
        self.tensors = replacement
