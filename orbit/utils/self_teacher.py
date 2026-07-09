"""EMA / lagged snapshots of the student adapter, used as OPD self-teachers.

Adapter tensors are MB-scale, so shadowing them is ~free — this is the
capability that is prohibitive with full-model teachers. Torch-only module
(no megatron imports) so it stays CPU unit-testable.
"""

import torch


class SelfTeacherBuffer:
    def __init__(
        self,
        named_tensors: dict[str, torch.Tensor],
        mode: str,
        decay: float = 0.999,
        interval: int = 1,
    ):
        if mode not in ("ema", "lag"):
            raise ValueError(f"SelfTeacherBuffer mode must be 'ema' or 'lag', got {mode!r}.")
        self.mode = mode
        self.decay = decay
        self.interval = interval
        self._step = 0
        # Step-0 init: early training distills toward the starting adapter
        # state (standard mean-teacher warmup behavior).
        self.tensors = {name: t.detach().clone() for name, t in named_tensors.items()}

    @torch.no_grad()
    def update(self, named_tensors: dict[str, torch.Tensor]) -> None:
        if set(named_tensors) != set(self.tensors):
            raise ValueError("SelfTeacherBuffer.update: adapter param keys changed since init.")
        self._step += 1
        if self.mode == "ema":
            for name, t in named_tensors.items():
                self.tensors[name].mul_(self.decay).add_(t.detach(), alpha=1.0 - self.decay)
        elif self._step % self.interval == 0:
            for name, t in named_tensors.items():
                self.tensors[name].copy_(t.detach())
