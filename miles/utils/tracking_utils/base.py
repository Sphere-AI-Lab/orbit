"""
Shared tracking interface for experiment logging backends.

Each backend implements ``init / log / finish``, and :class:`TrackingManager` fans out
calls to every active backend.

To add a new backend:
--------------------
1. Subclass :class:`TrackingBackend`.
2. Register it in :data:`BACKEND_REGISTRY`.
3. Add a corresponding ``--use-<name>`` CLI flag in ``arguments.py``.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class TrackingBackend(ABC):
    # Interface every logging backend must satisfy.

    @abstractmethod
    def init(self, args, *, primary: bool = True, **kwargs) -> bool | None: ...

    @abstractmethod
    def log(self, metrics: dict[str, Any], step: int | None = None, **kwargs) -> None: ...

    @abstractmethod
    def finish(self) -> None: ...


# Thin adapters for backwards compatibility to keep wandb_utils and tensorboard_utils untouched.
class WandbBackend(TrackingBackend):
    # Delegates to the existing ``wandb_utils`` helpers.

    def __init__(self) -> None:
        self._last_row_step = -1

    def init(self, args, *, primary: bool = True, **kwargs) -> bool | None:
        from . import wandb_utils

        if primary:
            return wandb_utils.init_wandb_primary(args, **kwargs)
        else:
            return wandb_utils.init_wandb_secondary(args, **kwargs)

    def log(self, metrics: dict[str, Any], step: int | None = None, **kwargs) -> None:
        import wandb

        # W&B has one global row step, while Miles has independent logical axes
        # such as train/step, rollout/step, and eval/step. Use a monotonically
        # increasing row id only for W&B history ordering; charts still use the
        # logical axes declared via ``wandb.define_metric(..., step_metric=...)``.
        row_step = self._next_row_step()
        wandb.log(metrics, step=row_step)

    def _next_row_step(self) -> int:
        row_step = time.time_ns() // 1_000_000
        if row_step <= self._last_row_step:
            row_step = self._last_row_step + 1
        self._last_row_step = row_step
        return row_step

    def finish(self) -> None:
        import wandb

        wandb.finish()


class TensorboardBackend(TrackingBackend):
    def __init__(self) -> None:
        self._adapter = None

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        from .tensorboard_utils import _TensorboardAdapter

        self._adapter = _TensorboardAdapter(args)

    def log(self, metrics: dict[str, Any], step: int | None = None, *, step_key: str | None = None, **kwargs) -> None:
        if self._adapter is not None:
            # Strip the caller's exact step-key entry (e.g. "train/step",
            # "rollout/step") — tensorboard receives step as an explicit
            # argument instead. Matching by exact key rather than endswith
            # avoids dropping user metrics that happen to end in "/step".
            data = {k: v for k, v in metrics.items() if k != step_key}
            self._adapter.log(data=data, step=step)

    def finish(self) -> None:
        if self._adapter is not None:
            self._adapter.finish()


class MlflowBackend(TrackingBackend):

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        from . import mlflow_utils

        mlflow_utils.init_mlflow(args, primary=primary, **kwargs)

    def log(self, metrics: dict[str, Any], step: int | None = None, **kwargs) -> None:
        from . import mlflow_utils

        mlflow_utils.log_metrics(metrics, step=step)

    def finish(self) -> None:
        from . import mlflow_utils

        mlflow_utils.finish()


class PrometheusBackend(TrackingBackend):
    # Wraps the existing Ray-actor based prometheus collector. The actor lifetime is
    # tied to the Ray job, so finish() is intentionally a no-op.

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        from .prometheus_utils import init_prometheus

        init_prometheus(args, start_server=primary)

    def log(self, metrics: dict[str, Any], step: int | None = None, **kwargs) -> None:
        from .prometheus_utils import get_prometheus

        prom = get_prometheus()
        assert prom is not None, (
            "Prometheus collector is not initialized; ensure init_tracking(..., primary=...) ran on the "
            "driver and workers can resolve the miles_prometheus_collector Ray actor."
        )
        prom.update.remote(metrics)

    def finish(self) -> None:
        return


# Registry that maps backend name → (class, args-flag attribute)

BACKEND_REGISTRY: dict[str, tuple[type[TrackingBackend], str]] = {
    "wandb": (WandbBackend, "use_wandb"),
    "tensorboard": (TensorboardBackend, "use_tensorboard"),
    "mlflow": (MlflowBackend, "use_mlflow"),
    "prometheus": (PrometheusBackend, "use_prometheus"),
}


class TrackingManager:
    # Initializes and logs to every enabled backend; used internally by ``tracking_utils``.

    def __init__(self) -> None:
        self._backends: list[TrackingBackend] = []
        self._tracking_requested = False
        self._warned_no_backends = False

    def init(self, args, *, primary: bool = True, **kwargs) -> None:
        for name, (cls, flag) in BACKEND_REGISTRY.items():
            if getattr(args, flag, False):
                self._tracking_requested = True
                logger.info("Initialising tracking backend: %s", name)
                backend = cls()
                if backend.init(args, primary=primary, **kwargs) is False:
                    logger.warning(
                        "Tracking backend %s did not initialize; metrics for this process will be skipped", name
                    )
                    continue
                self._backends.append(backend)

    def log(self, metrics: dict[str, Any], step: int | None = None, step_key: str | None = None) -> None:
        if not self._backends:
            if self._tracking_requested and not self._warned_no_backends:
                logger.warning("Dropping tracking metrics because no backend is initialized: %s", sorted(metrics)[:8])
                self._warned_no_backends = True
            return
        for backend in self._backends:
            backend.log(metrics, step=step, step_key=step_key)

    def finish(self) -> None:
        for backend in self._backends:
            try:
                backend.finish()
            except Exception:
                logger.exception(
                    "Error finishing tracking backend %s",
                    type(backend).__name__,
                )
        self._backends.clear()
