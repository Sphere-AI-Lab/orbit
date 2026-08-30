import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    days, remainder = divmod(total_seconds, 24 * 60 * 60)
    hours, remainder = divmod(remainder, 60 * 60)
    minutes, seconds = divmod(remainder, 60)
    formatted = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    if days:
        return f"{days}d {formatted}"
    return formatted


@dataclass(frozen=True)
class TrainingETAReport:
    rollout_id: int
    start_rollout_id: int
    num_rollout: int
    rollouts_total: int
    rollouts_completed: int
    rollouts_remaining: int
    elapsed_seconds: float
    last_rollout_seconds: float
    avg_rollout_seconds: float
    eta_seconds: float
    eta_wall_time: float

    def to_metrics(self, step: int) -> dict[str, float | int]:
        return {
            "rollout/step": step,
            "progress/rollout_seconds": self.last_rollout_seconds,
            "progress/avg_rollout_seconds": self.avg_rollout_seconds,
            "progress/elapsed_seconds": self.elapsed_seconds,
            "progress/eta_seconds": self.eta_seconds,
            "progress/rollouts_completed": self.rollouts_completed,
            "progress/rollouts_remaining": self.rollouts_remaining,
            "progress/rollouts_total": self.rollouts_total,
        }

    def to_log_message(self) -> str:
        last_rollout_id = max(self.start_rollout_id, self.num_rollout - 1)
        eta_at = datetime.fromtimestamp(self.eta_wall_time).strftime("%Y-%m-%d %H:%M:%S")
        return (
            f"rollout={self.rollout_id}/{last_rollout_id} "
            f"completed={self.rollouts_completed}/{self.rollouts_total} "
            f"remaining={self.rollouts_remaining} "
            f"elapsed={format_duration(self.elapsed_seconds)} "
            f"last={format_duration(self.last_rollout_seconds)} "
            f"avg={format_duration(self.avg_rollout_seconds)} "
            f"eta_remaining={format_duration(self.eta_seconds)} "
            f"eta_at={eta_at}"
        )


class TrainingETA:
    def __init__(
        self,
        *,
        start_rollout_id: int,
        num_rollout: int,
        monotonic_fn: Callable[[], float] = time.monotonic,
        wall_time_fn: Callable[[], float] = time.time,
    ):
        self.start_rollout_id = start_rollout_id
        self.num_rollout = num_rollout
        self.rollouts_total = max(num_rollout - start_rollout_id, 0)
        self._monotonic_fn = monotonic_fn
        self._wall_time_fn = wall_time_fn
        self._started_at = monotonic_fn()
        self._rollout_started_at: float | None = None

    def mark_rollout_start(self, rollout_id: int) -> None:
        del rollout_id
        self._rollout_started_at = self._monotonic_fn()

    def mark_rollout_done(self, rollout_id: int) -> TrainingETAReport:
        now = self._monotonic_fn()
        wall_now = self._wall_time_fn()
        completed = max(0, min(rollout_id - self.start_rollout_id + 1, self.rollouts_total))
        remaining = max(self.rollouts_total - completed, 0)
        elapsed = max(0.0, now - self._started_at)
        rollout_started_at = self._rollout_started_at if self._rollout_started_at is not None else self._started_at
        last_rollout_seconds = max(0.0, now - rollout_started_at)
        avg_rollout_seconds = elapsed / completed if completed else 0.0
        eta_seconds = avg_rollout_seconds * remaining
        self._rollout_started_at = None
        return TrainingETAReport(
            rollout_id=rollout_id,
            start_rollout_id=self.start_rollout_id,
            num_rollout=self.num_rollout,
            rollouts_total=self.rollouts_total,
            rollouts_completed=completed,
            rollouts_remaining=remaining,
            elapsed_seconds=elapsed,
            last_rollout_seconds=last_rollout_seconds,
            avg_rollout_seconds=avg_rollout_seconds,
            eta_seconds=eta_seconds,
            eta_wall_time=wall_now + eta_seconds,
        )
