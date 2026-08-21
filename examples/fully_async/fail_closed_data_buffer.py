"""Fail-closed staleness guard for the fully-async data buffer.

``DefaultDataBuffer`` treats unobservable staleness as admissible: a group whose
samples carry no ``oldest_weight_version`` (or a ``get()`` without a trainer
weight version) skips the ``--max-weight-staleness`` filter entirely. This fork
already lost a 15-hour run to exactly that failure mode — multi-turn generation
silently never recorded weight versions, so the filter recycled nothing — and
therefore requires the filter to be fail-closed: when a staleness bound is
configured, staleness must be observable for every trained group.
"""

from miles.rollout.fully_async_data_buffer import DataBufferInput, DefaultDataBuffer, iter_samples


class FailClosedDataBuffer(DefaultDataBuffer):
    """DefaultDataBuffer that refuses to train on groups of unobservable staleness."""

    async def get(self, current_version: int | None = None, **context) -> DataBufferInput:
        max_staleness = self._args.max_weight_staleness
        if max_staleness is not None and current_version is None:
            raise RuntimeError(
                "max_weight_staleness is configured but the trainer provided no current weight "
                "version; refusing to admit groups the staleness filter cannot observe"
            )
        entry = await super().get(current_version=current_version, **context)
        if max_staleness is not None:
            missing = sum(1 for s in iter_samples(entry.group) if s.oldest_weight_version is None)
            if missing:
                raise RuntimeError(
                    f"{missing} sample(s) in a completed group report no rollout weight version; "
                    "the max_weight_staleness filter would run fail-open on them"
                )
        return entry
