from collections.abc import Iterator

from miles.utils.types import Sample


def iter_group_samples(group) -> Iterator[Sample]:
    """Yield every sample leaf from a possibly nested rollout group."""
    for item in group:
        if isinstance(item, list):
            yield from iter_group_samples(item)
        elif isinstance(item, Sample):
            yield item
        else:
            raise TypeError(f"rollout group contains unsupported item {type(item).__name__}")


def group_has_aborted_sample(group) -> bool:
    """Return whether any leaf marks the generation attempt as aborted."""
    return any(sample.status == Sample.Status.ABORTED for sample in iter_group_samples(group))


def prepare_partial_retry_group(group: list[Sample], rollout_id: int) -> list[Sample]:
    """Validate and annotate a flat partial group for a later continuation."""
    if not group or any(not isinstance(sample, Sample) for sample in group):
        raise ValueError("partial rollout retry requires a non-empty, flat prompt group")
    for sample in group:
        if sample.response and "start_rollout_id" not in sample.metadata:
            sample.metadata["start_rollout_id"] = rollout_id
    return group
