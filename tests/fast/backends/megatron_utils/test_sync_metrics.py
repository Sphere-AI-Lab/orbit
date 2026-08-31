"""Regression coverage for the weight-sync payload tracker used by PEFT transports."""

import pytest
import torch

from tests.ci.ci_register import register_cpu_ci


register_cpu_ci(est_time=10, suite="stage-a-fast")


def test_payload_tracker_is_available_and_counts_the_wire_payload():
    """Removing sync_metrics must fail before any PEFT transport can be built."""
    try:
        from miles.backends.megatron_utils.update_weight.sync_metrics import WeightSyncPayloadTracker
    except ModuleNotFoundError as exc:
        pytest.fail(f"PEFT transports require the weight-sync metrics module: {exc}")

    tracker = WeightSyncPayloadTracker()
    tracker.record(
        [
            ("adapter_a", torch.zeros(4, dtype=torch.float32)),
            ("adapter_b", torch.zeros(3, dtype=torch.bfloat16)),
        ]
    )

    assert tracker.payload_bytes == 22
    assert tracker.num_tensors == 2
    assert tracker.num_records == 1
