from argparse import Namespace
import asyncio
from types import SimpleNamespace

import pytest
import torch

from orbit.backends.megatron_utils.peft_transport.backends import nccl
from orbit.backends.megatron_utils.peft_transport.registry import PeftMethodSpec
from orbit.backends.megatron_utils.peft_utils import PeftSyncSpec


class Remote:
    def __init__(self, fn):
        self.remote = fn


@pytest.mark.parametrize("fully_async", [True, False])
def test_nccl_stage_precedes_short_activation_pause(monkeypatch, fully_async):
    events = []

    def record(name, result):
        def call(*args, **kwargs):
            events.append(name)
            return result

        return Remote(call)

    engine = SimpleNamespace(
        update_adapter_from_distributed=record("stage", {"success": True, "staged_adapter_version": "7"}),
        activate_adapter_version=record("activate", {"success": True, "active_adapter_version": "7"}),
        pause_generation=record("pause", None),
        continue_generation=record("resume", None),
    )
    backend = nccl.NcclBackend(
        args=Namespace(
            peft_method="oft",
            adapter_double_buffer=True,
            fully_async=fully_async,
            peft_distributed_transport="nccl",
            pause_generation_mode="in_place",
        ),
        method_spec=PeftMethodSpec(
            name="oft",
            sglang_load_format="oft_adapter",
            weight_name_predicate=lambda n: True,
            dedupe_by_storage=False,
            payload_shaper=None,
            sample_names="oft_R",
            label="OFT",
        ),
        sync_spec=PeftSyncSpec(
            method="oft", adapter_name="orbit_oft", adapter_config={}, sync_transport="oft_adapter"
        ),
    )
    backend._engines = [engine]
    backend._lock = SimpleNamespace(acquire=Remote(lambda: True), release=Remote(lambda: True))
    monkeypatch.setattr(nccl.ray, "get", lambda refs: refs)
    monkeypatch.setattr(nccl.dist, "broadcast", lambda *a, **kw: SimpleNamespace(wait=lambda: None))
    backend.send_adapter([("layer.oft_R", torch.ones(1))], weight_version=7)
    assert events == (["stage", "pause", "activate", "resume"] if fully_async else ["stage", "activate"])


@pytest.mark.parametrize("double_buffer", [True, False])
def test_driver_preserves_next_batch_without_waiting_before_overlap_push(monkeypatch, double_buffer):
    import train_async as driver

    events = []

    class BatchFuture:
        def __init__(self, step):
            self.step = step

        def __await__(self):
            async def get():
                if self.step == 1:
                    assert ("push0" in events) is double_buffer
                events.append(f"batch{self.step}")
                return self.step

            return get().__await__()

    class Actor:
        async def update_weights(self, rollout_id=None):
            events.append(f"push{rollout_id}")

        async def train(self, rollout_id, data):
            assert data == rollout_id, "pending batch must be consumed exactly once"
            events.append(f"train{rollout_id}")

    async def done(*args, **kwargs):
        return None

    actor = Actor()
    manager = SimpleNamespace(generate=Remote(BatchFuture), dispose=Remote(done))

    async def models(*args):
        return actor, None

    for name in (
        "configure_logger",
        "maybe_start_periodic_pyspy_dump",
        "init_tracking",
        "maybe_start_mini_ft_controller",
        "set_progress",
        "write_train_status",
        "remove_rollout_data_refs",
        "validate_async_off_policy_correction",
    ):
        monkeypatch.setattr(driver, name, lambda *a, **kw: None)
    monkeypatch.setattr(driver.object_store, "init_instance", lambda *a, **kw: None)
    monkeypatch.setattr(driver, "create_placement_groups", lambda args: {"rollout": None})
    monkeypatch.setattr(driver, "create_rollout_manager", lambda *a: (manager, 2))
    monkeypatch.setattr(driver, "create_training_models", models)
    monkeypatch.setattr(driver, "EvalDispatcher", lambda *a: SimpleNamespace(drain=done))
    monkeypatch.setattr(driver, "should_run_periodic_action", lambda *a: False)
    args = Namespace(
        colocate=False,
        fully_async=True,
        peft_method="oft",
        adapter_double_buffer=double_buffer,
        peft_distributed_transport="nccl",
        pause_generation_mode="in_place",
        control_server_port=None,
        check_weight_update_equal=False,
        eval_interval=None,
        start_rollout_id=0,
        num_rollout=2,
        offload_train=False,
        use_critic=False,
        save_trigger_sentinel=None,
        save_interval=None,
        update_weights_interval=1,
        debug_exit_after_rollout=None,
    )
    asyncio.run(driver.train(args))
    assert events.count("batch1") == 1
    assert events.count("train1") == 1
