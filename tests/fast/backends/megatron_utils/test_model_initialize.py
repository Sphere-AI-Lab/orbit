import sys
import types
from argparse import Namespace
from contextlib import ExitStack
from unittest.mock import MagicMock, patch

import pytest


def _stub_module(name: str, attrs: dict[str, object] | None = None, is_package: bool = False) -> types.ModuleType:
    module = types.ModuleType(name)
    if is_package:
        module.__path__ = []
    if attrs is not None:
        for attr_name, value in attrs.items():
            setattr(module, attr_name, value)
    sys.modules[name] = module
    return module


class _DummyDDP:
    pass


class _DummyModel:
    pass


class _DummyOptimizer:
    pass


class _DummyChainedOptimizer:
    pass


class _DummyDistributedOptimizer:
    pass


class _DummyScheduler:
    pass


class _DummyOptimizerConfig:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _FakeModelChunk:
    role: str | None = None


@pytest.fixture(scope="module", autouse=True)
def _mock_megatron_environment():
    original_modules = dict(sys.modules)
    try:
        _stub_module("megatron", is_package=True)
        core_module = _stub_module("megatron.core", is_package=True)
        core_module.mpu = types.SimpleNamespace()
        core_module.tensor_parallel = types.SimpleNamespace(model_parallel_cuda_manual_seed=MagicMock())
        _stub_module(
            "megatron.core.distributed",
            {
                "DistributedDataParallel": _DummyDDP,
                "finalize_model_grads": MagicMock(),
            },
        )
        _stub_module(
            "megatron.core.enums",
            {"ModelType": types.SimpleNamespace(encoder_or_decoder="encoder_or_decoder")},
        )
        _stub_module("megatron.core.models", is_package=True)
        _stub_module("megatron.core.models.gpt", {"GPTModel": _DummyModel})
        _stub_module(
            "megatron.core.optimizer",
            {
                "OptimizerConfig": _DummyOptimizerConfig,
                "get_megatron_optimizer": MagicMock(),
            },
            is_package=True,
        )
        _stub_module("megatron.core.optimizer.muon", {"get_megatron_muon_optimizer": MagicMock()})
        _stub_module("megatron.core.optimizer.distrib_optimizer", {"DistributedOptimizer": _DummyDistributedOptimizer})
        _stub_module(
            "megatron.core.optimizer.optimizer",
            {
                "ChainedOptimizer": _DummyChainedOptimizer,
                "MegatronOptimizer": _DummyOptimizer,
            },
        )
        _stub_module("megatron.core.optimizer_param_scheduler", {"OptimizerParamScheduler": _DummyScheduler})
        _stub_module("megatron.core.packed_seq_params", {"PackedSeqParams": MagicMock()})
        _stub_module("megatron.core.pipeline_parallel", {"get_forward_backward_func": MagicMock()})
        _stub_module("megatron.core.transformer", is_package=True)
        _stub_module("megatron.core.transformer.utils", {"sharded_state_dict_default": MagicMock()})
        _stub_module("megatron.core.utils", {"get_model_config": MagicMock()})
        _stub_module("megatron.core.config", {"set_experimental_flag": MagicMock()})
        _stub_module("megatron.core.num_microbatches_calculator", {"init_num_microbatches_calculator": MagicMock()})
        _stub_module("megatron.training", is_package=True)
        _stub_module(
            "megatron.training.global_vars",
            {
                "get_args": MagicMock(),
                "_build_tokenizer": MagicMock(),
                "set_args": MagicMock(),
            },
        )
        _stub_module("megatron.training.training", {"get_model": MagicMock()})
        _stub_module(
            "megatron.training.checkpointing",
            {
                "load_checkpoint": MagicMock(),
                "save_checkpoint": MagicMock(),
            },
        )
        _stub_module("sglang.srt.debug_utils", is_package=True)
        _stub_module(
            "sglang.srt.debug_utils.dumper",
            {
                "DumperConfig": MagicMock(),
                "_get_rank": MagicMock(return_value=0),
                "dumper": MagicMock(),
            },
        )
        _stub_module(
            "miles.backends.megatron_utils.bridge_lora_helpers",
            {
                "_ensure_model_list": MagicMock(),
                "_setup_lora_model_via_bridge": MagicMock(),
            },
        )
        _stub_module("miles.backends.megatron_utils.model_provider", {"get_model_provider_func": MagicMock()})
        yield
    finally:
        sys.modules.clear()
        sys.modules.update(original_modules)


def _patch_initialize_side_effects(stack: ExitStack) -> None:
    stack.enter_context(patch("miles.backends.megatron_utils.model.clear_memory"))
    stack.enter_context(patch("miles.backends.megatron_utils.model.check_peak_gpu_memory_after_load"))
    stack.enter_context(patch("miles.backends.megatron_utils.model.check_model_hashes"))


def test_initialize_does_not_step_scheduler_restored_from_checkpoint():
    from miles.backends.megatron_utils.model import initialize_model_and_optimizer

    args = Namespace(use_checkpoint_opt_param_scheduler=True, global_batch_size=8)
    model = [_FakeModelChunk()]
    optimizer = object()
    opt_param_scheduler = MagicMock()

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "miles.backends.megatron_utils.model.setup_model_and_optimizer",
                return_value=(model, optimizer, opt_param_scheduler),
            )
        )
        stack.enter_context(patch("miles.backends.megatron_utils.model.load_checkpoint", return_value=(100, 0)))
        _patch_initialize_side_effects(stack)
        result = initialize_model_and_optimizer(args)

    assert result == (model, optimizer, opt_param_scheduler, 100)
    opt_param_scheduler.step.assert_not_called()


def test_initialize_steps_scheduler_when_checkpoint_did_not_restore_it():
    from miles.backends.megatron_utils.model import initialize_model_and_optimizer

    args = Namespace(use_checkpoint_opt_param_scheduler=False, global_batch_size=8)
    model = [_FakeModelChunk()]
    optimizer = object()
    opt_param_scheduler = MagicMock()

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "miles.backends.megatron_utils.model.setup_model_and_optimizer",
                return_value=(model, optimizer, opt_param_scheduler),
            )
        )
        stack.enter_context(patch("miles.backends.megatron_utils.model.load_checkpoint", return_value=(100, 0)))
        _patch_initialize_side_effects(stack)
        result = initialize_model_and_optimizer(args)

    assert result == (model, optimizer, opt_param_scheduler, 100)
    opt_param_scheduler.step.assert_called_once_with(increment=800)


@pytest.mark.parametrize("tracker", [None, "release"], ids=["hf-base", "release-base"])
@pytest.mark.parametrize(
    ("adapter_result", "expected_start"),
    [
        ((True, None), 0),
        ((False, None), 0),
        ((True, 0), 1),
        ((True, 7), 8),
    ],
    ids=["weights-only", "load-failed", "iteration-zero", "iteration-seven"],
)
def test_bridge_bootstrap_start_id_uses_actual_adapter_state(tmp_path, tracker, adapter_result, expected_start):
    from miles.backends.megatron_utils import checkpoint as checkpoint_module

    base = tmp_path / "base"
    base.mkdir()
    if tracker is None:
        (base / "config.json").write_text("{}")
        base_loader = "_load_checkpoint_hf"
    else:
        (base / "latest_checkpointed_iteration.txt").write_text(tracker)
        base_loader = "_load_checkpoint_megatron"

    args = Namespace(
        load=str(base),
        lora_adapter_path=str(tmp_path / "adapter"),
        megatron_to_hf_mode="bridge",
        start_rollout_id=None,
    )
    with (
        patch.object(checkpoint_module, "get_args", return_value=args),
        patch.object(checkpoint_module, "is_lora_enabled", return_value=True),
        patch.object(checkpoint_module, "load_lora_adapter", return_value=adapter_result),
        patch.object(checkpoint_module, base_loader, return_value=(0, 0)),
    ):
        loaded_iteration, _ = checkpoint_module.load_checkpoint([], object(), object(), None, False)

    assert checkpoint_module.resolve_start_rollout_id_after_load(args, loaded_iteration) == expected_start


def test_numeric_bridge_checkpoint_uses_loaded_iteration_for_weights_only_adapter(tmp_path):
    from miles.backends.megatron_utils import checkpoint as checkpoint_module

    base = tmp_path / "base"
    base.mkdir()
    (base / "latest_checkpointed_iteration.txt").write_text("17")
    args = Namespace(
        load=str(base),
        lora_adapter_path=str(tmp_path / "adapter"),
        megatron_to_hf_mode="bridge",
        start_rollout_id=None,
    )
    with (
        patch.object(checkpoint_module, "get_args", return_value=args),
        patch.object(checkpoint_module, "is_lora_enabled", return_value=True),
        patch.object(checkpoint_module, "load_lora_adapter", return_value=(True, None)),
        # The tracker says 17, but --ckpt-step can make Megatron actually load 7.
        patch.object(checkpoint_module, "_load_checkpoint_megatron", return_value=(7, 0)),
    ):
        loaded_iteration, _ = checkpoint_module.load_checkpoint([], object(), object(), None, False)

    assert checkpoint_module.resolve_start_rollout_id_after_load(args, loaded_iteration) == 8
