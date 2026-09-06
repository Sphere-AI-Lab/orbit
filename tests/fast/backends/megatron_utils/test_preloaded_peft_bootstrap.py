"""Fresh PEFT bootstrap must not reload the base after adapters are attached."""

from argparse import Namespace
from unittest.mock import Mock

import pytest
import torch
from megatron.core.transformer.module import Float16Module

from orbit.backends.megatron_utils import checkpoint, low_precision_bootstrap


@pytest.fixture
def bootstrap(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "iter_0000000"
    checkpoint_dir.mkdir()
    (checkpoint_dir / ".metadata").write_text("{}")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("0")
    args = Namespace(
        load=str(tmp_path),
        finetune=True,
        peft_method="oft",
        megatron_to_hf_mode="bridge",
        fp16=False,
        bf16=True,
        load_main_params_from_ckpt=False,
        start_rollout_id=None,
    )
    chunks = []
    for value in (2.0, 3.0):
        chunk = torch.nn.Module()
        chunk.register_parameter("weight", torch.nn.Parameter(torch.full((2, 2), value), requires_grad=False))
        chunk.adapter = torch.nn.Module()
        chunk.adapter.register_parameter("oft_r", torch.nn.Parameter(torch.zeros(2, 2)))
        chunks.append(chunk)
    low_precision_bootstrap._mark_dist_checkpoint_as_loaded(chunks, checkpoint_dir)
    config = Namespace(fp16=False, bf16=True, virtual_pipeline_model_parallel_size=None)
    model = [Float16Module(config, chunk) for chunk in chunks]
    legacy_loader = Mock(return_value=(17, 123))
    hf_loader = Mock(return_value=(0, 0))
    disk_load = Mock(side_effect=AssertionError("Unexpected second base model load"))
    monkeypatch.setattr(checkpoint, "get_args", lambda: args)
    monkeypatch.setattr(checkpoint, "_load_checkpoint_megatron", legacy_loader)
    monkeypatch.setattr(checkpoint, "_load_checkpoint_hf", hf_loader)
    monkeypatch.setattr(low_precision_bootstrap, "_restore_modelopt_state_before_load", disk_load)
    return Namespace(
        args=args,
        chunks=chunks,
        model=model,
        optimizer=Mock(),
        scheduler=object(),
        checkpoint_dir=checkpoint_dir,
        legacy_loader=legacy_loader,
        hf_loader=hf_loader,
        disk_load=disk_load,
    )


@pytest.mark.parametrize(
    ("peft_method", "direct_iteration_path", "start_rollout_id"),
    [("oft", False, None), ("lora", True, None), ("oft", False, 7)],
)
def test_fresh_preloaded_peft_preserves_base_and_refreshes_optimizer(
    bootstrap, peft_method, direct_iteration_path, start_rollout_id
):
    bootstrap.args.peft_method = peft_method
    bootstrap.args.start_rollout_id = start_rollout_id
    if direct_iteration_path:
        bootstrap.args.load = str(bootstrap.checkpoint_dir)
    parameters = [parameter for chunk in bootstrap.chunks for parameter in chunk.parameters()]
    original_values = [parameter.detach().clone() for parameter in parameters]

    result = checkpoint.load_checkpoint(bootstrap.model, bootstrap.optimizer, bootstrap.scheduler, {}, False)

    bootstrap.legacy_loader.assert_not_called()
    bootstrap.hf_loader.assert_not_called()
    bootstrap.disk_load.assert_not_called()
    bootstrap.optimizer.reload_model_params.assert_called_once_with()
    assert result == (0, 0)
    assert checkpoint.resolve_start_rollout_id_after_load(bootstrap.args, result[0]) == (
        0 if start_rollout_id is None else start_rollout_id
    )
    for parameter, original in zip(parameters, original_values, strict=True):
        assert torch.equal(parameter, original)
    assert all(not chunk.weight.requires_grad and chunk.adapter.oft_r.requires_grad for chunk in bootstrap.chunks)


@pytest.mark.parametrize(
    "condition",
    ["resume", "default", "wrong_source", "partial_marker", "wrong_prefix", "non_peft", "in_memory"],
)
def test_other_checkpoint_loads_keep_legacy_dispatch(bootstrap, condition, tmp_path):
    context = {}
    if condition == "resume":
        bootstrap.args.finetune = False
    elif condition == "default":
        del bootstrap.args.finetune
    elif condition == "wrong_source":
        low_precision_bootstrap._mark_dist_checkpoint_as_loaded(bootstrap.chunks, tmp_path / "another_source")
    elif condition == "partial_marker":
        del bootstrap.chunks[1]._orbit_loaded_dist_checkpoint_path
    elif condition == "wrong_prefix":
        low_precision_bootstrap._mark_dist_checkpoint_as_loaded(bootstrap.chunks, bootstrap.checkpoint_dir, "teacher.")
    elif condition == "non_peft":
        bootstrap.args.peft_method = "none"
    elif condition == "in_memory":
        context = {"local_checkpoint_manager": object()}
        bootstrap.args.load = str(tmp_path / "no_disk_checkpoint")

    result = checkpoint.load_checkpoint(bootstrap.model, bootstrap.optimizer, bootstrap.scheduler, context, True)

    assert result == (17, 123)
    bootstrap.legacy_loader.assert_called_once_with(
        ddp_model=bootstrap.model,
        optimizer=bootstrap.optimizer,
        opt_param_scheduler=bootstrap.scheduler,
        checkpointing_context=context,
        skip_load_to_model_and_opt=True,
    )
    bootstrap.hf_loader.assert_not_called()
    bootstrap.disk_load.assert_not_called()
    bootstrap.optimizer.reload_model_params.assert_not_called()
    assert bootstrap.args.start_rollout_id is None


def test_preloaded_finetune_with_lora_checkpoint_preserves_resume_iteration(bootstrap, monkeypatch):
    bootstrap.args.peft_method = "lora"
    bootstrap.args.lora_adapter_path = "/adapter/checkpoint"
    bootstrap.legacy_loader.return_value = (0, 0)
    adapter_loader = Mock(return_value=(True, 7))
    monkeypatch.setattr(checkpoint, "load_lora_adapter", adapter_loader)

    result = checkpoint.load_checkpoint(bootstrap.model, bootstrap.optimizer, bootstrap.scheduler, {}, False)

    assert result == (7, 0)
    assert checkpoint.resolve_start_rollout_id_after_load(bootstrap.args, result[0]) == 8
    bootstrap.legacy_loader.assert_called_once()
    adapter_loader.assert_called_once_with(
        bootstrap.model,
        bootstrap.args.lora_adapter_path,
        optimizer=bootstrap.optimizer,
        opt_param_scheduler=bootstrap.scheduler,
    )
    bootstrap.disk_load.assert_not_called()
    bootstrap.optimizer.reload_model_params.assert_not_called()


@pytest.mark.parametrize("adapter_path_name", ["oft_adapter_path", "peft_adapter_path"])
def test_other_explicit_adapter_paths_keep_legacy_dispatch(bootstrap, monkeypatch, adapter_path_name):
    setattr(bootstrap.args, adapter_path_name, "/adapter/checkpoint")
    # Keep the existing adapter-loader contract outside this dispatch test.
    monkeypatch.setattr(checkpoint, "load_peft_adapter", Mock(return_value=(False, None)))

    result = checkpoint.load_checkpoint(bootstrap.model, bootstrap.optimizer, bootstrap.scheduler, {}, False)

    assert result == (17, 123)
    bootstrap.legacy_loader.assert_called_once()
    bootstrap.disk_load.assert_not_called()
    bootstrap.optimizer.reload_model_params.assert_not_called()
    assert bootstrap.args.start_rollout_id is None


@pytest.mark.parametrize("ckpt_step", [0, 7])
def test_explicit_checkpoint_step_does_not_reuse_latest_preload(bootstrap, tmp_path, ckpt_step):
    selected_dir = tmp_path / f"iter_{ckpt_step:07d}"
    selected_dir.mkdir(exist_ok=True)
    (selected_dir / ".metadata").write_text("{}")
    latest_dir = tmp_path / "iter_0000017"
    latest_dir.mkdir()
    (latest_dir / ".metadata").write_text("{}")
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("17")
    low_precision_bootstrap._mark_dist_checkpoint_as_loaded(bootstrap.chunks, latest_dir)
    bootstrap.args.ckpt_step = ckpt_step

    result = checkpoint.load_checkpoint(bootstrap.model, bootstrap.optimizer, bootstrap.scheduler, {}, False)

    bootstrap.legacy_loader.assert_called_once_with(
        ddp_model=bootstrap.model,
        optimizer=bootstrap.optimizer,
        opt_param_scheduler=bootstrap.scheduler,
        checkpointing_context={},
        skip_load_to_model_and_opt=False,
    )
    assert result == (17, 123)
    assert bootstrap.args.ckpt_step == ckpt_step
    assert bootstrap.args.start_rollout_id is None
    bootstrap.disk_load.assert_not_called()
    bootstrap.optimizer.reload_model_params.assert_not_called()


def test_hf_bootstrap_keeps_existing_dispatch(bootstrap, tmp_path):
    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}")
    bootstrap.args.load = str(hf_dir)

    result = checkpoint.load_checkpoint(bootstrap.model, bootstrap.optimizer, bootstrap.scheduler, {}, False)

    assert result == (0, 0)
    bootstrap.hf_loader.assert_called_once_with(
        ddp_model=bootstrap.model,
        optimizer=bootstrap.optimizer,
        args=bootstrap.args,
        load_path=str(hf_dir),
    )
    bootstrap.legacy_loader.assert_not_called()
    bootstrap.disk_load.assert_not_called()
    assert bootstrap.args.start_rollout_id == 0
