import json
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from orbit.backends.megatron_utils import checkpoint as checkpoint_module
from orbit.backends.megatron_utils import model as model_module


def _make_distributed_checkpoint(tmp_path, *, iteration: int, common_state: dict) -> tuple:
    checkpoint_root = tmp_path / "checkpoint"
    checkpoint_dir = checkpoint_root / f"iter_{iteration:07d}"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text(str(iteration))
    (checkpoint_dir / ".metadata").write_bytes(b"test metadata")
    torch.save(common_state, checkpoint_dir / "common.pt")
    return checkpoint_root, checkpoint_dir


def _args(load_path) -> Namespace:
    return Namespace(
        load=str(load_path),
        ckpt_step=None,
        megatron_to_hf_mode=None,
    )


def _model(role: str):
    return [SimpleNamespace(role=role)]


def _load_with_spies(
    monkeypatch,
    args,
    *,
    role: str,
    is_value_model: bool = False,
    load_training_state: bool = True,
):
    calls = []

    def full_loader(**kwargs):
        observed_kwargs = dict(kwargs)
        observed_kwargs["_observed_load"] = args.load
        observed_kwargs["_observed_ckpt_step"] = args.ckpt_step
        observed_kwargs["_observed_ckpt_step_truthy"] = bool(args.ckpt_step)
        observed_root = checkpoint_module.Path(args.load)
        tracker_path = observed_root / checkpoint_module._MEGATRON_TRACKER_FILE
        observed_kwargs["_observed_tracker"] = tracker_path.read_text().strip()
        selected_path = observed_root / f"iter_{int(args.ckpt_step):07d}"
        observed_kwargs["_observed_selected_path"] = str(selected_path.resolve(strict=True))
        calls.append(("full", observed_kwargs))
        return 17, 23

    def model_only_loader(**kwargs):
        calls.append(("model_only", kwargs))
        return 0, 0

    monkeypatch.setattr(checkpoint_module, "get_args", lambda: args)
    monkeypatch.setattr(checkpoint_module, "is_peft_enabled", lambda _args: False)
    monkeypatch.setattr(checkpoint_module, "_load_checkpoint_megatron", full_loader)
    monkeypatch.setattr(checkpoint_module, "_load_checkpoint_dist", model_only_loader)
    result = checkpoint_module.load_checkpoint(
        _model(role),
        object(),
        object(),
        checkpointing_context={},
        skip_load_to_model_and_opt=False,
        is_value_model=is_value_model,
        load_training_state=load_training_state,
    )
    return result, calls


def test_marked_actor_checkpoint_routes_to_megatron_full_loader(tmp_path, monkeypatch):
    checkpoint_root, _ = _make_distributed_checkpoint(
        tmp_path,
        iteration=17,
        common_state={"checkpoint_version": 3.0, "iteration": 17},
    )
    checkpoint_module._write_orbit_training_checkpoint_marker(
        checkpoint_root,
        17,
        "actor",
        optimizer_state_saved=True,
        scheduler_state_saved=True,
    )

    args = _args(checkpoint_root)
    result, calls = _load_with_spies(monkeypatch, args, role="actor")

    assert result == (17, 23)
    assert [kind for kind, _ in calls] == ["full"]
    assert args._orbit_training_checkpoint_loaded is True
    assert args._orbit_optimizer_scheduler_state_restored is True


def test_converted_model_checkpoint_stays_on_model_only_loader(tmp_path, monkeypatch):
    checkpoint_root, _ = _make_distributed_checkpoint(
        tmp_path,
        iteration=0,
        common_state={"checkpoint_version": 3.0, "iteration": 0},
    )

    args = _args(checkpoint_root)
    result, calls = _load_with_spies(monkeypatch, args, role="actor")

    assert result == (0, 0)
    assert [kind for kind, _ in calls] == ["model_only"]
    assert args._orbit_training_checkpoint_loaded is False
    assert args._orbit_optimizer_scheduler_state_restored is False


def test_explicit_model_only_load_ignores_training_marker(tmp_path, monkeypatch):
    checkpoint_root, _ = _make_distributed_checkpoint(
        tmp_path,
        iteration=6,
        common_state={"checkpoint_version": 3.0, "iteration": 6},
    )
    checkpoint_module._write_orbit_training_checkpoint_marker(
        checkpoint_root,
        6,
        "actor",
        optimizer_state_saved=True,
        scheduler_state_saved=True,
    )

    args = _args(checkpoint_root)
    result, calls = _load_with_spies(
        monkeypatch,
        args,
        role="actor",
        load_training_state=False,
    )

    assert result == (0, 0)
    assert [kind for kind, _ in calls] == ["model_only"]
    assert args._orbit_training_checkpoint_loaded is False
    assert args._orbit_optimizer_scheduler_state_restored is False


@pytest.mark.parametrize(
    ("loaded_iteration", "load_training_state", "expected_training_resume"),
    [
        (0, True, False),  # Megatron release/base load
        (7, False, False),  # reference or other explicit model-only load
        (7, True, True),
    ],
)
def test_legacy_megatron_resume_flag_requires_training_iteration_intent(
    tmp_path,
    monkeypatch,
    loaded_iteration,
    load_training_state,
    expected_training_resume,
):
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("release" if loaded_iteration == 0 else "7")
    args = _args(tmp_path)
    calls = []
    monkeypatch.setattr(checkpoint_module, "get_args", lambda: args)
    monkeypatch.setattr(checkpoint_module, "is_distributed_checkpoint", lambda _path: False)
    monkeypatch.setattr(checkpoint_module, "_is_megatron_checkpoint", lambda _path: True)
    monkeypatch.setattr(
        checkpoint_module,
        "_load_checkpoint_megatron",
        lambda **kwargs: calls.append(kwargs) or (loaded_iteration, 0),
    )
    monkeypatch.setattr(checkpoint_module, "is_peft_enabled", lambda _args: False)

    result = checkpoint_module.load_checkpoint(
        _model("actor"),
        object(),
        object(),
        checkpointing_context={},
        skip_load_to_model_and_opt=False,
        load_training_state=load_training_state,
    )

    assert result == (loaded_iteration, 0)
    assert len(calls) == 1
    assert args._orbit_training_checkpoint_loaded is expected_training_resume
    assert args._orbit_optimizer_scheduler_state_restored is expected_training_resume


def test_scalar_head_model_only_critic_checkpoint_is_not_treated_as_resume(tmp_path, monkeypatch):
    checkpoint_root, _ = _make_distributed_checkpoint(
        tmp_path,
        iteration=9,
        common_state={
            "checkpoint_version": 3.0,
            "iteration": 9,
            # Model-only critic exports may retain Orbit args and a nonzero
            # iteration. Neither is evidence that optimizer/scheduler state exists.
            "args": Namespace(
                save=str(tmp_path / "unrelated"),
                critic_save=str(tmp_path / "checkpoint"),
            ),
        },
    )

    result, calls = _load_with_spies(
        monkeypatch,
        _args(checkpoint_root),
        role="critic",
        is_value_model=False,
    )

    assert result == (0, 0)
    assert [kind for kind, _ in calls] == ["model_only"]
    assert calls[0][1]["is_value_model"] is False


def test_legacy_actor_checkpoint_with_optimizer_and_scheduler_resumes(tmp_path, monkeypatch):
    checkpoint_root, _ = _make_distributed_checkpoint(
        tmp_path,
        iteration=11,
        common_state={
            "checkpoint_version": 3.0,
            "iteration": 11,
            "optimizer": {"state": "sharded"},
            "opt_param_scheduler": {"num_steps": 11},
            "args": Namespace(save=str(tmp_path / "checkpoint"), critic_save=str(tmp_path / "critic")),
        },
    )

    result, calls = _load_with_spies(monkeypatch, _args(checkpoint_root), role="actor")

    assert result == (17, 23)
    assert [kind for kind, _ in calls] == ["full"]


def test_legacy_sharded_optimizer_metadata_is_training_state(tmp_path, monkeypatch):
    checkpoint_root, _ = _make_distributed_checkpoint(
        tmp_path,
        iteration=12,
        common_state={
            "checkpoint_version": 3.0,
            "iteration": 12,
            # torch_dist may keep optimizer tensors entirely in sharded files.
            "opt_param_scheduler": {"num_steps": 12},
            "args": Namespace(save=str(tmp_path / "checkpoint"), critic_save=str(tmp_path / "critic")),
        },
    )
    from megatron.core.dist_checkpointing import serialization

    monkeypatch.setattr(
        serialization,
        "load_tensors_metadata",
        lambda _path: {
            "model.decoder.layers.0.weight": object(),
            "chained_0.optimizer.state.exp_avg.decoder.layers.0.weight": object(),
        },
    )

    result, calls = _load_with_spies(monkeypatch, _args(checkpoint_root), role="actor")

    assert result == (17, 23)
    assert [kind for kind, _ in calls] == ["full"]


def test_actor_training_checkpoint_is_model_only_for_critic_bootstrap(tmp_path, monkeypatch):
    checkpoint_root, _ = _make_distributed_checkpoint(
        tmp_path,
        iteration=13,
        common_state={
            "checkpoint_version": 3.0,
            "iteration": 13,
            "optimizer": {"state": "sharded"},
            "opt_param_scheduler": {"num_steps": 13},
        },
    )
    checkpoint_module._write_orbit_training_checkpoint_marker(
        checkpoint_root,
        13,
        "actor",
        optimizer_state_saved=True,
        scheduler_state_saved=True,
    )

    result, calls = _load_with_spies(monkeypatch, _args(checkpoint_root), role="critic")

    assert result == (0, 0)
    assert [kind for kind, _ in calls] == ["model_only"]


def test_marked_checkpoint_without_optimizer_fails_resume_explicitly(tmp_path, monkeypatch):
    checkpoint_root, _ = _make_distributed_checkpoint(
        tmp_path,
        iteration=4,
        common_state={"checkpoint_version": 3.0, "iteration": 4},
    )
    checkpoint_module._write_orbit_training_checkpoint_marker(
        checkpoint_root,
        4,
        "actor",
        optimizer_state_saved=False,
        scheduler_state_saved=False,
    )
    args = _args(checkpoint_root)
    args.no_load_optim = False
    args.finetune = False
    monkeypatch.setattr(checkpoint_module, "get_args", lambda: args)

    with pytest.raises(RuntimeError, match="saved without complete optimizer/scheduler state"):
        checkpoint_module.load_checkpoint(
            _model("actor"),
            object(),
            object(),
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
            load_training_state=True,
        )


def test_marked_checkpoint_without_optimizer_allows_explicit_model_warm_start(tmp_path, monkeypatch):
    checkpoint_root, _ = _make_distributed_checkpoint(
        tmp_path,
        iteration=4,
        common_state={"checkpoint_version": 3.0, "iteration": 4},
    )
    checkpoint_module._write_orbit_training_checkpoint_marker(
        checkpoint_root,
        4,
        "actor",
        optimizer_state_saved=False,
        scheduler_state_saved=False,
    )
    args = _args(checkpoint_root)
    args.no_load_optim = True
    args.finetune = False

    result, calls = _load_with_spies(monkeypatch, args, role="actor")

    assert result == (0, 0)
    assert [kind for kind, _ in calls] == ["model_only"]
    assert args._orbit_training_checkpoint_loaded is False
    assert args._orbit_optimizer_scheduler_state_restored is False


@pytest.mark.parametrize("warm_start_flag", ["no_load_optim", "finetune"])
def test_marked_complete_checkpoint_honors_explicit_model_only_warm_start(
    tmp_path,
    monkeypatch,
    warm_start_flag,
):
    checkpoint_root, _ = _make_distributed_checkpoint(
        tmp_path,
        iteration=10,
        common_state={"checkpoint_version": 3.0, "iteration": 10},
    )
    checkpoint_module._write_orbit_training_checkpoint_marker(
        checkpoint_root,
        10,
        "actor",
        optimizer_state_saved=True,
        scheduler_state_saved=True,
    )
    args = _args(checkpoint_root)
    setattr(args, warm_start_flag, True)

    result, calls = _load_with_spies(monkeypatch, args, role="actor")

    assert result == (0, 0)
    assert [kind for kind, _ in calls] == ["model_only"]
    assert args._orbit_training_checkpoint_loaded is False
    assert args._orbit_optimizer_scheduler_state_restored is False


def test_invalid_training_marker_fails_closed(tmp_path, monkeypatch):
    checkpoint_root, checkpoint_dir = _make_distributed_checkpoint(
        tmp_path,
        iteration=3,
        common_state={"checkpoint_version": 3.0, "iteration": 3},
    )
    marker_path = checkpoint_dir / checkpoint_module._ORBIT_TRAINING_CHECKPOINT_MARKER
    marker_path.write_text("{not-json")
    monkeypatch.setattr(checkpoint_module, "get_args", lambda: _args(checkpoint_root))

    with pytest.raises(RuntimeError, match="invalid Orbit training checkpoint marker"):
        checkpoint_module.load_checkpoint(
            _model("actor"),
            object(),
            object(),
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
            load_training_state=True,
        )


def test_save_wrapper_writes_atomic_role_marker(tmp_path, monkeypatch):
    args = Namespace(save=str(tmp_path), no_save_optim=False, async_save=True)
    calls = []

    def megatron_save(*save_args, **save_kwargs):
        calls.append((save_args, save_kwargs))
        return "saved"

    monkeypatch.setattr(checkpoint_module, "get_args", lambda: args)
    monkeypatch.setattr(checkpoint_module, "_save_checkpoint_megatron", megatron_save)

    optimizer = object()
    scheduler = object()
    result = checkpoint_module.save_checkpoint(5, _model("actor"), optimizer, scheduler, release=False)

    assert result == "saved"
    assert len(calls) == 1
    marker_path = tmp_path / "iter_0000005" / checkpoint_module._ORBIT_TRAINING_CHECKPOINT_MARKER
    marker = json.loads(marker_path.read_text())
    assert marker == {
        "format": "orbit.training_checkpoint",
        "version": 1,
        "iteration": 5,
        "role": "actor",
        "optimizer_state_saved": True,
        "scheduler_state_saved": True,
    }


def test_marker_helper_accepts_direct_iteration_directory(tmp_path):
    checkpoint_dir = tmp_path / "iter_0000005"

    marker_path = checkpoint_module._write_orbit_training_checkpoint_marker(
        checkpoint_dir,
        5,
        "actor",
        optimizer_state_saved=True,
        scheduler_state_saved=True,
    )

    assert marker_path.parent == checkpoint_dir
    assert not (checkpoint_dir / "iter_0000005").exists()


def test_direct_iteration_training_checkpoint_is_pinned_for_full_load(tmp_path, monkeypatch):
    checkpoint_root, checkpoint_dir = _make_distributed_checkpoint(
        tmp_path,
        iteration=17,
        common_state={"checkpoint_version": 3.0, "iteration": 17},
    )
    checkpoint_module._write_orbit_training_checkpoint_marker(
        checkpoint_dir,
        17,
        "actor",
        optimizer_state_saved=True,
        scheduler_state_saved=True,
    )
    args = _args(checkpoint_dir)

    result, calls = _load_with_spies(monkeypatch, args, role="actor")

    assert result == (17, 23)
    assert [kind for kind, _ in calls] == ["full"]
    assert calls[0][1]["_observed_load"] != str(checkpoint_root)
    assert calls[0][1]["_observed_tracker"] == "17"
    assert calls[0][1]["_observed_selected_path"] == str(checkpoint_dir.resolve())
    assert calls[0][1]["_observed_ckpt_step"] == 17
    assert args.load == str(checkpoint_dir)
    assert args.ckpt_step is None


def test_explicit_checkpoint_step_zero_is_selected_and_pinned(tmp_path, monkeypatch):
    checkpoint_root, checkpoint_dir = _make_distributed_checkpoint(
        tmp_path,
        iteration=0,
        common_state={"checkpoint_version": 3.0, "iteration": 0},
    )
    checkpoint_module._write_orbit_training_checkpoint_marker(
        checkpoint_root,
        0,
        "actor",
        optimizer_state_saved=True,
        scheduler_state_saved=True,
    )
    later_dir = checkpoint_root / "iter_0000008"
    later_dir.mkdir()
    (later_dir / ".metadata").write_bytes(b"later metadata")
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text("8")
    args = _args(checkpoint_root)
    args.ckpt_step = 0

    result, calls = _load_with_spies(monkeypatch, args, role="actor")

    assert result == (17, 23)
    assert [kind for kind, _ in calls] == ["full"]
    assert calls[0][1]["_observed_load"] != str(checkpoint_root)
    assert calls[0][1]["_observed_tracker"] == "0"
    assert calls[0][1]["_observed_selected_path"] == str(checkpoint_dir.resolve())
    assert calls[0][1]["_observed_ckpt_step"] == 0
    assert calls[0][1]["_observed_ckpt_step_truthy"] is True
    assert args.ckpt_step == 0
    assert type(args.ckpt_step) is int
    assert checkpoint_dir.is_dir()


def test_explicit_checkpoint_step_zero_pins_model_only_load(tmp_path, monkeypatch):
    checkpoint_root, checkpoint_dir = _make_distributed_checkpoint(
        tmp_path,
        iteration=0,
        common_state={"checkpoint_version": 3.0, "iteration": 0},
    )
    later_dir = checkpoint_root / "iter_0000008"
    later_dir.mkdir()
    (later_dir / ".metadata").write_bytes(b"later metadata")
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text("8")
    args = _args(checkpoint_root)
    args.ckpt_step = 0

    result, calls = _load_with_spies(monkeypatch, args, role="actor")

    assert result == (0, 0)
    assert [kind for kind, _ in calls] == ["model_only"]
    assert calls[0][1]["load_path"] == str(checkpoint_dir)


def test_unfinalized_async_marker_does_not_override_tracked_checkpoint(tmp_path, monkeypatch):
    checkpoint_root, _ = _make_distributed_checkpoint(
        tmp_path,
        iteration=5,
        common_state={"checkpoint_version": 3.0, "iteration": 5},
    )
    # An async save can write its Orbit marker before torch_dist finalizes
    # .metadata and advances the tracker. The incomplete directory is ignored.
    checkpoint_module._write_orbit_training_checkpoint_marker(
        checkpoint_root,
        6,
        "actor",
        optimizer_state_saved=True,
        scheduler_state_saved=True,
    )

    result, calls = _load_with_spies(monkeypatch, _args(checkpoint_root), role="actor")

    assert result == (0, 0)
    assert [kind for kind, _ in calls] == ["model_only"]


@pytest.mark.parametrize("selection", ["direct", "ckpt_step"])
def test_full_load_ignores_parent_release_tracker(tmp_path, monkeypatch, selection):
    checkpoint_root, checkpoint_dir = _make_distributed_checkpoint(
        tmp_path,
        iteration=17,
        common_state={"checkpoint_version": 3.0, "iteration": 17},
    )
    checkpoint_module._write_orbit_training_checkpoint_marker(
        checkpoint_root,
        17,
        "actor",
        optimizer_state_saved=True,
        scheduler_state_saved=True,
    )
    release_dir = checkpoint_root / "release"
    release_dir.mkdir()
    (release_dir / ".metadata").write_bytes(b"release metadata")
    (checkpoint_root / "latest_checkpointed_iteration.txt").write_text("release")
    args = _args(checkpoint_dir if selection == "direct" else checkpoint_root)
    if selection == "ckpt_step":
        args.ckpt_step = 17

    result, calls = _load_with_spies(monkeypatch, args, role="actor")

    assert result == (17, 23)
    assert [kind for kind, _ in calls] == ["full"]
    assert calls[0][1]["_observed_tracker"] == "17"
    assert calls[0][1]["_observed_selected_path"] == str(checkpoint_dir.resolve())
    assert args.load == str(checkpoint_dir if selection == "direct" else checkpoint_root)
    assert (checkpoint_root / "latest_checkpointed_iteration.txt").read_text() == "release"


@pytest.mark.parametrize(
    ("remnant", "use_alias"),
    [("marker", False), ("distcp", False), ("common", False), ("marker", True)],
)
def test_incomplete_direct_distributed_checkpoint_fails_closed(
    tmp_path,
    monkeypatch,
    remnant,
    use_alias,
):
    checkpoint_dir = tmp_path / "iter_0000003"
    checkpoint_dir.mkdir()
    if remnant == "marker":
        checkpoint_module._write_orbit_training_checkpoint_marker(
            checkpoint_dir,
            3,
            "actor",
            optimizer_state_saved=True,
            scheduler_state_saved=True,
        )
    else:
        remnant_name = "__0_0.distcp" if remnant == "distcp" else "common.pt"
        (checkpoint_dir / remnant_name).write_bytes(b"partial checkpoint state")

    load_path = checkpoint_dir
    if use_alias:
        load_path = tmp_path / "unfinished-actor"
        load_path.symlink_to(checkpoint_dir, target_is_directory=True)
    args = _args(load_path)
    monkeypatch.setattr(checkpoint_module, "get_args", lambda: args)

    with pytest.raises(RuntimeError, match=r"incomplete distributed checkpoint.*missing finalized \.metadata"):
        checkpoint_module.load_checkpoint(
            _model("actor"),
            object(),
            object(),
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
            load_training_state=True,
        )


def test_valid_legacy_direct_iteration_is_not_rejected_as_incomplete(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "iter_0000007"
    legacy_rank_dir = checkpoint_dir / "mp_rank_00"
    legacy_rank_dir.mkdir(parents=True)
    (legacy_rank_dir / "model_optim_rng.pt").write_bytes(b"legacy checkpoint")
    args = _args(checkpoint_dir)
    calls = []
    monkeypatch.setattr(checkpoint_module, "get_args", lambda: args)
    monkeypatch.setattr(checkpoint_module, "is_distributed_checkpoint", lambda _path: False)
    monkeypatch.setattr(checkpoint_module, "_is_megatron_checkpoint", lambda _path: True)
    monkeypatch.setattr(
        checkpoint_module,
        "_load_checkpoint_megatron",
        lambda **kwargs: calls.append(kwargs) or (7, 0),
    )
    monkeypatch.setattr(checkpoint_module, "is_peft_enabled", lambda _args: False)

    result = checkpoint_module.load_checkpoint(
        _model("actor"),
        object(),
        object(),
        checkpointing_context={},
        skip_load_to_model_and_opt=False,
        load_training_state=True,
    )

    assert result == (7, 0)
    assert len(calls) == 1


def test_symlink_alias_to_marked_iteration_resumes(tmp_path, monkeypatch):
    _, checkpoint_dir = _make_distributed_checkpoint(
        tmp_path,
        iteration=17,
        common_state={"checkpoint_version": 3.0, "iteration": 17},
    )
    checkpoint_module._write_orbit_training_checkpoint_marker(
        checkpoint_dir,
        17,
        "actor",
        optimizer_state_saved=True,
        scheduler_state_saved=True,
    )
    alias_path = tmp_path / "actor-latest"
    alias_path.symlink_to(checkpoint_dir, target_is_directory=True)
    args = _args(alias_path)

    result, calls = _load_with_spies(monkeypatch, args, role="actor")

    assert result == (17, 23)
    assert [kind for kind, _ in calls] == ["full"]
    assert calls[0][1]["_observed_tracker"] == "17"
    assert calls[0][1]["_observed_selected_path"] == str(checkpoint_dir.resolve())
    assert args.load == str(alias_path)


def test_failed_full_load_restores_args_and_removes_temporary_tracker(tmp_path, monkeypatch):
    checkpoint_root, checkpoint_dir = _make_distributed_checkpoint(
        tmp_path,
        iteration=17,
        common_state={"checkpoint_version": 3.0, "iteration": 17},
    )
    checkpoint_module._write_orbit_training_checkpoint_marker(
        checkpoint_root,
        17,
        "actor",
        optimizer_state_saved=True,
        scheduler_state_saved=True,
    )
    args = _args(checkpoint_dir)
    temporary_roots = []

    def failing_loader(**_kwargs):
        temporary_roots.append(checkpoint_module.Path(args.load))
        assert (temporary_roots[-1] / "latest_checkpointed_iteration.txt").read_text().strip() == "17"
        raise RuntimeError("synthetic load failure")

    monkeypatch.setattr(checkpoint_module, "get_args", lambda: args)
    monkeypatch.setattr(checkpoint_module, "is_peft_enabled", lambda _args: False)
    monkeypatch.setattr(checkpoint_module, "_load_checkpoint_megatron", failing_loader)

    with pytest.raises(RuntimeError, match="synthetic load failure"):
        checkpoint_module.load_checkpoint(
            _model("actor"),
            object(),
            object(),
            checkpointing_context={},
            skip_load_to_model_and_opt=False,
            load_training_state=True,
        )

    assert args.load == str(checkpoint_dir)
    assert args.ckpt_step is None
    assert len(temporary_roots) == 1
    assert not temporary_roots[0].exists()


def test_full_model_actor_resume_does_not_double_advance_restored_scheduler(monkeypatch):
    args = Namespace(
        load="/checkpoint",
        global_batch_size=8,
        fp16=False,
        bf16=False,
    )
    model = [SimpleNamespace()]
    optimizer = object()
    scheduler_steps = []
    scheduler = SimpleNamespace(step=lambda *, increment: scheduler_steps.append(increment))
    load_calls = []

    monkeypatch.setattr(model_module, "should_preload_low_precision_model_before_optimizer", lambda *a, **k: False)
    monkeypatch.setattr(
        model_module,
        "setup_model_and_optimizer",
        lambda *a, **k: (model, optimizer, scheduler),
    )
    monkeypatch.setattr(model_module, "_critic_output_layer_needs_reinit", lambda *a, **k: False)
    monkeypatch.setattr(model_module, "clear_memory", lambda: None)
    monkeypatch.setattr(model_module, "check_peak_gpu_memory_after_load", lambda *a, **k: None)
    monkeypatch.setattr(model_module, "check_model_hashes", lambda *a, **k: None)

    def load_checkpoint(*load_args, **load_kwargs):
        load_calls.append((load_args, load_kwargs))
        args._orbit_training_checkpoint_loaded = True
        args._orbit_optimizer_scheduler_state_restored = True
        return 7, 0

    monkeypatch.setattr(model_module, "load_checkpoint", load_checkpoint)

    loaded_model, loaded_optimizer, loaded_scheduler, iteration = model_module.initialize_model_and_optimizer(
        args,
        role="actor",
    )

    assert loaded_model is model
    assert loaded_optimizer is optimizer
    assert loaded_scheduler is scheduler
    assert iteration == 7
    assert model[0].role == "actor"
    assert load_calls[0][1]["load_training_state"] is True
    assert scheduler_steps == []


def test_full_model_actor_model_only_load_still_initializes_scheduler(monkeypatch):
    args = Namespace(
        load="/checkpoint",
        global_batch_size=8,
        fp16=False,
        bf16=False,
    )
    model = [SimpleNamespace()]
    optimizer = object()
    scheduler_steps = []
    scheduler = SimpleNamespace(step=lambda *, increment: scheduler_steps.append(increment))

    monkeypatch.setattr(model_module, "should_preload_low_precision_model_before_optimizer", lambda *a, **k: False)
    monkeypatch.setattr(
        model_module,
        "setup_model_and_optimizer",
        lambda *a, **k: (model, optimizer, scheduler),
    )
    monkeypatch.setattr(model_module, "_critic_output_layer_needs_reinit", lambda *a, **k: False)
    monkeypatch.setattr(model_module, "clear_memory", lambda: None)
    monkeypatch.setattr(model_module, "check_peak_gpu_memory_after_load", lambda *a, **k: None)
    monkeypatch.setattr(model_module, "check_model_hashes", lambda *a, **k: None)

    def load_checkpoint(*load_args, **load_kwargs):
        args._orbit_training_checkpoint_loaded = False
        args._orbit_optimizer_scheduler_state_restored = False
        return 7, 0

    monkeypatch.setattr(model_module, "load_checkpoint", load_checkpoint)

    _, _, _, iteration = model_module.initialize_model_and_optimizer(args, role="actor")

    assert iteration == 7
    assert scheduler_steps == [56]
