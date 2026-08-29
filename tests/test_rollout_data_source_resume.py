from types import SimpleNamespace

import pytest
import torch

from miles.rollout.data_source import DataSource, RolloutDataSource, _resolve_rollout_dataset_state_location


class _Dataset:
    def __init__(self, size: int):
        self.size = size
        self.samples = [SimpleNamespace(value=index) for index in range(size)]
        self.shuffle_calls = []

    def __len__(self):
        return self.size

    def shuffle(self, epoch_id):
        self.shuffle_calls.append(epoch_id)


def _args(load_path, *, adapter_path=None, save_path=None, shuffle=True):
    return SimpleNamespace(
        load=str(load_path) if load_path is not None else None,
        save=str(save_path) if save_path is not None else None,
        peft_adapter_path=str(adapter_path) if adapter_path is not None else None,
        lora_adapter_path=None,
        oft_adapter_path=None,
        rollout_global_dataset=True,
        rollout_shuffle=shuffle,
        n_samples_per_prompt=2,
    )


def _data_source(args, *, dataset_size=10):
    source = object.__new__(RolloutDataSource)
    source.args = args
    source.dataset = _Dataset(dataset_size)
    source.sample_offset = 0
    source.epoch_id = 0
    source.sample_group_index = 0
    source.sample_index = 0
    source.metadata = {}
    source._latest_completed_rollout_id = None
    source._rollout_state_snapshots = {}
    return source


def _write_state(root, rollout_id, **overrides):
    state = {
        "sample_offset": 2,
        "epoch_id": 1,
        "sample_group_index": 2,
        "sample_index": 4,
        "metadata": {"source": "checkpoint"},
    }
    state.update(overrides)
    path = root / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    return path


def test_canonical_peft_resume_loads_state_from_actor_root(tmp_path):
    base_root = tmp_path / "base"
    adapter_path = tmp_path / "actor" / "iter_0000001" / "adapter"
    adapter_path.mkdir(parents=True)
    _write_state(
        tmp_path / "actor",
        1,
        sample_offset=4,
        sample_group_index=4,
        sample_index=8,
        metadata={"resumed": True},
    )
    source = _data_source(_args(base_root, adapter_path=adapter_path))

    source.load(1)

    assert source.sample_offset == 4
    assert source.epoch_id == 1
    assert source.sample_group_index == 4
    assert source.sample_index == 8
    assert source.metadata == {"resumed": True}
    assert source.dataset.shuffle_calls == [1]


def test_full_checkpoint_resume_keeps_load_as_state_root(tmp_path):
    checkpoint_root = tmp_path / "full-checkpoint"
    _write_state(checkpoint_root, 3, epoch_id=2)
    source = _data_source(_args(checkpoint_root))

    source.load(3)

    assert source.sample_offset == 2
    assert source.epoch_id == 2
    assert source.sample_group_index == 2
    assert source.sample_index == 4
    assert source.dataset.shuffle_calls == [2]


def test_direct_full_checkpoint_resume_loads_state_from_parent_root(tmp_path):
    checkpoint_root = tmp_path / "full-checkpoint"
    direct_iteration_path = checkpoint_root / "iter_0000003"
    direct_iteration_path.mkdir(parents=True)
    _write_state(checkpoint_root, 3, sample_offset=4, sample_group_index=4, sample_index=8)
    source = _data_source(_args(direct_iteration_path))

    source.load(3)

    assert source.sample_offset == 4
    assert source.sample_group_index == 4
    assert source.sample_index == 8
    assert source.dataset.shuffle_calls == [1]


def test_full_checkpoint_resume_requires_dataset_state(tmp_path):
    checkpoint_root = tmp_path / "full-checkpoint"
    checkpoint_root.mkdir()
    source = _data_source(_args(checkpoint_root))

    with pytest.raises(FileNotFoundError, match="required rollout dataset checkpoint"):
        source.load(3)

    assert source.sample_offset == 0
    assert source.dataset.shuffle_calls == []


def test_direct_full_checkpoint_requires_state_at_parent_root(tmp_path):
    checkpoint_root = tmp_path / "full-checkpoint"
    direct_iteration_path = checkpoint_root / "iter_0000003"
    direct_iteration_path.mkdir(parents=True)
    _write_state(direct_iteration_path, 3, sample_offset=7)
    source = _data_source(_args(direct_iteration_path))

    with pytest.raises(FileNotFoundError, match=str(checkpoint_root / "rollout")):
        source.load(3)

    assert source.sample_offset == 0
    assert source.dataset.shuffle_calls == []


def test_direct_full_checkpoint_iteration_must_match_rollout_id(tmp_path):
    direct_iteration_path = tmp_path / "full-checkpoint" / "iter_0000004"
    direct_iteration_path.mkdir(parents=True)
    source = _data_source(_args(direct_iteration_path))

    with pytest.raises(ValueError, match="checkpoint iteration 4, rollout id 3"):
        source.load(3)

    assert source.sample_offset == 0
    assert source.dataset.shuffle_calls == []


def test_full_checkpoint_symlink_to_iteration_uses_resolved_parent_root(tmp_path):
    checkpoint_root = tmp_path / "full-checkpoint"
    iteration_path = checkpoint_root / "iter_0000003"
    iteration_path.mkdir(parents=True)
    alias_path = tmp_path / "latest"
    alias_path.symlink_to(iteration_path, target_is_directory=True)
    _write_state(checkpoint_root, 3, sample_offset=4, sample_group_index=4, sample_index=8)
    source = _data_source(_args(alias_path))

    source.load(3)

    assert source.sample_offset == 4
    assert source.sample_group_index == 4
    assert source.sample_index == 8


def test_full_checkpoint_symlink_iteration_must_match_rollout_id(tmp_path):
    checkpoint_root = tmp_path / "full-checkpoint"
    iteration_path = checkpoint_root / "iter_0000004"
    iteration_path.mkdir(parents=True)
    alias_path = tmp_path / "latest"
    alias_path.symlink_to(iteration_path, target_is_directory=True)
    source = _data_source(_args(alias_path))

    with pytest.raises(ValueError, match="checkpoint iteration 4, rollout id 3"):
        source.load(3)

    assert source.sample_offset == 0
    assert source.dataset.shuffle_calls == []


def test_full_checkpoint_root_alias_uses_resolved_root(tmp_path):
    checkpoint_root = tmp_path / "full-checkpoint"
    checkpoint_root.mkdir()
    alias_path = tmp_path / "checkpoint-alias"
    alias_path.symlink_to(checkpoint_root, target_is_directory=True)
    _write_state(checkpoint_root, 3, sample_offset=4, sample_group_index=4, sample_index=8)
    source = _data_source(_args(alias_path))

    source.load(3)

    assert source.sample_offset == 4
    assert source.sample_group_index == 4
    assert source.sample_index == 8


def test_arbitrary_weights_only_adapter_keeps_load_as_state_root(tmp_path):
    base_root = tmp_path / "base"
    canonical_adapter_path = tmp_path / "actor" / "iter_0000001" / "adapter"
    canonical_adapter_path.mkdir(parents=True)
    canonical_args = _args(base_root, adapter_path=canonical_adapter_path)

    # The rollout manager requests -1 when the actor treated this as a
    # weights-only warm start, even if its path happens to look canonical.
    assert _resolve_rollout_dataset_state_location(canonical_args, -1) == (base_root, False)

    arbitrary_adapter_path = tmp_path / "exported-adapter"
    arbitrary_adapter_path.mkdir()
    arbitrary_args = _args(base_root, adapter_path=arbitrary_adapter_path)

    assert _resolve_rollout_dataset_state_location(arbitrary_args, -1) == (base_root, False)
    assert _resolve_rollout_dataset_state_location(arbitrary_args, 4) == (base_root, False)

    direct_model_only_path = tmp_path / "base-checkpoints" / "iter_0000004"
    direct_model_only_path.mkdir(parents=True)
    direct_model_only_args = _args(direct_model_only_path)
    assert _resolve_rollout_dataset_state_location(direct_model_only_args, -1) == (
        direct_model_only_path,
        False,
    )
    model_only_source = _data_source(direct_model_only_args)
    model_only_source.load(-1)
    assert model_only_source.sample_offset == 0


def test_canonical_peft_iteration_mismatch_fails_instead_of_loading_base_state(tmp_path):
    base_root = tmp_path / "base"
    adapter_path = tmp_path / "actor" / "iter_0000002" / "adapter"
    adapter_path.mkdir(parents=True)
    _write_state(base_root, 1, sample_offset=7)
    source = _data_source(_args(base_root, adapter_path=adapter_path))

    with pytest.raises(ValueError, match="adapter iteration 2, rollout id 1"):
        source.load(1)

    assert source.sample_offset == 0
    assert source.dataset.shuffle_calls == []


def test_canonical_peft_resume_requires_derived_dataset_state(tmp_path):
    base_root = tmp_path / "base"
    adapter_path = tmp_path / "actor" / "iter_0000001" / "adapter"
    adapter_path.mkdir(parents=True)
    _write_state(base_root, 1, sample_offset=7)
    source = _data_source(_args(base_root, adapter_path=adapter_path))

    with pytest.raises(FileNotFoundError, match="required rollout dataset checkpoint"):
        source.load(1)

    assert source.sample_offset == 0
    assert source.dataset.shuffle_calls == []


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"sample_offset": 11}, "sample_offset exceeds"),
        ({"sample_index": True}, "invalid sample_index"),
        ({"metadata": []}, "metadata must be a dictionary"),
    ],
)
def test_rollout_dataset_state_is_validated_before_mutation(tmp_path, overrides, message):
    checkpoint_root = tmp_path / "checkpoint"
    _write_state(checkpoint_root, 2, **overrides)
    source = _data_source(_args(checkpoint_root), dataset_size=10)

    with pytest.raises(RuntimeError, match=message):
        source.load(2)

    assert source.sample_offset == 0
    assert source.epoch_id == 0
    assert source.sample_group_index == 0
    assert source.sample_index == 0
    assert source.metadata == {}
    assert source.dataset.shuffle_calls == []


def test_delayed_async_save_uses_completed_rollout_snapshot(tmp_path):
    save_root = tmp_path / "save"
    source = _data_source(_args(tmp_path / "base", save_path=save_root))

    source.get_samples(2)
    source.metadata = {"nested": {"rollout": 0}}
    source.mark_rollout_complete(0, snapshot_for_save=True)

    # This models generate(1) running on the serialized RolloutManager before
    # the already-queued save(0) method gets its turn.
    source.get_samples(2)
    source.metadata["nested"]["rollout"] = 1
    source.mark_rollout_complete(1, snapshot_for_save=False)
    source.save(0)

    state = torch.load(
        save_root / "rollout" / "global_dataset_state_dict_0.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert state == {
        "sample_offset": 2,
        "epoch_id": 0,
        "sample_group_index": 2,
        "sample_index": 4,
        "metadata": {"nested": {"rollout": 0}},
    }
    assert source.sample_offset == 4
    assert source.sample_group_index == 4
    assert source.sample_index == 8


def test_delayed_save_without_snapshot_rejects_later_live_cursor(tmp_path):
    source = _data_source(_args(tmp_path / "base", save_path=tmp_path / "save"))
    source.get_samples(2)
    source.mark_rollout_complete(0, snapshot_for_save=False)
    source.get_samples(2)
    source.mark_rollout_complete(1, snapshot_for_save=False)

    with pytest.raises(RuntimeError, match="no immutable state"):
        source.save(0)


def test_rollout_manager_marks_completed_state_before_next_generate(monkeypatch):
    import miles.ray.rollout as rollout_module

    class _RecordingDataSource:
        def __init__(self):
            self.dataset = _Dataset(10)
            self.cursor = 0
            self.marks = []

        def mark_rollout_complete(self, rollout_id, *, snapshot_for_save):
            self.marks.append((rollout_id, self.cursor, snapshot_for_save))

    manager_class = rollout_module.RolloutManager.__ray_metadata__.modified_class
    manager = object.__new__(manager_class)
    manager.args = SimpleNamespace(
        ci_test=False,
        use_fault_tolerance=False,
        rollout_global_dataset=True,
        rollout_batch_size=2,
        save_interval=1,
        num_rollout=2,
        opd_defer_full_vocab_scoring=False,
    )
    manager.data_source = _RecordingDataSource()
    manager.train_parallel_config = {"dp_size": 1}
    manager.health_monitoring_resume = lambda: None

    def get_rollout_data(rollout_id):
        manager.data_source.cursor += 2
        return [SimpleNamespace(rollout_id=rollout_id)], {}

    manager._get_rollout_data = get_rollout_data
    manager._save_debug_rollout_data = lambda *_args, **_kwargs: None
    manager._convert_samples_to_train_data = lambda data: data
    manager._split_train_data_by_dp = lambda data, _dp_size: data
    monkeypatch.setattr(rollout_module, "_log_rollout_data", lambda *_args, **_kwargs: None)

    manager.generate(0)
    manager.generate(1)

    assert manager.data_source.marks == [(0, 2, True), (1, 4, True)]

    class _DuckTypedDataSource:
        def __init__(self):
            self.dataset = _Dataset(10)
            self.cursor = 0

    manager.data_source = _DuckTypedDataSource()
    manager._get_rollout_data = get_rollout_data
    manager.generate(0)
    assert manager.data_source.cursor == 2

    failed_source = _RecordingDataSource()
    manager.data_source = failed_source
    manager._get_rollout_data = get_rollout_data

    def fail_split(_data, _dp_size):
        raise RuntimeError("split failed")

    manager._split_train_data_by_dp = fail_split
    with pytest.raises(RuntimeError, match="split failed"):
        manager.generate(0)
    assert failed_source.marks == []


def test_custom_data_source_does_not_need_snapshot_hook():
    class _CustomDataSource(DataSource):
        def get_samples(self, num_samples):
            return []

        def add_samples(self, samples):
            return None

        def save(self, rollout_id):
            return None

        def load(self, rollout_id=None):
            return None

    source = _CustomDataSource()
    assert source.mark_rollout_complete(0, snapshot_for_save=True) is None


def test_no_global_dataset_snapshot_and_save_are_noops(tmp_path):
    args = _args(tmp_path / "base", save_path=tmp_path / "save")
    args.rollout_global_dataset = False
    source = _data_source(args)

    source.mark_rollout_complete(0, snapshot_for_save=True)
    source.save(0)

    assert source._latest_completed_rollout_id is None
    assert source._rollout_state_snapshots == {}
    assert not (tmp_path / "save").exists()
