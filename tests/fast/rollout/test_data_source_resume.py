from types import SimpleNamespace

import torch

from orbit.rollout.data_source import RolloutDataSource


class _Dataset:
    def __init__(self, size: int):
        self.samples = [SimpleNamespace(value=index) for index in range(size)]

    def __len__(self):
        return len(self.samples)


def _data_source(save_path):
    source = object.__new__(RolloutDataSource)
    source.args = SimpleNamespace(
        save=str(save_path),
        rollout_global_dataset=True,
        rollout_shuffle=False,
        n_samples_per_prompt=2,
    )
    source.dataset = _Dataset(10)
    source.sample_offset = 0
    source.epoch_id = 0
    source.sample_group_index = 0
    source.sample_index = 0
    source.metadata = {}
    source._latest_completed_rollout_id = None
    source._rollout_state_snapshots = {}
    return source


def test_external_async_save_uses_completed_rollout_snapshot(tmp_path):
    source = _data_source(tmp_path)
    source.get_samples(2)
    source.metadata = {"nested": {"rollout": 0}}
    source.mark_rollout_complete(0, snapshot_for_save=True)

    source.get_samples(2)
    source.metadata["nested"]["rollout"] = 1
    source.mark_rollout_complete(1, snapshot_for_save=True)
    source.save(0)

    state = torch.load(
        tmp_path / "rollout" / "global_dataset_state_dict_0.pt",
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


def test_external_save_snapshots_are_bounded_to_the_async_pipeline_depth(tmp_path):
    source = _data_source(tmp_path)

    for rollout_id in range(100):
        source.sample_offset = rollout_id + 1
        source.mark_rollout_complete(rollout_id, snapshot_for_save=True)

    assert list(source._rollout_state_snapshots) == [98, 99]

    source.save(98)
    state = torch.load(
        tmp_path / "rollout" / "global_dataset_state_dict_98.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert state["sample_offset"] == 99
    assert list(source._rollout_state_snapshots) == [99]
