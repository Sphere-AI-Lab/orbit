from types import SimpleNamespace

import pytest
import torch

from orbit.rollout.data_source import RolloutDataSource, _resolve_rollout_dataset_state_location


class _Dataset:
    def __init__(self, size: int):
        self.size = size
        self.shuffle_calls = []

    def __len__(self):
        return self.size

    def shuffle(self, epoch_id):
        self.shuffle_calls.append(epoch_id)


def _args(load_path, *, adapter_path=None, shuffle=True):
    return SimpleNamespace(
        load=str(load_path) if load_path is not None else None,
        peft_adapter_path=str(adapter_path) if adapter_path is not None else None,
        lora_adapter_path=None,
        oft_adapter_path=None,
        rollout_global_dataset=True,
        rollout_shuffle=shuffle,
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

    with pytest.raises(FileNotFoundError, match="required PEFT rollout dataset checkpoint"):
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
