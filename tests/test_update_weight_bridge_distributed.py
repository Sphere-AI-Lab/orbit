"""Unit tests for bridge-aware disaggregated weight sync.

Covers the updater selection matrix, the v1 scope guards, and the lockstep
chunk-drain contract (every rank consumes every chunk; only the source
broadcasts). GPU transport is exercised by the condor smokes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import miles.backends.megatron_utils.actor as actor_mod
import orbit.megatron.update_weight_bridge as bridge_mod
from miles.backends.megatron_utils.actor import _select_update_weight_cls
from orbit.megatron.update_weight_bridge import (
    UpdateWeightFromDistributedBridge,
)


def _args(**overrides):
    values = {
        "colocate": False,
        "peft_method": "none",
        "update_weight_transfer_mode": "broadcast",
        "megatron_to_hf_mode": "raw",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


# ---------------------------------------------------------------------------
# Selection matrix
# ---------------------------------------------------------------------------


def test_colocate_selects_tensor_updater():
    assert _select_update_weight_cls(_args(colocate=True)) is actor_mod.UpdateWeightFromTensor


def test_peft_selects_tensor_updater_even_disaggregated():
    assert _select_update_weight_cls(_args(peft_method="oft")) is actor_mod.UpdateWeightFromTensor


def test_disaggregated_raw_selects_name_based_broadcast():
    assert _select_update_weight_cls(_args()) is actor_mod.UpdateWeightFromDistributed


def test_disaggregated_bridge_selects_bridge_broadcast():
    assert (
        _select_update_weight_cls(_args(megatron_to_hf_mode="bridge")) is UpdateWeightFromDistributedBridge
    )


# ---------------------------------------------------------------------------
# v1 scope guards
# ---------------------------------------------------------------------------


def _make_updater(monkeypatch, *, rank=0, quantization_config=None, is_lora=False):
    created = {}

    class FakeIterator:
        def __init__(self):
            created["iterator"] = self

        def get_hf_weight_chunks(self, weights):
            yield from weights

    monkeypatch.setattr(
        bridge_mod.HfWeightIteratorBase, "create", staticmethod(lambda **kw: FakeIterator())
    )
    updater = UpdateWeightFromDistributedBridge(
        _args(megatron_to_hf_mode="bridge"),
        model=[],
        weights_getter=lambda: [],
        model_name="nemotronhconfig",
        quantization_config=quantization_config,
        is_lora=is_lora,
    )
    return updater


def test_rejects_peft(monkeypatch):
    with pytest.raises(ValueError, match="PEFT"):
        _make_updater(monkeypatch, is_lora=True)


def test_rejects_quantized_checkpoints(monkeypatch):
    with pytest.raises(ValueError, match="quantized"):
        _make_updater(monkeypatch, quantization_config={"quant_method": "compressed-tensors"})


# ---------------------------------------------------------------------------
# Lockstep drain: all ranks consume, only source ships
# ---------------------------------------------------------------------------


def _drain(monkeypatch, rank: int):
    updater = _make_updater(monkeypatch)
    chunks = [[("a", 1)], [("b", 2)], [("c", 3)]]
    updater.weights_getter = lambda: chunks

    consumed = []
    shipped = []

    monkeypatch.setattr(bridge_mod.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(bridge_mod.dist, "barrier", lambda group=None: None)
    monkeypatch.setattr(bridge_mod, "get_gloo_group", lambda: None)
    monkeypatch.setattr(
        UpdateWeightFromDistributedBridge,
        "_pause_and_prepare_engines",
        lambda self: None,
    )
    monkeypatch.setattr(
        UpdateWeightFromDistributedBridge,
        "_finalize_and_resume_engines",
        lambda self, post_load_weights=False: None,
    )
    monkeypatch.setattr(
        UpdateWeightFromDistributedBridge,
        "_update_weight_implementation",
        lambda self, tensors, pbar=None: shipped.append(list(tensors)),
    )

    real_chunks = updater._hf_weight_iterator.get_hf_weight_chunks

    def counting_chunks(weights):
        for chunk in real_chunks(weights):
            consumed.append(chunk)
            yield chunk

    updater._hf_weight_iterator.get_hf_weight_chunks = counting_chunks
    monkeypatch.setattr(bridge_mod, "tqdm", lambda *a, **k: None)

    updater.update_weights()
    return consumed, shipped, updater


def test_source_rank_consumes_and_ships_every_chunk(monkeypatch):
    consumed, shipped, updater = _drain(monkeypatch, rank=0)
    assert len(consumed) == 3
    assert shipped == [[("a", 1)], [("b", 2)], [("c", 3)]]
    assert updater.weight_version == 1


def test_non_source_rank_consumes_everything_but_ships_nothing(monkeypatch):
    consumed, shipped, _ = _drain(monkeypatch, rank=1)
    assert len(consumed) == 3
    assert shipped == []
