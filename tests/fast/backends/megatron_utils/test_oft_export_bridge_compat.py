"""CPU coverage for the submission OFT exporter, chunk stream, and payload."""

import json
from argparse import Namespace
from types import MethodType
from unittest.mock import Mock

import pytest
import torch
from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.model_bridge import HFWeightTuple
from megatron.bridge.orbit.conversion import oft_export
from sglang.srt.oft.integration import reconstruct_oft_staging

from orbit.backends.megatron_utils.peft_transport._gather import coalesce_oft_hf_weight_chunks
from orbit.backends.megatron_utils.peft_transport._payload import build_oft_flattened_payload
from orbit.backends.megatron_utils.peft_transport.backends.nccl import _flatten_meta_to_json
from orbit.backends.megatron_utils.update_weight.hf_weight_iterator_bridge import HfWeightIteratorBridge


@pytest.fixture
def export_env(monkeypatch):
    bridge = AutoBridge.__new__(AutoBridge)
    monkeypatch.setattr(AutoBridge, "from_hf_pretrained", Mock(return_value=bridge))
    model = torch.nn.Module()
    model.config = Namespace(share_embeddings_and_output_weights=False)
    args = Namespace(
        hf_checkpoint="/unused/checkpoint",
        update_weight_buffer_size=1,
        vocab_size=2,
        q_lora_rank=None,
        colocate=False,
    )
    iterator = HfWeightIteratorBridge(
        args, [model], model_name="qwen3", quantization_config=None, peft_method="oft"
    )
    stream = Mock()
    native_bridge = Mock(return_value=Namespace(stream_oft_adapter_weights_megatron_to_hf=stream))
    # Execute the real free-function API, isolating only distributed weight gathering.
    monkeypatch.setattr(oft_export, "oft_export_bridge_for", native_bridge)
    return Namespace(iterator=iterator, bridge=bridge, stream=stream, native_bridge=native_bridge)


def _oft_weights(*, with_megatron_names=False):
    names = [
        "model.layers.0.self_attn.q_proj.oft_R.weight",
        "model.layers.0.self_attn.k_proj.oft_R.weight",
        "model.layers.0.self_attn.v_proj.oft_R.weight",
        "model.layers.0.mlp.gate_proj.oft_R.weight",
        "model.layers.0.mlp.up_proj.oft_R.weight",
    ]
    return [
        HFWeightTuple(
            name,
            torch.arange(6, dtype=torch.float32).reshape(2, 3) + index,
            f"decoder.layers.0.adapter_{index}.oft_r" if with_megatron_names else None,
        )
        for index, name in enumerate(names)
    ]


def _assert_coalesced_siblings(iterator, weights):
    source_chunks, coalesced = coalesce_oft_hf_weight_chunks(iterator.get_hf_weight_chunks({}))
    assert source_chunks == len(weights)
    assert len(coalesced) == 1
    assert [name for name, _ in coalesced[0]] == [item.param_name for item in weights]
    for (_, tensor), source in zip(coalesced[0], weights, strict=True):
        assert tensor is source.weight
        assert tensor.device.type == "cpu"


@pytest.mark.parametrize("q_lora_rank", [None, 4])
def test_submission_free_exporter_preserves_sglang_siblings(export_env, q_lora_rank):
    export_env.iterator.args.q_lora_rank = q_lora_rank
    weights = _oft_weights()
    export_env.stream.return_value = iter(weights)

    _assert_coalesced_siblings(export_env.iterator, weights)

    export_env.native_bridge.assert_called_once_with(export_env.bridge)
    export_env.stream.assert_called_once_with(
        export_env.iterator.model,
        cpu=False,
        show_progress=False,
        export_format=oft_export.OFTExportFormat.SGLANG,
    )


@pytest.mark.parametrize("with_megatron_names", [False, True])
def test_legacy_bound_oft_exporter_remains_supported(export_env, with_megatron_names):
    export_env.iterator.args.q_lora_rank = 4
    weights = _oft_weights(with_megatron_names=with_megatron_names)
    calls = []

    def export_legacy(bridge, model, *, cpu, show_progress):
        calls.append((bridge, model, cpu, show_progress))
        return iter(weights)

    export_env.bridge.export_oft_adapter_weights = MethodType(export_legacy, export_env.bridge)

    _assert_coalesced_siblings(export_env.iterator, weights)

    assert calls == [(export_env.bridge, export_env.iterator.model, False, False)]
    export_env.native_bridge.assert_not_called()


@pytest.mark.parametrize("weight_type", ["base", "lora"])
def test_base_and_lora_keep_their_export_and_postprocessing(export_env, weight_type):
    embedding = HFWeightTuple(
        "model.embed_tokens.weight",
        torch.arange(12, dtype=torch.float32).reshape(4, 3),
        "embedding.word_embeddings.weight",
    )
    lora = HFWeightTuple(
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight",
        torch.ones(2, 3),
        "decoder.layers.0.self_attention.linear_qkv.adapter.linear_in.weight",
    )
    tasks = []
    export_env.bridge.get_conversion_tasks = Mock(return_value=tasks)
    export_env.bridge.export_hf_weights = Mock(return_value=iter([embedding, lora]))
    export_env.bridge.export_adapter_weights = Mock(return_value=iter([embedding, lora]))

    chunks = list(export_env.iterator.get_hf_weight_chunks({}, weight_type=weight_type))

    assert len(chunks) == 1 and len(chunks[0]) == 1
    name, tensor = chunks[0][0]
    if weight_type == "base":
        export_env.bridge.export_hf_weights.assert_called_once_with(
            export_env.iterator.model, cpu=False, conversion_tasks=tasks, merge_adapter_weights=False
        )
        export_env.bridge.export_adapter_weights.assert_not_called()
        assert name == embedding.param_name
        assert torch.equal(tensor, embedding.weight[:2])
    else:
        export_env.bridge.export_adapter_weights.assert_called_once_with(
            export_env.iterator.model, cpu=False, show_progress=False
        )
        export_env.bridge.export_hf_weights.assert_not_called()
        assert name == lora.param_name
        assert tensor is lora.weight
    export_env.native_bridge.assert_not_called()


def test_oft_payload_roundtrips_aliases_clones_and_distinct_storage_views():
    storage = torch.arange(18, dtype=torch.float32).reshape(2, 3, 3)
    source = storage[0]
    tensors = [
        source,
        source.view_as(source),
        source.clone(),
        source.T,
        source.reshape(1, 9),
        storage[1],
        source.view(torch.int32),
    ]
    names = [f"model.layers.0.{leaf}.oft_R.weight" for leaf in ("q", "k", "v", "gate", "up", "down", "dtype")]
    named_tensors = list(zip(names, tensors, strict=True))

    payload = build_oft_flattened_payload(named_tensors)

    assert payload.extra["entries"] == list(zip(names, [0, 0, 1, 2, 3, 4, 5], strict=True))
    assert [meta.name for meta in payload.metadata] == [names[index] for index in (0, 2, 3, 4, 5, 6)]
    assert payload.flat_tensor.device.type == "cpu"
    assert payload.flat_tensor.dtype == torch.uint8
    assert payload.flat_tensor.numel() == 6 * source.nbytes
    assert not payload.flat_tensor.requires_grad
    wire_metadata = json.loads(
        json.dumps(
            {
                "metadata": [_flatten_meta_to_json(meta) for meta in payload.metadata],
                "extra": {"entries": [[name, index] for name, index in payload.extra["entries"]]},
            }
        )
    )

    reconstructed = reconstruct_oft_staging([("__flattened__", payload.flat_tensor)], wire_metadata)

    assert [name for name, _ in reconstructed] == names
    for (_, tensor), original in zip(reconstructed, tensors, strict=True):
        assert tensor.dtype == original.dtype
        assert torch.equal(tensor, original)
    assert reconstructed[0][1] is reconstructed[1][1]
    assert reconstructed[0][1] is not reconstructed[2][1]
