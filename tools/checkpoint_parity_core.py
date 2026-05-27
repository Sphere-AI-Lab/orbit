from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


def _bootstrap_repo_root_on_sys_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    repo_root_resolved = repo_root.resolve()
    for entry in sys.path:
        try:
            if Path(entry or ".").resolve() == repo_root_resolved:
                return
        except OSError:
            continue
    sys.path.insert(0, str(repo_root))


_bootstrap_repo_root_on_sys_path()


@dataclass
class CheckpointParityReport:
    hf_model_path: str
    megatron_dist_path: str
    missing_checkpoint_keys: list[str]
    missing_scale_keys: list[str]
    missing_extra_state_keys: list[str]
    mismatched_shapes: list[dict[str, object]]

    @property
    def ok(self) -> bool:
        return not (
            self.missing_checkpoint_keys
            or self.missing_scale_keys
            or self.missing_extra_state_keys
            or self.mismatched_shapes
        )


@dataclass
class RuntimeParityThresholds:
    max_logprob_abs_diff: float
    min_topk_agreement: float


@dataclass
class RuntimeParityReport:
    tokenizer_match: bool
    max_logit_abs_diff: float
    max_logprob_abs_diff: float
    next_token_match_rate: float
    topk_agreement: float
    thresholds: RuntimeParityThresholds

    @property
    def ok(self) -> bool:
        return (
            self.tokenizer_match
            # Step-0 parity is not safe to report as healthy when the chosen next token diverges.
            and self.next_token_match_rate >= 1.0
            and self.max_logprob_abs_diff <= self.thresholds.max_logprob_abs_diff
            and self.topk_agreement >= self.thresholds.min_topk_agreement
        )


def resolve_megatron_dist_checkpoint_dir(checkpoint_path: str) -> str:
    checkpoint_dir = Path(checkpoint_path)
    if (checkpoint_dir / ".metadata").exists():
        return str(checkpoint_dir)

    tracker_path = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if tracker_path.exists():
        iteration = int(tracker_path.read_text().strip())
        resolved_dir = checkpoint_dir / f"iter_{iteration:07d}"
        if (resolved_dir / ".metadata").exists():
            return str(resolved_dir)

    candidates = [candidate for candidate in sorted(checkpoint_dir.glob("iter_*")) if (candidate / ".metadata").exists()]
    if candidates:
        return str(candidates[-1])

    return str(checkpoint_dir)


def read_dcp_tensor_keys(checkpoint_dir: str) -> set[str]:
    from torch.distributed.checkpoint import FileSystemReader

    resolved_dir = resolve_megatron_dist_checkpoint_dir(checkpoint_dir)
    reader = FileSystemReader(resolved_dir)
    metadata = reader.read_metadata()
    return {str(key) for key in metadata.state_dict_metadata.keys()}


def load_hf_weight_index(model_dir: str) -> dict[str, str]:
    index_path = Path(model_dir) / "model.safetensors.index.json"
    if not index_path.exists():
        safetensors_path = Path(model_dir) / "model.safetensors"
        if not safetensors_path.exists():
            return {}
        from safetensors import safe_open

        with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
            return {str(key): safetensors_path.name for key in handle.keys()}
    return json.loads(index_path.read_text())["weight_map"]


def _load_hf_config(model_dir: str) -> dict[str, object]:
    return json.loads((Path(model_dir) / "config.json").read_text())


def _load_optional_hf_config(model_dir: str) -> dict[str, object]:
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text())


def _is_dense_qwen3_hf_checkpoint(model_dir: str) -> bool:
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return False
    config = json.loads(config_path.read_text())
    architectures = config.get("architectures")
    return config.get("model_type") == "qwen3" and isinstance(architectures, list) and "Qwen3ForCausalLM" in architectures


def _is_qwen3_moe_hf_checkpoint(model_dir: str) -> bool:
    config = _load_optional_hf_config(model_dir)
    architectures = config.get("architectures")
    return config.get("model_type") == "qwen3_moe" or (
        isinstance(architectures, list) and "Qwen3MoeForCausalLM" in architectures
    )


def _build_qwen3_moe_common_expected_checkpoint_keys(
    model_dir: str, hf_weight_map: dict[str, str]
) -> set[str]:
    dims = _dense_decoder_int4_dimensions(model_dir)
    keys: set[str] = {
        "embedding.word_embeddings.weight",
        "decoder.final_layernorm.weight",
        "decoder.final_layernorm._extra_state/shard_0_1",
    }
    if "lm_head.weight" in hf_weight_map:
        keys.add("output_layer.weight")
    for layer_idx in range(dims["num_hidden_layers"]):
        layer = f"decoder.layers.{layer_idx}"
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "input_layernorm.weight"):
            keys.add(f"{layer}.self_attention.linear_qkv.layer_norm_weight")
        if _dense_decoder_nvfp4_hf_key_is_present(
            hf_weight_map, layer_idx, "post_attention_layernorm.weight"
        ):
            keys.add(f"{layer}.pre_mlp_layernorm.weight")
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "self_attn.q_norm.weight"):
            keys.add(f"{layer}.self_attention.q_layernorm.weight")
            keys.add(f"{layer}.self_attention.q_layernorm._extra_state/shard_0_1")
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "self_attn.k_norm.weight"):
            keys.add(f"{layer}.self_attention.k_layernorm.weight")
            keys.add(f"{layer}.self_attention.k_layernorm._extra_state/shard_0_1")
    return keys


def _qwen3_moe_hf_expert_indices(hf_weight_map: dict[str, str], layer_idx: int) -> set[int]:
    prefix = f"model.layers.{layer_idx}.mlp.experts."
    indices: set[int] = set()
    for key in hf_weight_map:
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix) :]
        expert_idx, _, suffix = remainder.partition(".")
        if expert_idx.isdigit() and suffix in {
            "gate_proj.weight",
            "up_proj.weight",
            "down_proj.weight",
        }:
            indices.add(int(expert_idx))
    return indices


def _build_qwen3_moe_fp8_expected_checkpoint_keys(
    model_dir: str, hf_weight_map: dict[str, str]
) -> set[str]:
    dims = _dense_decoder_int4_dimensions(model_dir)
    keys = _build_qwen3_moe_common_expected_checkpoint_keys(model_dir, hf_weight_map)
    for layer_idx in range(dims["num_hidden_layers"]):
        layer = f"decoder.layers.{layer_idx}"
        if _dense_decoder_nvfp4_all_hf_keys_present(
            hf_weight_map,
            layer_idx,
            ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"),
        ):
            keys.add(f"{layer}.self_attention.linear_qkv.weight_w")
            keys.add(f"{layer}.self_attention.linear_qkv.weight_scale_inv")
            keys.add(f"{layer}.self_attention.linear_qkv._extra_state/shard_0_1")
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "self_attn.o_proj.weight"):
            keys.add(f"{layer}.self_attention.linear_proj.weight_w")
            keys.add(f"{layer}.self_attention.linear_proj.weight_scale_inv")
            keys.add(f"{layer}.self_attention.linear_proj._extra_state/shard_0_1")
        # Megatron stores MoE FP8 experts in TEGroupedMLP layout: stacked
        # grouped-GEMM weight tensors at the shared module path with a
        # per-expert numbered suffix, plus per-expert _extra_state under a
        # nested `experts.{E}` path. fc1 carries the SwiGLU gate/up split as
        # `weight{E}_w`/`weight{E}_v`; fc2 has no SwiGLU half so the weight is
        # stored as `weight{E}` directly.
        for expert_idx in sorted(_qwen3_moe_hf_expert_indices(hf_weight_map, layer_idx)):
            keys.add(f"{layer}.mlp.experts.linear_fc1.weight{expert_idx}_w")
            keys.add(f"{layer}.mlp.experts.linear_fc1.weight{expert_idx}_v")
            keys.add(f"{layer}.mlp.experts.linear_fc1.weight{expert_idx}_scale_inv")
            keys.add(f"{layer}.mlp.experts.linear_fc2.weight{expert_idx}")
            keys.add(f"{layer}.mlp.experts.linear_fc2.weight{expert_idx}_scale_inv")
            keys.add(
                f"{layer}.mlp.experts.experts.{expert_idx}.linear_fc1._extra_state/shard_0_1"
            )
            keys.add(
                f"{layer}.mlp.experts.experts.{expert_idx}.linear_fc2._extra_state/shard_0_1"
            )
    return keys


def _add_nvfp4_quantizer_keys(keys: set[str], prefix: str, *, expert_idx: str = "") -> None:
    keys.add(f"{prefix}.weight_quantizer._scale{expert_idx}")
    keys.add(f"{prefix}.weight_quantizer._double_scale{expert_idx}")
    keys.add(f"{prefix}.weight_quantizer._amax{expert_idx}")
    keys.add(f"{prefix}.input_quantizer._amax")


def _build_qwen3_moe_nvfp4_expected_checkpoint_keys(
    model_dir: str, hf_weight_map: dict[str, str]
) -> set[str]:
    dims = _dense_decoder_int4_dimensions(model_dir)
    keys = _build_qwen3_moe_common_expected_checkpoint_keys(model_dir, hf_weight_map)
    for layer_idx in range(dims["num_hidden_layers"]):
        layer = f"decoder.layers.{layer_idx}"
        if _dense_decoder_nvfp4_all_hf_keys_present(
            hf_weight_map,
            layer_idx,
            ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight"),
        ):
            prefix = f"{layer}.self_attention.linear_qkv"
            keys.add(f"{prefix}.weight")
            keys.add(f"{prefix}._extra_state/shard_0_1")
            _add_nvfp4_quantizer_keys(keys, prefix)
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "self_attn.o_proj.weight"):
            prefix = f"{layer}.self_attention.linear_proj"
            keys.add(f"{prefix}.weight")
            keys.add(f"{prefix}._extra_state/shard_0_1")
            _add_nvfp4_quantizer_keys(keys, prefix)
        # Megatron stores MoE NVFP4 experts in TEGroupedMLP layout: stacked
        # grouped-GEMM weight tensors at the shared module path with a
        # per-expert numbered suffix. fc1 (SwiGLU gate+up) carries
        # `weight{E}_w`/`weight{E}_v`; fc2 (down_proj) carries `weight{E}`
        # alone — same asymmetry as the FP8 layout. Per-expert modelopt scale
        # tensors share the numeric suffix; input_quantizer._amax is shared
        # once per module. _extra_state lives under the nested `experts.{E}`
        # path, not the shared module path.
        fc1_prefix = f"{layer}.mlp.experts.linear_fc1"
        fc2_prefix = f"{layer}.mlp.experts.linear_fc2"
        for expert_idx in sorted(_qwen3_moe_hf_expert_indices(hf_weight_map, layer_idx)):
            keys.add(f"{fc1_prefix}.weight{expert_idx}_w")
            keys.add(f"{fc1_prefix}.weight{expert_idx}_v")
            keys.add(f"{fc2_prefix}.weight{expert_idx}")
            _add_nvfp4_quantizer_keys(keys, fc1_prefix, expert_idx=str(expert_idx))
            _add_nvfp4_quantizer_keys(keys, fc2_prefix, expert_idx=str(expert_idx))
            keys.add(
                f"{layer}.mlp.experts.experts.{expert_idx}.linear_fc1._extra_state/shard_0_1"
            )
            keys.add(
                f"{layer}.mlp.experts.experts.{expert_idx}.linear_fc2._extra_state/shard_0_1"
            )
    return keys


def _build_dense_qwen3_expected_shapes(model_dir: str) -> dict[str, tuple[int, ...]]:
    config = _load_hf_config(model_dir)
    num_hidden_layers = int(config["num_hidden_layers"])
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    num_attention_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(config["num_key_value_heads"])
    head_dim = int(config.get("head_dim", hidden_size // num_attention_heads))
    vocab_size = int(config["vocab_size"])
    q_size = num_attention_heads * head_dim
    kv_size = num_key_value_heads * head_dim
    return {
        "embedding.word_embeddings.weight": (vocab_size, hidden_size),
        "decoder.final_layernorm.weight": (hidden_size,),
        "decoder.layers.self_attention.q_layernorm.weight": (num_hidden_layers, head_dim),
        "decoder.layers.self_attention.k_layernorm.weight": (num_hidden_layers, head_dim),
        "decoder.layers.self_attention.linear_qkv.weight": (num_hidden_layers, q_size + kv_size + kv_size, hidden_size),
        "decoder.layers.self_attention.linear_qkv.layer_norm_weight": (num_hidden_layers, hidden_size),
        "decoder.layers.self_attention.linear_proj.weight": (num_hidden_layers, hidden_size, q_size),
        "decoder.layers.mlp.linear_fc1.weight_w": (num_hidden_layers, intermediate_size, hidden_size),
        "decoder.layers.mlp.linear_fc1.weight_v": (num_hidden_layers, intermediate_size, hidden_size),
        "decoder.layers.mlp.linear_fc1.layer_norm_weight": (num_hidden_layers, hidden_size),
        "decoder.layers.mlp.linear_fc2.weight": (num_hidden_layers, hidden_size, intermediate_size),
    }


def _build_dense_qwen3_expected_checkpoint_keys(model_dir: str) -> set[str]:
    config = _load_hf_config(model_dir)
    num_hidden_layers = int(config["num_hidden_layers"])
    checkpoint_keys = set(_build_dense_qwen3_expected_shapes(model_dir))
    checkpoint_keys.add("decoder.final_layernorm._extra_state/shard_0_1")
    extra_state_bases = (
        "decoder.layers.self_attention.q_layernorm",
        "decoder.layers.self_attention.k_layernorm",
        "decoder.layers.self_attention.linear_qkv",
        "decoder.layers.self_attention.linear_proj",
        "decoder.layers.mlp.linear_fc1",
        "decoder.layers.mlp.linear_fc2",
    )
    for base in extra_state_bases:
        for layer_idx in range(num_hidden_layers):
            checkpoint_keys.add(f"{base}._extra_state/shard_{layer_idx}_{num_hidden_layers}")
    return checkpoint_keys


def _build_dense_qwen3_expected_scale_roots(model_dir: str) -> set[str]:
    config = _load_hf_config(model_dir)
    num_hidden_layers = int(config["num_hidden_layers"])
    scale_suffix_roots = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    return {
        f"model.layers.{layer_idx}.{root}"
        for layer_idx in range(num_hidden_layers)
        for root in scale_suffix_roots
    }


def _build_dense_qwen3_expected_dcp_scale_keys(model_dir: str, scale_suffix: str) -> set[str]:
    return {
        f"decoder.layers.self_attention.linear_qkv.{scale_suffix}",
        f"decoder.layers.self_attention.linear_proj.{scale_suffix}",
        f"decoder.layers.mlp.linear_fc1.{scale_suffix}",
        f"decoder.layers.mlp.linear_fc2.{scale_suffix}",
    }


def _load_nvfp4_hf_config(model_dir: str) -> dict[str, object]:
    config = _load_hf_config(model_dir)
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        return config

    merged_config = dict(text_config)
    for inherited_key in ("quantization_config", "vocab_size"):
        if inherited_key not in merged_config and inherited_key in config:
            merged_config[inherited_key] = config[inherited_key]
    return merged_config


def _dense_decoder_nvfp4_group_size(model_dir: str) -> int:
    config = _load_nvfp4_hf_config(model_dir)
    quantization_config = config.get("quantization_config", {})
    if isinstance(quantization_config, dict):
        return int(quantization_config.get("group_size", 16))
    return 16


def _dense_decoder_nvfp4_dimensions(model_dir: str) -> dict[str, int]:
    config = _load_nvfp4_hf_config(model_dir)
    hidden_size = int(config["hidden_size"])
    num_attention_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(config.get("num_key_value_heads", num_attention_heads))
    head_dim = int(config.get("head_dim", hidden_size // num_attention_heads))
    q_size = num_attention_heads * head_dim
    kv_size = num_key_value_heads * head_dim
    return {
        "num_hidden_layers": int(config["num_hidden_layers"]),
        "hidden_size": hidden_size,
        "intermediate_size": int(config["intermediate_size"]),
        "vocab_size": int(config["vocab_size"]),
        "head_dim": head_dim,
        "q_size": q_size,
        "kv_size": kv_size,
        "qkv_size": q_size + kv_size + kv_size,
        "group_size": _dense_decoder_nvfp4_group_size(model_dir),
    }


def _dense_decoder_nvfp4_div(value: int, divisor: int, name: str) -> int:
    if value % divisor != 0:
        raise ValueError(f"NVFP4 {name}={value} must be divisible by group size {divisor}.")
    return value // divisor


def _dense_decoder_nvfp4_hf_key(layer_idx: int, suffix: str) -> str:
    return f"model.layers.{layer_idx}.{suffix}"


def _dense_decoder_nvfp4_hf_key_is_present(
    hf_weight_map: dict[str, str],
    layer_idx: int,
    suffix: str,
) -> bool:
    return _dense_decoder_nvfp4_hf_key(layer_idx, suffix) in hf_weight_map


def _dense_decoder_nvfp4_all_hf_keys_present(
    hf_weight_map: dict[str, str],
    layer_idx: int,
    suffixes: tuple[str, ...],
) -> bool:
    return all(_dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, suffix) for suffix in suffixes)


def _dense_decoder_nvfp4_quantizer_key_prefixes(
    model_dir: str,
    hf_weight_map: dict[str, str],
) -> set[str]:
    dims = _dense_decoder_nvfp4_dimensions(model_dir)
    prefixes: set[str] = set()
    for layer_idx in range(dims["num_hidden_layers"]):
        layer_prefix = f"decoder.layers.{layer_idx}"
        if _dense_decoder_nvfp4_all_hf_keys_present(
            hf_weight_map,
            layer_idx,
            (
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
            ),
        ):
            prefixes.add(f"{layer_prefix}.self_attention.linear_qkv")
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "self_attn.o_proj.weight"):
            prefixes.add(f"{layer_prefix}.self_attention.linear_proj")
        if _dense_decoder_nvfp4_all_hf_keys_present(
            hf_weight_map,
            layer_idx,
            (
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
            ),
        ):
            prefixes.add(f"{layer_prefix}.mlp.linear_fc1")
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "mlp.down_proj.weight"):
            prefixes.add(f"{layer_prefix}.mlp.linear_fc2")
    return prefixes


def _build_dense_decoder_nvfp4_expected_quantizer_keys(
    model_dir: str,
    hf_weight_map: dict[str, str],
) -> set[str]:
    quantizer_suffixes = (
        "weight_quantizer._scale",
        "weight_quantizer._double_scale",
        "weight_quantizer._amax",
        "input_quantizer._amax",
    )
    return {
        f"{prefix}.{suffix}"
        for prefix in _dense_decoder_nvfp4_quantizer_key_prefixes(model_dir, hf_weight_map)
        for suffix in quantizer_suffixes
    }


def _build_dense_decoder_nvfp4_expected_checkpoint_keys(
    model_dir: str,
    hf_weight_map: dict[str, str],
) -> set[str]:
    dims = _dense_decoder_nvfp4_dimensions(model_dir)
    checkpoint_keys = {
        "embedding.word_embeddings.weight",
        "decoder.final_layernorm.weight",
        "decoder.final_layernorm._extra_state/shard_0_1",
    }
    if "lm_head.weight" in hf_weight_map:
        checkpoint_keys.add("output_layer.weight")

    for layer_idx in range(dims["num_hidden_layers"]):
        layer_prefix = f"decoder.layers.{layer_idx}"
        checkpoint_keys.add(f"{layer_prefix}.self_attention.core_attention._extra_state/shard_0_1")

        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "input_layernorm.weight"):
            checkpoint_keys.add(f"{layer_prefix}.self_attention.linear_qkv.layer_norm_weight")
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "post_attention_layernorm.weight"):
            checkpoint_keys.add(f"{layer_prefix}.mlp.linear_fc1.layer_norm_weight")
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "self_attn.q_norm.weight"):
            checkpoint_keys.add(f"{layer_prefix}.self_attention.q_layernorm.weight")
            checkpoint_keys.add(f"{layer_prefix}.self_attention.q_layernorm._extra_state/shard_0_1")
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "self_attn.k_norm.weight"):
            checkpoint_keys.add(f"{layer_prefix}.self_attention.k_layernorm.weight")
            checkpoint_keys.add(f"{layer_prefix}.self_attention.k_layernorm._extra_state/shard_0_1")

        if _dense_decoder_nvfp4_all_hf_keys_present(
            hf_weight_map,
            layer_idx,
            (
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
            ),
        ):
            checkpoint_keys.add(f"{layer_prefix}.self_attention.linear_qkv.weight")
            checkpoint_keys.add(f"{layer_prefix}.self_attention.linear_qkv._extra_state/shard_0_1")
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "self_attn.o_proj.weight"):
            checkpoint_keys.add(f"{layer_prefix}.self_attention.linear_proj.weight")
            checkpoint_keys.add(f"{layer_prefix}.self_attention.linear_proj._extra_state/shard_0_1")
        if _dense_decoder_nvfp4_all_hf_keys_present(
            hf_weight_map,
            layer_idx,
            (
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
            ),
        ):
            checkpoint_keys.add(f"{layer_prefix}.mlp.linear_fc1.weight_w")
            checkpoint_keys.add(f"{layer_prefix}.mlp.linear_fc1.weight_v")
            checkpoint_keys.add(f"{layer_prefix}.mlp.linear_fc1._extra_state/shard_0_1")
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "mlp.down_proj.weight"):
            checkpoint_keys.add(f"{layer_prefix}.mlp.linear_fc2.weight")
            checkpoint_keys.add(f"{layer_prefix}.mlp.linear_fc2._extra_state/shard_0_1")
    return checkpoint_keys


def _build_dense_decoder_nvfp4_expected_shapes(
    model_dir: str,
    hf_weight_map: dict[str, str],
) -> dict[str, tuple[int, ...]]:
    dims = _dense_decoder_nvfp4_dimensions(model_dir)
    hidden_size = dims["hidden_size"]
    intermediate_size = dims["intermediate_size"]
    group_size = dims["group_size"]
    packed_hidden = hidden_size // 2
    shapes: dict[str, tuple[int, ...]] = {
        "embedding.word_embeddings.weight": (dims["vocab_size"], hidden_size),
        "decoder.final_layernorm.weight": (hidden_size,),
    }
    if "lm_head.weight" in hf_weight_map:
        shapes["output_layer.weight"] = (dims["vocab_size"], hidden_size)

    hidden_scale_cols = _dense_decoder_nvfp4_div(hidden_size, group_size, "hidden_size")
    q_size_scale_cols = _dense_decoder_nvfp4_div(dims["q_size"], group_size, "q_size")
    intermediate_scale_cols = _dense_decoder_nvfp4_div(
        intermediate_size,
        group_size,
        "intermediate_size",
    )

    for layer_idx in range(dims["num_hidden_layers"]):
        layer_prefix = f"decoder.layers.{layer_idx}"
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "input_layernorm.weight"):
            shapes[f"{layer_prefix}.self_attention.linear_qkv.layer_norm_weight"] = (hidden_size,)
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "post_attention_layernorm.weight"):
            shapes[f"{layer_prefix}.mlp.linear_fc1.layer_norm_weight"] = (hidden_size,)
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "self_attn.q_norm.weight"):
            shapes[f"{layer_prefix}.self_attention.q_layernorm.weight"] = (dims["head_dim"],)
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "self_attn.k_norm.weight"):
            shapes[f"{layer_prefix}.self_attention.k_layernorm.weight"] = (dims["head_dim"],)

        if _dense_decoder_nvfp4_all_hf_keys_present(
            hf_weight_map,
            layer_idx,
            (
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
            ),
        ):
            prefix = f"{layer_prefix}.self_attention.linear_qkv"
            shapes[f"{prefix}.weight"] = (dims["qkv_size"], packed_hidden)
            shapes[f"{prefix}.weight_quantizer._scale"] = (dims["qkv_size"], hidden_scale_cols)
            shapes[f"{prefix}.weight_quantizer._double_scale"] = ()
            shapes[f"{prefix}.weight_quantizer._amax"] = ()
            shapes[f"{prefix}.input_quantizer._amax"] = ()
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "self_attn.o_proj.weight"):
            prefix = f"{layer_prefix}.self_attention.linear_proj"
            shapes[f"{prefix}.weight"] = (hidden_size, dims["q_size"] // 2)
            shapes[f"{prefix}.weight_quantizer._scale"] = (hidden_size, q_size_scale_cols)
            shapes[f"{prefix}.weight_quantizer._double_scale"] = ()
            shapes[f"{prefix}.weight_quantizer._amax"] = ()
            shapes[f"{prefix}.input_quantizer._amax"] = ()
        if _dense_decoder_nvfp4_all_hf_keys_present(
            hf_weight_map,
            layer_idx,
            (
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
            ),
        ):
            prefix = f"{layer_prefix}.mlp.linear_fc1"
            shapes[f"{prefix}.weight_w"] = (intermediate_size, packed_hidden)
            shapes[f"{prefix}.weight_v"] = (intermediate_size, packed_hidden)
            shapes[f"{prefix}.weight_quantizer._scale"] = (intermediate_size * 2, hidden_scale_cols)
            shapes[f"{prefix}.weight_quantizer._double_scale"] = ()
            shapes[f"{prefix}.weight_quantizer._amax"] = ()
            shapes[f"{prefix}.input_quantizer._amax"] = ()
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "mlp.down_proj.weight"):
            prefix = f"{layer_prefix}.mlp.linear_fc2"
            shapes[f"{prefix}.weight"] = (hidden_size, intermediate_size // 2)
            shapes[f"{prefix}.weight_quantizer._scale"] = (hidden_size, intermediate_scale_cols)
            shapes[f"{prefix}.weight_quantizer._double_scale"] = ()
            shapes[f"{prefix}.weight_quantizer._amax"] = ()
            shapes[f"{prefix}.input_quantizer._amax"] = ()
    return shapes


def _find_missing_dense_decoder_nvfp4_hf_bundle_keys(
    model_dir: str,
    hf_weight_map: dict[str, str],
) -> list[str]:
    dims = _dense_decoder_nvfp4_dimensions(model_dir)
    missing_bundle_keys: list[str] = []
    quantized_roots = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    )
    bundle_suffixes = ("weight_scale", "weight_scale_2", "input_scale")
    for layer_idx in range(dims["num_hidden_layers"]):
        for root in quantized_roots:
            module_key = _dense_decoder_nvfp4_hf_key(layer_idx, root)
            if f"{module_key}.weight" not in hf_weight_map:
                continue
            missing_bundle_keys.extend(
                f"{module_key}.{suffix}"
                for suffix in bundle_suffixes
                if f"{module_key}.{suffix}" not in hf_weight_map
            )
    return sorted(missing_bundle_keys)


def _is_kimi_k25_hf_checkpoint(model_dir: str) -> bool:
    config = _load_optional_hf_config(model_dir)
    architectures = config.get("architectures")
    return config.get("model_type") == "kimi_k25" or (
        isinstance(architectures, list) and "KimiK25ForConditionalGeneration" in architectures
    )


def _kimi_nvfp4_dimensions(model_dir: str) -> dict[str, int]:
    config = _load_nvfp4_hf_config(model_dir)
    hidden_size = int(config["hidden_size"])
    num_attention_heads = int(config["num_attention_heads"])
    q_lora_rank = int(config["q_lora_rank"])
    kv_lora_rank = int(config["kv_lora_rank"])
    qk_nope_head_dim = int(config["qk_nope_head_dim"])
    qk_rope_head_dim = int(config["qk_rope_head_dim"])
    v_head_dim = int(config["v_head_dim"])
    routed_experts = int(config.get("n_routed_experts", config.get("num_experts", 0)))
    return {
        "num_hidden_layers": int(config["num_hidden_layers"]),
        "hidden_size": hidden_size,
        "intermediate_size": int(config["intermediate_size"]),
        "moe_intermediate_size": int(config["moe_intermediate_size"]),
        "vocab_size": int(config["vocab_size"]),
        "q_lora_rank": q_lora_rank,
        "kv_lora_rank": kv_lora_rank,
        "q_up_size": num_attention_heads * (qk_nope_head_dim + qk_rope_head_dim),
        "kv_down_size": kv_lora_rank + qk_rope_head_dim,
        "kv_up_size": num_attention_heads * (qk_nope_head_dim + v_head_dim),
        "linear_proj_in": num_attention_heads * v_head_dim,
        "routed_experts": routed_experts,
        "group_size": _dense_decoder_nvfp4_group_size(model_dir),
    }


def _kimi_nvfp4_hf_key(layer_idx: int, suffix: str) -> str:
    return f"language_model.model.layers.{layer_idx}.{suffix}"


def _kimi_nvfp4_hf_key_is_present(
    hf_weight_map: dict[str, str],
    layer_idx: int,
    suffix: str,
) -> bool:
    return _kimi_nvfp4_hf_key(layer_idx, suffix) in hf_weight_map


def _kimi_nvfp4_hf_any_key_present(hf_weight_map: dict[str, str], keys: tuple[str, ...]) -> bool:
    return any(key in hf_weight_map for key in keys)


def _kimi_nvfp4_hf_bundle_is_present(
    hf_weight_map: dict[str, str],
    module_key: str,
) -> bool:
    return all(
        key in hf_weight_map
        for key in (
            f"{module_key}.weight",
            f"{module_key}.weight_scale",
            f"{module_key}.weight_scale_2",
            f"{module_key}.input_scale",
        )
    )


def _kimi_nvfp4_hf_expert_indices(hf_weight_map: dict[str, str], layer_idx: int) -> set[int]:
    prefix = _kimi_nvfp4_hf_key(layer_idx, "mlp.experts.")
    indices: set[int] = set()
    for key in hf_weight_map:
        if not key.startswith(prefix):
            continue
        remainder = key[len(prefix) :]
        expert_idx, _, suffix = remainder.partition(".")
        if expert_idx.isdigit() and suffix in {
            "gate_proj.weight",
            "up_proj.weight",
            "down_proj.weight",
        }:
            indices.add(int(expert_idx))
    return indices


def _add_kimi_nvfp4_dense_mlp_expected_keys(
    *,
    checkpoint_keys: set[str],
    quantizer_keys: set[str],
    hf_layer_prefix: str,
    hf_weight_map: dict[str, str],
    megatron_mlp_prefix: str,
) -> None:
    fc1_hf_modules = (f"{hf_layer_prefix}.gate_proj", f"{hf_layer_prefix}.up_proj")
    if all(_kimi_nvfp4_hf_bundle_is_present(hf_weight_map, module_key) for module_key in fc1_hf_modules):
        fc1_prefix = f"{megatron_mlp_prefix}.linear_fc1"
        checkpoint_keys.add(f"{fc1_prefix}.weight_w")
        checkpoint_keys.add(f"{fc1_prefix}.weight_v")
        checkpoint_keys.add(f"{fc1_prefix}._extra_state/shard_0_1")
        _add_nvfp4_quantizer_keys(quantizer_keys, fc1_prefix)

    fc2_hf_module = f"{hf_layer_prefix}.down_proj"
    if _kimi_nvfp4_hf_bundle_is_present(hf_weight_map, fc2_hf_module):
        fc2_prefix = f"{megatron_mlp_prefix}.linear_fc2"
        checkpoint_keys.add(f"{fc2_prefix}.weight")
        checkpoint_keys.add(f"{fc2_prefix}._extra_state/shard_0_1")
        _add_nvfp4_quantizer_keys(quantizer_keys, fc2_prefix)


def _build_kimi_nvfp4_expected_checkpoint_keys(
    model_dir: str,
    hf_weight_map: dict[str, str],
) -> set[str]:
    dims = _kimi_nvfp4_dimensions(model_dir)
    checkpoint_keys = {
        "language_model.embedding.word_embeddings.weight",
        "language_model.decoder.final_layernorm.weight",
        "language_model.decoder.final_layernorm._extra_state/shard_0_1",
    }
    tied_word_embeddings = _int4_hf_uses_tied_word_embeddings(model_dir)
    if (
        not tied_word_embeddings
        and _kimi_nvfp4_hf_any_key_present(hf_weight_map, ("language_model.lm_head.weight",))
    ):
        checkpoint_keys.add("language_model.output_layer.weight")

    quantizer_keys: set[str] = set()
    for layer_idx in range(dims["num_hidden_layers"]):
        layer_prefix = f"language_model.decoder.layers.{layer_idx}"
        checkpoint_keys.add(f"{layer_prefix}.self_attention.core_attention._extra_state/shard_0_1")

        if _kimi_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "input_layernorm.weight"):
            checkpoint_keys.add(f"{layer_prefix}.input_layernorm.weight")
            checkpoint_keys.add(f"{layer_prefix}.input_layernorm._extra_state/shard_0_1")

        if _kimi_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "post_attention_layernorm.weight"):
            if _kimi_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "mlp.gate_proj.weight"):
                checkpoint_keys.add(f"{layer_prefix}.mlp.linear_fc1.layer_norm_weight")
            else:
                checkpoint_keys.add(f"{layer_prefix}.pre_mlp_layernorm.weight")
                checkpoint_keys.add(f"{layer_prefix}.pre_mlp_layernorm._extra_state/shard_0_1")

        if _kimi_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "self_attn.q_a_layernorm.weight"):
            checkpoint_keys.add(f"{layer_prefix}.self_attention.linear_q_up_proj.layer_norm_weight")
        if _kimi_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "self_attn.kv_a_layernorm.weight"):
            checkpoint_keys.add(f"{layer_prefix}.self_attention.linear_kv_up_proj.layer_norm_weight")

        attention_mappings = {
            "self_attn.q_a_proj.weight": "linear_q_down_proj",
            "self_attn.q_b_proj.weight": "linear_q_up_proj",
            "self_attn.kv_a_proj_with_mqa.weight": "linear_kv_down_proj",
            "self_attn.kv_b_proj.weight": "linear_kv_up_proj",
            "self_attn.o_proj.weight": "linear_proj",
        }
        for hf_suffix, megatron_module in attention_mappings.items():
            if _kimi_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, hf_suffix):
                prefix = f"{layer_prefix}.self_attention.{megatron_module}"
                checkpoint_keys.add(f"{prefix}.weight")
                checkpoint_keys.add(f"{prefix}._extra_state/shard_0_1")

        _add_kimi_nvfp4_dense_mlp_expected_keys(
            checkpoint_keys=checkpoint_keys,
            quantizer_keys=quantizer_keys,
            hf_layer_prefix=_kimi_nvfp4_hf_key(layer_idx, "mlp"),
            hf_weight_map=hf_weight_map,
            megatron_mlp_prefix=f"{layer_prefix}.mlp",
        )

        if _kimi_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "mlp.gate.weight"):
            checkpoint_keys.add(f"{layer_prefix}.mlp.router.weight")
        if _kimi_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "mlp.gate.e_score_correction_bias"):
            checkpoint_keys.add(f"{layer_prefix}.mlp.router.expert_bias")

        _add_kimi_nvfp4_dense_mlp_expected_keys(
            checkpoint_keys=checkpoint_keys,
            quantizer_keys=quantizer_keys,
            hf_layer_prefix=_kimi_nvfp4_hf_key(layer_idx, "mlp.shared_experts"),
            hf_weight_map=hf_weight_map,
            megatron_mlp_prefix=f"{layer_prefix}.mlp.shared_experts",
        )

        fc1_prefix = f"{layer_prefix}.mlp.experts.linear_fc1"
        fc2_prefix = f"{layer_prefix}.mlp.experts.linear_fc2"
        for expert_idx in sorted(_kimi_nvfp4_hf_expert_indices(hf_weight_map, layer_idx)):
            expert_hf_prefix = _kimi_nvfp4_hf_key(layer_idx, f"mlp.experts.{expert_idx}")
            if all(
                _kimi_nvfp4_hf_bundle_is_present(hf_weight_map, f"{expert_hf_prefix}.{projection}")
                for projection in ("gate_proj", "up_proj")
            ):
                checkpoint_keys.add(f"{fc1_prefix}.weight{expert_idx}_w")
                checkpoint_keys.add(f"{fc1_prefix}.weight{expert_idx}_v")
                _add_nvfp4_quantizer_keys(quantizer_keys, fc1_prefix, expert_idx=str(expert_idx))
                checkpoint_keys.add(
                    f"{layer_prefix}.mlp.experts.experts.{expert_idx}.linear_fc1._extra_state/shard_0_1"
                )
            if _kimi_nvfp4_hf_bundle_is_present(hf_weight_map, f"{expert_hf_prefix}.down_proj"):
                checkpoint_keys.add(f"{fc2_prefix}.weight{expert_idx}")
                _add_nvfp4_quantizer_keys(quantizer_keys, fc2_prefix, expert_idx=str(expert_idx))
                checkpoint_keys.add(
                    f"{layer_prefix}.mlp.experts.experts.{expert_idx}.linear_fc2._extra_state/shard_0_1"
                )

    return checkpoint_keys


def _build_kimi_nvfp4_expected_quantizer_keys(
    model_dir: str,
    hf_weight_map: dict[str, str],
) -> set[str]:
    dims = _kimi_nvfp4_dimensions(model_dir)
    checkpoint_keys: set[str] = set()
    quantizer_keys: set[str] = set()
    for layer_idx in range(dims["num_hidden_layers"]):
        layer_prefix = f"language_model.decoder.layers.{layer_idx}"
        _add_kimi_nvfp4_dense_mlp_expected_keys(
            checkpoint_keys=checkpoint_keys,
            quantizer_keys=quantizer_keys,
            hf_layer_prefix=_kimi_nvfp4_hf_key(layer_idx, "mlp"),
            hf_weight_map=hf_weight_map,
            megatron_mlp_prefix=f"{layer_prefix}.mlp",
        )
        _add_kimi_nvfp4_dense_mlp_expected_keys(
            checkpoint_keys=checkpoint_keys,
            quantizer_keys=quantizer_keys,
            hf_layer_prefix=_kimi_nvfp4_hf_key(layer_idx, "mlp.shared_experts"),
            hf_weight_map=hf_weight_map,
            megatron_mlp_prefix=f"{layer_prefix}.mlp.shared_experts",
        )
        fc1_prefix = f"{layer_prefix}.mlp.experts.linear_fc1"
        fc2_prefix = f"{layer_prefix}.mlp.experts.linear_fc2"
        for expert_idx in sorted(_kimi_nvfp4_hf_expert_indices(hf_weight_map, layer_idx)):
            expert_hf_prefix = _kimi_nvfp4_hf_key(layer_idx, f"mlp.experts.{expert_idx}")
            if all(
                _kimi_nvfp4_hf_bundle_is_present(hf_weight_map, f"{expert_hf_prefix}.{projection}")
                for projection in ("gate_proj", "up_proj")
            ):
                _add_nvfp4_quantizer_keys(quantizer_keys, fc1_prefix, expert_idx=str(expert_idx))
            if _kimi_nvfp4_hf_bundle_is_present(hf_weight_map, f"{expert_hf_prefix}.down_proj"):
                _add_nvfp4_quantizer_keys(quantizer_keys, fc2_prefix, expert_idx=str(expert_idx))
    return quantizer_keys


def _add_kimi_nvfp4_dense_mlp_expected_shapes(
    *,
    shapes: dict[str, tuple[int, ...]],
    hf_layer_prefix: str,
    hf_weight_map: dict[str, str],
    megatron_mlp_prefix: str,
    hidden_size: int,
    intermediate_size: int,
    group_size: int,
) -> None:
    hidden_scale_cols = _dense_decoder_nvfp4_div(hidden_size, group_size, "hidden_size")
    intermediate_scale_cols = _dense_decoder_nvfp4_div(
        intermediate_size,
        group_size,
        "intermediate_size",
    )
    fc1_hf_modules = (f"{hf_layer_prefix}.gate_proj", f"{hf_layer_prefix}.up_proj")
    if all(f"{module_key}.weight" in hf_weight_map for module_key in fc1_hf_modules):
        prefix = f"{megatron_mlp_prefix}.linear_fc1"
        shapes[f"{prefix}.weight_w"] = (intermediate_size, hidden_size // 2)
        shapes[f"{prefix}.weight_v"] = (intermediate_size, hidden_size // 2)
        shapes[f"{prefix}.weight_quantizer._scale"] = (intermediate_size * 2, hidden_scale_cols)
        shapes[f"{prefix}.weight_quantizer._double_scale"] = ()
        shapes[f"{prefix}.weight_quantizer._amax"] = ()
        shapes[f"{prefix}.input_quantizer._amax"] = ()

    if f"{hf_layer_prefix}.down_proj.weight" in hf_weight_map:
        prefix = f"{megatron_mlp_prefix}.linear_fc2"
        shapes[f"{prefix}.weight"] = (hidden_size, intermediate_size // 2)
        shapes[f"{prefix}.weight_quantizer._scale"] = (hidden_size, intermediate_scale_cols)
        shapes[f"{prefix}.weight_quantizer._double_scale"] = ()
        shapes[f"{prefix}.weight_quantizer._amax"] = ()
        shapes[f"{prefix}.input_quantizer._amax"] = ()


def _build_kimi_nvfp4_expected_shapes(
    model_dir: str,
    hf_weight_map: dict[str, str],
) -> dict[str, tuple[int, ...]]:
    dims = _kimi_nvfp4_dimensions(model_dir)
    hidden_size = dims["hidden_size"]
    group_size = dims["group_size"]
    shapes: dict[str, tuple[int, ...]] = {
        "language_model.embedding.word_embeddings.weight": (dims["vocab_size"], hidden_size),
        "language_model.decoder.final_layernorm.weight": (hidden_size,),
    }
    tied_word_embeddings = _int4_hf_uses_tied_word_embeddings(model_dir)
    if (
        not tied_word_embeddings
        and _kimi_nvfp4_hf_any_key_present(hf_weight_map, ("language_model.lm_head.weight",))
    ):
        shapes["language_model.output_layer.weight"] = (dims["vocab_size"], hidden_size)

    for layer_idx in range(dims["num_hidden_layers"]):
        layer_prefix = f"language_model.decoder.layers.{layer_idx}"
        if _kimi_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "input_layernorm.weight"):
            shapes[f"{layer_prefix}.input_layernorm.weight"] = (hidden_size,)
        if _kimi_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "post_attention_layernorm.weight"):
            if _kimi_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "mlp.gate_proj.weight"):
                shapes[f"{layer_prefix}.mlp.linear_fc1.layer_norm_weight"] = (hidden_size,)
            else:
                shapes[f"{layer_prefix}.pre_mlp_layernorm.weight"] = (hidden_size,)
        if _kimi_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "self_attn.q_a_layernorm.weight"):
            shapes[f"{layer_prefix}.self_attention.linear_q_up_proj.layer_norm_weight"] = (dims["q_lora_rank"],)
        if _kimi_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "self_attn.kv_a_layernorm.weight"):
            shapes[f"{layer_prefix}.self_attention.linear_kv_up_proj.layer_norm_weight"] = (
                dims["kv_lora_rank"],
            )

        attention_shapes = {
            "self_attn.q_a_proj.weight": ("linear_q_down_proj", (dims["q_lora_rank"], hidden_size)),
            "self_attn.q_b_proj.weight": ("linear_q_up_proj", (dims["q_up_size"], dims["q_lora_rank"])),
            "self_attn.kv_a_proj_with_mqa.weight": (
                "linear_kv_down_proj",
                (dims["kv_down_size"], hidden_size),
            ),
            "self_attn.kv_b_proj.weight": ("linear_kv_up_proj", (dims["kv_up_size"], dims["kv_lora_rank"])),
            "self_attn.o_proj.weight": ("linear_proj", (hidden_size, dims["linear_proj_in"])),
        }
        for hf_suffix, (megatron_module, shape) in attention_shapes.items():
            if _kimi_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, hf_suffix):
                shapes[f"{layer_prefix}.self_attention.{megatron_module}.weight"] = shape

        _add_kimi_nvfp4_dense_mlp_expected_shapes(
            shapes=shapes,
            hf_layer_prefix=_kimi_nvfp4_hf_key(layer_idx, "mlp"),
            hf_weight_map=hf_weight_map,
            megatron_mlp_prefix=f"{layer_prefix}.mlp",
            hidden_size=hidden_size,
            intermediate_size=dims["intermediate_size"],
            group_size=group_size,
        )

        if _kimi_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "mlp.gate.weight"):
            shapes[f"{layer_prefix}.mlp.router.weight"] = (dims["routed_experts"], hidden_size)
        if _kimi_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "mlp.gate.e_score_correction_bias"):
            shapes[f"{layer_prefix}.mlp.router.expert_bias"] = (dims["routed_experts"],)

        _add_kimi_nvfp4_dense_mlp_expected_shapes(
            shapes=shapes,
            hf_layer_prefix=_kimi_nvfp4_hf_key(layer_idx, "mlp.shared_experts"),
            hf_weight_map=hf_weight_map,
            megatron_mlp_prefix=f"{layer_prefix}.mlp.shared_experts",
            hidden_size=hidden_size,
            intermediate_size=dims["moe_intermediate_size"],
            group_size=group_size,
        )

        hidden_scale_cols = _dense_decoder_nvfp4_div(hidden_size, group_size, "hidden_size")
        moe_scale_cols = _dense_decoder_nvfp4_div(
            dims["moe_intermediate_size"],
            group_size,
            "moe_intermediate_size",
        )
        for expert_idx in sorted(_kimi_nvfp4_hf_expert_indices(hf_weight_map, layer_idx)):
            expert_hf_prefix = _kimi_nvfp4_hf_key(layer_idx, f"mlp.experts.{expert_idx}")
            if all(
                f"{expert_hf_prefix}.{projection}.weight" in hf_weight_map
                for projection in ("gate_proj", "up_proj")
            ):
                prefix = f"{layer_prefix}.mlp.experts.linear_fc1"
                shapes[f"{prefix}.weight{expert_idx}_w"] = (
                    dims["moe_intermediate_size"],
                    hidden_size // 2,
                )
                shapes[f"{prefix}.weight{expert_idx}_v"] = (
                    dims["moe_intermediate_size"],
                    hidden_size // 2,
                )
                shapes[f"{prefix}.weight_quantizer._scale{expert_idx}"] = (
                    dims["moe_intermediate_size"] * 2,
                    hidden_scale_cols,
                )
                shapes[f"{prefix}.weight_quantizer._double_scale{expert_idx}"] = ()
                shapes[f"{prefix}.weight_quantizer._amax{expert_idx}"] = ()
                shapes[f"{prefix}.input_quantizer._amax"] = ()
            if f"{expert_hf_prefix}.down_proj.weight" in hf_weight_map:
                prefix = f"{layer_prefix}.mlp.experts.linear_fc2"
                shapes[f"{prefix}.weight{expert_idx}"] = (hidden_size, dims["moe_intermediate_size"] // 2)
                shapes[f"{prefix}.weight_quantizer._scale{expert_idx}"] = (hidden_size, moe_scale_cols)
                shapes[f"{prefix}.weight_quantizer._double_scale{expert_idx}"] = ()
                shapes[f"{prefix}.weight_quantizer._amax{expert_idx}"] = ()
                shapes[f"{prefix}.input_quantizer._amax"] = ()
    return shapes


def _find_missing_kimi_nvfp4_hf_bundle_keys(
    model_dir: str,
    hf_weight_map: dict[str, str],
) -> list[str]:
    dims = _kimi_nvfp4_dimensions(model_dir)
    missing_bundle_keys: list[str] = []
    for layer_idx in range(dims["num_hidden_layers"]):
        module_roots = [
            _kimi_nvfp4_hf_key(layer_idx, f"mlp.{projection}")
            for projection in ("gate_proj", "up_proj", "down_proj")
        ]
        module_roots.extend(
            _kimi_nvfp4_hf_key(layer_idx, f"mlp.shared_experts.{projection}")
            for projection in ("gate_proj", "up_proj", "down_proj")
        )
        for expert_idx in sorted(_kimi_nvfp4_hf_expert_indices(hf_weight_map, layer_idx)):
            module_roots.extend(
                _kimi_nvfp4_hf_key(layer_idx, f"mlp.experts.{expert_idx}.{projection}")
                for projection in ("gate_proj", "up_proj", "down_proj")
            )

        for module_key in module_roots:
            if f"{module_key}.weight" not in hf_weight_map:
                continue
            missing_bundle_keys.extend(
                key
                for key in (
                    f"{module_key}.weight_scale",
                    f"{module_key}.weight_scale_2",
                    f"{module_key}.input_scale",
                )
                if key not in hf_weight_map
            )
    return sorted(missing_bundle_keys)


def _load_int4_hf_config(model_dir: str) -> dict[str, object]:
    config = _load_hf_config(model_dir)
    text_config = config.get("text_config")
    if not isinstance(text_config, dict):
        return config

    merged_config = dict(text_config)
    for inherited_key in ("quantization_config", "vocab_size"):
        if inherited_key not in merged_config and inherited_key in config:
            merged_config[inherited_key] = config[inherited_key]
    return merged_config


def _dense_decoder_int4_group_size(model_dir: str) -> int:
    config = _load_int4_hf_config(model_dir)
    quantization_config = config.get("quantization_config", {})
    if not isinstance(quantization_config, dict):
        return 32
    if "group_size" in quantization_config:
        return int(quantization_config["group_size"])

    for group in (quantization_config.get("config_groups", {}) or {}).values():
        if not isinstance(group, dict):
            continue
        weights_config = group.get("weights", {})
        if isinstance(weights_config, dict) and "group_size" in weights_config:
            return int(weights_config["group_size"])
        if "group_size" in group:
            return int(group["group_size"])

    for weights_key in ("weights", "weight_quantization"):
        weights_config = quantization_config.get(weights_key, {})
        if isinstance(weights_config, dict) and "group_size" in weights_config:
            return int(weights_config["group_size"])
    return 32


def _dense_decoder_int4_dimensions(model_dir: str) -> dict[str, int]:
    config = _load_int4_hf_config(model_dir)
    hidden_size = int(config["hidden_size"])
    num_attention_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(config.get("num_key_value_heads", num_attention_heads))
    head_dim = int(config.get("head_dim", hidden_size // num_attention_heads))
    q_size = num_attention_heads * head_dim
    kv_size = num_key_value_heads * head_dim
    return {
        "num_hidden_layers": int(config["num_hidden_layers"]),
        "hidden_size": hidden_size,
        "intermediate_size": int(config["intermediate_size"]),
        "moe_intermediate_size": int(config.get("moe_intermediate_size", config["intermediate_size"])),
        "vocab_size": int(config["vocab_size"]),
        "head_dim": head_dim,
        "q_size": q_size,
        "kv_size": kv_size,
        "qkv_size": q_size + kv_size + kv_size,
        "group_size": _dense_decoder_int4_group_size(model_dir),
    }


def _dense_decoder_int4_div(value: int, divisor: int, name: str) -> int:
    if value % divisor != 0:
        raise ValueError(f"INT4 {name}={value} must be divisible by {divisor}.")
    return value // divisor


def _dense_decoder_int4_hf_key(layer_idx: int, suffix: str) -> str:
    return f"model.layers.{layer_idx}.{suffix}"


def _dense_decoder_int4_hf_weight_is_present(
    hf_weight_map: dict[str, str],
    layer_idx: int,
    suffix: str,
) -> bool:
    weight_key = _dense_decoder_int4_hf_key(layer_idx, suffix)
    return weight_key in hf_weight_map or f"{weight_key}_packed" in hf_weight_map


def _dense_decoder_int4_all_hf_weights_present(
    hf_weight_map: dict[str, str],
    layer_idx: int,
    suffixes: tuple[str, ...],
) -> bool:
    return all(_dense_decoder_int4_hf_weight_is_present(hf_weight_map, layer_idx, suffix) for suffix in suffixes)


def _int4_hf_any_key_present(hf_weight_map: dict[str, str], keys: tuple[str, ...]) -> bool:
    return any(key in hf_weight_map for key in keys)


def _int4_hf_uses_tied_word_embeddings(model_dir: str) -> bool:
    config = _load_hf_config(model_dir)
    text_config = config.get("text_config")
    if "tie_word_embeddings" in config:
        return bool(config["tie_word_embeddings"])
    if isinstance(text_config, dict) and "tie_word_embeddings" in text_config:
        return bool(text_config["tie_word_embeddings"])
    return False


def _int4_hf_uses_qwen3_moe_megatron_layout(model_dir: str) -> bool:
    config = _load_int4_hf_config(model_dir)
    architectures = config.get("architectures")
    return config.get("model_type") == "qwen3_moe" or (
        isinstance(architectures, list) and "Qwen3MoeForCausalLM" in architectures
    )


def _int4_megatron_checkpoint_key_prefix(model_dir: str) -> str:
    config = _load_hf_config(model_dir)
    architectures = config.get("architectures")
    if config.get("model_type") == "kimi_k25" or (
        isinstance(architectures, list) and "KimiK25ForConditionalGeneration" in architectures
    ):
        return "language_model."
    return ""


def _prefix_int4_megatron_checkpoint_keys(model_dir: str, checkpoint_keys: set[str]) -> set[str]:
    prefix = _int4_megatron_checkpoint_key_prefix(model_dir)
    if not prefix:
        return checkpoint_keys
    return {f"{prefix}{key}" for key in checkpoint_keys}


def _prefix_int4_megatron_checkpoint_shapes(
    model_dir: str,
    checkpoint_shapes: dict[str, tuple[int, ...]],
) -> dict[str, tuple[int, ...]]:
    prefix = _int4_megatron_checkpoint_key_prefix(model_dir)
    if not prefix:
        return checkpoint_shapes
    return {f"{prefix}{key}": shape for key, shape in checkpoint_shapes.items()}


def _int4_checkpoint_triplet_key(prefix: str, suffix: str) -> str:
    if re.search(r"\.weight\d+$", prefix):
        return f"{prefix}_{suffix}"
    return f"{prefix}.weight_{suffix}"


def _add_int4_triplet_keys(checkpoint_keys: set[str], prefix: str) -> None:
    checkpoint_keys.add(_int4_checkpoint_triplet_key(prefix, "packed"))
    checkpoint_keys.add(_int4_checkpoint_triplet_key(prefix, "scale"))
    checkpoint_keys.add(_int4_checkpoint_triplet_key(prefix, "shape"))


def _add_int4_layer_extra_state_key(
    checkpoint_keys: set[str],
    *,
    dims: dict[str, int],
    layer_idx: int,
    suffix: str,
    per_layer: bool = False,
) -> None:
    if per_layer:
        checkpoint_keys.add(f"decoder.layers.{layer_idx}.{suffix}._extra_state/shard_0_1")
        return
    checkpoint_keys.add(f"decoder.layers.{suffix}._extra_state/shard_{layer_idx}_{dims['num_hidden_layers']}")


def _add_int4_triplet_shapes(
    shapes: dict[str, tuple[int, ...]],
    *,
    prefix: str,
    out_features: int,
    in_features: int,
    group_size: int,
    name: str,
) -> None:
    shapes[_int4_checkpoint_triplet_key(prefix, "packed")] = (
        out_features,
        _dense_decoder_int4_div(in_features, 8, f"{name} in_features"),
    )
    shapes[_int4_checkpoint_triplet_key(prefix, "scale")] = (
        out_features,
        _dense_decoder_int4_div(in_features, group_size, f"{name} in_features"),
    )
    shapes[_int4_checkpoint_triplet_key(prefix, "shape")] = (2,)


def _build_dense_decoder_int4_expected_checkpoint_keys(
    model_dir: str,
    hf_weight_map: dict[str, str],
) -> set[str]:
    dims = _dense_decoder_int4_dimensions(model_dir)
    qwen3_moe_layout = _int4_hf_uses_qwen3_moe_megatron_layout(model_dir)
    checkpoint_keys = {
        "embedding.word_embeddings.weight",
        "decoder.final_layernorm.weight",
        "decoder.final_layernorm._extra_state/shard_0_1",
    }
    tied_word_embeddings = _int4_hf_uses_tied_word_embeddings(model_dir)
    if (
        not tied_word_embeddings
        and _int4_hf_any_key_present(hf_weight_map, ("lm_head.weight", "language_model.lm_head.weight"))
    ):
        checkpoint_keys.add("output_layer.weight")
    if (
        not tied_word_embeddings
        and _int4_hf_any_key_present(hf_weight_map, ("lm_head.weight_packed", "language_model.lm_head.weight_packed"))
    ):
        _add_int4_triplet_keys(checkpoint_keys, "output_layer")

    for layer_idx in range(dims["num_hidden_layers"]):
        layer_prefix = f"decoder.layers.{layer_idx}"

        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "input_layernorm.weight"):
            checkpoint_keys.add(f"{layer_prefix}.self_attention.linear_qkv.layer_norm_weight")
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "post_attention_layernorm.weight"):
            if qwen3_moe_layout:
                checkpoint_keys.add(f"{layer_prefix}.pre_mlp_layernorm.weight")
            else:
                checkpoint_keys.add(f"{layer_prefix}.mlp.linear_fc1.layer_norm_weight")
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "self_attn.q_norm.weight"):
            checkpoint_keys.add(f"{layer_prefix}.self_attention.q_layernorm.weight")
            _add_int4_layer_extra_state_key(
                checkpoint_keys,
                dims=dims,
                layer_idx=layer_idx,
                suffix="self_attention.q_layernorm",
                per_layer=qwen3_moe_layout,
            )
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "self_attn.k_norm.weight"):
            checkpoint_keys.add(f"{layer_prefix}.self_attention.k_layernorm.weight")
            _add_int4_layer_extra_state_key(
                checkpoint_keys,
                dims=dims,
                layer_idx=layer_idx,
                suffix="self_attention.k_layernorm",
                per_layer=qwen3_moe_layout,
            )

        if _dense_decoder_int4_all_hf_weights_present(
            hf_weight_map,
            layer_idx,
            (
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
            ),
        ):
            prefix = f"{layer_prefix}.self_attention.linear_qkv"
            _add_int4_triplet_keys(checkpoint_keys, prefix)
            _add_int4_layer_extra_state_key(
                checkpoint_keys,
                dims=dims,
                layer_idx=layer_idx,
                suffix="self_attention.linear_qkv",
                per_layer=qwen3_moe_layout,
            )
        if _dense_decoder_int4_hf_weight_is_present(hf_weight_map, layer_idx, "self_attn.o_proj.weight"):
            prefix = f"{layer_prefix}.self_attention.linear_proj"
            _add_int4_triplet_keys(checkpoint_keys, prefix)
            _add_int4_layer_extra_state_key(
                checkpoint_keys,
                dims=dims,
                layer_idx=layer_idx,
                suffix="self_attention.linear_proj",
                per_layer=qwen3_moe_layout,
            )
        if _dense_decoder_int4_all_hf_weights_present(
            hf_weight_map,
            layer_idx,
            (
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
            ),
        ):
            prefix = f"{layer_prefix}.mlp.linear_fc1"
            _add_int4_triplet_keys(checkpoint_keys, prefix)
            _add_int4_layer_extra_state_key(
                checkpoint_keys,
                dims=dims,
                layer_idx=layer_idx,
                suffix="mlp.linear_fc1",
            )
        if _dense_decoder_int4_hf_weight_is_present(hf_weight_map, layer_idx, "mlp.down_proj.weight"):
            prefix = f"{layer_prefix}.mlp.linear_fc2"
            _add_int4_triplet_keys(checkpoint_keys, prefix)
            _add_int4_layer_extra_state_key(
                checkpoint_keys,
                dims=dims,
                layer_idx=layer_idx,
                suffix="mlp.linear_fc2",
            )
    return checkpoint_keys


def _build_dense_decoder_int4_expected_shapes(
    model_dir: str,
    hf_weight_map: dict[str, str],
) -> dict[str, tuple[int, ...]]:
    dims = _dense_decoder_int4_dimensions(model_dir)
    qwen3_moe_layout = _int4_hf_uses_qwen3_moe_megatron_layout(model_dir)
    hidden_size = dims["hidden_size"]
    intermediate_size = dims["intermediate_size"]
    group_size = dims["group_size"]
    shapes: dict[str, tuple[int, ...]] = {
        "embedding.word_embeddings.weight": (dims["vocab_size"], hidden_size),
        "decoder.final_layernorm.weight": (hidden_size,),
    }
    tied_word_embeddings = _int4_hf_uses_tied_word_embeddings(model_dir)
    if (
        not tied_word_embeddings
        and _int4_hf_any_key_present(hf_weight_map, ("lm_head.weight", "language_model.lm_head.weight"))
    ):
        shapes["output_layer.weight"] = (dims["vocab_size"], hidden_size)
    if (
        not tied_word_embeddings
        and _int4_hf_any_key_present(hf_weight_map, ("lm_head.weight_packed", "language_model.lm_head.weight_packed"))
    ):
        _add_int4_triplet_shapes(
            shapes,
            prefix="output_layer",
            out_features=dims["vocab_size"],
            in_features=hidden_size,
            group_size=group_size,
            name="output_layer",
        )

    for layer_idx in range(dims["num_hidden_layers"]):
        layer_prefix = f"decoder.layers.{layer_idx}"
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "input_layernorm.weight"):
            shapes[f"{layer_prefix}.self_attention.linear_qkv.layer_norm_weight"] = (hidden_size,)
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "post_attention_layernorm.weight"):
            if qwen3_moe_layout:
                shapes[f"{layer_prefix}.pre_mlp_layernorm.weight"] = (hidden_size,)
            else:
                shapes[f"{layer_prefix}.mlp.linear_fc1.layer_norm_weight"] = (hidden_size,)
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "self_attn.q_norm.weight"):
            shapes[f"{layer_prefix}.self_attention.q_layernorm.weight"] = (dims["head_dim"],)
        if _dense_decoder_nvfp4_hf_key_is_present(hf_weight_map, layer_idx, "self_attn.k_norm.weight"):
            shapes[f"{layer_prefix}.self_attention.k_layernorm.weight"] = (dims["head_dim"],)

        if _dense_decoder_int4_all_hf_weights_present(
            hf_weight_map,
            layer_idx,
            (
                "self_attn.q_proj.weight",
                "self_attn.k_proj.weight",
                "self_attn.v_proj.weight",
            ),
        ):
            _add_int4_triplet_shapes(
                shapes,
                prefix=f"{layer_prefix}.self_attention.linear_qkv",
                out_features=dims["qkv_size"],
                in_features=hidden_size,
                group_size=group_size,
                name="linear_qkv",
            )
        if _dense_decoder_int4_hf_weight_is_present(hf_weight_map, layer_idx, "self_attn.o_proj.weight"):
            _add_int4_triplet_shapes(
                shapes,
                prefix=f"{layer_prefix}.self_attention.linear_proj",
                out_features=hidden_size,
                in_features=dims["q_size"],
                group_size=group_size,
                name="linear_proj",
            )
        if _dense_decoder_int4_all_hf_weights_present(
            hf_weight_map,
            layer_idx,
            (
                "mlp.gate_proj.weight",
                "mlp.up_proj.weight",
            ),
        ):
            _add_int4_triplet_shapes(
                shapes,
                prefix=f"{layer_prefix}.mlp.linear_fc1",
                out_features=intermediate_size * 2,
                in_features=hidden_size,
                group_size=group_size,
                name="linear_fc1",
            )
        if _dense_decoder_int4_hf_weight_is_present(hf_weight_map, layer_idx, "mlp.down_proj.weight"):
            _add_int4_triplet_shapes(
                shapes,
                prefix=f"{layer_prefix}.mlp.linear_fc2",
                out_features=hidden_size,
                in_features=intermediate_size,
                group_size=group_size,
                name="linear_fc2",
            )
    return shapes


_INT4_HF_TRIPLET_KEY_RE = re.compile(r"^(?P<weight_key>.+\.weight)_(?P<suffix>packed|scale|shape)$")
_INT4_HF_EXPERT_TRIPLET_KEY_RE = re.compile(
    r"^(?P<prefix>.*model\.layers\.)(?P<layer_idx>\d+)\.mlp\.experts\."
    r"(?P<expert_idx>\d+)\.(?P<projection>gate_proj|up_proj|down_proj)\.weight_"
    r"(?P<suffix>packed|scale|shape)$"
)


def _find_missing_int4_hf_bundle_keys(hf_weight_map: dict[str, str]) -> list[str]:
    triplet_suffixes = ("packed", "scale", "shape")
    present_suffixes_by_weight_key: dict[str, set[str]] = {}
    for key in hf_weight_map:
        match = _INT4_HF_TRIPLET_KEY_RE.match(key)
        if match is None:
            continue
        present_suffixes_by_weight_key.setdefault(match.group("weight_key"), set()).add(match.group("suffix"))

    return sorted(
        f"{weight_key}_{suffix}"
        for weight_key, present_suffixes in present_suffixes_by_weight_key.items()
        for suffix in triplet_suffixes
        if suffix not in present_suffixes
    )


def _collect_int4_hf_expert_triplets(
    hf_weight_map: dict[str, str],
) -> dict[tuple[int, int, str], set[str]]:
    triplets_by_expert_projection: dict[tuple[int, int, str], set[str]] = {}
    for key in hf_weight_map:
        match = _INT4_HF_EXPERT_TRIPLET_KEY_RE.match(key)
        if match is None:
            continue
        expert_key = (
            int(match.group("layer_idx")),
            int(match.group("expert_idx")),
            match.group("projection"),
        )
        triplets_by_expert_projection.setdefault(expert_key, set()).add(match.group("suffix"))
    return triplets_by_expert_projection


def _build_moe_expert_int4_expected_checkpoint_keys(
    hf_weight_map: dict[str, str],
) -> set[str]:
    expert_triplets = _collect_int4_hf_expert_triplets(hf_weight_map)
    checkpoint_keys: set[str] = set()
    layer_expert_pairs = {(layer_idx, expert_idx) for layer_idx, expert_idx, _ in expert_triplets}
    for layer_idx, expert_idx in sorted(layer_expert_pairs):
        layer_prefix = f"decoder.layers.{layer_idx}.mlp.experts"
        if (
            (layer_idx, expert_idx, "gate_proj") in expert_triplets
            or (layer_idx, expert_idx, "up_proj") in expert_triplets
        ):
            _add_int4_triplet_keys(checkpoint_keys, f"{layer_prefix}.linear_fc1.weight{expert_idx}")
        if (layer_idx, expert_idx, "down_proj") in expert_triplets:
            _add_int4_triplet_keys(checkpoint_keys, f"{layer_prefix}.linear_fc2.weight{expert_idx}")
    return checkpoint_keys


def _build_moe_expert_int4_expected_shapes(
    model_dir: str,
    hf_weight_map: dict[str, str],
) -> dict[str, tuple[int, ...]]:
    dims = _dense_decoder_int4_dimensions(model_dir)
    hidden_size = dims["hidden_size"]
    moe_intermediate_size = dims["moe_intermediate_size"]
    group_size = dims["group_size"]
    shapes: dict[str, tuple[int, ...]] = {}
    expert_triplets = _collect_int4_hf_expert_triplets(hf_weight_map)
    layer_expert_pairs = {(layer_idx, expert_idx) for layer_idx, expert_idx, _ in expert_triplets}
    for layer_idx, expert_idx in sorted(layer_expert_pairs):
        layer_prefix = f"decoder.layers.{layer_idx}.mlp.experts"
        if (
            (layer_idx, expert_idx, "gate_proj") in expert_triplets
            or (layer_idx, expert_idx, "up_proj") in expert_triplets
        ):
            _add_int4_triplet_shapes(
                shapes,
                prefix=f"{layer_prefix}.linear_fc1.weight{expert_idx}",
                out_features=moe_intermediate_size * 2,
                in_features=hidden_size,
                group_size=group_size,
                name="expert linear_fc1",
            )
        if (layer_idx, expert_idx, "down_proj") in expert_triplets:
            _add_int4_triplet_shapes(
                shapes,
                prefix=f"{layer_prefix}.linear_fc2.weight{expert_idx}",
                out_features=hidden_size,
                in_features=moe_intermediate_size,
                group_size=group_size,
                name="expert linear_fc2",
            )
    return shapes


def _dense_qwen3_dcp_scale_key_is_present(model_dir: str, scale_key: str, checkpoint_keys: set[str]) -> bool:
    if scale_key in checkpoint_keys:
        return True

    config = _load_hf_config(model_dir)
    num_hidden_layers = int(config["num_hidden_layers"])
    if not scale_key.startswith("decoder.layers."):
        return False

    suffix = scale_key[len("decoder.layers.") :]
    return all(f"decoder.layers.{layer_idx}.{suffix}" in checkpoint_keys for layer_idx in range(num_hidden_layers))


def _find_missing_dense_qwen3_dcp_scale_keys(
    model_dir: str,
    checkpoint_keys: set[str],
    expected_scale_suffixes: tuple[str, ...],
) -> list[str]:
    missing_scale_keys: list[str] = []
    for scale_key in sorted(_build_dense_qwen3_expected_dcp_scale_keys(model_dir, expected_scale_suffixes[0])):
        if _dense_qwen3_dcp_scale_key_is_present(model_dir, scale_key, checkpoint_keys):
            continue
        alternate_present = any(
            _dense_qwen3_dcp_scale_key_is_present(
                model_dir,
                scale_key[: -len(expected_scale_suffixes[0])] + alternate_suffix,
                checkpoint_keys,
            )
            for alternate_suffix in expected_scale_suffixes[1:]
        )
        if not alternate_present:
            missing_scale_keys.append(scale_key)
    return missing_scale_keys


def _load_mcore_bridge_module():
    try:
        from megatron.bridge import AutoBridge
    except ImportError as exc:
        raise ImportError(
            "Megatron-Bridge is required for non-dense FP8 parity checks. "
            "Install megatron-bridge or set MEGATRON_BRIDGE_ROOT before running this tool."
        ) from exc

    return SimpleNamespace(AutoBridge=AutoBridge)


def _unwrap_model_for_parity(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def _iter_unwrapped_models_for_parity(model):
    if isinstance(model, (list, tuple)):
        for model_chunk in model:
            yield from _iter_unwrapped_models_for_parity(model_chunk)
        return

    yield _unwrap_model_for_parity(model)


def _build_expected_megatron_state_dict(hf_model_path: str) -> dict[str, object]:
    bridge_module = _load_mcore_bridge_module()
    bridge = bridge_module.AutoBridge.from_hf_pretrained(hf_model_path)
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.perform_initialization = False
    provider.finalize()
    model = provider.provide_distributed_model(
        wrap_with_ddp=False,
        use_cpu_initialization=True,
        mixed_precision_wrapper=None,
    )
    state_dict: dict[str, object] = {}
    for model_chunk in _iter_unwrapped_models_for_parity(model):
        state_dict.update(model_chunk.sharded_state_dict())
    return state_dict


def build_expected_megatron_keyset(hf_model_path: str) -> set[str]:
    return {str(key) for key in _build_expected_megatron_state_dict(hf_model_path).keys()}


def _tensor_shape(value: object) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        size = getattr(value, "size", None)
        if callable(size):
            shape = size()
        elif size is not None:
            shape = size
    if shape is None:
        return None
    return tuple(int(dim) for dim in tuple(shape))


def _canonical_extra_state_key(key: str) -> str:
    key = key.replace(".mlp.experts.experts.", ".mlp.experts.")
    shard_match = re.match(r"(.+\._extra_state)/shard_(\d+)_(\d+)$", key)
    if shard_match:
        base_key, shard_idx, shard_count = shard_match.groups()
        if int(shard_count) == 1:
            return base_key
        return f"{base_key}/shard_{int(shard_idx)}"

    indexed_match = re.match(r"(.+\._extra_state)(\d+)$", key)
    if indexed_match:
        base_key, shard_idx = indexed_match.groups()
        return f"{base_key}/shard_{int(shard_idx)}"

    return key


def _checkpoint_extra_state_key_is_present(
    expected_key: str,
    checkpoint_keys: set[str],
    checkpoint_canonical_extra_state_keys: set[str] | None = None,
) -> bool:
    if "._extra_state" not in expected_key:
        return False
    if expected_key in checkpoint_keys:
        return True

    if checkpoint_canonical_extra_state_keys is None:
        checkpoint_canonical_extra_state_keys = {
            _canonical_extra_state_key(key) for key in checkpoint_keys if "._extra_state" in key
        }

    expected_extra_state_key = _canonical_extra_state_key(expected_key)
    if expected_extra_state_key in checkpoint_canonical_extra_state_keys:
        return True
    return any(
        checkpoint_key.startswith(f"{expected_extra_state_key}/shard_")
        for checkpoint_key in checkpoint_canonical_extra_state_keys
    )


def _missing_expected_checkpoint_keys(expected_checkpoint_keys: set[str], checkpoint_keys: set[str]) -> list[str]:
    checkpoint_canonical_extra_state_keys = {
        _canonical_extra_state_key(key) for key in checkpoint_keys if "._extra_state" in key
    }
    missing_keys: list[str] = []
    for expected_key in sorted(expected_checkpoint_keys):
        if expected_key in checkpoint_keys:
            continue
        if _checkpoint_extra_state_key_is_present(
            expected_key,
            checkpoint_keys,
            checkpoint_canonical_extra_state_keys,
        ):
            continue
        missing_keys.append(expected_key)
    return missing_keys


def _dense_qwen3_layer_extra_state_alias(expected_key: str, num_hidden_layers: int) -> str | None:
    match = re.match(r"decoder\.layers\.(.+\._extra_state)/shard_(\d+)_(\d+)$", expected_key)
    if match is None:
        return None

    suffix, layer_idx, layer_count = match.groups()
    layer_idx_int = int(layer_idx)
    if int(layer_count) != num_hidden_layers or layer_idx_int >= num_hidden_layers:
        return None
    return f"decoder.layers.{layer_idx_int}.{suffix}"


def _modelopt_weight_w_alias(key: str) -> str | None:
    if not key.endswith(".weight"):
        return None
    return f"{key}_w"


def _dense_qwen3_fc1_fused_alias(key: str) -> str | None:
    for suffix in (".mlp.linear_fc1.weight_w", ".mlp.linear_fc1.weight_v"):
        if key.endswith(suffix):
            return f"{key[:-len(suffix)]}.mlp.linear_fc1.weight"
    return None


def _dense_qwen3_fc1_fused_shape(split_shape: tuple[int, ...]) -> tuple[int, ...] | None:
    if len(split_shape) < 2:
        return None
    return (split_shape[0], split_shape[1] * 2, *split_shape[2:])


def _dense_qwen3_weight_key_is_present(key: str, checkpoint_keys: set[str]) -> bool:
    if key in checkpoint_keys:
        return True
    modelopt_key = _modelopt_weight_w_alias(key)
    return modelopt_key is not None and modelopt_key in checkpoint_keys


def _dense_qwen3_dcp_checkpoint_key_is_present(
    model_dir: str,
    expected_key: str,
    checkpoint_keys: set[str],
    checkpoint_canonical_extra_state_keys: set[str],
) -> bool:
    if _dense_qwen3_weight_key_is_present(expected_key, checkpoint_keys):
        return True
    fc1_fused_key = _dense_qwen3_fc1_fused_alias(expected_key)
    if fc1_fused_key is not None and fc1_fused_key in checkpoint_keys:
        return True
    if _checkpoint_extra_state_key_is_present(
        expected_key,
        checkpoint_keys,
        checkpoint_canonical_extra_state_keys,
    ):
        return True

    if not expected_key.startswith("decoder.layers."):
        return False

    config = _load_hf_config(model_dir)
    num_hidden_layers = int(config["num_hidden_layers"])

    extra_state_alias = _dense_qwen3_layer_extra_state_alias(expected_key, num_hidden_layers)
    if extra_state_alias is not None:
        return _checkpoint_extra_state_key_is_present(
            extra_state_alias,
            checkpoint_keys,
            checkpoint_canonical_extra_state_keys,
        )

    suffix = expected_key[len("decoder.layers.") :]
    return all(
        _dense_qwen3_weight_key_is_present(
            f"decoder.layers.{layer_idx}.{suffix}",
            checkpoint_keys,
        )
        or (
            (fc1_fused_key := _dense_qwen3_fc1_fused_alias(f"decoder.layers.{layer_idx}.{suffix}"))
            is not None
            and fc1_fused_key in checkpoint_keys
        )
        for layer_idx in range(num_hidden_layers)
    )


def _missing_dense_qwen3_dcp_checkpoint_keys(
    model_dir: str,
    expected_checkpoint_keys: set[str],
    checkpoint_keys: set[str],
) -> list[str]:
    checkpoint_canonical_extra_state_keys = {
        _canonical_extra_state_key(key) for key in checkpoint_keys if "._extra_state" in key
    }
    return [
        expected_key
        for expected_key in sorted(expected_checkpoint_keys)
        if not _dense_qwen3_dcp_checkpoint_key_is_present(
            model_dir,
            expected_key,
            checkpoint_keys,
            checkpoint_canonical_extra_state_keys,
        )
    ]


def _find_mismatched_shapes(
    *,
    expected_checkpoint_keys: set[str],
    expected_shapes: dict[str, tuple[int, ...]],
    checkpoint_keys: set[str],
    checkpoint_shapes: dict[str, tuple[int, ...]],
) -> list[dict[str, object]]:
    return [
        {
            "key": key,
            "expected_shape": list(expected_shapes[key]),
            "checkpoint_shape": list(checkpoint_shapes[key]),
        }
        for key in sorted(expected_checkpoint_keys & checkpoint_keys)
        if key in expected_shapes and key in checkpoint_shapes and expected_shapes[key] != checkpoint_shapes[key]
    ]


def _find_dense_qwen3_dcp_mismatched_shapes(
    model_dir: str,
    expected_shapes: dict[str, tuple[int, ...]],
    checkpoint_keys: set[str],
    checkpoint_shapes: dict[str, tuple[int, ...]],
) -> list[dict[str, object]]:
    mismatched_shapes = _find_mismatched_shapes(
        expected_checkpoint_keys=set(expected_shapes),
        expected_shapes=expected_shapes,
        checkpoint_keys=checkpoint_keys,
        checkpoint_shapes=checkpoint_shapes,
    )

    config = _load_hf_config(model_dir)
    num_hidden_layers = int(config["num_hidden_layers"])
    for expected_key in sorted(expected_shapes):
        if not expected_key.startswith("decoder.layers.") or _dense_qwen3_weight_key_is_present(
            expected_key,
            checkpoint_keys,
        ):
            continue
        expected_shape = expected_shapes[expected_key]
        if not expected_shape:
            continue

        fc1_fused_key = _dense_qwen3_fc1_fused_alias(expected_key)
        if fc1_fused_key is not None and fc1_fused_key in checkpoint_keys:
            if expected_key.endswith(".weight_v"):
                continue
            expected_fused_shape = _dense_qwen3_fc1_fused_shape(expected_shape)
            if (
                expected_fused_shape is not None
                and fc1_fused_key in checkpoint_shapes
                and expected_fused_shape != checkpoint_shapes[fc1_fused_key]
            ):
                mismatched_shapes.append(
                    {
                        "key": fc1_fused_key,
                        "expected_shape": list(expected_fused_shape),
                        "checkpoint_shape": list(checkpoint_shapes[fc1_fused_key]),
                    }
                )
            continue

        suffix = expected_key[len("decoder.layers.") :]
        per_layer_expected_shape = expected_shape[1:]
        if expected_key.endswith(".weight_v") and all(
            (fc1_fused_key := _dense_qwen3_fc1_fused_alias(f"decoder.layers.{layer_idx}.{suffix}"))
            is not None
            and fc1_fused_key in checkpoint_keys
            for layer_idx in range(num_hidden_layers)
        ):
            continue
        for layer_idx in range(num_hidden_layers):
            checkpoint_key = f"decoder.layers.{layer_idx}.{suffix}"
            checkpoint_key = checkpoint_key if checkpoint_key in checkpoint_keys else f"{checkpoint_key}_w"
            fc1_fused_key = _dense_qwen3_fc1_fused_alias(f"decoder.layers.{layer_idx}.{suffix}")
            if fc1_fused_key is not None and fc1_fused_key in checkpoint_keys:
                expected_fused_shape = _dense_qwen3_fc1_fused_shape(per_layer_expected_shape)
                if (
                    expected_fused_shape is not None
                    and fc1_fused_key in checkpoint_shapes
                    and expected_fused_shape != checkpoint_shapes[fc1_fused_key]
                ):
                    mismatched_shapes.append(
                        {
                            "key": fc1_fused_key,
                            "expected_shape": list(expected_fused_shape),
                            "checkpoint_shape": list(checkpoint_shapes[fc1_fused_key]),
                        }
                    )
                continue
            if checkpoint_key not in checkpoint_keys or checkpoint_key not in checkpoint_shapes:
                continue
            checkpoint_shape = checkpoint_shapes[checkpoint_key]
            if per_layer_expected_shape != checkpoint_shape:
                mismatched_shapes.append(
                    {
                        "key": checkpoint_key,
                        "expected_shape": list(per_layer_expected_shape),
                        "checkpoint_shape": list(checkpoint_shape),
                    }
                )

    return sorted(mismatched_shapes, key=lambda entry: str(entry["key"]))


def _hf_weight_is_known_unquantized(model_dir: str, weight_key: str) -> bool:
    if not weight_key.endswith(".weight"):
        return False

    module_key = weight_key[: -len(".weight")]
    config = _load_optional_hf_config(model_dir)
    quantization_config = config.get("quantization_config", {})
    modules_to_not_convert = set()
    if isinstance(quantization_config, dict):
        modules_to_not_convert = set(quantization_config.get("modules_to_not_convert", []) or [])

    if module_key in modules_to_not_convert:
        return True
    if module_key in {"lm_head", "model.embed_tokens", "model.norm"}:
        return True
    return module_key.endswith(
        (
            ".input_layernorm",
            ".post_attention_layernorm",
            ".self_attn.q_norm",
            ".self_attn.k_norm",
            ".mlp.gate",
        )
    )


def _find_missing_hf_scale_keys(
    model_dir: str,
    hf_weight_map: dict[str, str],
    expected_scale_suffixes: tuple[str, ...],
) -> list[str]:
    missing_scale_keys: list[str] = []
    for key in sorted(hf_weight_map):
        if not key.endswith(".weight"):
            continue
        if _hf_weight_is_known_unquantized(model_dir, key):
            continue
        base_key = key[: -len(".weight")]
        if any(f"{base_key}.{suffix}" in hf_weight_map for suffix in expected_scale_suffixes):
            continue
        missing_scale_keys.append(f"{base_key}.{expected_scale_suffixes[0]}")
    return missing_scale_keys


def _read_dcp_tensor_shapes(checkpoint_dir: str) -> dict[str, tuple[int, ...]]:
    from torch.distributed.checkpoint import FileSystemReader

    resolved_dir = resolve_megatron_dist_checkpoint_dir(checkpoint_dir)
    reader = FileSystemReader(resolved_dir)
    metadata = reader.read_metadata()
    tensor_shapes: dict[str, tuple[int, ...]] = {}
    for key, entry in metadata.state_dict_metadata.items():
        shape = _tensor_shape(entry)
        if shape is not None:
            tensor_shapes[str(key)] = shape
    return tensor_shapes


def compare_expected_fp8_metadata(
    *,
    hf_model_path: str,
    megatron_dist_path: str,
    expected_scale_suffixes: tuple[str, ...] = ("weight_scale_inv", "weight_scale"),
) -> CheckpointParityReport:
    resolved_megatron_dist_path = resolve_megatron_dist_checkpoint_dir(megatron_dist_path)
    checkpoint_keys = read_dcp_tensor_keys(resolved_megatron_dist_path)
    hf_weight_map = load_hf_weight_index(hf_model_path)
    checkpoint_shapes = _read_dcp_tensor_shapes(resolved_megatron_dist_path)
    if _is_qwen3_moe_hf_checkpoint(hf_model_path):
        expected_checkpoint_keys = _build_qwen3_moe_fp8_expected_checkpoint_keys(
            hf_model_path, hf_weight_map
        )
        expected_shapes: dict[str, tuple[int, ...]] = {}
        # Per-expert keys end in `weight{E}_scale_inv`; dense keys end in
        # `weight_scale_inv`. Match both by anchoring on the `_scale_inv` tail.
        missing_scale_keys = [
            key
            for key in sorted(expected_checkpoint_keys)
            if key.endswith("_scale_inv") and key not in checkpoint_keys
        ]
        missing_checkpoint_keys = _missing_expected_checkpoint_keys(
            expected_checkpoint_keys,
            checkpoint_keys,
        )
        mismatched_shapes: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    elif _is_dense_qwen3_hf_checkpoint(hf_model_path):
        expected_checkpoint_keys = _build_dense_qwen3_expected_checkpoint_keys(hf_model_path)
        expected_shapes = _build_dense_qwen3_expected_shapes(hf_model_path)
        expected_scale_roots = _build_dense_qwen3_expected_scale_roots(hf_model_path)
        missing_hf_scale_keys = [
            f"{base_key}.{expected_scale_suffixes[0]}"
            for base_key in sorted(expected_scale_roots)
            if f"{base_key}.weight" in hf_weight_map
            and not any(f"{base_key}.{suffix}" in hf_weight_map for suffix in expected_scale_suffixes)
        ]
        missing_scale_keys = sorted(
            set(missing_hf_scale_keys)
            | set(
                _find_missing_dense_qwen3_dcp_scale_keys(
                    hf_model_path,
                    checkpoint_keys,
                    expected_scale_suffixes,
                )
            )
        )
        missing_checkpoint_keys = _missing_dense_qwen3_dcp_checkpoint_keys(
            hf_model_path,
            expected_checkpoint_keys,
            checkpoint_keys,
        )
        mismatched_shapes = _find_dense_qwen3_dcp_mismatched_shapes(
            hf_model_path,
            expected_shapes,
            checkpoint_keys,
            checkpoint_shapes,
        )
    else:
        expected_state_dict = _build_expected_megatron_state_dict(hf_model_path)
        expected_checkpoint_keys = {str(key) for key in expected_state_dict.keys()}
        expected_shapes = {
            str(key): shape
            for key, value in expected_state_dict.items()
            if (shape := _tensor_shape(value)) is not None
        }
        missing_scale_keys = _find_missing_hf_scale_keys(
            hf_model_path,
            hf_weight_map,
            expected_scale_suffixes,
        )
        missing_checkpoint_keys = _missing_expected_checkpoint_keys(expected_checkpoint_keys, checkpoint_keys)
        mismatched_shapes = _find_mismatched_shapes(
            expected_checkpoint_keys=expected_checkpoint_keys,
            expected_shapes=expected_shapes,
            checkpoint_keys=checkpoint_keys,
            checkpoint_shapes=checkpoint_shapes,
        )
    return CheckpointParityReport(
        hf_model_path=hf_model_path,
        megatron_dist_path=resolved_megatron_dist_path,
        missing_checkpoint_keys=missing_checkpoint_keys,
        missing_scale_keys=missing_scale_keys,
        missing_extra_state_keys=sorted(
            key for key in missing_checkpoint_keys if "._extra_state" in key
        ),
        mismatched_shapes=mismatched_shapes,
    )


def compare_expected_nvfp4_metadata(
    *,
    hf_model_path: str,
    megatron_dist_path: str,
) -> CheckpointParityReport:
    resolved_megatron_dist_path = resolve_megatron_dist_checkpoint_dir(megatron_dist_path)
    checkpoint_keys = read_dcp_tensor_keys(resolved_megatron_dist_path)
    hf_weight_map = load_hf_weight_index(hf_model_path)
    checkpoint_shapes = _read_dcp_tensor_shapes(resolved_megatron_dist_path)
    if _is_kimi_k25_hf_checkpoint(hf_model_path):
        expected_checkpoint_keys = _build_kimi_nvfp4_expected_checkpoint_keys(
            hf_model_path,
            hf_weight_map,
        )
        expected_quantizer_keys = _build_kimi_nvfp4_expected_quantizer_keys(
            hf_model_path,
            hf_weight_map,
        )
        expected_shapes = _build_kimi_nvfp4_expected_shapes(
            hf_model_path,
            hf_weight_map,
        )
        missing_checkpoint_keys = _missing_expected_checkpoint_keys(
            expected_checkpoint_keys,
            checkpoint_keys,
        )
        missing_scale_keys = sorted(
            set(_find_missing_kimi_nvfp4_hf_bundle_keys(hf_model_path, hf_weight_map))
            | set(expected_quantizer_keys - checkpoint_keys)
        )
        mismatched_shapes = _find_mismatched_shapes(
            expected_checkpoint_keys=set(expected_shapes),
            expected_shapes=expected_shapes,
            checkpoint_keys=checkpoint_keys,
            checkpoint_shapes=checkpoint_shapes,
        )
        return CheckpointParityReport(
            hf_model_path=hf_model_path,
            megatron_dist_path=resolved_megatron_dist_path,
            missing_checkpoint_keys=missing_checkpoint_keys,
            missing_scale_keys=missing_scale_keys,
            missing_extra_state_keys=sorted(
                key for key in missing_checkpoint_keys if "._extra_state" in key
            ),
            mismatched_shapes=mismatched_shapes,
        )
    if _is_qwen3_moe_hf_checkpoint(hf_model_path):
        expected_checkpoint_keys = _build_qwen3_moe_nvfp4_expected_checkpoint_keys(
            hf_model_path, hf_weight_map
        )
        missing_checkpoint_keys = _missing_expected_checkpoint_keys(
            expected_checkpoint_keys,
            checkpoint_keys,
        )
        missing_scale_keys = sorted(
            key
            for key in missing_checkpoint_keys
            if ".weight_quantizer." in key or ".input_quantizer." in key
        )
        return CheckpointParityReport(
            hf_model_path=hf_model_path,
            megatron_dist_path=resolved_megatron_dist_path,
            missing_checkpoint_keys=missing_checkpoint_keys,
            missing_scale_keys=missing_scale_keys,
            missing_extra_state_keys=sorted(
                key for key in missing_checkpoint_keys if "._extra_state" in key
            ),
            mismatched_shapes=[],
        )
    expected_checkpoint_keys = _build_dense_decoder_nvfp4_expected_checkpoint_keys(
        hf_model_path,
        hf_weight_map,
    )
    expected_quantizer_keys = _build_dense_decoder_nvfp4_expected_quantizer_keys(
        hf_model_path,
        hf_weight_map,
    )
    expected_shapes = _build_dense_decoder_nvfp4_expected_shapes(
        hf_model_path,
        hf_weight_map,
    )
    missing_checkpoint_keys = _missing_expected_checkpoint_keys(
        expected_checkpoint_keys,
        checkpoint_keys,
    )
    missing_scale_keys = sorted(
        set(_find_missing_dense_decoder_nvfp4_hf_bundle_keys(hf_model_path, hf_weight_map))
        | set(expected_quantizer_keys - checkpoint_keys)
    )
    mismatched_shapes = _find_mismatched_shapes(
        expected_checkpoint_keys=set(expected_shapes),
        expected_shapes=expected_shapes,
        checkpoint_keys=checkpoint_keys,
        checkpoint_shapes=checkpoint_shapes,
    )
    return CheckpointParityReport(
        hf_model_path=hf_model_path,
        megatron_dist_path=resolved_megatron_dist_path,
        missing_checkpoint_keys=missing_checkpoint_keys,
        missing_scale_keys=missing_scale_keys,
        missing_extra_state_keys=sorted(
            key for key in missing_checkpoint_keys if "._extra_state" in key
        ),
        mismatched_shapes=mismatched_shapes,
    )


def compare_expected_int4_metadata(
    *,
    hf_model_path: str,
    megatron_dist_path: str,
) -> CheckpointParityReport:
    resolved_megatron_dist_path = resolve_megatron_dist_checkpoint_dir(megatron_dist_path)
    checkpoint_keys = read_dcp_tensor_keys(resolved_megatron_dist_path)
    hf_weight_map = load_hf_weight_index(hf_model_path)
    checkpoint_shapes = _read_dcp_tensor_shapes(resolved_megatron_dist_path)
    expected_checkpoint_keys = (
        _build_dense_decoder_int4_expected_checkpoint_keys(hf_model_path, hf_weight_map)
        | _build_moe_expert_int4_expected_checkpoint_keys(hf_weight_map)
    )
    expected_checkpoint_keys = _prefix_int4_megatron_checkpoint_keys(hf_model_path, expected_checkpoint_keys)
    expected_shapes = {
        **_build_dense_decoder_int4_expected_shapes(hf_model_path, hf_weight_map),
        **_build_moe_expert_int4_expected_shapes(hf_model_path, hf_weight_map),
    }
    expected_shapes = _prefix_int4_megatron_checkpoint_shapes(hf_model_path, expected_shapes)
    missing_checkpoint_keys = _missing_expected_checkpoint_keys(
        expected_checkpoint_keys,
        checkpoint_keys,
    )
    missing_scale_keys = sorted(
        set(_find_missing_int4_hf_bundle_keys(hf_weight_map))
        | {
            key
            for key in missing_checkpoint_keys
            if key.endswith(("_scale", "_shape"))
        }
    )
    mismatched_shapes = _find_mismatched_shapes(
        expected_checkpoint_keys=set(expected_shapes),
        expected_shapes=expected_shapes,
        checkpoint_keys=checkpoint_keys,
        checkpoint_shapes=checkpoint_shapes,
    )
    return CheckpointParityReport(
        hf_model_path=hf_model_path,
        megatron_dist_path=resolved_megatron_dist_path,
        missing_checkpoint_keys=missing_checkpoint_keys,
        missing_scale_keys=missing_scale_keys,
        missing_extra_state_keys=sorted(
            key for key in missing_checkpoint_keys if "._extra_state" in key
        ),
        mismatched_shapes=mismatched_shapes,
    )


def write_report_json(path: str, payload: dict[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
