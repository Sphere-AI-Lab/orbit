"""Bridge / PEFT model setup helpers.

Extracted from ``model.py`` to keep the main training module focused on
forward / backward / optimizer logic.
"""

from __future__ import annotations

from argparse import Namespace
from collections import Counter
import logging
from types import SimpleNamespace
import torch

from orbit.megatron.bridge_provider_overrides import apply_bridge_provider_overrides
from orbit.megatron.low_precision_bootstrap import (
    configure_provider_for_low_precision,
    is_distributed_checkpoint,
    load_dist_checkpoint,
    resolve_bridge_load_path,
)
from orbit.megatron.peft_utils import detect_peft_variant

logger = logging.getLogger(__name__)


def _ensure_model_list(model):
    return model if isinstance(model, list) else [model]


_PRELOADED_CHECKPOINT_IDENTITY_ATTRS = (
    "_orbit_loaded_dist_checkpoint_path",
    "_orbit_loaded_dist_checkpoint_prefix",
    "_orbit_restored_modelopt_checkpoint_path",
)


def _propagate_preloaded_checkpoint_identity(source_chunks, transformed_chunks) -> None:
    """Keep a pre-wrap base load attached to the PEFT-wrapped model.

    Some Bridge PEFT transforms return replacement top-level modules. Without
    copying Orbit's load identity, the ordinary initialization path loads the
    same model-only checkpoint a second time into the wrapped model and strict
    distributed-checkpoint loading then rejects the intentionally fresh adapter
    parameters.
    """
    source_chunks = _ensure_model_list(source_chunks)
    transformed_chunks = _ensure_model_list(transformed_chunks)
    if len(source_chunks) != len(transformed_chunks):
        raise RuntimeError(
            "PEFT wrapping changed the number of model chunks after base checkpoint preload: "
            f"{len(source_chunks)} -> {len(transformed_chunks)}"
        )
    for source, transformed in zip(source_chunks, transformed_chunks, strict=True):
        for attr_name in _PRELOADED_CHECKPOINT_IDENTITY_ATTRS:
            if hasattr(source, attr_name):
                setattr(transformed, attr_name, getattr(source, attr_name))


def _materialize_runtime_device(model_chunks):
    from megatron.bridge.models.common.unimodal import to_empty_if_meta_device

    target_device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    for model_chunk in _ensure_model_list(model_chunks):
        to_empty_if_meta_device(model_chunk, device=target_device)


def _adapter_debug_name(module_name: str, module) -> str | None:
    adapter = getattr(module, "adapter", None)
    if adapter is None:
        return None
    return getattr(adapter, "_oft_debug_name", None) or module_name


_CANONICAL_OFT_ADAPTER_LEAVES = {
    "adapter_q": "linear_q",
    "adapter_k": "linear_k",
    "adapter_v": "linear_v",
    "adapter_gate": "linear_fc1_gate",
    "adapter_up": "linear_fc1_up",
}


def _replace_module_leaf(module_name: str, leaf_name: str) -> str:
    parts = [part for part in module_name.split(".") if part]
    if not parts:
        return leaf_name
    parts[-1] = leaf_name
    return ".".join(parts)


def _grouped_expert_fc1_aliases(debug_name: str) -> list[str]:
    """For the grouped-expert FC1 canonical-OFT fallback, emit two virtual
    debug names that cover the split target names. Megatron-Bridge attaches
    a single unfused ``OFTLinear`` at ``...mlp.experts.linear_fc1`` because
    grouped expert FC1 cannot be split per-expert (the gate/up rows are
    interleaved across experts in the fused weight); the post-wrap validator
    still wants to see both ``linear_fc1_gate`` and ``linear_fc1_up`` so it
    can confirm the canonical target list was honored.
    """
    # Keep this predicate in sync with megatron.bridge.orbit.oft.canonical_oft._should_treat_linear_fc1_as_unfused.
    if not debug_name.endswith(".mlp.experts.linear_fc1"):
        return []
    return [
        _replace_module_leaf(debug_name, "linear_fc1_gate"),
        _replace_module_leaf(debug_name, "linear_fc1_up"),
    ]


def _adapter_debug_names(module_name: str, module) -> list[str]:
    # CanonicalOFT split wrappers (OFTLinearSplitQKV / OFTLinearSplitFC1UpGate)
    # expose per-slice adapters as adapter_q/k/v/gate/up — not a singular
    # `adapter` — so the legacy single-adapter probe counts zero wraps even
    # though linear_qkv / linear_fc1 are wrapped. Walk both shapes.
    names: list[str] = []
    debug = _adapter_debug_name(module_name, module)
    if debug is not None:
        names.append(debug)
        names.extend(_grouped_expert_fc1_aliases(debug))
    for attr_name, leaf_name in _CANONICAL_OFT_ADAPTER_LEAVES.items():
        split_adapter = getattr(module, attr_name, None)
        if split_adapter is None:
            continue
        names.append(
            getattr(split_adapter, "_oft_debug_name", None)
            or _replace_module_leaf(module_name, leaf_name)
        )
    return names


def _dsv4_native_moe_oft_target_names(module_name: str, module) -> list[str]:
    names = []
    for proj in ("w1", "w2", "w3"):
        if getattr(module, f"{proj}_oft_r", None) is not None:
            names.append(f"{module_name}.{proj}" if module_name else proj)
    return names


def _target_is_dsv4_native_moe_oft(target: str) -> bool:
    return ".experts." in target and _target_leaf_name(target) in {"w1", "w2", "w3"}


def _summarize_peft_wrapped_modules(
    model_chunks,
    *,
    include_dsv4_native_moe_oft: bool = False,
) -> dict[str, object]:
    names: list[str] = []
    for chunk_idx, model_chunk in enumerate(_ensure_model_list(model_chunks)):
        named_modules = getattr(model_chunk, "named_modules", None)
        if named_modules is None:
            continue
        for module_name, module in named_modules():
            for debug_name in _adapter_debug_names(module_name, module):
                names.append(f"chunk{chunk_idx}.{debug_name}")
            if include_dsv4_native_moe_oft:
                for native_name in _dsv4_native_moe_oft_target_names(module_name, module):
                    names.append(f"chunk{chunk_idx}.{native_name}")

    targets = Counter(name.rsplit(".", 1)[-1] for name in names)
    return {
        "total": len(names),
        "targets": targets,
        "examples": names[:20],
    }


def _target_leaf_name(target: str) -> str:
    return target.rsplit(".", 1)[-1]


def _assert_peft_wrapped_modules(
    model_chunks,
    *,
    peft_method: str,
    requested_target_modules: list[str] | tuple[str, ...] | None = None,
) -> None:
    if peft_method != "oft":
        return

    include_dsv4_native_moe_oft = bool(
        requested_target_modules
        and any(_target_is_dsv4_native_moe_oft(target) for target in requested_target_modules)
    )
    summary = _summarize_peft_wrapped_modules(
        model_chunks,
        include_dsv4_native_moe_oft=include_dsv4_native_moe_oft,
    )
    if summary["total"] == 0:
        raise RuntimeError("OFT did not wrap any Megatron modules; check --target-modules and Kimi MLA naming.")
    if requested_target_modules:
        missing_targets = [
            target
            for target in requested_target_modules
            if summary["targets"].get(target, 0) == 0
            and summary["targets"].get(_target_leaf_name(target), 0) == 0
        ]
        if missing_targets:
            raise RuntimeError(
                "OFT target modules matched no Megatron modules after wrapping: "
                f"{missing_targets}. Wrapped targets={dict(summary['targets'])}. "
                "Check --target-modules and Kimi MLA naming."
            )

    logger.info(
        "OFT wrapped Megatron modules: total=%s targets=%s examples=%s",
        summary["total"],
        dict(summary["targets"]),
        summary["examples"],
    )


def _make_value_model_hook(hidden_size: int, sequence_parallel: bool):
    """Create a pre-wrap hook that replaces the output layer with a value head."""
    from megatron.core import parallel_state
    from miles.backends.megatron_utils.model_provider import replace_output_layer_with_value_head

    value_config = SimpleNamespace(hidden_size=hidden_size, sequence_parallel=sequence_parallel)

    def hook(model):
        model_post_process = []
        if (
            parallel_state.get_pipeline_model_parallel_world_size() > 1
            and parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None
        ):
            for i in range(parallel_state.get_virtual_pipeline_model_parallel_world_size()):
                model_post_process.append(parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=i))
        else:
            model_post_process.append(parallel_state.is_pipeline_last_stage())

        model_list = _ensure_model_list(model)
        assert len(model_post_process) == len(model_list), "Model list length and post process list length must match."

        for index, model_chunk in enumerate(model_list):
            if not model_post_process[index]:
                continue
            replace_output_layer_with_value_head(model_chunk, value_config)

    return hook


def _make_peft_pre_wrap_hook(
    peft,
    *,
    load_path: str | None,
    is_value_model: bool,
    peft_method: str,
    post_peft_hooks=None,
    requested_target_modules: list[str] | tuple[str, ...] | None = None,
):
    post_peft_hooks = tuple(post_peft_hooks or ())

    def hook(model_chunks):
        model_list = _ensure_model_list(model_chunks)
        if load_path and is_distributed_checkpoint(load_path):
            load_dist_checkpoint(model_list, load_path, is_value_model=is_value_model)
        preloaded_chunks = list(model_list)
        transformed = _ensure_model_list(peft(model_list, training=True))
        for post_peft_hook in post_peft_hooks:
            maybe_transformed = post_peft_hook(transformed)
            if maybe_transformed is not None:
                transformed = _ensure_model_list(maybe_transformed)
        # A PEFT transform or any subsequent hook may replace a top-level
        # model chunk. Attach the preload identity to the final chunks that
        # Bridge will wrap so the ordinary initialization load sees it after
        # unwrapping DDP/mixed-precision wrappers.
        _propagate_preloaded_checkpoint_identity(preloaded_chunks, transformed)
        _assert_peft_wrapped_modules(
            transformed,
            peft_method=peft_method,
            requested_target_modules=requested_target_modules,
        )
        # Optional structural audit dump (env-gated; no-op in production).
        from orbit.audit.peft_wrap import dump_megatron_audit
        dump_megatron_audit(transformed)
        _materialize_runtime_device(transformed)
        peft.set_params_to_save(transformed)
        return transformed

    return hook


def _oft_validator_variant(args: Namespace) -> str:
    """Pick the Megatron name variant the post-wrap PEFT validator should
    compare against. Mirrors ``create_oft_instance`` so the validator's
    expected-target list aligns with whichever wrapper was actually attached:

        peft_variant == "dsv4"            -> "dsv4"   (DSV4OFT)
        --oft-type oft (legacy shared-R)  -> detect_peft_variant(args)
                                             ("standard" for fused QKV/FC1
                                              models, "mla" for MLA models)
        --oft-type canonical_oft (default)-> "canonical" (CanonicalOFT split)
    """
    variant = detect_peft_variant(args)
    if variant == "dsv4":
        return "dsv4"
    if getattr(args, "oft_type", "canonical_oft") == "oft":
        return variant
    return "canonical"


def _bridge_is_value_model(hf_config, role: str = "actor") -> bool:
    """The adapter-mode critic reuses the value-model machinery: pre-DDP value-head
    hook plus is_value_model-aware base-weight loading."""
    if role == "critic":
        return True
    return (
        "ForTokenClassification" in hf_config.architectures[0]
        or "ForSequenceClassification" in hf_config.architectures[0]
    )


def _setup_peft_model_via_bridge(args: Namespace, role: str = "actor") -> list:
    """Build Megatron model with PEFT using Megatron-Bridge."""
    from megatron.bridge import AutoBridge
    from megatron.bridge.training.config import DistributedDataParallelConfig
    from transformers import AutoConfig
    from orbit.megatron.peft_utils import (
        convert_target_modules_to_megatron,
        create_peft_instance,
    )

    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
    provider = bridge.to_megatron_provider(load_weights=False)

    apply_bridge_provider_overrides(
        provider,
        args,
        variable_seq_lengths=True,
        default_moe_token_dispatcher_type="alltoall",
        default_moe_router_load_balancing_type="none",
    )
    configure_provider_for_low_precision(provider, resolve_bridge_load_path(args, hf_config=hf_config))
    provider.finalize()

    peft = create_peft_instance(args)
    requested_target_modules = None
    if getattr(args, "peft_method", "none") == "oft":
        requested_target_modules = convert_target_modules_to_megatron(
            args.target_modules,
            variant=_oft_validator_variant(args),
        )

    is_value_model = _bridge_is_value_model(hf_config, role=role)
    post_peft_hooks = []
    if is_value_model:
        hidden_size = hf_config.text_config.hidden_size if hasattr(hf_config, "text_config") else hf_config.hidden_size
        post_peft_hooks.append(_make_value_model_hook(hidden_size, provider.sequence_parallel))

    provider.register_pre_wrap_hook(
        _make_peft_pre_wrap_hook(
            peft,
            load_path=resolve_bridge_load_path(args, hf_config=hf_config),
            is_value_model=is_value_model,
            peft_method=getattr(args, "peft_method", "none"),
            post_peft_hooks=post_peft_hooks,
            requested_target_modules=requested_target_modules,
        )
    )

    use_distributed_optimizer = "muon" not in (getattr(args, "optimizer", None) or "").lower()
    ddp_config = DistributedDataParallelConfig(use_distributed_optimizer=use_distributed_optimizer)
    ddp_config.finalize()
    extra_kwargs = {}
    if getattr(provider, "experimental_attention_variant", None) == "dsv4":
        # DeepSeek V4 native quant weights include dtypes that cannot be
        # blanket-cast by Megatron's Float16Module wrapper. Keep base weights
        # in their native quant format; the OFT adapter tensors carry the
        # training dtype selected in create_oft_instance().
        extra_kwargs["mixed_precision_wrapper"] = None
    return provider.provide_distributed_model(wrap_with_ddp=True, ddp_config=ddp_config, **extra_kwargs)
