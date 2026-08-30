"""LoRA utilities for Megatron backend using Megatron-Bridge PEFT integration."""

import logging
from argparse import Namespace
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist

# ORBIT-SEAM: base's `import os` / `from pathlib import Path` / `get_parallel_state` import dropped:
# the adapter save/load bodies that used them now live in miles.orbit.megatron.peft_utils
# ORBIT-SEAM: LORA_ADAPTER_NAME / lora_rollout_enabled re-exported from upstream's new
# miles.utils.lora home (base defined LORA_ADAPTER_NAME here); is_lora_enabled is NOT re-exported -
# orbit redefines it below on top of the peft_method source of truth
from miles.utils.lora import LORA_ADAPTER_NAME, lora_rollout_enabled  # noqa: F401  (re-exported)

# ORBIT-SEAM: PEFT-method detection, orbit's variant-aware target-module conversion, and adapter
# checkpoint save/load delegate to miles.orbit.megatron.peft_utils. The orbit converters are aliased so
# they do not shadow the base/upstream `lora_type=`-flavoured ones defined further down (which
# multi_lora_utils and the sglang engine still import from this module).
from miles.orbit.megatron.peft_utils import (
    PeftCheckpointPreflight,
    get_peft_method,
    load_peft_adapter_checkpoint,
    resolve_target_modules_hf,
    save_peft_adapter_checkpoint,
)
from miles.orbit.megatron.peft_utils import (
    convert_target_modules_to_megatron as _orbit_convert_target_modules_to_megatron,
)
from miles.orbit.megatron.peft_utils import (
    parse_exclude_modules as _orbit_parse_exclude_modules,
)

logger = logging.getLogger(__name__)


# ORBIT-SEAM: base's lora_rank/lora_adapter_path check (now upstream's miles.utils.lora
# is_lora_enabled) replaced by orbit's single PEFT-method source of truth (LoRA vs OFT vs none)
def is_lora_enabled(args: Namespace) -> bool:
    return get_peft_method(args) == "lora"



# ---------------------------------------------------------------------------
# Unified HF <-> Megatron module name mappings
# ---------------------------------------------------------------------------

# Standard LoRA: merged Q/K/V and merged up/gate
_STANDARD_LORA_HF_TO_MEGATRON = {
    "q_proj": "linear_qkv",
    "k_proj": "linear_qkv",
    "v_proj": "linear_qkv",
    "o_proj": "linear_proj",
    "gate_proj": "linear_fc1",
    "up_proj": "linear_fc1",
    "down_proj": "linear_fc2",
    # GDN (Qwen3.5/Qwen3-Next): both slices live in the single fused megatron in_proj
    "in_proj_qkvz": "in_proj",
    "in_proj_ba": "in_proj",
}

_STANDARD_LORA_ALL_MODULES = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]

# CanonicalLoRA: Split Q/K/V and up/gate
_CANONICAL_LORA_HF_TO_MEGATRON = {
    "q_proj": "linear_q",
    "k_proj": "linear_k",
    "v_proj": "linear_v",
    "o_proj": "linear_proj",
    "gate_proj": "linear_fc1_gate",
    "up_proj": "linear_fc1_up",
    "down_proj": "linear_fc2",
    "in_proj_qkvz": "in_proj",
    "in_proj_ba": "in_proj",
}

_CANONICAL_LORA_ALL_MODULES = [
    "linear_q",
    "linear_k",
    "linear_v",
    "linear_proj",
    "linear_fc1_up",
    "linear_fc1_gate",
    "linear_fc2",
]

# Megatron -> HF (inverse mapping, one-to-many)
# Covers both standard LoRA (merged) and CanonicalLoRA (split) module names.
_MEGATRON_TO_HF_MODULES = {
    # Standard LoRA (merged layers)
    "linear_qkv": ["q_proj", "k_proj", "v_proj"],
    "linear_proj": ["o_proj"],
    "linear_fc1": ["gate_proj", "up_proj"],
    "linear_fc2": ["down_proj"],
    # CanonicalLoRA (split layers)
    "linear_q": ["q_proj"],
    "linear_k": ["k_proj"],
    "linear_v": ["v_proj"],
    "linear_fc1_gate": ["gate_proj"],
    "linear_fc1_up": ["up_proj"],
    # GDN linear attention: SGLang serves the fused in_proj as two modules
    "in_proj": ["in_proj_qkvz", "in_proj_ba"],
}

_HF_MODULE_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "in_proj_qkvz",
    "in_proj_ba",
}

# DeepSeek / Kimi MLA (HF names on checkpoint; Megatron uses linear_* from Megatron-Bridge mappings).
_MLA_HF_TO_MEGATRON = {
    "q_a_proj": "linear_q_down_proj",
    "kv_a_proj_with_mqa": "linear_kv_down_proj",
    "q_b_proj": "linear_q_up_proj",
    "kv_b_proj": "linear_kv_up_proj",
    # DSA indexer (GLM-5 / DeepSeek-V3.2): HF/SGLang leaf names vs Megatron-Bridge linear_* names.
    "wq_b": "linear_wq_b",
    "wk": "linear_wk",
    "weights_proj": "linear_weights_proj",
}
_MEGATRON_MLA_TO_HF = {v: k for k, v in _MLA_HF_TO_MEGATRON.items()}

# Empty: dropping a module here makes sglang silently skip its shipped adapter tensors.
_SGLANG_UNSUPPORTED_HF_TARGETS = frozenset()


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------


def lora_base_cpu_backup_enabled(args: Namespace) -> bool:
    """LoRA + --colocate + --lora-base-cpu-backup all set."""
    return is_lora_enabled(args) and getattr(args, "colocate", False) and getattr(args, "lora_base_cpu_backup", False)


def sglang_lora_target_all_sentinel(args) -> bool:
    """Hand SGLang the ``"all"`` shorthand so it auto-detects module names (required for Inkling)."""
    from miles.utils.chat_template_utils.inkling import is_inkling_checkpoint

    return is_inkling_checkpoint(getattr(args, "hf_checkpoint", None) or "")


_marked_lora_grad_params_cache: dict[int, list] = {}


def reduce_marked_lora_grads(model: Sequence[torch.nn.Module]) -> None:
    """Sum partial grads of replicated LoRA params over their tagged group ("tp"|"ep"), before the DP reduce-scatter."""
    from megatron.core import parallel_state as ps

    key = id(model[0]) if model else 0
    marked = _marked_lora_grad_params_cache.get(key)
    if marked is None:
        marked = []
        for chunk in model:
            for param in chunk.parameters():
                group_name = getattr(param, "_lora_grad_sum_group", None)
                if group_name is not None and param.requires_grad:
                    marked.append((param, group_name))
        _marked_lora_grad_params_cache[key] = marked
    if not marked:
        return
    groups = {
        "tp": (ps.get_tensor_model_parallel_group(), ps.get_tensor_model_parallel_world_size()),
        "ep": (ps.get_expert_model_parallel_group(), ps.get_expert_model_parallel_world_size()),
    }
    for group_name in ("tp", "ep"):
        group, size = groups[group_name]
        if size <= 1:
            continue
        grads = []
        for param, g_name in marked:
            if g_name != group_name:
                continue
            grad = getattr(param, "main_grad", None)
            if grad is None:
                grad = param.grad
            if grad is not None:
                grads.append(grad)
        for dt in {g.dtype for g in grads}:
            gs = [g for g in grads if g.dtype == dt]
            if len(gs) == 1:
                dist.all_reduce(gs[0], op=dist.ReduceOp.SUM, group=group)
                continue
            flat = torch._utils._flatten_dense_tensors(gs)
            dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
            for g, red in zip(gs, torch._utils._unflatten_dense_tensors(flat, gs), strict=False):
                g.copy_(red)


def is_lora_model(model: Sequence[torch.nn.Module]) -> bool:
    """Check if model has LoRA layers applied."""
    for model_chunk in model:
        if hasattr(model_chunk.module, "peft_config"):
            return True
        for name, _ in model_chunk.named_parameters():
            if "lora_" in name or "adapter" in name:
                return True
    return False


def is_lora_weight_name(name: str) -> bool:
    # ORBIT-SEAM: docstring dropped (logic unchanged): part of the file's delegation-focused rewrite
    return ".lora_A." in name or ".lora_B." in name


# ORBIT-SEAM: orbit's MLA-aware variant selector, replacing the lora_type-only class-name switch for
# the miles.orbit.megatron.peft_utils converters used by create_lora_instance below. The base/upstream
# `lora_type=` helpers underneath are retained unchanged - multi_lora_utils and the sglang engine
# still import them from this module.
def _lora_variant(args: Namespace) -> str:
    if getattr(args, "multi_latent_attention", False):
        return "mla"
    lora_type_name = getattr(args, "lora_type", "lora").lower()
    return "canonical" if lora_type_name == "canonical_lora" else "standard"


def _is_adapter_param_name(name: str) -> bool:
    """Check if a parameter name belongs to a LoRA adapter (Megatron internal naming)."""
    return "lora_" in name or (".adapter." in name and ("linear_in" in name or "linear_out" in name))


_param_grad_buffer_patched = False


def patch_param_grad_buffer_for_colocate_mode_lora() -> None:
    """Patch _ParamAndGradBuffer to use disable_param_buffers_cpu_backup=True.

    In colocate mode with offload_train, torch_memory_saver.pause(tag="default")
    offloads default-region GPU memory.  During LoRA training, base weights are
    frozen (requires_grad=False) so DDP only creates buffers for adapter params.

    This patch ensures those buffers are allocated in the "param_buffer" region
    (enable_cpu_backup=False), making them invisible to pause(tag="default") —
    eliminating the need for resume()/pause() around update_weights.

    The patch is idempotent and only takes effect once.
    """
    global _param_grad_buffer_patched
    if _param_grad_buffer_patched:
        return
    _param_grad_buffer_patched = True

    from megatron.core.distributed.param_and_grad_buffer import _ParamAndGradBuffer

    _original_init = _ParamAndGradBuffer.__init__

    def _patched_init(self, *args, **kwargs):
        # Megatron reads these flags from ddp_config (its first ctor argument).
        ddp_config = kwargs.get("ddp_config", args[0] if args else None)
        ddp_config.disable_param_buffers_cpu_backup = True
        ddp_config.disable_grad_buffers_cpu_backup = True
        _original_init(self, *args, **kwargs)

    _ParamAndGradBuffer.__init__ = _patched_init
    logger.info("Patched _ParamAndGradBuffer.__init__ for LoRA colocate mode (disable cpu backup)")


# ---------------------------------------------------------------------------
# Module name conversion
# ---------------------------------------------------------------------------


def _get_lora_class_name(lora_type: type | object | None) -> str:
    """Resolve LoRA type to its class name string."""
    if lora_type is None:
        return "CanonicalLoRA"
    if isinstance(lora_type, type):
        return lora_type.__name__
    return type(lora_type).__name__


def convert_target_modules_to_megatron(
    hf_modules: str | list[str],
    lora_type: type | object | None = None,
) -> list[str]:
    """Convert HuggingFace LoRA target module names to Megatron format.

    HF:  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    Megatron (LoRA):          linear_qkv, linear_proj, linear_fc1, linear_fc2
    Megatron (CanonicalLoRA): linear_q, linear_k, linear_v, linear_proj,
                              linear_fc1_up, linear_fc1_gate, linear_fc2

    Special values: "all", "all-linear", "all_linear" -> all standard linear modules.
    If input is already in Megatron format, returns as-is.
    """
    class_name = _get_lora_class_name(lora_type)
    is_canonical = class_name == "CanonicalLoRA"

    all_modules = _CANONICAL_LORA_ALL_MODULES if is_canonical else _STANDARD_LORA_ALL_MODULES
    hf_to_megatron = _CANONICAL_LORA_HF_TO_MEGATRON if is_canonical else _STANDARD_LORA_HF_TO_MEGATRON

    # Handle special "all-linear" variants
    if isinstance(hf_modules, str):
        if hf_modules in ("all", "all-linear", "all_linear"):
            return list(all_modules)
        hf_modules = [hf_modules]
    elif isinstance(hf_modules, list) and len(hf_modules) == 1:
        if hf_modules[0] in ("all", "all-linear", "all_linear"):
            return list(all_modules)

    if isinstance(hf_modules, tuple):
        hf_modules = list(hf_modules)

    # Check if already in Megatron format (standard / canonical / Kimi MLA linear_*).
    if all(m not in _HF_MODULE_NAMES and m not in _MLA_HF_TO_MEGATRON for m in hf_modules if "*" not in m):
        return list(hf_modules)

    # Convert HF names to Megatron names (dedup while preserving order)
    megatron_modules: list[str] = []
    for module in hf_modules:
        if module in _MLA_HF_TO_MEGATRON:
            megatron_name = _MLA_HF_TO_MEGATRON[module]
        else:
            megatron_name = hf_to_megatron.get(module, module)
        if megatron_name not in megatron_modules:
            megatron_modules.append(megatron_name)

    return megatron_modules


def convert_target_modules_to_hf(megatron_modules: list[str]) -> list[str]:
    """Convert Megatron LoRA target module names to HuggingFace format.

    Supports both standard LoRA and CanonicalLoRA module names.

    Megatron standard:   linear_qkv, linear_proj, linear_fc1, linear_fc2
    Megatron canonical:  linear_q, linear_k, linear_v, linear_proj,
                         linear_fc1_up, linear_fc1_gate, linear_fc2
    HF:                  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    Kimi MLA Megatron:   linear_q_down_proj -> q_a_proj, linear_kv_down_proj -> kv_a_proj_with_mqa, ...

    Wildcards (``*.layers.2.mlp.experts.linear_fc1``) get the last dotted
    segment mapped to an HF leaf name; SGLang uses the result to choose
    adapter-buffer types, not to scope by layer.
    """
    if isinstance(megatron_modules, tuple):
        megatron_modules = list(megatron_modules)
    hf_modules: list[str] = []
    for module in megatron_modules:
        lookup_key = module.rsplit(".", 1)[-1] if "." in module else module
        if lookup_key in _MEGATRON_MLA_TO_HF:
            hf_modules.append(_MEGATRON_MLA_TO_HF[lookup_key])
        elif lookup_key in _MEGATRON_TO_HF_MODULES:
            hf_modules.extend(_MEGATRON_TO_HF_MODULES[lookup_key])
        else:
            # same-name passthrough; SGLang needs the leaf, not a path or pattern
            hf_modules.append(lookup_key)
    seen: set[str] = set()
    unique: list[str] = []
    for m in hf_modules:
        if m not in seen:
            seen.add(m)
            unique.append(m)
    return unique


def target_modules_hf_for_sglang_rollout(args: Namespace) -> list[str]:
    """HF target_modules for SGLang LoRA init/sync (minus _SGLANG_UNSUPPORTED_HF_TARGETS, currently empty)."""
    raw = list(args.target_modules) if args.target_modules else []
    hf = convert_target_modules_to_hf(raw)
    out = [m for m in hf if m not in _SGLANG_UNSUPPORTED_HF_TARGETS]
    dropped = set(hf) - set(out)
    if dropped:
        logger.warning(
            "target_modules_hf_for_sglang_rollout: omitting %s for SGLang (unsupported by default "
            "get_hidden_dim); Megatron should not train LoRA on these if rollout sync is required.",
            sorted(dropped),
        )
    return out


# ---------------------------------------------------------------------------
# Model setup helpers (used by model.py)
# ---------------------------------------------------------------------------


def parse_exclude_modules(args: Namespace, lora_type=None) -> list[str]:
    """Parse and convert exclude_modules argument."""
    exclude_modules: list[str] = []
    raw = getattr(args, "exclude_modules", None)
    if raw:
        if isinstance(raw, str):
            exclude_modules = [m.strip() for m in raw.split(",")]
        else:
            exclude_modules = list(raw)
        exclude_modules = convert_target_modules_to_megatron(exclude_modules, lora_type=lora_type)
    return exclude_modules


def create_lora_instance(args: Namespace):
    # ORBIT-SEAM: docstring dropped (logic below rewritten to delegate); still creates the
    # LoRA/CanonicalLoRA instance, now via _lora_variant
    from megatron.bridge.peft.canonical_lora import CanonicalLoRA
    from megatron.bridge.peft.lora import LoRA

    # ORBIT-SEAM: base's lora_type_name lookup + if/else lora_cls selection replaced by _lora_variant,
    # which additionally special-cases multi_latent_attention (MLA) to force the "mla" variant
    variant = _lora_variant(args)
    # CanonicalLoRA splits merged Q/K/V; MLA already exposes split projections,
    # so plain LoRA is the right wrapper there.
    lora_cls = CanonicalLoRA if variant == "canonical" else LoRA

    # ORBIT-SEAM: converters come from miles.orbit.megatron.peft_utils and take variant= (from
    # _lora_variant) instead of lora_type=lora_cls; aliased at import so the base/upstream
    # `lora_type=` helpers above stay available to their own importers
    target_modules = _orbit_convert_target_modules_to_megatron(args.target_modules, variant=variant)
    exclude_modules = _orbit_parse_exclude_modules(args, variant=variant)

    lora_kwargs = dict(
        target_modules=target_modules,
        exclude_modules=exclude_modules,
        dim=args.lora_rank,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        # ORBIT-SEAM: args attribute renamed lora_A_init_method -> lora_a_init_method (lowercase 'a'):
        # matches orbit's argument-registration casing convention
        lora_A_init_method=getattr(args, "lora_a_init_method", "xavier"),
        lora_B_init_method=getattr(args, "lora_B_init_method", "zero"),
    )
    # shared-outer grouped-expert LoRA (SGLang PR #21466); per-expert is the default
    if getattr(args, "experts_shared_outer_loras", False):
        assert lora_cls is LoRA, "--experts-shared-outer-loras requires the standard LoRA adapter type"
        lora_kwargs["experts_shared_outer_loras"] = True

    lora = lora_cls(**lora_kwargs)

    logger.info(
        f"Created {lora_cls.__name__}: rank={args.lora_rank}, alpha={args.lora_alpha}, "
        # ORBIT-SEAM: log line gains a_init=<lora_a_init_method> alongside the existing fields -
        # reflects the renamed arg above
        f"dropout={args.lora_dropout}, a_init={getattr(args, 'lora_a_init_method', 'xavier')}, "
        f"target_modules={target_modules}, exclude_modules={exclude_modules}"
    )
    return lora


# ORBIT-SEAM: relocated here from the end of the base file (base's checkpoint save/load section
# header comment stood at this spot) so save_lora_checkpoint's build_config lambda below can see it;
# target-module HF listing now delegates to miles.orbit.megatron.peft_utils.resolve_target_modules_hf
# instead of computing target_modules_hf inline via the removed convert_target_modules_to_hf
def build_lora_sync_config(args: Namespace) -> dict[str, Any]:
    """Build LoRA config dict for syncing weights to SGLang engines."""
    # ORBIT-SEAM: target-module listing comes from orbit's PEFT-aware resolve_target_modules_hf
    # instead of upstream's target_modules_hf_for_sglang_rollout; upstream's Inkling "all-linear"
    # sentinel is kept in front of it.
    if sglang_lora_target_all_sentinel(args):
        target_modules_hf: Any = "all-linear"
    else:
        target_modules_hf = resolve_target_modules_hf(args)
    return {
        "peft_type": "LORA",
        "r": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "target_modules": target_modules_hf,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }


def save_lora_checkpoint(
    model: Sequence[torch.nn.Module],
    args: Namespace,
    save_dir: str,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
    iteration: int | None = None,
    # ORBIT-SEAM: active_student_version param added - threaded through to
    # miles.orbit.megatron.peft_utils.save_peft_adapter_checkpoint for OPD self-teacher version tagging
    active_student_version: str | None = None,
) -> str:
    # ORBIT-SEAM: base's ~90-line LoRA save (Megatron-native per-rank .pt write + HF PEFT bridge
    # export + adapter_config.json + optimizer/scheduler training-state write) excised and replaced
    # by a single delegation to miles.orbit.megatron.peft_utils.save_peft_adapter_checkpoint(method="lora"),
    # which implements the same two-format save plus PEFT method dispatch and the self-teacher sidecar
    return save_peft_adapter_checkpoint(
        model,
        args,
        save_dir,
        method="lora",
        build_config=lambda: build_lora_sync_config(args),
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
        iteration=iteration,
        active_student_version=active_student_version,
    )


def load_lora_adapter(
    model: Sequence[torch.nn.Module],
    adapter_path: str,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
    # ORBIT-SEAM: 3 params added (expected_iteration, expected_active_student_version,
    # checkpoint_preflight) - support orbit's checkpoint-preflight validation and self-teacher-version
    # consistency check, threaded through to miles.orbit.megatron.peft_utils.load_peft_adapter_checkpoint
    expected_iteration: int | None = None,
    expected_active_student_version: str | None = None,
    checkpoint_preflight: PeftCheckpointPreflight | None = None,
) -> tuple[bool, int | None]:
    # ORBIT-SEAM: base's ~90-line LoRA load (native .pt tensor restore + HF-PEFT-not-supported
    # warning + optimizer/scheduler training-state restore) plus the standalone _load_training_state()
    # helper (removed entirely) excised and replaced by a delegation to
    # miles.orbit.megatron.peft_utils.load_peft_adapter_checkpoint(label="LoRA"), which folds in
    # checkpoint-preflight validation and self-teacher-version checks
    return load_peft_adapter_checkpoint(
        model,
        adapter_path,
        label="LoRA",
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
        expected_iteration=expected_iteration,
        expected_active_student_version=expected_active_student_version,
        checkpoint_preflight=checkpoint_preflight,
    )


# ORBIT-SEAM: base's build_lora_sync_config (inline target_modules_hf computation via
# convert_target_modules_to_hf, then this same dict literal) stood at the end of the file here -
# it was relocated above create_lora_instance and now delegates to
# miles.orbit.megatron.peft_utils.resolve_target_modules_hf; see the ORBIT-SEAM stamp there
