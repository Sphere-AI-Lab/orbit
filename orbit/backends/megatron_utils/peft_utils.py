import json
import logging
import os
from argparse import Namespace
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
import torch.distributed as dist
from megatron.core import mpu
from safetensors.torch import save_file as safetensors_save_file

from orbit.backends.training_utils.parallel import get_parallel_state
from orbit.backends.megatron_utils.update_weight.common import is_dsv4_grouped_moe_oft_param_name

logger = logging.getLogger(__name__)


LORA_SYNC_TRANSPORT = "lora_adapter"
OFT_SYNC_TRANSPORT = "oft_adapter"

Variant = Literal["standard", "canonical", "mla", "dsv4"]


@dataclass(frozen=True)
class PeftSyncSpec:
    method: str
    adapter_name: str
    adapter_config: dict
    sync_transport: str


def get_peft_method(args) -> str:
    return getattr(args, "peft_method", "none")


def is_peft_enabled(args) -> bool:
    return get_peft_method(args) != "none"


from megatron.bridge.peft.param_names import (
    CANONICAL_OFT_SLICE_NAMES,
    is_peft_adapter_param_name,
)


def is_adapter_param_name(name: str) -> bool:
    return is_peft_adapter_param_name(name) or is_dsv4_grouped_moe_oft_param_name(name)


def _maybe_legacy_canonical_oft_key(name: str) -> str | None:
    """If ``name`` is a CanonicalOFT split slice (``...adapter_q.oft_r``), return
    the legacy shared-R key (``...adapter.oft_r``) it would have lived under in
    a pre-fix checkpoint. Returns ``None`` for non-split keys."""
    for slice_name in CANONICAL_OFT_SLICE_NAMES:
        token = f".adapter_{slice_name}."
        if token in name:
            return name.replace(token, ".adapter.", 1)
    return None


def is_peft_model(model: Sequence[torch.nn.Module]) -> bool:
    for model_chunk in model:
        for name, _ in model_chunk.named_parameters():
            if is_adapter_param_name(name):
                return True
    return False


def validate_peft_checkpoint_type(adapter_dir: Path, expected_method: str) -> dict:
    config_path = Path(adapter_dir) / "adapter_config.json"
    if not config_path.exists():
        return {}

    with config_path.open() as f:
        config = json.load(f)

    actual_type = config.get("peft_type")
    expected_type = expected_method.upper()
    if actual_type is not None and actual_type.upper() != expected_type:
        raise ValueError(
            f"PEFT checkpoint at {adapter_dir} has peft_type={actual_type}, expected {expected_type}."
        )
    return config


def create_peft_instance(args):
    method = get_peft_method(args)
    if method == "oft":
        from .oft_utils import create_oft_instance

        return create_oft_instance(args)
    if method == "lora":
        from .lora_utils import create_lora_instance

        return create_lora_instance(args)
    return None


def build_peft_sync_spec(args) -> PeftSyncSpec | None:
    method = get_peft_method(args)
    if method == "oft":
        from .oft_utils import OFT_ADAPTER_NAME, build_oft_sync_config

        return PeftSyncSpec(
            method="oft",
            adapter_name=OFT_ADAPTER_NAME,
            adapter_config=build_oft_sync_config(args),
            sync_transport=OFT_SYNC_TRANSPORT,
        )
    if method == "lora":
        from .lora_utils import LORA_ADAPTER_NAME, build_lora_sync_config

        return PeftSyncSpec(
            method="lora",
            adapter_name=LORA_ADAPTER_NAME,
            adapter_config=build_lora_sync_config(args),
            sync_transport=LORA_SYNC_TRANSPORT,
        )
    return None


def save_peft_checkpoint(
    model,
    args,
    save_dir,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
    iteration: int | None = None,
) -> str:
    method = get_peft_method(args)
    if method == "lora":
        from .lora_utils import save_lora_checkpoint

        return save_lora_checkpoint(
            model,
            args,
            save_dir,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            iteration=iteration,
        )
    if method == "oft":
        from .oft_utils import save_oft_checkpoint

        return save_oft_checkpoint(
            model,
            args,
            save_dir,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            iteration=iteration,
        )
    raise ValueError(f"Cannot save PEFT checkpoint when peft_method={method!r}.")


def load_peft_adapter(
    model,
    args,
    adapter_path: str,
    *,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
) -> tuple[bool, int | None]:
    method = get_peft_method(args)
    adapter_dir = Path(adapter_path)

    if method == "lora":
        from .lora_utils import load_lora_adapter

        validate_peft_checkpoint_type(adapter_dir, expected_method=method)
        return load_lora_adapter(
            model,
            adapter_path,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
        )
    if method == "oft":
        from .oft_utils import load_oft_adapter

        validate_peft_checkpoint_type(adapter_dir, expected_method=method)
        return load_oft_adapter(
            model,
            adapter_path,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
        )
    raise ValueError(f"Cannot load PEFT adapter when peft_method={method!r}.")


# ---------------------------------------------------------------------------
# Shared training-state save/load (used by save/load_{lora,oft}_checkpoint)
# ---------------------------------------------------------------------------


def save_training_state(
    adapter_dir: Path,
    optimizer: Any | None,
    opt_param_scheduler: Any | None,
    iteration: int | None,
) -> None:
    if optimizer is None:
        return

    rank = dist.get_rank() if dist.is_initialized() else 0
    state_path = Path(adapter_dir) / f"training_state_rank{rank}.pt"
    torch.save(
        {
            "iteration": iteration,
            "optimizer": optimizer.state_dict(),
            "opt_param_scheduler": opt_param_scheduler.state_dict() if opt_param_scheduler else None,
        },
        state_path,
    )
    logger.info(f"Saved optimizer/scheduler state to {state_path.parent}")


def load_training_state(
    adapter_dir: Path,
    optimizer: Any | None,
    opt_param_scheduler: Any | None,
) -> int | None:
    if optimizer is None:
        return None

    rank = dist.get_rank() if dist.is_initialized() else 0
    state_path = Path(adapter_dir) / f"training_state_rank{rank}.pt"
    if not state_path.exists():
        return None

    training_state = torch.load(state_path, map_location="cpu", weights_only=False)

    optimizer.load_state_dict(training_state["optimizer"])
    logger.info("Restored optimizer state from PEFT checkpoint")

    if opt_param_scheduler is not None and training_state.get("opt_param_scheduler") is not None:
        opt_param_scheduler.load_state_dict(training_state["opt_param_scheduler"])
        logger.info("Restored LR scheduler state from PEFT checkpoint")

    iteration = training_state.get("iteration")
    if iteration is not None:
        logger.info(f"Resuming PEFT training from iteration {iteration}")
    return iteration


# ---------------------------------------------------------------------------
# Shared HF <-> Megatron module-name mappings (PEFT-neutral)
# ---------------------------------------------------------------------------


# Standard PEFT: merged Q/K/V and merged up/gate (default for LoRA and OFT).
_STANDARD_HF_TO_MEGATRON = {
    "q_proj": "linear_qkv",
    "k_proj": "linear_qkv",
    "v_proj": "linear_qkv",
    "o_proj": "linear_proj",
    "gate_proj": "linear_fc1",
    "up_proj": "linear_fc1",
    "down_proj": "linear_fc2",
    "embed_tokens": "word_embeddings",
    "lm_head": "output_layer",
}

_STANDARD_ALL_MODULES = ["linear_qkv", "linear_proj", "linear_fc1", "linear_fc2"]

# CanonicalLoRA: Split Q/K/V and up/gate (LoRA-only variant).
_CANONICAL_HF_TO_MEGATRON = {
    "q_proj": "linear_q",
    "k_proj": "linear_k",
    "v_proj": "linear_v",
    "o_proj": "linear_proj",
    "gate_proj": "linear_fc1_gate",
    "up_proj": "linear_fc1_up",
    "down_proj": "linear_fc2",
    "embed_tokens": "word_embeddings",
    "lm_head": "output_layer",
}

_CANONICAL_ALL_MODULES = [
    "linear_q",
    "linear_k",
    "linear_v",
    "linear_proj",
    "linear_fc1_up",
    "linear_fc1_gate",
    "linear_fc2",
]

# Fused Megatron leaf names accepted under variant="canonical" for backwards
# compat with legacy launchers. CanonicalOFT itself rejects these (it asserts
# in __post_init__), so we expand them to the split forms before the HF map.
_CANONICAL_FUSED_MEGATRON_TO_SPLIT = {
    "linear_qkv": ["linear_q", "linear_k", "linear_v"],
    "linear_fc1": ["linear_fc1_gate", "linear_fc1_up"],
}

# Multi-Latent Attention (DeepSeek v2/v3/v4): split low-rank Q/K/V projections.
# ``q_proj`` is used only when ``q_lora_rank is None`` (DeepSeek-Lite path);
# otherwise the model exposes ``q_a_proj`` / ``q_b_proj``. KV is always factored.
_MLA_HF_TO_MEGATRON = {
    "q_proj": "linear_q_proj",
    "q_a_proj": "linear_q_down_proj",
    "q_b_proj": "linear_q_up_proj",
    "kv_a_proj_with_mqa": "linear_kv_down_proj",
    "kv_b_proj": "linear_kv_up_proj",
    "o_proj": "linear_proj",
    "gate_proj": "linear_fc1",
    "up_proj": "linear_fc1",
    "down_proj": "linear_fc2",
}

_MLA_ALL_MODULES = [
    "linear_q_proj",
    "linear_q_down_proj",
    "linear_q_up_proj",
    "linear_kv_down_proj",
    "linear_kv_up_proj",
    "linear_proj",
    "linear_fc1",
    "linear_fc2",
]

# DeepSeek V4 uses native attention sublayer names. Its grouped attention
# exposes wq_a/wq_b/wkv/wo_a/wo_b directly; wkv must not match nested
# compressor modules, so use full-name globs instead of bare leaf names.
_DSV4_HF_TO_MEGATRON: dict[str, str] = {
    "wq_a": "*.self_attention.wq_a",
    "wq_b": "*.self_attention.wq_b",
    "wkv": "*.self_attention.wkv",
    "wo_a": "*.self_attention.wo_a",
    "wo_b": "*.self_attention.wo_b",
    "w1": "*.experts.*.w1",
    "w2": "*.experts.*.w2",
    "w3": "*.experts.*.w3",
}

_DSV4_ALL_MODULES = [
    "*.self_attention.wq_a",
    "*.self_attention.wq_b",
    "*.self_attention.wkv",
    "*.self_attention.wo_a",
    "*.self_attention.wo_b",
]

DSV4_MOE_HF_TARGET_MODULES = ["w1", "w2", "w3"]
DSV4_MOE_MEGATRON_TARGET_MODULES = [
    "*.experts.*.w1",
    "*.experts.*.w2",
    "*.experts.*.w3",
]

# Megatron -> HF (inverse mapping, one-to-many).
_MEGATRON_TO_HF_MODULES = {
    # Standard (merged layers)
    "linear_qkv": ["q_proj", "k_proj", "v_proj"],
    "linear_proj": ["o_proj"],
    "linear_fc1": ["gate_proj", "up_proj"],
    "linear_fc2": ["down_proj"],
    # Canonical (split layers)
    "linear_q": ["q_proj"],
    "linear_k": ["k_proj"],
    "linear_v": ["v_proj"],
    "linear_fc1_gate": ["gate_proj"],
    "linear_fc1_up": ["up_proj"],
    # MLA
    "linear_q_proj": ["q_proj"],
    "linear_q_down_proj": ["q_a_proj"],
    "linear_q_up_proj": ["q_b_proj"],
    "linear_kv_down_proj": ["kv_a_proj_with_mqa"],
    "linear_kv_up_proj": ["kv_b_proj"],
    # all-mode adds
    "word_embeddings": ["embed_tokens"],
    "output_layer": ["lm_head"],
}

_HF_MODULE_NAMES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    # MLA-only HF names
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    # all-mode adds
    "embed_tokens",
    "lm_head",
}

DEFAULT_HF_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

DEFAULT_MLA_HF_TARGET_MODULES = [
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

DEFAULT_DSV4_HF_TARGET_MODULES = [
    "wq_a",
    "wq_b",
    "wkv",
    "wo_a",
    "wo_b",
]


def detect_peft_variant(args: Namespace) -> Variant:
    """Pick the right HF<->Megatron mapping variant from runtime args.

    MLA models advertise themselves via Megatron's ``--multi-latent-attention``
    flag. DeepSeek V4 also sets MLA, but it uses native wq_a/wq_b/wkv/wo_a/wo_b
    sublayer names and must be selected explicitly with ``--peft-variant=dsv4``.
    CanonicalLoRA opts in via ``--peft-variant=canonical`` (LoRA-only).
    Everything else falls back to the standard merged-QKV mapping.
    """
    explicit = getattr(args, "peft_variant", "standard")
    if explicit == "dsv4":
        return "dsv4"
    if getattr(args, "multi_latent_attention", False):
        return "mla"
    if explicit == "canonical":
        return "canonical"
    return "standard"


def convert_target_modules_to_megatron(
    hf_modules: str | list[str],
    variant: Variant = "standard",
) -> list[str]:
    """Convert HuggingFace module names to Megatron module names.

    HF (standard):  q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    HF (MLA):       q_proj | q_a_proj/q_b_proj, kv_a_proj_with_mqa, kv_b_proj,
                    o_proj, gate_proj, up_proj, down_proj
    Megatron (standard):  linear_qkv, linear_proj, linear_fc1, linear_fc2
    Megatron (canonical): linear_q, linear_k, linear_v, linear_proj,
                          linear_fc1_up, linear_fc1_gate, linear_fc2
    Megatron (mla):       linear_q_proj | linear_q_down_proj/linear_q_up_proj,
                          linear_kv_down_proj, linear_kv_up_proj, linear_proj,
                          linear_fc1, linear_fc2

    ``variant`` controls LoRA's variant selection and OFT's explicit
    ``--oft-type oft`` compatibility path. Default OFT calls this with
    ``variant="canonical"`` so HF Q/K/V and gate/up targets map to split
    Megatron names for CanonicalOFT; the fused Megatron names ``linear_qkv``
    and ``linear_fc1`` are also accepted under this variant and are expanded
    into ``linear_q/k/v`` and ``linear_fc1_gate/linear_fc1_up`` respectively,
    so legacy launchers continue to work without an explicit migration.
    ``--oft-type oft`` calls this with the detected runtime variant, so
    fused-QKV models map Q/K/V to ``linear_qkv`` and gate/up to ``linear_fc1``
    for the legacy shared-R OFT wrapper.

    Special values: "all", "all-linear", "all_linear" -> all linear modules
    for the selected variant. If input is already in Megatron format, returns
    as-is.
    """
    if variant not in ("standard", "canonical", "mla", "dsv4"):
        raise ValueError(f"variant must be 'standard', 'canonical', 'mla', or 'dsv4', got {variant!r}")

    if variant == "dsv4":
        all_modules = _DSV4_ALL_MODULES
        hf_to_megatron = _DSV4_HF_TO_MEGATRON
    elif variant == "mla":
        all_modules = _MLA_ALL_MODULES
        hf_to_megatron = _MLA_HF_TO_MEGATRON
    elif variant == "canonical":
        all_modules = _CANONICAL_ALL_MODULES
        hf_to_megatron = _CANONICAL_HF_TO_MEGATRON
    else:
        all_modules = _STANDARD_ALL_MODULES
        hf_to_megatron = _STANDARD_HF_TO_MEGATRON

    if isinstance(hf_modules, str):
        if hf_modules in ("all", "all-linear", "all_linear"):
            return list(all_modules)
        hf_modules = [hf_modules]
    elif isinstance(hf_modules, list) and len(hf_modules) == 1:
        if hf_modules[0] in ("all", "all-linear", "all_linear"):
            return list(all_modules)

    if variant == "canonical":
        # Expand legacy fused names (linear_qkv, linear_fc1) into canonical split forms; see docstring.
        expanded_modules: list[str] = []
        for module in hf_modules:
            split_modules = _CANONICAL_FUSED_MEGATRON_TO_SPLIT.get(module, [module])
            for split_module in split_modules:
                if split_module not in expanded_modules:
                    expanded_modules.append(split_module)
        hf_modules = expanded_modules

    known_hf_names = set(hf_to_megatron)
    if all(m not in known_hf_names for m in hf_modules if "*" not in m):
        return hf_modules

    megatron_modules: list[str] = []
    for module in hf_modules:
        megatron_name = hf_to_megatron.get(module, module)
        if megatron_name not in megatron_modules:
            megatron_modules.append(megatron_name)
    return megatron_modules


def convert_target_modules_to_hf(megatron_modules: list[str]) -> list[str]:
    """Convert Megatron module names to HuggingFace module names.

    Supports both standard and canonical Megatron names.
    """
    hf_modules: list[str] = []
    for module in megatron_modules:
        if module in _MEGATRON_TO_HF_MODULES:
            hf_modules.extend(_MEGATRON_TO_HF_MODULES[module])
        else:
            hf_modules.append(module)
    return hf_modules


def parse_exclude_modules(args: Namespace, variant: Variant = "standard") -> list[str]:
    """Parse and convert the ``--exclude-modules`` argument to Megatron names."""
    exclude_modules: list[str] = []
    raw = getattr(args, "exclude_modules", None)
    if raw:
        if isinstance(raw, str):
            exclude_modules = [m.strip() for m in raw.split(",")]
        else:
            exclude_modules = list(raw)
        exclude_modules = convert_target_modules_to_megatron(exclude_modules, variant=variant)
    return exclude_modules


def resolve_target_modules_hf(args: Namespace) -> list[str]:
    """HF-format target modules from ``args``; falls back to the full all-linear set."""
    modules = getattr(args, "target_modules", None)
    variant = detect_peft_variant(args)
    if not modules:
        if variant == "dsv4":
            return list(DEFAULT_DSV4_HF_TARGET_MODULES)
        if variant == "mla":
            return list(DEFAULT_MLA_HF_TARGET_MODULES)
        return list(DEFAULT_HF_TARGET_MODULES)
    if variant == "dsv4":
        return list(modules) if isinstance(modules, list) else [m.strip() for m in modules.split(",")]
    return convert_target_modules_to_hf(list(modules))


# ---------------------------------------------------------------------------
# Shared PEFT checkpoint save/load (consumed by lora_utils and oft_utils)
# ---------------------------------------------------------------------------


def _to_peft_canonical_key(name: str) -> str:
    """Wrap a megatron-bridge adapter weight name into peft on-disk form.

    megatron-bridge emits LoRA/OFT adapter keys in two forms depending on
    the bridge version:

    * Suffix-only (older / unit-test inputs):
      ``model.layers.X.<mod>.lora_A`` / ``lora_B`` / ``oft_R``
    * With trailing ``.weight`` (bridge export_adapter_weights output):
      ``model.layers.X.<mod>.lora_A.weight`` etc.

    HF PEFT 0.19.x writes adapter safetensors with keys shaped
    ``base_model.model.<orig>.<adapter_suffix>.weight`` (the adapter name
    ``default`` is *not* in the on-disk key — peft's loader injects it
    during ``set_peft_model_state_dict`` via
    ``_insert_adapter_name_into_state_dict``).  Saving with ``.default.``
    already in the key causes peft to produce ``.default.default.weight``
    on load and miss every adapter tensor.

    This helper:
    * Prefixes with ``base_model.model.``
    * Validates the recognised adapter suffix (raises ``ValueError`` if
      none matches, so the save aborts rather than silently producing a
      non-loadable adapter)
    * Ensures the result ends in ``.weight`` (OFT bridge keys lack it,
      LoRA bridge keys already have it).
    """
    suffixes = ("oft_R", "lora_A", "lora_B")
    stripped = name[: -len(".weight")] if name.endswith(".weight") else name
    if not any(stripped.endswith(f".{suffix}") for suffix in suffixes):
        raise ValueError(
            f"cannot wrap adapter weight '{name}' to peft canonical form: "
            f"expected suffix in {suffixes}"
        )
    return f"base_model.model.{stripped}.weight"


def _save_peft_hf_artifacts(
    save_path: Path,
    state_dict: dict[str, torch.Tensor],
    config: dict[str, Any],
    *,
    base_model_name_or_path: str,
) -> None:
    """Write ``adapter_model.safetensors`` + ``adapter_config.json``.

    The on-disk format is HF-PEFT canonical so ``peft.PeftModel.from_pretrained``
    can load it directly: tensor keys are wrapped via
    ``_to_peft_canonical_key`` and the config carries
    ``base_model_name_or_path``.

    Tensors are cloned to guarantee distinct safetensors storage; safetensors
    forbids storage sharing, and the Megatron-Bridge LoRA exporter emits the
    fused-QKV / fused-gate-up A-side as a single tensor aliased under three HF
    names. ``.clone()`` also yields a contiguous tensor.
    """
    if not base_model_name_or_path:
        raise ValueError(
            "_save_peft_hf_artifacts: base_model_name_or_path must be a non-empty "
            "string (typically args.hf_checkpoint)"
        )

    serializable = {
        _to_peft_canonical_key(name): tensor.detach().clone()
        for name, tensor in state_dict.items()
    }
    safetensors_save_file(serializable, str(save_path / "adapter_model.safetensors"))

    enriched_config = dict(config)
    enriched_config["base_model_name_or_path"] = base_model_name_or_path
    with open(save_path / "adapter_config.json", "w") as f:
        json.dump(enriched_config, f, indent=2)

    os.sync()
    logger.info(f"Saved HF PEFT adapter to {save_path} with {len(serializable)} tensors")


def save_peft_adapter_checkpoint(
    model: Sequence[torch.nn.Module],
    args: Namespace,
    save_dir: str,
    *,
    method: Literal["lora", "oft"],
    build_config: Any,  # callable() -> dict
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
    iteration: int | None = None,
) -> str:
    """Save a PEFT adapter checkpoint (native per-rank shards + HF artifacts).

    Both LoRA and OFT use this helper; the only method-specific pieces are the
    bridge exporter and the ``adapter_config.json`` contents.
    """
    from megatron.bridge import AutoBridge

    from orbit.utils import megatron_bridge_utils

    save_path = Path(save_dir)
    is_dp_rank_0 = get_parallel_state().intra_dp.rank == 0
    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()

    if is_dp_rank_0:
        save_path.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    # Megatron-native format (per TP/PP rank, fast resume)
    if is_dp_rank_0:
        adapter_state = {
            name: param.data.cpu()
            for chunk in model
            for name, param in chunk.named_parameters()
            if is_adapter_param_name(name)
        }
        native_path = save_path / f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt"
        torch.save(adapter_state, native_path)
        logger.info(f"Saved {len(adapter_state)} adapter tensors (native) to {native_path}")

    # HF PEFT format — bridge export is TP-collective, so every rank calls it.
    bridge = AutoBridge.from_hf_pretrained(args.hf_checkpoint, trust_remote_code=True)
    exporter = (
        bridge.export_oft_adapter_weights if method == "oft" else bridge.export_adapter_weights
    )

    state_dict: dict[str, torch.Tensor] = {}
    with megatron_bridge_utils.patch_megatron_model(model):
        # megatron-bridge >=0.5 yields a 2-tuple (hf_name, tensor); older
        # versions yielded 3-tuples. Positional unpack handles both.
        for item in exporter(model, cpu=True, show_progress=False):
            hf_name, weight = item[0], item[1]
            state_dict[hf_name] = weight

    if is_dp_rank_0 and tp_rank == 0:
        _save_peft_hf_artifacts(
            save_path,
            state_dict,
            config=build_config(),
            base_model_name_or_path=args.hf_checkpoint,
        )

    save_training_state(save_path, optimizer, opt_param_scheduler, iteration)

    if dist.is_initialized():
        dist.barrier()

    return str(save_path)


def load_peft_adapter_checkpoint(
    model: Sequence[torch.nn.Module],
    adapter_path: str,
    *,
    label: str,
    optimizer: Any | None = None,
    opt_param_scheduler: Any | None = None,
) -> tuple[bool, int | None]:
    """Load a PEFT adapter checkpoint from Megatron-native shards.

    ``label`` is the user-visible method name (``"LoRA"`` / ``"OFT"``) used in
    log messages. If only an HF PEFT artifact exists (no native shards), warns
    and returns ``(False, None)``.
    """
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        logger.warning(f"{label} adapter path does not exist: {adapter_dir}")
        return False, None

    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()

    native_path = adapter_dir / f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt"
    if native_path.exists():
        state_dict = torch.load(native_path, map_location="cpu", weights_only=True)
        loaded = 0
        legacy_oft_broadcasts = 0
        for chunk in model:
            for name, param in chunk.named_parameters():
                if name in state_dict:
                    param.data.copy_(state_dict[name].to(device=param.device))
                    loaded += 1
                    continue
                # CanonicalOFT legacy-checkpoint compatibility: a pre-fix run
                # saved a single shared `adapter.oft_r` for the fused QKV/FC1
                # wrappers. The new model has `adapter_q/k/v/gate/up.oft_r`;
                # broadcast the legacy single R into each slice so resume
                # remains possible (the first step diverges, which is the
                # intended math correction).
                legacy_key = _maybe_legacy_canonical_oft_key(name)
                if legacy_key is not None and legacy_key in state_dict:
                    param.data.copy_(state_dict[legacy_key].to(device=param.device))
                    loaded += 1
                    legacy_oft_broadcasts += 1
        if legacy_oft_broadcasts:
            logger.warning(
                f"{label}: broadcast {legacy_oft_broadcasts} legacy shared-R OFT "
                f"tensors into CanonicalOFT split slices. First training step "
                f"after resume will diverge from the legacy shared-R math."
            )
        if loaded == 0:
            logger.warning(
                f"{label} adapter checkpoint at {native_path} did not match any local adapter tensors"
            )
            return False, None
        logger.info(f"Loaded {loaded} adapter tensors from Megatron-native checkpoint: {native_path}")
        iteration = load_training_state(adapter_dir, optimizer, opt_param_scheduler)
        return True, iteration

    hf_safetensors = adapter_dir / "adapter_model.safetensors"
    hf_bin = adapter_dir / "adapter_model.bin"
    if hf_safetensors.exists() or hf_bin.exists():
        found = hf_safetensors if hf_safetensors.exists() else hf_bin
        logger.warning(
            f"Found HF PEFT adapter at {found} but direct HF PEFT loading into "
            f"Megatron is not yet supported. Please save using Megatron-native format "
            f"(adapter_megatron_tp*_pp*.pt files) for checkpoint resume."
        )
        return False, None

    logger.warning(f"No adapter checkpoint found at {adapter_dir}")
    return False, None


def load_adapter_tensors_for_teacher(
    model: Sequence[torch.nn.Module],
    adapter_path: str,
) -> dict[str, torch.Tensor]:
    """Load a frozen teacher adapter as a name->tensor dict (never into params).

    Requires Megatron-native shards (adapter_megatron_tp{tp}_pp{pp}.pt) as
    written by save_peft_checkpoint; HF-only artifacts are rejected — the
    engine side consumes those, the trainer side needs native names/shapes.
    """
    adapter_dir = Path(adapter_path)
    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    native_path = adapter_dir / f"adapter_megatron_tp{tp_rank}_pp{pp_rank}.pt"
    if not native_path.exists():
        raise FileNotFoundError(
            f"OPD teacher adapter needs Megatron-native shards, missing {native_path}. "
            "Save the teacher with orbit's save_peft_checkpoint (HF-only artifacts are not "
            "loadable trainer-side)."
        )
    state_dict = torch.load(native_path, map_location="cpu", weights_only=True)
    tensors: dict[str, torch.Tensor] = {}
    for chunk in model:
        for name, param in chunk.named_parameters():
            if not is_adapter_param_name(name):
                continue
            source = state_dict.get(name)
            if source is None:
                legacy_key = _maybe_legacy_canonical_oft_key(name)
                if legacy_key is not None:
                    source = state_dict.get(legacy_key)
            if source is None:
                raise KeyError(f"Teacher adapter shard {native_path} is missing tensor {name!r}.")
            tensors[name] = source.to(device=param.device, dtype=param.dtype)
    return tensors
