from __future__ import annotations

import contextlib
import logging
import os
import re
from pathlib import Path

import torch
from torch import nn

from orbit.megatron.tensor_semantics import should_skip_named_tensor_for_meta_validation

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def _torch_load_weights_only_false():
    """Force ``torch.load`` to ``weights_only=False`` for the duration of the block.

    PyTorch 2.6 flipped the ``weights_only`` default to ``True``. ModelOpt
    serialises its per-module quantizer extra_state with pickle protocol 4,
    whose opcodes are not whitelisted by the new safe Unpickler (typical
    failure: ``WeightsUnpickler error: Unsupported operand 149``). The
    checkpoints we restore here come from our trusted Megatron-Bridge direct
    conversion path, so loading them with the legacy unpickler is safe; scope
    the override to a single restore call so the global default is unchanged.
    """

    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = patched_load
    try:
        yield
    finally:
        torch.load = original_load

_LEGACY_MODEL_OPT_CONFIG_KEYS = {
    "max_calibrate": {"use_sequential"},
}
_MODEL_OPT_COMPAT_PATCH_ATTR = "_orbit_modelopt_config_compat_patched"


def _quantization_config_from_hf_config(hf_config):
    if hf_config is None:
        return None
    if isinstance(hf_config, dict):
        quantization_config = hf_config.get("quantization_config")
    else:
        quantization_config = getattr(hf_config, "quantization_config", None)

    if quantization_config is not None and hasattr(quantization_config, "to_dict"):
        quantization_config = quantization_config.to_dict()
    return quantization_config


def hf_config_is_low_precision(hf_config) -> bool:
    return bool(_quantization_config_from_hf_config(hf_config))


def resolve_distributed_checkpoint_dir(path: str | os.PathLike[str] | None) -> Path | None:
    if not path:
        return None

    checkpoint_dir = Path(path)
    if not checkpoint_dir.is_dir():
        return None

    if (checkpoint_dir / ".metadata").is_file():
        return checkpoint_dir

    latest_step_file = checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not latest_step_file.is_file():
        return None

    try:
        latest_step = latest_step_file.read_text().strip()
    except OSError:
        latest_step = ""

    if latest_step.isdigit():
        latest_iter_dir = checkpoint_dir / f"iter_{int(latest_step):07d}"
        if (latest_iter_dir / ".metadata").is_file():
            return latest_iter_dir
    elif latest_step:
        named_latest_dir = checkpoint_dir / latest_step
        if (named_latest_dir / ".metadata").is_file():
            return named_latest_dir

    candidates = sorted(
        candidate.parent for candidate in checkpoint_dir.glob("iter_*/.metadata") if candidate.parent.is_dir()
    )
    if candidates:
        return candidates[-1]

    return None


def is_distributed_checkpoint(path: str | os.PathLike[str] | None) -> bool:
    return resolve_distributed_checkpoint_dir(path) is not None


def is_legacy_megatron_checkpoint(path: str | os.PathLike[str] | None) -> bool:
    if not path:
        return False
    checkpoint_dir = Path(path)
    return (checkpoint_dir / "latest_checkpointed_iteration.txt").is_file() or bool(
        re.fullmatch(r"iter_\d{7}", checkpoint_dir.name)
    )


def is_megatron_checkpoint_path(path: str | os.PathLike[str] | None) -> bool:
    return is_legacy_megatron_checkpoint(path) or is_distributed_checkpoint(path)


def load_hf_config(args):
    from transformers import AutoConfig

    if not getattr(args, "hf_checkpoint", None):
        raise ValueError("hf_checkpoint is required to inspect low-precision bridge bootstrap settings.")
    return AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)


def requires_low_precision_dist_checkpoint(args, *, hf_config=None) -> bool:
    if getattr(args, "megatron_to_hf_mode", None) != "bridge":
        return False
    if hf_config is None:
        if not getattr(args, "hf_checkpoint", None):
            return False
        hf_config = load_hf_config(args)
    return hf_config_is_low_precision(hf_config)


def resolve_bridge_load_path(args, *, hf_config=None) -> str | None:
    load_path = getattr(args, "load", None)
    if load_path is not None:
        return load_path
    if requires_low_precision_dist_checkpoint(args, hf_config=hf_config):
        return getattr(args, "ref_load", None)
    return getattr(args, "ref_load", None) or getattr(args, "hf_checkpoint", None)


def validate_low_precision_bootstrap_args(args, *, hf_config=None) -> None:
    if not requires_low_precision_dist_checkpoint(args, hf_config=hf_config):
        return

    if getattr(args, "critic_mode", None) == "adapter":
        raise ValueError(
            "--critic-mode adapter does not support low-precision/quantized bridge checkpoints: "
            "one-trunk aliasing currently shares Parameters only, while quantized trunk weights and scales "
            "are stored in checkpoint-created buffers. Use --critic-mode full or a high-precision base model."
        )

    load_path = resolve_bridge_load_path(args, hf_config=hf_config)
    if not load_path:
        raise ValueError(
            "Low-precision bridge runs require --load or --ref-load to point to a Megatron checkpoint; "
            "HF-only startup is not supported."
        )
    if not os.path.exists(load_path):
        raise ValueError(
            f"Low-precision bridge runs require an existing Megatron checkpoint path, got {load_path!r}. "
            "HF-only startup is not supported."
        )
    if not is_megatron_checkpoint_path(load_path):
        raise ValueError(
            "Low-precision bridge runs require --load/--ref-load to point to a Megatron checkpoint directory; "
            "HF-only startup is not supported."
        )


def should_preload_low_precision_model_before_optimizer(args, *, role: str, hf_config=None) -> bool:
    peft_method = getattr(args, "peft_method", None)
    peft_enabled = peft_method not in (None, "", "none")
    if role != "actor" or not peft_enabled:
        return False
    return requires_low_precision_dist_checkpoint(args, hf_config=hf_config)


def detect_int4_checkpoint(dist_weight_path: str) -> tuple[bool, bool]:
    """Detect whether a distributed checkpoint carries INT4 packed tensors."""
    resolved_dist_dir = resolve_distributed_checkpoint_dir(dist_weight_path)
    if resolved_dist_dir is None:
        return False, False

    try:
        from torch.distributed.checkpoint import FileSystemReader

        metadata = FileSystemReader(str(resolved_dist_dir)).read_metadata()
        checkpoint_keys = list(metadata.state_dict_metadata.keys())
    except Exception:
        return False, False

    has_expert_int4 = False
    has_dense_int4 = False
    for checkpoint_key in checkpoint_keys:
        key_text = str(checkpoint_key)
        if "_packed" not in key_text or "weight" not in key_text:
            continue
        if "experts" in key_text:
            has_expert_int4 = True
        else:
            has_dense_int4 = True
        if has_expert_int4 and has_dense_int4:
            break

    return has_expert_int4, has_dense_int4


def detect_nvfp4_checkpoint(dist_weight_path: str) -> bool:
    """Detect whether a distributed checkpoint carries NVFP4 quantizer scale tensors."""
    resolved_dist_dir = resolve_distributed_checkpoint_dir(dist_weight_path)
    if resolved_dist_dir is None:
        return False

    try:
        from torch.distributed.checkpoint import FileSystemReader

        metadata = FileSystemReader(str(resolved_dist_dir)).read_metadata()
        checkpoint_keys = list(metadata.state_dict_metadata.keys())
    except Exception:
        return False

    scale_key_re = re.compile(r"weight_quantizer\._scale(\d+)?$")
    return any(scale_key_re.search(str(checkpoint_key)) is not None for checkpoint_key in checkpoint_keys)


def detect_fp8_checkpoint(dist_weight_path: str) -> bool:
    """Detect direct-write FP8 checkpoints with native weight scale tensors."""
    resolved_dist_dir = resolve_distributed_checkpoint_dir(dist_weight_path)
    if resolved_dist_dir is None:
        return False

    try:
        from torch.distributed.checkpoint import FileSystemReader

        metadata = FileSystemReader(str(resolved_dist_dir)).read_metadata()
        checkpoint_keys = list(metadata.state_dict_metadata.keys())
    except Exception:
        return False

    scale_key_re = re.compile(r"\.weight\d*_scale_inv$")
    return any(scale_key_re.search(str(checkpoint_key)) is not None for checkpoint_key in checkpoint_keys)


def _modelopt_common_state_path(dist_weight_path: str | os.PathLike[str] | None) -> Path | None:
    resolved_dist_dir = resolve_distributed_checkpoint_dir(dist_weight_path)
    if resolved_dist_dir is None:
        return None
    common_state_path = resolved_dist_dir / "modelopt_state" / "common.pt"
    return common_state_path if common_state_path.is_file() else None


def _checkpoint_has_modelopt_quantizer_keys(dist_weight_path: str | os.PathLike[str] | None) -> bool:
    resolved_dist_dir = resolve_distributed_checkpoint_dir(dist_weight_path)
    if resolved_dist_dir is None:
        return False

    try:
        from torch.distributed.checkpoint import FileSystemReader

        metadata = FileSystemReader(str(resolved_dist_dir)).read_metadata()
        checkpoint_keys = metadata.state_dict_metadata.keys()
    except Exception:
        return False

    return any("weight_quantizer" in str(key) or "input_quantizer" in str(key) for key in checkpoint_keys)


def has_modelopt_checkpoint_state(dist_weight_path: str | os.PathLike[str] | None) -> bool:
    """Detect whether a distributed checkpoint carries nontrivial ModelOpt state."""
    common_state_path = _modelopt_common_state_path(dist_weight_path)
    if common_state_path is None:
        return _checkpoint_has_modelopt_quantizer_keys(dist_weight_path)

    try:
        modelopt_state = torch.load(common_state_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        logger.warning("Could not inspect ModelOpt state at %s: %s", common_state_path, exc)
        return True

    modes = modelopt_state.get("modelopt_state_dict", [])
    for mode in modes:
        mode_name = mode[0] if isinstance(mode, (list, tuple)) and mode else None
        if mode_name != "kd_loss":
            return True

    return _checkpoint_has_modelopt_quantizer_keys(dist_weight_path)


def configure_provider_for_low_precision(provider, load_path: str | None) -> bool:
    if not is_distributed_checkpoint(load_path):
        return False

    has_expert_int4, has_dense_int4 = detect_int4_checkpoint(load_path)
    if has_expert_int4 or has_dense_int4:
        checkpoint_kind = "INT4"
    elif detect_nvfp4_checkpoint(load_path):
        checkpoint_kind = "NVFP4"
    elif detect_fp8_checkpoint(load_path):
        checkpoint_kind = "FP8"
    else:
        return False

    provider.init_model_with_meta_device = True
    provider.perform_initialization = False
    logger.info(
        "Enabled meta-device provider init for %s distributed checkpoint %s "
        "(init_model_with_meta_device=True, perform_initialization=False)",
        checkpoint_kind,
        load_path,
    )
    return True


def filter_missing_oft_adapter_keys(sharded_state_dict, checkpoint_keys):
    checkpoint_key_set = {str(key) for key in checkpoint_keys}
    missing_oft_keys = []
    filtered_state_dict = {}
    for key, value in sharded_state_dict.items():
        key_text = str(key)
        if _is_oft_adapter_param_name(key_text) and key_text not in checkpoint_key_set:
            missing_oft_keys.append(key_text)
            continue
        filtered_state_dict[key] = value
    return filtered_state_dict, missing_oft_keys


def _is_oft_adapter_param_name(name: str) -> bool:
    lowered = name.lower()
    canonical_suffixes = (
        "adapter_q.oft_r",
        "adapter_k.oft_r",
        "adapter_v.oft_r",
        "adapter_gate.oft_r",
        "adapter_up.oft_r",
    )
    return (
        lowered.endswith("adapter.oft_r")
        or lowered.endswith(canonical_suffixes)
        or lowered.endswith(("w1_oft_r", "w2_oft_r", "w3_oft_r"))
    )


def filter_missing_extra_state_keys(sharded_state_dict, checkpoint_keys):
    checkpoint_key_set = {str(key) for key in checkpoint_keys}
    missing_extra_state_keys = []
    filtered_state_dict = {}
    for key, value in sharded_state_dict.items():
        key_text = str(key)
        if "._extra_state" in key_text and key_text not in checkpoint_key_set:
            missing_extra_state_keys.append(key_text)
            continue
        filtered_state_dict[key] = value
    return filtered_state_dict, missing_extra_state_keys


def _loaded_tensor_payload(value):
    if isinstance(value, torch.Tensor):
        return value

    payload = getattr(value, "data", None)
    if isinstance(payload, torch.Tensor):
        return payload

    return None


def _replace_meta_module_tensor(module: nn.Module, name: str, payload: torch.Tensor, *, is_buffer: bool) -> bool:
    if "." in name:
        parent_name, leaf_name = name.rsplit(".", 1)
        owner = module.get_submodule(parent_name)
    else:
        owner = module
        leaf_name = name

    target_owner = owner
    target_tensor = owner._buffers.get(leaf_name) if is_buffer else owner._parameters.get(leaf_name)
    if (target_tensor is None or target_tensor.device.type != "meta") and hasattr(owner, "to_wrap"):
        wrapped_owner = owner.to_wrap
        wrapped_tensor = wrapped_owner._buffers.get(leaf_name) if is_buffer else wrapped_owner._parameters.get(leaf_name)
        if wrapped_tensor is not None and wrapped_tensor.device.type == "meta":
            target_owner = wrapped_owner
            target_tensor = wrapped_tensor

    if is_buffer:
        if target_tensor is None or target_tensor.device.type != "meta":
            return False
        target_owner._buffers[leaf_name] = payload
        return True

    param = target_tensor
    if param is None or param.device.type != "meta":
        return False

    if param.requires_grad and not (payload.is_floating_point() or payload.is_complex()):
        raise RuntimeError(
            "Cannot materialize meta parameter "
            f"{name!r} from non-floating payload dtype={payload.dtype} "
            f"with requires_grad={param.requires_grad}."
        )

    replacement = nn.Parameter(payload, requires_grad=param.requires_grad)
    try:
        from megatron.core import tensor_parallel

        tensor_parallel.copy_tensor_model_parallel_attributes(replacement, param)
    except Exception:
        pass

    for attr_name, attr_value in vars(param).items():
        if attr_name.startswith("_") or attr_name in {"data", "grad"}:
            continue
        setattr(replacement, attr_name, attr_value)

    target_owner._parameters[leaf_name] = replacement
    return True


_NVFP4_EXPERT_PACKED_WEIGHT_RE = re.compile(
    r"^(?P<module>.*\.experts\.linear_fc[12])\.weight(?P<idx>\d+)(?:_[wv])?$"
)
_NVFP4_EXPERT_QUANTIZER_RE = re.compile(
    r"^(?P<module>.*\.experts\.linear_fc[12])\.weight_quantizer\._(?:scale|double_scale|amax)\d+$"
)
_NVFP4_QUANTIZER_DIGIT_RE = re.compile(r"\.(_scale|_double_scale|_amax)\d+$")
_NVFP4_DENSE_PACKED_WEIGHT_SUFFIXES = (".weight_w", ".weight_v", ".weight")


def _collect_nvfp4_expert_quantizer_modules(loaded_keys: set[str], prefix: str = "") -> set[str]:
    modules = set()
    for key in loaded_keys:
        lookup_key = key
        if prefix and lookup_key.startswith(prefix):
            lookup_key = lookup_key[len(prefix) :]
        match = _NVFP4_EXPERT_QUANTIZER_RE.match(lookup_key)
        if match is not None:
            modules.add(match.group("module"))
    return modules


def _has_nvfp4_expert_quantizer_sibling(
    lookup_key: str,
    loaded_keys: set[str],
    prefix: str = "",
    *,
    expert_quantizer_modules: set[str] | None = None,
) -> bool:
    match = _NVFP4_EXPERT_PACKED_WEIGHT_RE.match(lookup_key)
    if match is None:
        return False

    module_path = match.group("module")
    if expert_quantizer_modules is not None and module_path in expert_quantizer_modules:
        return True

    expert_idx = match.group("idx")
    candidates = {
        f"{module_path}.weight_quantizer._scale{expert_idx}",
        f"{module_path}.weight_quantizer._double_scale{expert_idx}",
        f"{module_path}.weight_quantizer._amax{expert_idx}",
    }
    if prefix:
        candidates.update(f"{prefix}{candidate}" for candidate in tuple(candidates))
    if any(candidate in loaded_keys for candidate in candidates):
        return True

    if expert_quantizer_modules is None:
        return module_path in _collect_nvfp4_expert_quantizer_modules(loaded_keys, prefix=prefix)
    return False


def _nvfp4_dense_module_path_from_packed_weight(lookup_key: str) -> str | None:
    for suffix in _NVFP4_DENSE_PACKED_WEIGHT_SUFFIXES:
        if lookup_key.endswith(suffix):
            return lookup_key[: -len(suffix)]
    return None


def _has_nvfp4_dense_quantizer_sibling(lookup_key: str, loaded_keys: set[str], prefix: str = "") -> bool:
    module_path = _nvfp4_dense_module_path_from_packed_weight(lookup_key)
    if module_path is None:
        return False

    candidates = {
        f"{module_path}.weight_quantizer._scale",
        f"{module_path}.weight_quantizer._double_scale",
        f"{module_path}.weight_quantizer._amax",
    }
    if prefix:
        candidates.update(f"{prefix}{candidate}" for candidate in tuple(candidates))
    return any(candidate in loaded_keys for candidate in candidates)


def _cleanup_nvfp4_dense_split_meta_parameters(module: nn.Module, sharded_state_dict, prefix: str = "") -> int:
    loaded_keys = {str(key) for key in sharded_state_dict}
    module_paths = set()

    for key in loaded_keys:
        lookup_key = str(key)
        if prefix and lookup_key.startswith(prefix):
            lookup_key = lookup_key[len(prefix) :]
        if not lookup_key.endswith((".weight_w", ".weight_v")):
            continue
        if not _has_nvfp4_dense_quantizer_sibling(lookup_key, loaded_keys, prefix=prefix):
            continue
        module_path = _nvfp4_dense_module_path_from_packed_weight(lookup_key)
        if module_path is not None:
            module_paths.add(module_path)

    removed = 0
    for module_path in sorted(module_paths):
        try:
            owner = module.get_submodule(module_path)
        except AttributeError:
            continue

        weight = owner._parameters.get("weight")
        if weight is None or weight.device.type == "meta":
            continue

        for leaf_name in ("weight_w", "weight_v"):
            split_param = owner._parameters.get(leaf_name)
            if split_param is not None and split_param.device.type == "meta":
                del owner._parameters[leaf_name]
                removed += 1

    return removed


def _materialize_loaded_meta_module_tensors(module: nn.Module, sharded_state_dict, prefix: str = "") -> int:
    rebound = 0

    # Pre-compute the loaded-key set for the NVFP4 dense-weight sibling check below.
    loaded_keys = {str(key) for key in sharded_state_dict}
    nvfp4_expert_quantizer_modules = _collect_nvfp4_expert_quantizer_modules(loaded_keys, prefix=prefix)

    for key, value in sharded_state_dict.items():
        lookup_key = str(key)
        if prefix and lookup_key.startswith(prefix):
            lookup_key = lookup_key[len(prefix) :]
        if lookup_key.endswith(("_packed", "_scale", "_shape")) or "._extra_state" in lookup_key:
            continue
        # NVFP4: skip quantizer scale/double_scale/amax entries (both bare dense and
        # digit-suffixed expert variants). They are consumed by
        # register_nvfp4_buffers_after_load_dense / register_nvfp4_buffers_after_load
        # later in load_dist_checkpoint -- no module-side attribute path exists for
        # them since we don't carry modelopt's weight_quantizer submodule at runtime.
        if ".weight_quantizer." in lookup_key or lookup_key.endswith("_amax"):
            continue
        if _NVFP4_QUANTIZER_DIGIT_RE.search(lookup_key):
            continue
        # NVFP4 expert: skip per-expert packed weights (weight{N}, weight{N}_w/_v)
        # only when NVFP4 quantizer siblings are present. FP8 grouped experts use
        # the same weight{N} names for real runtime parameters after the FP8
        # split-weight register combines weight{N}_{w,v}.
        if _has_nvfp4_expert_quantizer_sibling(
            lookup_key,
            loaded_keys,
            prefix=prefix,
            expert_quantizer_modules=nvfp4_expert_quantizer_modules,
        ):
            continue
        # NVFP4 dense: skip packed `weight` and split SwiGLU `weight_w`/`weight_v`
        # keys when quantizer siblings are loaded. The dense register hook consumes
        # these entries and rebuilds the runtime bf16 `weight` Parameter.
        if _has_nvfp4_dense_quantizer_sibling(lookup_key, loaded_keys, prefix=prefix):
            continue

        payload = _loaded_tensor_payload(value)
        if payload is None or payload.device.type == "meta":
            continue

        if _replace_meta_module_tensor(module, lookup_key, payload, is_buffer=False):
            rebound += 1
            continue
        if _replace_meta_module_tensor(module, lookup_key, payload, is_buffer=True):
            rebound += 1
    return rebound


def _checkpoint_uses_explicit_layer_keys(checkpoint_keys) -> bool:
    layer_key_re = re.compile(r"(^|\.)layers\.\d+\.")
    return any(layer_key_re.search(str(key)) for key in checkpoint_keys)


def _dist_checkpoint_already_loaded(parallel_model, dist_weight_path, prefix="") -> bool:
    expected_path = os.path.abspath(os.path.expanduser(str(dist_weight_path)))
    expected_prefix = prefix or ""
    if not parallel_model:
        return False
    for model in parallel_model:
        if getattr(model, "_orbit_loaded_dist_checkpoint_path", None) != expected_path:
            return False
        if getattr(model, "_orbit_loaded_dist_checkpoint_prefix", None) != expected_prefix:
            return False
    return True


def _mark_dist_checkpoint_as_loaded(parallel_model, dist_weight_path, prefix="") -> None:
    loaded_path = os.path.abspath(os.path.expanduser(str(dist_weight_path)))
    loaded_prefix = prefix or ""
    for model in parallel_model:
        model._orbit_loaded_dist_checkpoint_path = loaded_path
        model._orbit_loaded_dist_checkpoint_prefix = loaded_prefix


def _modelopt_state_already_restored(parallel_model, checkpoint_path) -> bool:
    expected_path = os.path.abspath(os.path.expanduser(str(checkpoint_path)))
    return bool(parallel_model) and all(
        getattr(model, "_orbit_restored_modelopt_checkpoint_path", None) == expected_path
        for model in parallel_model
    )


def _mark_modelopt_state_restored(parallel_model, checkpoint_path) -> None:
    restored_path = os.path.abspath(os.path.expanduser(str(checkpoint_path)))
    for model in parallel_model:
        model._orbit_restored_modelopt_checkpoint_path = restored_path


def _sanitize_modelopt_mode_config(mode, config):
    legacy_keys = _LEGACY_MODEL_OPT_CONFIG_KEYS.get(str(mode))
    if not legacy_keys or not isinstance(config, dict):
        return config

    removed_keys = sorted(key for key in legacy_keys if key in config)
    if not removed_keys:
        return config

    sanitized = dict(config)
    for key in removed_keys:
        sanitized.pop(key)
    logger.info(
        "Dropped legacy ModelOpt config key(s) for %s: %s",
        mode,
        ", ".join(removed_keys),
    )
    return sanitized


def _patch_modelopt_config_compatibility() -> bool:
    try:
        import modelopt.torch.opt.conversion as mto
    except ImportError:
        return False

    manager_cls = getattr(mto, "ModeloptStateManager", None)
    if manager_cls is None:
        return False

    original_get_config_class = manager_cls.get_config_class
    if getattr(original_get_config_class, _MODEL_OPT_COMPAT_PATCH_ATTR, False):
        return True

    def compat_get_config_class(mode, config):
        return original_get_config_class(mode, _sanitize_modelopt_mode_config(mode, config))

    setattr(compat_get_config_class, _MODEL_OPT_COMPAT_PATCH_ATTR, True)
    manager_cls.get_config_class = staticmethod(compat_get_config_class)
    return True


def _restore_modelopt_state_before_load(parallel_model, checkpoint_path) -> bool:
    # NVFP4 now follows the INT4-style design: dense + expert pre-load shape
    # transforms and post-load buffer registers handle quantizer state directly,
    # so the modelopt-based restore path is not needed for NVFP4 checkpoints.
    if detect_nvfp4_checkpoint(checkpoint_path):
        return False
    # NVFP4 actor/ref need ModelOpt to register the per-module quantizer
    # buffers (``weight_quantizer._scale`` etc.) before the DCP load can match
    # the saved shapes. FP8 doesn't need this — FP8 quant scales live in
    # regular tensor keys (``weight_scale_inv``) — so the FP8 launcher exports
    # ORBIT_RESTORE_MODELOPT_STATE=0 to skip the restore. Default remains "1"
    # so existing NVFP4 / future low-precision flows keep their current
    # behaviour.
    if os.environ.get("ORBIT_RESTORE_MODELOPT_STATE", "1") == "0":
        logger.info(
            "ORBIT_RESTORE_MODELOPT_STATE=0; skipping ModelOpt state restore for %s",
            checkpoint_path,
        )
        return False

    try:
        from megatron.bridge.training.post_training.checkpointing import (
            has_modelopt_state,
            load_modelopt_state,
        )
    except ImportError:
        return False

    if not has_modelopt_state(checkpoint_path):
        return False
    if _modelopt_state_already_restored(parallel_model, checkpoint_path):
        return True

    _patch_modelopt_config_compatibility()
    with _torch_load_weights_only_false():
        load_modelopt_state(parallel_model, checkpoint_path)
    _mark_modelopt_state_restored(parallel_model, checkpoint_path)
    logger.info(
        "Restored ModelOpt state before distributed checkpoint load from %s",
        checkpoint_path,
    )
    return True


def _infer_materialization_device(module: nn.Module) -> torch.device:
    for tensor in list(module.parameters()) + list(module.buffers()):
        if tensor.device.type != "meta":
            return tensor.device
    return torch.device("cpu")


def _materialize_meta_parameter(module: nn.Module, name: str, target_device: torch.device) -> torch.Tensor | None:
    named_params = dict(module.named_parameters())
    param = named_params.get(name)
    if param is None or param.device.type != "meta":
        return None

    payload = torch.zeros(param.shape, dtype=param.dtype, device=target_device)
    replaced = _replace_meta_module_tensor(module, name, payload, is_buffer=False)
    if not replaced:
        return None
    return payload


def _zero_oft_adapter_parameters(module: nn.Module, missing_oft_keys: list[str]) -> None:
    if not missing_oft_keys:
        return

    target_device = _infer_materialization_device(module)
    named_params = dict(module.named_parameters())
    zeroed_any = False
    with torch.no_grad():
        for key in missing_oft_keys:
            param = named_params.get(key)
            if param is not None and param.device.type == "meta":
                _materialize_meta_parameter(module, key, target_device)
                named_params = dict(module.named_parameters())
                param = named_params.get(key)
            if param is not None:
                param.zero_()
                zeroed_any = True
        if not zeroed_any:
            for name, param in named_params.items():
                if _is_oft_adapter_param_name(name):
                    if param.device.type == "meta":
                        _materialize_meta_parameter(module, name, target_device)
                        named_params = dict(module.named_parameters())
                        param = named_params[name]
                    param.zero_()


def _remaining_meta_tensor_names(module: nn.Module) -> list[str]:
    remaining = []
    for name, param in module.named_parameters():
        if param.device.type == "meta" and not should_skip_named_tensor_for_meta_validation(module, name, param):
            remaining.append(name)
    for name, buffer in module.named_buffers():
        if buffer.device.type == "meta" and not should_skip_named_tensor_for_meta_validation(module, name, buffer):
            remaining.append(name)
    return remaining


def _raise_if_meta_tensors_remain(module: nn.Module, *, checkpoint_path: str, prefix: str = "") -> None:
    remaining = _remaining_meta_tensor_names(module)
    if not remaining:
        return

    sample = ", ".join(sorted(remaining)[:8])
    if len(remaining) > 8:
        sample += ", ..."
    raise ValueError(
        "Remaining meta tensors after distributed checkpoint load from "
        f"{checkpoint_path!r} (prefix={prefix!r}): {sample}"
    )


def _import_dist_checkpointing():
    from megatron.core import dist_checkpointing

    return dist_checkpointing


def _materialize_meta_sharded_tensor_data(sharded_state_dict) -> None:
    try:
        from megatron.core.dist_checkpointing import mapping as mcore_mapping
    except (ImportError, AttributeError):
        return

    MCoreShardedTensor = getattr(mcore_mapping, "ShardedTensor", None)
    MCoreShardedTensorFactory = getattr(mcore_mapping, "ShardedTensorFactory", None)

    def _materialize_data(value, *, local_shape=None, dtype=None) -> None:
        payload = getattr(value, "data", None)
        if not isinstance(payload, torch.Tensor) or payload.device.type != "meta":
            return

        local_shape = local_shape or getattr(value, "local_shape", None) or tuple(payload.shape)
        dtype = dtype or getattr(value, "dtype", payload.dtype)
        value.data = torch.empty(local_shape, dtype=dtype, device="cpu")

    def _walk(value) -> None:
        if isinstance(value, dict):
            for nested in value.values():
                _walk(nested)
            return
        if isinstance(value, (list, tuple)):
            for nested in value:
                _walk(nested)
            return

        if MCoreShardedTensorFactory is not None and isinstance(value, MCoreShardedTensorFactory):
            payload = getattr(value, "data", None)
            if isinstance(payload, torch.Tensor):
                _materialize_data(value, local_shape=tuple(payload.shape), dtype=payload.dtype)
            return

        if MCoreShardedTensor is None or not isinstance(value, MCoreShardedTensor):
            return

        _materialize_data(value)

    _walk(sharded_state_dict)


def _unwrap_parallel_model(parallel_model):
    from megatron.core.utils import unwrap_model

    unwrapped = unwrap_model(parallel_model)
    if isinstance(unwrapped, list):
        return unwrapped
    return [unwrapped]


def _get_assume_ok_unexpected_strict_handling():
    try:
        from megatron.core.dist_checkpointing.serialization import StrictHandling

        return StrictHandling.ASSUME_OK_UNEXPECTED
    except Exception:
        return None


def load_dist_checkpoint(parallel_model, dist_weight_path: str, *, is_value_model: bool = False, prefix: str = ""):
    """Load a Megatron distributed checkpoint directly into the current model."""
    resolved_dist_dir = resolve_distributed_checkpoint_dir(dist_weight_path)
    if resolved_dist_dir is None:
        raise ValueError(f"{dist_weight_path!r} is not a Megatron distributed checkpoint directory.")
    resolved_dist_path = str(resolved_dist_dir)

    unwrapped_parallel_model = _unwrap_parallel_model(parallel_model)
    if _dist_checkpoint_already_loaded(unwrapped_parallel_model, resolved_dist_path, prefix):
        return

    _restore_modelopt_state_before_load(unwrapped_parallel_model, resolved_dist_path)

    dist_checkpointing = _import_dist_checkpointing()
    checkpoint_tensor_keys = dist_checkpointing.load_tensors_metadata(resolved_dist_path).keys()
    force_explicit_layer_schema = _checkpoint_uses_explicit_layer_keys(checkpoint_tensor_keys)
    strict_handling = _get_assume_ok_unexpected_strict_handling()

    has_expert_int4, has_dense_int4 = detect_int4_checkpoint(resolved_dist_path)
    int4_mode = has_expert_int4 or has_dense_int4
    nvfp4_mode = detect_nvfp4_checkpoint(resolved_dist_path)
    fp8_mode = detect_fp8_checkpoint(resolved_dist_path)

    orig_apply_swiglu = None
    register_int4_buffers_after_load = None
    transform_sharded_state_dict_for_int4 = None
    register_int4_buffers_after_load_dense = None
    transform_sharded_state_dict_for_int4_dense = None
    register_fp8_scale_inv_buffers_after_load = None
    transform_sharded_state_dict_for_fp8 = None

    if fp8_mode:
        from megatron.bridge.orbit.quant.fp8_utils import (
            register_fp8_scale_inv_buffers_after_load,
            transform_sharded_state_dict_for_fp8,
        )

    if int4_mode:
        import megatron.core.transformer.moe.experts as moe_experts

        if has_expert_int4:
            from megatron.bridge.orbit.quant.int4_utils import (
                register_int4_buffers_after_load,
                transform_sharded_state_dict_for_int4,
            )
        if has_dense_int4:
            from megatron.bridge.orbit.low_precision.int4 import (
                register_int4_buffers_after_load_dense,
                transform_sharded_state_dict_for_int4_dense,
            )

        orig_apply_swiglu = moe_experts.apply_swiglu_sharded_factory

        def _safe_apply_swiglu(original_sh_ten, sharded_offsets, singleton_local_shards=False):
            local_shape = getattr(original_sh_ten, "local_shape", None)
            if local_shape is not None and len(local_shape) > 0 and local_shape[0] == 0:
                return original_sh_ten
            return orig_apply_swiglu(original_sh_ten, sharded_offsets, singleton_local_shards)

        moe_experts.apply_swiglu_sharded_factory = _safe_apply_swiglu

    nvfp4_transform_dense = None
    nvfp4_transform_expert = None
    nvfp4_register_dense = None
    nvfp4_register_expert = None

    if nvfp4_mode:
        from megatron.bridge.orbit.low_precision.nvfp4 import (
            register_nvfp4_buffers_after_load_dense,
            transform_sharded_state_dict_for_nvfp4_dense,
        )
        from megatron.bridge.orbit.quant.nvfp4_utils import (
            register_nvfp4_buffers_after_load,
            transform_sharded_state_dict_for_nvfp4,
        )
        nvfp4_transform_dense = transform_sharded_state_dict_for_nvfp4_dense
        nvfp4_transform_expert = transform_sharded_state_dict_for_nvfp4
        nvfp4_register_dense = register_nvfp4_buffers_after_load_dense
        nvfp4_register_expert = register_nvfp4_buffers_after_load

    try:
        for model in unwrapped_parallel_model:
            sharded_state_dict_metadata = {"non_homogeneous_layers": True} if force_explicit_layer_schema else None
            sharded_state_dict = model.sharded_state_dict(prefix=prefix, metadata=sharded_state_dict_metadata)

            if is_value_model:
                for key in list(sharded_state_dict.keys()):
                    if "output_layer" in str(key):
                        sharded_state_dict.pop(key)

            sharded_state_dict, _ = filter_missing_extra_state_keys(sharded_state_dict, checkpoint_tensor_keys)

            missing_oft_adapter_keys = []
            if any(str(key).endswith("adapter.oft_r") for key in sharded_state_dict):
                sharded_state_dict, missing_oft_adapter_keys = filter_missing_oft_adapter_keys(
                    sharded_state_dict,
                    checkpoint_tensor_keys,
                )

            if fp8_mode and transform_sharded_state_dict_for_fp8 is not None:
                sharded_state_dict = transform_sharded_state_dict_for_fp8(sharded_state_dict)
            if has_expert_int4 and transform_sharded_state_dict_for_int4 is not None:
                sharded_state_dict = transform_sharded_state_dict_for_int4(sharded_state_dict)
            if has_dense_int4 and transform_sharded_state_dict_for_int4_dense is not None:
                sharded_state_dict = transform_sharded_state_dict_for_int4_dense(sharded_state_dict)
            if nvfp4_mode and nvfp4_transform_dense is not None:
                sharded_state_dict = nvfp4_transform_dense(sharded_state_dict, checkpoint_tensor_keys)
            if nvfp4_mode and nvfp4_transform_expert is not None:
                sharded_state_dict = nvfp4_transform_expert(sharded_state_dict)
            if int4_mode or fp8_mode or nvfp4_mode:
                _materialize_meta_sharded_tensor_data(sharded_state_dict)

            if strict_handling is None:
                loaded_state_dict = dist_checkpointing.load(sharded_state_dict, resolved_dist_path)
            else:
                loaded_state_dict = dist_checkpointing.load(
                    sharded_state_dict,
                    resolved_dist_path,
                    strict=strict_handling,
                )

            post_load_state_dict = loaded_state_dict if loaded_state_dict is not None else sharded_state_dict
            if fp8_mode and register_fp8_scale_inv_buffers_after_load is not None:
                register_fp8_scale_inv_buffers_after_load(model, post_load_state_dict)
            _materialize_loaded_meta_module_tensors(model, post_load_state_dict, prefix=prefix)
            if has_expert_int4 and register_int4_buffers_after_load is not None:
                register_int4_buffers_after_load(model, post_load_state_dict)
            if has_dense_int4 and register_int4_buffers_after_load_dense is not None:
                register_int4_buffers_after_load_dense(model, post_load_state_dict)
            if nvfp4_mode and nvfp4_register_dense is not None:
                nvfp4_register_dense(model, post_load_state_dict)
                _cleanup_nvfp4_dense_split_meta_parameters(model, post_load_state_dict, prefix=prefix)
            if nvfp4_mode and nvfp4_register_expert is not None:
                nvfp4_register_expert(model, post_load_state_dict)
            _zero_oft_adapter_parameters(model, missing_oft_adapter_keys)
            _raise_if_meta_tensors_remain(model, checkpoint_path=resolved_dist_path, prefix=prefix)
            _mark_dist_checkpoint_as_loaded([model], resolved_dist_path, prefix)
    finally:
        if int4_mode:
            import megatron.core.transformer.moe.experts as moe_experts

            moe_experts.apply_swiglu_sharded_factory = orig_apply_swiglu
