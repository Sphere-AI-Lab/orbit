import json
import os
from pathlib import Path


def _load_adapter_config(adapter_path: str) -> dict:
    config_path = Path(adapter_path) / "adapter_config.json"
    with config_path.open() as f:
        return json.load(f)


def _adapter_type(config: dict, explicit_type: str | None) -> str:
    adapter_type = explicit_type or config.get("peft_type")
    if not adapter_type:
        raise ValueError("adapter_type was not provided and adapter_config.json has no peft_type")
    return str(adapter_type).upper()


def build_sglang_engine_config(args):
    """Build SGLang engine kwargs and per-request adapter kwargs.

    For PEFT adapter eval, SGLang should load the base model and preload the
    adapter directly. This avoids materializing a full merged HF checkpoint for
    every saved training step.
    """
    engine_kwargs = {
        "model_path": args.model_name_or_path,
        "dtype": args.dtype,
        "mem_fraction_static": args.gpu_memory_utilization,
        "tp_size": args.num_gpus,
        "context_length": args.max_model_len,
    }
    attention_backend = os.environ.get("SGLANG_ATTENTION_BACKEND")
    if attention_backend:
        engine_kwargs["attention_backend"] = attention_backend
    generate_kwargs = {}
    tokenizer_path = args.model_name_or_path

    adapter_path = getattr(args, "adapter_path", None)
    if not adapter_path:
        return engine_kwargs, generate_kwargs, tokenizer_path

    config = _load_adapter_config(adapter_path)
    base_model_path = config.get("base_model_name_or_path")
    if not base_model_path:
        raise ValueError(f"{adapter_path}/adapter_config.json missing base_model_name_or_path")

    adapter_name = getattr(args, "adapter_name", None) or "orbit_eval_adapter"
    adapter_type = _adapter_type(config, getattr(args, "adapter_type", None))
    engine_kwargs["model_path"] = base_model_path
    tokenizer_path = base_model_path

    if adapter_type == "LORA":
        engine_kwargs.update(
            {
                "enable_lora": True,
                "lora_paths": {adapter_name: adapter_path},
                "lora_backend": getattr(args, "lora_backend", "triton"),
                "max_loras_per_batch": 1,
            }
        )
        generate_kwargs["lora_path"] = adapter_name
    elif adapter_type == "OFT":
        engine_kwargs.update(
            {
                "peft_method": "oft",
                "peft_paths": {adapter_name: adapter_path},
                "oft_backend": getattr(args, "oft_backend", "triton"),
                "max_ofts_per_batch": 2,
            }
        )
        generate_kwargs["adapter_path"] = adapter_name
    else:
        raise ValueError(f"Unsupported direct SGLang adapter_type={adapter_type!r}")

    return engine_kwargs, generate_kwargs, tokenizer_path
