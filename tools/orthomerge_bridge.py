#!/usr/bin/env python
"""Bridge utilities for comparing original OrthoMerge and Orbit OFT adapter runs."""
from __future__ import annotations

import argparse
import inspect
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_DSV4_GROUPED_MOE_OFT_PARAM_NAMES = frozenset({"w1_oft_r", "w2_oft_r", "w3_oft_r"})


def is_oft_key(name: str) -> bool:
    """Return whether a tensor name is treated as an OFT generator by oft-original."""
    parts = name.lower().replace("/", ".").split(".")
    if any("classifier" in part for part in parts):
        return False
    return any(part == "oft_r" or part in _DSV4_GROUPED_MOE_OFT_PARAM_NAMES for part in parts)


def adapter_weight_file(adapter_dir: str | Path) -> Path:
    """Return the adapter weight file, preferring safetensors over PyTorch .bin."""
    adapter_path = Path(adapter_dir)
    safetensors_path = adapter_path / "adapter_model.safetensors"
    if safetensors_path.exists():
        return safetensors_path
    bin_path = adapter_path / "adapter_model.bin"
    if bin_path.exists():
        return bin_path
    raise FileNotFoundError(f"no adapter_model.safetensors or adapter_model.bin in {adapter_path}")


def _torch_load_weights(weight_path: Path) -> Any:
    try:
        signature = inspect.signature(torch.load)
    except (TypeError, ValueError):
        return torch.load(str(weight_path), map_location="cpu", weights_only=True)
    supports_weights_only = "weights_only" in signature.parameters or any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if supports_weights_only:
        return torch.load(str(weight_path), map_location="cpu", weights_only=True)
    return torch.load(str(weight_path), map_location="cpu")


def load_adapter_state(adapter_dir: str | Path) -> dict[str, torch.Tensor]:
    """Load an adapter state dict from adapter_model.safetensors or adapter_model.bin."""
    weight_path = adapter_weight_file(adapter_dir)
    if weight_path.suffix == ".safetensors":
        state = load_file(str(weight_path))
    else:
        state = _torch_load_weights(weight_path)
        if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"adapter weights at {weight_path} did not contain a state dict")
    non_tensor_keys = [key for key, value in state.items() if not isinstance(value, torch.Tensor)]
    if non_tensor_keys:
        raise TypeError(f"adapter weights at {weight_path} contain non-tensor keys: {non_tensor_keys[:5]}")
    return dict(state)


def _read_adapter_config(adapter_dir: str | Path) -> dict[str, Any]:
    config_path = Path(adapter_dir) / "adapter_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text())


def _json_safe_float(value: float) -> float | str:
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def _json_metric(value: float) -> tuple[float | str, bool]:
    finite = math.isfinite(value)
    return _json_safe_float(value), finite


def _json_safe(data: Any) -> Any:
    if isinstance(data, float):
        return _json_safe_float(data)
    if isinstance(data, dict):
        return {key: _json_safe(value) for key, value in data.items()}
    if isinstance(data, (list, tuple)):
        return [_json_safe(value) for value in data]
    return data


def _json_text(data: Any) -> str:
    return json.dumps(_json_safe(data), indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(data: Any, output: str | Path | None = None) -> None:
    text = _json_text(data)
    if output is None:
        print(text, end="")
    else:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text)


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    data = tensor.detach().cpu()
    finite = bool(torch.isfinite(data).all().item())
    if data.numel() == 0:
        max_abs = 0.0
        fro_norm = 0.0
    else:
        as_float = data.float()
        max_abs = float(as_float.abs().max().item())
        fro_norm = float(torch.linalg.vector_norm(as_float.reshape(-1), ord=2).item())
    max_abs_value, max_abs_finite = _json_metric(max_abs)
    fro_norm_value, fro_norm_finite = _json_metric(fro_norm)
    return {
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "all_finite": finite,
        "max_abs": max_abs_value,
        "max_abs_finite": max_abs_finite,
        "fro_norm": fro_norm_value,
        "fro_norm_finite": fro_norm_finite,
    }


def summarize_adapter(adapter_dir: str | Path) -> dict[str, Any]:
    """Summarize tensor counts and numeric properties for one adapter directory."""
    adapter_path = Path(adapter_dir)
    state = load_adapter_state(adapter_path)
    tensor_summaries = {}
    all_finite = True
    num_oft = 0
    for key in sorted(state):
        is_oft = is_oft_key(key)
        summary = _tensor_summary(state[key])
        summary["is_oft"] = is_oft
        tensor_summaries[key] = summary
        all_finite = all_finite and summary["all_finite"]
        num_oft += int(is_oft)
    return {
        "path": str(adapter_path),
        "weight_file": str(adapter_weight_file(adapter_path)),
        "num_tensors": len(state),
        "num_oft_tensors": num_oft,
        "num_non_oft_tensors": len(state) - num_oft,
        "all_finite": all_finite,
        "tensors": tensor_summaries,
    }


def _tensor_diff(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float]:
    ref = reference.detach().cpu().float()
    cand = candidate.detach().cpu().float()
    diff = ref - cand
    if diff.numel() == 0:
        return 0.0, 0.0
    max_abs_diff = float(diff.abs().max().item())
    diff_norm = float(torch.linalg.vector_norm(diff.reshape(-1), ord=2).item())
    ref_norm = float(torch.linalg.vector_norm(ref.reshape(-1), ord=2).item())
    if ref_norm == 0.0:
        relative_frobenius = 0.0 if diff_norm == 0.0 else math.inf
    else:
        relative_frobenius = diff_norm / ref_norm
    return max_abs_diff, relative_frobenius


def _merge_global_max_abs_diff(current: float, candidate: float) -> float:
    if math.isnan(current) or math.isnan(candidate):
        return math.nan
    if math.isinf(current) or math.isinf(candidate):
        return math.inf
    return max(current, candidate)


def compare_adapters(reference_dir: str | Path, candidate_dir: str | Path) -> dict[str, Any]:
    """Compare two adapter state dicts and report key, shape, dtype, and numeric deltas."""
    reference_path = Path(reference_dir)
    candidate_path = Path(candidate_dir)
    reference = load_adapter_state(reference_path)
    candidate = load_adapter_state(candidate_path)
    reference_keys = set(reference)
    candidate_keys = set(candidate)
    missing = sorted(reference_keys - candidate_keys)
    extra = sorted(candidate_keys - reference_keys)
    reports: dict[str, Any] = {}
    global_max_abs_diff = 0.0
    num_different = len(missing) + len(extra)

    for key in missing:
        ref_summary = _tensor_summary(reference[key])
        reports[key] = {
            "status": "missing",
            "is_oft": is_oft_key(key),
            "reference_shape": ref_summary["shape"],
            "candidate_shape": None,
            "reference_dtype": ref_summary["dtype"],
            "candidate_dtype": None,
            "same_shape": False,
            "same_dtype": False,
            "max_abs_diff": None,
            "max_abs_diff_finite": None,
            "relative_frobenius": None,
            "relative_frobenius_finite": None,
        }
    for key in extra:
        cand_summary = _tensor_summary(candidate[key])
        reports[key] = {
            "status": "extra",
            "is_oft": is_oft_key(key),
            "reference_shape": None,
            "candidate_shape": cand_summary["shape"],
            "reference_dtype": None,
            "candidate_dtype": cand_summary["dtype"],
            "same_shape": False,
            "same_dtype": False,
            "max_abs_diff": None,
            "max_abs_diff_finite": None,
            "relative_frobenius": None,
            "relative_frobenius_finite": None,
        }

    for key in sorted(reference_keys & candidate_keys):
        ref_tensor = reference[key]
        cand_tensor = candidate[key]
        ref_summary = _tensor_summary(ref_tensor)
        cand_summary = _tensor_summary(cand_tensor)
        same_shape = tuple(ref_tensor.shape) == tuple(cand_tensor.shape)
        same_dtype = ref_tensor.dtype == cand_tensor.dtype
        if same_shape:
            max_abs_diff, relative_frobenius = _tensor_diff(ref_tensor, cand_tensor)
            global_max_abs_diff = _merge_global_max_abs_diff(global_max_abs_diff, max_abs_diff)
        else:
            max_abs_diff = None
            relative_frobenius = None
        if max_abs_diff is None:
            max_abs_diff_value = None
            max_abs_diff_finite = None
        else:
            max_abs_diff_value, max_abs_diff_finite = _json_metric(max_abs_diff)
        if relative_frobenius is None:
            relative_frobenius_value = None
            relative_frobenius_finite = None
        else:
            relative_frobenius_value, relative_frobenius_finite = _json_metric(relative_frobenius)
        differs = (not same_shape) or (not same_dtype) or (max_abs_diff is not None and max_abs_diff != 0.0)
        num_different += int(differs)
        reports[key] = {
            "status": "compared",
            "is_oft": is_oft_key(key),
            "reference_shape": ref_summary["shape"],
            "candidate_shape": cand_summary["shape"],
            "reference_dtype": ref_summary["dtype"],
            "candidate_dtype": cand_summary["dtype"],
            "same_shape": same_shape,
            "same_dtype": same_dtype,
            "max_abs_diff": max_abs_diff_value,
            "max_abs_diff_finite": max_abs_diff_finite,
            "relative_frobenius": relative_frobenius_value,
            "relative_frobenius_finite": relative_frobenius_finite,
        }

    global_max_abs_diff_value, global_max_abs_diff_finite = _json_metric(global_max_abs_diff)
    return {
        "reference_path": str(reference_path),
        "candidate_path": str(candidate_path),
        "same_keys": not missing and not extra,
        "missing_keys": missing,
        "extra_keys": extra,
        "global_max_abs_diff": global_max_abs_diff_value,
        "global_max_abs_diff_finite": global_max_abs_diff_finite,
        "num_different_tensors": num_different,
        "tensors": reports,
    }


def write_manifest(
    adapter_dirs: list[str | Path],
    output: str | Path,
    base_model: str | None = None,
) -> dict[str, Any]:
    """Write a deterministic manifest sorted by adapter directory name."""
    adapters = sorted((Path(path) for path in adapter_dirs), key=lambda path: (path.name, str(path)))
    entries = []
    config_bases = []
    for adapter_path in adapters:
        config = _read_adapter_config(adapter_path)
        config_base = config.get("base_model_name_or_path")
        if config_base:
            config_bases.append(config_base)
        summary = summarize_adapter(adapter_path)
        entries.append({
            "name": adapter_path.name,
            "path": str(adapter_path),
            "weight_file": summary["weight_file"],
            "num_tensors": summary["num_tensors"],
            "num_oft_tensors": summary["num_oft_tensors"],
            "oft_block_size": config.get("oft_block_size"),
        })

    inferred_base_model = base_model
    unique_config_bases = sorted(set(config_bases))
    if inferred_base_model is None and len(unique_config_bases) == 1:
        inferred_base_model = unique_config_bases[0]

    manifest = {
        "base_model": inferred_base_model,
        "adapters": entries,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_json_text(manifest))
    return manifest


def build_reference_command(
    original_repo: str | Path,
    base_model: str,
    adapters: list[str | Path],
    output_dir: str | Path,
    gpu: int,
) -> list[str]:
    """Build the original OrthoMerge adapter-only command."""
    script = Path(original_repo) / "merge" / "OrthoMerge_OFT_models.py"
    return [
        sys.executable,
        str(script),
        "--language_model_name",
        str(base_model),
        "--adapter_paths",
        *(str(path) for path in adapters),
        "--output_merged_adapter_dir",
        str(output_dir),
        "--gpu",
        str(gpu),
        "--just_merge_adapter",
    ]


def build_orbit_command(
    adapters: list[str | Path],
    output_dir: str | Path,
    method: str = "oft-original",
) -> list[str]:
    """Build the local Orbit adapter merge command."""
    script = REPO_ROOT / "tools" / "merge_oft_adapters.py"
    return [
        sys.executable,
        str(script),
        "--method",
        method,
        "--adapters",
        *(str(path) for path in adapters),
        "--output",
        str(output_dir),
    ]


def _load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _select_adapter_paths(manifest: dict[str, Any], count: str) -> list[Path]:
    adapters = [Path(item["path"]) for item in manifest.get("adapters", [])]
    if count == "all":
        selected = adapters
    else:
        selected = adapters[:int(count)]
    if len(selected) < 2:
        raise ValueError(f"selected {len(selected)} adapter(s); need at least 2")
    if count != "all" and len(selected) != int(count):
        raise ValueError(f"manifest has {len(adapters)} adapter(s), cannot select {count}")
    return selected


def _preflight_reference_adapters(adapter_paths: list[Path]) -> None:
    for adapter_path in adapter_paths:
        config = _read_adapter_config(adapter_path)
        block_size = config.get("oft_block_size")
        if block_size != 32:
            raise ValueError(
                "run-reference requires adapters compatible with the original script's "
                f"hardcoded block_size=32; {adapter_path} has oft_block_size={block_size!r}"
            )


def _run_command(cmd: list[str]) -> int:
    print(shlex.join(cmd), flush=True)
    return subprocess.run(cmd, check=False).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bridge OrthoMerge reference runs and Orbit OFT merges.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser("manifest", help="write a deterministic adapter manifest")
    manifest_parser.add_argument("--output", required=True, help="manifest JSON path")
    manifest_parser.add_argument("--base-model", default=None, help="optional base model override")
    manifest_parser.add_argument("adapters", nargs="+", help="adapter directories")

    summarize_parser = subparsers.add_parser("summarize", help="summarize one or more adapters")
    summarize_parser.add_argument("adapters", nargs="+", help="adapter directories")
    summarize_parser.add_argument("--output", default=None, help="optional JSON output path")

    compare_parser = subparsers.add_parser("compare", help="compare two adapter directories")
    compare_parser.add_argument("reference", help="reference adapter directory")
    compare_parser.add_argument("candidate", help="candidate adapter directory")
    compare_parser.add_argument("--output", default=None, help="optional JSON output path")

    reference_parser = subparsers.add_parser("run-reference", help="run original OrthoMerge adapter merge")
    reference_parser.add_argument("--manifest", required=True, help="manifest JSON path")
    reference_parser.add_argument("--original-repo", required=True, help="path to original OrthoMerge repo")
    reference_parser.add_argument("--output", required=True, help="output directory")
    reference_parser.add_argument("--gpu", type=int, default=0, help="GPU id for original script")
    reference_parser.add_argument("--count", choices=("2", "3", "all"), default="all", help="adapter count to merge")

    orbit_parser = subparsers.add_parser("run-orbit", help="run local Orbit OFT adapter merge")
    orbit_parser.add_argument("--manifest", required=True, help="manifest JSON path")
    orbit_parser.add_argument("--output", required=True, help="output directory")
    orbit_parser.add_argument("--method", default="oft-original", help="Orbit merge method")
    orbit_parser.add_argument("--count", choices=("2", "3", "all"), default="all", help="adapter count to merge")

    args = parser.parse_args(argv)

    if args.command == "manifest":
        result = write_manifest(args.adapters, args.output, base_model=args.base_model)
        _write_json(result)
        return 0
    if args.command == "summarize":
        summaries = [summarize_adapter(adapter) for adapter in args.adapters]
        _write_json(summaries[0] if len(summaries) == 1 else summaries, args.output)
        return 0
    if args.command == "compare":
        _write_json(compare_adapters(args.reference, args.candidate), args.output)
        return 0
    if args.command == "run-reference":
        manifest = _load_manifest(args.manifest)
        base_model = manifest.get("base_model")
        if not base_model:
            raise ValueError("run-reference requires manifest base_model")
        adapters = _select_adapter_paths(manifest, args.count)
        _preflight_reference_adapters(adapters)
        cmd = build_reference_command(args.original_repo, base_model, adapters, args.output, args.gpu)
        return _run_command(cmd)
    if args.command == "run-orbit":
        manifest = _load_manifest(args.manifest)
        adapters = _select_adapter_paths(manifest, args.count)
        cmd = build_orbit_command(adapters, args.output, method=args.method)
        return _run_command(cmd)

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
