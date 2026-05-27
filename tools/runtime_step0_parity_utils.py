from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import tempfile
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


TOPK = 8
SGLANG_MEGATRON_WEIGHT_COMPARE_DEBUG_ENV = "ORBIT_RUNTIME_PARITY_WEIGHT_COMPARE"
SGLANG_MEGATRON_WEIGHT_COMPARE_LAYERS_ENV = "ORBIT_RUNTIME_PARITY_WEIGHT_COMPARE_LAYERS"
SGLANG_MEGATRON_WEIGHT_COMPARE_TRUNCATE_SIZE_ENV = "ORBIT_RUNTIME_PARITY_WEIGHT_COMPARE_TRUNCATE_SIZE"
SGLANG_MEGATRON_WEIGHT_COMPARE_INCLUDE_SCALES_ENV = "ORBIT_RUNTIME_PARITY_WEIGHT_COMPARE_INCLUDE_SCALES"
SGLANG_MEGATRON_ACTIVATION_COMPARE_DEBUG_ENV = "ORBIT_RUNTIME_PARITY_ACTIVATION_COMPARE"
SGLANG_MEGATRON_ACTIVATION_COMPARE_INTERNAL_ENV = "ORBIT_RUNTIME_PARITY_ACTIVATION_COMPARE_INTERNAL"
SGLANG_MEGATRON_ACTIVATION_COMPARE_LAYERS_ENV = "ORBIT_RUNTIME_PARITY_ACTIVATION_COMPARE_LAYERS"
SGLANG_TENSOR_DUMP_DIR_ENV = "ORBIT_RUNTIME_PARITY_TENSOR_DUMP_DIR"
SGLANG_HTTP_LOG_LEVEL_ENV = "VERL_SGLANG_HTTP_LOG_LEVEL"
SGLANG_ATTENTION_BACKEND_ENV = "SGLANG_ATTENTION_BACKEND"
VERL_SGLANG_ATTENTION_BACKEND_ENV = "VERL_SGLANG_ATTENTION_BACKEND"
SGLANG_FP8_GEMM_BACKEND_ENV = "SGLANG_FP8_GEMM_BACKEND"
VERL_SGLANG_FP8_GEMM_BACKEND_ENV = "VERL_SGLANG_FP8_GEMM_BACKEND"
SGLANG_FP4_GEMM_BACKEND_ENV = "SGLANG_FP4_GEMM_BACKEND"
VERL_SGLANG_FP4_GEMM_BACKEND_ENV = "VERL_SGLANG_FP4_GEMM_BACKEND"
SGLANG_LEGACY_FP4_GEMM_BACKEND_ENV = "SGLANG_FLASHINFER_FP4_GEMM_BACKEND"
MEGATRON_OFT_FP8_ACTIVATION_QUANT_ENV = "MEGATRON_OFT_FP8_ACTIVATION_QUANT"

PRECISION_PROFILE_DEFAULTS = {
    "fp8": {
        "rollout_quantization": "fp8",
        "restore_modelopt_state": True,
    },
    "nvfp4": {
        "rollout_quantization": "modelopt_fp4",
        "restore_modelopt_state": True,
    },
    "int4": {
        "rollout_quantization": None,
        "restore_modelopt_state": False,
    },
}

ROLLOUT_QUANTIZATION_ALIASES = {
    "fp4": "modelopt_fp4",
    "nvfp4": "modelopt_fp4",
    "modelopt_nvfp4": "modelopt_fp4",
    "int4": None,
}

FP8_MODE_CHOICES = ("w8a8", "w8a16")
FP8_W8A16_TORCHAO_CONFIG = "fp8wo"
FP8_W8A8_PARITY_MODE_CHOICES = ("activation_qdq", "block_linear")


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


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from checkpoint_parity_core import (  # noqa: E402
    RuntimeParityReport,
    RuntimeParityThresholds,
    resolve_megatron_dist_checkpoint_dir,
    write_report_json,
)


def _parse_rollout_quantization(value: str) -> str | None:
    normalized = value.strip().lower()
    if normalized in {"", "0", "false", "none", "null", "off", "disabled"}:
        return None
    return ROLLOUT_QUANTIZATION_ALIASES.get(normalized, normalized)


def _parse_precision_profile(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in PRECISION_PROFILE_DEFAULTS:
        supported = ", ".join(sorted(PRECISION_PROFILE_DEFAULTS))
        raise argparse.ArgumentTypeError(
            f"Unsupported precision profile {value!r}; expected one of: {supported}."
        )
    return normalized


def _resolve_precision_profile_settings(
    *,
    precision_profile: str,
    rollout_quantization: str | None,
    restore_modelopt_state: bool | None,
) -> tuple[str, str | None, bool]:
    normalized_profile = _parse_precision_profile(precision_profile)
    profile_defaults = PRECISION_PROFILE_DEFAULTS[normalized_profile]

    if rollout_quantization is None:
        resolved_rollout_quantization = profile_defaults["rollout_quantization"]
    else:
        resolved_rollout_quantization = _parse_rollout_quantization(rollout_quantization)

    if restore_modelopt_state is None:
        resolved_restore_modelopt_state = bool(profile_defaults["restore_modelopt_state"])
    else:
        resolved_restore_modelopt_state = restore_modelopt_state

    return normalized_profile, resolved_rollout_quantization, resolved_restore_modelopt_state


def _parse_megatron_activation_fp8(value: str) -> str | None:
    normalized = value.strip().lower()
    if normalized in {"auto", "config", "default"}:
        return "auto"
    if normalized in {"", "0", "false", "none", "null", "off", "disabled"}:
        return None
    if normalized in {"e4m3", "hybrid"}:
        return normalized
    raise argparse.ArgumentTypeError(
        f"Unsupported Megatron activation FP8 mode {value!r}; expected none, e4m3, hybrid, or auto."
    )


def _parse_megatron_oft_fp8_activation_quant(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"auto", "w8a8", "none"}:
        return normalized
    if normalized in {"", "0", "false", "off", "disabled"}:
        return "none"
    raise argparse.ArgumentTypeError(
        f"Unsupported Megatron OFT FP8 activation quant mode {value!r}; expected auto, none, or w8a8."
    )


def _none_if_blank(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def parse_args(
    default_precision_profile: str = "fp8",
    default_fp8_mode: str = "w8a8",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check step-0 runtime parity between SGLang rollout and Megatron actor")
    parser.add_argument("--hf-model", required=True)
    parser.add_argument("--megatron-dist", required=True)
    parser.add_argument("--prompt-file", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument(
        "--precision-profile",
        type=_parse_precision_profile,
        default=default_precision_profile,
        help="Runtime precision profile. Supported values: fp8, nvfp4, int4.",
    )
    parser.add_argument(
        "--fp8-mode",
        choices=FP8_MODE_CHOICES,
        default=default_fp8_mode,
        help=(
            "FP8 quantization mode for the SGLang rollout when --precision-profile=fp8. "
            "'w8a8' (block-FP8 weights + dynamic e4m3 activations, the SGLang FP8 default) "
            "or 'w8a16' (torchao fp8wo weight-only FP8 with bfloat16 activations, for high-precision "
            "source weights rather than already-serialized FP8 checkpoints). "
            "Selecting 'w8a16' overrides the rollout-quantization/torchao defaults to "
            f"rollout_quantization=none and sglang_torchao_config={FP8_W8A16_TORCHAO_CONFIG} "
            "unless those flags are explicitly set."
        ),
    )
    parser.add_argument(
        "--rollout-quantization",
        default=None,
        help=(
            "SGLang quantization mode. Defaults to the selected precision profile. "
            'Aliases: "nvfp4" -> "modelopt_fp4", "int4" -> none.'
        ),
    )
    parser.add_argument(
        "--sglang-torchao-config",
        default=None,
        help=(
            "Optional SGLang torchao_config override. Use with --rollout-quantization none for "
            "weight-only FP8 rollout modes such as fp8wo."
        ),
    )
    parser.add_argument(
        "--sglang-fp8-gemm-backend",
        default=None,
        help=(
            "Optional SGLang --fp8-gemm-backend override for block-FP8 rollout diagnostics "
            "(for example: triton, cutlass, deep_gemm, flashinfer_trtllm)."
        ),
    )
    parser.add_argument(
        "--megatron-activation-fp8",
        type=_parse_megatron_activation_fp8,
        default="none",
        help=(
            "Megatron actor TE activation FP8 mode. Default 'none' keeps activations high precision while "
            "still restoring low-precision/ModelOpt checkpoint weights. Use 'e4m3' or 'hybrid' to exercise TE FP8 "
            "activation kernels, or 'auto' to keep the provider config unchanged."
        ),
    )
    parser.add_argument(
        "--megatron-oft-fp8-activation-quant",
        type=_parse_megatron_oft_fp8_activation_quant,
        default="auto",
        help=(
            "OFTLinear activation quantization for native FP8 base weights. 'auto' selects w8a8 for "
            "--fp8-mode w8a8 and none otherwise. This is separate from TE --megatron-activation-fp8."
        ),
    )
    parser.add_argument(
        "--megatron-fp8-w8a8-parity-mode",
        choices=FP8_W8A8_PARITY_MODE_CHOICES,
        default="activation_qdq",
        help=(
            "Megatron-side diagnostic used only when --megatron-oft-fp8-activation-quant=w8a8. "
            "'activation_qdq' applies dynamic FP8 activation QDQ before the normal dense parity path; "
            "'block_linear' replaces FP8 linears with a SGLang-style block-scaled FP8 matmul approximation."
        ),
    )
    parser.add_argument(
        "--restore-modelopt-state",
        dest="restore_modelopt_state",
        action="store_true",
        default=None,
        help="Restore ModelOpt sidecar state before loading the Megatron checkpoint.",
    )
    parser.add_argument(
        "--no-restore-modelopt-state",
        dest="restore_modelopt_state",
        action="store_false",
        help="Disable ModelOpt sidecar restore for BF16/non-ModelOpt checkpoints.",
    )
    parser.add_argument(
        "--dequant-fp8-weights-for-parity",
        action="store_true",
        help=(
            "Temporarily dequantize native packed FP8 Megatron weights with weight_scale_inv during "
            "this parity check only. This is a diagnostic path for checkpoints that store raw FP8 "
            "codes in BF16 weight tensors plus sibling scale tensors."
        ),
    )
    parser.add_argument(
        "--dequant-nvfp4-weights-for-parity",
        action="store_true",
        help=(
            "Temporarily dequantize ModelOpt NVFP4 Megatron weights into dense tensors during this parity "
            "check only. This diagnostic path avoids TensorRT-LLM real-quant GEMM requirements."
        ),
    )
    parser.add_argument(
        "--dequant-int4-weights-for-parity",
        dest="dequant_int4_weights_for_parity",
        action="store_true",
        default=None,
        help=(
            "Temporarily dequantize dense INT4 Megatron weights into dense tensors during this parity check only. "
            "Defaults to enabled for the int4 precision profile."
        ),
    )
    parser.add_argument(
        "--no-dequant-int4-weights-for-parity",
        dest="dequant_int4_weights_for_parity",
        action="store_false",
        help="Disable the parity-only dense INT4 dequantization hooks.",
    )
    parser.add_argument("--max-logprob-abs-diff", type=float, default=0.02)
    parser.add_argument("--min-topk-agreement", type=float, default=0.95)
    parser.add_argument("--report-json", default=None)
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=int(os.getenv("ORBIT_RUNTIME_PARITY_MAX_PROMPTS", "0")),
        help="Only run the first N prompts from --prompt-file. 0 means all prompts.",
    )
    args = parser.parse_args()
    args.sglang_torchao_config = _none_if_blank(args.sglang_torchao_config)
    args.sglang_fp8_gemm_backend = _none_if_blank(args.sglang_fp8_gemm_backend)
    if args.precision_profile == "fp8" and args.fp8_mode == "w8a16":
        if args.rollout_quantization is None:
            args.rollout_quantization = "none"
        if args.sglang_torchao_config is None:
            args.sglang_torchao_config = FP8_W8A16_TORCHAO_CONFIG
    (
        args.precision_profile,
        args.rollout_quantization,
        args.restore_modelopt_state,
    ) = _resolve_precision_profile_settings(
        precision_profile=args.precision_profile,
        rollout_quantization=args.rollout_quantization,
        restore_modelopt_state=args.restore_modelopt_state,
    )
    if args.megatron_oft_fp8_activation_quant == "auto":
        if args.precision_profile == "fp8" and args.fp8_mode == "w8a8":
            args.megatron_oft_fp8_activation_quant = "w8a8"
        else:
            args.megatron_oft_fp8_activation_quant = "none"
    if args.dequant_int4_weights_for_parity is None:
        args.dequant_int4_weights_for_parity = args.precision_profile == "int4"
    return args


def load_prompts_jsonl(path: str) -> list[str]:
    return [json.loads(line)["prompt"] for line in Path(path).read_text().splitlines() if line.strip()]


def _reserve_local_port() -> int:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


class HttpServerAdapter:
    """Small local SGLang HTTP adapter used by the runtime parity CLI."""

    def __init__(self, *, first_rank_in_node: bool = False, **kwargs: Any) -> None:
        from sglang.srt.server_args import ServerArgs

        from orbit.backends.sglang_utils.sglang_engine import launch_server_process

        self.server_args = ServerArgs(**kwargs)
        self.node_rank = self.server_args.node_rank
        self.process = launch_server_process(self.server_args) if first_rank_in_node else None

    def _make_request(self, endpoint: str, payload: dict[str, Any] | None = None, method: str = "POST") -> Any:
        import requests

        url = f"http://{self.server_args.host}:{self.server_args.port}/{endpoint}"
        if method.upper() == "GET":
            response = requests.get(url, timeout=60)
        else:
            response = requests.post(url, json=payload or {}, timeout=60)
        response.raise_for_status()
        if not response.content:
            return {}
        return response.json()

    def generate(
        self,
        *,
        prompt: str,
        sampling_params: dict[str, Any],
        return_logprob: bool,
        top_logprobs_num: int,
    ) -> dict[str, Any]:
        return self._make_request(
            "generate",
            {
                "text": prompt,
                "sampling_params": sampling_params,
                "return_logprob": return_logprob,
                "top_logprobs_num": top_logprobs_num,
            },
        )

    def shutdown(self) -> None:
        if self.process is None:
            return
        from sglang.srt.utils import kill_process_tree

        kill_process_tree(self.process.pid)


def _build_actor_override_transformer_config(
    *, restore_modelopt_state: bool, megatron_activation_fp8: str | None
) -> dict[str, object]:
    override_transformer_config: dict[str, object] = {
        "restore_modelopt_state": restore_modelopt_state,
        # ModelOpt GPT specs default to arbitrary masks when CP=1, but the THD
        # parity path relies on causal/padding_causal semantics in TE.
        "use_arbitrary_attention_mask": False,
    }
    if megatron_activation_fp8 != "auto":
        override_transformer_config["fp8"] = megatron_activation_fp8
    return override_transformer_config


def _tokenize_prompts(tokenizer_path: str, prompts: list[str]) -> list[list[int]]:
    from orbit.utils.processing_utils import load_tokenizer

    tokenizer = load_tokenizer(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return [_normalize_tokenized_output(tokenizer(prompt)) for prompt in prompts]


def _normalize_tokenized_output(tokenized_output: Any) -> list[int]:
    token_ids = tokenized_output
    if isinstance(tokenized_output, dict) and "input_ids" in tokenized_output:
        token_ids = tokenized_output["input_ids"]
    elif hasattr(tokenized_output, "input_ids"):
        token_ids = tokenized_output.input_ids

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)
    if isinstance(token_ids, list) and len(token_ids) == 1 and isinstance(token_ids[0], list | tuple):
        token_ids = list(token_ids[0])
    if not isinstance(token_ids, list):
        raise TypeError(f"token_ids must be list-like token ids, got {type(token_ids).__name__}: {token_ids!r}")
    return [int(token_id.item() if hasattr(token_id, "item") else token_id) for token_id in token_ids]


def _looks_like_tokenizer_dir(path: Path) -> bool:
    return path.is_dir() and any((path / name).exists() for name in ("tokenizer.json", "tokenizer_config.json"))


def _find_megatron_tokenizer_dir(megatron_dist: str) -> Path | None:
    checkpoint_dir = Path(megatron_dist)
    candidate_roots = [checkpoint_dir]
    if checkpoint_dir.name.startswith("iter_") and checkpoint_dir.parent not in candidate_roots:
        candidate_roots.append(checkpoint_dir.parent)

    candidates: list[Path] = []
    for candidate_root in candidate_roots:
        candidates.append(candidate_root / "tokenizer")
        candidates.extend(sorted(candidate_root.glob("iter_*/tokenizer")))
    for candidate in candidates:
        if _looks_like_tokenizer_dir(candidate):
            return candidate
    return None


def load_rollout_prompt_token_ids(hf_model: str, prompts: list[str]) -> list[list[int]]:
    return _tokenize_prompts(hf_model, prompts)


def load_actor_prompt_token_ids(*, hf_model: str, megatron_dist: str, prompts: list[str]) -> list[list[int]]:
    resolved_megatron_dist = resolve_megatron_dist_checkpoint_dir(megatron_dist)
    tokenizer_path = _find_megatron_tokenizer_dir(resolved_megatron_dist)
    # The rollout HTTP path does not expose prompt token IDs directly, so we compare the strongest
    # real provenance available from each side: HF model tokenizer for rollout, bundled dist-checkpoint
    # tokenizer for the Megatron actor when present, otherwise the HF tokenizer as fallback.
    actor_tokenizer_path = str(tokenizer_path) if tokenizer_path is not None else hf_model
    return _tokenize_prompts(actor_tokenizer_path, prompts)


def _coerce_int(value: Any) -> int:
    if isinstance(value, list | tuple):
        if not value:
            raise ValueError("Expected a non-empty token-id payload")
        value = value[0]
    return int(value)


def _coerce_float(value: Any) -> float:
    if isinstance(value, list | tuple):
        if not value:
            raise ValueError("Expected a non-empty score payload")
        value = value[0]
    return float(value)


def _looks_like_integral_token_id(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    if isinstance(value, str):
        try:
            int(value)
            return True
        except (TypeError, ValueError):
            return False
    return False


def _parse_topk_payload(
    payload: Any,
    *,
    fallback_token: int,
    fallback_logprob: float,
    topk: int,
) -> tuple[list[int], list[float]]:
    tokens: list[int] = []
    logprobs: list[float] = []

    if isinstance(payload, dict):
        for key, value in payload.items():
            try:
                tokens.append(int(key))
                logprobs.append(float(value))
            except (TypeError, ValueError):
                continue
    elif isinstance(payload, list | tuple):
        for item in payload:
            if isinstance(item, dict):
                token_value = None
                for token_key in ("token_id", "token_ids", "token", "id"):
                    if token_key in item:
                        token_value = item[token_key]
                        break
                score_value = None
                for score_key in ("logprob", "log_prob", "score", "value"):
                    if score_key in item:
                        score_value = item[score_key]
                        break
                if token_value is None or score_value is None:
                    continue
                tokens.append(_coerce_int(token_value))
                logprobs.append(_coerce_float(score_value))
            elif isinstance(item, list | tuple) and len(item) >= 2:
                first, second = item[0], item[1]
                if _looks_like_integral_token_id(first):
                    tokens.append(int(first))
                    logprobs.append(float(second))
                elif _looks_like_integral_token_id(second):
                    tokens.append(int(second))
                    logprobs.append(float(first))
                else:
                    try:
                        tokens.append(int(first))
                        logprobs.append(float(second))
                    except (TypeError, ValueError):
                        continue

    if fallback_token in tokens:
        chosen_index = tokens.index(fallback_token)
        chosen_token = tokens.pop(chosen_index)
        chosen_logprob = logprobs.pop(chosen_index)
        tokens.insert(0, chosen_token)
        logprobs.insert(0, chosen_logprob)
    else:
        tokens.insert(0, fallback_token)
        logprobs.insert(0, fallback_logprob)

    if not tokens:
        tokens = [fallback_token]
        logprobs = [fallback_logprob]

    while len(tokens) < topk:
        tokens.append(tokens[-1])
        logprobs.append(logprobs[-1])

    return tokens[:topk], logprobs[:topk]


def _extract_sglang_step0(
    output: dict[str, Any],
    *,
    topk: int,
) -> tuple[int, float, list[int], list[float]]:
    meta_info = output.get("meta_info", {})
    if not isinstance(meta_info, dict):
        raise ValueError(f"Unexpected SGLang response meta_info payload: {meta_info!r}")

    output_ids = output.get("output_ids")
    if not isinstance(output_ids, list) or not output_ids:
        raise ValueError(f"SGLang response did not contain output_ids: {output!r}")
    chosen_token = _coerce_int(output_ids[0])

    entries = meta_info.get("output_token_logprobs")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"SGLang response did not contain output_token_logprobs: {output!r}")
    entry = entries[0]

    topk_payload = meta_info.get("output_top_logprobs")
    if isinstance(topk_payload, list) and topk_payload:
        topk_payload = topk_payload[0]

    if isinstance(entry, dict):
        chosen_logprob = _coerce_float(entry.get("logprob", entry.get("log_prob")))
        chosen_token = _coerce_int(entry.get("token_id", entry.get("token_ids", chosen_token)))
        topk_payload = entry.get("top_logprobs", entry.get("topk", topk_payload))
    elif isinstance(entry, list | tuple) and len(entry) >= 2:
        chosen_logprob = _coerce_float(entry[0])
        chosen_token = _coerce_int(entry[1])
        if len(entry) >= 3 and entry[2] is not None:
            topk_payload = entry[2]
    else:
        raise ValueError(f"Unexpected SGLang logprob entry: {entry!r}")

    topk_tokens, topk_logprobs = _parse_topk_payload(
        topk_payload,
        fallback_token=chosen_token,
        fallback_logprob=chosen_logprob,
        topk=topk,
    )
    return chosen_token, chosen_logprob, topk_tokens, topk_logprobs


def _stack_int_rows(rows: list[list[int]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.long)


def _stack_float_rows(rows: list[list[float]]) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


def _truncate_debug_repr(value: Any, *, limit: int = 320) -> str:
    text = repr(value)
    if len(text) <= limit:
        return text
    return f"{text[: limit - 3]}..."


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _get_device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _sglang_megatron_weight_compare_enabled() -> bool:
    return _env_flag(SGLANG_MEGATRON_WEIGHT_COMPARE_DEBUG_ENV)


def _sglang_megatron_weight_compare_include_scales() -> bool:
    return _env_flag(SGLANG_MEGATRON_WEIGHT_COMPARE_INCLUDE_SCALES_ENV)


def _sglang_megatron_activation_compare_enabled() -> bool:
    return _env_flag(SGLANG_MEGATRON_ACTIVATION_COMPARE_DEBUG_ENV)


def _sglang_megatron_activation_compare_internal_enabled() -> bool:
    return _env_flag(SGLANG_MEGATRON_ACTIVATION_COMPARE_INTERNAL_ENV)


def _sglang_megatron_weight_compare_truncate_size() -> int:
    raw = os.getenv(SGLANG_MEGATRON_WEIGHT_COMPARE_TRUNCATE_SIZE_ENV, "4")
    try:
        value = int(raw)
    except ValueError:
        value = 4
    return max(1, value)


def _read_hf_config_num_hidden_layers(model_path: str) -> int:
    try:
        config = json.loads((Path(model_path) / "config.json").read_text())
    except Exception:
        return 1
    try:
        return max(1, int(config.get("num_hidden_layers", 1)))
    except (TypeError, ValueError):
        return 1


def _read_hf_config_dict(model_path: str) -> dict[str, Any]:
    try:
        config = json.loads((Path(model_path) / "config.json").read_text())
    except Exception:
        return {}
    return config if isinstance(config, dict) else {}


def _weight_name_is_scale(name: str) -> bool:
    return name.endswith(".weight_scale_inv") or name.endswith(".weight_scale")


def _build_hf_style_loaded_weight_compare_names(
    *,
    layer_indices: list[int],
    include_scales: bool,
    scale_suffixes: tuple[str, ...] = ("weight_scale_inv", "weight_scale"),
    include_qk_norms: bool = True,
) -> list[str]:
    names: list[str] = [
        "model.embed_tokens.weight",
        "lm_head.weight",
        "model.norm.weight",
    ]
    per_layer_suffixes = [
        "input_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "post_attention_layernorm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    if include_qk_norms:
        per_layer_suffixes.extend(
            [
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
            ]
        )
    if include_scales:
        scale_modules = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]
        per_layer_suffixes.extend(
            f"{module_name}.{scale_suffix}"
            for scale_suffix in scale_suffixes
            for module_name in scale_modules
        )

    for layer_idx in layer_indices:
        names.extend(f"model.layers.{layer_idx}.{suffix}" for suffix in per_layer_suffixes)

    deduped: list[str] = []
    for name in names:
        if name not in deduped:
            deduped.append(name)
    return deduped


def _coerce_sglang_weight_payload_to_tensor(payload: Any) -> torch.Tensor | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        if "parameter" in payload:
            payload = payload["parameter"]
        else:
            return None
    try:
        tensor = torch.as_tensor(payload, dtype=torch.float32)
    except (TypeError, ValueError):
        return None
    return tensor.detach().cpu().clone()


def _truncate_weight_sample_for_sglang_compare(tensor: torch.Tensor, *, truncate_size: int) -> torch.Tensor:
    tensor = _clone_weight_for_compare(tensor)
    if tensor.ndim == 0:
        return tensor.reshape(1)
    return tensor[:truncate_size].contiguous()


def _collect_sglang_loaded_weight_samples(
    *,
    adapter: Any,
    names: list[str],
    truncate_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    samples: dict[str, torch.Tensor] = {}
    errors: dict[str, str] = {}
    for name in names:
        try:
            payload = adapter._make_request(
                "get_weights_by_name",
                {"name": name, "truncate_size": truncate_size},
                method="POST",
            )
        except Exception as exc:
            errors[name] = str(exc)
            continue

        tensor = _coerce_sglang_weight_payload_to_tensor(payload)
        if tensor is None:
            errors[name] = "missing"
            continue
        samples[name] = tensor
    return samples, errors


def _resolve_loaded_weight_compare_scale_suffixes(quantization: str | None) -> tuple[str, ...]:
    if _parse_rollout_quantization(quantization or "") == "modelopt_fp4":
        return ("weight_scale",)
    return ("weight_scale_inv", "weight_scale")


def _resolve_loaded_weight_compare_include_qk_norms(model_path: str) -> bool:
    config = _read_hf_config_dict(model_path)
    for key in ("use_qk_norm", "use_qk_norms", "qk_norm", "qk_layernorm", "use_qk_layernorm"):
        if key in config:
            return bool(config[key])

    model_type = str(config.get("model_type", "")).lower()
    architectures = [str(item).lower() for item in config.get("architectures", []) if isinstance(item, str)]
    if model_type.startswith("qwen3"):
        return True
    return any("qwen3" in architecture for architecture in architectures)


def _resolve_sglang_megatron_activation_compare_layer_indices(model_path: str) -> list[int]:
    layer_count = _read_hf_config_num_hidden_layers(model_path)
    raw_layers = (
        os.getenv(SGLANG_MEGATRON_ACTIVATION_COMPARE_LAYERS_ENV)
        or os.getenv("ORBIT_RUNTIME_PARITY_HF_MEGATRON_ACTIVATION_COMPARE_LAYERS")
        or os.getenv(SGLANG_MEGATRON_WEIGHT_COMPARE_LAYERS_ENV)
        or os.getenv("ORBIT_RUNTIME_PARITY_HF_MEGATRON_WEIGHT_COMPARE_LAYERS")
    )
    return _resolve_weight_compare_layer_indices(raw_layers, layer_count=layer_count)


def _make_sglang_tensor_dump_dir() -> str:
    base_dir = Path(os.getenv(SGLANG_TENSOR_DUMP_DIR_ENV, tempfile.gettempdir()))
    base_dir.mkdir(parents=True, exist_ok=True)
    return tempfile.mkdtemp(prefix=f"verl_sglang_tensor_dump_pid{os.getpid()}_", dir=str(base_dir))


def _resolve_sglang_attention_backend_for_parity() -> str | None:
    for env_name in (SGLANG_ATTENTION_BACKEND_ENV, VERL_SGLANG_ATTENTION_BACKEND_ENV):
        value = os.getenv(env_name)
        if value:
            return value

    try:
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 9:
            return "flashinfer"
    except Exception:
        return None
    return None


def _resolve_sglang_fp8_gemm_backend_for_parity() -> str | None:
    for env_name in (SGLANG_FP8_GEMM_BACKEND_ENV, VERL_SGLANG_FP8_GEMM_BACKEND_ENV):
        value = os.getenv(env_name)
        if value:
            normalized = value.strip()
            if normalized:
                return normalized
    return None


def _normalize_sglang_fp4_gemm_backend(value: str) -> str:
    normalized = value.strip()
    if normalized in {"cutlass", "cudnn", "trtllm"}:
        return f"flashinfer_{normalized}"
    return normalized


def _resolve_sglang_fp4_gemm_backend_for_parity() -> str | None:
    for env_name in (
        SGLANG_FP4_GEMM_BACKEND_ENV,
        VERL_SGLANG_FP4_GEMM_BACKEND_ENV,
        SGLANG_LEGACY_FP4_GEMM_BACKEND_ENV,
    ):
        value = os.getenv(env_name)
        if value:
            return _normalize_sglang_fp4_gemm_backend(value)
    return None


def _maybe_emit_sglang_raw_response_debug(
    *,
    idx: int,
    prompt: str,
    response: dict[str, Any],
) -> bool:
    if os.getenv("ORBIT_RUNTIME_PARITY_RAW_RESPONSE", "0") != "1":
        return False
    if getattr(_maybe_emit_sglang_raw_response_debug, "_emitted", False):
        return False

    meta_info = response.get("meta_info", {})
    if not isinstance(meta_info, dict):
        meta_info = {"raw_meta_info": meta_info}

    prompt_text = prompt.replace("\n", "\\n")
    print(
        f"ORBIT_RUNTIME_PARITY_RAW_RESPONSE idx={idx} prompt={prompt_text!r} "
        f"output_ids_type={type(response.get('output_ids')).__name__} "
        f"output_token_logprobs_type={type(meta_info.get('output_token_logprobs')).__name__} "
        f"output_top_logprobs_type={type(meta_info.get('output_top_logprobs')).__name__} "
        f"output_ids={_truncate_debug_repr(response.get('output_ids'))} "
        f"output_token_logprobs={_truncate_debug_repr(meta_info.get('output_token_logprobs'))} "
        f"output_top_logprobs={_truncate_debug_repr(meta_info.get('output_top_logprobs'))}",
        flush=True,
    )
    _maybe_emit_sglang_raw_response_debug._emitted = True
    return True


def collect_sglang_outputs(
    *,
    model_path: str,
    prompts: list[str],
    quantization: str | None,
    sglang_torchao_config: str | None = None,
    sglang_fp8_gemm_backend: str | None = None,
    topk: int = TOPK,
) -> dict[str, object]:
    token_ids = load_rollout_prompt_token_ids(model_path, prompts)
    port = _reserve_local_port()
    adapter_kwargs: dict[str, Any] = {
        "model_path": model_path,
        "host": "127.0.0.1",
        "port": port,
        "quantization": quantization,
        "first_rank_in_node": True,
        "trust_remote_code": True,
        "log_level_http": os.getenv(SGLANG_HTTP_LOG_LEVEL_ENV, "warning"),
    }
    if sglang_torchao_config:
        adapter_kwargs["torchao_config"] = sglang_torchao_config
    attention_backend = _resolve_sglang_attention_backend_for_parity()
    if attention_backend is not None:
        adapter_kwargs["attention_backend"] = attention_backend
    if sglang_fp8_gemm_backend is None:
        sglang_fp8_gemm_backend = _resolve_sglang_fp8_gemm_backend_for_parity()
    if sglang_fp8_gemm_backend:
        adapter_kwargs["fp8_gemm_runner_backend"] = sglang_fp8_gemm_backend
    fp4_gemm_backend = _resolve_sglang_fp4_gemm_backend_for_parity()
    if fp4_gemm_backend is not None:
        adapter_kwargs["fp4_gemm_runner_backend"] = fp4_gemm_backend
    sglang_activation_dump_dir: str | None = None
    sglang_activation_layer_indices: list[int] = []
    if _sglang_megatron_activation_compare_enabled():
        sglang_activation_layer_indices = _resolve_sglang_megatron_activation_compare_layer_indices(model_path)
        sglang_activation_dump_dir = _make_sglang_tensor_dump_dir()
        sglang_activation_internal = _sglang_megatron_activation_compare_internal_enabled()
        adapter_kwargs["debug_tensor_dump_output_folder"] = sglang_activation_dump_dir
        adapter_kwargs["debug_tensor_dump_layers"] = sglang_activation_layer_indices
        adapter_kwargs["disable_cuda_graph"] = True
        if sglang_activation_internal:
            os.environ["TENSOR_DUMP_LAYER_OUTPUT_ONLY"] = "0"
        else:
            os.environ.setdefault("TENSOR_DUMP_LAYER_OUTPUT_ONLY", "1")
        print(
            f"ORBIT_RUNTIME_PARITY_ACTIVATION_COMPARE tensor_dump_dir={sglang_activation_dump_dir} "
            f"layers={sglang_activation_layer_indices} internal={sglang_activation_internal}",
            flush=True,
        )
    adapter = HttpServerAdapter(**adapter_kwargs)

    chosen_logprobs: list[float] = []
    topk_tokens_rows: list[list[int]] = []
    topk_logprobs_rows: list[list[float]] = []
    sglang_weight_samples: dict[str, torch.Tensor] = {}
    sglang_weight_errors: dict[str, str] = {}
    sglang_activation_stages: dict[str, torch.Tensor] = {}
    try:
        for idx, prompt in enumerate(prompts):
            response = adapter.generate(
                prompt=prompt,
                sampling_params={
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "max_new_tokens": 1,
                },
                return_logprob=True,
                top_logprobs_num=topk,
            )
            _maybe_emit_sglang_raw_response_debug(idx=idx, prompt=prompt, response=response)
            _, chosen_logprob, topk_tokens, topk_logprobs = _extract_sglang_step0(response, topk=topk)
            chosen_logprobs.append(chosen_logprob)
            topk_tokens_rows.append(topk_tokens)
            topk_logprobs_rows.append(topk_logprobs)

        if _sglang_megatron_weight_compare_enabled():
            layer_count = _read_hf_config_num_hidden_layers(model_path)
            raw_layers = os.getenv(SGLANG_MEGATRON_WEIGHT_COMPARE_LAYERS_ENV) or os.getenv(
                "ORBIT_RUNTIME_PARITY_HF_MEGATRON_WEIGHT_COMPARE_LAYERS"
            )
            layer_indices = _resolve_weight_compare_layer_indices(raw_layers, layer_count=layer_count)
            include_scales = _sglang_megatron_weight_compare_include_scales()
            truncate_size = _sglang_megatron_weight_compare_truncate_size()
            names = _build_hf_style_loaded_weight_compare_names(
                layer_indices=layer_indices,
                include_scales=include_scales,
                scale_suffixes=_resolve_loaded_weight_compare_scale_suffixes(quantization),
                include_qk_norms=_resolve_loaded_weight_compare_include_qk_norms(model_path),
            )
            sglang_weight_samples, sglang_weight_errors = _collect_sglang_loaded_weight_samples(
                adapter=adapter,
                names=names,
                truncate_size=truncate_size,
            )
        if sglang_activation_dump_dir is not None:
            sglang_activation_stages = _collect_sglang_tensor_dump_layer_outputs(
                dump_dir=sglang_activation_dump_dir,
                token_ids=token_ids,
                layer_indices=sglang_activation_layer_indices,
                model_config=_read_hf_config_dict(model_path),
            )
            print(
                f"ORBIT_RUNTIME_PARITY_ACTIVATION_COMPARE "
                f"collected_sglang_stages={sorted(sglang_activation_stages)}",
                flush=True,
            )
    finally:
        adapter.shutdown()

    observed_scores = _stack_float_rows(topk_logprobs_rows)
    result = {
        "token_ids": token_ids,
        "logprobs": torch.tensor(chosen_logprobs, dtype=torch.float32),
        "topk_tokens": _stack_int_rows(topk_tokens_rows),
        "topk_logprobs": observed_scores,
        # The real rollout path exposes per-step top-k scores, not dense vocab logits.
        "logits": observed_scores,
    }
    if _sglang_megatron_weight_compare_enabled():
        result["sglang_weight_samples"] = sglang_weight_samples
        result["sglang_weight_errors"] = sglang_weight_errors
    if _sglang_megatron_activation_compare_enabled():
        result["sglang_activation_stages"] = sglang_activation_stages
    return result


def run_sglang_rollout_step0(
    *,
    hf_model: str,
    prompts: list[str],
    rollout_quantization: str | None = "fp8",
    sglang_torchao_config: str | None = None,
    sglang_fp8_gemm_backend: str | None = None,
) -> dict[str, object]:
    return collect_sglang_outputs(
        model_path=hf_model,
        prompts=prompts,
        quantization=rollout_quantization,
        sglang_torchao_config=sglang_torchao_config,
        sglang_fp8_gemm_backend=sglang_fp8_gemm_backend,
    )


def _ensure_real_single_rank_runtime(
    *,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
) -> None:
    if tensor_parallel_size != 1 or pipeline_parallel_size != 1:
        raise SystemExit(
            "step-0 runtime parity currently supports only the real single-rank preflight path "
            "(TP=1, PP=1). Use distributed bootstrap such as torchrun for multi-rank runtime parity."
        )


def _prepare_single_rank_dist() -> bool:
    import torch.distributed as dist

    if dist.is_initialized():
        if dist.get_world_size() != 1 or dist.get_rank() != 0:
            raise RuntimeError("Existing distributed context is incompatible with the step-0 single-rank preflight.")
        return False

    device_name = _get_device_name()
    if device_name != "cuda":
        raise RuntimeError(
            f"Real Megatron step-0 parity requires a CUDA runtime, got device backend {device_name!r}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the real Megatron step-0 parity path.")

    torch.cuda.set_device(0)
    port = _reserve_local_port()
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=0,
        world_size=1,
    )
    return True


def _extract_model_logits(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, list | tuple) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Could not extract logits tensor from Megatron output: {type(output).__name__}")


def _select_last_token_logits_from_packed_output(
    logits: torch.Tensor,
    *,
    lengths: torch.Tensor,
    cu_seqlens_q_padded: torch.Tensor | list[int],
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError(f"Expected packed Megatron logits with shape [1, packed_seq, vocab], got {tuple(logits.shape)}")
    if logits.shape[0] != 1:
        raise ValueError(f"Expected single packed batch dimension, got {logits.shape[0]}")

    packed_logits = logits[0]
    cu_seqlens_q_padded = torch.as_tensor(cu_seqlens_q_padded, device=packed_logits.device, dtype=lengths.dtype)
    start_positions = cu_seqlens_q_padded[:-1]
    last_positions = start_positions + lengths - 1
    return packed_logits[last_positions]


def _select_last_token_logits_from_nested_output(
    logits: torch.Tensor,
    *,
    lengths: torch.Tensor,
) -> torch.Tensor:
    if getattr(logits, "is_nested", False):
        flat_logits = logits.values()
        offsets = torch.as_tensor(logits.offsets(), device=flat_logits.device, dtype=lengths.dtype)
        last_positions = offsets[:-1] + lengths - 1
        return flat_logits[last_positions]

    if logits.ndim != 3:
        raise ValueError(
            f"Expected postprocessed Megatron logits with shape [batch, seq, vocab] or nested jagged tensor, got {tuple(logits.shape)}"
        )

    batch_positions = torch.arange(logits.shape[0], device=logits.device)
    last_positions = lengths.to(device=logits.device) - 1
    return logits[batch_positions, last_positions]


def _build_padded_prompt_batch(
    token_ids: list[list[int]],
    *,
    device: torch.device,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = len(token_ids)
    max_len = max(len(ids) for ids in token_ids)

    input_ids = torch.full((batch_size, max_len), fill_value=pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    for row_idx, ids in enumerate(token_ids):
        seq_len = len(ids)
        input_ids[row_idx, :seq_len] = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask[row_idx, :seq_len] = True
        position_ids[row_idx, :seq_len] = torch.arange(seq_len, dtype=torch.long, device=device)

    return input_ids, attention_mask, position_ids


def _maybe_emit_actor_forward_compare_debug(
    *,
    prompts: list[str],
    no_padding_result: dict[str, object],
    packed_result: dict[str, object],
) -> bool:
    if os.getenv("ORBIT_RUNTIME_PARITY_ACTOR_FORWARD_COMPARE", "0") != "1":
        return False

    no_padding_topk_tokens = torch.as_tensor(no_padding_result.get("topk_tokens", []), dtype=torch.long)
    packed_topk_tokens = torch.as_tensor(packed_result.get("topk_tokens", []), dtype=torch.long)
    no_padding_topk_logprobs = torch.as_tensor(no_padding_result.get("topk_logprobs", []), dtype=torch.float32)
    packed_topk_logprobs = torch.as_tensor(packed_result.get("topk_logprobs", []), dtype=torch.float32)
    no_padding_logprobs = torch.as_tensor(no_padding_result.get("logprobs", []), dtype=torch.float32)
    packed_logprobs = torch.as_tensor(packed_result.get("logprobs", []), dtype=torch.float32)

    row_count = min(
        len(prompts),
        no_padding_topk_tokens.shape[0] if no_padding_topk_tokens.ndim >= 1 else 0,
        packed_topk_tokens.shape[0] if packed_topk_tokens.ndim >= 1 else 0,
    )
    for idx in range(row_count):
        prompt_text = prompts[idx].replace("\n", "\\n")
        print(
            f"ORBIT_RUNTIME_PARITY_ACTOR_FORWARD_COMPARE idx={idx} "
            f"prompt={prompt_text!r} "
            f"nopad_chosen_logprob={float(no_padding_logprobs[idx]) if idx < no_padding_logprobs.numel() else 'n/a'} "
            f"packed_chosen_logprob={float(packed_logprobs[idx]) if idx < packed_logprobs.numel() else 'n/a'} "
            f"nopad_topk_tokens={no_padding_topk_tokens[idx].tolist()} "
            f"nopad_topk_logprobs={[float(v) for v in no_padding_topk_logprobs[idx].tolist()]} "
            f"packed_topk_tokens={packed_topk_tokens[idx].tolist()} "
            f"packed_topk_logprobs={[float(v) for v in packed_topk_logprobs[idx].tolist()]}",
            flush=True,
        )

    return True


def _run_actor_packed_forward_debug(
    *,
    packed_forward_fn: Any,
    module: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    mtp_config: Any,
):
    return packed_forward_fn(
        module,
        input_ids,
        attention_mask,
        position_ids,
        {},
        data_format="thd",
        mtp_config=mtp_config,
    )


def _collect_hf_direct_outputs(
    *,
    model_path: str,
    token_ids: list[list[int]],
    topk: int,
    device: torch.device,
) -> dict[str, object]:
    from transformers import AutoModelForCausalLM

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.float32 if device.type == "cpu" else "auto",
    }
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()
    model = model.to(device)

    pad_token_id = getattr(getattr(model, "generation_config", None), "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(getattr(model, "config", None), "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(getattr(model, "config", None), "eos_token_id", None)
    if pad_token_id is None:
        raise ValueError("Direct HF compare requires a pad_token_id or eos_token_id fallback.")

    input_ids, attention_mask, _ = _build_padded_prompt_batch(token_ids, device=device, pad_token_id=pad_token_id)
    lengths = torch.tensor([len(ids) for ids in token_ids], dtype=torch.long, device=device)

    with torch.inference_mode():
        output = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = _extract_model_logits(output).to(torch.float32)
        next_token_logits = _select_last_token_logits_from_nested_output(
            logits,
            lengths=lengths,
        )
        next_token_logprobs = torch.log_softmax(next_token_logits, dim=-1)
        topk = min(topk, next_token_logprobs.shape[-1])
        topk_logprobs, topk_tokens = torch.topk(next_token_logprobs, k=topk, dim=-1)

    return {
        "logprobs": topk_logprobs[:, 0].cpu(),
        "topk_tokens": topk_tokens.cpu(),
        "topk_logprobs": topk_logprobs.cpu(),
    }


def _extract_first_tensor_from_debug_output(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            tensor = _extract_first_tensor_from_debug_output(item)
            if tensor is not None:
                return tensor
        return None
    if isinstance(value, dict):
        for item in value.values():
            tensor = _extract_first_tensor_from_debug_output(item)
            if tensor is not None:
                return tensor
    return None


def _clone_activation_for_debug(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu").clone()


def _activation_tensor_for_compare(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(device="cpu", dtype=torch.float32)


def _select_last_token_hidden_for_debug(hidden: torch.Tensor, *, lengths: torch.Tensor) -> torch.Tensor:
    hidden = _clone_activation_for_debug(hidden)
    lengths = lengths.detach().to(device="cpu", dtype=torch.long)

    batch_size = int(lengths.numel())
    total_tokens = int(lengths.sum().item()) if lengths.numel() else 0
    final_token_indices = torch.cumsum(lengths, dim=0) - 1 if lengths.numel() else torch.zeros(0, dtype=torch.long)

    if hidden.ndim == 0:
        return hidden.reshape(1, 1)
    if hidden.ndim == 1:
        if hidden.shape[0] == batch_size:
            return hidden.reshape(batch_size, 1)
        return hidden.reshape(1, -1)
    if hidden.ndim == 2:
        if hidden.shape[0] == batch_size:
            return hidden.reshape(batch_size, -1)
        if total_tokens > 0 and hidden.shape[0] == total_tokens:
            return hidden.index_select(0, final_token_indices).reshape(batch_size, -1)
        if hidden.shape[1] == batch_size:
            return hidden.transpose(0, 1).reshape(batch_size, -1)
        return hidden.reshape(hidden.shape[0], -1)

    if hidden.shape[0] == batch_size:
        row_indices = torch.arange(batch_size, dtype=torch.long)
        return hidden[row_indices, lengths - 1].reshape(batch_size, -1)
    if hidden.shape[1] == batch_size:
        row_indices = torch.arange(batch_size, dtype=torch.long)
        return hidden[lengths - 1, row_indices].reshape(batch_size, -1)
    if total_tokens > 0 and hidden.shape[0] == total_tokens:
        return hidden.index_select(0, final_token_indices).reshape(batch_size, -1)
    return hidden.reshape(hidden.shape[0], -1)


def _split_packed_qkv_output_for_hf_compare(
    packed_qkv: torch.Tensor,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    num_query_groups: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv_denominator = num_attention_heads + 2 * num_key_value_heads
    if qkv_denominator <= 0 or packed_qkv.shape[-1] % qkv_denominator != 0:
        raise ValueError(
            "Cannot infer Megatron QKV activation units from "
            f"qkv_cols={packed_qkv.shape[-1]}, num_attention_heads={num_attention_heads}, "
            f"num_key_value_heads={num_key_value_heads}."
        )

    units_per_head = packed_qkv.shape[-1] // qkv_denominator
    q_size = units_per_head * num_attention_heads
    kv_size = units_per_head * num_key_value_heads
    q_size_chunk = q_size // num_query_groups
    kv_size_chunk = kv_size // num_query_groups

    q_parts: list[torch.Tensor] = []
    k_parts: list[torch.Tensor] = []
    v_parts: list[torch.Tensor] = []
    for qkv_chunk in packed_qkv.chunk(num_query_groups, dim=-1):
        q_parts.append(qkv_chunk[..., :q_size_chunk])
        k_parts.append(qkv_chunk[..., q_size_chunk : q_size_chunk + kv_size_chunk])
        v_parts.append(qkv_chunk[..., q_size_chunk + kv_size_chunk : q_size_chunk + 2 * kv_size_chunk])

    return torch.cat(q_parts, dim=-1), torch.cat(k_parts, dim=-1), torch.cat(v_parts, dim=-1)


def _split_sglang_qkv_output_for_hf_compare(
    packed_qkv: torch.Tensor,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv_denominator = num_attention_heads + 2 * num_key_value_heads
    if qkv_denominator <= 0 or packed_qkv.shape[-1] % qkv_denominator != 0:
        raise ValueError(
            "Cannot infer SGLang QKV activation units from "
            f"qkv_cols={packed_qkv.shape[-1]}, num_attention_heads={num_attention_heads}, "
            f"num_key_value_heads={num_key_value_heads}."
        )

    units_per_head = packed_qkv.shape[-1] // qkv_denominator
    q_size = units_per_head * num_attention_heads
    kv_size = units_per_head * num_key_value_heads
    q_proj = packed_qkv[..., :q_size]
    k_proj = packed_qkv[..., q_size : q_size + kv_size]
    v_proj = packed_qkv[..., q_size + kv_size : q_size + 2 * kv_size]
    return q_proj, k_proj, v_proj


def _split_fc1_output_for_hf_compare(packed_fc1: torch.Tensor, *, intermediate_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    if packed_fc1.shape[-1] != 2 * intermediate_size:
        raise ValueError(
            f"Expected packed FC1 activation last-dim {2 * intermediate_size}, got {packed_fc1.shape[-1]}."
        )
    return packed_fc1[..., :intermediate_size], packed_fc1[..., intermediate_size:]


def _read_positive_int_from_config(config: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        if key not in config:
            continue
        try:
            value = int(config[key])
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _effective_layernorm_weight_for_debug(weight: torch.Tensor, *, config: Any) -> torch.Tensor:
    weight = weight.detach().to(dtype=torch.float32)
    if bool(getattr(config, "layernorm_zero_centered_gamma", False)):
        return weight + 1.0
    return weight


def _apply_fused_layernorm_for_debug(
    hidden: torch.Tensor,
    *,
    fused_module: Any,
    config: Any,
) -> torch.Tensor | None:
    weight = getattr(fused_module, "layer_norm_weight", None)
    if not isinstance(weight, torch.Tensor):
        return None

    eps = float(getattr(fused_module, "eps", getattr(config, "layernorm_epsilon", 1.0e-5)))
    hidden_f32 = hidden.detach().to(dtype=torch.float32)
    weight_f32 = _effective_layernorm_weight_for_debug(weight, config=config).to(device=hidden_f32.device)
    normalization = str(getattr(config, "normalization", "RMSNorm")).lower()

    if normalization == "layernorm":
        mean = hidden_f32.mean(dim=-1, keepdim=True)
        centered = hidden_f32 - mean
        variance = centered.pow(2).mean(dim=-1, keepdim=True)
        normalized = centered * torch.rsqrt(variance + eps)
    else:
        variance = hidden_f32.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_f32 * torch.rsqrt(variance + eps)

    output = normalized * weight_f32
    bias = getattr(fused_module, "layer_norm_bias", None)
    if isinstance(bias, torch.Tensor):
        output = output + bias.detach().to(device=hidden_f32.device, dtype=torch.float32)
    if hidden.is_floating_point():
        output = output.to(dtype=hidden.dtype)
    return output


def _attach_activation_capture_hooks(root: Any, module_paths: dict[str, str]) -> tuple[dict[str, torch.Tensor], list[Any]]:
    activations: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    for name, path in module_paths.items():
        try:
            target_module = root.get_submodule(path)
        except Exception:
            continue

        def _hook(_module, _args, output, *, _name=name):
            tensor = _extract_first_tensor_from_debug_output(output)
            if tensor is None or _name in activations:
                return
            activations[_name] = _clone_activation_for_debug(tensor)

        handles.append(target_module.register_forward_hook(_hook))

    return activations, handles


def _attach_fused_layernorm_activation_capture_hooks(
    root: Any,
    module_paths: dict[str, str],
    *,
    config: Any,
) -> tuple[dict[str, torch.Tensor], list[Any]]:
    activations: dict[str, torch.Tensor] = {}
    handles: list[Any] = []

    for name, path in module_paths.items():
        try:
            target_module = root.get_submodule(path)
        except Exception:
            continue
        if not isinstance(getattr(target_module, "layer_norm_weight", None), torch.Tensor):
            continue

        def _hook(module, args, *, _name=name):
            if _name in activations:
                return
            hidden = _extract_first_tensor_from_debug_output(args)
            if hidden is None:
                return
            raw_input_name = _activation_stage_input_name(_name)
            if raw_input_name not in activations:
                activations[raw_input_name] = _clone_activation_for_debug(hidden)
            normalized = _apply_fused_layernorm_for_debug(hidden, fused_module=module, config=config)
            if normalized is None:
                return
            activations[_name] = _clone_activation_for_debug(normalized)

        handles.append(target_module.register_forward_pre_hook(_hook))

    return activations, handles


def _activation_stage_name(layer_idx: int, name: str) -> str:
    return f"layer{layer_idx}.{name}"


def _activation_stage_input_name(stage_name: str) -> str:
    return f"{stage_name}_input"


def _collect_hf_direct_activation_outputs(
    *,
    model_path: str,
    token_ids: list[list[int]],
    device: torch.device,
    raw_layer_indices: str | None = None,
) -> dict[str, torch.Tensor]:
    from transformers import AutoModelForCausalLM

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.float32 if device.type == "cpu" else "auto",
    }
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    model.eval()
    model = model.to(device)

    pad_token_id = getattr(getattr(model, "generation_config", None), "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(getattr(model, "config", None), "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(getattr(model, "config", None), "eos_token_id", None)
    if pad_token_id is None:
        raise ValueError("HF activation compare requires a pad_token_id or eos_token_id fallback.")

    input_ids, attention_mask, _ = _build_padded_prompt_batch(token_ids, device=device, pad_token_id=pad_token_id)
    lengths = torch.tensor([len(ids) for ids in token_ids], dtype=torch.long, device=device)

    layer_count = len(model.model.layers)
    layer_indices = _resolve_weight_compare_layer_indices(raw_layer_indices, layer_count=layer_count)
    module_paths = {"embed_tokens": "model.embed_tokens"}
    for layer_idx in layer_indices:
        module_paths.update(
            {
                _activation_stage_name(layer_idx, "input_layernorm"): f"model.layers.{layer_idx}.input_layernorm",
                _activation_stage_name(layer_idx, "q_proj"): f"model.layers.{layer_idx}.self_attn.q_proj",
                _activation_stage_name(layer_idx, "k_proj"): f"model.layers.{layer_idx}.self_attn.k_proj",
                _activation_stage_name(layer_idx, "v_proj"): f"model.layers.{layer_idx}.self_attn.v_proj",
                _activation_stage_name(layer_idx, "o_proj"): f"model.layers.{layer_idx}.self_attn.o_proj",
                _activation_stage_name(layer_idx, "post_attention_layernorm"): f"model.layers.{layer_idx}.post_attention_layernorm",
                _activation_stage_name(layer_idx, "gate_proj"): f"model.layers.{layer_idx}.mlp.gate_proj",
                _activation_stage_name(layer_idx, "up_proj"): f"model.layers.{layer_idx}.mlp.up_proj",
                _activation_stage_name(layer_idx, "down_proj"): f"model.layers.{layer_idx}.mlp.down_proj",
                _activation_stage_name(layer_idx, "layer_output"): f"model.layers.{layer_idx}",
            }
        )

    activations, handles = _attach_activation_capture_hooks(model, module_paths)
    try:
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for handle in handles:
            handle.remove()

    logits = _extract_model_logits(output).to(torch.float32)
    next_token_logits = _select_last_token_logits_from_nested_output(logits, lengths=lengths)

    stage_outputs: dict[str, torch.Tensor] = {
        "final_logits": next_token_logits.detach().to(device="cpu", dtype=torch.float32),
    }
    for name in module_paths:
        tensor = activations.get(name)
        if tensor is None:
            continue
        stage_outputs[name] = _select_last_token_hidden_for_debug(tensor, lengths=lengths)

    return stage_outputs


def _collect_megatron_packed_activation_outputs(
    *,
    module: Any,
    packed_forward_fn: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    lengths: torch.Tensor,
    mtp_config: Any,
    raw_layer_indices: str | None = None,
) -> dict[str, torch.Tensor]:
    compare_root = _unwrap_megatron_module_for_weight_compare(module)
    layer_count = len(compare_root.decoder.layers)
    layer_indices = _resolve_weight_compare_layer_indices(raw_layer_indices, layer_count=layer_count)
    module_paths = {"embed_tokens": "embedding.word_embeddings"}
    fused_layernorm_module_paths: dict[str, str] = {}
    for layer_idx in layer_indices:
        module_paths.update(
            {
                _activation_stage_name(layer_idx, "input_layernorm"): f"decoder.layers.{layer_idx}.input_layernorm",
                _activation_stage_name(layer_idx, "qkv_proj"): f"decoder.layers.{layer_idx}.self_attention.linear_qkv",
                _activation_stage_name(layer_idx, "o_proj"): f"decoder.layers.{layer_idx}.self_attention.linear_proj",
                _activation_stage_name(layer_idx, "post_attention_layernorm"): f"decoder.layers.{layer_idx}.pre_mlp_layernorm",
                _activation_stage_name(layer_idx, "fc1_proj"): f"decoder.layers.{layer_idx}.mlp.linear_fc1",
                _activation_stage_name(layer_idx, "fc2_proj"): f"decoder.layers.{layer_idx}.mlp.linear_fc2",
                _activation_stage_name(layer_idx, "layer_output"): f"decoder.layers.{layer_idx}",
            }
        )
        fused_layernorm_module_paths.update(
            {
                _activation_stage_name(layer_idx, "input_layernorm"): (
                    f"decoder.layers.{layer_idx}.self_attention.linear_qkv"
                ),
                _activation_stage_name(layer_idx, "post_attention_layernorm"): (
                    f"decoder.layers.{layer_idx}.mlp.linear_fc1"
                ),
            }
        )

    activations, handles = _attach_activation_capture_hooks(compare_root, module_paths)
    fused_layernorm_activations, fused_layernorm_handles = _attach_fused_layernorm_activation_capture_hooks(
        compare_root,
        fused_layernorm_module_paths,
        config=compare_root.config,
    )
    handles.extend(fused_layernorm_handles)
    try:
        packed_output = _run_actor_packed_forward_debug(
            packed_forward_fn=packed_forward_fn,
            module=module,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            mtp_config=mtp_config,
        )
    finally:
        for handle in handles:
            handle.remove()

    logits = _extract_model_logits(packed_output).to(torch.float32)
    next_token_logits = _select_last_token_logits_from_nested_output(logits, lengths=lengths)
    config = compare_root.config
    num_query_groups = getattr(config, "num_query_groups", getattr(config, "num_key_value_heads", None))
    num_key_value_heads = getattr(config, "num_key_value_heads", num_query_groups)
    intermediate_size = getattr(config, "intermediate_size", getattr(config, "ffn_hidden_size", None))

    stage_outputs: dict[str, torch.Tensor] = {
        "final_logits": next_token_logits.detach().to(device="cpu", dtype=torch.float32),
    }

    for layer_idx in layer_indices:
        for name in (
            _activation_stage_input_name(_activation_stage_name(layer_idx, "input_layernorm")),
            _activation_stage_name(layer_idx, "input_layernorm"),
            _activation_stage_name(layer_idx, "o_proj"),
            _activation_stage_input_name(_activation_stage_name(layer_idx, "post_attention_layernorm")),
            _activation_stage_name(layer_idx, "post_attention_layernorm"),
            _activation_stage_name(layer_idx, "layer_output"),
        ):
            tensor = fused_layernorm_activations.get(name)
            if tensor is None:
                tensor = activations.get(name)
            if tensor is None:
                continue
            stage_outputs[name] = _select_last_token_hidden_for_debug(tensor, lengths=lengths)

        qkv_tensor = activations.get(_activation_stage_name(layer_idx, "qkv_proj"))
        if qkv_tensor is not None and num_query_groups is not None and num_key_value_heads is not None:
            q_proj, k_proj, v_proj = _split_packed_qkv_output_for_hf_compare(
                _select_last_token_hidden_for_debug(qkv_tensor, lengths=lengths),
                num_attention_heads=config.num_attention_heads,
                num_key_value_heads=num_key_value_heads,
                num_query_groups=num_query_groups,
            )
            stage_outputs[_activation_stage_name(layer_idx, "q_proj")] = q_proj
            stage_outputs[_activation_stage_name(layer_idx, "k_proj")] = k_proj
            stage_outputs[_activation_stage_name(layer_idx, "v_proj")] = v_proj

        fc1_tensor = activations.get(_activation_stage_name(layer_idx, "fc1_proj"))
        if fc1_tensor is not None and intermediate_size is not None:
            gate_proj, up_proj = _split_fc1_output_for_hf_compare(
                _select_last_token_hidden_for_debug(fc1_tensor, lengths=lengths),
                intermediate_size=intermediate_size,
            )
            stage_outputs[_activation_stage_name(layer_idx, "gate_proj")] = gate_proj
            stage_outputs[_activation_stage_name(layer_idx, "up_proj")] = up_proj

        fc2_tensor = activations.get(_activation_stage_name(layer_idx, "fc2_proj"))
        if fc2_tensor is not None:
            stage_outputs[_activation_stage_name(layer_idx, "down_proj")] = _select_last_token_hidden_for_debug(
                fc2_tensor,
                lengths=lengths,
            )

    for name in ("embed_tokens",):
        tensor = activations.get(name)
        if tensor is None:
            continue
        stage_outputs[name] = _select_last_token_hidden_for_debug(tensor, lengths=lengths)

    return stage_outputs


def _sglang_dump_prompt_index(
    *,
    dumped_input_ids: torch.Tensor,
    token_ids: list[list[int]],
) -> int | None:
    dumped_ids = [int(value) for value in dumped_input_ids.reshape(-1).to(dtype=torch.long).tolist()]
    for idx, prompt_ids in enumerate(token_ids):
        if dumped_ids == prompt_ids:
            return idx
    return None


def _iter_sglang_tensor_dump_files(dump_dir: str) -> list[Path]:
    root = Path(dump_dir)
    if not root.exists():
        return []
    return sorted(root.glob("TP*_PP*_Rank*_pid*/Pass*.pt"))


def _extract_sglang_decoder_layer_output_for_debug(value: Any) -> torch.Tensor | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        hidden_states = value[0]
        residual = value[1]
        if isinstance(hidden_states, torch.Tensor) and isinstance(residual, torch.Tensor):
            if tuple(hidden_states.shape) == tuple(residual.shape):
                return hidden_states + residual
            return hidden_states
    return _extract_first_tensor_from_debug_output(value)


def _extract_sglang_layernorm_input_for_debug(value: Any) -> torch.Tensor | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2 and isinstance(value[1], torch.Tensor):
        return value[1]
    return None


def _collect_sglang_tensor_dump_layer_outputs(
    *,
    dump_dir: str,
    token_ids: list[list[int]],
    layer_indices: list[int],
    model_config: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    model_config = model_config or {}
    num_attention_heads = _read_positive_int_from_config(model_config, "num_attention_heads", "n_head")
    num_key_value_heads = _read_positive_int_from_config(model_config, "num_key_value_heads", "num_kv_heads")
    if num_key_value_heads is None:
        num_key_value_heads = num_attention_heads
    intermediate_size = _read_positive_int_from_config(model_config, "intermediate_size", "ffn_hidden_size")
    rows_by_stage: dict[str, dict[int, torch.Tensor]] = {}

    def _record_stage(
        *,
        stage_name: str,
        prompt_idx: int,
        tensor: torch.Tensor | None,
        seq_lens: torch.Tensor,
    ) -> None:
        if tensor is None:
            return
        rows_by_stage.setdefault(stage_name, {})[prompt_idx] = _select_last_token_hidden_for_debug(
            tensor,
            lengths=seq_lens.to(dtype=torch.long),
        ).reshape(1, -1)

    for dump_file in _iter_sglang_tensor_dump_files(dump_dir):
        try:
            dump = torch.load(dump_file, map_location="cpu")
        except Exception as exc:
            print(f"ORBIT_RUNTIME_PARITY_TENSOR_DUMP file={dump_file} status=load_failed error={exc!r}", flush=True)
            continue
        if not isinstance(dump, dict):
            continue

        dumped_input_ids = dump.get("model.forward_batch_info.input_ids")
        if not isinstance(dumped_input_ids, torch.Tensor):
            continue
        prompt_idx = _sglang_dump_prompt_index(dumped_input_ids=dumped_input_ids, token_ids=token_ids)
        if prompt_idx is None:
            continue

        seq_lens = dump.get("model.forward_batch_info.seq_lens")
        if not isinstance(seq_lens, torch.Tensor):
            seq_lens = torch.tensor([len(token_ids[prompt_idx])], dtype=torch.long)

        for layer_idx in layer_indices:
            for stage_name, dump_key in {
                "input_layernorm": f"model.layers.{layer_idx}.input_layernorm",
                "o_proj": f"model.layers.{layer_idx}.self_attn.o_proj",
                "post_attention_layernorm": f"model.layers.{layer_idx}.post_attention_layernorm",
                "down_proj": f"model.layers.{layer_idx}.mlp.down_proj",
            }.items():
                dump_value = dump.get(dump_key)
                _record_stage(
                    stage_name=_activation_stage_name(layer_idx, stage_name),
                    prompt_idx=prompt_idx,
                    tensor=_extract_first_tensor_from_debug_output(dump_value),
                    seq_lens=seq_lens,
                )
                if stage_name in {"input_layernorm", "post_attention_layernorm"}:
                    _record_stage(
                        stage_name=_activation_stage_input_name(_activation_stage_name(layer_idx, stage_name)),
                        prompt_idx=prompt_idx,
                        tensor=_extract_sglang_layernorm_input_for_debug(dump_value),
                        seq_lens=seq_lens,
                    )

            _record_stage(
                stage_name=_activation_stage_name(layer_idx, "layer_output"),
                prompt_idx=prompt_idx,
                tensor=_extract_sglang_decoder_layer_output_for_debug(dump.get(f"model.layers.{layer_idx}")),
                seq_lens=seq_lens,
            )

            qkv_tensor = _extract_first_tensor_from_debug_output(
                dump.get(f"model.layers.{layer_idx}.self_attn.qkv_proj")
            )
            if qkv_tensor is not None and num_attention_heads is not None and num_key_value_heads is not None:
                try:
                    q_proj, k_proj, v_proj = _split_sglang_qkv_output_for_hf_compare(
                        _select_last_token_hidden_for_debug(qkv_tensor, lengths=seq_lens.to(dtype=torch.long)),
                        num_attention_heads=num_attention_heads,
                        num_key_value_heads=num_key_value_heads,
                    )
                except ValueError as exc:
                    print(
                        f"ORBIT_RUNTIME_PARITY_TENSOR_DUMP file={dump_file} layer={layer_idx} "
                        f"stage=qkv_proj status=split_failed error={exc}",
                        flush=True,
                    )
                else:
                    rows_by_stage.setdefault(_activation_stage_name(layer_idx, "q_proj"), {})[prompt_idx] = q_proj
                    rows_by_stage.setdefault(_activation_stage_name(layer_idx, "k_proj"), {})[prompt_idx] = k_proj
                    rows_by_stage.setdefault(_activation_stage_name(layer_idx, "v_proj"), {})[prompt_idx] = v_proj

            gate_up_tensor = _extract_first_tensor_from_debug_output(
                dump.get(f"model.layers.{layer_idx}.mlp.gate_up_proj")
            )
            if gate_up_tensor is not None and intermediate_size is not None:
                try:
                    gate_proj, up_proj = _split_fc1_output_for_hf_compare(
                        _select_last_token_hidden_for_debug(gate_up_tensor, lengths=seq_lens.to(dtype=torch.long)),
                        intermediate_size=intermediate_size,
                    )
                except ValueError as exc:
                    print(
                        f"ORBIT_RUNTIME_PARITY_TENSOR_DUMP file={dump_file} layer={layer_idx} "
                        f"stage=gate_up_proj status=split_failed error={exc}",
                        flush=True,
                    )
                else:
                    rows_by_stage.setdefault(_activation_stage_name(layer_idx, "gate_proj"), {})[prompt_idx] = gate_proj
                    rows_by_stage.setdefault(_activation_stage_name(layer_idx, "up_proj"), {})[prompt_idx] = up_proj

    stage_outputs: dict[str, torch.Tensor] = {}
    for stage_name, rows_by_idx in rows_by_stage.items():
        rows: list[torch.Tensor] = []
        for prompt_idx in range(len(token_ids)):
            row = rows_by_idx.get(prompt_idx)
            if row is None:
                break
            rows.append(row)
        if len(rows) == len(token_ids):
            stage_outputs[stage_name] = torch.cat(rows, dim=0)
    return stage_outputs


def _maybe_emit_hf_direct_compare_debug(
    *,
    prompts: list[str],
    actor_result: dict[str, object],
    hf_result: dict[str, object],
) -> bool:
    if os.getenv("ORBIT_RUNTIME_PARITY_HF_DIRECT_COMPARE", "0") != "1":
        return False

    actor_topk_tokens = torch.as_tensor(actor_result.get("topk_tokens", []), dtype=torch.long)
    hf_topk_tokens = torch.as_tensor(hf_result.get("topk_tokens", []), dtype=torch.long)
    actor_topk_logprobs = torch.as_tensor(actor_result.get("topk_logprobs", []), dtype=torch.float32)
    hf_topk_logprobs = torch.as_tensor(hf_result.get("topk_logprobs", []), dtype=torch.float32)
    actor_logprobs = torch.as_tensor(actor_result.get("logprobs", []), dtype=torch.float32)
    hf_logprobs = torch.as_tensor(hf_result.get("logprobs", []), dtype=torch.float32)

    row_count = min(
        len(prompts),
        actor_topk_tokens.shape[0] if actor_topk_tokens.ndim >= 1 else 0,
        hf_topk_tokens.shape[0] if hf_topk_tokens.ndim >= 1 else 0,
    )
    for idx in range(row_count):
        prompt_text = prompts[idx].replace("\n", "\\n")
        print(
            f"ORBIT_RUNTIME_PARITY_HF_DIRECT_COMPARE idx={idx} "
            f"prompt={prompt_text!r} "
            f"actor_chosen_logprob={float(actor_logprobs[idx]) if idx < actor_logprobs.numel() else 'n/a'} "
            f"hf_chosen_logprob={float(hf_logprobs[idx]) if idx < hf_logprobs.numel() else 'n/a'} "
            f"actor_topk_tokens={actor_topk_tokens[idx].tolist()} "
            f"actor_topk_logprobs={[float(v) for v in actor_topk_logprobs[idx].tolist()]} "
            f"hf_topk_tokens={hf_topk_tokens[idx].tolist()} "
            f"hf_topk_logprobs={[float(v) for v in hf_topk_logprobs[idx].tolist()]}",
            flush=True,
        )

    return True


def _maybe_emit_hf_megatron_activation_compare_debug(
    *,
    prompts: list[str],
    megatron_stages: dict[str, torch.Tensor],
    hf_stages: dict[str, torch.Tensor],
) -> bool:
    if os.getenv("ORBIT_RUNTIME_PARITY_HF_MEGATRON_ACTIVATION_COMPARE", "0") != "1":
        return False

    ordered_names = sorted(set(megatron_stages) | set(hf_stages), key=_activation_compare_sort_key)
    bad_threshold = float(os.getenv("ORBIT_RUNTIME_PARITY_HF_MEGATRON_ACTIVATION_EXPLOSION_THRESHOLD", "10.0"))
    first_bad: tuple[str, float, float] | None = None

    for name in ordered_names:
        megatron_tensor = megatron_stages.get(name)
        hf_tensor = hf_stages.get(name)
        if megatron_tensor is None or hf_tensor is None:
            print(
                f"ORBIT_RUNTIME_PARITY_HF_MEGATRON_ACTIVATION_COMPARE stage={name} "
                f"status=missing megatron_present={megatron_tensor is not None} hf_present={hf_tensor is not None}",
                flush=True,
            )
            continue

        shape_match = tuple(megatron_tensor.shape) == tuple(hf_tensor.shape)
        if shape_match:
            diff = torch.abs(
                _activation_tensor_for_compare(megatron_tensor) - _activation_tensor_for_compare(hf_tensor)
            )
            max_abs_diff = float(diff.max().item()) if diff.numel() else 0.0
            mean_abs_diff = float(diff.mean().item()) if diff.numel() else 0.0
            if first_bad is None and max_abs_diff > bad_threshold:
                first_bad = (name, max_abs_diff, mean_abs_diff)
        else:
            max_abs_diff = float("nan")
            mean_abs_diff = float("nan")

        print(
            f"ORBIT_RUNTIME_PARITY_HF_MEGATRON_ACTIVATION_COMPARE stage={name} "
            f"prompt_count={len(prompts)} "
            f"shape_match={shape_match} "
            f"megatron_shape={tuple(megatron_tensor.shape)} hf_shape={tuple(hf_tensor.shape)} "
            f"megatron_dtype={megatron_tensor.dtype} hf_dtype={hf_tensor.dtype} "
            f"max_abs_diff={max_abs_diff} mean_abs_diff={mean_abs_diff} "
            f"megatron_sample={_tensor_sample_for_debug(megatron_tensor)} "
            f"hf_sample={_tensor_sample_for_debug(hf_tensor)}",
            flush=True,
        )

    if first_bad is None:
        print(
            f"ORBIT_RUNTIME_PARITY_HF_MEGATRON_ACTIVATION_FIRST_BAD threshold={bad_threshold} status=none",
            flush=True,
        )
    else:
        stage, max_abs_diff, mean_abs_diff = first_bad
        print(
            f"ORBIT_RUNTIME_PARITY_HF_MEGATRON_ACTIVATION_FIRST_BAD threshold={bad_threshold} "
            f"stage={stage} max_abs_diff={max_abs_diff} mean_abs_diff={mean_abs_diff}",
            flush=True,
        )

    return True


def _maybe_emit_sglang_megatron_activation_compare_debug(
    *,
    prompts: list[str],
    megatron_stages: dict[str, torch.Tensor],
    sglang_stages: dict[str, torch.Tensor],
) -> bool:
    if not _sglang_megatron_activation_compare_enabled():
        return False

    ordered_names = sorted(set(megatron_stages) | set(sglang_stages), key=_activation_compare_sort_key)
    bad_threshold = float(os.getenv("ORBIT_RUNTIME_PARITY_SGLANG_MEGATRON_ACTIVATION_EXPLOSION_THRESHOLD", "10.0"))
    first_bad: tuple[str, float, float] | None = None

    for name in ordered_names:
        megatron_tensor = megatron_stages.get(name)
        sglang_tensor = sglang_stages.get(name)
        if megatron_tensor is None or sglang_tensor is None:
            print(
                f"ORBIT_RUNTIME_PARITY_ACTIVATION_COMPARE stage={name} "
                f"status=missing megatron_present={megatron_tensor is not None} "
                f"sglang_present={sglang_tensor is not None}",
                flush=True,
            )
            continue

        shape_match = tuple(megatron_tensor.shape) == tuple(sglang_tensor.shape)
        if shape_match:
            diff = torch.abs(
                _activation_tensor_for_compare(megatron_tensor) - _activation_tensor_for_compare(sglang_tensor)
            )
            max_abs_diff = float(diff.max().item()) if diff.numel() else 0.0
            mean_abs_diff = float(diff.mean().item()) if diff.numel() else 0.0
            if first_bad is None and max_abs_diff > bad_threshold:
                first_bad = (name, max_abs_diff, mean_abs_diff)
        else:
            max_abs_diff = float("nan")
            mean_abs_diff = float("nan")

        print(
            f"ORBIT_RUNTIME_PARITY_ACTIVATION_COMPARE stage={name} "
            f"prompt_count={len(prompts)} "
            f"shape_match={shape_match} "
            f"megatron_shape={tuple(megatron_tensor.shape)} sglang_shape={tuple(sglang_tensor.shape)} "
            f"megatron_dtype={megatron_tensor.dtype} sglang_dtype={sglang_tensor.dtype} "
            f"max_abs_diff={max_abs_diff} mean_abs_diff={mean_abs_diff} "
            f"megatron_sample={_tensor_sample_for_debug(megatron_tensor)} "
            f"sglang_sample={_tensor_sample_for_debug(sglang_tensor)}",
            flush=True,
        )

    if first_bad is None:
        print(
            f"ORBIT_RUNTIME_PARITY_SGLANG_MEGATRON_ACTIVATION_FIRST_BAD threshold={bad_threshold} status=none",
            flush=True,
        )
    else:
        stage, max_abs_diff, mean_abs_diff = first_bad
        print(
            f"ORBIT_RUNTIME_PARITY_SGLANG_MEGATRON_ACTIVATION_FIRST_BAD threshold={bad_threshold} "
            f"stage={stage} max_abs_diff={max_abs_diff} mean_abs_diff={mean_abs_diff}",
            flush=True,
        )

    return True


def _clone_weight_for_compare(tensor: torch.Tensor) -> torch.Tensor:
    if hasattr(tensor, "detach"):
        tensor = tensor.detach()
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    return tensor.to(device="cpu", dtype=torch.float32).clone()


def _get_scale_inv_for_compare(module: Any) -> torch.Tensor | None:
    scale_inv = _get_weight_compare_attr(module, "weight_scale_inv")
    if scale_inv is not None:
        return _clone_weight_for_compare(scale_inv)

    scale = _get_weight_compare_attr(module, "weight_scale")
    if scale is not None:
        return _clone_weight_for_compare(1.0 / scale.float())

    return None


def _get_quantizer_scale_for_compare(module: Any) -> torch.Tensor | None:
    quantizer = _get_weight_compare_attr(module, "weight_quantizer")
    scale = getattr(quantizer, "_scale", None)
    if isinstance(scale, torch.Tensor):
        return _clone_weight_for_compare(scale)
    return None


def _unwrap_megatron_module_for_weight_compare(module: Any) -> Any:
    while hasattr(module, "module") and not hasattr(module, "decoder"):
        module = module.module
    return module


def _iter_weight_compare_module_views(module: Any):
    seen: set[int] = set()
    pending = [module]
    while pending:
        current = pending.pop(0)
        if current is None:
            continue
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        yield current

        wrapped = getattr(current, "to_wrap", None)
        if wrapped is not None:
            pending.append(wrapped)

        nested = getattr(current, "module", None)
        if nested is not None and nested is not current:
            pending.append(nested)


def _get_weight_compare_attr(module: Any, attr: str) -> Any:
    for candidate in _iter_weight_compare_module_views(module):
        if hasattr(candidate, attr):
            return getattr(candidate, attr)
    return None


def _require_weight_compare_attr(module: Any, attr: str, *, context: str) -> Any:
    value = _get_weight_compare_attr(module, attr)
    if value is None:
        raise AttributeError(f"{type(module).__name__} is missing required attribute {attr!r} for {context}")
    return value


def _dequantize_fp8_weight_for_parity(
    weight: torch.Tensor,
    scale_inv: torch.Tensor,
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor | None:
    if weight.ndim != 2 or scale_inv.numel() == 0:
        return None

    if scale_inv.numel() == 1:
        return (weight.float() * scale_inv.float().item()).to(out_dtype)

    if scale_inv.ndim != 2:
        return None

    out_features, in_features = weight.shape
    scale_rows, scale_cols = scale_inv.shape
    if scale_rows <= 0 or scale_cols <= 0:
        return None

    if out_features % scale_rows == 0 and in_features % scale_cols == 0:
        block_rows = out_features // scale_rows
        block_cols = in_features // scale_cols
        scaled = weight.float().reshape(scale_rows, block_rows, scale_cols, block_cols)
        scaled = scaled * scale_inv.float().to(weight.device).unsqueeze(1).unsqueeze(3)
        return scaled.reshape(out_features, in_features).to(out_dtype)

    block_rows = (out_features + scale_rows - 1) // scale_rows
    block_cols = (in_features + scale_cols - 1) // scale_cols
    expanded_scale = scale_inv.float().to(weight.device)
    expanded_scale = expanded_scale.repeat_interleave(block_rows, dim=0)
    expanded_scale = expanded_scale.repeat_interleave(block_cols, dim=1)
    expanded_scale = expanded_scale[:out_features, :in_features]
    return (weight.float() * expanded_scale).to(out_dtype)


def _looks_like_packed_fp8_weight_for_parity(weight: torch.Tensor, scale_inv: torch.Tensor) -> bool:
    if weight.numel() == 0 or scale_inv.numel() == 0:
        return False
    try:
        weight_max = float(weight.detach().abs().max().cpu())
        scale_max = float(scale_inv.detach().abs().max().cpu())
    except Exception:
        return True
    return weight_max > 8.0 and 0.0 < scale_max < 1.0


class _Fp8WeightDequantParityHandle:
    def __init__(self, handles: list[Any], active_original_weights: dict[int, tuple[Any, torch.Tensor]]):
        self._handles = handles
        self._active_original_weights = active_original_weights

    def remove(self) -> None:
        for module, original_weight_data in list(self._active_original_weights.values()):
            try:
                module.weight.data = original_weight_data
            except Exception:
                pass
        self._active_original_weights.clear()
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass


def _attach_fp8_weight_dequant_parity_hooks(root: Any) -> _Fp8WeightDequantParityHandle:
    handles: list[Any] = []
    active_original_weights: dict[int, tuple[Any, torch.Tensor]] = {}
    seen_weight_ids: set[int] = set()
    attached_names: list[str] = []

    def _restore_module_weight(module: Any) -> None:
        original = active_original_weights.pop(id(module), None)
        if original is not None:
            _, original_weight_data = original
            module.weight.data = original_weight_data

    def _pre_hook(module: Any, _args: Any) -> None:
        weight = getattr(module, "weight", None)
        scale_inv = getattr(module, "weight_scale_inv", None)
        if not isinstance(weight, torch.Tensor) or not isinstance(scale_inv, torch.Tensor):
            return
        if not _looks_like_packed_fp8_weight_for_parity(weight, scale_inv):
            return

        dequantized = _dequantize_fp8_weight_for_parity(
            weight,
            scale_inv.to(device=weight.device),
            out_dtype=weight.dtype,
        )
        if dequantized is None:
            return

        active_original_weights[id(module)] = (module, weight.data)
        module.weight.data = dequantized.contiguous()

    def _post_hook(module: Any, _args: Any, _output: Any) -> None:
        _restore_module_weight(module)

    for name, module in root.named_modules():
        weight = getattr(module, "weight", None)
        scale_inv = getattr(module, "weight_scale_inv", None)
        if not isinstance(weight, torch.Tensor) or not isinstance(scale_inv, torch.Tensor):
            continue
        if weight.ndim != 2 or scale_inv.numel() == 0:
            continue
        weight_id = id(weight)
        if weight_id in seen_weight_ids:
            continue
        seen_weight_ids.add(weight_id)
        handles.append(module.register_forward_pre_hook(_pre_hook))
        handles.append(module.register_forward_hook(_post_hook))
        attached_names.append(name)

    print(
        "VERL_FP8_WEIGHT_DEQUANT_PARITY_HOOKS "
        f"attached={len(attached_names)} "
        f"sample={attached_names[:8]}",
        flush=True,
    )
    return _Fp8WeightDequantParityHandle(handles, active_original_weights)


def _is_float8_dtype(dtype: torch.dtype) -> bool:
    return dtype in {
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
    }


def _fp8_activation_qdq_per_token_group_for_parity(
    tensor: torch.Tensor,
    *,
    group_size: int = 128,
) -> torch.Tensor:
    """Simulate SGLang's dynamic per-token-group FP8 activation quantization."""
    if not tensor.is_floating_point():
        return tensor
    hidden_size = tensor.shape[-1]
    if hidden_size % group_size != 0:
        raise ValueError(
            f"FP8 activation parity quantization requires hidden size divisible by {group_size}, "
            f"got {hidden_size}."
        )

    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        return tensor

    original_dtype = tensor.dtype
    original_shape = tensor.shape
    grouped = tensor.reshape(-1, hidden_size // group_size, group_size).float()
    max_abs = grouped.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(max_abs, min=1e-10) / float(torch.finfo(fp8_dtype).max)
    quantized = torch.clamp(grouped / scale, min=-float(torch.finfo(fp8_dtype).max), max=float(torch.finfo(fp8_dtype).max))
    dequantized = quantized.to(fp8_dtype).to(torch.float32) * scale
    dequantized = dequantized.reshape(original_shape).to(original_dtype)
    return tensor + (dequantized - tensor).detach()


def _fp8_quantize_activation_per_token_group_for_parity(
    tensor: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not tensor.is_floating_point():
        raise TypeError("FP8 activation quantization requires a floating-point tensor.")
    hidden_size = tensor.shape[-1]
    if hidden_size % group_size != 0:
        raise ValueError(
            f"FP8 activation parity quantization requires hidden size divisible by {group_size}, "
            f"got {hidden_size}."
        )

    fp8_dtype = getattr(torch, "float8_e4m3fn", None)
    if fp8_dtype is None:
        raise RuntimeError("torch.float8_e4m3fn is required for FP8 W8A8 parity.")

    tensor_2d = tensor.contiguous().reshape(-1, hidden_size)
    grouped = tensor_2d.float().reshape(tensor_2d.shape[0], hidden_size // group_size, group_size)
    fp8_info = torch.finfo(fp8_dtype)
    scale = grouped.abs().amax(dim=-1).clamp_min(1e-10) / float(fp8_info.max)
    quantized = torch.clamp(
        grouped / scale.unsqueeze(-1),
        float(fp8_info.min),
        float(fp8_info.max),
    ).to(fp8_dtype)
    return quantized.reshape_as(tensor_2d), scale


def _expand_fp8_weight_block_scales_for_parity(
    weight_scale_inv: torch.Tensor,
    *,
    out_features: int,
    block_n: int,
    k_block_index: int,
) -> torch.Tensor:
    if weight_scale_inv.ndim == 0 or weight_scale_inv.numel() == 1:
        return weight_scale_inv.reshape(1).float().expand(out_features)
    if weight_scale_inv.ndim != 2:
        raise ValueError(
            "FP8 W8A8 block-linear parity expects scalar or 2D weight scales, "
            f"got shape {tuple(weight_scale_inv.shape)}."
        )

    if k_block_index >= weight_scale_inv.shape[1]:
        raise ValueError(
            f"Weight scale is missing K block {k_block_index}; shape={tuple(weight_scale_inv.shape)}."
        )

    scale_column = weight_scale_inv[:, k_block_index].float()
    if scale_column.numel() == out_features:
        return scale_column

    expanded = scale_column.repeat_interleave(block_n)
    if expanded.numel() < out_features:
        raise ValueError(
            "Weight scale rows do not cover all output features for FP8 W8A8 block-linear parity: "
            f"scale_rows={scale_column.numel()} block_n={block_n} out_features={out_features}."
        )
    return expanded[:out_features]


def _fp8_w8a8_block_linear_for_parity(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    block_size: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """Approximate SGLang Triton W8A8 block-FP8 linear math for parity debugging."""

    if weight.ndim != 2:
        raise ValueError(f"FP8 W8A8 block-linear parity expects a 2D weight, got {tuple(weight.shape)}.")
    if not hidden.is_floating_point():
        raise TypeError("FP8 W8A8 block-linear parity expects floating-point activations.")

    block_n, block_k = block_size
    out_features, in_features = weight.shape
    if hidden.shape[-1] != in_features:
        raise ValueError(
            "FP8 W8A8 block-linear parity shape mismatch: "
            f"hidden last dim={hidden.shape[-1]} weight shape={tuple(weight.shape)}."
        )

    hidden_2d = hidden.contiguous().reshape(-1, in_features)
    q_hidden, activation_scale = _fp8_quantize_activation_per_token_group_for_parity(
        hidden_2d,
        group_size=block_k,
    )
    q_hidden_f32 = q_hidden.float()
    weight_f32 = weight.to(device=hidden.device).float()
    weight_scale_inv = weight_scale_inv.to(device=hidden.device)

    output = torch.zeros(
        (hidden_2d.shape[0], out_features),
        device=hidden.device,
        dtype=torch.float32,
    )
    num_k_blocks = (in_features + block_k - 1) // block_k
    for k_block_index in range(num_k_blocks):
        k_start = k_block_index * block_k
        k_end = min(k_start + block_k, in_features)
        partial = q_hidden_f32[:, k_start:k_end] @ weight_f32[:, k_start:k_end].T
        activation_block_scale = activation_scale[:, k_block_index].unsqueeze(1)
        weight_block_scale = _expand_fp8_weight_block_scales_for_parity(
            weight_scale_inv,
            out_features=out_features,
            block_n=block_n,
            k_block_index=k_block_index,
        ).unsqueeze(0)
        output += partial * activation_block_scale * weight_block_scale

    if bias is not None:
        output += bias.to(device=hidden.device, dtype=torch.float32)

    return output.to(dtype=hidden.dtype).reshape(*hidden.shape[:-1], out_features)


def _get_first_floating_tensor_arg(args: Any) -> torch.Tensor | None:
    for arg in args:
        if isinstance(arg, torch.Tensor) and arg.is_floating_point():
            return arg
    return None


def _replace_first_tensor_in_output(value: Any, replacement: torch.Tensor) -> Any:
    if isinstance(value, torch.Tensor):
        return replacement
    if isinstance(value, tuple):
        replaced = False
        updated = []
        for item in value:
            if not replaced and isinstance(item, torch.Tensor):
                updated.append(replacement)
                replaced = True
            else:
                updated.append(item)
        return tuple(updated)
    if isinstance(value, list):
        replaced = False
        updated = []
        for item in value:
            if not replaced and isinstance(item, torch.Tensor):
                updated.append(replacement)
                replaced = True
            else:
                updated.append(item)
        return updated
    return replacement


class _Fp8ActivationQdqParityHandle:
    def __init__(self, handles: list[Any]):
        self._handles = handles

    def remove(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass


def _attach_fp8_activation_qdq_parity_hooks(
    root: Any,
    *,
    group_size: int = 128,
) -> _Fp8ActivationQdqParityHandle:
    handles: list[Any] = []
    seen_modules: set[int] = set()
    attached_names: list[str] = []

    def _pre_hook(module: Any, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
        scale_inv = getattr(module, "weight_scale_inv", None)
        if not isinstance(scale_inv, torch.Tensor):
            return None

        updated_args = list(args)
        for index, arg in enumerate(updated_args):
            if isinstance(arg, torch.Tensor) and arg.is_floating_point():
                updated_args[index] = _fp8_activation_qdq_per_token_group_for_parity(arg, group_size=group_size)
                return tuple(updated_args)
        return None

    for name, module in list(root.named_modules()):
        weight = getattr(module, "weight", None)
        scale_inv = getattr(module, "weight_scale_inv", None)
        if not isinstance(weight, torch.Tensor) or not isinstance(scale_inv, torch.Tensor):
            continue
        if not _is_float8_dtype(weight.dtype):
            continue
        module_id = id(module)
        if module_id in seen_modules:
            continue
        seen_modules.add(module_id)
        handles.append(module.register_forward_pre_hook(_pre_hook))
        attached_names.append(name)

    print(
        "VERL_FP8_ACTIVATION_QDQ_PARITY_HOOKS "
        f"attached={len(attached_names)} group_size={group_size} "
        f"sample={attached_names[:8]}",
        flush=True,
    )
    return _Fp8ActivationQdqParityHandle(handles)


class _Fp8BlockLinearParityHandle:
    def __init__(
        self,
        handles: list[Any],
        active_original_weights: dict[int, tuple[Any, Any]],
        saved_forward_inputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]],
        original_quantizers: list[tuple[Any, str, Any]],
    ):
        self._handles = handles
        self._active_original_weights = active_original_weights
        self._saved_forward_inputs = saved_forward_inputs
        self._original_quantizers = original_quantizers

    def remove(self) -> None:
        for module, original_weight in list(self._active_original_weights.values()):
            try:
                _set_module_weight_for_nvfp4_parity(module, original_weight)
            except Exception:
                pass
        self._active_original_weights.clear()
        self._saved_forward_inputs.clear()
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        for module, attr, original_quantizer in reversed(self._original_quantizers):
            try:
                _set_module_attr_for_parity(module, attr, original_quantizer)
            except Exception:
                pass


def _attach_fp8_block_linear_parity_hooks(
    root: Any,
    *,
    block_size: tuple[int, int] = (128, 128),
) -> _Fp8BlockLinearParityHandle:
    handles: list[Any] = []
    active_original_weights: dict[int, tuple[Any, Any]] = {}
    saved_forward_inputs: dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]] = {}
    original_quantizers: list[tuple[Any, str, Any]] = []
    seen_modules: set[int] = set()
    attached_names: list[str] = []

    def _restore_module_weight(module: Any) -> None:
        original = active_original_weights.pop(id(module), None)
        if original is not None:
            _, original_weight = original
            _set_module_weight_for_nvfp4_parity(module, original_weight)

    def _make_dense_weight_like_original(original_weight: Any, dense_weight: torch.Tensor) -> Any:
        if isinstance(original_weight, torch.nn.Parameter):
            requires_grad = bool(getattr(original_weight, "requires_grad", False)) and dense_weight.is_floating_point()
            return torch.nn.Parameter(dense_weight, requires_grad=requires_grad)
        return dense_weight

    def _pre_hook(module: Any, args: tuple[Any, ...]) -> None:
        hidden = _get_first_floating_tensor_arg(args)
        weight = getattr(module, "weight", None)
        scale_inv = getattr(module, "weight_scale_inv", None)
        if hidden is None or not isinstance(weight, torch.Tensor) or not isinstance(scale_inv, torch.Tensor):
            return
        if not _is_float8_dtype(weight.dtype):
            return

        original_weight = weight
        active_original_weights[id(module)] = (module, original_weight)
        saved_forward_inputs[id(module)] = (
            hidden,
            original_weight.detach(),
            scale_inv.detach(),
            getattr(module, "bias", None),
        )
        dequantized = _dequantize_fp8_weight_for_parity(
            original_weight.to(device=hidden.device),
            scale_inv.to(device=hidden.device),
            out_dtype=hidden.dtype,
        )
        if dequantized is None:
            dequantized = original_weight.to(device=hidden.device, dtype=hidden.dtype)
        _set_module_weight_for_nvfp4_parity(
            module,
            _make_dense_weight_like_original(original_weight, dequantized.contiguous()),
        )

    def _post_hook(module: Any, _args: Any, output: Any) -> Any:
        saved = saved_forward_inputs.pop(id(module), None)
        try:
            if saved is None:
                return output
            hidden, weight, scale_inv, bias = saved
            replacement = _fp8_w8a8_block_linear_for_parity(
                hidden,
                weight.to(device=hidden.device),
                scale_inv.to(device=hidden.device),
                bias=bias,
                block_size=block_size,
            )
            return _replace_first_tensor_in_output(output, replacement)
        finally:
            _restore_module_weight(module)

    for name, module in list(root.named_modules()):
        weight = getattr(module, "weight", None)
        scale_inv = getattr(module, "weight_scale_inv", None)
        quantizer = getattr(module, "weight_quantizer", None)
        if not isinstance(weight, torch.Tensor) or not isinstance(scale_inv, torch.Tensor):
            continue
        if not _is_float8_dtype(weight.dtype):
            continue
        module_id = id(module)
        if module_id in seen_modules:
            continue
        seen_modules.add(module_id)
        if callable(quantizer):
            original_quantizers.append((module, "weight_quantizer", quantizer))
            _set_module_attr_for_parity(module, "weight_quantizer", torch.nn.Identity())
        handles.append(module.register_forward_pre_hook(_pre_hook))
        handles.append(module.register_forward_hook(_post_hook))
        attached_names.append(name)

    print(
        "VERL_FP8_BLOCK_LINEAR_PARITY_HOOKS "
        f"attached={len(attached_names)} block_size={block_size} "
        f"sample={attached_names[:8]}",
        flush=True,
    )
    return _Fp8BlockLinearParityHandle(
        handles,
        active_original_weights,
        saved_forward_inputs,
        original_quantizers,
    )


def _set_module_attr_for_parity(module: Any, attr: str, value: Any) -> None:
    modules = getattr(module, "_modules", None)
    if isinstance(modules, dict) and attr in modules and not isinstance(value, torch.nn.Module) and value is not None:
        modules.pop(attr, None)
        object.__setattr__(module, attr, value)
        return
    setattr(module, attr, value)


class _ModelOptFp8WeightQuantizerBypassHandle:
    def __init__(
        self,
        handles: list[Any],
        active_original_weights: dict[int, tuple[Any, Any]],
        original_quantizers: list[tuple[Any, str, Any]],
    ):
        self._handles = handles
        self._active_original_weights = active_original_weights
        self._original_quantizers = original_quantizers

    def remove(self) -> None:
        for module, original_weight in list(self._active_original_weights.values()):
            try:
                _set_module_weight_for_nvfp4_parity(module, original_weight)
            except Exception:
                pass
        self._active_original_weights.clear()
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        for module, attr, original_quantizer in reversed(self._original_quantizers):
            try:
                _set_module_attr_for_parity(module, attr, original_quantizer)
            except Exception:
                pass


def _attach_modelopt_fp8_weight_quantizer_bypass_hooks(root: Any) -> _ModelOptFp8WeightQuantizerBypassHandle:
    handles: list[Any] = []
    active_original_weights: dict[int, tuple[Any, Any]] = {}
    original_quantizers: list[tuple[Any, str, Any]] = []
    seen_modules: set[int] = set()
    attached_names: list[str] = []

    def _restore_module_weight(module: Any) -> None:
        original = active_original_weights.pop(id(module), None)
        if original is not None:
            _, original_weight = original
            _set_module_weight_for_nvfp4_parity(module, original_weight)

    def _make_dense_weight_like_original(original_weight: Any, dense_weight: torch.Tensor) -> Any:
        if isinstance(original_weight, torch.nn.Parameter):
            requires_grad = bool(getattr(original_weight, "requires_grad", False)) and dense_weight.is_floating_point()
            return torch.nn.Parameter(dense_weight, requires_grad=requires_grad)
        return dense_weight

    def _pre_hook(module: Any, args: Any) -> None:
        weight = getattr(module, "weight", None)
        scale_inv = getattr(module, "weight_scale_inv", None)
        if not isinstance(weight, torch.Tensor):
            return
        if not _is_float8_dtype(weight.dtype):
            return

        device, out_dtype = _first_tensor_arg_device_and_dtype(args, weight)
        dequantized = None
        if isinstance(scale_inv, torch.Tensor):
            dequantized = _dequantize_fp8_weight_for_parity(
                weight.to(device=device),
                scale_inv.to(device=device),
                out_dtype=out_dtype,
            )
        if dequantized is None:
            dequantized = weight.to(device=device, dtype=out_dtype)

        active_original_weights[id(module)] = (module, weight)
        _set_module_weight_for_nvfp4_parity(
            module,
            _make_dense_weight_like_original(weight, dequantized.contiguous()),
        )

    def _post_hook(module: Any, _args: Any, _output: Any) -> None:
        _restore_module_weight(module)

    for name, module in list(root.named_modules()):
        weight = getattr(module, "weight", None)
        quantizer = getattr(module, "weight_quantizer", None)
        if not isinstance(weight, torch.Tensor):
            continue
        if not _is_float8_dtype(weight.dtype) or not callable(quantizer):
            continue
        module_id = id(module)
        if module_id in seen_modules:
            continue
        seen_modules.add(module_id)
        original_quantizers.append((module, "weight_quantizer", quantizer))
        _set_module_attr_for_parity(module, "weight_quantizer", torch.nn.Identity())
        handles.append(module.register_forward_pre_hook(_pre_hook))
        handles.append(module.register_forward_hook(_post_hook))
        attached_names.append(name)

    print(
        "VERL_MODELOPT_FP8_WEIGHT_QUANTIZER_BYPASS_HOOKS "
        f"attached={len(attached_names)} "
        f"sample={attached_names[:8]}",
        flush=True,
    )
    return _ModelOptFp8WeightQuantizerBypassHandle(handles, active_original_weights, original_quantizers)


def _move_parity_tensor_to_weight_device(value: Any, weight: torch.Tensor) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=weight.device)
    return value


def _nvfp4_weight_dequant_dtype_for_parity(weight: torch.Tensor) -> torch.dtype | None:
    metadata = getattr(weight, "metadata", None)
    if isinstance(metadata, dict) and isinstance(metadata.get("dtype"), torch.dtype):
        return metadata["dtype"]
    return None


def _looks_like_nvfp4_qtensor_for_parity(qtensor: Any) -> bool:
    qtensor_name = type(qtensor).__name__.lower()
    metadata = getattr(qtensor, "metadata", None)
    metadata_class = None
    if isinstance(metadata, dict):
        metadata_class = metadata.get("qtensor_class")
    metadata_name = getattr(metadata_class, "__name__", "").lower()
    return "nvfp4" in qtensor_name or "nvfp4" in metadata_name


def _dequantize_nvfp4_weight_for_parity(
    weight: torch.Tensor,
    quantizer: Any,
) -> torch.Tensor | None:
    get_qtensor = getattr(weight, "get_qtensor", None)
    if not callable(get_qtensor):
        return None
    scale = getattr(quantizer, "_scale", None)
    if not isinstance(scale, torch.Tensor):
        return None

    try:
        qtensor = get_qtensor()
    except Exception:
        return None
    if not _looks_like_nvfp4_qtensor_for_parity(qtensor):
        return None

    dequantize = getattr(qtensor, "dequantize", None)
    if not callable(dequantize):
        return None

    kwargs = {
        "scale": _move_parity_tensor_to_weight_device(scale, weight),
        "block_sizes": {-1: 16},
        "fast": True,
    }
    double_scale = getattr(quantizer, "_double_scale", None)
    if double_scale is not None:
        kwargs["double_scale"] = _move_parity_tensor_to_weight_device(double_scale, weight)

    try:
        dequantized = dequantize(**kwargs)
    except TypeError:
        kwargs.pop("fast", None)
        try:
            dequantized = dequantize(**kwargs)
        except Exception:
            return None
    except Exception:
        return None

    if isinstance(dequantized, (tuple, list)):
        if not dequantized:
            return None
        dequantized = dequantized[0]
    if not isinstance(dequantized, torch.Tensor):
        return None

    out_dtype = _nvfp4_weight_dequant_dtype_for_parity(weight)
    if out_dtype is not None:
        dequantized = dequantized.to(dtype=out_dtype)
    return dequantized.to(device=weight.device)


def _set_module_weight_for_nvfp4_parity(owner: Any, weight_value: Any) -> None:
    parameters = getattr(owner, "_parameters", None)
    if isinstance(parameters, dict) and "weight" in parameters:
        dict.__setitem__(parameters, "weight", weight_value)
        return

    buffers = getattr(owner, "_buffers", None)
    if isinstance(buffers, dict) and "weight" in buffers:
        dict.__setitem__(buffers, "weight", weight_value)
        return

    object.__setattr__(owner, "weight", weight_value)


def _disable_quantizer_for_nvfp4_parity(quantizer: Any) -> None:
    disable = getattr(quantizer, "disable", None)
    if callable(disable):
        disable()
    elif hasattr(quantizer, "_disabled"):
        setattr(quantizer, "_disabled", True)


def _restore_quantizer_disabled_state_for_nvfp4_parity(quantizer: Any, disabled_state: Any) -> None:
    if disabled_state is not None:
        try:
            setattr(quantizer, "_disabled", disabled_state)
        except Exception:
            pass


class _Nvfp4WeightDequantParityHandle:
    def __init__(self, handles: list[Any], active_original_weights: dict[int, tuple[Any, Any, Any, Any]]):
        self._handles = handles
        self._active_original_weights = active_original_weights

    def remove(self) -> None:
        for owner, original_weight, quantizer, disabled_state in list(self._active_original_weights.values()):
            try:
                _set_module_weight_for_nvfp4_parity(owner, original_weight)
                _restore_quantizer_disabled_state_for_nvfp4_parity(quantizer, disabled_state)
            except Exception:
                pass
        self._active_original_weights.clear()
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass


def _find_nvfp4_weight_owner_for_parity(module: Any) -> tuple[Any, torch.Tensor, Any] | None:
    for candidate in _iter_weight_compare_module_views(module):
        weight = getattr(candidate, "weight", None)
        quantizer = getattr(candidate, "weight_quantizer", None)
        if not isinstance(weight, torch.Tensor) or quantizer is None:
            continue
        get_qtensor = getattr(weight, "get_qtensor", None)
        if not callable(get_qtensor):
            continue
        try:
            qtensor = get_qtensor()
        except Exception:
            continue
        if _looks_like_nvfp4_qtensor_for_parity(qtensor):
            return candidate, weight, quantizer
    return None


def _attach_nvfp4_weight_dequant_parity_hooks(root: Any) -> _Nvfp4WeightDequantParityHandle:
    handles: list[Any] = []
    active_original_weights: dict[int, tuple[Any, Any, Any, Any]] = {}
    hook_owner_by_module_id: dict[int, Any] = {}
    seen_owner_ids: set[int] = set()
    attached_names: list[str] = []

    def _restore_owner_weight(owner: Any) -> None:
        original = active_original_weights.pop(id(owner), None)
        if original is not None:
            _, original_weight, quantizer, disabled_state = original
            _set_module_weight_for_nvfp4_parity(owner, original_weight)
            _restore_quantizer_disabled_state_for_nvfp4_parity(quantizer, disabled_state)

    def _make_dense_weight_like_original(original_weight: Any, dense_weight: torch.Tensor) -> Any:
        if isinstance(original_weight, torch.nn.Parameter):
            requires_grad = bool(getattr(original_weight, "requires_grad", False)) and dense_weight.is_floating_point()
            return torch.nn.Parameter(dense_weight, requires_grad=requires_grad)
        return dense_weight

    def _pre_hook(module: Any, _args: Any) -> None:
        owner_info = _find_nvfp4_weight_owner_for_parity(module)
        if owner_info is None:
            return
        owner, weight, quantizer = owner_info
        dequantized = _dequantize_nvfp4_weight_for_parity(weight, quantizer)
        if dequantized is None:
            return

        original_weight = getattr(owner, "weight")
        disabled_state = getattr(quantizer, "_disabled", None)
        active_original_weights[id(owner)] = (owner, original_weight, quantizer, disabled_state)
        _disable_quantizer_for_nvfp4_parity(quantizer)
        _set_module_weight_for_nvfp4_parity(
            owner,
            _make_dense_weight_like_original(original_weight, dequantized.contiguous()),
        )

    def _post_hook(module: Any, _args: Any, _output: Any) -> None:
        owner = hook_owner_by_module_id.get(id(module))
        if owner is None:
            owner_info = _find_nvfp4_weight_owner_for_parity(module)
            if owner_info is None:
                return
            owner, _, _ = owner_info
        _restore_owner_weight(owner)

    for name, module in root.named_modules():
        owner_info = _find_nvfp4_weight_owner_for_parity(module)
        if owner_info is None:
            continue
        owner, _, _ = owner_info
        owner_id = id(owner)
        if owner_id in seen_owner_ids:
            continue
        seen_owner_ids.add(owner_id)
        hook_owner_by_module_id[id(module)] = owner
        handles.append(module.register_forward_pre_hook(_pre_hook))
        handles.append(module.register_forward_hook(_post_hook))
        attached_names.append(name)

    print(
        "VERL_NVFP4_WEIGHT_DEQUANT_PARITY_HOOKS "
        f"attached={len(attached_names)} "
        f"sample={attached_names[:8]}",
        flush=True,
    )
    return _Nvfp4WeightDequantParityHandle(handles, active_original_weights)


def _first_tensor_arg_device_and_dtype(args: Any, fallback: torch.Tensor) -> tuple[torch.device, torch.dtype]:
    for arg in args:
        if isinstance(arg, torch.Tensor):
            dtype = arg.dtype if arg.is_floating_point() else fallback.dtype
            return arg.device, dtype
    return fallback.device, fallback.dtype


def _dequantize_int4_weight_for_parity(
    weight_packed: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    device: torch.device,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    packed = weight_packed.to(device=device)
    scale = weight_scale.to(device=device)
    local_out, local_packed_in = packed.shape
    local_in = local_packed_in * 8

    shifts = torch.arange(8, device=device, dtype=torch.int32) * 4
    unpacked = ((packed.unsqueeze(-1).to(torch.int32) >> shifts) & 0xF).float()
    unpacked = unpacked.reshape(local_out, local_in) - 8

    scale = scale.float()
    if scale.ndim == 1:
        scale = scale.view(local_out, -1)
    else:
        scale = scale.reshape(local_out, -1)

    groups = max(1, scale.shape[1])
    elements_per_group = max(1, local_in // groups)
    expanded_scale = scale.repeat_interleave(elements_per_group, dim=1)
    if expanded_scale.shape[1] < local_in:
        expanded_scale = torch.nn.functional.pad(
            expanded_scale,
            (0, local_in - expanded_scale.shape[1]),
            value=expanded_scale[:, -1:].mean(),
        )
    expanded_scale = expanded_scale[:, :local_in]
    return (unpacked * expanded_scale).to(dtype=out_dtype)


class _Int4WeightDequantParityHandle:
    def __init__(self, handles: list[Any], active_original_weights: dict[int, tuple[Any, Any]]):
        self._handles = handles
        self._active_original_weights = active_original_weights

    def remove(self) -> None:
        for module, original_weight in list(self._active_original_weights.values()):
            try:
                _set_module_weight_for_nvfp4_parity(module, original_weight)
            except Exception:
                pass
        self._active_original_weights.clear()
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass


def _attach_int4_weight_dequant_parity_hooks(root: Any) -> _Int4WeightDequantParityHandle:
    handles: list[Any] = []
    active_original_weights: dict[int, tuple[Any, Any]] = {}
    seen_modules: set[int] = set()
    attached_names: list[str] = []

    def _restore_module_weight(module: Any) -> None:
        original = active_original_weights.pop(id(module), None)
        if original is not None:
            _, original_weight = original
            _set_module_weight_for_nvfp4_parity(module, original_weight)

    def _make_dense_weight_like_original(original_weight: Any, dense_weight: torch.Tensor) -> Any:
        if isinstance(original_weight, torch.nn.Parameter):
            requires_grad = bool(getattr(original_weight, "requires_grad", False)) and dense_weight.is_floating_point()
            return torch.nn.Parameter(dense_weight, requires_grad=requires_grad)
        return dense_weight

    def _pre_hook(module: Any, args: Any) -> None:
        weight = getattr(module, "weight", None)
        packed = getattr(module, "weight_packed", None)
        scale = getattr(module, "weight_scale", None)
        if not isinstance(weight, torch.Tensor) or not isinstance(packed, torch.Tensor) or not isinstance(scale, torch.Tensor):
            return
        if packed.ndim != 2 or scale.numel() == 0:
            return

        device, out_dtype = _first_tensor_arg_device_and_dtype(args, weight)
        dequantized = _dequantize_int4_weight_for_parity(
            packed,
            scale,
            device=device,
            out_dtype=out_dtype,
        )
        active_original_weights[id(module)] = (module, weight)
        _set_module_weight_for_nvfp4_parity(
            module,
            _make_dense_weight_like_original(weight, dequantized.contiguous()),
        )

    def _post_hook(module: Any, _args: Any, _output: Any) -> None:
        _restore_module_weight(module)

    for name, module in root.named_modules():
        weight = getattr(module, "weight", None)
        packed = getattr(module, "weight_packed", None)
        scale = getattr(module, "weight_scale", None)
        shape = getattr(module, "weight_shape", None)
        if not isinstance(weight, torch.Tensor):
            continue
        if not isinstance(packed, torch.Tensor) or not isinstance(scale, torch.Tensor) or not isinstance(shape, torch.Tensor):
            continue
        module_id = id(module)
        if module_id in seen_modules:
            continue
        seen_modules.add(module_id)
        handles.append(module.register_forward_pre_hook(_pre_hook))
        handles.append(module.register_forward_hook(_post_hook))
        attached_names.append(name)

    print(
        "VERL_INT4_WEIGHT_DEQUANT_PARITY_HOOKS "
        f"attached={len(attached_names)} "
        f"sample={attached_names[:8]}",
        flush=True,
    )
    return _Int4WeightDequantParityHandle(handles, active_original_weights)


def _resolve_layernorm_weight_for_hf_compare(
    *,
    layer: Any,
    linear_module: Any,
    linear_context: str,
    fallback_module_attr: str,
    fallback_context: str,
) -> Any:
    weight = _get_weight_compare_attr(linear_module, "layer_norm_weight")
    if weight is not None:
        return weight

    fallback_module = getattr(layer, fallback_module_attr, None)
    if fallback_module is None:
        raise AttributeError(
            f"{type(linear_module).__name__} is missing required attribute 'layer_norm_weight' for {linear_context}"
        )

    return _require_weight_compare_attr(fallback_module, "weight", context=fallback_context)


def _split_packed_qkv_for_hf_compare(
    packed_qkv: torch.Tensor,
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    num_query_groups: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv_denominator = num_attention_heads + 2 * num_key_value_heads
    if qkv_denominator <= 0 or packed_qkv.shape[0] % qkv_denominator != 0:
        raise ValueError(
            "Cannot infer Megatron QKV head units from "
            f"qkv_rows={packed_qkv.shape[0]}, num_attention_heads={num_attention_heads}, "
            f"num_key_value_heads={num_key_value_heads}."
        )

    units_per_head = packed_qkv.shape[0] // qkv_denominator
    q_size = units_per_head * num_attention_heads
    kv_size = units_per_head * num_key_value_heads
    q_size_chunk = q_size // num_query_groups
    kv_size_chunk = kv_size // num_query_groups

    q_parts: list[torch.Tensor] = []
    k_parts: list[torch.Tensor] = []
    v_parts: list[torch.Tensor] = []
    for qkv_chunk in packed_qkv.chunk(num_query_groups, dim=0):
        q_parts.append(qkv_chunk[:q_size_chunk])
        k_parts.append(qkv_chunk[q_size_chunk : q_size_chunk + kv_size_chunk])
        v_parts.append(qkv_chunk[q_size_chunk + kv_size_chunk : q_size_chunk + 2 * kv_size_chunk])

    return torch.cat(q_parts, dim=0), torch.cat(k_parts, dim=0), torch.cat(v_parts, dim=0)


def _resolve_weight_compare_layer_indices(raw: str | None, *, layer_count: int) -> list[int]:
    if layer_count <= 0:
        return []

    raw = raw or "0"
    indices: list[int] = []
    for item in raw.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token == "all":
            candidates = range(layer_count)
        elif token == "first":
            candidates = (0,)
        elif token in {"middle", "mid"}:
            candidates = (layer_count // 2,)
        elif token == "last":
            candidates = (layer_count - 1,)
        else:
            value = int(token)
            if value < 0:
                value = layer_count + value
            candidates = (value,)

        for idx in candidates:
            if idx < 0 or idx >= layer_count:
                raise ValueError(f"Weight-compare layer index {idx} is outside 0..{layer_count - 1}.")
            if idx not in indices:
                indices.append(idx)

    return indices or [0]


def _extract_megatron_global_weights_for_hf_compare(module: Any) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}

    embedding = getattr(module, "embedding", None)
    word_embeddings = getattr(embedding, "word_embeddings", None)
    if word_embeddings is not None and getattr(word_embeddings, "weight", None) is not None:
        weights["model.embed_tokens.weight"] = _clone_weight_for_compare(word_embeddings.weight)

    output_layer = getattr(module, "output_layer", None)
    output_weight = getattr(output_layer, "weight", None)
    if output_weight is None and hasattr(module, "shared_embedding_or_output_weight"):
        output_weight = module.shared_embedding_or_output_weight()
    if output_weight is not None:
        weights["lm_head.weight"] = _clone_weight_for_compare(output_weight)

    return weights


def _extract_megatron_dense_layer_weights_for_hf_compare(module: Any, layer_idx: int) -> dict[str, torch.Tensor]:
    module = _unwrap_megatron_module_for_weight_compare(module)
    layer = module.decoder.layers[layer_idx]
    config = module.config

    qkv_weight = _require_weight_compare_attr(
        layer.self_attention.linear_qkv,
        "weight",
        context=f"layer {layer_idx} self_attention.linear_qkv",
    )
    num_attention_heads = config.num_attention_heads
    num_query_groups = getattr(config, "num_query_groups", getattr(config, "num_key_value_heads", None))
    if num_query_groups is None or num_query_groups <= 0:
        raise ValueError("Megatron weight compare requires a positive num_query_groups or num_key_value_heads.")
    num_key_value_heads = getattr(config, "num_key_value_heads", num_query_groups)

    q_weight, k_weight, v_weight = _split_packed_qkv_for_hf_compare(
        qkv_weight,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        num_query_groups=num_query_groups,
    )

    fc1_weight = _require_weight_compare_attr(layer.mlp.linear_fc1, "weight", context=f"layer {layer_idx} mlp.linear_fc1")
    intermediate_size = getattr(config, "intermediate_size", getattr(config, "ffn_hidden_size", None))
    if intermediate_size is None:
        raise ValueError("Megatron weight compare requires intermediate_size or ffn_hidden_size.")

    weights = {
        f"model.layers.{layer_idx}.input_layernorm.weight": _clone_weight_for_compare(
            _resolve_layernorm_weight_for_hf_compare(
                layer=layer,
                linear_module=layer.self_attention.linear_qkv,
                linear_context=f"layer {layer_idx} self_attention.linear_qkv",
                fallback_module_attr="input_layernorm",
                fallback_context=f"layer {layer_idx} input_layernorm",
            )
        ),
        f"model.layers.{layer_idx}.self_attn.q_proj.weight": _clone_weight_for_compare(q_weight),
        f"model.layers.{layer_idx}.self_attn.k_proj.weight": _clone_weight_for_compare(k_weight),
        f"model.layers.{layer_idx}.self_attn.v_proj.weight": _clone_weight_for_compare(v_weight),
        f"model.layers.{layer_idx}.self_attn.o_proj.weight": _clone_weight_for_compare(
            _require_weight_compare_attr(
                layer.self_attention.linear_proj,
                "weight",
                context=f"layer {layer_idx} self_attention.linear_proj",
            )
        ),
        f"model.layers.{layer_idx}.post_attention_layernorm.weight": _clone_weight_for_compare(
            _resolve_layernorm_weight_for_hf_compare(
                layer=layer,
                linear_module=layer.mlp.linear_fc1,
                linear_context=f"layer {layer_idx} mlp.linear_fc1",
                fallback_module_attr="pre_mlp_layernorm",
                fallback_context=f"layer {layer_idx} pre_mlp_layernorm",
            )
        ),
        f"model.layers.{layer_idx}.mlp.gate_proj.weight": _clone_weight_for_compare(fc1_weight[:intermediate_size]),
        f"model.layers.{layer_idx}.mlp.up_proj.weight": _clone_weight_for_compare(fc1_weight[intermediate_size:]),
        f"model.layers.{layer_idx}.mlp.down_proj.weight": _clone_weight_for_compare(
            _require_weight_compare_attr(layer.mlp.linear_fc2, "weight", context=f"layer {layer_idx} mlp.linear_fc2")
        ),
    }
    weights.update(_extract_megatron_global_weights_for_hf_compare(module))

    q_layernorm = getattr(layer.self_attention, "q_layernorm", None)
    if q_layernorm is not None and getattr(q_layernorm, "weight", None) is not None:
        weights[f"model.layers.{layer_idx}.self_attn.q_norm.weight"] = _clone_weight_for_compare(q_layernorm.weight)

    k_layernorm = getattr(layer.self_attention, "k_layernorm", None)
    if k_layernorm is not None and getattr(k_layernorm, "weight", None) is not None:
        weights[f"model.layers.{layer_idx}.self_attn.k_norm.weight"] = _clone_weight_for_compare(k_layernorm.weight)

    final_layernorm = getattr(module.decoder, "final_layernorm", None)
    if final_layernorm is not None and getattr(final_layernorm, "weight", None) is not None:
        weights["model.norm.weight"] = _clone_weight_for_compare(final_layernorm.weight)

    qkv_scale_inv = _get_scale_inv_for_compare(layer.self_attention.linear_qkv)
    if qkv_scale_inv is not None:
        q_scale, k_scale, v_scale = _split_packed_qkv_for_hf_compare(
            qkv_scale_inv,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            num_query_groups=num_query_groups,
        )
        weights.update(
            {
                f"model.layers.{layer_idx}.self_attn.q_proj.weight_scale_inv": q_scale,
                f"model.layers.{layer_idx}.self_attn.k_proj.weight_scale_inv": k_scale,
                f"model.layers.{layer_idx}.self_attn.v_proj.weight_scale_inv": v_scale,
            }
        )

    linear_proj_scale_inv = _get_scale_inv_for_compare(layer.self_attention.linear_proj)
    if linear_proj_scale_inv is not None:
        weights[f"model.layers.{layer_idx}.self_attn.o_proj.weight_scale_inv"] = linear_proj_scale_inv

    fc1_scale_inv = _get_scale_inv_for_compare(layer.mlp.linear_fc1)
    if fc1_scale_inv is not None:
        fc1_scale_split = fc1_scale_inv.shape[0] // 2
        weights[f"model.layers.{layer_idx}.mlp.gate_proj.weight_scale_inv"] = fc1_scale_inv[:fc1_scale_split]
        weights[f"model.layers.{layer_idx}.mlp.up_proj.weight_scale_inv"] = fc1_scale_inv[fc1_scale_split:]

    fc2_scale_inv = _get_scale_inv_for_compare(layer.mlp.linear_fc2)
    if fc2_scale_inv is not None:
        weights[f"model.layers.{layer_idx}.mlp.down_proj.weight_scale_inv"] = fc2_scale_inv

    qkv_scale = _get_quantizer_scale_for_compare(layer.self_attention.linear_qkv)
    if qkv_scale is not None:
        q_scale, k_scale, v_scale = _split_packed_qkv_for_hf_compare(
            qkv_scale,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            num_query_groups=num_query_groups,
        )
        weights.update(
            {
                f"model.layers.{layer_idx}.self_attn.q_proj.weight_scale": q_scale,
                f"model.layers.{layer_idx}.self_attn.k_proj.weight_scale": k_scale,
                f"model.layers.{layer_idx}.self_attn.v_proj.weight_scale": v_scale,
            }
        )

    linear_proj_scale = _get_quantizer_scale_for_compare(layer.self_attention.linear_proj)
    if linear_proj_scale is not None:
        weights[f"model.layers.{layer_idx}.self_attn.o_proj.weight_scale"] = linear_proj_scale

    fc1_scale = _get_quantizer_scale_for_compare(layer.mlp.linear_fc1)
    if fc1_scale is not None:
        fc1_scale_split = fc1_scale.shape[0] // 2
        weights[f"model.layers.{layer_idx}.mlp.gate_proj.weight_scale"] = fc1_scale[:fc1_scale_split]
        weights[f"model.layers.{layer_idx}.mlp.up_proj.weight_scale"] = fc1_scale[fc1_scale_split:]

    fc2_scale = _get_quantizer_scale_for_compare(layer.mlp.linear_fc2)
    if fc2_scale is not None:
        weights[f"model.layers.{layer_idx}.mlp.down_proj.weight_scale"] = fc2_scale

    return weights


def _collect_hf_reference_layer_weights(
    *,
    model_path: str,
    layer_idx: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    from transformers import AutoModelForCausalLM

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.float32 if device.type == "cpu" else "auto",
    }
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    try:
        model.eval()
        model = model.to(device)
        layer = model.model.layers[layer_idx]
        weights = {
            f"model.layers.{layer_idx}.input_layernorm.weight": _clone_weight_for_compare(layer.input_layernorm.weight),
            f"model.layers.{layer_idx}.self_attn.q_proj.weight": _clone_weight_for_compare(layer.self_attn.q_proj.weight),
            f"model.layers.{layer_idx}.self_attn.k_proj.weight": _clone_weight_for_compare(layer.self_attn.k_proj.weight),
            f"model.layers.{layer_idx}.self_attn.v_proj.weight": _clone_weight_for_compare(layer.self_attn.v_proj.weight),
            f"model.layers.{layer_idx}.self_attn.o_proj.weight": _clone_weight_for_compare(layer.self_attn.o_proj.weight),
            f"model.layers.{layer_idx}.post_attention_layernorm.weight": _clone_weight_for_compare(
                layer.post_attention_layernorm.weight
            ),
            f"model.layers.{layer_idx}.mlp.gate_proj.weight": _clone_weight_for_compare(layer.mlp.gate_proj.weight),
            f"model.layers.{layer_idx}.mlp.up_proj.weight": _clone_weight_for_compare(layer.mlp.up_proj.weight),
            f"model.layers.{layer_idx}.mlp.down_proj.weight": _clone_weight_for_compare(layer.mlp.down_proj.weight),
        }
        embed_tokens = getattr(model.model, "embed_tokens", None)
        if embed_tokens is not None and getattr(embed_tokens, "weight", None) is not None:
            weights["model.embed_tokens.weight"] = _clone_weight_for_compare(embed_tokens.weight)

        lm_head = getattr(model, "lm_head", None)
        if lm_head is not None and getattr(lm_head, "weight", None) is not None:
            weights["lm_head.weight"] = _clone_weight_for_compare(lm_head.weight)

        final_norm = getattr(model.model, "norm", None)
        if final_norm is not None and getattr(final_norm, "weight", None) is not None:
            weights["model.norm.weight"] = _clone_weight_for_compare(final_norm.weight)

        q_norm = getattr(layer.self_attn, "q_norm", None)
        if q_norm is not None and getattr(q_norm, "weight", None) is not None:
            weights[f"model.layers.{layer_idx}.self_attn.q_norm.weight"] = _clone_weight_for_compare(q_norm.weight)

        k_norm = getattr(layer.self_attn, "k_norm", None)
        if k_norm is not None and getattr(k_norm, "weight", None) is not None:
            weights[f"model.layers.{layer_idx}.self_attn.k_norm.weight"] = _clone_weight_for_compare(k_norm.weight)

        scale_modules = {
            f"model.layers.{layer_idx}.self_attn.q_proj.weight_scale_inv": layer.self_attn.q_proj,
            f"model.layers.{layer_idx}.self_attn.k_proj.weight_scale_inv": layer.self_attn.k_proj,
            f"model.layers.{layer_idx}.self_attn.v_proj.weight_scale_inv": layer.self_attn.v_proj,
            f"model.layers.{layer_idx}.self_attn.o_proj.weight_scale_inv": layer.self_attn.o_proj,
            f"model.layers.{layer_idx}.mlp.gate_proj.weight_scale_inv": layer.mlp.gate_proj,
            f"model.layers.{layer_idx}.mlp.up_proj.weight_scale_inv": layer.mlp.up_proj,
            f"model.layers.{layer_idx}.mlp.down_proj.weight_scale_inv": layer.mlp.down_proj,
        }
        for key, linear_module in scale_modules.items():
            scale_inv = _get_scale_inv_for_compare(linear_module)
            if scale_inv is not None:
                weights[key] = scale_inv
        return weights
    finally:
        del model
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


def _tensor_sample_for_debug(tensor: torch.Tensor, *, count: int = 4) -> list[float]:
    flat = tensor.reshape(-1)[:count].to(torch.float32)
    return [float(value) for value in flat.tolist()]


def _activation_compare_sort_key(name: str) -> tuple[int, int, str]:
    if name == "embed_tokens":
        return (-2, 0, name)
    if name == "final_logits":
        return (10**9, 0, name)

    if name.startswith("layer"):
        layer_and_stage = name[len("layer") :]
        layer_token, _, stage_name = layer_and_stage.partition(".")
        if layer_token.isdigit():
            stage_priority = {
                "input_layernorm_input": 0,
                "input_layernorm": 1,
                "q_proj": 2,
                "k_proj": 3,
                "v_proj": 4,
                "o_proj": 5,
                "post_attention_layernorm_input": 6,
                "post_attention_layernorm": 7,
                "gate_proj": 8,
                "up_proj": 9,
                "down_proj": 10,
                "layer_output": 11,
            }.get(stage_name, 100)
            return (int(layer_token), stage_priority, name)

    return (10**8, 0, name)


def _maybe_emit_hf_megatron_weight_compare_debug(
    *,
    layer_idx: int,
    megatron_weights: dict[str, torch.Tensor],
    hf_weights: dict[str, torch.Tensor],
) -> bool:
    if os.getenv("ORBIT_RUNTIME_PARITY_HF_MEGATRON_WEIGHT_COMPARE", "0") != "1":
        return False

    for name in sorted(set(megatron_weights) | set(hf_weights)):
        megatron_tensor = megatron_weights.get(name)
        hf_tensor = hf_weights.get(name)
        if megatron_tensor is None or hf_tensor is None:
            print(
                f"ORBIT_RUNTIME_PARITY_HF_MEGATRON_WEIGHT_COMPARE layer={layer_idx} name={name} "
                f"status=missing megatron_present={megatron_tensor is not None} hf_present={hf_tensor is not None}",
                flush=True,
            )
            continue

        shape_match = tuple(megatron_tensor.shape) == tuple(hf_tensor.shape)
        if shape_match:
            diff = torch.abs(megatron_tensor - hf_tensor)
            max_abs_diff = float(diff.max().item()) if diff.numel() else 0.0
            mean_abs_diff = float(diff.mean().item()) if diff.numel() else 0.0
        else:
            max_abs_diff = float("nan")
            mean_abs_diff = float("nan")

        print(
            f"ORBIT_RUNTIME_PARITY_HF_MEGATRON_WEIGHT_COMPARE layer={layer_idx} name={name} "
            f"shape_match={shape_match} "
            f"megatron_shape={tuple(megatron_tensor.shape)} hf_shape={tuple(hf_tensor.shape)} "
            f"max_abs_diff={max_abs_diff} mean_abs_diff={mean_abs_diff} "
            f"megatron_sample={_tensor_sample_for_debug(megatron_tensor)} "
            f"hf_sample={_tensor_sample_for_debug(hf_tensor)}",
            flush=True,
        )

    return True


def _maybe_emit_sglang_megatron_weight_compare_debug(
    *,
    layer_idx: int,
    megatron_weights: dict[str, torch.Tensor],
    sglang_weight_samples: dict[str, torch.Tensor] | None,
    sglang_weight_errors: dict[str, str] | None,
    truncate_size: int,
    include_scales: bool,
) -> bool:
    if not _sglang_megatron_weight_compare_enabled():
        return False

    sglang_weight_samples = sglang_weight_samples or {}
    sglang_weight_errors = sglang_weight_errors or {}
    names = sorted(set(megatron_weights) | set(sglang_weight_samples))
    for name in names:
        if not include_scales and _weight_name_is_scale(name):
            continue

        megatron_tensor = megatron_weights.get(name)
        sglang_tensor = sglang_weight_samples.get(name)
        if megatron_tensor is None or sglang_tensor is None:
            print(
                f"ORBIT_RUNTIME_PARITY_WEIGHT_COMPARE layer={layer_idx} name={name} "
                f"status=missing megatron_present={megatron_tensor is not None} "
                f"sglang_present={sglang_tensor is not None} "
                f"sglang_error={sglang_weight_errors.get(name)!r}",
                flush=True,
            )
            continue

        megatron_sample = _truncate_weight_sample_for_sglang_compare(
            megatron_tensor,
            truncate_size=truncate_size,
        )
        sglang_sample = _truncate_weight_sample_for_sglang_compare(
            sglang_tensor,
            truncate_size=truncate_size,
        )
        shape_match = tuple(megatron_sample.shape) == tuple(sglang_sample.shape)
        if shape_match:
            diff = torch.abs(megatron_sample - sglang_sample)
            max_abs_diff = float(diff.max().item()) if diff.numel() else 0.0
            mean_abs_diff = float(diff.mean().item()) if diff.numel() else 0.0
        else:
            max_abs_diff = float("nan")
            mean_abs_diff = float("nan")

        print(
            f"ORBIT_RUNTIME_PARITY_WEIGHT_COMPARE layer={layer_idx} name={name} "
            f"shape_match={shape_match} "
            f"megatron_shape={tuple(megatron_sample.shape)} sglang_shape={tuple(sglang_sample.shape)} "
            f"max_abs_diff={max_abs_diff} mean_abs_diff={mean_abs_diff} "
            f"megatron_sample={_tensor_sample_for_debug(megatron_sample)} "
            f"sglang_sample={_tensor_sample_for_debug(sglang_sample)}",
            flush=True,
        )

    return True


def _maybe_emit_runtime_output_debug(
    *,
    stage: str,
    prompts: list[str],
    rollout_result: dict[str, object],
    actor_result: dict[str, object],
) -> bool:
    if os.getenv("ORBIT_RUNTIME_PARITY_OUTPUT_COMPARE", "0") != "1":
        return False

    rollout_topk_tokens = torch.as_tensor(rollout_result.get("topk_tokens", []), dtype=torch.long)
    actor_topk_tokens = torch.as_tensor(actor_result.get("topk_tokens", []), dtype=torch.long)
    rollout_topk_logprobs = torch.as_tensor(rollout_result.get("topk_logprobs", []), dtype=torch.float32)
    actor_topk_logprobs = torch.as_tensor(actor_result.get("topk_logprobs", []), dtype=torch.float32)
    rollout_logprobs = torch.as_tensor(rollout_result.get("logprobs", []), dtype=torch.float32)
    actor_logprobs = torch.as_tensor(actor_result.get("logprobs", []), dtype=torch.float32)

    row_count = min(
        len(prompts),
        rollout_topk_tokens.shape[0] if rollout_topk_tokens.ndim >= 1 else 0,
        actor_topk_tokens.shape[0] if actor_topk_tokens.ndim >= 1 else 0,
    )
    for idx in range(row_count):
        prompt_text = prompts[idx].replace("\n", "\\n")
        print(
            f"ORBIT_RUNTIME_PARITY_OUTPUT stage={stage} idx={idx} "
            f"prompt={prompt_text!r} "
            f"rollout_chosen_logprob={float(rollout_logprobs[idx]) if idx < rollout_logprobs.numel() else 'n/a'} "
            f"actor_chosen_logprob={float(actor_logprobs[idx]) if idx < actor_logprobs.numel() else 'n/a'} "
            f"rollout_topk_tokens={rollout_topk_tokens[idx].tolist()} "
            f"rollout_topk_logprobs={[float(v) for v in rollout_topk_logprobs[idx].tolist()]} "
            f"actor_topk_tokens={actor_topk_tokens[idx].tolist()} "
            f"actor_topk_logprobs={[float(v) for v in actor_topk_logprobs[idx].tolist()]}",
            flush=True,
        )

    return True


def _ensure_engine_model_on_runtime_device(engine: Any) -> None:
    engine.to(device=_get_device_name(), model=True, optimizer=False, grad=False)


def collect_megatron_outputs(
    *,
    hf_model: str,
    megatron_dist: str,
    prompts: list[str],
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    restore_modelopt_state: bool = True,
    megatron_activation_fp8: str | None = "none",
    megatron_oft_fp8_activation_quant: str | None = None,
    megatron_fp8_w8a8_parity_mode: str = "activation_qdq",
    dequant_fp8_weights_for_parity: bool = False,
    dequant_nvfp4_weights_for_parity: bool = False,
    dequant_int4_weights_for_parity: bool = False,
    sglang_weight_samples: dict[str, torch.Tensor] | None = None,
    sglang_weight_errors: dict[str, str] | None = None,
    sglang_activation_stages: dict[str, torch.Tensor] | None = None,
    topk: int = TOPK,
) -> dict[str, object]:
    resolved_megatron_dist = resolve_megatron_dist_checkpoint_dir(megatron_dist)
    _ensure_real_single_rank_runtime(
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
    )

    from verl.trainer.config.config import CheckpointConfig
    from verl.models.mcore import get_mcore_forward_fn, get_mcore_forward_no_padding_fn
    from verl.workers.config import HFModelConfig, McoreEngineConfig, McoreOptimizerConfig
    from verl.workers.engine.megatron.transformer_impl import MegatronEngineWithLMHead

    created_dist = _prepare_single_rank_dist()
    engine = None
    debug_handles = []
    previous_oft_fp8_activation_quant = os.environ.get(MEGATRON_OFT_FP8_ACTIVATION_QUANT_ENV)
    set_oft_fp8_activation_quant = megatron_oft_fp8_activation_quant not in (None, "auto")
    if set_oft_fp8_activation_quant:
        os.environ[MEGATRON_OFT_FP8_ACTIVATION_QUANT_ENV] = str(megatron_oft_fp8_activation_quant)
    try:
        model_config = HFModelConfig(path=hf_model, trust_remote_code=True)
        engine_config = McoreEngineConfig(
            forward_only=True,
            use_mbridge=True,
            vanilla_mbridge=False,
            use_dist_checkpointing=True,
            dist_checkpointing_path=resolved_megatron_dist,
            tensor_model_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_parallel_size,
            use_remove_padding=True,
            override_transformer_config=_build_actor_override_transformer_config(
                restore_modelopt_state=restore_modelopt_state,
                megatron_activation_fp8=megatron_activation_fp8,
            ),
        )
        optimizer_config = McoreOptimizerConfig()
        checkpoint_config = CheckpointConfig(load_contents=["model"])
        engine = MegatronEngineWithLMHead(
            model_config=model_config,
            engine_config=engine_config,
            optimizer_config=optimizer_config,
            checkpoint_config=checkpoint_config,
        )
        engine.initialize()
        _ensure_engine_model_on_runtime_device(engine)

        token_ids = load_actor_prompt_token_ids(
            hf_model=hf_model,
            megatron_dist=resolved_megatron_dist,
            prompts=prompts,
        )
        device = torch.device("cuda", 0)
        pad_token_id = model_config.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = model_config.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("Megatron actor parity requires a tokenizer pad_token_id or eos_token_id fallback.")
        nested_ids = torch.nested.as_nested_tensor(
            [torch.tensor(ids, dtype=torch.long, device=device) for ids in token_ids],
            layout=torch.jagged,
        )
        lengths = torch.tensor([len(ids) for ids in token_ids], dtype=torch.long, device=device)
        padded_input_ids, padded_attention_mask, padded_position_ids = _build_padded_prompt_batch(
            token_ids,
            device=device,
            pad_token_id=pad_token_id,
        )

        with engine.eval_mode():
            module = engine.module[0] if isinstance(engine.module, list) else engine.module
            if dequant_fp8_weights_for_parity:
                compare_root = _unwrap_megatron_module_for_weight_compare(module)
                debug_handles.append(_attach_fp8_weight_dequant_parity_hooks(compare_root))
            use_fp8_block_linear_parity = (
                megatron_oft_fp8_activation_quant == "w8a8"
                and megatron_fp8_w8a8_parity_mode == "block_linear"
            )
            if use_fp8_block_linear_parity:
                compare_root = _unwrap_megatron_module_for_weight_compare(module)
                debug_handles.append(_attach_fp8_block_linear_parity_hooks(compare_root))
            elif not dequant_fp8_weights_for_parity and (
                _env_flag("VERL_KEEP_NATIVE_FP8_WEIGHTS") or _env_flag("MEGATRON_KEEP_NATIVE_FP8_WEIGHTS")
            ):
                compare_root = _unwrap_megatron_module_for_weight_compare(module)
                debug_handles.append(_attach_modelopt_fp8_weight_quantizer_bypass_hooks(compare_root))
            if megatron_oft_fp8_activation_quant == "w8a8" and not use_fp8_block_linear_parity:
                compare_root = _unwrap_megatron_module_for_weight_compare(module)
                debug_handles.append(_attach_fp8_activation_qdq_parity_hooks(compare_root))
            if dequant_nvfp4_weights_for_parity:
                compare_root = _unwrap_megatron_module_for_weight_compare(module)
                debug_handles.append(_attach_nvfp4_weight_dequant_parity_hooks(compare_root))
            if dequant_int4_weights_for_parity:
                compare_root = _unwrap_megatron_module_for_weight_compare(module)
                debug_handles.append(_attach_int4_weight_dequant_parity_hooks(compare_root))
            forward_fn = get_mcore_forward_no_padding_fn(model_config.hf_config)
            output = forward_fn(
                module,
                nested_ids,
                {},
                vision_model=hasattr(model_config.hf_config, "vision_config"),
                pad_token_id=model_config.tokenizer.pad_token_id,
                data_format="thd" if engine_config.use_remove_padding else "bshd",
                mtp_enable_train=model_config.mtp.enable and model_config.mtp.enable_train,
            )
        logits = _extract_model_logits(output).to(torch.float32)
        next_token_logits = _select_last_token_logits_from_nested_output(
            logits,
            lengths=lengths,
        )
        next_token_logprobs = torch.log_softmax(next_token_logits, dim=-1)
        topk = min(topk, next_token_logprobs.shape[-1])
        topk_logprobs, topk_tokens = torch.topk(next_token_logprobs, k=topk, dim=-1)

        observed_scores = topk_logprobs.cpu()
        result = {
            "token_ids": token_ids,
            "logprobs": topk_logprobs[:, 0].cpu(),
            "topk_tokens": topk_tokens.cpu(),
            "topk_logprobs": observed_scores,
            # The real Megatron path computes dense logits, but we align on the observable top-k score table
            # because the rollout API does not expose dense vocab logits in the same shape.
            "logits": observed_scores,
        }

        if os.getenv("ORBIT_RUNTIME_PARITY_ACTOR_FORWARD_COMPARE", "0") == "1":
            with engine.eval_mode():
                packed_forward_fn = get_mcore_forward_fn(model_config.hf_config)
                packed_output = _run_actor_packed_forward_debug(
                    packed_forward_fn=packed_forward_fn,
                    module=module,
                    input_ids=padded_input_ids,
                    attention_mask=padded_attention_mask,
                    position_ids=padded_position_ids,
                    mtp_config=model_config.mtp,
                )
            packed_logits = _extract_model_logits(packed_output).to(torch.float32)
            packed_next_token_logits = _select_last_token_logits_from_nested_output(
                packed_logits,
                lengths=lengths,
            )
            packed_next_token_logprobs = torch.log_softmax(packed_next_token_logits, dim=-1)
            packed_topk_logprobs, packed_topk_tokens = torch.topk(packed_next_token_logprobs, k=topk, dim=-1)
            packed_result = {
                "logprobs": packed_topk_logprobs[:, 0].cpu(),
                "topk_tokens": packed_topk_tokens.cpu(),
                "topk_logprobs": packed_topk_logprobs.cpu(),
            }
            _maybe_emit_actor_forward_compare_debug(
                prompts=prompts,
                no_padding_result=result,
                packed_result=packed_result,
            )

        if os.getenv("ORBIT_RUNTIME_PARITY_HF_MEGATRON_ACTIVATION_COMPARE", "0") == "1":
            activation_compare_layers = os.getenv("ORBIT_RUNTIME_PARITY_HF_MEGATRON_ACTIVATION_COMPARE_LAYERS")
            with engine.eval_mode():
                packed_forward_fn = get_mcore_forward_fn(model_config.hf_config)
                megatron_activation_stages = _collect_megatron_packed_activation_outputs(
                    module=module,
                    packed_forward_fn=packed_forward_fn,
                    input_ids=padded_input_ids,
                    attention_mask=padded_attention_mask,
                    position_ids=padded_position_ids,
                    lengths=lengths,
                    mtp_config=model_config.mtp,
                    raw_layer_indices=activation_compare_layers,
                )
            try:
                engine.to(device="cpu", model=True, optimizer=False, grad=False)
            except Exception:
                pass
            hf_activation_stages = _collect_hf_direct_activation_outputs(
                model_path=hf_model,
                token_ids=token_ids,
                device=device,
                raw_layer_indices=activation_compare_layers,
            )
            _maybe_emit_hf_megatron_activation_compare_debug(
                prompts=prompts,
                megatron_stages=megatron_activation_stages,
                hf_stages=hf_activation_stages,
            )

        if _sglang_megatron_activation_compare_enabled():
            activation_compare_layers = (
                os.getenv(SGLANG_MEGATRON_ACTIVATION_COMPARE_LAYERS_ENV)
                or os.getenv("ORBIT_RUNTIME_PARITY_HF_MEGATRON_ACTIVATION_COMPARE_LAYERS")
                or os.getenv(SGLANG_MEGATRON_WEIGHT_COMPARE_LAYERS_ENV)
                or os.getenv("ORBIT_RUNTIME_PARITY_HF_MEGATRON_WEIGHT_COMPARE_LAYERS")
            )
            with engine.eval_mode():
                packed_forward_fn = get_mcore_forward_fn(model_config.hf_config)
                megatron_activation_stages = _collect_megatron_packed_activation_outputs(
                    module=module,
                    packed_forward_fn=packed_forward_fn,
                    input_ids=padded_input_ids,
                    attention_mask=padded_attention_mask,
                    position_ids=padded_position_ids,
                    lengths=lengths,
                    mtp_config=model_config.mtp,
                    raw_layer_indices=activation_compare_layers,
                )
            _maybe_emit_sglang_megatron_activation_compare_debug(
                prompts=prompts,
                megatron_stages=megatron_activation_stages,
                sglang_stages=sglang_activation_stages or {},
            )

        if os.getenv("ORBIT_RUNTIME_PARITY_HF_DIRECT_COMPARE", "0") == "1":
            try:
                engine.to(device="cpu", model=True, optimizer=False, grad=False)
            except Exception:
                pass
            hf_result = _collect_hf_direct_outputs(
                model_path=hf_model,
                token_ids=token_ids,
                topk=topk,
                device=device,
            )
            _maybe_emit_hf_direct_compare_debug(
                prompts=prompts,
                actor_result=result,
                hf_result=hf_result,
            )

        if os.getenv("ORBIT_RUNTIME_PARITY_HF_MEGATRON_WEIGHT_COMPARE", "0") == "1":
            compare_root = _unwrap_megatron_module_for_weight_compare(module)
            compare_layer_indices = _resolve_weight_compare_layer_indices(
                os.getenv("ORBIT_RUNTIME_PARITY_HF_MEGATRON_WEIGHT_COMPARE_LAYERS"),
                layer_count=len(compare_root.decoder.layers),
            )
            try:
                engine.to(device="cpu", model=True, optimizer=False, grad=False)
            except Exception:
                pass
            for layer_idx in compare_layer_indices:
                megatron_weights = _extract_megatron_dense_layer_weights_for_hf_compare(module, layer_idx=layer_idx)
                hf_weights = _collect_hf_reference_layer_weights(
                    model_path=hf_model,
                    layer_idx=layer_idx,
                    device=device,
                )
                _maybe_emit_hf_megatron_weight_compare_debug(
                    layer_idx=layer_idx,
                    megatron_weights=megatron_weights,
                    hf_weights=hf_weights,
                )

        if _sglang_megatron_weight_compare_enabled():
            compare_root = _unwrap_megatron_module_for_weight_compare(module)
            compare_layer_indices = _resolve_weight_compare_layer_indices(
                os.getenv(SGLANG_MEGATRON_WEIGHT_COMPARE_LAYERS_ENV)
                or os.getenv("ORBIT_RUNTIME_PARITY_HF_MEGATRON_WEIGHT_COMPARE_LAYERS"),
                layer_count=len(compare_root.decoder.layers),
            )
            truncate_size = _sglang_megatron_weight_compare_truncate_size()
            include_scales = _sglang_megatron_weight_compare_include_scales()
            for layer_idx in compare_layer_indices:
                megatron_weights = _extract_megatron_dense_layer_weights_for_hf_compare(module, layer_idx=layer_idx)
                _maybe_emit_sglang_megatron_weight_compare_debug(
                    layer_idx=layer_idx,
                    megatron_weights=megatron_weights,
                    sglang_weight_samples=sglang_weight_samples,
                    sglang_weight_errors=sglang_weight_errors,
                    truncate_size=truncate_size,
                    include_scales=include_scales,
                )

        return result
    finally:
        for handle in debug_handles:
            try:
                handle.remove()
            except Exception:
                pass
        if engine is not None:
            try:
                engine.to(device="cpu", model=True, optimizer=False, grad=False)
            except Exception:
                pass
        if set_oft_fp8_activation_quant:
            if previous_oft_fp8_activation_quant is None:
                os.environ.pop(MEGATRON_OFT_FP8_ACTIVATION_QUANT_ENV, None)
            else:
                os.environ[MEGATRON_OFT_FP8_ACTIVATION_QUANT_ENV] = previous_oft_fp8_activation_quant
        if created_dist:
            import torch.distributed as dist

            dist.destroy_process_group()


def run_megatron_actor_step0(
    *,
    hf_model: str,
    megatron_dist: str,
    prompts: list[str],
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    restore_modelopt_state: bool = True,
    megatron_activation_fp8: str | None = "none",
    megatron_oft_fp8_activation_quant: str | None = None,
    megatron_fp8_w8a8_parity_mode: str = "activation_qdq",
    dequant_fp8_weights_for_parity: bool = False,
    dequant_nvfp4_weights_for_parity: bool = False,
    dequant_int4_weights_for_parity: bool = False,
    sglang_weight_samples: dict[str, torch.Tensor] | None = None,
    sglang_weight_errors: dict[str, str] | None = None,
    sglang_activation_stages: dict[str, torch.Tensor] | None = None,
) -> dict[str, object]:
    return collect_megatron_outputs(
        hf_model=hf_model,
        megatron_dist=megatron_dist,
        prompts=prompts,
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        restore_modelopt_state=restore_modelopt_state,
        megatron_activation_fp8=megatron_activation_fp8,
        megatron_oft_fp8_activation_quant=megatron_oft_fp8_activation_quant,
        megatron_fp8_w8a8_parity_mode=megatron_fp8_w8a8_parity_mode,
        dequant_fp8_weights_for_parity=dequant_fp8_weights_for_parity,
        dequant_nvfp4_weights_for_parity=dequant_nvfp4_weights_for_parity,
        dequant_int4_weights_for_parity=dequant_int4_weights_for_parity,
        sglang_weight_samples=sglang_weight_samples,
        sglang_weight_errors=sglang_weight_errors,
        sglang_activation_stages=sglang_activation_stages,
    )


def compare_runtime_outputs(
    *,
    rollout_result: dict[str, object],
    actor_result: dict[str, object],
    thresholds: RuntimeParityThresholds,
) -> RuntimeParityReport:
    rollout_logits = torch.as_tensor(rollout_result["logits"], dtype=torch.float32)
    actor_logits = torch.as_tensor(actor_result["logits"], dtype=torch.float32)
    rollout_logprobs = torch.as_tensor(rollout_result["logprobs"], dtype=torch.float32)
    actor_logprobs = torch.as_tensor(actor_result["logprobs"], dtype=torch.float32)
    rollout_topk = torch.as_tensor(rollout_result["topk_tokens"], dtype=torch.long)
    actor_topk = torch.as_tensor(actor_result["topk_tokens"], dtype=torch.long)

    if rollout_logits.shape != actor_logits.shape:
        raise ValueError(f"Observed score tables differ in shape: {rollout_logits.shape} vs {actor_logits.shape}")
    if rollout_logprobs.shape != actor_logprobs.shape:
        raise ValueError(f"Chosen-token logprob tensors differ in shape: {rollout_logprobs.shape} vs {actor_logprobs.shape}")
    if rollout_topk.shape != actor_topk.shape:
        raise ValueError(f"Top-k token tables differ in shape: {rollout_topk.shape} vs {actor_topk.shape}")

    tokenizer_match = rollout_result["token_ids"] == actor_result["token_ids"]
    max_logprob_abs_diff = float(torch.max(torch.abs(rollout_logprobs - actor_logprobs)))
    return RuntimeParityReport(
        tokenizer_match=tokenizer_match,
        # On the current real single-rank path, both runtimes expose aligned top-k score tables rather than
        # dense vocab logits, so this metric is computed on those observable sampled-position scores.
        max_logit_abs_diff=float(torch.max(torch.abs(rollout_logits - actor_logits))),
        max_logprob_abs_diff=max_logprob_abs_diff,
        next_token_match_rate=float((rollout_topk[:, 0] == actor_topk[:, 0]).to(torch.float32).mean()),
        topk_agreement=float((rollout_topk == actor_topk).to(torch.float32).mean()),
        thresholds=thresholds,
    )


def main(default_precision_profile: str = "fp8") -> None:
    args = parse_args(default_precision_profile=default_precision_profile)
    resolved_megatron_dist = resolve_megatron_dist_checkpoint_dir(args.megatron_dist)
    _ensure_real_single_rank_runtime(
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
    )

    prompts = load_prompts_jsonl(args.prompt_file)
    if not prompts:
        raise SystemExit(f"No prompts found in {args.prompt_file}")
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]

    thresholds = RuntimeParityThresholds(
        max_logprob_abs_diff=args.max_logprob_abs_diff,
        min_topk_agreement=args.min_topk_agreement,
    )
    rollout_result = run_sglang_rollout_step0(
        hf_model=args.hf_model,
        prompts=prompts,
        rollout_quantization=args.rollout_quantization,
        sglang_torchao_config=args.sglang_torchao_config,
        sglang_fp8_gemm_backend=args.sglang_fp8_gemm_backend,
    )
    actor_result = run_megatron_actor_step0(
        hf_model=args.hf_model,
        megatron_dist=resolved_megatron_dist,
        prompts=prompts,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        restore_modelopt_state=args.restore_modelopt_state,
        megatron_activation_fp8=args.megatron_activation_fp8,
        megatron_oft_fp8_activation_quant=args.megatron_oft_fp8_activation_quant,
        megatron_fp8_w8a8_parity_mode=args.megatron_fp8_w8a8_parity_mode,
        dequant_fp8_weights_for_parity=args.dequant_fp8_weights_for_parity,
        dequant_nvfp4_weights_for_parity=args.dequant_nvfp4_weights_for_parity,
        dequant_int4_weights_for_parity=args.dequant_int4_weights_for_parity,
        sglang_weight_samples=rollout_result.get("sglang_weight_samples"),
        sglang_weight_errors=rollout_result.get("sglang_weight_errors"),
        sglang_activation_stages=rollout_result.get("sglang_activation_stages"),
    )
    report = compare_runtime_outputs(
        rollout_result=rollout_result,
        actor_result=actor_result,
        thresholds=thresholds,
    )
    _maybe_emit_runtime_output_debug(
        stage="final_compare",
        prompts=prompts,
        rollout_result=rollout_result,
        actor_result=actor_result,
    )
    if args.report_json:
        write_report_json(args.report_json, asdict(report))
    print(f"runtime parity: {'ok' if report.ok else 'not ok'}")
    print(f"tokenizer match: {report.tokenizer_match}")
    print(f"max observed-score abs diff: {report.max_logit_abs_diff:.6f}")
    print(f"max logprob abs diff: {report.max_logprob_abs_diff:.6f}")
    print(f"next-token match rate: {report.next_token_match_rate:.6f}")
    print(f"top-k agreement: {report.topk_agreement:.6f}")
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
