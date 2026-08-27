#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import re
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

# Force mcore async strategy everywhere (modelopt's save_sharded_modelopt_state
# calls dist_checkpointing.save without passing async_strategy, which defaults
# to "nvrx" and fails when nvidia-resiliency-ext is unavailable/mismatched).
from megatron.core import dist_checkpointing as _dc

_orig_dc_save = _dc.save


def _dc_save_with_mcore(*args, **kwargs):
    kwargs.setdefault("async_strategy", "mcore")
    return _orig_dc_save(*args, **kwargs)


_dc.save = _dc_save_with_mcore

from megatron.bridge.orbit.low_precision.common import (
    TensorSpillManager,
    build_single_rank_meta_provider,
    patch_meta_init_for_te_modules,
)
from megatron.bridge.orbit.low_precision.nvfp4 import (
    apply_modelopt_nvfp4_to_meta_model,
    build_nvfp4_direct_model_state_dict,
    collect_nvfp4_target_module_names,
    is_nvfp4_source,
)
from megatron.bridge.training.checkpointing import (
    get_checkpoint_name,
    save_checkpoint,
    save_tokenizer_assets,
)
from megatron.bridge.training.config import CheckpointConfig, ConfigContainer, LoggerConfig
from megatron.bridge.training.model_load_save import temporary_distributed_context
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.core.optimizer import OptimizerConfig


def _format_elapsed(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def _log_stage_start(message: str) -> float:
    print(message, flush=True)
    return time.monotonic()


def _log_stage_done(message: str, start_time: float, *, extra: str | None = None) -> None:
    suffix = f" | {extra}" if extra else ""
    print(
        f"{message} in {_format_elapsed(time.monotonic() - start_time)}{suffix}",
        flush=True,
    )


_DECODER_LAYER_RE = re.compile(r"decoder\.layers\.(\d+)\.")


def _parse_debug_layer_range(value: str) -> tuple[int, int]:
    try:
        start_str, end_str = value.split(":", 1)
        start = int(start_str)
        end = int(end_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid debug layer range {value!r}. Expected START:END, e.g. 0:4."
        ) from exc

    if start < 0 or end < 0:
        raise argparse.ArgumentTypeError("Debug layer range indices must be non-negative.")
    if end <= start:
        raise argparse.ArgumentTypeError(f"Invalid debug layer range {value!r}. END must be greater than START.")

    return start, end


def _task_decoder_layer_idx(task: Any) -> int | None:
    if task is None:
        return None
    param_name = getattr(task, "param_name", None)
    if not isinstance(param_name, str):
        return None
    match = _DECODER_LAYER_RE.search(param_name)
    return int(match.group(1)) if match else None


def _filter_conversion_tasks_for_debug_layer_range(
    conversion_tasks: list[Any],
    layer_range: tuple[int, int],
) -> list[Any]:
    start, end = layer_range
    filtered = []
    for task in conversion_tasks:
        layer_idx = _task_decoder_layer_idx(task)
        if layer_idx is None:
            continue
        if start <= layer_idx < end:
            filtered.append(task)
    return filtered


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _format_num_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024.0 or unit == "PiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


class _SaveProgressMonitor:
    def __init__(self, checkpoint_root: str | Path, interval_sec: float = 10.0):
        self.checkpoint_root = Path(checkpoint_root)
        self.interval_sec = max(interval_sec, 1.0)
        self._start_time = 0.0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="nvfp4-direct-save-progress",
            daemon=True,
        )
        self._last_snapshot: tuple[int, int] | None = None

    def start(self) -> None:
        self._start_time = time.monotonic()
        print(
            f"Save progress monitor enabled for {self.checkpoint_root} " f"(interval={self.interval_sec:.1f}s)",
            flush=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=self.interval_sec + 1.0)
        self._emit_progress(final=True)

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_sec):
            self._emit_progress()

    def _snapshot(self) -> tuple[int, int]:
        root = self.checkpoint_root
        if not root.exists():
            return 0, 0

        file_count = 0
        total_bytes = 0
        for path in root.rglob("*"):
            try:
                if not path.is_file():
                    continue
                stat_result = path.stat()
            except FileNotFoundError:
                continue
            file_count += 1
            total_bytes += stat_result.st_size
        return file_count, total_bytes

    def _emit_progress(self, *, final: bool = False) -> None:
        snapshot = self._snapshot()
        if not final and snapshot == self._last_snapshot:
            return

        self._last_snapshot = snapshot
        file_count, total_bytes = snapshot
        elapsed = _format_elapsed(time.monotonic() - self._start_time)
        label = "final save progress" if final else "save progress"
        print(
            f"[{label}] elapsed {elapsed} | files {file_count} | written {_format_num_bytes(total_bytes)}",
            flush=True,
        )


def _maybe_create_save_progress_monitor(path: str | Path) -> _SaveProgressMonitor | None:
    if not _env_flag_enabled("MEGATRON_BRIDGE_DIRECT_SAVE_PROGRESS"):
        return None

    try:
        interval_sec = float(os.environ.get("MEGATRON_BRIDGE_DIRECT_SAVE_PROGRESS_INTERVAL", "10"))
    except ValueError:
        interval_sec = 10.0

    return _SaveProgressMonitor(path, interval_sec=interval_sec)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Direct-write HF NVFP4 -> Megatron checkpoint conversion",
    )
    parser.add_argument("--hf-model-path", required=True)
    parser.add_argument("--megatron-path", required=True)
    parser.add_argument(
        "--debug-layer-range",
        type=_parse_debug_layer_range,
        help=(
            "Optional half-open decoder layer range START:END for debug-only partial "
            "conversion, e.g. 0:4 converts decoder.layers.0-3 only. "
            "The resulting checkpoint is intended for save-path debugging, not loading."
        ),
    )
    return parser.parse_args(argv)


def _save_direct_checkpoint(
    provider: Any,
    path: str,
    model_state: dict[str, Any],
    *,
    model_list: list[Any],
    pg_collection: Any,
    hf_tokenizer_path: str | None,
    hf_tokenizer_kwargs: dict[str, Any] | None,
) -> None:
    storage_writers_per_rank = int(os.environ.get("MEGATRON_BRIDGE_DIRECT_STORAGE_WRITERS_PER_RANK", "16"))
    tokenizer_config = None
    if hf_tokenizer_path is not None:
        tokenizer_config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=str(hf_tokenizer_path),
            hf_tokenizer_kwargs=hf_tokenizer_kwargs or {},
        )

    state = GlobalState()
    if hasattr(state, "train_state") and hasattr(state.train_state, "step"):
        state.train_state.step = 0

    state.cfg = ConfigContainer(
        model=provider,
        train=None,
        optimizer=OptimizerConfig(use_distributed_optimizer=False),
        ddp=None,
        scheduler=None,
        dataset=None,
        logger=LoggerConfig(),
        tokenizer=tokenizer_config,
        checkpoint=CheckpointConfig(
            async_save=False,
            async_strategy="mcore",
            save=str(path),
            save_optim=False,
            save_rng=False,
            ckpt_format="torch_dist",
            dist_ckpt_optim_fully_reshardable=True,
            fully_parallel_save=False,
            storage_writers_per_rank=storage_writers_per_rank,
        ),
        dist=None,
    )

    prebuilt_state_dict = {
        "checkpoint_version": 3.0,
        "iteration": 0,
        "model": model_state,
    }

    t0 = time.monotonic()
    progress_monitor = _maybe_create_save_progress_monitor(path)
    print("Saving checkpoint...", flush=True)
    try:
        if progress_monitor is not None:
            progress_monitor.start()
        save_checkpoint(
            state=state,
            model=model_list,
            optimizer=None,
            opt_param_scheduler=None,
            num_floating_point_operations_so_far=0,
            prebuilt_state_dict=prebuilt_state_dict,
            pg_collection=pg_collection,
        )
    finally:
        if progress_monitor is not None:
            progress_monitor.stop()
    print(
        f"Checkpoint saved in {time.strftime('%H:%M:%S', time.gmtime(time.monotonic() - t0))}",
        flush=True,
    )

    if tokenizer_config is not None:
        tokenizer = build_tokenizer(tokenizer_config)
        checkpoint_name = get_checkpoint_name(str(path), 0, release=False)
        save_tokenizer_assets(tokenizer, tokenizer_config, checkpoint_name)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    auto_bridge, provider = build_single_rank_meta_provider(
        args.hf_model_path,
        trust_remote_code=True,
    )
    if not is_nvfp4_source(auto_bridge.hf_pretrained.config):
        raise ValueError("Source model is not an NVFP4 HuggingFace checkpoint")

    if hasattr(provider, "finalize"):
        provider.finalize()

    patch_meta_init_for_te_modules()

    trust_remote_code = getattr(auto_bridge.hf_pretrained, "trust_remote_code", False)
    tokenizer_kwargs = {"trust_remote_code": True} if trust_remote_code else None

    with temporary_distributed_context(backend="gloo"):
        stage_start = _log_stage_start("Building Megatron meta model...")
        megatron_model = provider.provide_distributed_model(
            wrap_with_ddp=False,
            use_cpu_initialization=True,
            init_model_with_meta_device=True,
            mixed_precision_wrapper=None,
        )
        _log_stage_done("Built Megatron meta model", stage_start)

        stage_start = _log_stage_start("Building conversion tasks...")
        conversion_tasks = auto_bridge._model_bridge.build_conversion_tasks(
            auto_bridge.hf_pretrained,
            megatron_model,
        )
        _log_stage_done(
            "Built conversion tasks",
            stage_start,
            extra=f"{len(conversion_tasks)} tasks",
        )

        if args.debug_layer_range is not None:
            start, end = args.debug_layer_range
            stage_start = _log_stage_start(
                f"Filtering conversion tasks for debug decoder layer range [{start}, {end})..."
            )
            original_task_count = len(conversion_tasks)
            conversion_tasks = _filter_conversion_tasks_for_debug_layer_range(
                conversion_tasks,
                args.debug_layer_range,
            )
            if not conversion_tasks:
                raise ValueError(f"Debug layer range [{start}, {end}) selected no conversion tasks.")
            _log_stage_done(
                "Filtered conversion tasks for debug layer range",
                stage_start,
                extra=f"{len(conversion_tasks)}/{original_task_count} tasks kept",
            )
            print(
                "Debug partial-conversion mode is enabled. "
                "The output checkpoint is only meant for conversion/save validation "
                "and is not expected to be loadable.",
                flush=True,
            )

        stage_start = _log_stage_start("Collecting NVFP4 target modules...")
        module_names = collect_nvfp4_target_module_names(
            conversion_tasks,
            auto_bridge.hf_pretrained.state,
            show_progress=True,
        )
        _log_stage_done(
            "Collected NVFP4 target modules",
            stage_start,
            extra=f"{len(module_names)} modules",
        )
        stage_start = _log_stage_start(f"Applying ModelOpt NVFP4 modules to {len(module_names)} target modules...")
        apply_modelopt_nvfp4_to_meta_model(
            megatron_model[0],
            module_names=module_names,
        )
        _log_stage_done(
            "Applied ModelOpt NVFP4 modules",
            stage_start,
            extra=f"{len(module_names)} modules",
        )

        pg_collection = get_pg_collection(megatron_model)
        stage_start = _log_stage_start("Building sharded state dict template...")
        model_template = megatron_model[0].sharded_state_dict(metadata={"dp_cp_group": pg_collection.dp_cp})
        _log_stage_done("Built sharded state dict template", stage_start)

        checkpoint_parent = Path(args.megatron_path).resolve().parent
        checkpoint_parent.mkdir(parents=True, exist_ok=True)
        spill_parent = Path(os.environ.get("MEGATRON_BRIDGE_DIRECT_SPILL_DIR", str(checkpoint_parent))).resolve()
        spill_parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(
            prefix=f".{Path(args.megatron_path).name}.nvfp4_spill_",
            dir=spill_parent,
        ) as spill_dir:
            spill_manager = TensorSpillManager(spill_dir)
            print(f"Using spill directory for direct tensors: {spill_dir}", flush=True)

            stage_start = _log_stage_start("Preparing direct NVFP4 model state dict...")
            model_state = build_nvfp4_direct_model_state_dict(
                auto_bridge._model_bridge,
                auto_bridge.hf_pretrained,
                megatron_model,
                model_template,
                conversion_tasks=conversion_tasks,
                spill_manager=spill_manager,
            )
            _log_stage_done("Prepared direct NVFP4 model state dict", stage_start)

            _save_direct_checkpoint(
                provider,
                args.megatron_path,
                model_state,
                model_list=megatron_model,
                pg_collection=pg_collection,
                hf_tokenizer_path=args.hf_model_path,
                hf_tokenizer_kwargs=tokenizer_kwargs,
            )

    print(f"Done. Direct NVFP4 Megatron checkpoint saved to: {args.megatron_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
