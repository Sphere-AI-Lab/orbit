#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Direct-write HF FP8 -> Megatron checkpoint conversion for Qwen3 MoE."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any


def build_single_rank_meta_provider(*args, **kwargs):
    from megatron.bridge.orbit.low_precision.common import build_single_rank_meta_provider as _impl

    return _impl(*args, **kwargs)


def patch_meta_init_for_te_modules(*args, **kwargs):
    from megatron.bridge.orbit.low_precision.common import patch_meta_init_for_te_modules as _impl

    return _impl(*args, **kwargs)


def build_fp8_direct_model_state_dict(*args, **kwargs):
    from megatron.bridge.orbit.low_precision.fp8 import build_fp8_direct_model_state_dict as _impl

    return _impl(*args, **kwargs)


def apply_modelopt_fp8_to_meta_model(*args, **kwargs):
    from megatron.bridge.orbit.low_precision.fp8 import apply_modelopt_fp8_to_meta_model as _impl

    return _impl(*args, **kwargs)


def temporary_distributed_context(*args, **kwargs):
    from megatron.bridge.training.model_load_save import temporary_distributed_context as _impl

    return _impl(*args, **kwargs)


def get_pg_collection(*args, **kwargs):
    from megatron.bridge.training.utils.pg_utils import get_pg_collection as _impl

    return _impl(*args, **kwargs)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct-write HF FP8 -> Megatron checkpoint conversion",
    )
    parser.add_argument("--hf-model-path", required=True)
    parser.add_argument("--megatron-path", required=True)
    return parser.parse_args(argv)


def _format_elapsed(seconds: float) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def _format_num_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if value < 1024.0 or unit == "PiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _env_flag_enabled(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


class _SaveProgressMonitor:
    def __init__(self, checkpoint_root: str | Path, interval_sec: float = 10.0):
        self.checkpoint_root = Path(checkpoint_root)
        self.interval_sec = max(interval_sec, 1.0)
        self._start_time = 0.0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="fp8-direct-save-progress",
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
    from megatron.bridge.training.checkpointing import (
        get_checkpoint_name,
        save_checkpoint,
        save_tokenizer_assets,
    )
    from megatron.bridge.training.config import CheckpointConfig, ConfigContainer, LoggerConfig
    from megatron.bridge.training.state import GlobalState
    from megatron.bridge.training.tokenizers.config import TokenizerConfig
    from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
    from megatron.core.optimizer import OptimizerConfig

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
            storage_writers_per_rank=16,
        ),
        dist=None,
    )

    save_checkpoint(
        state=state,
        model=model_list,
        optimizer=None,
        opt_param_scheduler=None,
        num_floating_point_operations_so_far=0,
        prebuilt_state_dict={"checkpoint_version": 3.0, "iteration": 0, "model": model_state},
        pg_collection=pg_collection,
    )

    if tokenizer_config is not None:
        tokenizer = build_tokenizer(tokenizer_config)
        checkpoint_name = get_checkpoint_name(str(path), 0, release=False)
        save_tokenizer_assets(tokenizer, tokenizer_config, checkpoint_name)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print(f"Converting FP8 checkpoint directly: {args.hf_model_path} -> {args.megatron_path}")

    auto_bridge, provider = build_single_rank_meta_provider(args.hf_model_path)
    bridge = auto_bridge._model_bridge

    if hasattr(provider, "finalize"):
        provider.finalize()

    patch_meta_init_for_te_modules()

    with temporary_distributed_context(backend="gloo"):
        t0 = time.monotonic()
        print("Building Megatron meta model...", flush=True)
        meta_model = provider.provide_distributed_model(
            wrap_with_ddp=False,
            use_cpu_initialization=True,
            init_model_with_meta_device=True,
            mixed_precision_wrapper=None,
        )
        print(f"Built Megatron meta model in {time.monotonic() - t0:.2f}s", flush=True)

        apply_modelopt_fp8_to_meta_model(meta_model[0])
        pg_collection = get_pg_collection(meta_model)
        model_template = meta_model[0].sharded_state_dict(metadata={"dp_cp_group": pg_collection.dp_cp})
        model_state = build_fp8_direct_model_state_dict(
            bridge,
            auto_bridge.hf_pretrained,
            meta_model,
            model_template,
        )
        save_progress_monitor = _maybe_create_save_progress_monitor(args.megatron_path)
        if save_progress_monitor is not None:
            save_progress_monitor.start()
        try:
            _save_direct_checkpoint(
                provider,
                args.megatron_path,
                model_state,
                model_list=meta_model,
                pg_collection=pg_collection,
                hf_tokenizer_path=args.hf_model_path,
                hf_tokenizer_kwargs=None,
            )
        finally:
            if save_progress_monitor is not None:
                save_progress_monitor.stop()

    print(f"Done. Direct FP8 Megatron checkpoint saved to: {args.megatron_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
