from __future__ import annotations

from argparse import Namespace
from collections import defaultdict
from collections.abc import Callable, Iterable

import torch

from orbit.peft.megatron.state_mode import resolve_training_state_mode
from orbit.backends.megatron_utils.update_weight.common import is_named_adapter_tensor
from orbit.utils.tensor_backper import TensorBackuper

TensorSource = Callable[[], Iterable[tuple[str, torch.Tensor]]]


class AdapterStateManager:
    mode = "adapter"

    def __init__(self, source_getter: TensorSource):
        self._source_getter = source_getter
        self._backups: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)

    @property
    def backup_tags(self):
        return list(self._backups)

    def get(self, tag: str):
        return self._backups[tag]

    @torch.no_grad()
    def backup(self, tag: str) -> None:
        backup_dict = self._backups[tag]
        try:
            for name, tensor in self._source_getter():
                if not is_named_adapter_tensor(name):
                    continue
                backup_dict[name] = tensor.detach().cpu().clone()
        except Exception:
            self._backups.pop(tag, None)
            raise

    @torch.no_grad()
    def copy(self, *, src_tag: str, dst_tag: str):
        src = self._backups[src_tag]
        dst = self._backups[dst_tag]
        for name, tensor in src.items():
            dst[name] = tensor.clone()

    @torch.no_grad()
    def restore(self, tag: str) -> None:
        backup_dict = self._backups[tag]
        for name, tensor in self._source_getter():
            if name not in backup_dict:
                continue
            tensor.copy_(backup_dict[name], non_blocking=True)
        torch.cuda.synchronize()


class FullModelStateManager:
    mode = "full_model"

    def __init__(self, source_getter: TensorSource, single_tag: str | None):
        self._backuper = TensorBackuper.create(source_getter=source_getter, single_tag=single_tag)

    @property
    def backup_tags(self):
        return self._backuper.backup_tags

    def get(self, tag: str):
        return self._backuper.get(tag)

    def backup(self, tag: str) -> None:
        self._backuper.backup(tag)

    def copy(self, *, src_tag: str, dst_tag: str):
        self._backuper.copy(src_tag=src_tag, dst_tag=dst_tag)

    def restore(self, tag: str) -> None:
        self._backuper.restore(tag)


def create_model_state_manager(args: Namespace, source_getter: TensorSource, single_tag: str | None):
    if resolve_training_state_mode(args) == "adapter":
        return AdapterStateManager(source_getter=source_getter)
    return FullModelStateManager(source_getter=source_getter, single_tag=single_tag)
