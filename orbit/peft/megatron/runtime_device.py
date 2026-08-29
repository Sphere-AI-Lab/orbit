from __future__ import annotations

import torch

from orbit.peft.megatron.tensor_semantics import should_skip_named_tensor_for_meta_validation


def _resolve_target_device(target_device: torch.device | None) -> torch.device:
    if target_device is not None:
        return target_device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _move_tensor_to_device(tensor: torch.Tensor, target_device: torch.device) -> torch.Tensor:
    if tensor.device == target_device or tensor.device.type == "meta":
        return tensor
    return tensor.to(target_device, non_blocking=True)


def _remaining_meta_tensor_names(model_chunks) -> list[str]:
    remaining = []

    for model_chunk in model_chunks:
        for name, param in model_chunk.named_parameters():
            if param.device.type == "meta" and not should_skip_named_tensor_for_meta_validation(model_chunk, name, param):
                remaining.append(name)

        for name, buffer in model_chunk.named_buffers():
            if buffer.device.type == "meta" and not should_skip_named_tensor_for_meta_validation(
                model_chunk, name, buffer
            ):
                remaining.append(name)

        for module_name, module in model_chunk.named_modules():
            for attr_name, attr_value in vars(module).items():
                if attr_name in module._parameters or attr_name in module._buffers or attr_name in module._modules:
                    continue
                if not isinstance(attr_value, torch.Tensor) or attr_value.device.type != "meta":
                    continue
                full_name = f"{module_name}.{attr_name}" if module_name else attr_name
                remaining.append(full_name)

    return remaining


def _raise_if_meta_tensors_remain(model_chunks, target_device: torch.device) -> None:
    remaining = _remaining_meta_tensor_names(model_chunks)
    if not remaining:
        return

    sample = ", ".join(sorted(remaining)[:8])
    if len(remaining) > 8:
        sample += ", ..."
    raise ValueError(
        "Remaining meta tensors after runtime device finalization to "
        f"{target_device!s}: {sample}"
    )


def finalize_runtime_device(model_chunks, target_device: torch.device | None = None) -> int:
    model_chunks = list(model_chunks)
    target_device = _resolve_target_device(target_device)
    moved = 0

    for model_chunk in model_chunks:
        for module in model_chunk.modules():
            for name, param in list(module._parameters.items()):
                if param is None:
                    continue
                moved_param = _move_tensor_to_device(param.data, target_device)
                if moved_param.data_ptr() == param.data.data_ptr():
                    continue
                param.data = moved_param
                moved += 1

            for name, buffer in list(module._buffers.items()):
                if buffer is None:
                    continue
                moved_buffer = _move_tensor_to_device(buffer, target_device)
                if moved_buffer.data_ptr() == buffer.data_ptr():
                    continue
                module._buffers[name] = moved_buffer
                moved += 1

            for attr_name, attr_value in vars(module).items():
                if attr_name in module._parameters or attr_name in module._buffers or attr_name in module._modules:
                    continue
                if not isinstance(attr_value, torch.Tensor):
                    continue
                moved_attr = _move_tensor_to_device(attr_value, target_device)
                if moved_attr.data_ptr() == attr_value.data_ptr():
                    continue
                setattr(module, attr_name, moved_attr)
                moved += 1

    _raise_if_meta_tensors_remain(model_chunks, target_device)
    return moved
