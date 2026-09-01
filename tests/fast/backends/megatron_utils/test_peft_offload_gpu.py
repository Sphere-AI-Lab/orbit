import pytest
import torch

from miles.backends.megatron_utils import peft_offload


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


class _TinyPeftModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.base_weight = torch.nn.Parameter(
            torch.arange(16, device="cuda", dtype=torch.float32).view(4, 4),
            requires_grad=False,
        )
        self.lora_a = torch.nn.Parameter(torch.ones(4, 2, device="cuda"))


def test_flat_peft_offload_round_trip_on_gpu() -> None:
    model = _TinyPeftModel()
    base_expected = model.base_weight.detach().cpu().clone()
    adapter_expected = model.lora_a.detach().cpu().clone()

    try:
        peft_offload.offload_megatron_frozen_base_to_cpu([model])
        torch.cuda.synchronize()
        assert model.base_weight.device.type == "cpu"
        assert model.lora_a.device.type == "cuda"
        torch.testing.assert_close(model.base_weight, base_expected)

        peft_offload.load_megatron_frozen_base_to_gpu([model])
        torch.cuda.synchronize()
        assert model.base_weight.device.type == "cuda"
        torch.testing.assert_close(model.base_weight.cpu(), base_expected)

        peft_offload.offload_megatron_adapter_to_cpu([model])
        torch.cuda.synchronize()
        assert model.lora_a.device.type == "cpu"
        torch.testing.assert_close(model.lora_a, adapter_expected)

        peft_offload.load_megatron_adapter_to_gpu([model])
        torch.cuda.synchronize()
        assert model.lora_a.device.type == "cuda"
        torch.testing.assert_close(model.lora_a.cpu(), adapter_expected)
    finally:
        peft_offload._FLAT_GROUPS.pop(id(model), None)
        peft_offload._ADAPTER_FLAT_GROUPS.pop(id(model), None)
