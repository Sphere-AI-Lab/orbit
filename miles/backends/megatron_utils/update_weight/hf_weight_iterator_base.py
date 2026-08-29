from abc import ABC, abstractmethod


class HfWeightIteratorBase(ABC):
    @staticmethod
    # ORBIT-SEAM: peft_method supersedes is_lora so OFT routes like LoRA through the weight iterators (is_lora kept as compat alias)
    def create(args, model, *, peft_method="none", is_lora=None, **kwargs):
        from .hf_weight_iterator_bridge import HfWeightIteratorBridge
        from .hf_weight_iterator_direct import HfWeightIteratorDirect

        if is_lora is not None and peft_method == "none":
            peft_method = "lora" if is_lora else "none"

        c = {
            "raw": HfWeightIteratorDirect,
            "bridge": HfWeightIteratorBridge,
        }[args.megatron_to_hf_mode]

        return c(args, model, peft_method=peft_method, **kwargs)

    # ORBIT-SEAM: peft_method plumbed to subclasses
    def __init__(self, args, model, model_name, quantization_config, *, peft_method="none", is_lora=None):
        if is_lora is not None and peft_method == "none":
            peft_method = "lora" if is_lora else "none"

        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.peft_method = peft_method
        self.is_lora = peft_method == "lora"

    @abstractmethod
    def get_hf_weight_chunks(self, megatron_local_weights):
        """
        Mental model of the API:
        megatron_model.to_hf_magically().named_parameters()
        """
        raise NotImplementedError
