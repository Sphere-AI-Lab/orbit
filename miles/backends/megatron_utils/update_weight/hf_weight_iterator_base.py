from abc import ABC, abstractmethod
from orbit.megatron.hf_weight_iterator_ext import OrbitHfWeightIteratorExtensions


# ORBIT-SEAM: __init__ (which carries orbit's peft_method plumbing) lives in the mixin
class HfWeightIteratorBase(OrbitHfWeightIteratorExtensions, ABC):
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

    @abstractmethod
    def get_hf_weight_chunks(self, megatron_local_weights, weight_type="base"):
        """
        Mental model of the API:
        megatron_model.to_hf_magically().named_parameters()
        """
        raise NotImplementedError
