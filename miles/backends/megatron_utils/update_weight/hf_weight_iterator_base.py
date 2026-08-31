from abc import ABC, abstractmethod

# ORBIT-SEAM: orbit's peft_method-aware constructor lives in the home layer
# (orbit/megatron/hf_weight_iterator_ext.py); the class below lists its mixin first.
from orbit.megatron.hf_weight_iterator_ext import OrbitHfWeightIteratorExtensions


class HfWeightIteratorBase(OrbitHfWeightIteratorExtensions, ABC):
    @staticmethod
    def create(args, model, *, is_lora=False, **kwargs):
        from .hf_weight_iterator_bridge import HfWeightIteratorBridge
        from .hf_weight_iterator_direct import HfWeightIteratorDirect

        c = {
            "raw": HfWeightIteratorDirect,
            "bridge": HfWeightIteratorBridge,
        }[args.megatron_to_hf_mode]

        return c(args, model, is_lora=is_lora, **kwargs)

    @abstractmethod
    def get_hf_weight_chunks(self, megatron_local_weights):
        """
        Mental model of the API:
        megatron_model.to_hf_magically().named_parameters()
        """
        raise NotImplementedError
