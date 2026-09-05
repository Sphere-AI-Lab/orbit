def force_native_forward_after_init(cls):
    if getattr(cls, "_orbit_force_native_ops_patched", False):
        return

    original_init = cls.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._forward_method = self.forward_native

    cls.__init__ = patched_init
    cls._orbit_force_native_ops_patched = True


def patch_sglang_native_ops():
    from sglang.srt.layers import activation, layernorm, rotary_embedding

    for cls in (
        activation.GeluAndMul,
        activation.SiluAndMul,
        layernorm.Gemma3RMSNorm,
        layernorm.GemmaRMSNorm,
        layernorm.RMSNorm,
        rotary_embedding.RotaryEmbedding,
    ):
        force_native_forward_after_init(cls)
