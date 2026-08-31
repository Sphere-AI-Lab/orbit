from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec


# ORBIT-SEAM: dropped upstream's `moe_use_legacy_grouped_gemm=` kwarg. It broke the
# `--megatron-to-hf-mode raw` path -- the DEFAULT mode, bit-rotted because every recipe
# passes `bridge` -- twice over: no option registers that argparse dest (AttributeError),
# and the pinned Megatron-LM no longer accepts the parameter on the TE/local spec
# builders at all (TypeError). It survives there only on the *inference* spec
# (gpt_layer_specs.py get_gpt_layer_with_inference_submodules). Deleting it is
# behaviour-neutral against a call that could not execute. Belongs upstream.
def get_glm_spec(args, config, vp_stage):
    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=args.num_experts,
        moe_grouped_gemm=args.moe_grouped_gemm,
        qk_layernorm=args.qk_layernorm,
        multi_latent_attention=args.multi_latent_attention,
        post_self_attn_layernorm=args.post_self_attn_layernorm,
        post_mlp_layernorm=args.post_mlp_layernorm,
    )
    return transformer_layer_spec
