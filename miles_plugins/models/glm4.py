from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec


# ORBIT-SEAM: upstream reads an argparse dest nothing registers -- no orbit, miles or
# pinned-Megatron option defines --moe-use-legacy-grouped-gemm -- so this raised
# AttributeError at actor init on the `--megatron-to-hf-mode raw` path, which is the
# DEFAULT mode (it bit-rotted because every recipe passes `bridge`). getattr restores
# Megatron's own default for the parameter (gpt_layer_specs.py: Optional[bool] = False),
# so this is a no-op for anyone not setting it. Belongs upstream; kept minimal here.
def get_glm_spec(args, config, vp_stage):
    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=args.num_experts,
        moe_grouped_gemm=args.moe_grouped_gemm,
        qk_layernorm=args.qk_layernorm,
        multi_latent_attention=args.multi_latent_attention,
        moe_use_legacy_grouped_gemm=getattr(args, "moe_use_legacy_grouped_gemm", False),
        post_self_attn_layernorm=args.post_self_attn_layernorm,
        post_mlp_layernorm=args.post_mlp_layernorm,
    )
    return transformer_layer_spec
