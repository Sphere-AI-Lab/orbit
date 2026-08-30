import sys

import pytest

from tests.fast.launch_scripts.model_args_harness import expand_model_args


def _skip_unless_megatron_registers(*option_strings: str) -> None:
    """Skip when the vendored Megatron-LM does not register these CLI flags.

    orbit: these flags belong to Megatron's own parser, not to miles/orbit, and orbit
    pins a Megatron-LM commit that predates them (`--post-self-attn-layernorm`,
    `--post-mlp-layernorm`, and the YaRN trio `--original-max-position-embeddings`
    / `--beta-fast` / `--beta-slow`). On a Megatron that has them these tests run
    exactly as upstream wrote them; on orbit's pin argparse would SystemExit(2)
    before reaching the assertion under test, which says nothing about miles/orbit.
    Pair this skip with a Megatron pin bump, not with weakened assertions.
    """
    from megatron.training.arguments import parse_args as megatron_parse_args

    registered: set[str] = set()

    def _collect(parser):
        for action in parser._actions:
            registered.update(action.option_strings)
        # Raise to stop megatron's parse_args before it consumes sys.argv.
        raise _StopParsing

    try:
        megatron_parse_args(extra_args_provider=_collect)
    except _StopParsing:
        pass

    missing = [option for option in option_strings if option not in registered]
    if missing:
        pytest.skip(f"vendored Megatron-LM does not register {', '.join(missing)}")


class _StopParsing(Exception):
    pass


def test_post_layernorm_flags_propagate_to_megatron(monkeypatch):
    pytest.importorskip("megatron.training.arguments")
    _skip_unless_megatron_registers("--post-self-attn-layernorm", "--post-mlp-layernorm")

    import torch
    from megatron.training.arguments import core_transformer_config_from_args

    import miles.backends.megatron_utils.arguments as megatron_arguments
    import miles.utils.arguments as miles_arguments

    monkeypatch.setattr(miles_arguments, "miles_validate_args", lambda args: None)
    monkeypatch.setattr(megatron_arguments, "validate_args", lambda args: None)

    argv = [
        "pytest",
        "--train-backend",
        "megatron",
        "--rollout-batch-size",
        "1",
        "--num-layers",
        "1",
        "--hidden-size",
        "8",
        "--num-attention-heads",
        "1",
        "--post-self-attn-layernorm",
        "--post-mlp-layernorm",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    args = miles_arguments.parse_args()

    assert args.post_self_attn_layernorm is True
    assert args.post_mlp_layernorm is True

    if args.bf16:
        args.params_dtype = torch.bfloat16
    elif args.fp16:
        args.params_dtype = torch.float16
    else:
        args.params_dtype = torch.float32

    # apply_rope_fusion requires TransformerEngine >= 1.4, which is GPU-only
    # and not installed on CPU CI. This test only validates post-layernorm flag
    # propagation, so disable the fused kernel to avoid TransformerConfig
    # __post_init__ validation failure.
    args.apply_rope_fusion = False

    config = core_transformer_config_from_args(args)

    assert config.post_self_attn_layernorm is True
    assert config.post_mlp_layernorm is True


@pytest.mark.parametrize(
    ("model_type", "beta_fast"),
    [("kimi-k2", 1), ("kimi-k2-thinking", 1), ("kimi-k25", 32), ("kimi-k25_2layer", 32)],
)
def test_kimi_yarn_flags_propagate_to_megatron(monkeypatch, model_type, beta_fast):
    pytest.importorskip("megatron.training.arguments")
    _skip_unless_megatron_registers("--original-max-position-embeddings", "--beta-fast", "--beta-slow")

    import torch
    from megatron.core.transformer.transformer_config import MLATransformerConfig
    from megatron.training.arguments import core_transformer_config_from_args

    import miles.backends.megatron_utils.arguments as megatron_arguments
    import miles.utils.arguments as miles_arguments

    monkeypatch.setattr(miles_arguments, "miles_validate_args", lambda args: None)
    monkeypatch.setattr(megatron_arguments, "validate_args", lambda args: None)
    monkeypatch.setattr(
        sys,
        "argv",
        ["pytest", "--train-backend", "megatron", "--rollout-batch-size", "1", *expand_model_args(model_type)],
    )

    args = miles_arguments.parse_args()
    if args.bf16:
        args.params_dtype = torch.bfloat16
    elif args.fp16:
        args.params_dtype = torch.float16
    else:
        args.params_dtype = torch.float32

    # Fused MoE permutation requires Transformer Engine, which is not installed
    # on CPU CI and is unrelated to the YaRN configuration under test.
    args.moe_permute_fusion = False

    config = core_transformer_config_from_args(args)

    assert isinstance(config, MLATransformerConfig)
    assert config.rope_type == "yarn"
    assert config.original_max_position_embeddings == 4096
    assert config.beta_fast == beta_fast
    assert config.beta_slow == 1
