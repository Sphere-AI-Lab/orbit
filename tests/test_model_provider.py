from types import SimpleNamespace

from miles.backends.megatron_utils.model_provider import (
    LinearForLastLayer,
    replace_output_layer_with_value_head,
)


def test_replace_output_layer_with_value_head_uses_scalar_output():
    config = SimpleNamespace(hidden_size=8, sequence_parallel=False)
    model = SimpleNamespace(config=config, output_layer=None)

    replace_output_layer_with_value_head(model, config)

    assert isinstance(model.output_layer, LinearForLastLayer)
    assert model.output_layer.in_features == 8
    assert model.output_layer.out_features == 1
