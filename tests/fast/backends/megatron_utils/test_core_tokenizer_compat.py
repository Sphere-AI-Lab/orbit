from argparse import Namespace

import pytest


@pytest.mark.parametrize(
    ("vocab_size", "tp_size", "expected"),
    [(151936, 1, 151936), (151937, 1, 152064), (257, 2, 512)],
)
def test_vocab_padding_imports_with_current_core(vocab_size, tp_size, expected):
    from orbit.backends.megatron_utils.arguments import _vocab_size_with_padding

    args = Namespace(make_vocab_size_divisible_by=128, tensor_model_parallel_size=tp_size, rank=0)

    assert _vocab_size_with_padding(vocab_size, args) == expected
