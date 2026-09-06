from argparse import Namespace

import pytest


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ({"use_gloo_process_groups": True}, True),
        ({"use_gloo_process_groups": False}, False),
        ({"enable_gloo_process_groups": False, "use_gloo_process_groups": True}, False),
        ({"enable_gloo_process_groups": True}, True),
    ],
)
def test_gloo_flag_is_available_to_orbit_consumers(flags, expected):
    from orbit.backends.megatron_utils.arguments import set_default_megatron_args

    args = Namespace(
        optimizer="adam",
        fp16=False,
        seq_length=4096,
        vocab_size=None,
        tokenizer_model="unused",
        tokenizer_type="HuggingFaceTokenizer",
        **flags,
    )

    set_default_megatron_args(args)

    assert args.enable_gloo_process_groups is expected
