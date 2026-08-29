"""Gemma-4 bridge weight-sync tolerances (port of miles 6ccc2cab companion hunks).

Gemma-4 has params with no HF mapping (post_shared_expert_layernorm -> the
bridge yields a None conversion task) and buffer-like params (layer_scalar /
scale) that are absent from the optimizer-backed new_weight_dict.
_process_conversion_tasks must pass both through untouched instead of crashing
on the first weight-sync cycle.
"""

import dataclasses

from miles.backends.megatron_utils.update_weight.hf_weight_iterator_bridge import _process_conversion_tasks


@dataclasses.dataclass
class _Task:
    param_weight: object
    vp_stage: int
    param_name: str


def test_none_task_passes_through():
    out = list(_process_conversion_tasks([None], {}))
    assert out == [None]


def test_missing_weight_dict_key_keeps_task_untouched():
    task = _Task(param_weight=object(), vp_stage=0, param_name="decoder.layers.0.mlp.layer_scalar")
    out = list(_process_conversion_tasks([task], {}))
    assert out[0] is task


def test_param_weight_none_keeps_task_untouched():
    task = _Task(param_weight=None, vp_stage=0, param_name="decoder.layers.0.mlp.linear_fc1.weight")
    out = list(_process_conversion_tasks([task], {}))
    assert out[0] is task
