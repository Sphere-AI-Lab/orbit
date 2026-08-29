from argparse import Namespace

import pytest

from orbit.peft.megatron.peft_utils import PeftSyncSpec
from orbit.backends.megatron_utils.update_weight.update_weight_from_tensor import (
    UpdateWeightFromTensor,
)
from orbit.utils.opd_teacher_spec import parse_teacher_spec, should_promote_teacher


def test_non_self_sources_never_promote():
    assert not should_promote_teacher("adapter", 1, 0)
    assert not should_promote_teacher("base", 1, 0)
    assert not should_promote_teacher("load", 1, 0)


def test_no_interval_never_promotes():
    assert not should_promote_teacher("self_ema", None, 0)


def test_promotes_at_startup_and_on_interval():
    # rollout_id 0 = startup promotion (fills the empty engine slot before
    # the first scored rollout).
    assert should_promote_teacher("self_ema", 3, 0)
    assert not should_promote_teacher("self_ema", 3, 1)
    assert not should_promote_teacher("self_ema", 3, 2)
    assert should_promote_teacher("self_ema", 3, 3)
    assert should_promote_teacher("self_lag", 1, 7)


def test_pool_mode_rejects_legacy_promotion_before_gather_or_transport():
    events = []
    updater = object.__new__(UpdateWeightFromTensor)
    updater._peft_args = Namespace(
        peft_method="lora",
        adapter_double_buffer=False,
        opd_teacher_spec=parse_teacher_spec("self:ema", None),
        ultra_teacher_pool_plan=object(),
    )
    updater._peft_sync_spec = PeftSyncSpec(
        method="lora",
        adapter_name="orbit_lora",
        adapter_config={},
        sync_transport="lora_adapter",
    )
    updater.weights_getter = lambda: events.append("gather")
    updater._hf_weight_iterator = None
    updater._peft_transport = None
    updater.weight_version = 11

    with pytest.raises(ValueError, match="legacy|teacher|destination|forbidden"):
        updater.push_teacher_adapter()

    assert events == []
    assert updater.weight_version == 11
