"""A run's wandb name must not be its group.

`init_wandb_primary` derived `name` from `--wandb-group`. That is fine for a
single run and wrong for a sweep: `sweep.py` sets the group to the METHOD, so
E4's seven FullFT arms share one group and its twenty-one LoRA arms share
another -- which is what makes the dashboard readable -- and every one of those
runs then also carried the name "full" or "lora". The learning rate, which is
the axis the sweep exists to vary, was visible only by opening a run's config.

`--wandb-run-name` separates the two. Default unchanged: no flag, name falls
back to the group.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _args(**overrides):
    base = dict(
        use_wandb=True,
        wandb_group="full",
        wandb_run_name=None,
        wandb_random_suffix=False,
        wandb_project="p",
        wandb_team=None,
        wandb_mode=None,
        wandb_key=None,
        wandb_host=None,
        wandb_dir=None,
        rank=0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _init_kwargs(args) -> dict:
    """Run `init_wandb_primary` with wandb stubbed, and return its init kwargs."""
    from miles.utils.tracking_utils import wandb_utils

    captured = {}

    def _fake_init(**kwargs):
        captured.update(kwargs)

    with patch.object(wandb_utils, "wandb") as fake_wandb:
        fake_wandb.init.side_effect = _fake_init
        fake_wandb.util.generate_id.return_value = "abc123"
        fake_wandb.Settings.side_effect = lambda **kw: kw
        # upstream now imports generate_id directly (from wandb.sdk.lib.runid), so
        # stubbing the wandb module attribute alone no longer pins the suffix.
        with patch.object(wandb_utils, "generate_id", return_value="abc123"), \
             patch.object(wandb_utils, "_init_wandb_common"), \
             patch.object(wandb_utils, "_compute_config_for_logging", return_value={}), \
             patch.object(wandb_utils, "_is_offline_mode", return_value=True):
            wandb_utils.init_wandb_primary(args)
    return captured


def test_the_name_still_falls_back_to_the_group():
    """The historical behaviour, preserved for every caller that passes no
    run name."""
    kwargs = _init_kwargs(_args())
    assert kwargs["group"] == "full"
    assert kwargs["name"] == "full"


def test_an_explicit_run_name_overrides_the_group():
    kwargs = _init_kwargs(_args(wandb_run_name="lora-r1-all-lr5e-06-s0"))
    assert kwargs["group"] == "full", "the group is untouched"
    assert kwargs["name"] == "lora-r1-all-lr5e-06-s0"


def test_arms_sharing_a_group_get_distinct_names():
    """The property the sweep needs: same group, one name each."""
    names = {
        _init_kwargs(_args(wandb_run_name=f"full-na-na-lr{lr}-s0"))["name"]
        for lr in ("5e-07", "1e-06", "3e-06", "7e-06", "2e-05", "4e-05", "0.0001")
    }
    assert len(names) == 7


def test_the_random_suffix_path_still_appends_the_rank():
    """Unchanged when no name is given: group gets an id, name gets the rank."""
    kwargs = _init_kwargs(_args(wandb_random_suffix=True))
    assert kwargs["group"] == "full_abc123"
    assert kwargs["name"] == "full_abc123-RANK_0"


def test_an_explicit_name_keeps_the_rank_under_random_suffix():
    """Two ranks writing one name would collide, so the rank stays appended
    even when the name is given."""
    kwargs = _init_kwargs(_args(wandb_random_suffix=True, wandb_run_name="arm", rank=3))
    assert kwargs["name"] == "arm-RANK_3"


def test_missing_attribute_does_not_raise():
    """Callers built before the flag existed pass an args object with no
    `wandb_run_name` at all."""
    args = _args()
    del args.wandb_run_name
    assert _init_kwargs(args)["name"] == "full"


@pytest.mark.parametrize("flag", ["--wandb-run-name"])
def test_the_flag_is_registered(flag):
    from pathlib import Path

    # The orbit-added arguments moved out of miles/utils/arguments.py and into
    # miles/orbit/arguments.py in the Phase-2 registration refactor
    # (docs/superpowers/plans/2026-08-29-phase2-arguments-registration.md).
    source = (Path(__file__).resolve().parents[3] / "miles" / "orbit" / "arguments.py").read_text(encoding="utf-8")
    assert f'"{flag}"' in source
