"""PEFT train actors need expandable segments; full fine-tuning does not.

Measured 2026-08-06 on 8xB200, LoRA r1 gsm8k, rank 0 at `before update_weights`:
`allocated 0.09 GB` against `reserved 65.71 GB`, of which `inactive_split
65.62 GB` -- 100% of the gap -- spread over `segments 17`. That is a few MB of
straggler blocks pinning ~3.9 GB apiece. `empty_cache()` may only return a
segment that is entirely free, so it returns nothing every rollout, and the
colocated SGLang engine's `cuMemCreate` then fails at resume. On the 80 GB H100
that killed the arm at rollout 2; B200 only has more room to hide it.

`active_GB - allocated_GB` measured 0.00, so nothing was awaiting stream release
-- the gap is fragmentation alone, and expandable segments address exactly that
by mapping physical pages on demand instead of whole segments.

Full fine-tuning is deliberately left alone: its pool is tight (`reserved`
tracks `allocated` to within 0.07 GB, measured on the same node), it has no such
gap, and its arms are already producing completed 149/149 runs whose allocator
behaviour there is no reason to perturb mid-campaign.
"""

from __future__ import annotations

from argparse import Namespace

from miles.ray.train.actor_factory import _build_train_actor_env

_KEY = "PYTORCH_CUDA_ALLOC_CONF"


def _args(**overrides):
    """The finalised-argument surface `_build_train_actor_env` reads, nothing more."""
    base = dict(
        peft_method="none",
        train_env_vars={},
        train_backend="megatron",
        dumper_source_patcher_config_train=None,
    )
    base.update(overrides)
    return Namespace(**base)


def test_peft_actors_get_expandable_segments(monkeypatch):
    monkeypatch.delenv(_KEY, raising=False)

    env = _build_train_actor_env(_args(peft_method="lora"))

    assert env[_KEY] == "expandable_segments:True"


def test_oft_is_peft_too_and_gets_the_same_treatment(monkeypatch):
    """OFT takes the identical frozen-base offload route as LoRA and was
    measured hitting the same `func=resume` OOM, so it must not be excluded by
    a predicate that only recognises LoRA."""
    monkeypatch.delenv(_KEY, raising=False)

    env = _build_train_actor_env(_args(peft_method="oft"))

    assert env[_KEY] == "expandable_segments:True"


def test_full_finetuning_is_left_alone(monkeypatch):
    monkeypatch.delenv(_KEY, raising=False)

    env = _build_train_actor_env(_args(peft_method="none"))

    assert _KEY not in env


def test_an_explicit_shell_setting_wins(monkeypatch):
    """Someone pinning the allocator from the environment -- to compare against
    the un-fixed behaviour, or to select a different backend -- must not have it
    silently overwritten."""
    monkeypatch.setenv(_KEY, "max_split_size_mb:128")

    env = _build_train_actor_env(_args(peft_method="lora"))

    assert env[_KEY] == "max_split_size_mb:128"


def test_train_env_vars_beat_the_default(monkeypatch):
    monkeypatch.delenv(_KEY, raising=False)

    env = _build_train_actor_env(_args(peft_method="lora", train_env_vars={_KEY: "expandable_segments:False"}))

    assert env[_KEY] == "expandable_segments:False"
