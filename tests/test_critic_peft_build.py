import argparse
from types import SimpleNamespace

import pytest

import miles.backends.megatron_utils.model as model_mod
from orbit.megatron.bridge_peft_helpers import _bridge_is_value_model


def _args(**overrides):
    defaults = dict(
        peft_method="lora",
        megatron_to_hf_mode="bridge",
        advantage_estimator="ppo",
        use_critic=True,
        critic_mode="adapter",
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_bridge_is_value_model_for_critic_role():
    cfg = SimpleNamespace(architectures=["Qwen2ForCausalLM"])
    assert _bridge_is_value_model(cfg, role="critic")
    assert not _bridge_is_value_model(cfg, role="actor")


def test_bridge_is_value_model_for_classifier_architectures():
    cfg = SimpleNamespace(architectures=["Qwen2ForSequenceClassification"])
    assert _bridge_is_value_model(cfg, role="actor")


def _capture_peft_build(calls):
    def fake(args, role="actor"):
        calls["role"] = role
        return ["peft"]

    return fake


def test_build_model_routes_adapter_critic_to_peft_path(monkeypatch):
    calls = {}
    monkeypatch.setattr(model_mod, "_setup_peft_model_via_bridge", _capture_peft_build(calls))
    monkeypatch.setattr(model_mod, "get_model", lambda *a, **k: ["full"])
    assert model_mod._build_model(_args(), role="critic") == ["peft"]
    assert calls["role"] == "critic"


def test_build_model_keeps_full_mode_critic_on_full_path(monkeypatch):
    monkeypatch.setattr(
        model_mod, "_setup_peft_model_via_bridge", lambda args, role="actor": ["peft"]
    )
    monkeypatch.setattr(model_mod, "get_model", lambda *a, **k: ["full"])
    monkeypatch.setattr(model_mod, "get_model_provider_func", lambda *a, **k: None)
    assert model_mod._build_model(_args(critic_mode="full"), role="critic") == ["full"]


def test_build_model_actor_path_unchanged(monkeypatch):
    calls = {}
    monkeypatch.setattr(model_mod, "_setup_peft_model_via_bridge", _capture_peft_build(calls))
    assert model_mod._build_model(_args(), role="actor") == ["peft"]
    assert calls["role"] == "actor"
