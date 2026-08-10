"""--critic-mode head build path: plain (non-PEFT) builder with the freeze
applied inside the provider, BEFORE the DDP wrap, so grad buffers and
optimizer state cover only the value head."""

import argparse

import torch

from orbit.backends.megatron_utils import model as model_mod

HIDDEN = 4


class _ToyCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = torch.nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.output_layer = torch.nn.Linear(HIDDEN, 1, bias=False)


def test_head_critic_provider_freezes_before_wrap():
    provider = lambda *a, **k: _ToyCritic()  # noqa: E731
    wrapped = model_mod._head_critic_provider(provider)
    module = wrapped()
    assert not module.trunk.weight.requires_grad
    assert module.output_layer.weight.requires_grad


def _head_args():
    return argparse.Namespace(
        use_critic=True,
        critic_mode="head",
        peft_method="none",
        megatron_to_hf_mode="bridge",
    )


def test_build_model_routes_head_critic_to_plain_builder(monkeypatch):
    captured = {}

    def fake_get_model(provider, _model_type):
        captured["module"] = provider()
        return [captured["module"]]

    monkeypatch.setattr(model_mod, "get_model", fake_get_model)
    monkeypatch.setattr(model_mod, "get_model_provider_func", lambda a, r: (lambda *x, **k: _ToyCritic()))
    monkeypatch.setattr(
        model_mod, "_setup_peft_model_via_bridge",
        lambda a, role: (_ for _ in ()).throw(AssertionError("head critic must not use the peft bridge")),
    )

    result = model_mod._build_model(_head_args(), role="critic")
    assert result == [captured["module"]]
    # the freeze happened inside the provider, before any wrap
    assert not captured["module"].trunk.weight.requires_grad
    assert captured["module"].output_layer.weight.requires_grad
