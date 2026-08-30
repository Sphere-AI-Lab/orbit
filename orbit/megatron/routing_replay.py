"""Orbit's R3 (routing replay) wiring for Megatron MoE routers.

Routes Megatron's per-layer ``TopKRouter``/``DeepSeekV4Gate`` replay hook through
orbit's ``routing_replay_manager`` instead of the base's inference-only
``RouterReplay``, plus the post-build pass that installs it on every non-MTP MoE
router. ``wire_routing_replay_to_models`` is called once after
``initialize_model_and_optimizer``.

Lifted out of ``miles/backends/megatron_utils/replay_utils.py`` (which is now
byte-pristine again): this is orbit-authored code that never needed to live
inside the vendored tree -- nothing upstream references it.
"""

from __future__ import annotations

import logging

from miles.utils.replay_base import routing_replay_manager

logger = logging.getLogger(__name__)


class _OrbitRoutingReplayAdapter:
    """Matches Megatron's ``RouterReplay.get_replay_topk`` signature but delegates
    record/replay to orbit's ``routing_replay_manager``. Stateless; a single
    instance is shared across every ``TopKRouter`` — per-layer state lives in
    the manager's ``replays`` list, selected by the forward pre-hook installed
    by ``register_to_module``."""

    __slots__ = ()

    def get_replay_topk(self, scores, topk, num_groups, group_topk, _compute_topk):
        new_fn = routing_replay_manager.get_topk_fn(_compute_topk, return_probs=True)
        return new_fn(scores, topk, num_groups=num_groups, group_topk=group_topk)


_R3_ADAPTER = _OrbitRoutingReplayAdapter()


def wire_routing_replay_to_models(models) -> None:
    """Post-build R3 wiring. Call once after ``initialize_model_and_optimizer``.

    Walks each MoE ``TopKRouter`` in ``models`` and registers it with
    ``routing_replay_manager``. Routers inside MTP MoE layers (where the
    enclosing ``MoELayer.is_mtp_layer`` is True) are skipped — MTP uses fresh
    routing, never replay.

    This avoids monkey-patching ``TopKRouter.__init__``: the wiring is a clean
    post-build pass over the model graph.
    """
    if not getattr(routing_replay_manager, "enabled", False):
        return

    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.moe.router import TopKRouter

    try:
        from megatron.core.transformer.experimental_attention_variant.deepseek_v4 import DeepSeekV4Gate
    except Exception:
        DeepSeekV4Gate = None

    if not isinstance(models, (list, tuple)):
        models = [models]

    wired = 0
    for model in models:
        root = model.module if hasattr(model, "module") else model
        # Build {name: module} so we can look up the enclosing MoELayer.
        named = dict(root.named_modules())
        for name, module in named.items():
            is_standard_router = isinstance(module, TopKRouter)
            is_dsv4_gate = DeepSeekV4Gate is not None and isinstance(module, DeepSeekV4Gate)
            if not (is_standard_router or is_dsv4_gate):
                continue
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            parent = named.get(parent_name)
            if is_standard_router and isinstance(parent, MoELayer) and getattr(parent, "is_mtp_layer", False):
                continue
            routing_replay_manager.register_to_module(module, "routing_replay")
            # Route Megatron's existing `router_replay` flow through orbit's
            # manager. ``self.router_replay = None`` from upstream init gets
            # overwritten so the inference-only ``RouterReplay`` is bypassed.
            # DSV4's custom gate also checks this attribute directly.
            module.router_replay = _R3_ADAPTER
            wired += 1

    if wired:
        logger.info("megatron R3: wired %d MoE TopKRouter(s) to routing_replay_manager", wired)
