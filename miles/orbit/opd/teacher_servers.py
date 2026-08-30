"""OPD teacher server / pool config builders.

Home for the managed-teacher sglang model-config helpers lifted out of
miles/ray/rollout.py: parsing the ``--opd-teacher-pool`` manifest and
building the scoring-correctness ``ModelConfig`` for the single managed
teacher (``--opd-serve-teacher``) and for each served pool entry.
"""

from __future__ import annotations

OPD_TEACHER_MODEL_NAME = "opd_teacher"


def _opd_teacher_pool(args):
    """Parsed --opd-teacher-pool manifest, or None. Cached on args: the pool is
    read by placement sizing, engine injection, and validation."""
    path = getattr(args, "opd_teacher_pool", None)
    if path is None:
        return None
    cached = getattr(args, "_opd_teacher_pool_parsed", None)
    if cached is None:
        from miles.orbit.opd.opd_teacher_pool import parse_teacher_pool

        cached = parse_teacher_pool(path)
        args._opd_teacher_pool_parsed = cached
    return cached


def _teacher_server_overrides(
    mem_fraction: float | None,
    max_running_requests: int | None = None,
    max_prefill_tokens: int | None = None,
) -> dict:
    overrides = {
        "enable_return_hidden_states": True,
        "disable_radix_cache": True,
        "chunked_prefill_size": -1,
    }
    if mem_fraction is not None:
        overrides["mem_fraction_static"] = mem_fraction
    if max_running_requests is not None:
        overrides["max_running_requests"] = max_running_requests
    if max_prefill_tokens is not None:
        overrides["max_prefill_tokens"] = max_prefill_tokens
    return overrides


def _opd_teacher_model_config(args) -> "ModelConfig | None":
    """ModelConfig for the managed frozen OPD teacher (--opd-serve-teacher), or None.

    Served like any other sglang_config model -- own router, own ServerGroup, riding the
    same offload/health machinery -- but never weight-synced (update_weights=False). The
    scoring-correctness server flags are baked in: radix cache off (a cache hit skips the
    forward pass, so no hidden states / input logprobs for the matched prefix) and chunked
    prefill off (only the last chunk of a chunked prefill returns hidden states,
    sgl-project/sglang#8066).
    """
    if not getattr(args, "opd_serve_teacher", False):
        return None
    from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig

    return ModelConfig(
        name=OPD_TEACHER_MODEL_NAME,
        model_path=args.teacher_hf_checkpoint,
        update_weights=False,
        num_gpus_per_engine=args.opd_teacher_num_gpus,
        server_groups=[
            ServerGroupConfig(
                worker_type="regular",
                num_gpus=args.opd_teacher_num_gpus,
                overrides=_teacher_server_overrides(
                    args.opd_teacher_mem_fraction,
                    getattr(args, "opd_teacher_max_running_requests", None),
                    getattr(args, "opd_teacher_max_prefill_tokens", None),
                ),
            )
        ],
    )


def _opd_teacher_pool_model_configs(args) -> "list[ModelConfig]":
    """One sglang model entry per served pool teacher (--opd-teacher-pool)."""
    pool = _opd_teacher_pool(args)
    if pool is None:
        return []
    from miles.backends.sglang_utils.sglang_config import ModelConfig, ServerGroupConfig

    return [
        ModelConfig(
            name=entry.served_model_name,
            model_path=entry.model_path,
            update_weights=False,
            num_gpus_per_engine=entry.num_gpus_per_engine or entry.num_gpus,
            server_groups=[
                ServerGroupConfig(
                    worker_type="regular",
                    num_gpus=entry.num_gpus,
                    overrides=_teacher_server_overrides(entry.mem_fraction),
                )
            ],
        )
        for entry in pool.served
    ]
