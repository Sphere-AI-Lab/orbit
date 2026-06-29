"""Configuration parsing for the envpack Miles plugin."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal


class EnvpackConfigError(ValueError):
    pass


POOL_ENV_CONFIG_RENDER_ONLY_ALLOWLIST: dict[str, frozenset[str]] = {
    "sokoban": frozenset({"sokoban_render_style", "tiny_scale", "raw_plane_scale"}),
    "frozenlake": frozenset(),
}


ApiMode = Literal["in_process", "session"]


@dataclass(frozen=True, slots=True)
class EnvpackPoolConfig:
    env: str
    profile: str = "default_local"
    pool_id: str | None = None
    env_config: dict[str, Any] = field(default_factory=dict)
    runtime_config: dict[str, Any] = field(default_factory=dict)
    factory: str | None = None

    @property
    def resolved_pool_id(self) -> str:
        return self.pool_id or f"{self.env}:{self.profile}"


@dataclass(frozen=True, slots=True)
class EnvpackRolloutConfig:
    max_turns: int | None = None
    response_length_per_turn: int | None = None


@dataclass(frozen=True, slots=True)
class EnvpackHttpConfig:
    timeout_s: float = 60.0
    max_retries: int = 3
    retry_backoff_s: float = 0.25
    auth_token_env: str | None = "ENVPACK_AUTH_TOKEN"


@dataclass(frozen=True, slots=True)
class EnvpackRefillConfig:
    max_attempts: int = 3
    backoff_s: float = 0.5


@dataclass(frozen=True, slots=True)
class EnvpackCurriculumStage:
    until: int | None
    solve_steps: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EnvpackCurriculumConfig:
    enabled: bool = False
    stages: tuple[EnvpackCurriculumStage, ...] = ()


@dataclass(frozen=True, slots=True)
class EnvpackAdapterConfig:
    api: ApiMode
    pools: tuple[EnvpackPoolConfig, ...]
    rollout: EnvpackRolloutConfig
    http: EnvpackHttpConfig = field(default_factory=EnvpackHttpConfig)
    refill: EnvpackRefillConfig = field(default_factory=EnvpackRefillConfig)
    curriculum: EnvpackCurriculumConfig = field(default_factory=EnvpackCurriculumConfig)
    server: str | None = None
    reject_group_rm: bool = True
    reject_partial_rollout: bool = True
    reject_external_rm: bool = True
    allow_unimplemented_generate: bool = False

    def pool_for_env(self, env_name: str) -> EnvpackPoolConfig:
        matches = [pool for pool in self.pools if pool.env == env_name]
        if not matches:
            known = ", ".join(pool.env for pool in self.pools)
            raise EnvpackConfigError(f"no envpack pool for env {env_name!r}; known envs: {known}")
        if len(matches) > 1:
            known = ", ".join(pool.resolved_pool_id for pool in matches)
            raise EnvpackConfigError(f"multiple pools for env {env_name!r}; select pool_id explicitly: {known}")
        return matches[0]

    def pool_by_id(self, pool_id: str) -> EnvpackPoolConfig:
        for pool in self.pools:
            if pool.resolved_pool_id == pool_id:
                return pool
        known = ", ".join(pool.resolved_pool_id for pool in self.pools)
        raise EnvpackConfigError(f"unknown envpack pool_id {pool_id!r}; known pools: {known}")


def load_envpack_config(args) -> EnvpackAdapterConfig:
    label, raw = _raw_adapter_config(args)
    if raw is None:
        raise EnvpackConfigError(
            "missing `envpack_adapter` config. Add it to --custom-config-path before using "
            "miles_plugins.envpack_adapter.*"
        )
    if not isinstance(raw, dict):
        raise EnvpackConfigError(f"`{label}` config must be a dict, got {type(raw).__name__}")
    _reject_unknown(
        raw,
        {
            "api",
            "server",
            "env",
            "profile",
            "pool_id",
            "pools",
            "rollout",
            "http",
            "refill",
            "curriculum",
            "guards",
        },
        label,
    )

    api = raw.get("api", "in_process")
    if api not in {"in_process", "session"}:
        raise EnvpackConfigError(f"{label}.api must be one of: in_process, session")
    server = raw.get("server")
    if api == "session" and not server:
        raise EnvpackConfigError(f"{label}.server is required when {label}.api=session")
    if api == "in_process" and server:
        raise EnvpackConfigError(f"{label}.server is only valid when {label}.api=session")

    pools = _parse_pools(raw, label)
    rollout = _parse_rollout(raw.get("rollout") or {}, label)
    http = _parse_http(raw.get("http") or {}, label)
    refill = _parse_refill(raw.get("refill") or {}, label)
    curriculum = _parse_curriculum(raw.get("curriculum") or {}, label)
    guards = _parse_guards(raw.get("guards") or {}, label)
    return EnvpackAdapterConfig(
        api=api,
        server=server,
        pools=tuple(pools),
        rollout=rollout,
        http=http,
        refill=refill,
        curriculum=curriculum,
        **guards,
    )


def _raw_adapter_config(args) -> tuple[str, Any]:
    raw_new = getattr(args, "envpack_adapter", None)
    raw_legacy = getattr(args, "envpack", None)
    if raw_new is not None and raw_legacy is not None:
        raise EnvpackConfigError("use either `envpack_adapter` or legacy `envpack` config, not both")
    if raw_new is not None:
        return "envpack_adapter", raw_new
    return "envpack", raw_legacy


def validate_runtime_args(args, config: EnvpackAdapterConfig) -> None:
    if config.reject_partial_rollout and getattr(args, "partial_rollout", False):
        raise EnvpackConfigError("envpack adapter MVP does not support partial_rollout")
    if config.reject_group_rm and getattr(args, "group_rm", False):
        raise EnvpackConfigError("envpack adapter MVP expects envpack/rubric reward, not group_rm")
    if config.reject_external_rm:
        if getattr(args, "rm_type", None):
            raise EnvpackConfigError("envpack adapter MVP rejects rm_type; reward comes from envpack")
        if getattr(args, "custom_rm_path", None):
            raise EnvpackConfigError("envpack adapter MVP rejects custom_rm_path; reward comes from envpack")
    if getattr(args, "rollout_external", False):
        raise EnvpackConfigError("envpack adapter is a custom_generate path, not Miles rollout_external")
    middleware_paths = getattr(args, "miles_router_middleware_paths", None) or []
    if getattr(args, "use_miles_router", False) and any("RadixTreeMiddleware" in path for path in middleware_paths):
        raise EnvpackConfigError(
            "envpack adapter MVP does not support Miles RadixTreeMiddleware postprocessing; "
            "disable the middleware or add multi-turn envpack-aware postprocessing first"
        )
    if getattr(args, "use_rollout_routing_replay", False):
        raise EnvpackConfigError(
            "envpack generate path does not preserve R3 yet; disable use_rollout_routing_replay "
            "or implement routed-expert delegation before training"
        )


def validate_pool_env_config_overrides(
    env_name: str, pool_env_config: dict[str, Any], *, context: str
) -> dict[str, Any]:
    overrides = copy.deepcopy(pool_env_config or {})
    if not overrides:
        return overrides
    env_name = _normalize_env_name(env_name)
    allowed = POOL_ENV_CONFIG_RENDER_ONLY_ALLOWLIST.get(env_name, frozenset())
    unsafe = sorted(set(overrides) - allowed)
    if unsafe:
        raise EnvpackConfigError(
            f"{context} cannot apply structural pool.env_config overrides for env {env_name!r}: {unsafe}. "
            "pool.env_config is reserved for render-only launch overrides. Move task-shaping keys into "
            "samples.jsonl metadata.envpack.env_config or EnvSpec envs[].config."
        )
    return overrides


def _parse_pools(raw: dict[str, Any], label: str) -> list[EnvpackPoolConfig]:
    if raw.get("pools") is not None:
        if raw.get("env") is not None:
            raise EnvpackConfigError(f"use either {label}.env or {label}.pools, not both")
        pool_items = raw["pools"]
        if not isinstance(pool_items, list) or not pool_items:
            raise EnvpackConfigError(f"{label}.pools must be a non-empty list")
        return [_parse_pool(item, f"{label}.pools[{idx}]") for idx, item in enumerate(pool_items)]

    env_name = raw.get("env")
    if env_name is None:
        raise EnvpackConfigError(f"{label}.env or {label}.pools is required")
    return [
        _parse_pool(
            {
                "env": env_name,
                "profile": raw.get("profile", "default_local"),
                "pool_id": raw.get("pool_id"),
            },
            label,
        )
    ]


def _parse_pool(raw: dict[str, Any], label: str) -> EnvpackPoolConfig:
    if not isinstance(raw, dict):
        raise EnvpackConfigError(f"{label} must be a dict")
    _reject_unknown(raw, {"env", "profile", "pool_id", "env_config", "runtime_config", "factory"}, label)
    env_name = _normalize_env_name(raw.get("env"))
    return EnvpackPoolConfig(
        env=env_name,
        profile=str(raw.get("profile", "default_local")),
        pool_id=None if raw.get("pool_id") is None else str(raw["pool_id"]),
        env_config=dict(raw.get("env_config") or {}),
        runtime_config=dict(raw.get("runtime_config") or {}),
        factory=None if raw.get("factory") is None else str(raw["factory"]),
    )


def _parse_rollout(raw: dict[str, Any], label: str) -> EnvpackRolloutConfig:
    if not isinstance(raw, dict):
        raise EnvpackConfigError(f"{label}.rollout must be a dict")
    _reject_unknown(raw, {"max_turns", "response_length_per_turn"}, f"{label}.rollout")
    max_turns = raw.get("max_turns")
    response_length_per_turn = raw.get("response_length_per_turn")
    return EnvpackRolloutConfig(
        max_turns=None if max_turns is None else int(max_turns),
        response_length_per_turn=None if response_length_per_turn is None else int(response_length_per_turn),
    )


def _parse_http(raw: dict[str, Any], label: str) -> EnvpackHttpConfig:
    if not isinstance(raw, dict):
        raise EnvpackConfigError(f"{label}.http must be a dict")
    _reject_unknown(raw, {"timeout_s", "max_retries", "retry_backoff_s", "auth_token_env"}, f"{label}.http")
    timeout_s = float(raw.get("timeout_s", 60.0))
    max_retries = int(raw.get("max_retries", 3))
    retry_backoff_s = float(raw.get("retry_backoff_s", 0.25))
    raw_auth_token_env = raw.get("auth_token_env", "ENVPACK_AUTH_TOKEN")
    auth_token_env = None if raw_auth_token_env in {None, ""} else str(raw_auth_token_env)
    if timeout_s <= 0:
        raise EnvpackConfigError(f"{label}.http.timeout_s must be > 0")
    if max_retries < 0:
        raise EnvpackConfigError(f"{label}.http.max_retries must be >= 0")
    if retry_backoff_s < 0:
        raise EnvpackConfigError(f"{label}.http.retry_backoff_s must be >= 0")
    return EnvpackHttpConfig(
        timeout_s=timeout_s,
        max_retries=max_retries,
        retry_backoff_s=retry_backoff_s,
        auth_token_env=auth_token_env,
    )


def _parse_refill(raw: dict[str, Any], label: str) -> EnvpackRefillConfig:
    if not isinstance(raw, dict):
        raise EnvpackConfigError(f"{label}.refill must be a dict")
    _reject_unknown(raw, {"max_attempts", "backoff_s"}, f"{label}.refill")
    max_attempts = int(raw.get("max_attempts", 3))
    backoff_s = float(raw.get("backoff_s", 0.5))
    if max_attempts < 1:
        raise EnvpackConfigError(f"{label}.refill.max_attempts must be >= 1")
    if backoff_s < 0:
        raise EnvpackConfigError(f"{label}.refill.backoff_s must be >= 0")
    return EnvpackRefillConfig(max_attempts=max_attempts, backoff_s=backoff_s)


def _parse_curriculum(raw: dict[str, Any], label: str) -> EnvpackCurriculumConfig:
    if not isinstance(raw, dict):
        raise EnvpackConfigError(f"{label}.curriculum must be a dict")
    _reject_unknown(raw, {"enabled", "stages"}, f"{label}.curriculum")
    enabled = bool(raw.get("enabled", False))
    raw_stages = raw.get("stages") or ()
    if not enabled:
        if raw_stages:
            raise EnvpackConfigError(f"{label}.curriculum.stages requires enabled: true")
        return EnvpackCurriculumConfig()
    if not isinstance(raw_stages, list) or not raw_stages:
        raise EnvpackConfigError(f"{label}.curriculum.stages must be a non-empty list when enabled")

    stages: list[EnvpackCurriculumStage] = []
    last_until = -1
    for idx, item in enumerate(raw_stages):
        if not isinstance(item, dict):
            raise EnvpackConfigError(f"{label}.curriculum.stages[{idx}] must be a dict")
        _reject_unknown(item, {"until", "solve_steps"}, f"{label}.curriculum.stages[{idx}]")
        raw_until = item.get("until")
        until = None if raw_until is None else int(raw_until)
        if until is not None:
            if until <= last_until:
                raise EnvpackConfigError(f"{label}.curriculum.stages[{idx}].until must be increasing")
            if until < 1:
                raise EnvpackConfigError(f"{label}.curriculum.stages[{idx}].until must be >= 1")
            last_until = until
        elif idx != len(raw_stages) - 1:
            raise EnvpackConfigError(f"{label}.curriculum only allows until: null on the final stage")

        raw_steps = item.get("solve_steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise EnvpackConfigError(f"{label}.curriculum.stages[{idx}].solve_steps must be a non-empty list")
        solve_steps = tuple(sorted({int(value) for value in raw_steps}))
        if any(value < 0 for value in solve_steps):
            raise EnvpackConfigError(f"{label}.curriculum.stages[{idx}].solve_steps must be non-negative")
        stages.append(EnvpackCurriculumStage(until=until, solve_steps=solve_steps))

    if stages[-1].until is not None:
        raise EnvpackConfigError(f"{label}.curriculum final stage must use until: null")
    return EnvpackCurriculumConfig(enabled=True, stages=tuple(stages))


def _parse_guards(raw: dict[str, Any], label: str) -> dict[str, bool]:
    if not isinstance(raw, dict):
        raise EnvpackConfigError(f"{label}.guards must be a dict")
    known = {"reject_group_rm", "reject_partial_rollout", "reject_external_rm", "allow_unimplemented_generate"}
    _reject_unknown(raw, known, f"{label}.guards")
    return {key: bool(raw.get(key, True if key.startswith("reject_") else False)) for key in known}


def _normalize_env_name(name: Any) -> str:
    if not name:
        raise EnvpackConfigError("env name is required")
    normalized = str(name).strip().lower()
    aliases = {
        "sokoban": "sokoban",
        "frozenlake": "frozenlake",
        "frozen_lake": "frozenlake",
    }
    if normalized not in aliases:
        raise EnvpackConfigError(f"unsupported envpack env {name!r}")
    return aliases[normalized]


def _reject_unknown(raw: dict[str, Any], known: set[str], label: str) -> None:
    unknown = sorted(set(raw) - known)
    if unknown:
        raise EnvpackConfigError(f"unknown keys in {label}: {unknown}")
