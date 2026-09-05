"""Runtime construction for the envpack Orbit plugin."""

import copy
import json
import os
from dataclasses import asdict, dataclass

from orbit_plugins.envpack_adapter.config import EnvpackAdapterConfig, EnvpackConfigError, EnvpackPoolConfig

_LOCAL_BUNDLES: dict[str, object] = {}
_SESSION_BUNDLES: dict[str, object] = {}


@dataclass(slots=True)
class SessionClientBundle:
    client: object
    env_configs_by_pool: dict[str, dict]

    def env_config(self, pool_id: str) -> dict:
        try:
            env_config = self.env_configs_by_pool[pool_id]
        except KeyError as exc:
            known = ", ".join(sorted(self.env_configs_by_pool))
            raise EnvpackConfigError(f"unknown envpack pool_id {pool_id!r}; known pools: {known}") from exc
        return copy.deepcopy(env_config)


def get_client_bundle(config: EnvpackAdapterConfig, args=None):
    if config.api == "in_process":
        return get_in_process_client_bundle(config, args=args)
    if config.api == "session":
        return get_session_client_bundle(config, args=args)
    raise EnvpackConfigError(f"unsupported envpack_adapter.api {config.api!r}")


def build_in_process_client(config: EnvpackAdapterConfig, args=None):
    if config.api != "in_process":
        raise EnvpackConfigError("build_in_process_client requires envpack_adapter.api=in_process")
    try:
        from envpack.client import EnvProfileRequest, build_local_client
    except Exception as exc:
        raise EnvpackConfigError(
            "envpack is not importable. Run `pip install -e thirdparty/envpack` "
            "or add the envpack repo to PYTHONPATH on every Orbit worker."
        ) from exc

    requests = [
        EnvProfileRequest(
            env_name=pool.env,
            profile_name=pool.profile,
            pool_id=pool.resolved_pool_id,
            env_config_overrides=pool.env_config,
            runtime_config_overrides=resolve_pool_runtime_demand(args, pool),
            factory=_resolve_factory(pool),
        )
        for pool in config.pools
    ]
    return build_local_client(requests)


def get_in_process_client_bundle(config: EnvpackAdapterConfig, args=None):
    """Return the process-local envpack client bundle for this config.

    The bundle owns the in-process Orchestrator and InstancePools. It must be
    shared across concurrent sample generations so pool capacity actually
    bounds active episodes.
    """

    key = _cache_key(config, args)
    bundle = _LOCAL_BUNDLES.get(key)
    if bundle is None:
        bundle = build_in_process_client(config, args=args)
        _LOCAL_BUNDLES[key] = bundle
    return bundle


def build_session_client(config: EnvpackAdapterConfig, args=None):
    if config.api != "session":
        raise EnvpackConfigError("build_session_client requires envpack_adapter.api=session")
    if not config.server:
        raise EnvpackConfigError("envpack_adapter.server is required when envpack_adapter.api=session")
    try:
        from envpack.client import RemoteEnvpackClient
    except Exception as exc:
        raise EnvpackConfigError(
            "envpack is not importable. Run `pip install -e thirdparty/envpack` "
            "or add the envpack repo to PYTHONPATH on every Orbit worker."
        ) from exc

    return SessionClientBundle(
        client=RemoteEnvpackClient(
            config.server,
            timeout_s=config.http.timeout_s,
            max_retries=config.http.max_retries,
            retry_backoff_s=config.http.retry_backoff_s,
            auth_token=_auth_token_from_env(config.http.auth_token_env),
        ),
        env_configs_by_pool={pool.resolved_pool_id: copy.deepcopy(pool.env_config) for pool in config.pools},
    )


def get_session_client_bundle(config: EnvpackAdapterConfig, args=None):
    """Return the process-local HTTP session client bundle for this config.

    The remote server owns runtime capacity and env lifecycle. The Orbit worker
    keeps only a lightweight HTTP client and the env_config view needed for
    metadata plus per-sample create requests.
    """

    key = _cache_key(config, args)
    bundle = _SESSION_BUNDLES.get(key)
    if bundle is None:
        bundle = build_session_client(config, args=args)
        _SESSION_BUNDLES[key] = bundle
    return bundle


def resolve_pool_runtime_demand(args, pool: EnvpackPoolConfig) -> dict:
    """Resolve runtime demand for a pool.

    The adapter reports Orbit' desired active episode throughput. Envpack is
    responsible for mapping that demand to num_instances/per-instance capacity
    according to the target environment profile.
    """

    runtime_config = dict(pool.runtime_config)
    capacity = resolve_orbit_active_episode_capacity(args)
    if capacity is None:
        return runtime_config

    explicit_layout = "num_instances" in runtime_config and "max_active_episodes_per_instance" in runtime_config
    if "desired_concurrency" not in runtime_config and not explicit_layout:
        runtime_config["desired_concurrency"] = capacity
    return runtime_config


def resolve_orbit_active_episode_capacity(args) -> int | None:
    if args is None:
        return None
    required = ("sglang_server_concurrency", "rollout_num_gpus", "rollout_num_gpus_per_engine")
    values = []
    for name in required:
        value = getattr(args, name, None)
        if value is None:
            return None
        values.append(int(value))

    sglang_server_concurrency, rollout_num_gpus, rollout_num_gpus_per_engine = values
    if sglang_server_concurrency < 1 or rollout_num_gpus < 1 or rollout_num_gpus_per_engine < 1:
        return None
    return max(1, sglang_server_concurrency * rollout_num_gpus // rollout_num_gpus_per_engine)


def _resolve_factory(pool: EnvpackPoolConfig):
    factory_name = pool.factory or "gym"
    if pool.env == "sokoban" and factory_name in {"gym", "sokoban_gym"}:
        from envpack.envs.sokoban import make_gym_sokoban_session

        return make_gym_sokoban_session
    if pool.env == "frozenlake" and factory_name in {"gym", "frozenlake_gym"}:
        from envpack.envs.frozenlake import make_gym_frozenlake_session

        return make_gym_frozenlake_session
    raise EnvpackConfigError(f"unsupported envpack factory {factory_name!r} for env {pool.env!r}")


def _cache_key(config: EnvpackAdapterConfig, args=None) -> str:
    payload = asdict(config)
    if config.api == "in_process":
        payload["resolved_runtime_config"] = [
            {
                "pool_id": pool.resolved_pool_id,
                "runtime_config": resolve_pool_runtime_demand(args, pool),
            }
            for pool in config.pools
        ]
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _auth_token_from_env(env_name: str | None) -> str | None:
    if not env_name:
        return None
    token = os.environ.get(env_name)
    return token or None
