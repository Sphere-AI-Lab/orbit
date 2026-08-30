import dataclasses
import ipaddress
import logging
import multiprocessing
import os
import time
from urllib.parse import quote

import ray
import requests
import sglang_router
from packaging.version import parse
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from urllib3.exceptions import NewConnectionError

# ORBIT-SEAM: lora_utils' convert_target_modules_to_hf/is_lora_enabled replaced by
# orbit.megatron.peft_utils' unified (LoRA+OFT) equivalents below; upstream's
# lora_base_cpu_backup_enabled is re-anchored into the peft dispatch further down
from miles.backends.megatron_utils.lora_utils import lora_base_cpu_backup_enabled
from orbit.megatron.oft_utils import OFT_ADAPTER_NAME
from orbit.megatron.peft_utils import convert_target_modules_to_hf, get_peft_method
from miles.ray.ray_actor import RayActor
from miles.ray.rollout.sglang_server_actor import SGLangServerActor
from miles.utils.env_report import collect_and_print_node_env_report
from miles.utils.http_utils import get_host_info
# ORBIT-SEAM: upstream's lora_rollout_enabled / is_multi_lora_enabled drove the enable_lora
# (multi-tenant LoRAManager) branches of _compute_server_args that orbit's peft_method dispatch
# replaces, so they are not imported here
from miles.utils.lora import LORA_ADAPTER_NAME

# ORBIT-SEAM: launch-env, MoE-parity and shm-refcount helpers moved to orbit/sglang/
# (P1 lift-out, Phase 3 slice 3c); imported here both to call from the base functions
# that stay in this file (launch_server_process, _init_normal, _compute_server_args)
# and to re-export the names tests import directly from this module.
from orbit.sglang.launch import (
    _configure_peft_cache_kwargs,
    _launch_server_with_orbit_compat,
    _prepare_child_native_ops_env,
    _prepare_child_peft_cache_env,  # noqa: F401  (re-exported: tests/test_sglang_native_ops.py)
)
from orbit.sglang.server_args import _configure_megatron_moe_parity_kwargs, _training_adapter_dtype_arg
from orbit.sglang.shm_refcounts import _balance_broadcast_shm_refcounts  # noqa: F401  (re-exported: tests/test_peft_broadcast_shm_refcount.py)
# ORBIT-SEAM: orbit-added adapter/teacher SGLangEngine methods moved to this home mixin (P2, Phase 3 slice 3c)
from orbit.sglang.engine_ext import OrbitEngineExtensions

logger = logging.getLogger(__name__)


def get_base_gpu_id(args, rank):
    num_gpus = min(args.num_gpus_per_node, args.rollout_num_gpus_per_engine)
    if args.colocate:
        start_index = (rank * num_gpus) % args.num_gpus_per_node
    else:
        num_actor_gpus = 0 if args.debug_rollout_only else args.actor_num_gpus_per_node * args.actor_num_nodes
        start_index = (num_actor_gpus + rank * num_gpus) % args.num_gpus_per_node
    return start_index


def _to_local_gpu_id(physical_gpu_id: int) -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("HIP_VISIBLE_DEVICES")
    if not cvd:
        return physical_gpu_id  # no remapping
    # CUDA_VISIBLE_DEVICES can be like "4,5,6,7"
    visible = [int(x) for x in cvd.split(",") if x.strip() != ""]
    # In a remapped process, valid torch device indices are 0..len(visible)-1
    if physical_gpu_id in visible:
        return visible.index(physical_gpu_id)
    # If we're already getting local IDs, allow them
    if 0 <= physical_gpu_id < len(visible):
        return physical_gpu_id
    raise RuntimeError(
        f"GPU id {physical_gpu_id} is not valid under CUDA_VISIBLE_DEVICES={cvd}. "
        f"Expected one of {visible} (physical) or 0..{len(visible)-1} (local)."
    )


def _get_gpu_uuids(gpu_ids: list[int]) -> list[str | None]:
    """Best-effort NVML UUIDs so the dashboard can reconcile GPU index
    spaces across processes; None entries when NVML is unavailable."""
    try:
        import pynvml

        pynvml.nvmlInit()
        return [str(pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(i))) for i in gpu_ids]
    except Exception:
        return [None] * len(gpu_ids)


# ORBIT-SEAM: launch_server_process gained a force_native_ops passthrough and now
# spawns the orbit-compat entrypoint below (native-ops env prep + PEFT radix-cache
# env prep, both homed in orbit/sglang/launch.py) instead of sglang's launch_server
# directly; the host-bracket-strip that used to happen here moved into _init_normal
# (ServerArgs is read-only after __post_init__ resolves it in sglang v0.5.18)
def launch_server_process(server_args: ServerArgs, force_native_ops: bool = False) -> multiprocessing.Process:

    multiprocessing.set_start_method("spawn", force=True)
    _prepare_child_native_ops_env(force_native_ops)
    p = multiprocessing.Process(target=_launch_server_with_orbit_compat, args=(server_args, force_native_ops))
    p.start()

    if server_args.node_rank != 0:
        return

    _wait_server_healthy(
        base_url=server_args.url(),
        api_key=server_args.api_key,
        is_process_alive=lambda: p.is_alive(),
    )

    return p


def _launch_sglang_server(server_args: ServerArgs, bundle_indices: list[int]):
    """Host the Ray HTTP server in a same-job child actor. Returns (actor, scheduler_actors)."""
    placement_group = ray.util.get_current_placement_group()
    assert placement_group is not None
    http_actor = (
        ray.remote(SGLangServerActor)
        .options(
            num_cpus=0.2,
            num_gpus=0,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=placement_group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=bundle_indices[0],
            ),
        )
        .remote()
    )
    scheduler_actors = ray.get(http_actor.start.remote(server_args, bundle_indices=bundle_indices))
    _wait_server_healthy(
        base_url=server_args.url(),
        api_key=server_args.api_key,
        is_process_alive=lambda: ray.get(http_actor.is_alive.remote()),
    )
    return http_actor, scheduler_actors


def _wait_server_healthy(base_url, api_key, is_process_alive):
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "Authorization": f"Bearer {api_key}",
    }

    with requests.Session() as session:
        while True:
            try:
                response = session.get(f"{base_url}/health_generate", headers=headers)
                if response.status_code == 200:
                    break
            except requests.RequestException:
                pass

            if not is_process_alive():
                raise Exception("Server process terminated unexpectedly.")

            time.sleep(2)

        # use flush_cache to make sure the working queue is empty, so that we can do offload
        while True:
            try:
                response = session.get(f"{base_url}/flush_cache", headers=headers)
                if response.status_code == 200:
                    break

            except requests.RequestException:
                pass

            if not is_process_alive():
                raise Exception("Server process terminated unexpectedly.")

            time.sleep(2)


# ORBIT-SEAM: orbit adapter/teacher engine methods live in the home mixin (P2, Phase 3 slice 3c)
class SGLangEngine(OrbitEngineExtensions, RayActor):
    def __init__(
        self,
        args,
        rank: int,
        worker_type: str = "regular",
        base_gpu_id: int | None = None,
        sglang_overrides: dict | None = None,
        num_gpus_per_engine: int | None = None,
        pg_bundles: list[int] | None = None,
    ):
        self.args = args
        self.rank = rank
        self.worker_type = worker_type
        self.base_gpu_id = base_gpu_id
        self.sglang_overrides = sglang_overrides or {}
        self.num_gpus_per_engine = num_gpus_per_engine
        self.pg_bundles = pg_bundles
        self._scheduler_actors = []
        self._sglang_server_actor = None
        self.process = None
        # ORBIT-SEAM: tracks whether this sglang build exposes /post_process_weights
        # (older builds 404); lazily set the first time post_process_weights() runs
        self._supports_post_process_weights: bool | None = None
        # ORBIT-SEAM: same probe for the newer /begin_weight_update//end_weight_update
        # session endpoints (the pinned sglang build predates them; sessionless builds
        # load weights directly, so skipping the session is safe)
        self._supports_weight_update_session: bool | None = None

    def get_topology_info(self) -> dict:
        """Placement facts for the dashboard timeline. ``base_gpu_id`` is
        node-physical, so these ids match the NVML order the GPU sampler uses."""
        from miles.utils.misc import get_current_node_ip

        if self.base_gpu_id is None:  # external engines: placement unknown
            gpu_ids = []
        else:
            gpus_on_node = min(self.num_gpus_per_engine, self.args.num_gpus_per_node)
            gpu_ids = list(range(self.base_gpu_id, self.base_gpu_id + gpus_on_node))
        return dict(
            url=f"http://{self.server_host}:{self.server_port}",
            node_ip=get_current_node_ip(),
            gpu_ids=gpu_ids,
            gpu_uuids=_get_gpu_uuids(gpu_ids),
            worker_type=self.worker_type,
            node_rank=self.node_rank,
        )

    def init(
        self,
        dist_init_addr,
        port,
        nccl_port,
        host=None,
        disaggregation_bootstrap_port=None,
        router_ip=None,
        router_port=None,
        engine_info_bootstrap_port=None,
    ):
        if env_report := self.args.env_report:
            collect_and_print_node_env_report(
                role="rollout",
                rank=self.rank,
                partial_env_report=env_report,
            )

        self.router_ip = router_ip if router_ip is not None else self.args.sglang_router_ip
        self.router_port = router_port if router_port is not None else self.args.sglang_router_port

        host = host or get_host_info()[1]

        def _format_v6_uri(addr):
            if not addr or addr.startswith("["):
                return addr
            try:
                if ipaddress.ip_address(addr).version == 6:
                    return f"[{addr}]"
            except ValueError:
                pass
            return addr

        host = _format_v6_uri(host)
        ip_part, port_part = dist_init_addr.rsplit(":", 1)
        dist_init_addr = f"{_format_v6_uri(ip_part)}:{port_part}"
        server_args_dict, external_engine_need_check_fields = _compute_server_args(
            self.args,
            self.rank,
            dist_init_addr,
            nccl_port,
            host,
            port,
            self.worker_type,
            disaggregation_bootstrap_port,
            base_gpu_id=self.base_gpu_id,
            engine_info_bootstrap_port=engine_info_bootstrap_port,
            sglang_overrides=self.sglang_overrides,
            num_gpus_per_engine=self.num_gpus_per_engine,
        )

        # ORBIT-SEAM: exposes engine node count for update_adapter_from_rank_tensors's
        # single-host guard (home mixin, orbit/sglang/engine_ext.py)
        self.nnodes = server_args_dict["nnodes"]
        self.node_rank = server_args_dict["node_rank"]
        self.server_host = server_args_dict["host"]  # with [] if ipv6
        self.server_port = server_args_dict["port"]

        if self.args.rollout_external:
            self._init_external(server_args_dict, external_engine_need_check_fields=external_engine_need_check_fields)
        else:
            self._init_normal(server_args_dict)

    def _init_external(self, expect_server_args, external_engine_need_check_fields):
        logger.info(f"Use external SGLang engine (rank={self.rank}, expect_server_args={expect_server_args})")

        def _get_actual_server_args():
            response = requests.get(f"http://{self.server_host}:{self.server_port}/get_server_info")
            response.raise_for_status()
            return response.json()

        def _sanity_check_server_args(actual_server_args, expect_server_args):
            for name in external_engine_need_check_fields:
                expect_value = expect_server_args.get(name)
                actual_value = actual_server_args.get(name)
                assert (
                    actual_value == expect_value
                ), f"{name=} {expect_value=} {actual_value=} {expect_server_args=} {actual_server_args=}"

        _wait_server_healthy(
            base_url=f"http://{self.server_host}:{self.server_port}",
            api_key=None,
            is_process_alive=lambda: True,
        )
        actual_server_args = _get_actual_server_args()
        _sanity_check_server_args(actual_server_args, expect_server_args)

    def _init_normal(self, server_args_dict):
        use_rdt = self.args.update_weight_transfer_mode == "rdt"
        if use_rdt:
            if self.node_rank != 0:
                # For a multi-node engine, the node-0 server's RayEngine spawns
                # the SchedulerActors of ALL ranks (placed cross-node via the
                # placement group), so non-zero node ranks launch nothing.
                return
            server_args_dict["use_ray"] = True
            server_args_dict["enable_rdt_weight_sync"] = True
            assert self.pg_bundles
        logger.info(
            f"Launch HttpServerEngineAdapter at: {self.server_host}:{self.server_port}"
            f"{' (use_ray=True for RDT)' if use_rdt else ''}"
        )
        # ORBIT-SEAM: ServerArgs is read-only after __post_init__ resolves it (v0.5.18); the
        # bracket-stripped host must travel in as a constructor argument, not be
        # assigned after (self.server_host above keeps the bracketed form -- it's
        # used for URL construction elsewhere in this class). launch_server_process
        # also gained the force_native_ops passthrough (orbit/sglang/native_ops.py).
        server_args = ServerArgs(**{**server_args_dict, "host": server_args_dict["host"].strip("[]")})
        if use_rdt:
            self._sglang_server_actor, self._scheduler_actors = _launch_sglang_server(
                server_args, bundle_indices=self.pg_bundles
            )
        else:
            self.process = launch_server_process(
                server_args,
                force_native_ops=getattr(self.args, "sglang_force_native_ops", False),
            )

        # ORBIT-SEAM: use_miles_router renamed use_orbit_router (orbit's naming split)
        if self.node_rank == 0 and self.router_ip and self.router_port:
            if parse(sglang_router.__version__) <= parse("0.2.1") or self.args.use_orbit_router:
                assert (
                    self.worker_type == "regular"
                ), "pd disaggregation is not supported in old router or miles router."
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/add_worker?url=http://{self.server_host}:{self.server_port}"
                )
            else:
                payload = {
                    "url": f"http://{self.server_host}:{self.server_port}",
                    "worker_type": self.worker_type,
                }
                if self.worker_type == "prefill":
                    payload["bootstrap_port"] = server_args_dict["disaggregation_bootstrap_port"]
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/workers",
                    json=payload,
                )
            response.raise_for_status()

    def _make_request(self, endpoint: str, payload: dict | None = None):
        """Make a POST request to the specified endpoint with the given payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)

        Returns:
            The JSON response from the server
        """
        if self.node_rank != 0:
            return

        url = f"http://{self.server_host}:{self.server_port}/{endpoint}"
        response = requests.post(url, json=payload or {})
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if hasattr(e, "add_note"):
                e.add_note(f"{response.text=}")
            raise
        return response.json()

    def health_generate(self, timeout: float = 5.0) -> bool:
        """Run /health_generate on the underlying SGLang HTTP server.

        Args:
            timeout: Timeout for the health request in seconds.

        Returns:
            True if the server responds with HTTP 200.

        Raises:
            requests.RequestException: If the request fails for any reason, including timeout.
        """
        if self.node_rank != 0:
            return True

        response = requests.get(
            f"http://{self.server_host}:{self.server_port}/health_generate",
            timeout=timeout,
        )
        response.raise_for_status()
        return True

    # ORBIT-SEAM: adapter_config/adapter_name passthrough for PEFT tensor updates;
    # populated by the home mixin's update_adapter_from_ray_tensor/update_adapter_from_rank_tensors
    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = False,
        weight_version: str | None = None,
        selector: str = "all",
        adapter_config: dict | None = None,
        adapter_name: str | None = None,
    ):
        """
        Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.

        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """
        payload = {
            "serialized_named_tensors": serialized_named_tensors,
            "load_format": load_format,
            "flush_cache": flush_cache,
            "selector": selector,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        if adapter_config is not None:
            payload["adapter_config"] = adapter_config
        if adapter_name is not None:
            payload["adapter_name"] = adapter_name
        return self._make_request(
            "update_weights_from_tensor",
            payload,
        )

    # ORBIT-SEAM: comment wording only (TODO -> Follow-up), no behavior change
    def get_remote_instance_transfer_engine_info(self, rank: int):
        # Follow-up: will be changed to `remote_instance_transfer_engine_info` when the sglang side is ready.
        response = requests.get(
            f"http://{self.server_host}:{self.server_port}/get_remote_instance_transfer_engine_info",
            params={"rank": rank},
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json()["remote_instance_transfer_engine_info"]

    def get_parallelism_info(self, rank: int):
        response = requests.get(
            f"http://{self.server_host}:{self.server_port}/parallelism_config",
            params={"rank": rank},
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json()

    def get_server_info(self):
        response = requests.get(
            f"http://{self.server_host}:{self.server_port}/server_info",
            timeout=5.0,
        )
        response.raise_for_status()
        return response.json()

    def load_lora_adapter_from_tensors(
        self,
        lora_name: str,
        config_dict: dict,
        serialized_tensors: str | None = None,
        serialized_named_tensors: list | None = None,
        load_format: str | None = None,
        pinned: bool = False,
        added_tokens_config: dict | None = None,
        upsert: bool = False,
        expected_checksums: dict | None = None,
    ):
        """Load a LoRA adapter from either transport (exactly one of the two).

        ``serialized_named_tensors[tp_rank]`` is bytes for that TP rank; ``serialized_tensors``
        is the whole adapter. With ``upsert``, the already-loaded ``lora_name`` is overwritten
        in place (no unload/register).
        """
        if (serialized_tensors is None) == (serialized_named_tensors is None):
            raise ValueError("pass exactly one of serialized_tensors / serialized_named_tensors")
        payload = {
            "lora_name": lora_name,
            "config_dict": config_dict,
            "pinned": pinned,
        }
        if serialized_tensors is not None:
            payload["serialized_tensors"] = serialized_tensors
        else:
            payload["serialized_named_tensors"] = serialized_named_tensors
        if upsert:
            payload["upsert"] = True
        if load_format is not None:
            payload["load_format"] = load_format
        if added_tokens_config is not None:
            payload["added_tokens_config"] = added_tokens_config
        if expected_checksums is not None:
            payload["expected_checksums"] = expected_checksums

        return self._make_request(
            "load_lora_adapter_from_tensors",
            payload,
        )

    def load_lora_adapter_from_distributed(
        self,
        lora_name: str,
        config_dict: dict,
        names: list,
        dtypes: list,
        shapes: list,
        group_name: str,
        pinned: bool = False,
        added_tokens_config: dict | None = None,
        upsert: bool = False,
    ):
        """Load a LoRA adapter: only metadata is sent; weights arrive via NCCL broadcast over ``group_name``.
        With ``upsert``, the already-loaded ``lora_name`` is overwritten in place (no unload/register)."""
        payload = {
            "lora_name": lora_name,
            "config_dict": config_dict,
            "names": names,
            "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
            "shapes": shapes,
            "group_name": group_name,
            "pinned": pinned,
            "upsert": upsert,
        }
        if added_tokens_config is not None:
            payload["added_tokens_config"] = added_tokens_config

        return self._make_request(
            "load_lora_adapter_from_distributed",
            payload,
        )

    # ORBIT-SEAM: raise_if_local_process_exited + timeout= arg added so flush_cache fails
    # fast (with the underlying process-exit cause) instead of retrying against a dead server
    def flush_cache(self):
        """Flush the cache of the server."""
        if self.node_rank != 0:
            return

        def raise_if_local_process_exited(cause: Exception | None = None):
            process = getattr(self, "process", None)
            if process is not None and not process.is_alive():
                error = RuntimeError(
                    f"SGLang server process exited before flush_cache completed "
                    f"({self.server_host}:{self.server_port})."
                )
                if cause is not None:
                    raise error from cause
                raise error

        raise_if_local_process_exited()
        # flush cache will not return status_code 200 when there are pending requests
        last_message = None
        for _ in range(60):
            try:
                response = requests.get(f"http://{self.server_host}:{self.server_port}/flush_cache", timeout=5.0)
                if response.status_code == 200:
                    break
                last_message = response.text
            except NewConnectionError as e:
                raise e
            except Exception as e:
                raise_if_local_process_exited(e)
                logger.info(f"Error flushing cache: {e}")
                last_message = str(e)
            time.sleep(1)
        else:
            raise TimeoutError(f"Timeout while flushing cache: {last_message}")

    def shutdown(self):
        if self.args.rollout_external:
            return
        if self._sglang_server_actor is None and self.process is None:
            # Non-zero node ranks of an RDT multi-node engine launch no server.
            return

        logger.info(f"Shutdown engine {self.server_host}:{self.server_port}...")
        if self.node_rank == 0:
            worker_url = f"http://{self.server_host}:{self.server_port}"
            response = None
            # ORBIT-SEAM: use_miles_router renamed use_orbit_router (orbit's naming split)
            if parse(sglang_router.__version__) <= parse("0.2.1") or self.args.use_orbit_router:
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/remove_worker?url=http://{self.server_host}:{self.server_port}"
                )
            elif parse(sglang_router.__version__) < parse("0.3.0"):
                worker_url = quote(worker_url, safe="")
                response = requests.delete(f"http://{self.router_ip}:{self.router_port}/workers/{worker_url}")
            else:
                try:
                    all_workers = requests.get(f"http://{self.router_ip}:{self.router_port}/workers").json()["workers"]
                    for worker in all_workers:
                        if worker["url"] == worker_url:
                            worker_id = worker["id"]
                            response = requests.delete(
                                f"http://{self.router_ip}:{self.router_port}/workers/{worker_id}"
                            )
                            break
                    else:
                        logger.warning(f"Worker {worker_url} not found in router during shutdown.")
                except Exception as e:
                    logger.warning(f"Failed to fetch workers list or remove worker: {e}")

            if response is not None:
                response.raise_for_status()
        if self._sglang_server_actor is not None:
            ray.kill(self._sglang_server_actor)
            self._sglang_server_actor = None
            self._scheduler_actors = []
            return
        kill_process_tree(self.process.pid)

    def get_weight_version(self):
        if self.node_rank != 0:
            return
        base = f"http://{self.server_host}:{self.server_port}"
        # new sglang change api from /get_weight_version to /model_info
        for endpoint in ("/model_info", "/get_weight_version"):
            response = requests.get(f"{base}{endpoint}")
            if response.status_code == 200:
                return response.json()["weight_version"]
        response.raise_for_status()

    # ORBIT-SEAM: removed base docstring: trivial one-liner mirrored below by the
    # undocumented unload_oft_adapter counterpart; kept the pair symmetric
    def unload_lora_adapter(self, lora_name: str):
        return self._make_request(
            "unload_lora_adapter",
            {"lora_name": lora_name},
        )

    def get_scheduler_actors(self) -> list:
        """Return this engine's SchedulerActor handles (RDT mode, use_ray=True)."""
        return self._scheduler_actors

    # ORBIT-SEAM: new method -- OFT counterpart to unload_lora_adapter above; not one of
    # the seven methods named for the P2 mixin move in the slice-3c spec, left here
    def unload_oft_adapter(self, adapter_name: str):
        return self._make_request(
            "unload_oft_adapter",
            {"adapter_name": adapter_name},
        )

    def release_memory_occupation(self, tags: list[str] = None):
        """Release memory occupation. Available tags: weights, kv_cache."""
        self.flush_cache()
        return self._make_request(
            "release_memory_occupation",
            {"tags": tags},
        )

    def resume_memory_occupation(self, tags: list[str] = None):
        """
        Available tags for multi-stage resume: weights, kv_cache
        """
        return self._make_request(
            "resume_memory_occupation",
            {"tags": tags},
        )

    def check_weights(
        self, action: str, allow_quant_error: bool = False, selector: str = "all", skip_list: list[str] | None = None
    ):
        payload = {"action": action, "allow_quant_error": allow_quant_error, "selector": selector}
        if skip_list is not None:
            # sglang's CheckWeightsReqInput names this field `skip_tensor_list`.
            payload["skip_tensor_list"] = skip_list
        return self._make_request("weights_checker", payload)

    def pull_weights(self, target_version: int):
        """Have the engine sync every host it spans to target_version: each host pulls the
        published weights (a full checkpoint copied as-is, or deltas verified per-tensor and
        applied onto the local checkpoint) into its local checkpoint dir. The engine reloads
        it afterwards via update_weights_from_disk."""
        return self._make_request(
            "pull_weights",
            {
                "local_checkpoint_dir": self.args.update_weight_local_checkpoint_dir,
                "source_dir": self.args.update_weight_disk_dir,
                "target_version": target_version,
            },
        )

    def update_weights_from_disk(
        self, model_path: str, load_format: str | None = None, weight_version: str | None = None
    ):
        """Reload weights from *model_path* without restarting the engine.

        Used for non-updatable (frozen) models that overlap with megatron (after offload,
        weights are restored from disk instead of CPU cache), and by disk-delta weight sync
        to reload the patched host-local checkpoint.
        """
        payload = {"model_path": model_path}
        if load_format is not None:
            payload["load_format"] = load_format
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request("update_weights_from_disk", payload)

    def init_weights_update_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        return self._make_request(
            "init_weights_update_group",
            {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
                "group_name": group_name,
                "backend": backend,
            },
        )

    def destroy_weights_update_group(self, group_name):
        try:
            return self._make_request(
                "destroy_weights_update_group",
                {
                    "group_name": group_name,
                },
            )
        except requests.exceptions.RequestException:
            # catch the case there the engine is just created and does not have the group.
            pass

    def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        flush_cache=False,
        weight_version: str | None = None,
        selector: str = "all",
    ):
        payload = {
            "names": names,
            "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
            "shapes": shapes,
            "group_name": group_name,
            "flush_cache": flush_cache,
            "selector": selector,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request(
            "update_weights_from_distributed",
            payload,
        )

    def pause_generation(self, mode: str = "retract"):
        response = requests.post(
            f"http://{self.server_host}:{self.server_port}/pause_generation",
            json={"mode": mode},
        )
        response.raise_for_status()
        return response

    def continue_generation(self):
        response = requests.post(f"http://{self.server_host}:{self.server_port}/continue_generation", json={})
        response.raise_for_status()
        return response

    def post_process_weights(
        self,
        restore_weights_before_load: bool = False,
        post_process_quantization: bool = False,
        post_load_weights: bool = False,
    ):
        """
        Update model weights from tensor data. The HTTP server will only post meta data, and the real weights will be copied directly from GPUs.
        Note: The model should be on GPUs rather than CPU for this functionality to work properly.
        If you encounter issues, ensure your model is loaded on GPU devices rather than CPU.
        """
        # ORBIT-SEAM: 404 tolerance for sglang builds that predate this endpoint (see
        # self._supports_post_process_weights above); a BF16/non-quantized/colocate
        # setup makes the step a no-op, so silently skipping it is safe
        if self._supports_post_process_weights is False:
            return None

        try:
            result = self._make_request(
                "post_process_weights",
                {
                    "restore_weights_before_load": restore_weights_before_load,
                    "post_process_quantization": post_process_quantization,
                    "post_load_weights": post_load_weights,
                },
            )
            self._supports_post_process_weights = True
            return result
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                # Older sglang versions (e.g. 0.5.9) do not expose this
                # endpoint. For a BF16, non-quantized, colocate setup the
                # post-process step is a no-op, so silently skip it.
                self._supports_post_process_weights = False
                return None
            raise

    def begin_weight_update(self, selector: str = "all"):
        """Open a weight-update session on the engine (restores packed weights for loading)."""
        # ORBIT-SEAM: 404 tolerance for sglang builds without the weight-update session
        # (mirrors the post_process_weights probe above)
        if self._supports_weight_update_session is False:
            return None
        try:
            result = self._make_request("begin_weight_update", {"selector": selector})
            self._supports_weight_update_session = True
            return result
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                self._supports_weight_update_session = False
                return None
            raise

    def end_weight_update(self):
        """Close the weight-update session (post-load + quant post-process on the full model)."""
        # ORBIT-SEAM: see begin_weight_update — sessionless builds skip the close too
        if self._supports_weight_update_session is False:
            return None
        try:
            result = self._make_request("end_weight_update", {})
            self._supports_weight_update_session = True
            return result
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                self._supports_weight_update_session = False
                return None
            raise

    def update_weight_version(self, weight_version: str):
        return self._make_request(
            "update_weight_version",
            {"new_version": weight_version, "abort_all_requests": False},
        )

    def start_profile(
        self,
        # The output directory
        output_dir: str | None = None,
        # If set, it profile as many as this number of steps.
        # If it is set, profiling is automatically stopped after this step, and
        # the caller doesn't need to run stop_profile.
        start_step: int | None = None,
        num_steps: int | None = None,
        activities: list[str] | None = None,
        profile_by_stage: bool = False,
        with_stack: bool | None = None,
        record_shapes: bool | None = None,
    ):
        response = requests.post(
            f"http://{self.server_host}:{self.server_port}/start_profile",
            json={
                "output_dir": output_dir,
                "start_step": start_step,
                "num_steps": num_steps,
                "activities": activities,
                "profile_by_stage": profile_by_stage,
                "with_stack": with_stack,
                "record_shapes": record_shapes,
            },
        )
        response.raise_for_status()
        return response

    def stop_profile(self):
        response = requests.post(f"http://{self.server_host}:{self.server_port}/stop_profile", json={})
        response.raise_for_status()
        return response

    def simulate_crash(self):
        if self.args.rollout_external or not getattr(self, "process", None):
            logger.info(
                "simulate_crash called but no local engine process exists (rollout_external=%s); skip kill",
                self.args.rollout_external,
            )
            return

        logger.info(f"Simulating crash on engine {self.server_host}:{self.server_port}...")
        self.shutdown()


# ORBIT-SEAM: _compute_server_args helpers (MoE-parity, target-module classification,
# adapter dtype selection) moved to orbit/sglang/server_args.py (P1 lift-out, Phase 3
# slice 3c); imported above and called directly below
def _compute_server_args(
    args,
    rank,
    dist_init_addr,
    nccl_port,
    host,
    port,
    worker_type: str = "regular",
    disaggregation_bootstrap_port: int | None = None,
    base_gpu_id: int | None = None,
    engine_info_bootstrap_port: int | None = None,
    sglang_overrides: dict | None = None,
    num_gpus_per_engine: int | None = None,
):
    _gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    nnodes = max(1, _gpus_per_engine // args.num_gpus_per_node)
    node_rank = rank % nnodes
    base = base_gpu_id if base_gpu_id is not None else get_base_gpu_id(args, rank)
    base = _to_local_gpu_id(base)
    kwargs = {
        "model_path": args.hf_checkpoint,
        "trust_remote_code": True,
        "random_seed": args.seed + rank,
        # memory
        "enable_memory_saver": args.offload_rollout,
        # distributed
        "host": host,
        "port": port,
        "nccl_port": nccl_port,
        "nnodes": nnodes,
        "node_rank": node_rank,
        "dist_init_addr": dist_init_addr,
        "gpu_id_step": 1,
        "base_gpu_id": base,
        # parallel
        "tp_size": _gpus_per_engine,
        "dp_size": args.sglang_dp_size,
        # ORBIT-SEAM: attn_cp_size / moe_dp_size parallelism knobs added
        "attn_cp_size": args.sglang_attn_cp_size,
        "moe_dp_size": args.sglang_moe_dp_size,
        "pp_size": args.sglang_pp_size,
        "ep_size": args.sglang_ep_size,
        # always skip warmup to prevent warmup timeout.
        "skip_server_warmup": True,
        # always enable draft weights cpu backup so that we run training without mtp weights.
        "enable_draft_weights_cpu_backup": True,
        # always serve /metrics so Prometheus scrapers can read engine stats.
        "enable_metrics": True,
    }

    if os.environ.get("MILES_SGLANG_DUMMY_LOAD") == "1":
        kwargs["load_format"] = "dummy"

    if worker_type == "prefill":
        kwargs["disaggregation_mode"] = "prefill"
        kwargs.setdefault("load_balance_method", "round_robin")
        assert (
            disaggregation_bootstrap_port is not None
        ), "disaggregation_bootstrap_port must be set for prefill worker"
        kwargs["disaggregation_bootstrap_port"] = disaggregation_bootstrap_port
    elif worker_type == "decode":
        kwargs["disaggregation_mode"] = "decode"
        kwargs["prefill_round_robin_balance"] = True

    if args.use_rollout_routing_replay:
        kwargs["enable_return_routed_experts"] = True
    if args.use_rollout_indexer_replay:
        kwargs["enable_return_indexer_topk"] = True
    if args.fp16:
        kwargs["dtype"] = "float16"
    if engine_info_bootstrap_port is not None:
        kwargs["engine_info_bootstrap_port"] = engine_info_bootstrap_port

    # ORBIT-SEAM: OPD same-base teacher engine-slot reservation (reserves a peft_paths
    # entry / max_ofts_per_batch slot for a frozen teacher adapter colocated with the
    # student); out of scope for this slice's P1 extraction (not one of the named
    # _compute_server_args helpers) -- lazy import keeps this file miles-clean at
    # module scope
    from orbit.opd.opd_teacher_spec import (
        OPD_TEACHER_ADAPTER_NAME,
        needs_engine_teacher_slot,
        parse_teacher_spec,
    )

    def _opd_teacher_spec_from_args(a):
        if getattr(a, "opd_type", None) != "sglang":
            return None
        return parse_teacher_spec(getattr(a, "opd_teacher", None), getattr(a, "opd_teacher_load", None))

    opd_teacher_spec = _opd_teacher_spec_from_args(args)
    external_opd_teacher = bool(
        getattr(args, "opd_teacher_url", None)
        or getattr(args, "opd_teacher_urls", None)
        or getattr(args, "opd_serve_teacher", False)
        or getattr(args, "opd_teacher_pool", None)
    )
    opd_teacher_slot = not external_opd_teacher and needs_engine_teacher_slot(opd_teacher_spec)

    # ORBIT-SEAM: base's single is_lora_enabled(args) check replaced by the unified
    # get_peft_method(args) (lora | oft | none) dispatch used throughout this function;
    # enable_adapter_cpu_backup (OFT-only) added alongside the pre-existing
    # enable_weights_cpu_backup guard
    peft_method = get_peft_method(args)
    if "enable_weights_cpu_backup" not in kwargs:
        kwargs["enable_weights_cpu_backup"] = args.offload_rollout
    if peft_method == "oft":
        if "enable_adapter_cpu_backup" not in kwargs:
            requested = getattr(args, "offload_rollout_adapter", None)
            kwargs["enable_adapter_cpu_backup"] = bool(requested) if requested is not None else False
        if kwargs["enable_adapter_cpu_backup"] and not kwargs["enable_weights_cpu_backup"]:
            raise ValueError(
                "--offload-rollout-adapter requires SGLang weights CPU backup; "
                "enable rollout weights CPU backup or pass "
                "--no-offload-rollout-adapter."
            )

    # ORBIT-SEAM: base's plain `if is_lora_enabled(args): kwargs["enable_lora"] = True; ...`
    # (upstream's multi-tenant LoRAManager) replaced end-to-end by this peft_method
    # dispatch over the fork's single-active peft/lora and peft/oft paths (see the
    # per-branch comments below for the rationale of each field)
    if peft_method == "lora":
        # Route LoRA through the fork's SINGLE-ACTIVE peft/lora (peft_method="lora"),
        # symmetric to the OFT branch below -- NOT upstream's multi-tenant
        # LoRAManager (enable_lora). The IPC weight-sync then goes through
        # update_weights_from_tensor(load_format="lora_adapter") in the
        # peft_transport IPC backend, matching the C1-validated single-active
        # streamed path. peft/lora is single-active: no lora_backend /
        # max_loras_per_batch pool (MoE-LoRA rides upstream's triton
        # fused_moe_lora by construction, so the old triton-backend check is moot).
        # Offload (BP-7): this branch only changes WHICH LoRA manager handles the
        # adapter; recover_updatable_engines / steady-state offload still mirror
        # args.offload_rollout as before.
        kwargs["peft_method"] = "lora"
        kwargs["peft_target_modules"] = convert_target_modules_to_hf(args.target_modules)
        kwargs["peft_max_lora_rank"] = max(getattr(args, "lora_rank", 0), 1)
        # Double-buffer weight-sync: the fork sizes LoRA's mem-pool to 2 slots
        # (active+staging) from this flag. Unlike OFT (which also bumps
        # max_ofts_per_batch above), LoRA has no slot-count knob, so this is the
        # sole init-time signal enabling stage/activate.
        kwargs["peft_double_buffer"] = bool(getattr(args, "adapter_double_buffer", False))

        lora_adapter_path = getattr(args, "lora_adapter_path", None)
        if lora_adapter_path is not None:
            kwargs["peft_paths"] = {LORA_ADAPTER_NAME: lora_adapter_path}
        else:
            logger.info("No pre-trained LoRA adapter_path provided, will use random initial weights")
    elif peft_method == "oft":
        # Same BP-7 decline as above; mirror args.offload_rollout until the
        # merge gate is verified for the OFT recovery / offload path too.
        kwargs["peft_method"] = "oft"
        kwargs["max_oft_block_size"] = args.oft_block_size
        kwargs["peft_target_modules"] = convert_target_modules_to_hf(args.target_modules)
        kwargs["oft_dtype"] = _training_adapter_dtype_arg(args)
        kwargs["oft_type"] = args.oft_type
        # max_ofts_per_batch includes the base-only request -- sglang's
        # init_memory_pool() eagerly loads the base into slot 0, so the
        # minimum usable value is 2 (base + 1 trained adapter). With the
        # previous value 1, the very first /update_weights_from_tensor
        # for an OFT adapter raised "No available buffer slots for direct
        # OFT loading. All slots are occupied." inside sglang's
        # oft/mem_pool.py::allocate_buffer_slot.
        kwargs["max_ofts_per_batch"] = 2
        if getattr(args, "adapter_double_buffer", False):
            kwargs["max_ofts_per_batch"] = max(kwargs["max_ofts_per_batch"], 3)
        # reserved orbit_teacher slot for OPD same-base teacher scoring
        if opd_teacher_slot:
            kwargs["max_ofts_per_batch"] += 1
        # Enable the fork's stage/activate double-buffer path (staging slot =
        # max_ofts_per_batch-1); paired with the max_ofts_per_batch=3 bump above.
        kwargs["peft_double_buffer"] = bool(getattr(args, "adapter_double_buffer", False))
        kwargs["oft_backend"] = getattr(args, "sglang_oft_backend", "triton")
        oft_adapter_path = getattr(args, "oft_adapter_path", None)
        if oft_adapter_path is not None:
            kwargs["peft_paths"] = {OFT_ADAPTER_NAME: oft_adapter_path}
        else:
            logger.info("No pre-trained OFT adapter_path provided, will use random initial weights")
        # Unified PEFT consumes one shared peft_paths map. Add a frozen teacher
        # after the student entry so both adapters reach the same OFT manager.
        if opd_teacher_slot and opd_teacher_spec.source == "adapter":
            kwargs.setdefault("peft_paths", {})[OPD_TEACHER_ADAPTER_NAME] = opd_teacher_spec.path

    # ORBIT-SEAM: upstream's LoRA+colocate host-RAM base-weight mirror, re-anchored out of the
    # `elif lora_rollout_enabled(args):` branch that orbit's peft_method dispatch above replaces
    if peft_method == "lora" and lora_base_cpu_backup_enabled(args):
        # Host-RAM mirror of the base weights so they survive
        # torch_memory_saver.pause() across rollout/training swaps without
        # needing to be re-shipped from the trainer. The trainer mirrors
        # this by skipping the base weight sync entirely (see
        # UpdateWeightFromTensor.update_weights).
        kwargs["enable_weights_cpu_backup"] = True
        logger.info(
            "LoRA + colocate: enabling SGLang enable_weights_cpu_backup=True; "
            "the trainer will skip per-step base weight sync."
        )

    # Last, so a per-group override wins over every args-derived default above.
    if sglang_overrides:
        kwargs.update(sglang_overrides)

    # ORBIT-SEAM: server_arg_fields precomputed upfront (was: unused_keys = set(kwargs.keys())
    # seeded before the loop, with unused_keys.discard(attr.name) removed per-attr inside the
    # loop below); the PEFT branches above and the peft/MoE-parity calls below also add keys
    # to kwargs, so a single set-difference against server_arg_fields after everything has
    # settled is the only computation that accounts for every source correctly
    server_arg_fields = {field.name for field in dataclasses.fields(ServerArgs)}
    for attr in dataclasses.fields(ServerArgs):
        if worker_type == "decode" and attr.name == "enable_hierarchical_cache":
            continue
        if hasattr(args, f"sglang_{attr.name}") and attr.name not in kwargs:
            kwargs[attr.name] = getattr(args, f"sglang_{attr.name}")

    _configure_peft_cache_kwargs(kwargs, peft_method)
    _configure_megatron_moe_parity_kwargs(kwargs, args, sglang_overrides)

    unused_keys = set(kwargs.keys()) - server_arg_fields

    # for compatibility with old args
    if len(unused_keys) > 0:
        logger.info(f"Warning: The following arguments is not supported in the current sglang: {unused_keys}.")
        for key in unused_keys:
            kwargs.pop(key)

    # ORBIT-SEAM: external_engine_need_check_fields relocated here (was computed right
    # after engine_info_bootstrap_port, before any PEFT/MoE-parity kwargs existed).
    # Compute the external-engine sanity-check field set after every kwargs
    # mutation has settled (PEFT branches add peft_method / enable_lora /
    # lora_backend / etc.; the dataclasses-fields auto-pass-through pulls in
    # any sglang_<attr> override; the unused_keys pop trims dead keys). If
    # we computed this earlier, _init_external would silently miss those
    # fields when validating an external SGLang server's startup args.
    external_engine_need_check_fields = [k for k in kwargs.keys() if k not in _EXTERNAL_ENGINE_SKIP_CHECK_FIELDS]

    return kwargs, external_engine_need_check_fields


# ORBIT-SEAM: list -> frozenset (immutable module-level constant; no behavior change,
# `in` membership tests above work identically on either container)
_EXTERNAL_ENGINE_SKIP_CHECK_FIELDS = frozenset({
    "model_path",
    "trust_remote_code",
    "random_seed",
    "nccl_port",
    "dist_init_addr",
    "skip_server_warmup",
    "enable_draft_weights_cpu_backup",
    "enable_metrics",
    "mem_fraction_static",
})
