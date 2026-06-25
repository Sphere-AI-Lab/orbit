import dataclasses
import ipaddress
import logging
import multiprocessing
import os
import time
from pathlib import Path
from urllib.parse import quote

import requests
import sglang_router
from packaging.version import parse
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import kill_process_tree
from urllib3.exceptions import NewConnectionError

from orbit.backends.megatron_utils.lora_utils import LORA_ADAPTER_NAME
from orbit.backends.megatron_utils.oft_utils import OFT_ADAPTER_NAME
from orbit.backends.megatron_utils.peft_utils import convert_target_modules_to_hf, get_peft_method
from orbit.backends.sglang_utils.native_ops import patch_sglang_native_ops
from orbit.ray.ray_actor import RayActor
from orbit.utils.env_report import collect_and_print_node_env_report
from orbit.utils.http_utils import get_host_info

logger = logging.getLogger(__name__)
_COMPAT_SITE_DIR = Path(__file__).resolve().parent / "compat_site"


def get_base_gpu_id(args, rank):
    num_gpus = min(args.num_gpus_per_node, args.rollout_num_gpus_per_engine)
    if args.colocate:
        start_index = (rank * num_gpus) % args.num_gpus_per_node
    else:
        num_actor_gpus = 0 if args.debug_rollout_only else args.actor_num_gpus_per_node * args.actor_num_nodes
        start_index = (num_actor_gpus + rank * num_gpus) % args.num_gpus_per_node
        if args.use_critic:
            num_critic_gpus = args.critic_num_gpus_per_node * args.critic_num_nodes
            start_index = (num_actor_gpus + num_critic_gpus + rank * num_gpus) % args.num_gpus_per_node
    return start_index


def _to_local_gpu_id(physical_gpu_id: int) -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
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


def _prepend_pythonpath(path: Path):
    current = os.environ.get("PYTHONPATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    path_str = str(path)
    if path_str not in entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([path_str, *entries])


def _prepare_child_native_ops_env(force_native_ops: bool):
    if not force_native_ops:
        return

    os.environ["ORBIT_SGLANG_FORCE_NATIVE_OPS"] = "1"
    _prepend_pythonpath(_COMPAT_SITE_DIR)


def _server_args_enable_peft(server_args: ServerArgs) -> bool:
    return bool(getattr(server_args, "enable_lora", False) or getattr(server_args, "enable_oft", False))


def _prepare_child_peft_cache_env(server_args: ServerArgs):
    if not _server_args_enable_peft(server_args):
        return

    # PEFT rollout requests rely on SGLang's adapter/version extra_key when
    # matching prefix cache entries. In the tested SGLang build, the Python
    # radix cache honors it while the experimental C++ radix tree drops it.
    previous = os.environ.get("SGLANG_EXPERIMENTAL_CPP_RADIX_TREE")
    if previous not in (None, "", "0", "false", "False"):
        logger.warning(
            "Disabling SGLang experimental C++ radix tree for PEFT rollout; "
            "the Python radix cache preserves adapter-specific prefix keys."
        )
    os.environ["SGLANG_EXPERIMENTAL_CPP_RADIX_TREE"] = "0"


def _launch_server_with_orbit_compat(server_args: ServerArgs, force_native_ops: bool):
    _prepare_child_peft_cache_env(server_args)

    if force_native_ops:
        patch_sglang_native_ops()

    from sglang.srt.entrypoints.http_server import launch_server

    launch_server(server_args)


def launch_server_process(server_args: ServerArgs, force_native_ops: bool = False) -> multiprocessing.Process:

    multiprocessing.set_start_method("spawn", force=True)
    server_args.host = server_args.host.strip("[]")
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


class SGLangEngine(RayActor):
    def __init__(
        self,
        args,
        rank: int,
        worker_type: str = "regular",
        base_gpu_id: int | None = None,
        sglang_overrides: dict | None = None,
        num_gpus_per_engine: int | None = None,
    ):
        self.args = args
        self.rank = rank
        self.worker_type = worker_type
        self.base_gpu_id = base_gpu_id
        self.sglang_overrides = sglang_overrides or {}
        self.num_gpus_per_engine = num_gpus_per_engine
        self._supports_post_process_weights: bool | None = None

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
        logger.info(f"Launch HttpServerEngineAdapter at: {self.server_host}:{self.server_port}")
        self.process = launch_server_process(
            ServerArgs(**server_args_dict),
            force_native_ops=getattr(self.args, "sglang_force_native_ops", False),
        )

        if self.node_rank == 0 and self.router_ip and self.router_port:
            if parse(sglang_router.__version__) <= parse("0.2.1") or self.args.use_orbit_router:
                assert (
                    self.worker_type == "regular"
                ), "pd disaggregation is not supported in old router or orbit router."
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

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = False,
        weight_version: str | None = None,
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
        serialized_tensors: str,
        config_dict: dict,
        load_format: str | None = None,
        pinned: bool = False,
        added_tokens_config: dict | None = None,
    ):
        """Load a LoRA adapter from serialized tensor data."""
        payload = {
            "lora_name": lora_name,
            "serialized_tensors": serialized_tensors,
            "config_dict": config_dict,
            "pinned": pinned,
        }
        if load_format is not None:
            payload["load_format"] = load_format
        if added_tokens_config is not None:
            payload["added_tokens_config"] = added_tokens_config

        return self._make_request(
            "load_lora_adapter_from_tensors",
            payload,
        )

    def load_oft_adapter_from_tensors(
        self,
        oft_name: str,
        serialized_tensors: str,
        config_dict: dict,
        pinned: bool = False,
    ):
        """Load an OFT adapter from serialized tensor data.

        Requires the sglang server to be launched with ``enable_oft=True``;
        orbit's SGLangEngine kwargs builder sets this automatically when OFT
        is configured.
        """
        payload = {
            "oft_name": oft_name,
            "serialized_tensors": serialized_tensors,
            "config_dict": config_dict,
            "pinned": pinned,
        }
        return self._make_request("load_oft_adapter_from_tensors", payload)

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
        for _ in range(60):
            try:
                response = requests.get(f"http://{self.server_host}:{self.server_port}/flush_cache", timeout=5.0)
                if response.status_code == 200:
                    break
            except NewConnectionError as e:
                raise e
            except Exception as e:
                raise_if_local_process_exited(e)
                logger.info(f"Error flushing cache: {e}")
                time.sleep(1)
                continue
        else:
            raise TimeoutError("Timeout while flushing cache.")

    def shutdown(self):
        if self.args.rollout_external:
            return

        logger.info(f"Shutdown engine {self.server_host}:{self.server_port}...")
        if self.node_rank == 0:
            worker_url = f"http://{self.server_host}:{self.server_port}"
            response = None
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

    def unload_lora_adapter(self, lora_name: str):
        return self._make_request(
            "unload_lora_adapter",
            {"lora_name": lora_name},
        )

    def unload_oft_adapter(self, oft_name: str):
        return self._make_request(
            "unload_oft_adapter",
            {"oft_name": oft_name},
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

    def check_weights(self, action: str):
        return self._make_request("weights_checker", {"action": action})

    def update_weights_from_disk(self, model_path: str, load_format: str | None = None):
        """Reload weights from *model_path* without restarting the engine.

        Used for non-updatable (frozen) models that overlap with megatron:
        after offload, weights are restored from disk instead of CPU cache.
        """
        payload = {"model_path": model_path}
        if load_format is not None:
            payload["load_format"] = load_format
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
        self, names, dtypes, shapes, group_name, flush_cache=False, weight_version: str | None = None
    ):
        payload = {
            "names": names,
            "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
            "shapes": shapes,
            "group_name": group_name,
            "flush_cache": flush_cache,
        }
        if weight_version is not None:
            payload["weight_version"] = weight_version
        return self._make_request(
            "update_weights_from_distributed",
            payload,
        )

    def update_adapter_from_distributed(
        self,
        *,
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
        group_name: str,
        weight_version: str,
        load_format: str,           # "lora_adapter" | "oft_adapter"
        adapter_config: dict,
        adapter_name: str,
        payload_metadata: dict | None = None,  # OFT FlattenedTensorBucket metadata
        adapter_version: str | None = None,
        double_buffer: bool = False,
    ):
        payload = {
            "names": names,
            "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
            "shapes": shapes,
            "group_name": group_name,
            "weight_version": weight_version,
            # v1 invariant: adapter_version == weight_version; callers that pass
            # adapter_version=None opt into emitting this canonical pair from
            # weight_version alone.
            "adapter_version": adapter_version if adapter_version is not None else weight_version,
            "load_format": load_format,
            "adapter_config": adapter_config,
            "adapter_name": adapter_name,
            "payload_metadata": payload_metadata,
            "double_buffer": double_buffer,
        }
        return self._make_request("update_adapter_from_distributed", payload)

    def activate_adapter_version(
        self,
        *,
        adapter_name: str,
        adapter_version: str,
        weight_version: str,
        load_format: str,
    ):
        return self._make_request(
            "activate_adapter_version",
            {
                "adapter_name": adapter_name,
                "adapter_version": adapter_version,
                "weight_version": weight_version,
                "load_format": load_format,
            },
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

    def update_weight_version(self, weight_version: str):
        return self._make_request(
            "update_weight_version",
            {"new_version": weight_version},
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


def _target_modules_request_moe_lora(target_modules) -> bool:
    if target_modules is None:
        return False
    if isinstance(target_modules, str):
        values = [part.strip().lower() for part in target_modules.split(",")]
    else:
        values = [str(part).strip().lower() for part in target_modules]
    return bool(
        {value for value in values if value}
        & {
            "all",
            "all-linear",
            "all_linear",
            "gate_proj",
            "up_proj",
            "down_proj",
            "linear_fc1",
            "linear_fc2",
            "linear_fc1_gate",
            "linear_fc1_up",
        }
    )


def _training_adapter_dtype_arg(args) -> str:
    if getattr(args, "fp16", False):
        return "float16"
    if getattr(args, "bf16", False):
        return "bfloat16"
    return "float32"


def _args_indicate_moe_model(args) -> bool:
    # num_experts is the authoritative MoE signal. moe_layer_freq is meaningful
    # only when MoE is active, and Megatron's parser defaults it to 1 even for
    # dense models, so it cannot be used as a fallback indicator.
    num_experts = getattr(args, "num_experts", None)
    if num_experts is None:
        return False
    try:
        return int(num_experts) > 0
    except (TypeError, ValueError):
        return False


def _configure_megatron_moe_parity_kwargs(kwargs: dict, args, sglang_overrides: dict | None) -> None:
    if not _args_indicate_moe_model(args):
        return

    explicit_overrides = set(sglang_overrides or {})

    if (
        not getattr(args, "moe_apply_probs_on_input", False)
        and "moe_megatron_weighted_swiglu" not in explicit_overrides
    ):
        if not kwargs.get("moe_megatron_weighted_swiglu", False):
            logger.info(
                "Megatron MoE rollout: enabling SGLang moe_megatron_weighted_swiglu "
                "to match Megatron Core weighted_bias_swiglu_impl."
            )
        kwargs["moe_megatron_weighted_swiglu"] = True

    if (
        str(getattr(args, "moe_router_dtype", "")).lower() == "fp32"
        and "moe_router_force_fp32" not in explicit_overrides
    ):
        if not kwargs.get("moe_router_force_fp32", False):
            logger.info(
                "Megatron MoE rollout: enabling SGLang moe_router_force_fp32 "
                "because Megatron moe_router_dtype=fp32."
            )
        kwargs["moe_router_force_fp32"] = True


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
        "attn_cp_size": args.sglang_attn_cp_size,
        "moe_dp_size": args.sglang_moe_dp_size,
        "pp_size": args.sglang_pp_size,
        "ep_size": args.sglang_ep_size,
        # always skip warmup to prevent warmup timeout.
        "skip_server_warmup": True,
        # always enable draft weights cpu backup so that we run training without mtp weights.
        "enable_draft_weights_cpu_backup": True,
    }

    if sglang_overrides:
        kwargs.update(sglang_overrides)

    if worker_type == "prefill":
        kwargs["disaggregation_mode"] = "prefill"
        kwargs["load_balance_method"] = "round_robin"
        assert (
            disaggregation_bootstrap_port is not None
        ), "disaggregation_bootstrap_port must be set for prefill worker"
        kwargs["disaggregation_bootstrap_port"] = disaggregation_bootstrap_port
    elif worker_type == "decode":
        kwargs["disaggregation_mode"] = "decode"
        kwargs["prefill_round_robin_balance"] = True

    if args.use_rollout_routing_replay:
        kwargs["enable_return_routed_experts"] = True
    if args.fp16:
        kwargs["dtype"] = "float16"
    if engine_info_bootstrap_port is not None:
        kwargs["engine_info_bootstrap_port"] = engine_info_bootstrap_port

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

    if peft_method == "lora":
        lora_backend = kwargs.get("lora_backend", getattr(args, "sglang_lora_backend", "csgmv") or "csgmv")
        if (
            lora_backend != "triton"
            and _args_indicate_moe_model(args)
            and _target_modules_request_moe_lora(args.target_modules)
        ):
            raise ValueError("MoE LoRA requires sglang_lora_backend='triton'.")
        # Decline BP-7 until the merge gate verifies that recover_updatable_engines and
        # steady-state offload do not depend on the dense CPU snapshot before
        # disabling it. Until that is verified (or a per-cycle dense re-sync
        # replacement is added in orbit/ray/rollout.py), keep orbit's existing
        # default of mirroring args.offload_rollout. Engine kwargs may still
        # override (used by the external-engine code path).
        kwargs["enable_lora"] = True
        kwargs["lora_backend"] = lora_backend
        kwargs["max_loras_per_batch"] = 1
        if getattr(args, "adapter_double_buffer", False):
            kwargs["max_loras_per_batch"] = max(kwargs["max_loras_per_batch"], 2)
        kwargs["max_lora_rank"] = max(getattr(args, "lora_rank", 0), 1)
        kwargs["lora_target_modules"] = convert_target_modules_to_hf(args.target_modules)

        lora_adapter_path = getattr(args, "lora_adapter_path", None)
        if lora_adapter_path is not None:
            kwargs["lora_paths"] = {LORA_ADAPTER_NAME: lora_adapter_path}
        else:
            logger.info("No pre-trained LoRA adapter_path provided, will use random initial weights")
    elif peft_method == "oft":
        # Same BP-7 decline as above; mirror args.offload_rollout until the
        # merge gate is verified for the OFT recovery / offload path too.
        kwargs["enable_oft"] = True
        kwargs["max_oft_block_size"] = args.oft_block_size
        kwargs["oft_target_modules"] = convert_target_modules_to_hf(args.target_modules)
        kwargs["oft_dtype"] = _training_adapter_dtype_arg(args)
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
        kwargs["oft_backend"] = getattr(args, "sglang_oft_backend", "triton")
        oft_adapter_path = getattr(args, "oft_adapter_path", None)
        if oft_adapter_path is not None:
            kwargs["oft_paths"] = {OFT_ADAPTER_NAME: oft_adapter_path}
        else:
            logger.info("No pre-trained OFT adapter_path provided, will use random initial weights")

    server_arg_fields = {field.name for field in dataclasses.fields(ServerArgs)}
    for attr in dataclasses.fields(ServerArgs):
        if worker_type == "decode" and attr.name == "enable_hierarchical_cache":
            continue
        if hasattr(args, f"sglang_{attr.name}") and attr.name not in kwargs:
            kwargs[attr.name] = getattr(args, f"sglang_{attr.name}")

    _configure_megatron_moe_parity_kwargs(kwargs, args, sglang_overrides)

    unused_keys = set(kwargs.keys()) - server_arg_fields

    # for compatibility with old args
    if len(unused_keys) > 0:
        logger.info(f"Warning: The following arguments is not supported in the current sglang: {unused_keys}.")
        for key in unused_keys:
            kwargs.pop(key)

    # Compute the external-engine sanity-check field set after every kwargs
    # mutation has settled (PEFT branches add enable_lora / enable_oft /
    # lora_backend / etc.; the dataclasses-fields auto-pass-through pulls in
    # any sglang_<attr> override; the unused_keys pop trims dead keys). If
    # we computed this earlier, _init_external would silently miss those
    # fields when validating an external SGLang server's startup args.
    external_engine_need_check_fields = [k for k in kwargs.keys() if k not in _EXTERNAL_ENGINE_SKIP_CHECK_FIELDS]

    return kwargs, external_engine_need_check_fields


_EXTERNAL_ENGINE_SKIP_CHECK_FIELDS = frozenset({
    "model_path",
    "trust_remote_code",
    "random_seed",
    "nccl_port",
    "dist_init_addr",
    "skip_server_warmup",
    "enable_draft_weights_cpu_backup",
    "mem_fraction_static",
})
