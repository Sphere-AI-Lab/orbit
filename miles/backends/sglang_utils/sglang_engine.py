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
from sglang.srt.utils import MultiprocessingSerializer, kill_process_tree
from urllib3.exceptions import NewConnectionError

from miles.backends.megatron_utils.lora_utils import (
    convert_target_modules_to_hf,
    lora_base_cpu_backup_enabled,
    sglang_lora_target_all_sentinel,
)
from miles.backends.megatron_utils.oft_utils import OFT_ADAPTER_NAME
from miles.backends.megatron_utils.peft_utils import get_peft_method
from miles.backends.sglang_utils.native_ops import patch_sglang_native_ops
from miles.ray.ray_actor import RayActor
from miles.utils.env_report import collect_and_print_node_env_report
from miles.utils.http_utils import get_host_info
from miles.utils.lora import LORA_ADAPTER_NAME, lora_rollout_enabled
from miles.utils.multi_lora import is_multi_lora_enabled

logger = logging.getLogger(__name__)
_COMPAT_SITE_DIR = Path(__file__).resolve().parent / "compat_site"


def _training_adapter_dtype_arg(args) -> str:
    if getattr(args, "fp16", False):
        return "fp16"
    if getattr(args, "bf16", False):
        return "bf16"
    return "fp32"


def _balance_broadcast_shm_refcounts(tensors: dict, consumer_count: int) -> int:
    """Pre-pay the shm refcount for a payload that ``consumer_count`` processes rebuild.

    torch's ``file_system`` reduce/rebuild pair is a 1-producer -> 1-consumer
    handshake: ``reduce_storage`` calls ``storage._shared_incref()`` exactly
    once per serialization, and every ``rebuild_storage_filename`` ends in a
    matching ``_shared_decref()`` (torch/multiprocessing/reductions.py). SGLang
    hands ONE payload to EVERY TP scheduler and each deserializes it
    (tp_worker.py:218), so a tp_size=N engine decrefs N times against that
    single incref. The manager unlinks the segment N-1 releases early — while
    this actor still holds ``tensors`` — and the rank that opens last dies with
    ``unable to open shared memory object ... No such file or directory (2)``.
    Ranks skew by roughly a batch, so it fires intermittently and always on the
    slowest rank: three e4 gsm8k LoRA arms died this way at rollouts 28, 65 and
    114 on 2026-08-04.

    One extra incref per ADDITIONAL consumer restores the pairing. Deduped by
    storage because ForkingPickler reduces a storage once however many tensors
    view it — increfing per tensor would leak the segment instead.

    Returns the number of increfs performed, for tests and diagnostics.
    """
    if consumer_count <= 1:
        return 0
    increfs = 0
    seen: set[int] = set()
    for tensor in tensors.values():
        storage = tensor.untyped_storage()
        key = storage.data_ptr()
        if key in seen:
            continue
        seen.add(key)
        for _ in range(consumer_count - 1):
            storage._shared_incref()
            increfs += 1
    return increfs


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


def _prepend_pythonpath(path: Path):
    current = os.environ.get("PYTHONPATH", "")
    entries = [entry for entry in current.split(os.pathsep) if entry]
    path_str = str(path)
    if path_str not in entries:
        os.environ["PYTHONPATH"] = os.pathsep.join([path_str, *entries])


def _prepare_child_native_ops_env(force_native_ops: bool):
    if not force_native_ops:
        return

    os.environ["MILES_SGLANG_FORCE_NATIVE_OPS"] = "1"
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


def _configure_peft_cache_kwargs(kwargs: dict, peft_method: str | None):
    if peft_method not in {"lora", "oft"}:
        return

    if kwargs.get("disable_radix_cache") is not True:
        logger.warning(
            "Disabling SGLang radix cache for PEFT rollout; cached prefixes can "
            "produce stale adapter activations and train-inference mismatch."
        )
    kwargs["disable_radix_cache"] = True


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


def _launch_server_with_miles_compat(server_args: ServerArgs, force_native_ops: bool):
    _prepare_child_peft_cache_env(server_args)

    if force_native_ops:
        patch_sglang_native_ops()

    from sglang.srt.entrypoints.http_server import launch_server

    launch_server(server_args)


def launch_server_process(server_args: ServerArgs, force_native_ops: bool = False) -> multiprocessing.Process:

    multiprocessing.set_start_method("spawn", force=True)
    _prepare_child_native_ops_env(force_native_ops)
    p = multiprocessing.Process(target=_launch_server_with_miles_compat, args=(server_args, force_native_ops))
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
        logger.info(f"Launch HttpServerEngineAdapter at: {self.server_host}:{self.server_port}")
        # Current SGLang resolves ServerArgs during construction and treats its
        # public fields as read-only afterward. Normalize IPv6 brackets before
        # constructing it instead of mutating the resolved object.
        server_args_dict = {**server_args_dict, "host": server_args_dict["host"].strip("[]")}
        self.process = launch_server_process(
            ServerArgs(**server_args_dict),
            force_native_ops=getattr(self.args, "sglang_force_native_ops", False),
        )

        if self.node_rank == 0 and self.router_ip and self.router_port:
            if parse(sglang_router.__version__) <= parse("0.2.1") or self.args.use_miles_router:
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

    def get_remote_instance_transfer_engine_info(self, rank: int):
        # TODO: will be changed to `remote_instance_transfer_engine_info` when the sglang side is ready.
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

    def _adapter_payload_consumers(self) -> int:
        """How many processes will rebuild one broadcast adapter payload.

        One scheduler per TP rank of this engine, and no more: SGLang's
        dynamic-LoRA path asserts ``dp_size == 1``
        (tokenizer_communicator_mixin.py:1137), so tp_size is the whole
        fan-out. Falls back to 1 — the no-op count — when neither value is set,
        which keeps single-GPU smokes on torch's own accounting.
        """
        per_engine = self.num_gpus_per_engine or getattr(self.args, "rollout_num_gpus_per_engine", None)
        return int(per_engine or 1)

    def load_lora_adapter_from_ray_tensors(
        self,
        lora_name: str,
        tensors: dict,
        config_dict: dict,
        load_format: str | None = None,
        pinned: bool = False,
        added_tokens_config: dict | None = None,
    ):
        """Load LoRA tensors received through Ray.

        The SGLang HTTP endpoint deserializes tensors inside the scheduler
        process with ``MultiprocessingSerializer``. Serializing in the trainer
        actor can embed multiprocessing resource-sharer handles with a
        different auth key, so distributed Ray transport serializes here,
        inside the SGLangEngine actor that owns the server process.

        Serialized under the ``file_system`` sharing strategy, never the
        ``file_descriptor`` default. The endpoint takes ONE payload and every
        TP-rank scheduler deserializes it, but a fd-strategy DupFd is
        redeemable exactly once: on a TP=2 engine, TP0's deserialize consumes
        the fd and TP1 dies on EOFError in recvfds (measured on the
        2026-08-04 B200 probe, reproduced deterministically on CPU). A
        file_system storage is a named shm segment any process can attach
        any number of times, which is what a broadcast payload needs.

        Attaching is not the whole story: the segment's *lifetime* still has to
        be paid for, once per rank. See
        ``_balance_broadcast_shm_refcounts``.
        """
        import torch.multiprocessing as torch_mp  # local: the module keeps torch off its import path

        old_strategy = torch_mp.get_sharing_strategy()
        torch_mp.set_sharing_strategy("file_system")
        try:
            serialized_tensors = MultiprocessingSerializer.serialize(tensors, output_str=True)
            _balance_broadcast_shm_refcounts(tensors, self._adapter_payload_consumers())
        finally:
            torch_mp.set_sharing_strategy(old_strategy)
        return self.load_lora_adapter_from_tensors(
            lora_name=lora_name,
            serialized_tensors=serialized_tensors,
            config_dict=config_dict,
            load_format=load_format,
            pinned=pinned,
            added_tokens_config=added_tokens_config,
        )

    def load_oft_adapter_from_tensors(
        self,
        adapter_name: str,
        serialized_tensors: str,
        config_dict: dict,
        pinned: bool = False,
    ):
        """Load an OFT adapter from serialized tensor data.

        Requires the sglang server to be launched with ``peft_method="oft"``;
        miles's SGLangEngine kwargs builder sets this automatically when OFT
        is configured.
        """
        payload = {
            "adapter_name": adapter_name,
            "serialized_tensors": serialized_tensors,
            "config_dict": config_dict,
            "pinned": pinned,
        }
        return self._make_request("load_oft_adapter_from_tensors", payload)

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

    def update_adapter_from_ray_tensor(
        self,
        *,
        flat_tensor,
        metadata: dict,
        entries: list,
        payload_tag: str,
        load_format: str,
        adapter_config: dict,
        adapter_name: str,
    ):
        """Update PEFT tensors received through Ray via SGLang's streamed loader.

        ``payload_tag`` is the per-method wire tag: sglang's
        normalize_{oft,lora}_weight_payload asserts on "flattened_oft_payload" /
        "flattened_lora_payload" respectively, so it must follow the method, not
        be hardcoded -- LoRA reaches this path too now that it has a shaper.

        SGLang selects ``serialized_named_tensors[tp_rank]``, so every TP
        scheduler needs its own serialized entry.  Serialize each entry under
        ``file_system`` sharing: each serialization supplies the one shared-
        memory ownership credit redeemed by its matching scheduler.  Unlike a
        broadcast payload, these entries have one consumer each and therefore
        require no refcount pre-payment.

        POSIX shared-memory names are local to the engine host, so this
        transport supports only a single-host SGLang engine.
        """
        if self.nnodes > 1:
            raise RuntimeError("Ray PEFT tensor serialization currently supports only a single-host SGLang engine.")

        import torch.multiprocessing as torch_mp

        old_strategy = torch_mp.get_sharing_strategy()
        torch_mp.set_sharing_strategy("file_system")
        try:
            serialized_rank_payloads = []
            for _ in range(self._adapter_payload_consumers()):
                inner = (
                    payload_tag,
                    MultiprocessingSerializer.serialize(flat_tensor),
                    metadata,
                    entries,
                )
                serialized_rank_payloads.append(MultiprocessingSerializer.serialize(inner, output_str=True))
        finally:
            torch_mp.set_sharing_strategy(old_strategy)
        return self.update_weights_from_tensor(
            serialized_named_tensors=serialized_rank_payloads,
            load_format=load_format,
            adapter_config=adapter_config,
            adapter_name=adapter_name,
        )

    def update_adapter_from_rank_tensors(
        self,
        *,
        rank_payloads: list[tuple],
        payload_tag: str,
        load_format: str,
        adapter_config: dict,
        adapter_name: str,
    ):
        """Serialize colocated PEFT TP shards inside the SGLang parent actor.

        Condor and other restricted runtimes can deny ``pidfd_getfd``, making
        CUDA IPC handles produced by trainer actors impossible for scheduler
        children to rebuild. CPU copies arrive here through Ray, then named
        shared-memory serialization gives each TP scheduler a parent-owned
        payload without crossing the restricted CUDA IPC boundary.

        ``payload_tag`` is the per-method wire tag SGLang asserts on --
        "flattened_oft_payload" / "flattened_lora_payload".
        """
        if self.nnodes > 1:
            raise RuntimeError(
                "PEFT rank-tensor serialization currently supports only a single-host " "SGLang engine."
            )

        import torch.multiprocessing as torch_mp

        old_strategy = torch_mp.get_sharing_strategy()
        torch_mp.set_sharing_strategy("file_system")
        try:
            serialized_rank_payloads = []
            for flat_tensor, metadata, entries in rank_payloads:
                inner = (
                    payload_tag,
                    MultiprocessingSerializer.serialize(flat_tensor),
                    metadata,
                    entries,
                )
                serialized_rank_payloads.append(MultiprocessingSerializer.serialize(inner, output_str=True))
            return self.update_weights_from_tensor(
                serialized_named_tensors=serialized_rank_payloads,
                load_format=load_format,
                adapter_config=adapter_config,
                adapter_name=adapter_name,
            )
        finally:
            torch_mp.set_sharing_strategy(old_strategy)

    def flush_cache(self):
        """Flush the cache of the server."""
        if self.node_rank != 0:
            return
        last_message = None
        for _ in range(60):
            try:
                response = requests.get(f"http://{self.server_host}:{self.server_port}/flush_cache")
                if response.status_code == 200:
                    break
                last_message = response.text
            except NewConnectionError as e:
                raise e
            except Exception as e:
                logger.info(f"Error flushing cache: {e}")
                last_message = str(e)
            time.sleep(1)
        else:
            raise TimeoutError(f"Timeout while flushing cache: {last_message}")

    def shutdown(self):
        if self.args.rollout_external:
            return

        logger.info(f"Shutdown engine {self.server_host}:{self.server_port}...")
        if self.node_rank == 0:
            worker_url = f"http://{self.server_host}:{self.server_port}"
            response = None
            if parse(sglang_router.__version__) <= parse("0.2.1") or self.args.use_miles_router:
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
        """Unload LoRA adapter."""
        return self._make_request(
            "unload_lora_adapter",
            {"lora_name": lora_name},
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

    def update_adapter_from_distributed(
        self,
        *,
        names: list[str],
        dtypes: list[str],
        shapes: list[list[int]],
        group_name: str,
        weight_version: str,
        load_format: str,
        adapter_config: dict,
        adapter_name: str,
        payload_metadata: dict | None = None,
        adapter_version: str | None = None,
        double_buffer: bool = False,
    ):
        payload = {
            "names": names,
            "dtypes": [str(dtype).replace("torch.", "") for dtype in dtypes],
            "shapes": shapes,
            "group_name": group_name,
            "weight_version": weight_version,
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

    def begin_weight_update(self, selector: str = "all"):
        """Open a weight-update session on the engine (restores packed weights for loading)."""
        return self._make_request("begin_weight_update", {"selector": selector})

    def end_weight_update(self):
        """Close the weight-update session (post-load + quant post-process on the full model)."""
        return self._make_request("end_weight_update", {})

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
        kwargs["peft_method"] = "lora"
        kwargs["peft_target_modules"] = convert_target_modules_to_hf(args.target_modules)
        kwargs["peft_max_lora_rank"] = max(getattr(args, "lora_rank", 0), 1)
        kwargs["peft_double_buffer"] = bool(getattr(args, "adapter_double_buffer", False))
        if args.lora_adapter_path is not None:
            kwargs["peft_paths"] = {LORA_ADAPTER_NAME: args.lora_adapter_path}
        else:
            logger.info("No pre-trained LoRA adapter_path provided, will use random initial weights")
    elif is_multi_lora_enabled(args):
        kwargs["enable_lora"] = True
        kwargs["max_loras_per_batch"] = args.multi_lora_n_adapters
        kwargs["max_lora_rank"] = max(getattr(args, "lora_rank", 0), 1)
        kwargs["lora_target_modules"] = convert_target_modules_to_hf(args.target_modules)
    elif lora_rollout_enabled(args):
        kwargs["enable_lora"] = True
        kwargs["max_loras_per_batch"] = 1
        kwargs["max_lora_rank"] = max(getattr(args, "lora_rank", 0), 1)
        if sglang_lora_target_all_sentinel(args):
            kwargs["lora_target_modules"] = ["all"]
        else:
            kwargs["lora_target_modules"] = convert_target_modules_to_hf(args.target_modules)

        if args.lora_adapter_path is not None and kwargs.get("load_format") != "dummy":
            kwargs["lora_paths"] = {LORA_ADAPTER_NAME: args.lora_adapter_path}
        elif args.lora_adapter_path is not None:
            logger.info("dummy base load: skipping startup lora_paths; adapter comes via weight-sync")
        else:
            logger.info("No pre-trained LoRA adapter_path provided, will use random initial weights")

        if lora_base_cpu_backup_enabled(args):
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
        # Enable the fork's stage/activate double-buffer path (staging slot =
        # max_ofts_per_batch-1); paired with the max_ofts_per_batch=3 bump above.
        kwargs["peft_double_buffer"] = bool(getattr(args, "adapter_double_buffer", False))
        kwargs["oft_backend"] = getattr(args, "sglang_oft_backend", "triton")
        oft_adapter_path = getattr(args, "oft_adapter_path", None)
        if oft_adapter_path is not None:
            kwargs["peft_paths"] = {OFT_ADAPTER_NAME: oft_adapter_path}
        else:
            logger.info("No pre-trained OFT adapter_path provided, will use random initial weights")

    # Last, so a per-group override wins over every args-derived default above.
    if sglang_overrides:
        kwargs.update(sglang_overrides)

    _configure_peft_cache_kwargs(kwargs, peft_method)
    _configure_megatron_moe_parity_kwargs(kwargs, args, sglang_overrides)

    external_engine_need_check_fields = [k for k in kwargs.keys() if k not in _EXTERNAL_ENGINE_SKIP_CHECK_FIELDS]

    unused_keys = set(kwargs.keys())
    for attr in dataclasses.fields(ServerArgs):
        if worker_type == "decode" and attr.name == "enable_hierarchical_cache":
            continue
        if hasattr(args, f"sglang_{attr.name}") and attr.name not in kwargs:
            kwargs[attr.name] = getattr(args, f"sglang_{attr.name}")
        unused_keys.discard(attr.name)

    # for compatibility with old args
    if len(unused_keys) > 0:
        logger.info(f"Warning: The following arguments is not supported in the current sglang: {unused_keys}.")
        for key in unused_keys:
            kwargs.pop(key)

    return kwargs, external_engine_need_check_fields


_EXTERNAL_ENGINE_SKIP_CHECK_FIELDS = [
    "model_path",
    "trust_remote_code",
    "random_seed",
    "nccl_port",
    "dist_init_addr",
    "skip_server_warmup",
    "enable_draft_weights_cpu_backup",
    "enable_metrics",
    "mem_fraction_static",
]
