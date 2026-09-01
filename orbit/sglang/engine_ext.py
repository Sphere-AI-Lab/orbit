"""Orbit's adapter-tensor-loading / distributed-PEFT-sync SGLangEngine methods.

Home mixin for the orbit-added ``SGLangEngine`` methods lifted out of
miles/backends/sglang_utils/sglang_engine.py (Phase 3 isolation, slice 3c):
Ray-transported LoRA/OFT tensor loading (with the shm-refcount fix), the
streamed-loader distributed adapter update path, double-buffer adapter version
activation, OFT adapter unload, and the 404-tolerant weight-update lifecycle
(``post_process_weights`` / ``begin_weight_update`` / ``end_weight_update``).
``SGLangEngine`` in the miles file lists ``OrbitEngineExtensions`` as its first
base; every method here runs with ``self`` bound to a live ``SGLangEngine``
instance and reaches base-class state/methods (``self.args``,
``self.num_gpus_per_engine``, ``self.nnodes``,
``self.load_lora_adapter_from_tensors``, ``self.update_weights_from_tensor``,
``self._make_request``, and the ``_supports_*`` probe flags seeded in
``SGLangEngine.__init__``) the normal attribute-lookup way -- no re-imports
needed for those.

Plain mixin: no ``__init__``, no state of its own.
"""

import requests
from sglang.srt.utils import MultiprocessingSerializer

from orbit.sglang.shm_refcounts import _balance_broadcast_shm_refcounts


class OrbitEngineExtensions:
    def _adapter_payload_consumers(self) -> int:
        """How many processes will rebuild one broadcast adapter payload.

        One scheduler per TP rank of this engine, and no more: SGLang's
        dynamic-LoRA path asserts ``dp_size == 1``
        (tokenizer_communicator_mixin.py:1137), so tp_size is the whole
        fan-out. Falls back to 1 — the no-op count — when neither value is set,
        which keeps single-GPU smokes on torch's own accounting.
        """
        per_engine = self.num_gpus_per_engine or getattr(
            self.args, "rollout_num_gpus_per_engine", None
        )
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
        orbit's SGLangEngine kwargs builder sets this automatically when OFT
        is configured.
        """
        payload = {
            "adapter_name": adapter_name,
            "serialized_tensors": serialized_tensors,
            "config_dict": config_dict,
            "pinned": pinned,
        }
        return self._make_request("load_oft_adapter_from_tensors", payload)

    def unload_oft_adapter(self, adapter_name: str):
        # OFT counterpart to the base class's unload_lora_adapter; kept
        # undocumented so the pair reads symmetrically.
        return self._make_request(
            "unload_oft_adapter",
            {"adapter_name": adapter_name},
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
        scheduler needs its own serialized entry. Serialize each entry under
        ``file_system`` sharing: each serialization supplies the one shared-
        memory ownership credit redeemed by its matching scheduler. Unlike a
        broadcast payload, these entries have one consumer each and therefore
        require no refcount pre-payment.

        POSIX shared-memory names are local to the engine host, so this
        transport supports only a single-host SGLang engine.
        """
        if self.nnodes > 1:
            raise RuntimeError(
                "Ray PEFT tensor serialization currently supports only a single-host "
                "SGLang engine."
            )

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
                serialized_rank_payloads.append(
                    MultiprocessingSerializer.serialize(inner, output_str=True)
                )
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
                "PEFT rank-tensor serialization currently supports only a single-host "
                "SGLang engine."
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
                serialized_rank_payloads.append(
                    MultiprocessingSerializer.serialize(inner, output_str=True)
                )
            return self.update_weights_from_tensor(
                serialized_named_tensors=serialized_rank_payloads,
                load_format=load_format,
                adapter_config=adapter_config,
                adapter_name=adapter_name,
            )
        finally:
            torch_mp.set_sharing_strategy(old_strategy)

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

    # --- weight-update lifecycle ------------------------------------------
    # Orbit runs against a range of sglang builds, so each of the three calls
    # below probes its endpoint once and remembers a 404 in the matching
    # ``self._supports_*`` flag seeded by ``SGLangEngine.__init__``. Skipping is
    # safe: builds without these endpoints load weights directly, and for a
    # BF16/non-quantized/colocate setup the post-process step is a no-op.

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

    def begin_weight_update(self, selector: str = "all"):
        """Open a weight-update session on the engine (restores packed weights for loading)."""
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
