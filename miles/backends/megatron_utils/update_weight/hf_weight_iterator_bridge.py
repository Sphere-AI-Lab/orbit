import dataclasses
import json
import os

from miles.backends.megatron_utils.lora_utils import is_lora_weight_name
from miles.utils import megatron_bridge_utils

from ..megatron_to_hf import postprocess_hf_param
from ..megatron_to_hf.processors import quantize_params
from ..misc_utils import strip_param_name_prefix
from .common import get_atomic_update_groups
from .hf_weight_iterator_base import HfWeightIteratorBase


class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from megatron.bridge import AutoBridge

        self._bridge = AutoBridge.from_hf_pretrained(self.args.hf_checkpoint, trust_remote_code=True)

        if (
            self.quantization_config is not None
            and self.quantization_config.get("quant_method") == "compressed-tensors"
        ):
            quantized_basenames = _load_quantized_param_basenames(self.args.hf_checkpoint)
            if quantized_basenames is not None:
                # Quantize exactly the params the checkpoint stores packed; the
                # published ignore list of multimodal checkpoints (e.g.
                # Kimi-K2.5 VL) omits vision_tower/mm_projector, so it cannot
                # be trusted as the sole quantization criterion.
                self.quantization_config = {
                    **self.quantization_config,
                    "_miles_quantized_basenames": quantized_basenames,
                }

    def get_hf_weight_chunks(self, megatron_local_weights, weight_type: str = "base"):
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in megatron_local_weights.items()}
        with megatron_bridge_utils.patch_megatron_model(self.model):
            # ORBIT-SEAM: upstream's inline lora/base export branch pulled into _export_named_weights,
            # which adds the OFT adapter-export path (self.peft_method) on top of upstream's weight_type
            named_weights = self._export_named_weights(renamed_megatron_local_weights, weight_type)

            # ORBIT-SEAM: 2-tuple/3-tuple compat shim. Newer megatron-bridge yields
            # HFWeightTuple(hf_name, tensor) -- a 2-tuple -- whereas older versions yielded
            # (hf_name, tensor, megatron_name). The megatron_name is only used to detect
            # embedding/output layers (postprocess_hf_param recognizes those by HF name too) and to
            # match atomic-update-group suffixes, so an empty megatron name is safe here.
            named_weights = map(_to_triple, named_weights)

            # Apply postprocess + quantization (when targeting a quantized rollout,
            # e.g. FP8 sglang). Base weights are quantized to match the rollout's
            # storage format so update_weights_from_tensor lands real weight + scale
            # pairs; LoRA adapters are passed through unchanged.
            named_weights = self._postprocess_and_quantize(named_weights, weight_type)

            # ORBIT-SEAM: the adapter-name predicate is PEFT-method aware (OFT tensors are named
            # .oft_R / .oft_r and never match is_lora_weight_name)
            if weight_type == "base":
                named_weights = ((h, w, m) for h, w, m in named_weights if not self._is_adapter_weight_name(h))
            elif weight_type == "lora":
                named_weights = ((h, w, m) for h, w, m in named_weights if self._is_adapter_weight_name(h))

            groups = get_atomic_update_groups(self.args, self.model_name)
            units = _stream_atomic_units(named_weights, groups)
            yield from _chunk_atomic_units_by_size(units, chunk_size=self.args.update_weight_buffer_size)

    # ORBIT-SEAM: upstream's inline weight_type export branch pulled into this helper and extended
    # with an OFT export path; self.peft_method selects the adapter exporter for weight_type="lora"
    def _export_named_weights(self, renamed_megatron_local_weights, weight_type: str):
        if weight_type == "lora":
            if self.peft_method == "oft":
                # The pinned Bridge (988d6426) exposes this as an AutoBridge method;
                # orbit-main's Bridge lineage as a free function taking the bridge
                # first. oft_adapter_exporter binds whichever is installed.
                from orbit.megatron.oft_export_compat import oft_adapter_exporter

                return oft_adapter_exporter(self._bridge)(
                    self.model,
                    cpu=False,
                    show_progress=False,
                )
            return self._bridge.export_adapter_weights(
                self.model,
                cpu=False,
                show_progress=False,
            )

        assert weight_type == "base", f"Unsupported weight_type: {weight_type!r}"
        conversion_tasks = self._bridge.get_conversion_tasks(self.model)
        conversion_tasks = _process_conversion_tasks(conversion_tasks, renamed_megatron_local_weights)
        return self._bridge.export_hf_weights(
            self.model,
            cpu=False,
            conversion_tasks=conversion_tasks,
            merge_adapter_weights=False,
        )

    # ORBIT-SEAM: PEFT-method aware adapter-name predicate (LoRA + OFT), replacing upstream's bare
    # is_lora_weight_name at the two filter sites above
    def _is_adapter_weight_name(self, hf_name: str) -> bool:
        if self.peft_method == "oft":
            from orbit.megatron.oft_utils import is_oft_weight_name

            return is_oft_weight_name(hf_name)
        return is_lora_weight_name(hf_name)

    def _postprocess_and_quantize(self, named_weights, weight_type: str):
        for hf_param_name, weight, megatron_param_name in named_weights:
            hf_name = hf_param_name.replace(".base_layer.", ".")
            weight = postprocess_hf_param(
                args=self.args,
                megatron_param_name=megatron_param_name,
                hf_param_name=hf_name,
                param=weight,
            )
            if weight_type == "base" and self.quantization_config is not None:
                # quantize_params expects the megatron name with the `module.module.`
                # prefix that the direct iterator uses; the bridge yields it without.
                qmegatron_name = f"module.module.{megatron_param_name}"
                for q_hf_name, q_weight in quantize_params(
                    self.args, qmegatron_name, [(hf_name, weight)], self.quantization_config
                ):
                    yield q_hf_name, q_weight, megatron_param_name
            else:
                yield hf_name, weight, megatron_param_name


def _load_quantized_param_basenames(hf_checkpoint):
    """Base names of params stored packed (`<base>.weight_packed`) in the checkpoint, or None if unknown."""
    index_path = os.path.join(hf_checkpoint, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        return None
    with open(index_path) as f:
        names = json.load(f)["weight_map"]
    return {n.removesuffix(".weight_packed") for n in names if n.endswith(".weight_packed")}


def _stream_atomic_units(items, atomic_update_groups):
    """Streaming counterpart of get_named_value_update_units: buffer items
    whose megatron name matches an AtomicUpdateGroup suffix until every
    suffix in the same (prefix, group.key) arrives, then yield together."""
    pending: dict[tuple[str, str], list] = {}
    for hf_name, weight, megatron_name in items:
        match = next(
            (
                (group, idx, suffix)
                for group in atomic_update_groups
                for idx, suffix in enumerate(group.suffixes)
                if megatron_name.endswith(suffix)
            ),
            None,
        )
        if match is None:
            yield [(hf_name, weight)]
            continue
        group, idx, suffix = match
        prefix = megatron_name[: -len(suffix)]
        slots = pending.setdefault((prefix, group.key), [None] * len(group.suffixes))
        slots[idx] = (hf_name, weight)
        if None not in slots:
            yield list(slots)
            del pending[(prefix, group.key)]
    assert not pending, f"Incomplete atomic update groups at end of stream: {sorted(pending)}"


def _chunk_atomic_units_by_size(units, chunk_size):
    """Pack atomic units into chunks <= chunk_size bytes, never splitting a unit."""
    bucket: list = []
    bucket_size = 0
    for unit in units:
        unit_size = sum(t.nbytes for _, t in unit)
        if bucket and bucket_size + unit_size >= chunk_size:
            yield bucket
            bucket = []
            bucket_size = 0
        bucket.extend(unit)
        bucket_size += unit_size
    if bucket:
        yield bucket


# ORBIT-SEAM: 2-tuple/3-tuple megatron-bridge export compat, see get_hf_weight_chunks
def _to_triple(item):
    if len(item) == 3:
        return item
    hf_name, tensor = item
    return hf_name, tensor, ""


def _process_conversion_tasks(vanilla_conversion_tasks, new_weight_dict):
    # ORBIT-SEAM: two None/missing-key guards added for model families whose conversion tasks
    # (Gemma-4) can be unmapped or buffer-like; base's hard assert on a missing weight_dict_key
    # replaced with a keep-as-is fallback since orbit now runs a broader model set through this path
    def _handle_one(task):
        if task is None:
            # no HF mapping (e.g. Gemma-4 post_shared_expert_layernorm)
            return task
        if task.param_weight is None:
            return task

        weight_dict_key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        if weight_dict_key not in new_weight_dict:
            # buffer-like params (Gemma-4 layer_scalar/scale) aren't in optimizer state; keep as-is
            return task
        new_param_weight = new_weight_dict[weight_dict_key]
        new_param_weight = new_param_weight.cuda()
        return dataclasses.replace(task, param_weight=new_param_weight)

    return _MapWithLen(_handle_one, vanilla_conversion_tasks)


class _MapWithLen:
    def __init__(self, fn, xs):
        self.fn = fn
        self.xs = xs

    def __len__(self):
        return len(self.xs)

    def __iter__(self):
        for x in self.xs:
            yield self.fn(x)
