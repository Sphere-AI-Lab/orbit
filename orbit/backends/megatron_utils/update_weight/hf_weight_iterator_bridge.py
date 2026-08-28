import dataclasses

from orbit.utils import megatron_bridge_utils
from orbit.utils.iter_utils import chunk_named_params_by_size

from ..megatron_to_hf import postprocess_hf_param
from ..misc_utils import strip_param_name_prefix
from .hf_weight_iterator_base import HfWeightIteratorBase


class HfWeightIteratorBridge(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        from megatron.bridge import AutoBridge

        self._bridge = AutoBridge.from_hf_pretrained(self.args.hf_checkpoint, trust_remote_code=True)

    def get_hf_weight_chunks(self, megatron_local_weights):
        # Follow-up: support quantization (e.g. modify megatron-bridge to provide megatron param name)
        renamed_megatron_local_weights = {strip_param_name_prefix(k): v for k, v in megatron_local_weights.items()}
        with megatron_bridge_utils.patch_megatron_model(self.model):
            named_weights = self._export_named_weights(renamed_megatron_local_weights)

            # Newer megatron-bridge yields HFWeightTuple(hf_name, tensor) -- a
            # 2-tuple -- whereas older versions yielded
            # (hf_name, tensor, megatron_name). The megatron_name was only
            # used by postprocess_hf_param to detect embedding/output layers;
            # we now recognize those by HF name instead, so an empty
            # megatron_param_name is fine for the newer API.
            def _to_triple(item):
                if len(item) == 3:
                    return item
                hf_name, tensor = item
                return hf_name, tensor, ""

            named_weights = (
                (
                    hf_param_name,
                    postprocess_hf_param(
                        args=self.args,
                        megatron_param_name=megatron_param_name,
                        hf_param_name=hf_param_name,
                        param=weight,
                    ),
                )
                for hf_param_name, weight, megatron_param_name in map(_to_triple, named_weights)
            )

            yield from chunk_named_params_by_size(named_weights, chunk_size=self.args.update_weight_buffer_size)

    def _export_named_weights(self, renamed_megatron_local_weights):
        if self.peft_method == "lora":
            return self._bridge.export_adapter_weights(
                self.model,
                cpu=False,
                show_progress=False,
            )
        if self.peft_method == "oft":
            # Free function (megatron.bridge.orbit namespace, post-reattach),
            # not a bridge method -- takes the bridge as an explicit first arg.
            from megatron.bridge.orbit.conversion.oft_export import export_oft_adapter_weights

            return export_oft_adapter_weights(
                self._bridge,
                self.model,
                cpu=False,
                show_progress=False,
            )

        conversion_tasks = self._bridge.get_conversion_tasks(self.model)
        conversion_tasks = _process_conversion_tasks(conversion_tasks, renamed_megatron_local_weights)
        return self._bridge.export_hf_weights(
            self.model,
            cpu=False,
            conversion_tasks=conversion_tasks,
        )


def _process_conversion_tasks(vanilla_conversion_tasks, new_weight_dict):
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
