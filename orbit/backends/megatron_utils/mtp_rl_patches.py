import logging

logger = logging.getLogger(__name__)

_PATCHED: set[str] = set()


def apply_mtp_in_rl_patches(args) -> None:
    """
    Patches:
    - ``MultiTokenPredictionLayer._get_embeddings`` (defensive ``position_ids``
      check + ``decoder_input.detach()`` + ``keep_graph=False``).
    - ``MultiTokenPredictionLayer._checkpointed_forward`` (handle non-tensor
      args correctly under activation checkpointing).
    - ``GPTModel.forward`` (accept ``mtp_kwargs`` and thread it through).
    - ``GPTModel._postprocess`` (use ``mtp_kwargs['mtp_labels']`` and detach
      output-layer parameters via ``torch.func.functional_call`` so MTP
      gradients don't flow back into the output layer in RL training).

    Gated on ``args.mtp_num_layers > 0`` so non-MTP runs see no behavior
    change. Note: caller code (orbit's training loop) must produce and pass
    ``mtp_kwargs={'mtp_labels': ...}`` for the new MTP branch to fire;
    without that, the patched ``_postprocess`` falls through identically to
    the unpatched version.
    """
    if "mtp_in_rl" in _PATCHED:
        return
    if int(getattr(args, "mtp_num_layers", 0) or 0) <= 0:
        return

    try:
        import torch
        from megatron.core import parallel_state, tensor_parallel
        from megatron.core.transformer.multi_token_prediction import (
            MTPLossAutoScaler,
            MTPLossLoggingHelper,
            MultiTokenPredictionLayer,
            roll_tensor,
        )
        from megatron.core.models.gpt.gpt_model import GPTModel
        from megatron.core.utils import make_viewless_tensor
    except Exception as e:
        logger.warning("megatron MTP-RL patch skipped: import failed (%s)", e)
        return

    # ----- MultiTokenPredictionLayer._get_embeddings -----
    def _patched_get_embeddings(self, input_ids, position_ids, embedding, hidden_states, packed_seq_params=None):
        input_ids, _ = roll_tensor(
            input_ids, shifts=-1, dims=-1, cp_group=self.cp_group, packed_seq_params=packed_seq_params
        )
        if position_ids is not None:
            position_ids, _ = roll_tensor(
                position_ids, shifts=-1, dims=-1, cp_group=self.cp_group, packed_seq_params=packed_seq_params
            )
        decoder_input = embedding(input_ids=input_ids, position_ids=position_ids)
        decoder_input = decoder_input.detach()
        hidden_states = make_viewless_tensor(inp=hidden_states, requires_grad=True, keep_graph=False)
        return input_ids, position_ids, decoder_input, hidden_states

    # ----- MultiTokenPredictionLayer._checkpointed_forward -----
    def _patched_checkpointed_forward(self, forward_func, *args_, **kwargs_):
        positional_specs = []
        kw_specs = []
        tensor_args = []

        for arg in args_:
            if torch.is_tensor(arg):
                positional_specs.append(("tensor", len(tensor_args)))
                tensor_args.append(arg)
            else:
                positional_specs.append(("const", arg))

        for key, value in kwargs_.items():
            if torch.is_tensor(value):
                kw_specs.append((key, ("tensor", len(tensor_args))))
                tensor_args.append(value)
            else:
                kw_specs.append((key, ("const", value)))

        def run(*flat_tensor_args):
            rebuilt_args = []
            for spec_type, payload in positional_specs:
                rebuilt_args.append(flat_tensor_args[payload] if spec_type == "tensor" else payload)
            rebuilt_kwargs = {}
            for key, (spec_type, payload) in kw_specs:
                rebuilt_kwargs[key] = flat_tensor_args[payload] if spec_type == "tensor" else payload
            return forward_func(*rebuilt_args, **rebuilt_kwargs)

        tensor_args_tuple = tuple(tensor_args)

        def checkpoint_handler():
            if self.config.fp8:
                from megatron.core.extensions.transformer_engine import te_checkpoint
                return te_checkpoint(
                    forward_func,
                    self.config.distribute_saved_activations,
                    tensor_parallel.random.get_cuda_rng_tracker,
                    parallel_state.get_tensor_model_parallel_group(),
                    *tensor_args_tuple,
                )
            return tensor_parallel.checkpoint(
                run, self.config.distribute_saved_activations, *tensor_args_tuple
            )

        if self.config.recompute_method == "uniform":
            assert self.config.recompute_num_layers == 1, "recompute_num_layers must be 1 for MTP recompute"
            return checkpoint_handler()
        if self.config.recompute_method == "block":
            import warnings as _warnings
            _warnings.warn("recompute_method == 'block' is not supported for MTP yet. Skipping recompute.")
            return forward_func(*args_, **kwargs_)
        raise ValueError("Invalid activation recompute method.")

    MultiTokenPredictionLayer._get_embeddings = _patched_get_embeddings
    MultiTokenPredictionLayer._checkpointed_forward = _patched_checkpointed_forward

    # ----- MTP-RL helper: process_mtp_loss with detached output_layer params -----
    def _process_mtp_loss_rl(
        hidden_states,
        mtp_labels,
        loss_mask,
        output_layer,
        output_weight,
        runtime_gather_output,
        is_training,
        compute_language_model_loss,
        config,
        cp_group=None,
        packed_seq_params=None,
        scale_logits_fn=None,
    ):
        hidden_states_list = torch.chunk(hidden_states, 1 + config.mtp_num_layers, dim=0)
        hidden_states = hidden_states_list[0]

        if loss_mask is None:
            loss_mask = torch.ones_like(mtp_labels)
        else:
            loss_mask, _ = roll_tensor(
                loss_mask, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params
            )
        original_num_tokens = loss_mask.sum()

        # Detach output_layer params/buffers once; reused per MTP layer.
        output_layer_params = {k: v.detach() for k, v in output_layer.named_parameters()}
        output_layer_buffers = dict(output_layer.named_buffers())

        for mtp_layer_number in range(config.mtp_num_layers):
            mtp_logits, _ = torch.func.functional_call(
                output_layer,
                {**output_layer_params, **output_layer_buffers},
                (hidden_states_list[mtp_layer_number + 1],),
                {
                    "weight": output_weight.detach() if output_weight is not None else None,
                    "runtime_gather_output": runtime_gather_output,
                },
            )
            if scale_logits_fn is not None:
                mtp_logits = scale_logits_fn(mtp_logits)
            mtp_labels, _ = roll_tensor(
                mtp_labels, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params
            )
            loss_mask, num_tokens = roll_tensor(
                loss_mask, shifts=-1, dims=-1, cp_group=cp_group, packed_seq_params=packed_seq_params
            )
            mtp_loss = compute_language_model_loss(mtp_labels, mtp_logits) * loss_mask
            if is_training:
                mtp_loss_for_log = (
                    torch.sum(mtp_loss) / num_tokens if num_tokens > 0 else mtp_loss.new_tensor(0.0)
                )
                MTPLossLoggingHelper.save_loss_to_tracker(
                    mtp_loss_for_log,
                    mtp_layer_number,
                    config.mtp_num_layers,
                    avg_group=parallel_state.get_data_parallel_group(with_context_parallel=True),
                )
            mtp_loss_scale = config.mtp_loss_scaling_factor / config.mtp_num_layers
            if config.calculate_per_token_loss:
                num_tokens_safe = torch.clamp(num_tokens, min=1)
                hidden_states = MTPLossAutoScaler.apply(
                    hidden_states,
                    mtp_loss_scale * mtp_loss * (original_num_tokens / num_tokens_safe),
                )
            else:
                safe_num_tokens = num_tokens.clamp(min=1)
                hidden_states = MTPLossAutoScaler.apply(
                    hidden_states, mtp_loss_scale * mtp_loss / safe_num_tokens
                )

        return hidden_states

    # GPTModel.forward picks up an `mtp_kwargs` kwarg; _postprocess reads it
    # via `self._mtp_kwargs`. Avoids copying the ~120-line forward body.
    orig_forward = GPTModel.forward
    orig_postprocess = GPTModel._postprocess

    def _patched_forward(self, *fa, **fk):
        self._mtp_kwargs = fk.pop("mtp_kwargs", None) or {}
        try:
            return orig_forward(self, *fa, **fk)
        finally:
            self._mtp_kwargs = None

    def _patched_postprocess(self, *pa, **pk):
        mtp_labels = (getattr(self, "_mtp_kwargs", None) or {}).get("mtp_labels")
        if mtp_labels is None or not self.config.mtp_num_layers:
            return orig_postprocess(self, *pa, **pk)

        # Bind hidden_states from positional or kwarg form.
        pa = list(pa)
        hidden_states = pa[0] if pa else pk["hidden_states"]

        output_weight = (
            self.shared_embedding_or_output_weight()
            if self.share_embeddings_and_output_weights
            else None
        )

        hidden_states = _process_mtp_loss_rl(
            hidden_states=hidden_states,
            mtp_labels=mtp_labels.clone(),
            loss_mask=pk.get("loss_mask"),
            output_layer=self.output_layer,
            output_weight=output_weight,
            runtime_gather_output=pk.get("runtime_gather_output"),
            is_training=self.training,
            compute_language_model_loss=self.compute_language_model_loss,
            config=self.config,
            cp_group=getattr(getattr(self, "pg_collection", None), "cp", None),
            packed_seq_params=pk.get("packed_seq_params"),
            scale_logits_fn=self._scale_logits if getattr(self.config, "use_mup", False) else None,
        )

        # Hand truncated hidden_states back; zero mtp_num_layers so upstream
        # skips its own (now-redundant) MTP branch.
        if pa:
            pa[0] = hidden_states
        else:
            pk["hidden_states"] = hidden_states
        saved_mtp_num = self.config.mtp_num_layers
        self.config.mtp_num_layers = 0
        try:
            return orig_postprocess(self, *pa, **pk)
        finally:
            self.config.mtp_num_layers = saved_mtp_num

    GPTModel.forward = _patched_forward
    GPTModel._postprocess = _patched_postprocess

    _PATCHED.add("mtp_in_rl")
    logger.info("megatron MTP-in-RL: patched MultiTokenPredictionLayer + GPTModel for MTP RL training")
