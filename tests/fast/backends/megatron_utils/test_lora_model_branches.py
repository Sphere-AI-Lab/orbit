"""Mock-based tests for LoRA branch logic in miles.backends.megatron_utils.model.

Validates that setup_model_and_optimizer, save, and save_hf_model correctly
route to LoRA-specific code paths depending on configuration — without GPU.
"""

from argparse import Namespace
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# _ensure_model_list
# ---------------------------------------------------------------------------


class TestEnsureModelList:
    def test_list_passthrough(self):
        from miles.backends.megatron_utils.model import _ensure_model_list

        models = [MagicMock(), MagicMock()]
        assert _ensure_model_list(models) is models

    def test_non_list_wrapped(self):
        from miles.backends.megatron_utils.model import _ensure_model_list

        model = MagicMock()
        result = _ensure_model_list(model)
        assert isinstance(result, list)
        assert result[0] is model


# ---------------------------------------------------------------------------
# should_disable_forward_pre_hook
# ---------------------------------------------------------------------------


class TestShouldDisableForwardPreHook:
    def test_both_true(self):
        from miles.backends.megatron_utils.model import should_disable_forward_pre_hook

        args = Namespace(use_distributed_optimizer=True, overlap_param_gather=True)
        assert should_disable_forward_pre_hook(args) is True

    def test_optimizer_false(self):
        from miles.backends.megatron_utils.model import should_disable_forward_pre_hook

        args = Namespace(use_distributed_optimizer=False, overlap_param_gather=True)
        assert should_disable_forward_pre_hook(args) is False

    def test_overlap_false(self):
        from miles.backends.megatron_utils.model import should_disable_forward_pre_hook

        args = Namespace(use_distributed_optimizer=True, overlap_param_gather=False)
        assert should_disable_forward_pre_hook(args) is False


# ---------------------------------------------------------------------------
# setup_model_and_optimizer — LoRA branch routing
# ---------------------------------------------------------------------------

_MODEL_MODULE = "miles.backends.megatron_utils.model"


class TestSetupModelAndOptimizerLoraBranch:
    """Verify that PEFT-enabled actor + bridge mode routes to _setup_peft_model_via_bridge."""

    # orbit: base's bridge branch is `is_lora_enabled(args) and role == "actor"`, where LoRA is
    # inferred from lora_rank. Orbit widened it to its whole PEFT layer (LoRA *and* OFT):
    # `is_peft_enabled(args)` reads `args.peft_method`, and the adapter-mode critic is admitted
    # alongside the actor. The helper is renamed `_setup_peft_model_via_bridge` and takes the
    # role. `peft_method` therefore replaces `lora_rank` as this fixture's on/off switch, and
    # the critic case pins `critic_mode="full"` so it stays outside the widened branch.
    # The optimizer/scheduler build is likewise one orbit helper (`_build_optimizer_and_scheduler`
    # in orbit/megatron/optim_build.py) rather than base's two inline megatron calls, so the two
    # optimizer patches collapse into one.
    def _make_args(self, peft_method="lora", role="actor", mode="bridge"):
        return Namespace(
            peft_method=peft_method,
            lora_rank=32,
            lora_adapter_path=None,
            multi_lora=False,
            use_critic=(role == "critic"),
            critic_mode="full",
            custom_model_provider_path=None,
            megatron_to_hf_mode=mode,
            moe_use_upcycling=False,
            debug_disable_optimizer=False,
            stream_optimizer_state_to_disk=False,
            load="/some/path",
            pretrained_checkpoint=None,
            # optimizer fields
            num_rollout=10,
            rollout_batch_size=8,
            n_samples_per_prompt=8,
            global_batch_size=32,
            lr_decay_iters=None,
            lr_wsd_decay_iters=None,
            lr_warmup_fraction=None,
            lr_warmup_iters=0,
            lr_warmup_init=0,
            lr=1e-5,
            min_lr=0,
            lr_decay_style="constant",
            start_weight_decay=0,
            end_weight_decay=0,
            weight_decay_incr_style="constant",
            use_checkpoint_opt_param_scheduler=False,
            override_opt_param_scheduler=False,
            lr_wsd_decay_style="linear",
            use_gloo_process_groups=False,
        )

    @patch(f"{_MODEL_MODULE}._build_optimizer_and_scheduler")
    @patch(f"{_MODEL_MODULE}._setup_peft_model_via_bridge")
    def test_lora_actor_bridge_routes_to_lora_setup(self, mock_lora_setup, mock_build_opt):
        from miles.backends.megatron_utils.model import setup_model_and_optimizer

        mock_lora_setup.return_value = [MagicMock()]
        mock_build_opt.return_value = (MagicMock(param_groups=[]), MagicMock())

        args = self._make_args(peft_method="lora", role="actor", mode="bridge")
        model, _, _ = setup_model_and_optimizer(args, role="actor")

        mock_lora_setup.assert_called_once_with(args, role="actor")

    @patch(f"{_MODEL_MODULE}._build_optimizer_and_scheduler")
    @patch(f"{_MODEL_MODULE}.get_model")
    @patch(f"{_MODEL_MODULE}.get_model_provider_func")
    @patch(f"{_MODEL_MODULE}._setup_peft_model_via_bridge")
    def test_lora_critic_skips_lora_setup(self, mock_lora_setup, mock_provider, mock_get_model, mock_build_opt):
        from miles.backends.megatron_utils.model import setup_model_and_optimizer

        mock_get_model.return_value = [MagicMock()]
        mock_build_opt.return_value = (MagicMock(param_groups=[]), MagicMock())

        args = self._make_args(peft_method="lora", role="critic", mode="bridge")
        setup_model_and_optimizer(args, role="critic")

        mock_lora_setup.assert_not_called()
        mock_get_model.assert_called_once()

    @patch(f"{_MODEL_MODULE}._build_optimizer_and_scheduler")
    @patch(f"{_MODEL_MODULE}.get_model")
    @patch(f"{_MODEL_MODULE}.get_model_provider_func")
    @patch(f"{_MODEL_MODULE}._setup_peft_model_via_bridge")
    def test_non_lora_skips_lora_setup(self, mock_lora_setup, mock_provider, mock_get_model, mock_build_opt):
        from miles.backends.megatron_utils.model import setup_model_and_optimizer

        mock_get_model.return_value = [MagicMock()]
        mock_build_opt.return_value = (MagicMock(param_groups=[]), MagicMock())

        args = self._make_args(peft_method="none", role="actor", mode="bridge")
        setup_model_and_optimizer(args, role="actor")

        mock_lora_setup.assert_not_called()
        mock_get_model.assert_called_once()

    @patch(f"{_MODEL_MODULE}._build_optimizer_and_scheduler")
    @patch(f"{_MODEL_MODULE}.get_model")
    @patch(f"{_MODEL_MODULE}._setup_peft_model_via_bridge")
    def test_lora_raw_mode_skips_bridge(self, mock_lora_setup, mock_get_model, mock_build_opt):
        from miles.backends.megatron_utils.model import setup_model_and_optimizer

        mock_get_model.return_value = [MagicMock()]
        mock_build_opt.return_value = (MagicMock(param_groups=[]), MagicMock())

        args = self._make_args(peft_method="lora", role="actor", mode="raw")
        setup_model_and_optimizer(args, role="actor")

        mock_lora_setup.assert_not_called()
        mock_get_model.assert_called_once()


# ---------------------------------------------------------------------------
# save — LoRA vs regular branch
# ---------------------------------------------------------------------------


class TestSaveLoRaBranch:
    @patch(f"{_MODEL_MODULE}.save_model_hashes")
    @patch(f"{_MODEL_MODULE}.enable_forward_pre_hook")
    @patch(f"{_MODEL_MODULE}.disable_forward_pre_hook")
    @patch(f"{_MODEL_MODULE}.should_disable_forward_pre_hook", return_value=False)
    @patch(f"{_MODEL_MODULE}.get_args")
    # orbit: base's LoRA-only save predicate/writer are orbit's PEFT (LoRA + OFT) pair,
    # `is_peft_model` / `save_checkpoint_with_peft`. Branch logic under test is unchanged.
    @patch(f"{_MODEL_MODULE}.save_checkpoint_with_peft")
    @patch(f"{_MODEL_MODULE}.is_peft_model", return_value=True)
    def test_lora_model_calls_lora_save(
        self, mock_is_lora, mock_save_lora, mock_get_args, mock_should, mock_disable, mock_enable, mock_save_hashes
    ):
        from miles.backends.megatron_utils.model import save

        model = [MagicMock()]
        save(42, model, MagicMock(), MagicMock())

        mock_save_lora.assert_called_once()

    @patch(f"{_MODEL_MODULE}.save_model_hashes")
    @patch(f"{_MODEL_MODULE}.enable_forward_pre_hook")
    @patch(f"{_MODEL_MODULE}.disable_forward_pre_hook")
    @patch(f"{_MODEL_MODULE}.should_disable_forward_pre_hook", return_value=False)
    @patch(f"{_MODEL_MODULE}.get_args")
    @patch(f"{_MODEL_MODULE}.save_checkpoint")
    @patch(f"{_MODEL_MODULE}.is_peft_model", return_value=False)
    def test_non_lora_model_calls_regular_save(
        self, mock_is_lora, mock_save_ckpt, mock_get_args, mock_should, mock_disable, mock_enable, mock_save_hashes
    ):
        from miles.backends.megatron_utils.model import save

        model = [MagicMock()]
        save(42, model, MagicMock(), MagicMock())

        mock_save_ckpt.assert_called_once()
