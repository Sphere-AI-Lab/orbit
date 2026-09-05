import argparse
import logging
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from orbit.backends.sglang_utils.arguments import add_sglang_arguments, collect_eval_sglang_overrides
from orbit.backends.sglang_utils.arguments import validate_args as validate_sglang_args
from orbit.utils.arguments import (
    _finalize_train_offload_args,
    _maybe_apply_dumper_overrides,
    _resolve_ft_components,
    _resolve_rollout_functions,
    _validate_model_response_trace_args,
    _validate_opd_dagger_args,
    _validate_opd_sglang_scoring_args,
    _validate_opd_task_reward_args,
    _validate_rematerialize_param_from_master_weight,
    get_orbit_extra_args_provider,
    hf_validate_args,
    orbit_validate_args,
    resolve_rollout_function_paths,
    resolve_train_offload_mode,
    validate_async_off_policy_correction,
    validate_skip_actor_forward_only,
)
from orbit.utils.misc import function_registry

PATH_ARGS = ["--rollout-function-path", "--custom-generate-function-path"]
REQUIRED_ARGS = ["--rollout-batch-size", "64"]


def _train_offload_args(**overrides) -> SimpleNamespace:
    defaults = dict(
        offload_train=True,
        offload_train_adapter=False,
        offload_train_grad_buffers=False,
        offload_train_optimizer=False,
        offload_train_async=False,
        offload_train_frozen_base_mode="auto",
        offload_train_target="cpu",
        peft_method="oft",
        peft_variant="standard",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_train_offload_auto_preserves_orbit_tms_default():
    assert resolve_train_offload_mode(_train_offload_args()) == "tms"


@pytest.mark.parametrize(
    "flag",
    [
        "offload_train_adapter",
        "offload_train_grad_buffers",
        "offload_train_optimizer",
        "offload_train_async",
    ],
)
def test_train_offload_auto_selects_flat_backend_for_orbit_controls(flag):
    assert resolve_train_offload_mode(_train_offload_args(**{flag: True})) == "flat"


def test_explicit_tms_rejects_orbit_only_controls():
    args = _train_offload_args(
        offload_train_frozen_base_mode="tms",
        offload_train_async=True,
    )

    with pytest.raises(ValueError, match="flat"):
        _finalize_train_offload_args(args)


def test_flat_offload_rejects_disk_and_dsv4():
    with pytest.raises(ValueError, match="CPU"):
        _finalize_train_offload_args(
            _train_offload_args(
                offload_train_frozen_base_mode="flat",
                offload_train_target="disk",
            )
        )

    with pytest.raises(NotImplementedError, match="DSV4"):
        _finalize_train_offload_args(
            _train_offload_args(
                offload_train_frozen_base_mode="flat",
                peft_variant="dsv4",
            )
        )


class TestValidateModelResponseTraceArgs:
    def test_trace_output_requires_an_explicit_positive_cap(self) -> None:
        args = SimpleNamespace(
            save_model_response_trace_dir="/tmp/traces",
            model_response_trace_max_samples_per_step=None,
        )

        with pytest.raises(ValueError, match="--model-response-trace-max-samples-per-step"):
            _validate_model_response_trace_args(args)

    def test_trace_output_accepts_an_explicit_positive_cap(self) -> None:
        args = SimpleNamespace(
            save_model_response_trace_dir="/tmp/traces",
            model_response_trace_max_samples_per_step=16,
        )

        _validate_model_response_trace_args(args)


def _hf_validation_args(*, untie_embeddings: bool) -> SimpleNamespace:
    return SimpleNamespace(
        model_name="qwen3-vl",
        context_parallel_size=1,
        untie_embeddings_and_output_weights=untie_embeddings,
    )


def test_hf_validate_args_uses_outer_tying_value_for_qwen3_vl_moe() -> None:
    text_config = SimpleNamespace(model_type="qwen3_vl_moe_text", tie_word_embeddings=True)
    hf_config = SimpleNamespace(model_type="qwen3_vl_moe", text_config=text_config, tie_word_embeddings=False)

    hf_validate_args(_hf_validation_args(untie_embeddings=True), hf_config)

    assert text_config.tie_word_embeddings is False


def test_hf_validate_args_rejects_qwen3_vl_moe_outer_tying_mismatch() -> None:
    text_config = SimpleNamespace(model_type="qwen3_vl_moe_text", tie_word_embeddings=False)
    hf_config = SimpleNamespace(model_type="qwen3_vl_moe", text_config=text_config, tie_word_embeddings=True)

    with pytest.raises(AssertionError, match="tie_word_embeddings in hf config True"):
        hf_validate_args(_hf_validation_args(untie_embeddings=True), hf_config)


def test_hf_validate_args_preserves_text_tying_for_other_composite_models() -> None:
    text_config = SimpleNamespace(model_type="synthetic_vlm_text", tie_word_embeddings=False)
    hf_config = SimpleNamespace(model_type="synthetic_vlm", text_config=text_config, tie_word_embeddings=True)

    hf_validate_args(_hf_validation_args(untie_embeddings=True), hf_config)

    assert text_config.tie_word_embeddings is False


def make_class_with_add_arguments():
    class MyFn:
        @classmethod
        def add_arguments(cls, parser):
            parser.add_argument("--my-custom-arg", type=int, default=42)

    return MyFn


def make_function_with_add_arguments():
    def my_fn():
        pass

    my_fn.add_arguments = lambda parser: parser.add_argument("--my-custom-arg", type=int, default=42)
    return my_fn


def make_function_without_add_arguments():
    def my_fn():
        pass

    return my_fn


@pytest.mark.parametrize("path_arg", PATH_ARGS)
class TestAddArgumentsSupport:

    @pytest.mark.parametrize("fn_factory", [make_class_with_add_arguments, make_function_with_add_arguments])
    def test_add_arguments_is_called_and_arg_is_parsed(self, path_arg, fn_factory):
        fn = fn_factory()
        with function_registry.temporary("test:fn", fn), patch.object(
            sys, "argv", ["test", path_arg, "test:fn", "--my-custom-arg", "100"] + REQUIRED_ARGS
        ):
            parser = argparse.ArgumentParser()
            get_orbit_extra_args_provider()(parser)
            args, _ = parser.parse_known_args()
            assert args.my_custom_arg == 100

    def test_skips_function_without_add_arguments(self, path_arg):
        fn = make_function_without_add_arguments()
        with function_registry.temporary("test:fn", fn), patch.object(
            sys, "argv", ["test", path_arg, "test:fn"] + REQUIRED_ARGS
        ):
            parser = argparse.ArgumentParser()
            get_orbit_extra_args_provider()(parser)


def test_lora_a_init_cli_matches_bridge_supported_methods():
    parser = argparse.ArgumentParser()
    get_orbit_extra_args_provider()(parser)

    args, _ = parser.parse_known_args(["--rollout-batch-size", "1", "--lora-a-init-method", "uniform"])
    assert args.lora_a_init_method == "uniform"

    with pytest.raises(SystemExit):
        parser.parse_known_args(["--rollout-batch-size", "1", "--lora-a-init-method", "normal"])


class TestMaybeApplyDumperOverrides:
    def _make_args(
        self,
        *,
        dumper_enable: bool = False,
        use_fault_tolerance: bool = False,
        router_disable_health_check: bool = False,
        rollout_health_check_interval: float = 30.0,
        start_rollout_id: int | None = None,
        num_rollout: int = 10,
        eval_interval: int | None = 5,
        save: str | None = "/tmp/checkpoint",
        save_interval: int | None = 5,
        save_retain_interval: int | None = 10,
    ) -> SimpleNamespace:
        return SimpleNamespace(
            dumper_enable=dumper_enable,
            use_fault_tolerance=use_fault_tolerance,
            router_disable_health_check=router_disable_health_check,
            rollout_health_check_interval=rollout_health_check_interval,
            start_rollout_id=start_rollout_id,
            num_rollout=num_rollout,
            eval_interval=eval_interval,
            save=save,
            save_interval=save_interval,
            save_retain_interval=save_retain_interval,
        )

    def test_noop_when_dumper_disabled(self) -> None:
        args = self._make_args(
            dumper_enable=False,
            use_fault_tolerance=True,
            rollout_health_check_interval=30.0,
        )
        _maybe_apply_dumper_overrides(args)

        assert args.use_fault_tolerance is True
        assert args.router_disable_health_check is False
        assert args.rollout_health_check_interval == 30.0
        assert args.num_rollout == 10
        assert args.eval_interval == 5
        assert args.save == "/tmp/checkpoint"
        assert args.save_interval == 5
        assert args.save_retain_interval == 10

    def test_disables_all_heartbeats(self) -> None:
        args = self._make_args(
            dumper_enable=True,
            use_fault_tolerance=True,
            rollout_health_check_interval=30.0,
        )
        _maybe_apply_dumper_overrides(args)

        assert args.use_fault_tolerance is False
        assert args.router_disable_health_check is True
        assert args.rollout_health_check_interval == 1e18

    def test_forces_single_rollout(self) -> None:
        args = self._make_args(dumper_enable=True, num_rollout=100)
        _maybe_apply_dumper_overrides(args)

        assert args.start_rollout_id == 0
        assert args.num_rollout == 1
        assert args.eval_interval is None
        assert args.save is None
        assert args.save_interval is None
        assert args.save_retain_interval is None

    def test_respects_start_rollout_id(self) -> None:
        args = self._make_args(dumper_enable=True, start_rollout_id=5, num_rollout=100)
        _maybe_apply_dumper_overrides(args)

        assert args.num_rollout == 6


def test_fully_async_eval_resolves_to_the_producer_itself():
    """Only the producer's own instance pauses on eval, and RolloutManager reuses one
    instance only when both paths match."""
    path = "orbit.rollout.fully_async_rollout.FullyAsyncRolloutFn"
    default = SimpleNamespace(rollout_function_path=None, eval_function_path=None, fully_async=True)
    assert resolve_rollout_function_paths(default) == (path, path)

    override = SimpleNamespace(rollout_function_path=None, eval_function_path="pkg.CustomEval", fully_async=True)
    assert resolve_rollout_function_paths(override) == (path, "pkg.CustomEval")


FAIL_CLOSED_BUFFER = "examples.fully_async.fail_closed_data_buffer.FailClosedDataBuffer"


def _fully_async_args(**overrides):
    """A fully-async arg namespace that reaches the generation-window derivation."""
    args = SimpleNamespace(
        fully_async=True,
        multi_lora=False,
        rollout_function_path=None,
        eval_function_path=None,
        colocate=False,
        partial_rollout=False,
        pause_generation_mode="retract",
        recompute_logprobs_via_prefill=False,
        rollout_all_samples_process_path=None,
        eval_num_gpus=0,
        rollout_batch_size=64,
        n_samples_per_prompt=8,
        fully_async_prefetch_batches=1,
        async_max_concurrent_samples=None,
        max_weight_staleness=None,
        fully_async_max_completed_queue_groups=2048,
        async_data_buffer_capacity_factor=2.0,
        custom_async_data_buffer_path=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_fully_async_rejects_abort_pause_mode():
    """Generation is always in flight, so aborting on every weight update would kill it."""
    args = _fully_async_args(pause_generation_mode="abort")

    with pytest.raises(AssertionError, match="pause-generation-mode abort"):
        _resolve_rollout_functions(args)

    args.pause_generation_mode = "retract"
    _resolve_rollout_functions(args)


def test_fully_async_prefetch_batches_sizes_the_generation_window():
    """The knob was inert after the upstream rewrite: nothing read it, so every recipe
    asking for depth 2 silently ran at depth 1."""
    args = _fully_async_args(fully_async_prefetch_batches=2)
    _resolve_rollout_functions(args)
    assert args.async_max_concurrent_samples == 64 * 2 * 8

    # depth 1 must reproduce upstream's own default bound (rollout_batch_size groups).
    default = _fully_async_args()
    _resolve_rollout_functions(default)
    assert default.async_max_concurrent_samples == 64 * 8


def test_fully_async_window_knobs_are_mutually_exclusive():
    args = _fully_async_args(fully_async_prefetch_batches=2, async_max_concurrent_samples=99)
    with pytest.raises(ValueError, match="pass only one"):
        _resolve_rollout_functions(args)

    # An explicit absolute bound alone is honored untouched.
    explicit = _fully_async_args(async_max_concurrent_samples=99)
    _resolve_rollout_functions(explicit)
    assert explicit.async_max_concurrent_samples == 99


def test_fully_async_prefetch_batches_must_be_at_least_one():
    """A recipe expanding an unset env var to 0 would derive a zero window and
    silently fall back to a single group."""
    for depth in (0, -1):
        with pytest.raises(ValueError, match=">= 1"):
            _resolve_rollout_functions(_fully_async_args(fully_async_prefetch_batches=depth))


def test_fully_async_defaults_to_the_fail_closed_data_buffer():
    """DefaultDataBuffer admits groups whose staleness it cannot observe — the fail-open
    mode that cost this fork a 15-hour run."""
    args = _fully_async_args(max_weight_staleness=2)
    _resolve_rollout_functions(args)
    assert args.custom_async_data_buffer_path == FAIL_CLOSED_BUFFER

    explicit = _fully_async_args(custom_async_data_buffer_path="pkg.CustomBuffer")
    _resolve_rollout_functions(explicit)
    assert explicit.custom_async_data_buffer_path == "pkg.CustomBuffer"


def test_recompute_logprobs_via_prefill_flag_is_parsed():
    parser = argparse.ArgumentParser()
    get_orbit_extra_args_provider()(parser)

    args = parser.parse_args(["--recompute-logprobs-via-prefill"] + REQUIRED_ARGS)

    assert args.recompute_logprobs_via_prefill is True


def test_sglang_mm_exact_scoring_suffix_is_opt_in():
    parser = argparse.ArgumentParser()
    get_orbit_extra_args_provider()(parser)

    default_args = parser.parse_args(REQUIRED_ARGS)
    enabled_args = parser.parse_args(["--sglang-mm-exact-scoring-suffix"] + REQUIRED_ARGS)

    assert default_args.sglang_mm_exact_scoring_suffix is False
    assert enabled_args.sglang_mm_exact_scoring_suffix is True


def test_opd_dagger_defaults_are_disabled():
    parser = argparse.ArgumentParser()
    get_orbit_extra_args_provider()(parser)

    args = parser.parse_args(REQUIRED_ARGS)

    assert args.opd_dagger_top_k == 0
    assert args.opd_dagger_coef == 0.0
    assert args.opd_dagger_loss == "cross_entropy"
    assert args.opd_log_task_reward is False
    assert args.opd_optimize_task_reward is False
    assert args.opd_task_reward_coef == 1.0


def test_opd_task_reward_logging_argument_is_parsed():
    parser = argparse.ArgumentParser()
    get_orbit_extra_args_provider()(parser)

    args = parser.parse_args(["--opd-log-task-reward"] + REQUIRED_ARGS)

    assert args.opd_log_task_reward is True


def _opd_task_reward_args(**overrides):
    values = {
        "opd_log_task_reward": True,
        "opd_optimize_task_reward": False,
        "opd_task_reward_coef": 1.0,
        "use_opd": True,
        "opd_type": "sglang",
        "rm_type": "deepscaler",
        "custom_rm_path": "orbit.rollout.on_policy_distillation.reward_func",
        "custom_reward_post_process_path": "orbit.rollout.on_policy_distillation.post_process_rewards",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_opd_task_reward_optimization_arguments_are_parsed():
    parser = argparse.ArgumentParser()
    get_orbit_extra_args_provider()(parser)

    args = parser.parse_args(
        ["--opd-log-task-reward", "--opd-optimize-task-reward", "--opd-task-reward-coef", "0.5"] + REQUIRED_ARGS
    )

    assert args.opd_optimize_task_reward is True
    assert args.opd_task_reward_coef == 0.5


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (_opd_task_reward_args(use_opd=False), r"requires --use-opd"),
        (_opd_task_reward_args(opd_type="megatron"), r"supported only with --opd-type=sglang"),
        (_opd_task_reward_args(custom_rm_path=None), r"requires --custom-rm-path"),
        (
            _opd_task_reward_args(custom_reward_post_process_path=None),
            r"requires --custom-reward-post-process-path",
        ),
        (_opd_task_reward_args(rm_type=None), r"requires a built-in --rm-type"),
        (_opd_task_reward_args(rm_type="remote_rm"), r"does not support remote_rm"),
        (
            _opd_task_reward_args(rm_type="boxed_remote_rm"),
            r"does not support remote_rm",
        ),
        (
            SimpleNamespace(opd_log_task_reward=False, opd_optimize_task_reward=True),
            r"requires --opd-log-task-reward",
        ),
        (
            SimpleNamespace(opd_log_task_reward=False, opd_task_reward_coef=-1.0),
            r"must be finite and non-negative",
        ),
    ],
)
def test_opd_task_reward_logging_rejects_incomplete_configuration(args, error):
    with pytest.raises(ValueError, match=error):
        _validate_opd_task_reward_args(args)


def test_opd_task_reward_logging_accepts_builtin_verifier():
    _validate_opd_task_reward_args(_opd_task_reward_args())


def test_opd_task_reward_optimization_accepts_builtin_verifier():
    _validate_opd_task_reward_args(_opd_task_reward_args(opd_optimize_task_reward=True, opd_task_reward_coef=0.5))


def _opd_sglang_scoring_args(**overrides) -> SimpleNamespace:
    values = {
        "use_opd": True,
        "opd_type": "sglang",
        "rm_url": "http://teacher:30000/generate",
        "custom_rm_path": "orbit.rollout.on_policy_distillation.reward_func",
        "custom_reward_post_process_path": "orbit.rollout.on_policy_distillation.post_process_rewards",
        "group_rm": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_opd_sglang_scoring_accepts_complete_production_wiring():
    _validate_opd_sglang_scoring_args(_opd_sglang_scoring_args())


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"rm_url": None}, r"requires --rm-url"),
        ({"rm_url": "teacher:30000/generate"}, r"valid HTTP\(S\)"),
        ({"custom_rm_path": None}, r"requires --custom-rm-path"),
        ({"custom_reward_post_process_path": None}, r"requires --custom-reward-post-process-path"),
        ({"group_rm": True}, r"does not support --group-rm"),
    ],
)
def test_opd_sglang_scoring_rejects_incomplete_or_incompatible_wiring(overrides, error):
    with pytest.raises(ValueError, match=error):
        _validate_opd_sglang_scoring_args(_opd_sglang_scoring_args(**overrides))


def test_opd_sglang_scoring_preserves_custom_hook_extension_points():
    _validate_opd_sglang_scoring_args(
        _opd_sglang_scoring_args(
            custom_rm_path="custom.reward_func",
            custom_reward_post_process_path="custom.post_process_rewards",
        )
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_opd": False},
        {"opd_type": "megatron"},
    ],
)
def test_opd_sglang_scoring_validation_does_not_change_other_modes(overrides):
    _validate_opd_sglang_scoring_args(
        _opd_sglang_scoring_args(
            rm_url=None,
            custom_rm_path=None,
            custom_reward_post_process_path=None,
            group_rm=True,
            **overrides,
        )
    )


def test_opd_dagger_arguments_are_parsed():
    parser = argparse.ArgumentParser()
    get_orbit_extra_args_provider()(parser)

    args = parser.parse_args(
        [
            "--opd-dagger-top-k",
            "2",
            "--opd-dagger-coef",
            "0.5",
            "--opd-dagger-loss",
            "explicit_cross_entropy",
        ]
        + REQUIRED_ARGS
    )

    assert args.opd_dagger_top_k == 2
    assert args.opd_dagger_coef == 0.5
    assert args.opd_dagger_loss == "explicit_cross_entropy"


def _opd_dagger_args(**overrides) -> SimpleNamespace:
    values = {
        "use_opd": True,
        "opd_type": "sglang",
        "opd_kl_coef": 1.0,
        "opd_log_prob_top_k": 0,
        "opd_dagger_top_k": 2,
        "opd_dagger_coef": 1.0,
        "opd_dagger_loss": "explicit_cross_entropy",
        "vocab_size": 8,
        "padded_vocab_size": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_opd_dagger_accepts_explicit_teacher_sparse_target_contract():
    _validate_opd_dagger_args(_opd_dagger_args())


def test_opd_dagger_accepts_complete_topk_rest_cross_entropy():
    _validate_opd_dagger_args(_opd_dagger_args(opd_dagger_loss="cross_entropy"))


def test_opd_dagger_accepts_sampled_rkld_hybrid_configuration():
    args = _opd_dagger_args(
        opd_kl_coef=1.0,
        opd_dagger_coef=1.0,
        opd_dagger_loss="cross_entropy",
    )

    _validate_opd_dagger_args(args)

    assert args.opd_log_prob_top_k == 0
    assert args.opd_kl_coef > 0
    assert args.opd_dagger_coef > 0


def test_opd_dagger_top_k_can_collect_targets_with_zero_loss_coefficient():
    _validate_opd_dagger_args(_opd_dagger_args(opd_dagger_coef=0.0, opd_dagger_loss="cross_entropy"))


def test_opd_dagger_accepts_megatron_padded_vocab_and_reports_masking(caplog):
    with caplog.at_level("INFO"):
        _validate_opd_dagger_args(_opd_dagger_args(padded_vocab_size=10))

    assert "exclude 2 Megatron dummy vocabulary logits" in caplog.text


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"opd_dagger_top_k": -1}, r"must be non-negative"),
        ({"opd_dagger_coef": -1.0}, r"must be non-negative"),
        ({"opd_dagger_loss": "invalid"}, r"Unsupported --opd-dagger-loss"),
        ({"opd_dagger_top_k": 0}, r"requires --opd-dagger-top-k > 0"),
        ({"opd_dagger_top_k": 9}, r"positive vocab_size >= top-k"),
        ({"padded_vocab_size": 7}, r"cannot be smaller than vocab_size"),
        ({"use_opd": False}, r"requires --use-opd"),
        ({"opd_type": "megatron"}, r"only with --opd-type=sglang"),
        ({"opd_log_prob_top_k": 2}, r"cannot be combined with legacy"),
    ],
)
def test_opd_dagger_rejects_incompatible_configuration(overrides, error):
    with pytest.raises(ValueError, match=error):
        _validate_opd_dagger_args(_opd_dagger_args(**overrides))


def test_sglang_parallel_sizes_keep_server_args_destinations():
    parser = add_sglang_arguments(argparse.ArgumentParser())
    args = parser.parse_args(
        [
            "--sglang-tp-size",
            "6",
            "--sglang-data-parallel-size",
            "2",
            "--sglang-pipeline-parallel-size",
            "3",
            "--sglang-expert-parallel-size",
            "4",
            "--sglang-attention-context-parallel-size",
            "5",
        ]
    )
    args.rollout_num_gpus_per_engine = 8
    args.true_on_policy_mode = False
    args.sglang_enable_dp_attention = True
    args.use_session_server = False

    validate_sglang_args(args)

    assert args.sglang_tp_size == 8
    assert args.sglang_dp_size == 2
    assert args.sglang_pp_size == 3
    assert args.sglang_ep_size == 4
    assert args.sglang_attn_cp_size == 5


class TestEvalSglangOverrides:
    """Unset means "inherit --sglang-*", so an unset flag must leave no attribute at all."""

    def _parse(self, argv):
        return add_sglang_arguments(argparse.ArgumentParser()).parse_args(argv)

    def test_unset_flags_produce_no_overrides(self):
        args = self._parse(["--sglang-mem-fraction-static", "0.7"])

        assert collect_eval_sglang_overrides(args) == {}
        assert not hasattr(args, "eval_sglang_mem_fraction_static")

    def test_set_flag_becomes_an_override_without_touching_the_base_family(self):
        args = self._parse(["--sglang-mem-fraction-static", "0.7", "--eval-sglang-mem-fraction-static", "0.9"])

        assert collect_eval_sglang_overrides(args) == {"mem_fraction_static": 0.9}
        assert args.sglang_mem_fraction_static == 0.7

    def test_boolean_can_be_turned_back_off(self):
        args = self._parse(["--sglang-enable-dp-attention", "--no-eval-sglang-enable-dp-attention"])

        assert args.sglang_enable_dp_attention is True
        assert collect_eval_sglang_overrides(args) == {"enable_dp_attention": False}

    def test_parallel_sizes_keep_server_args_destinations(self):
        args = self._parse(["--eval-sglang-data-parallel-size", "2", "--eval-sglang-expert-parallel-size", "4"])

        assert collect_eval_sglang_overrides(args) == {"dp_size": 2, "ep_size": 4}

    def test_tp_size_is_not_exposed(self):
        """A second TP knob could move tp_size off the bundles --eval-num-gpus-per-engine placed."""
        with pytest.raises(SystemExit):
            self._parse(["--eval-sglang-tp-size", "2"])


def test_custom_megatron_post_save_hook_path_is_parsed():
    parser = argparse.ArgumentParser()
    get_orbit_extra_args_provider()(parser)

    args = parser.parse_args(["--custom-megatron-post-save-hook-path", "pkg.module.hook"] + REQUIRED_ARGS)

    assert args.custom_megatron_post_save_hook_path == "pkg.module.hook"


def test_custom_megatron_post_save_hook_path_requires_save():
    parser = argparse.ArgumentParser()
    get_orbit_extra_args_provider()(parser)
    args = parser.parse_args(
        ["--custom-megatron-post-save-hook-path", "pkg.module.hook", "--num-rollout", "1"] + REQUIRED_ARGS
    )

    with pytest.raises(
        AssertionError,
        match="'--save' is required when custom_megatron_post_save_hook_path is set.",
    ):
        orbit_validate_args(args)


class TestBridgeResumeRolloutId:
    @staticmethod
    def _parse(load, extra=None):
        parser = argparse.ArgumentParser()
        get_orbit_extra_args_provider()(parser)
        return parser.parse_args(
            [
                "--megatron-to-hf-mode",
                "bridge",
                "--load",
                str(load),
                "--num-rollout",
                "1",
            ]
            + (extra or [])
            + REQUIRED_ARGS
        )

    @staticmethod
    def _checkpoint(tmp_path, tracker):
        checkpoint = tmp_path / "checkpoint"
        checkpoint.mkdir()
        (checkpoint / "latest_checkpointed_iteration.txt").write_text(tracker)
        return checkpoint

    @pytest.mark.parametrize(
        ("override", "value"),
        [
            (None, None),
            ("ckpt_step", 7),
            ("lora_adapter_path", "/adapter/iter_0000007"),
        ],
        ids=["latest", "ckpt-step", "lora-adapter"],
    )
    def test_numeric_tracker_defers_to_loaded_iteration(self, tmp_path, override, value):
        args = self._parse(self._checkpoint(tmp_path, "17"))
        if override is not None:
            setattr(args, override, value)

        orbit_validate_args(args)

        assert args.start_rollout_id is None

    @pytest.mark.parametrize("tracker", [None, "release"], ids=["hf-base", "release-base"])
    def test_lora_resume_defers_to_adapter_iteration(self, tmp_path, tracker):
        adapter = tmp_path / "adapter"
        adapter.mkdir()
        if tracker is None:
            checkpoint = tmp_path / "hf"
            checkpoint.mkdir()
            base_args = ["--hf-checkpoint", str(checkpoint)]
        else:
            checkpoint = self._checkpoint(tmp_path, tracker)
            base_args = []
        args = self._parse(
            checkpoint,
            base_args + ["--lora-adapter-path", str(adapter)],
        )

        orbit_validate_args(args)

        assert args.start_rollout_id is None

    def test_release_tracker_starts_at_zero(self, tmp_path):
        args = self._parse(self._checkpoint(tmp_path, "release"))

        orbit_validate_args(args)

        assert args.start_rollout_id == 0

    def test_explicit_rollout_id_is_preserved(self, tmp_path):
        args = self._parse(
            self._checkpoint(tmp_path, "17"),
            ["--start-rollout-id", "9"],
        )

        orbit_validate_args(args)

        assert args.start_rollout_id == 9

    def test_missing_checkpoint_falls_back_to_hf_at_zero(self, tmp_path):
        hf_checkpoint = tmp_path / "hf"
        hf_checkpoint.mkdir()
        args = self._parse(
            tmp_path / "missing",
            ["--hf-checkpoint", str(hf_checkpoint)],
        )

        orbit_validate_args(args)

        assert args.load == str(hf_checkpoint)
        assert args.start_rollout_id == 0


def test_dynamic_global_batch_size_requires_dynamic_batch_size():
    parser = argparse.ArgumentParser()
    get_orbit_extra_args_provider()(parser)
    args = parser.parse_args(["--use-dynamic-global-batch-size", "--num-rollout", "1"] + REQUIRED_ARGS)

    with pytest.raises(AssertionError, match="requires --use-dynamic-batch-size"):
        orbit_validate_args(args)


class TestCriticSaveDerivation:
    def _validate(self, extra):
        parser = argparse.ArgumentParser()
        get_orbit_extra_args_provider()(parser)
        args = parser.parse_args(extra + ["--num-rollout", "1"] + REQUIRED_ARGS)
        orbit_validate_args(args)
        return args

    def test_derives_sibling_dir_from_save(self):
        args = self._validate(["--advantage-estimator", "ppo", "--save", "/ckpts/run1"])
        assert args.critic_save == "/ckpts/run1_critic"

    def test_trailing_slash_is_stripped(self):
        args = self._validate(["--advantage-estimator", "ppo", "--save", "/ckpts/run1/"])
        assert args.critic_save == "/ckpts/run1_critic"

    def test_explicit_critic_save_is_respected(self):
        args = self._validate(
            ["--advantage-estimator", "ppo", "--save", "/ckpts/run1", "--critic-save", "/elsewhere/critic"]
        )
        assert args.critic_save == "/elsewhere/critic"

    def test_stays_none_without_save(self):
        args = self._validate(["--advantage-estimator", "ppo"])
        assert args.critic_save is None


class TestSessionServerV2Validation:
    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        get_orbit_extra_args_provider()(parser)
        return parser.parse_args(extra + ["--num-rollout", "1"] + REQUIRED_ARGS)

    @pytest.mark.parametrize(
        ("extra", "flag"),
        [
            (["--group-rm"], "--group-rm"),
            (["--partial-rollout"], "--partial-rollout"),
            (
                ["--true-on-policy-mode", "--recompute-logprobs-via-prefill"],
                "--recompute-logprobs-via-prefill",
            ),
        ],
    )
    def test_rejects_unsupported_list_consumers(self, extra, flag):
        args = self._parse(["--use-session-server", "v2", *extra])

        with pytest.raises(ValueError) as exc_info:
            orbit_validate_args(args)

        assert str(exc_info.value) == (f"--use-session-server v2 does not support {flag}; v2 returns list[Sample]")


class TestSessionMessageMatcherArgument:
    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        get_orbit_extra_args_provider()(parser)
        return parser.parse_args(extra + ["--num-rollout", "1"] + REQUIRED_ARGS)

    def test_defaults_to_strict(self):
        assert self._parse([]).session_message_matcher == "strict"

    @pytest.mark.parametrize(
        "selector",
        [
            "strict",
            "loose_tool_call",
            "role_content_only",
            "not_installed.matchers.same_message",
        ],
    )
    def test_preserves_selector_without_importing(self, selector):
        args = self._parse(["--session-message-matcher", selector])

        assert args.session_message_matcher == selector


class TestSessionServerPauseGenerationMode:
    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        get_orbit_extra_args_provider()(parser)
        return parser.parse_args(extra + ["--num-rollout", "1"] + REQUIRED_ARGS)

    def test_session_server_rejects_abort(self):
        args = self._parse(["--use-session-server", "--pause-generation-mode", "abort"])

        with pytest.raises(
            AssertionError, match="--use-session-server is incompatible with --pause-generation-mode=abort"
        ):
            orbit_validate_args(args)

    def test_abort_without_session_server_passes(self):
        orbit_validate_args(self._parse(["--pause-generation-mode", "abort"]))

    @pytest.mark.parametrize("mode", ["retract", "in_place"])
    def test_session_server_accepts_non_abort_modes(self, mode):
        orbit_validate_args(self._parse(["--use-session-server", "--pause-generation-mode", mode]))


class TestTitoFixedTemplateConfiguration:
    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        get_orbit_extra_args_provider()(parser)
        return parser.parse_args(extra + ["--num-rollout", "1"] + REQUIRED_ARGS)

    def test_removed_role_flag_is_rejected(self):
        with pytest.raises(SystemExit):
            self._parse(["--tito-allowed-append-roles", "tool"])

    @pytest.mark.parametrize(
        ("extra", "expect_warning"),
        [
            (["--use-session-server"], True),
            ([], False),
            (["--use-session-server", "--tito-model", "qwen3"], False),
        ],
    )
    def test_warns_only_for_default_model_session(self, caplog, extra, expect_warning):
        args = self._parse(extra)

        with caplog.at_level(logging.WARNING, logger="orbit.utils.arguments"):
            orbit_validate_args(args)

        target_records = [
            record
            for record in caplog.records
            if record.getMessage().startswith("--tito-model=default uses a best-effort four-role append surface.")
        ]
        assert len(target_records) == int(expect_warning)

    def test_named_family_requires_session_server(self):
        args = self._parse(["--tito-model", "qwen3"])
        with pytest.raises(ValueError, match="--tito-model=qwen3 requires --use-session-server"):
            orbit_validate_args(args)

    def test_named_family_resolves_registered_template_and_kwargs(self):
        args = self._parse(["--use-session-server", "--tito-model", "qwen3"])
        orbit_validate_args(args)
        assert args.chat_template_path.endswith("/qwen3_fixed.jinja")
        assert args.apply_chat_template_kwargs == {"clear_thinking": False}

    def test_named_family_rejects_custom_template(self):
        args = self._parse(
            [
                "--use-session-server",
                "--tito-model",
                "qwen3",
                "--chat-template-path",
                "/tmp/custom.jinja",
            ]
        )
        with pytest.raises(ValueError, match="cannot override the template registered"):
            orbit_validate_args(args)

    def test_named_family_rejects_conflicting_registered_kwarg(self):
        args = self._parse(
            [
                "--use-session-server",
                "--tito-model",
                "qwen3",
                "--apply-chat-template-kwargs",
                '{"clear_thinking": true}',
            ]
        )
        with pytest.raises(ValueError, match="clear_thinking=True conflicts"):
            orbit_validate_args(args)

    def test_named_family_accepts_same_registered_and_additional_kwargs(self):
        args = self._parse(
            [
                "--use-session-server",
                "--tito-model",
                "qwen3",
                "--apply-chat-template-kwargs",
                '{"clear_thinking": false, "enable_thinking": true}',
            ]
        )
        orbit_validate_args(args)
        assert args.apply_chat_template_kwargs == {
            "clear_thinking": False,
            "enable_thinking": True,
        }


def test_bridge_mode_rejects_critic(tmp_path):
    parser = argparse.ArgumentParser()
    get_orbit_extra_args_provider()(parser)
    args = parser.parse_args(
        [
            "--advantage-estimator",
            "ppo",
            "--megatron-to-hf-mode",
            "bridge",
            "--hf-checkpoint",
            str(tmp_path),
            "--num-rollout",
            "1",
        ]
        + REQUIRED_ARGS
    )

    with pytest.raises(
        AssertionError,
        match="Critic models are not supported with --megatron-to-hf-mode bridge",
    ):
        orbit_validate_args(args)


def test_critic_rejects_experimental_ft_trainer(tmp_path, monkeypatch):
    monkeypatch.setenv("ORBIT_EXPERIMENTAL_FT_TRAINER", "1")
    parser = argparse.ArgumentParser()
    get_orbit_extra_args_provider()(parser)
    args = parser.parse_args(
        ["--advantage-estimator", "ppo", "--hf-checkpoint", str(tmp_path), "--num-rollout", "1"] + REQUIRED_ARGS
    )

    with pytest.raises(AssertionError, match="ORBIT_EXPERIMENTAL_FT_TRAINER"):
        orbit_validate_args(args)


def test_critic_rejects_reward_level_kl(tmp_path):
    parser = argparse.ArgumentParser()
    get_orbit_extra_args_provider()(parser)
    args = parser.parse_args(
        [
            "--advantage-estimator",
            "ppo",
            "--kl-coef",
            "0.05",
            "--ref-load",
            str(tmp_path),
            "--hf-checkpoint",
            str(tmp_path),
            "--num-rollout",
            "1",
        ]
        + REQUIRED_ARGS
    )

    with pytest.raises(AssertionError, match="does not support reward-level KL"):
        orbit_validate_args(args)


class TestMultiLoRAValidation:
    def _parse(self, extra):
        parser = argparse.ArgumentParser()
        get_orbit_extra_args_provider()(parser)
        return parser.parse_args(
            [
                "--multi-lora-n-adapters",
                "2",
                "--lora-rank",
                "8",
                "--target-modules",
                "linear_qkv",
                "--num-rollout",
                "1",
            ]
            + extra
            + REQUIRED_ARGS
        )

    def test_rejects_multiple_tokenizer_workers(self):
        # Each sglang tokenizer worker holds its own LoRA registry, so per-step
        # upserts fail non-deterministically; fail at launch, not first push.
        args = self._parse(["--sglang-tokenizer-worker-num", "2"])

        with pytest.raises(AssertionError, match="sglang-tokenizer-worker-num 1"):
            orbit_validate_args(args)

    def test_accepts_default_single_tokenizer_worker(self):
        args = self._parse([])

        orbit_validate_args(args)

        assert args.multi_lora is True

    def test_defaults_rollout_fn_and_data_source_to_multi_lora(self):
        args = self._parse([])

        orbit_validate_args(args)

        assert args.rollout_function_path == "orbit.rollout.multi_lora.async_rollout.generate_rollout_multi_lora"
        assert args.data_source_path == "orbit.rollout.multi_lora.data_source.MultiLoRAAsyncDataSource"
        assert args.rollout_global_dataset is True

    def test_keeps_user_supplied_rollout_fn_and_data_source(self):
        args = self._parse(
            ["--rollout-function-path", "my.custom.rollout_fn", "--data-source-path", "my.custom.DataSource"]
        )

        orbit_validate_args(args)

        assert args.rollout_function_path == "my.custom.rollout_fn"
        assert args.data_source_path == "my.custom.DataSource"

    def test_empty_wait_is_a_registered_argument(self):
        assert self._parse([]).multi_lora_max_empty_wait_s == 30.0
        assert self._parse(["--multi-lora-max-empty-wait-s", "5"]).multi_lora_max_empty_wait_s == 5.0

    def test_rejects_non_adam_optimizer(self):
        # Per-slot optimizer isolation (state init, retirement cleanup, step
        # clocks) only implements Adam semantics. Muon has its own dedicated
        # rejection; anything else non-Adam trips the generic guard.
        args = self._parse([])
        args.optimizer = "muon"
        with pytest.raises(AssertionError, match="does not support Muon"):
            orbit_validate_args(args)

        args = self._parse([])
        args.optimizer = "sgd"
        with pytest.raises(AssertionError, match="requires --optimizer adam"):
            orbit_validate_args(args)

    def test_rejects_experimental_ft_trainer(self, monkeypatch):
        # The v2 train group has no reconcile_adapters.
        monkeypatch.setenv("ORBIT_EXPERIMENTAL_FT_TRAINER", "1")
        args = self._parse([])

        with pytest.raises(AssertionError, match="ORBIT_EXPERIMENTAL_FT_TRAINER"):
            orbit_validate_args(args)

    def test_rejects_pipeline_parallelism(self):
        # Adapter routing is not recompute-safe under a pipelined schedule.
        args = self._parse([])
        args.pipeline_model_parallel_size = 2
        with pytest.raises(AssertionError, match="pipeline-model-parallel-size 1"):
            orbit_validate_args(args)

    def test_rejects_bshd_qkv_format(self):
        # bshd interleaves samples in the sequence-major flattening the spans assume.
        args = self._parse([])
        args.qkv_format = "bshd"
        with pytest.raises(AssertionError, match="qkv-format thd"):
            orbit_validate_args(args)

    def test_rejects_shared_outer_expert_loras(self):
        # Per-expert layout only; the flag would switch sglang to a layout training never produces.
        args = self._parse([])
        args.experts_shared_outer_loras = True
        with pytest.raises(AssertionError, match="experts-shared-outer-loras"):
            orbit_validate_args(args)

    def test_accepts_expert_leaf_targets_without_expert_tp_flag(self):
        # --expert-tensor-parallel-size stays None until Megatron's own validate_args;
        # comparing the raw value here rejected every run that omitted the flag.
        args = self._parse(["--target-modules", "gate_proj,up_proj,down_proj"])
        args.expert_tensor_parallel_size = None

        orbit_validate_args(args)

        assert args.multi_lora is True


class TestResolveFtComponents:
    def test_disabled_with_no_components_returns_empty_without_warning(self, caplog) -> None:
        """use_fault_tolerance off and no ft_components yields an empty list and no warning."""
        args = SimpleNamespace(use_fault_tolerance=False, ft_components=None)
        with caplog.at_level(logging.WARNING, logger="orbit.utils.arguments"):
            result = _resolve_ft_components(args)

        assert result == []
        assert not any("--ft-components is ignored" in record.message for record in caplog.records)

    def test_disabled_with_components_returns_empty_and_warns(self, caplog) -> None:
        """use_fault_tolerance off but ft_components set returns empty list and logs an ignore warning."""
        args = SimpleNamespace(use_fault_tolerance=False, ft_components=["train"])
        with caplog.at_level(logging.WARNING, logger="orbit.utils.arguments"):
            result = _resolve_ft_components(args)

        assert result == []
        assert any(
            "--ft-components is ignored without --use-fault-tolerance" in record.message for record in caplog.records
        )

    def test_enabled_with_no_components_returns_default(self) -> None:
        """use_fault_tolerance on with no ft_components falls back to the default ['rollout']."""
        args = SimpleNamespace(use_fault_tolerance=True, ft_components=None)
        result = _resolve_ft_components(args)

        assert result == ["rollout"]

    def test_enabled_with_components_returns_distinct_copy(self) -> None:
        """use_fault_tolerance on with ft_components returns an equal but distinct list copy."""
        components = ["train", "rollout"]
        args = SimpleNamespace(use_fault_tolerance=True, ft_components=components)
        result = _resolve_ft_components(args)

        assert result == ["train", "rollout"]
        assert result is not components


@pytest.mark.parametrize(
    ("parallel_args", "expected"),
    [
        ([], (1, 1, 1, 1)),
        (
            [
                "--sglang-tensor-parallel-size",
                "2",
                "--sglang-data-parallel-size",
                "3",
                "--sglang-pipeline-parallel-size",
                "4",
                "--sglang-expert-parallel-size",
                "5",
                "--sglang-enable-dp-attention",
            ],
            (2, 3, 4, 5),
        ),
        (
            [
                "--sglang-tp-size",
                "2",
                "--sglang-dp-size",
                "3",
                "--sglang-pp-size",
                "4",
                "--sglang-ep-size",
                "5",
                "--sglang-enable-dp-attention",
            ],
            (2, 3, 4, 5),
        ),
    ],
)
def test_sglang_parallel_sizes_use_short_namespace_fields(parallel_args, expected):
    parser = argparse.ArgumentParser()
    add_sglang_arguments(parser)
    args = parser.parse_args(parallel_args)

    assert (args.sglang_tp_size, args.sglang_dp_size, args.sglang_pp_size, args.sglang_ep_size) == expected
    assert not hasattr(args, "sglang_tensor_parallel_size")
    assert not hasattr(args, "sglang_data_parallel_size")
    assert not hasattr(args, "sglang_pipeline_parallel_size")
    assert not hasattr(args, "sglang_expert_parallel_size")

    args.rollout_num_gpus_per_engine = 8
    args.true_on_policy_mode = False
    args.recompute_logprobs_via_prefill = False
    args.sglang_router_policy = None
    args.use_session_server = False

    validate_sglang_args(args)

    assert args.sglang_tp_size == 8
    assert (args.sglang_dp_size, args.sglang_pp_size, args.sglang_ep_size) == expected[1:]


def test_sglang_parallel_size_aliases_keep_last_value():
    parser = argparse.ArgumentParser()
    add_sglang_arguments(parser)

    args = parser.parse_args(["--sglang-data-parallel-size", "2", "--sglang-dp-size", "3"])

    assert args.sglang_dp_size == 3


def _make_async_ppo_args(**overrides) -> SimpleNamespace:
    defaults = dict(
        use_critic=True,
        use_rollout_logprobs=False,
        use_tis=False,
        keep_old_actor=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestValidateAsyncOffPolicyCorrection:
    def test_ppo_without_correction_is_rejected(self):
        with pytest.raises(AssertionError, match="behavior-policy correction"):
            validate_async_off_policy_correction(_make_async_ppo_args())

    @pytest.mark.parametrize("flag", ["use_rollout_logprobs", "use_tis", "keep_old_actor"])
    def test_ppo_with_any_correction_passes(self, flag):
        validate_async_off_policy_correction(_make_async_ppo_args(**{flag: True}))

    def test_non_ppo_estimators_are_unaffected(self):
        validate_async_off_policy_correction(_make_async_ppo_args(use_critic=False))


class TestValidateRematerializeParamFromMasterWeight:
    def _make_args(self, **overrides) -> SimpleNamespace:
        args = SimpleNamespace(
            rematerialize_param_from_master_weight=True,
            train_backend="megatron",
            lora_rank=0,
            lora_adapter_path=None,
            debug_disable_optimizer=False,
            indep_dp=False,
            colocate=True,
            offload_train=True,
            offload_train_target="cpu",
            use_distributed_optimizer=True,
            keep_old_actor=False,
            kl_coef=0,
            use_kl_loss=False,
            opd_teacher_load=None,
            use_precision_aware_optimizer=False,
            optimizer_cpu_offload=False,
            overlap_param_gather=False,
            compute_advantages_and_returns=True,
            num_critic_only_steps=0,
            debug_train_only=False,
            ci_test=False,
            check_rematerialize_param_from_master_weight=False,
            disable_param_buffers_cpu_backup=False,
        )
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_valid_config_forces_no_param_buffer_cpu_backup(self):
        args = self._make_args()
        _validate_rematerialize_param_from_master_weight(args)
        assert args.disable_param_buffers_cpu_backup is True

    def test_accepts_precision_aware_with_cpu_offload(self):
        args = self._make_args(use_precision_aware_optimizer=True, optimizer_cpu_offload=True)
        _validate_rematerialize_param_from_master_weight(args)
        assert args.disable_param_buffers_cpu_backup is True

    def test_ci_test_auto_enables_the_check(self):
        args = self._make_args(ci_test=True)
        _validate_rematerialize_param_from_master_weight(args)
        assert args.check_rematerialize_param_from_master_weight is True

    def test_check_stays_off_outside_ci(self):
        args = self._make_args()
        _validate_rematerialize_param_from_master_weight(args)
        assert args.check_rematerialize_param_from_master_weight is False

    def test_accepts_ref_and_teacher_tags(self):
        for overrides in ({"use_kl_loss": True}, {"kl_coef": 0.1}, {"opd_teacher_load": "/path/to/teacher"}):
            _validate_rematerialize_param_from_master_weight(self._make_args(**overrides))

    def test_debug_train_only_silently_disables(self):
        args = self._make_args(debug_train_only=True, colocate=False)
        _validate_rematerialize_param_from_master_weight(args)
        assert args.rematerialize_param_from_master_weight is False
        assert args.disable_param_buffers_cpu_backup is False

    def test_noop_when_disabled(self):
        args = self._make_args(rematerialize_param_from_master_weight=False, colocate=False)
        _validate_rematerialize_param_from_master_weight(args)
        assert args.disable_param_buffers_cpu_backup is False

    @pytest.mark.parametrize(
        "overrides",
        [
            {"train_backend": "fsdp"},
            {"lora_rank": 8},
            {"lora_adapter_path": "/path/to/adapter"},
            {"debug_disable_optimizer": True},
            {"indep_dp": True},
            {"colocate": False},
            {"offload_train": False},
            {"offload_train_target": "disk"},
            {"use_distributed_optimizer": False},
            {"keep_old_actor": True},
            {"use_precision_aware_optimizer": True},
            {"overlap_param_gather": True},
            {"compute_advantages_and_returns": False},
            {"num_critic_only_steps": 2},
        ],
    )
    def test_rejects_unsupported_config(self, overrides):
        with pytest.raises(AssertionError):
            _validate_rematerialize_param_from_master_weight(self._make_args(**overrides))

    def test_backend_is_checked_before_megatron_only_args(self):
        # An fsdp Namespace has none of the megatron args the later asserts read.
        args = SimpleNamespace(
            rematerialize_param_from_master_weight=True,
            train_backend="fsdp",
            debug_train_only=False,
        )
        with pytest.raises(AssertionError, match="Megatron"):
            _validate_rematerialize_param_from_master_weight(args)


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        ([], False),
        (["--skip-actor-forward-only"], True),
    ],
)
def test_skip_actor_forward_only_flag_is_parsed(extra_args, expected):
    parser = argparse.ArgumentParser()
    get_orbit_extra_args_provider()(parser)

    args = parser.parse_args(extra_args + REQUIRED_ARGS)

    assert args.skip_actor_forward_only is expected


def test_skip_actor_forward_only_is_gated_during_orbit_validation():
    parser = argparse.ArgumentParser()
    get_orbit_extra_args_provider()(parser)
    args = parser.parse_args(
        ["--skip-actor-forward-only", "--global-batch-size", "32", "--num-rollout", "1"] + REQUIRED_ARGS
    )
    vars(args).update(
        hidden_dropout=0.0,
        attention_dropout=0.0,
        lora_dropout=0.0,
        moe_input_jitter_eps=None,
        moe_router_force_biased=None,
        moe_router_force_load_balancing=False,
        moe_router_load_balancing_type="aux_loss",
    )

    with pytest.raises(AssertionError, match="--skip-actor-forward-only"):
        orbit_validate_args(args)


def _make_skip_actor_forward_only_args(**overrides) -> SimpleNamespace:
    defaults = dict(
        compute_advantages_and_returns=True,
        custom_megatron_before_log_prob_hook_path=None,
        custom_megatron_before_train_step_hook_path=None,
        custom_model_provider_path=None,
        dumper_enable=False,
        dumper_fwd_only=None,
        dumper_source_patcher_config_train=None,
        dump_details=None,
        get_mismatch_metrics=False,
        global_batch_size=64,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        keep_old_actor=False,
        kl_coef=0.0,
        lora_dropout=0.0,
        log_correct_samples=False,
        loss_type="policy_loss",
        moe_input_jitter_eps=None,
        moe_router_force_biased=None,
        moe_router_force_load_balancing=False,
        moe_router_load_balancing_type="aux_loss",
        multi_lora=False,
        n_samples_per_prompt=8,
        num_steps_per_rollout=None,
        rollout_batch_size=8,
        rollout_data_postprocess_path=None,
        save_debug_train_data=None,
        train_backend="megatron",
        true_on_policy_mode=False,
        use_dynamic_global_batch_size=False,
        use_indexer_replay=False,
        use_opd=False,
        use_rollout_entropy=False,
        use_rollout_indexer_replay=False,
        use_rollout_logprobs=False,
        use_rollout_routing_replay=False,
        use_routing_replay=False,
        use_tis=False,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestValidateSkipActorForwardOnly:
    def test_valid_single_step_configuration_passes(self):
        validate_skip_actor_forward_only(_make_skip_actor_forward_only_args())

    def test_zero_moe_input_jitter_passes(self):
        validate_skip_actor_forward_only(_make_skip_actor_forward_only_args(moe_input_jitter_eps=0.0))

    def test_tis_configuration_passes(self):
        validate_skip_actor_forward_only(_make_skip_actor_forward_only_args(use_tis=True))

    def test_rollout_logprobs_configuration_passes(self):
        validate_skip_actor_forward_only(_make_skip_actor_forward_only_args(use_rollout_logprobs=True))

    def test_rollout_logprobs_with_mismatch_metrics_passes(self):
        validate_skip_actor_forward_only(
            _make_skip_actor_forward_only_args(
                get_mismatch_metrics=True,
                use_rollout_logprobs=True,
            )
        )

    @pytest.mark.parametrize(
        "overrides",
        [
            {"dumper_enable": True},
            {"dumper_fwd_only": ["enable=true"]},
            {"dumper_enable": True, "dumper_fwd_only": ["enable=false"]},
            {"dump_details": "/tmp/details"},
            {
                "dump_details": "/tmp/details",
                "save_debug_train_data": "/tmp/details/train_data/{rollout_id}_{rank}.pt",
            },
        ],
    )
    def test_dumper_configuration_passes(self, overrides):
        validate_skip_actor_forward_only(_make_skip_actor_forward_only_args(**overrides))

    @pytest.mark.parametrize(
        "overrides",
        [
            {"train_backend": "fsdp"},
            {"loss_type": "custom_loss"},
            {"compute_advantages_and_returns": False},
            {"keep_old_actor": True},
            {"kl_coef": 0.1},
            {"use_opd": True},
            {"hidden_dropout": 0.1},
            {"attention_dropout": 0.1},
            {"lora_dropout": 0.1},
            {"moe_input_jitter_eps": 0.1},
            {"moe_router_force_load_balancing": True},
            {"moe_router_force_biased": 0.0},
            {"moe_router_load_balancing_type": ["sinkhorn"]},
            {"use_rollout_entropy": True},
            {"true_on_policy_mode": True},
            {"log_correct_samples": True},
            {"rollout_data_postprocess_path": "pkg.hook"},
            {"custom_megatron_before_log_prob_hook_path": "pkg.hook"},
            {"custom_megatron_before_train_step_hook_path": "pkg.hook"},
            {"custom_model_provider_path": "pkg.model_provider"},
            {"dumper_source_patcher_config_train": "patcher.yaml"},
            {"save_debug_train_data": "train-{rollout_id}.pt"},
            {"use_routing_replay": True},
            {"use_indexer_replay": True},
            {"num_steps_per_rollout": 2},
            {"global_batch_size": 32},
        ],
    )
    def test_incompatible_configuration_is_rejected(self, overrides):
        with pytest.raises(AssertionError, match="--skip-actor-forward-only"):
            validate_skip_actor_forward_only(_make_skip_actor_forward_only_args(**overrides))

    @pytest.mark.parametrize(
        ("base_flag", "rollout_flag"),
        [
            ("use_routing_replay", "use_rollout_routing_replay"),
            ("use_indexer_replay", "use_rollout_indexer_replay"),
        ],
    )
    def test_rollout_replay_is_compatible(self, base_flag, rollout_flag):
        validate_skip_actor_forward_only(
            _make_skip_actor_forward_only_args(
                **{
                    base_flag: True,
                    rollout_flag: True,
                }
            )
        )

    def test_dynamic_global_batch_size_defers_step_count_to_runtime(self):
        validate_skip_actor_forward_only(
            _make_skip_actor_forward_only_args(
                global_batch_size=32,
                use_dynamic_global_batch_size=True,
            )
        )
