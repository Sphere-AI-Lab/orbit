import argparse
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from miles.utils.arguments import (
    _maybe_apply_dumper_overrides,
    _validate_opd_dagger_args,
    _validate_opd_sglang_scoring_args,
    _validate_opd_task_reward_args,
    get_miles_extra_args_provider,
    hf_validate_args,
)
from miles.utils.misc import function_registry

PATH_ARGS = ["--rollout-function-path", "--custom-generate-function-path"]
REQUIRED_ARGS = ["--rollout-batch-size", "64"]


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
            get_miles_extra_args_provider()(parser)
            args, _ = parser.parse_known_args()
            assert args.my_custom_arg == 100

    def test_skips_function_without_add_arguments(self, path_arg):
        fn = make_function_without_add_arguments()
        with function_registry.temporary("test:fn", fn), patch.object(
            sys, "argv", ["test", path_arg, "test:fn"] + REQUIRED_ARGS
        ):
            parser = argparse.ArgumentParser()
            get_miles_extra_args_provider()(parser)


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


def test_recompute_logprobs_via_prefill_flag_is_parsed():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)

    args = parser.parse_args(["--recompute-logprobs-via-prefill"] + REQUIRED_ARGS)

    assert args.recompute_logprobs_via_prefill is True


def test_sglang_mm_exact_scoring_suffix_is_opt_in():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)

    default_args = parser.parse_args(REQUIRED_ARGS)
    enabled_args = parser.parse_args(["--sglang-mm-exact-scoring-suffix"] + REQUIRED_ARGS)

    assert default_args.sglang_mm_exact_scoring_suffix is False
    assert enabled_args.sglang_mm_exact_scoring_suffix is True


def test_opd_dagger_defaults_are_disabled():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)

    args = parser.parse_args(REQUIRED_ARGS)

    assert args.opd_dagger_top_k == 0
    assert args.opd_dagger_coef == 0.0
    assert args.opd_dagger_loss == "cross_entropy"
    assert args.opd_log_task_reward is False
    assert args.opd_optimize_task_reward is False
    assert args.opd_task_reward_coef == 1.0


def test_opd_task_reward_logging_argument_is_parsed():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)

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
        "custom_rm_path": "miles.rollout.on_policy_distillation.reward_func",
        "custom_reward_post_process_path": "miles.rollout.on_policy_distillation.post_process_rewards",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_opd_task_reward_optimization_arguments_are_parsed():
    parser = argparse.ArgumentParser()
    get_miles_extra_args_provider()(parser)

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
        "custom_rm_path": "miles.rollout.on_policy_distillation.reward_func",
        "custom_reward_post_process_path": "miles.rollout.on_policy_distillation.post_process_rewards",
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
    get_miles_extra_args_provider()(parser)

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
