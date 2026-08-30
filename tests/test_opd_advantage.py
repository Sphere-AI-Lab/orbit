import pytest
import torch

import miles.backends.training_utils.loss as training_loss


_ORIGINAL_VANILLA_TIS = training_loss.vanilla_tis_function

from miles.backends.training_utils.loss_hub.math_utils import (
    apply_opd_icepop_gate,
    apply_opd_kl_to_advantages,
    icepop_gate,
    opd_mopd_advantages,
)
from miles.utils.types import Sample


def test_sample_declares_teacher_log_probs_default_none():
    s = Sample(index=0, prompt="p", response="r", response_length=3)
    assert s.teacher_log_probs is None


def test_sample_validate_raises_on_teacher_log_probs_length_mismatch():
    s = Sample(
        index=0, prompt="p", tokens=[1, 2, 3], response="r", response_length=3, teacher_log_probs=[0.1, 0.2]
    )
    with pytest.raises(AssertionError, match="teacher_log_probs"):
        s.validate()


def test_sample_validate_passes_with_correct_teacher_log_probs_length():
    s = Sample(
        index=0, prompt="p", tokens=[1, 2, 3], response="r", response_length=3, teacher_log_probs=[0.1, 0.2, 0.3]
    )
    s.validate()


def test_opd_mopd_advantages_raises_without_teacher_log_probs():
    student_log_probs = [torch.tensor([0.1, 0.2, 0.3])]
    response_lengths = [3]

    with pytest.raises(ValueError, match="--opd-type") as excinfo:
        opd_mopd_advantages({"teacher_log_probs": None}, student_log_probs, response_lengths)
    # The advice must NOT mention --use-opd: pure MOPD + --use-opd is rejected
    # as mutually exclusive by _validate_opd_args, so following that advice
    # would trade one error for another.
    assert "--use-opd" not in str(excinfo.value)


def test_opd_mopd_advantages_matches_teacher_minus_student():
    student_log_probs = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([-0.5, -0.1])]
    teacher_log_probs = [torch.tensor([0.5, 0.4, 0.3]), torch.tensor([-0.2, -0.3])]
    response_lengths = [3, 2]
    rollout_data = {"teacher_log_probs": teacher_log_probs}

    advantages = opd_mopd_advantages(rollout_data, student_log_probs, response_lengths)

    for adv, teacher, student in zip(advantages, teacher_log_probs, student_log_probs, strict=True):
        torch.testing.assert_close(adv, teacher - student)


def test_opd_mopd_advantages_raises_on_length_mismatch():
    student_log_probs = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([-0.5, -0.1])]
    teacher_log_probs = [torch.tensor([0.5, 0.4, 0.3])]
    response_lengths = [3, 2]
    rollout_data = {"teacher_log_probs": teacher_log_probs}

    with pytest.raises(ValueError):
        opd_mopd_advantages(rollout_data, student_log_probs, response_lengths)


def test_apply_opd_kl_to_advantages_blends_reverse_kl():
    student_log_probs = [torch.tensor([0.1, 0.2, 0.3]), torch.tensor([-0.5, -0.1])]
    teacher_log_probs = [torch.tensor([0.5, 0.4, 0.3]), torch.tensor([-0.2, -0.3])]
    advantages = [torch.ones(3), torch.ones(2)]
    rollout_data = {"teacher_log_probs": teacher_log_probs}

    apply_opd_kl_to_advantages(1.0, rollout_data, advantages, student_log_probs)

    for adv, teacher, student in zip(advantages, teacher_log_probs, student_log_probs, strict=True):
        torch.testing.assert_close(adv, torch.ones_like(student) - (student - teacher))
    assert "opd_reverse_kl" in rollout_data


def test_apply_opd_kl_to_advantages_raises_without_teacher_log_probs():
    advantages = [torch.ones(3)]
    student_log_probs = [torch.tensor([0.1, 0.2, 0.3])]

    with pytest.raises(ValueError):
        apply_opd_kl_to_advantages(1.0, {"teacher_log_probs": None}, advantages, student_log_probs)


def test_apply_opd_kl_to_advantages_zero_coef_is_noop():
    student_log_probs = [torch.tensor([0.1, 0.2, 0.3])]
    teacher_log_probs = [torch.tensor([0.5, 0.4, 0.3])]
    advantages = [torch.ones(3)]
    rollout_data = {"teacher_log_probs": teacher_log_probs}

    apply_opd_kl_to_advantages(0.0, rollout_data, advantages, student_log_probs)

    torch.testing.assert_close(advantages[0], torch.ones(3))


# --- Phase 3 / Task 3.1: ICE-POP gate (shared with the PG icepop_function) ---


def test_icepop_gate_in_band_passes_ratio_through():
    ratio = torch.tensor([0.5, 1.0, 1.5, 2.0])
    weight = icepop_gate(ratio, 0.5, 2.0)
    torch.testing.assert_close(weight, ratio)


def test_icepop_gate_out_of_band_zeroed():
    ratio = torch.tensor([0.1, 1.0, 5.0])
    weight = icepop_gate(ratio, 0.5, 2.0)
    torch.testing.assert_close(weight, torch.tensor([0.0, 1.0, 0.0]))


def test_icepop_gate_matches_inline_torch_where():
    # Behavior-preservation: icepop_gate must equal the exact expression that
    # icepop_function used inline (loss.py) so the refactor is a no-op for the PG path.
    ratio = torch.tensor([-0.3, 0.0, 0.4999, 0.5, 1.0, 2.0, 2.0001, 7.3])
    low, high = 0.5, 2.0
    expected = torch.where((ratio >= low) & (ratio <= high), ratio, torch.zeros_like(ratio))
    torch.testing.assert_close(icepop_gate(ratio, low, high), expected)


def test_apply_opd_icepop_gate_zeros_out_of_band_keeps_in_band():
    # Build train vs rollout log-probs so tokens 0,1 are in-band with ratio == 1
    # (train == rollout => unchanged) and tokens 2,3 are out-of-band (=> zeroed).
    train = torch.tensor([0.0, -0.3, 0.0, -5.0])
    rollout = torch.tensor([0.0, -0.3, -5.0, 0.0])  # ratio = exp(0,0,+5,-5)
    advantages = [torch.tensor([1.5, -2.0, 3.0, -4.0])]
    rollout_data = {"log_probs": [train], "rollout_log_probs": [rollout]}

    apply_opd_icepop_gate(rollout_data, advantages, 0.5, 2.0)

    torch.testing.assert_close(advantages[0], torch.tensor([1.5, -2.0, 0.0, 0.0]))


def test_apply_opd_icepop_gate_reweights_in_band_by_ratio():
    # In-band tokens are importance-reweighted by the ratio (mirrors PG icepop:
    # pg_loss * ice_weight), not merely masked.
    train = torch.tensor([0.5])
    rollout = torch.tensor([0.0])  # ratio = exp(0.5) ~= 1.6487, inside [0, 2]
    advantages = [torch.tensor([2.0])]
    rollout_data = {"log_probs": [train], "rollout_log_probs": [rollout]}

    apply_opd_icepop_gate(rollout_data, advantages, 0.0, 2.0)

    torch.testing.assert_close(advantages[0], torch.tensor([2.0]) * torch.exp(torch.tensor([0.5])))


def test_apply_opd_icepop_gate_noop_when_ratio_one():
    # Parity: when train == rollout (ratio == 1 everywhere), the OPD advantage is
    # unchanged -- the same property that makes --opd-icepop off a no-op.
    lp = [torch.tensor([0.1, -0.2, 0.3])]
    advantages = [torch.tensor([1.0, -2.0, 3.0])]
    rollout_data = {"log_probs": lp, "rollout_log_probs": [lp[0].clone()]}

    apply_opd_icepop_gate(rollout_data, advantages, 0.0, 2.0)

    torch.testing.assert_close(advantages[0], torch.tensor([1.0, -2.0, 3.0]))


def test_apply_opd_icepop_gate_raises_without_rollout_log_probs():
    advantages = [torch.tensor([1.0, 2.0])]
    rollout_data = {"log_probs": [torch.tensor([0.0, 0.0])], "rollout_log_probs": None}

    with pytest.raises(ValueError, match="rollout_log_probs"):
        apply_opd_icepop_gate(rollout_data, advantages, 0.0, 2.0)


def _exercise_policy_ratio(
    monkeypatch: pytest.MonkeyPatch,
    *,
    collection_log_probs: torch.Tensor,
    rollout_log_probs: torch.Tensor,
    force_on_policy_ratio: bool,
) -> dict[str, torch.Tensor]:
    current_log_probs = torch.tensor([-0.4, 0.1, 0.8], requires_grad=True)
    captures: dict[str, torch.Tensor] = {}
    args = type(
        "Args",
        (),
        {
            "use_rollout_logprobs": False,
            "use_opsm": False,
            "advantage_estimator": "on_policy_distillation",
            "force_on_policy_ratio": force_on_policy_ratio,
            "entropy_coef": 0.0,
            "eps_clip": 0.2,
            "eps_clip_high": 0.2,
            "eps_clip_c": None,
            "get_mismatch_metrics": False,
            "use_tis": True,
            "tis_clip_low": 0.2,
            "tis_clip": 5.0,
            "custom_tis_function_path": None,
            "calculate_per_token_loss": True,
            "qkv_format": "thd",
            "custom_pg_loss_reducer_function_path": None,
            "use_kl_loss": False,
        },
    )()
    batch = {
        "advantages": [torch.ones(3)],
        "log_probs": [collection_log_probs],
        "rollout_log_probs": [rollout_log_probs],
        "response_lengths": [3],
        "total_lengths": [3],
        "loss_masks": [torch.ones(3)],
        "unconcat_tokens": [torch.arange(3)],
    }

    monkeypatch.setattr(training_loss, "get_parallel_state", lambda: object())
    monkeypatch.setattr(
        training_loss,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: {
            "log_probs": [current_log_probs],
            "entropy": [torch.zeros_like(current_log_probs)],
        },
    )

    def policy_loss(
        ppo_kl: torch.Tensor,
        advantages: torch.Tensor,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        captures["ppo_kl"] = ppo_kl.detach().clone()
        ratio = torch.exp(-ppo_kl)
        captures["ppo_ratio"] = ratio.detach().clone()
        return -(advantages * ratio), torch.zeros_like(ratio)

    monkeypatch.setattr(training_loss, "compute_policy_loss", policy_loss)
    def capture_tis(**kwargs):
        train = torch.cat(kwargs["train_log_probs"])
        rollout = torch.cat(kwargs["rollout_log_probs"])
        captures["tis_weight"] = torch.exp(train - rollout).clamp(
            min=kwargs["args"].tis_clip_low,
            max=kwargs["args"].tis_clip,
        )
        captures["tis_train_log_probs"] = train.clone()
        captures["tis_rollout_log_probs"] = rollout.clone()
        return _ORIGINAL_VANILLA_TIS(**kwargs)

    monkeypatch.setattr(training_loss, "vanilla_tis_function", capture_tis)

    def reduce(values: torch.Tensor) -> torch.Tensor:
        return values.mean()

    monkeypatch.setattr(
        training_loss,
        "get_sum_of_sample_mean",
        lambda *args, **kwargs: reduce,
    )
    monkeypatch.setattr(
        training_loss,
        "_response_masked_max",
        lambda values, **kwargs: values.max(),
    )

    loss, _ = training_loss.policy_loss_function(
        args,
        batch,
        torch.zeros(1, 3, 2, requires_grad=True),
        reduce,
    )
    loss.backward()
    assert current_log_probs.grad is not None
    captures["current_log_probs"] = current_log_probs.detach().clone()
    captures["current_grad"] = current_log_probs.grad.detach().clone()
    return captures


def test_force_on_policy_ratio_is_one_while_tis_remains_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = torch.log(torch.tensor([0.1, 1.0, 10.0]))
    rollout = torch.zeros(3)

    captures = _exercise_policy_ratio(
        monkeypatch,
        collection_log_probs=collection,
        rollout_log_probs=rollout,
        force_on_policy_ratio=True,
    )

    assert torch.equal(captures["ppo_kl"], torch.zeros(3))
    assert torch.equal(captures["ppo_ratio"], torch.ones(3))
    assert torch.equal(captures["tis_weight"], torch.tensor([0.2, 1.0, 5.0]))
    assert torch.equal(captures["tis_train_log_probs"], collection)
    assert torch.equal(captures["tis_rollout_log_probs"], rollout)
    assert torch.count_nonzero(captures["current_grad"]) > 0


def test_force_on_policy_ratio_ignores_collection_changes_but_tis_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rollout = torch.zeros(3)
    first = _exercise_policy_ratio(
        monkeypatch,
        collection_log_probs=torch.log(torch.tensor([0.1, 1.0, 10.0])),
        rollout_log_probs=rollout,
        force_on_policy_ratio=True,
    )
    second = _exercise_policy_ratio(
        monkeypatch,
        collection_log_probs=torch.log(torch.tensor([0.5, 2.0, 3.0])),
        rollout_log_probs=rollout,
        force_on_policy_ratio=True,
    )

    assert torch.equal(first["ppo_ratio"], second["ppo_ratio"])
    assert torch.equal(first["ppo_ratio"], torch.ones(3))
    assert not torch.equal(first["tis_weight"], second["tis_weight"])


def test_force_on_policy_ratio_ignores_rollout_changes_but_tis_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = torch.tensor([-0.4, 0.0, 0.7])
    first = _exercise_policy_ratio(
        monkeypatch,
        collection_log_probs=collection,
        rollout_log_probs=torch.zeros(3),
        force_on_policy_ratio=True,
    )
    second = _exercise_policy_ratio(
        monkeypatch,
        collection_log_probs=collection,
        rollout_log_probs=torch.tensor([1.0, -1.0, 0.2]),
        force_on_policy_ratio=True,
    )

    assert torch.equal(first["ppo_ratio"], second["ppo_ratio"])
    assert not torch.equal(first["tis_weight"], second["tis_weight"])


def test_disabling_force_on_policy_ratio_preserves_existing_ppo_ratio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    collection = torch.tensor([-1.0, 0.4, 1.2])
    captures = _exercise_policy_ratio(
        monkeypatch,
        collection_log_probs=collection,
        rollout_log_probs=torch.zeros(3),
        force_on_policy_ratio=False,
    )
    expected_kl = collection - captures["current_log_probs"]
    expected_ratio = torch.exp(-expected_kl)

    assert torch.equal(captures["ppo_kl"], expected_kl)
    assert torch.equal(captures["ppo_ratio"], expected_ratio)


def test_apply_opd_kl_is_noop_when_student_log_probs_none():
    # Critic path: teacher_log_probs never reaches the critic and, with KL off,
    # neither do student log-probs. The blend must be a silent no-op (miles
    # semantics), not a crash.
    advantages = [torch.tensor([1.0, 2.0])]
    rollout_data = {"teacher_log_probs": [torch.tensor([0.5, 0.5])]}

    apply_opd_kl_to_advantages(0.5, rollout_data, advantages, None)

    torch.testing.assert_close(advantages[0], torch.tensor([1.0, 2.0]))
    assert "opd_reverse_kl" not in rollout_data


def test_apply_opd_kl_raises_on_length_mismatch():
    advantages = [torch.tensor([1.0]), torch.tensor([2.0])]
    student_log_probs = [torch.tensor([0.1]), torch.tensor([0.2])]
    rollout_data = {"teacher_log_probs": [torch.tensor([0.3])]}

    with pytest.raises(ValueError, match="length mismatch"):
        apply_opd_kl_to_advantages(0.5, rollout_data, advantages, student_log_probs)


def test_apply_opd_kl_uses_precomputed_reverse_kl():
    # Top-k OPD: rollout-side scoring stores per-token reverse KL; the blend
    # must consume it directly (no teacher_log_probs required).
    advantages = [torch.tensor([1.0, 1.0])]
    rollout_data = {"opd_reverse_kl": [torch.tensor([0.2, 0.4])]}
    student_log_probs = [torch.tensor([-0.1, -0.2])]

    apply_opd_kl_to_advantages(0.5, rollout_data, advantages, student_log_probs)

    torch.testing.assert_close(advantages[0], torch.tensor([1.0 - 0.5 * 0.2, 1.0 - 0.5 * 0.4]))


def test_opd_mopd_advantages_uses_precomputed_reverse_kl():
    # Pure MOPD with top-k scoring: advantage = -reverse_kl per token.
    student_log_probs = [torch.tensor([-0.1, -0.2])]
    rollout_data = {"opd_reverse_kl": [torch.tensor([0.2, 0.4])]}

    out = opd_mopd_advantages(rollout_data, student_log_probs, [2])

    torch.testing.assert_close(out[0], torch.tensor([-0.2, -0.4]))
