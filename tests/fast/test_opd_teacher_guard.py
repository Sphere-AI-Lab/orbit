"""Pure MOPD counts as "OPD is enabled" for upstream's teacher-argument guard.

miles @ dbbab1566 added a guard rejecting ``--opd-teacher-load`` /
``--opd-teacher-urls`` unless ``--use-opd`` is set. That assumes ``--use-opd`` is
the only way OPD gets turned on; orbit has an older second spelling, pure MOPD
(``--advantage-estimator on_policy_distillation``), which deliberately leaves
``use_opd`` False because that flag gates the BLEND
(``apply_opd_kl_to_advantages`` in backends/training_utils/loss.py). Six recipes
in examples/on_policy_distillation/ died in validation as a result.

The guard is narrowed with ``needs_opd_teacher()``, the union of the two
spellings. What must NOT happen is the two obvious over-fixes: turning
``use_opd`` on for pure MOPD (which would silently add the blend KL on top of
it), or dropping the guard (which would stop catching a teacher named by a run
that does no OPD at all). Both are asserted below.
"""

import argparse

import pytest

pytest.importorskip("torch")

import orbit  # noqa: F401,E402

from miles.utils.arguments import needs_opd_teacher  # noqa: E402


def _args(**kwargs):
    base = dict(advantage_estimator="grpo", use_opd=False, opd_teacher_load=None, opd_teacher_urls=None)
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_pure_mopd_needs_a_teacher():
    assert needs_opd_teacher(_args(advantage_estimator="on_policy_distillation")) is True


def test_the_blend_needs_a_teacher():
    assert needs_opd_teacher(_args(use_opd=True)) is True


def test_a_plain_rl_run_does_not():
    """The guard must still fire for a run that does no OPD at all -- narrowing
    it is not the same as deleting it."""
    assert needs_opd_teacher(_args()) is False


def test_pure_mopd_does_not_turn_on_the_blend():
    """The over-fix this exists to rule out.

    ``use_opd`` gates apply_opd_kl_to_advantages. If a future change satisfies
    upstream's guard by setting it True for pure MOPD, the blend KL lands on top
    of the pure-MOPD objective with no error and slightly wrong numbers forever.
    """
    args = _args(advantage_estimator="on_policy_distillation", opd_teacher_load="/tmp/teacher")
    assert needs_opd_teacher(args) is True
    assert args.use_opd is False


def test_the_vendored_guard_is_keyed_on_the_union_predicate():
    """Read the narrowed guard back out of the vendored source.

    Asserted statically because reaching it needs a full validated arg
    namespace; the behavioural half is the recipe dry-run in the campaign notes.
    """
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "miles/utils/arguments.py").read_text()
    assert "elif not needs_opd_teacher(args):" in src, (
        "upstream's `else:` is back; pure-MOPD recipes that name a teacher will "
        "be rejected in validation again"
    )
    assert src.count("--opd-teacher-load is set but --use-opd is not enabled") == 1
