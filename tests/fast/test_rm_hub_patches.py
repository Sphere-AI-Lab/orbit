"""Orbit's reward-hub additions still happen with both vendored files pristine.

``miles/rollout/rm_hub/__init__.py`` used to carry two extra ``rm_type``
branches and a ``default_async_rm`` split; ``deepscaler.py`` used to carry a
refactor whose only purpose was to let the gemma reward reuse the boxed-answer
grader. All of it is expressed from orbit now:

* the two rm_types are a DELEGATING patch, so upstream still owns every other
  type and both NotImplementedError messages;
* ``default_async_rm`` is a lift whose only caller is orbit (opd_sglang);
* ``get_gemma_math_reward`` is a lift that CALLS upstream's grader rather than
  copying it, which is why deepscaler.py needed no patch at all.

An rm_type that quietly stops being recognised raises NotImplementedError, but
one that quietly stops going through upstream's body just returns a slightly
different number forever -- so the delegation is asserted, not assumed.
"""

import argparse
import ast
import asyncio
from pathlib import Path

import pytest

pytest.importorskip("torch")

import orbit  # noqa: F401,E402  -- importing orbit installs the patch

from miles.rollout import rm_hub  # noqa: E402
from miles.utils.types import Sample  # noqa: E402
from orbit.rewards.rm_hub_patches import _NoCustomRM, default_async_rm  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PATCH_MODULE = "orbit.rewards.rm_hub_patches"


def _args(**kwargs):
    return argparse.Namespace(**{"custom_rm_path": None, "rm_type": "", **kwargs})


def _sample(response, label="42", **metadata):
    return Sample(prompt="ignored", response=response, label=label, metadata=metadata)


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# the delegating patch on async_rm
# --------------------------------------------------------------------------


def test_the_patch_is_installed():
    assert rm_hub.async_rm.__module__ == PATCH_MODULE
    assert hasattr(rm_hub, "_orbit_unpatched_async_rm"), (
        "the pristine upstream function must be kept so the patch can delegate"
    )


def test_the_vendored_files_no_longer_carry_orbit_code():
    """The point of the move. If either fails, that file went dirty again.

    ``gemma_math`` is NOT checked for here, unlike on orbit-main-isolated: miles
    @ dbbab1566 ships that reward and dispatches it, so the name in the hub is
    upstream's own. What must stay gone is orbit's ``math_alignment`` branch and
    the ``default_async_rm`` split.
    """
    for rel in ("miles/rollout/rm_hub/__init__.py", "miles/rollout/rm_hub/deepscaler.py"):
        src = (REPO / rel).read_text()
        assert "ORBIT" not in src, rel
        assert "orbit" not in src, rel
    hub = (REPO / "miles/rollout/rm_hub/__init__.py").read_text()
    assert "math_alignment" not in hub and "default_async_rm" not in hub
def test_orbit_reaches_math_alignment_through_the_same_branch(monkeypatch):
    from orbit.rewards import math_alignment

    seen = {}

    def fake(response, label, metadata):
        seen.update(response=response, label=label, metadata=metadata)
        return True

    monkeypatch.setattr(math_alignment, "grade_math_alignment", fake)
    sample = _sample("\\boxed{42}", rm_type="math_alignment", dataset_name="math500")
    assert _run(rm_hub.async_rm(_args(), sample)) == 1
    assert seen["response"] == "\\boxed{42}"
    assert seen["metadata"]["dataset_name"] == "math500"


def test_orbits_types_inherit_upstreams_boxed_prefix_handling(monkeypatch):
    """The reason orbit resolves rm_type the way upstream does rather than
    reading it raw: ``boxed_<type>`` has to strip the prefix AND hand the bare
    extracted answer on, for orbit's types exactly as for upstream's. Reading
    the raw type instead would fall through to upstream and raise here.
    """
    from orbit.rewards import math_alignment

    seen = {}
    monkeypatch.setattr(
        math_alignment,
        "grade_math_alignment",
        lambda response, label, metadata: seen.update(response=response) or True,
    )
    sample = _sample("reasoning \\boxed{42}", rm_type="boxed_math_alignment")
    assert _run(rm_hub.async_rm(_args(), sample)) == 1
    assert seen == {"response": "42"}


def test_the_rm_type_can_come_from_args_as_well_as_metadata():
    assert _run(rm_hub.async_rm(_args(rm_type="gemma_math"), _sample("<channel|> \\boxed{42}"))) == 1


def test_upstreams_types_still_run_upstreams_body():
    """The delegation property. If this fails because orbit copied the dispatch,
    upstream's fixes to every grader stop reaching us."""
    sample = _sample("reasoning </think> \\boxed{42}", rm_type="deepscaler")
    assert _run(rm_hub.async_rm(_args(), sample)) == _run(
        rm_hub._orbit_unpatched_async_rm(_args(), sample)
    ) == 1


def test_an_unknown_type_still_raises_upstreams_error():
    """A patch that swallowed this would turn a misconfigured run into a silent
    stream of zero rewards."""
    with pytest.raises(NotImplementedError, match="nonsense"):
        _run(rm_hub.async_rm(_args(), _sample("x", rm_type="nonsense")))
    with pytest.raises(NotImplementedError, match="not specified"):
        _run(rm_hub.async_rm(_args(), _sample("x")))


def test_a_custom_rm_still_wins_over_orbits_types():
    """Upstream checks custom_rm_path first; orbit's branch must not jump it."""
    args = _args(
        custom_rm_path="tests.fast.test_rm_hub_patches._hijacking_rm", rm_type="gemma_math"
    )
    assert _run(rm_hub.async_rm(args, _sample("t <channel|> \\boxed{42}"))) == "hijacked"


async def _hijacking_rm(args, sample, **kwargs):
    return "hijacked"


# --------------------------------------------------------------------------
# default_async_rm: the lift orbit's OPD teacher scoring calls back into
# --------------------------------------------------------------------------


def test_default_async_rm_reaches_the_task_reward_past_a_custom_rm():
    """The case it exists for: OPD teacher scoring occupies the reward slot as a
    transport, and its eval samples still need the real task reward."""
    args = _args(custom_rm_path="tests.fast.test_rm_hub_patches._hijacking_rm")
    sample = _sample("reasoning </think> \\boxed{42}", rm_type="deepscaler")
    assert _run(default_async_rm(args, sample)) == 1
    # ...through the same slot the custom rm would otherwise have taken.
    assert _run(rm_hub.async_rm(args, sample)) == "hijacked"


def test_default_async_rm_still_sees_orbits_types():
    args = _args(custom_rm_path="tests.fast.test_rm_hub_patches._hijacking_rm")
    sample = _sample("t <channel|> \\boxed{42}", rm_type="gemma_math")
    assert _run(default_async_rm(args, sample)) == 1


def test_the_args_view_hides_only_custom_rm_path():
    args = _args(custom_rm_path="x", rm_type="deepscaler", rm_url="http://example")
    view = _NoCustomRM(args)
    assert view.custom_rm_path is None
    assert view.rm_type == "deepscaler"
    assert view.rm_url == "http://example"
    # A view, not a copy: the real namespace is untouched.
    assert args.custom_rm_path == "x"


def test_orbit_is_the_only_caller_so_it_is_not_imported_back_into_miles():
    """A lift whose sole consumer is orbit should leave no trace in the vendored
    tree; an import re-added there is how the file goes dirty again."""
    hits = []
    for path in (REPO / "miles").rglob("*.py"):
        if "default_async_rm" in path.read_text(errors="surrogateescape"):
            hits.append(str(path))
    assert not hits, hits


def test_the_opd_caller_imports_it_from_orbit():
    tree = ast.parse((REPO / "orbit" / "opd" / "opd_sglang.py").read_text())
    sources = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and any(a.name == "default_async_rm" for a in node.names)
    }
    assert sources == {PATCH_MODULE}


def test_the_patch_survives_the_hub_being_imported_before_orbit():
    """Import ORDER must not decide whether the patch takes effect.

    Both vendored callers bind the name at import time
    (``from .rm_hub import async_rm``), so a process that reaches the hub before
    it runs ``import orbit`` -- a Ray actor unpickling its worker class does
    exactly that -- would keep calling upstream's unpatched dispatch while the
    patch looks installed. orbit/patch/runtime.py::_repoint_reexports is what
    prevents it; run it for real rather than simulating it.
    """
    import os
    import subprocess
    import sys

    probe = (
        "import miles.rollout.rm_hub as pkg\n"  # BEFORE orbit
        "import orbit\n"
        "print(pkg.async_rm.__module__)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith(PATCH_MODULE), out.stdout


def test_both_vendored_callers_bind_the_name_at_import_time():
    """Why the test above is not paranoia: these are the stale bindings."""
    binders = []
    for rel in ("miles/rollout/sglang_rollout.py", "miles/rollout/inference_rollout/inference_rollout_common.py"):
        tree = ast.parse((REPO / rel).read_text())
        if any(
            isinstance(node, ast.ImportFrom) and any(a.name == "async_rm" for a in node.names)
            for node in tree.body
        ):
            binders.append(rel)
    assert binders, "no module-level `from ... import async_rm` left -- retire this test"
