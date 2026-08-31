"""Orbit's two additions to ``miles.rollout.rm_hub``, moved out of the vendored file.

The vendored ``async_rm`` used to carry both of them, plus a split that existed
only to expose the second one. Expressing them from here is what makes
``miles/rollout/rm_hub/__init__.py`` byte-pristine again.

* Two extra ``rm_type`` branches -- ``gemma_math`` and ``math_alignment``. This
  is a DELEGATING patch: orbit answers only the two types upstream has never
  heard of, and everything else -- ``custom_rm_path``, ``remote_rm``, the
  ``boxed_`` prefix, every rule-based grader, and both NotImplementedError
  messages -- goes straight through to upstream's body. Orbit carries about ten
  lines instead of the forty-line dispatch, and an upstream fix to any grader
  keeps reaching us.

  The rm_type resolution below duplicates six upstream lines, because deciding
  whether this is one of orbit's types means resolving the type the same way
  upstream does. That is the irreducible cost of adding a branch to a dispatch
  from outside; the alternative is copying the whole dispatch.

* ``default_async_rm`` -- the rule-based/remote entry point with
  ``--custom-rm-path`` deliberately bypassed, so a custom rm that has hijacked
  the reward slot for a non-reward transport (OPD teacher scoring) can still ask
  for the real task reward. Upstream has no such entry point, so this is a LIFT;
  its only caller is orbit/opd/opd_sglang.py, so it is NOT imported back into
  miles.

Nothing here imports miles at module scope -- ``import orbit`` executes this
module and must stay cheap (see orbit/patch/runtime.py).
"""

from __future__ import annotations

from orbit.patch import original, patch_function

_RM_HUB = "miles.rollout.rm_hub"

_REASON = (
    "orbit grades two reward types upstream has no spelling for: gemma_math "
    "(Gemma-4 closes thinking with <channel|>, not </think>) and math_alignment "
    "(dataset-specific aligned grading used by the eval suites)"
)

# Returned by the orbit dispatch when this sample is upstream's business.
_NOT_ORBITS = object()


class _NoCustomRM:
    """A read-only view of ``args`` whose ``custom_rm_path`` is ``None``.

    ``default_async_rm`` has to reach the rule-based dispatch while a custom rm
    is configured -- it is called FROM that custom rm. A view rather than a copy
    because ``args`` is the whole training namespace: copying it would be both
    wasteful and a second object that can drift from the real one mid-run.
    """

    __slots__ = ("_args",)

    custom_rm_path = None

    def __init__(self, args):
        self._args = args

    def __getattr__(self, name):
        return getattr(self._args, name)


def _orbit_rm_types(args, sample):
    """Grade the sample if its rm_type is one of orbit's; else ``_NOT_ORBITS``.

    The resolution order (metadata over args, ``boxed_`` prefix stripped after
    extracting the boxed answer) is upstream's, reproduced so orbit's two types
    behave exactly like the ones next to them -- ``boxed_gemma_math`` works for
    the same reason ``boxed_math`` does.
    """
    from miles.rollout.rm_hub.math_utils import extract_answer as extract_boxed_answer

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    rm_type = (metadata.get("rm_type") or args.rm_type or "").strip()
    response = sample.response
    label = sample.label
    if rm_type.startswith("boxed_"):
        response = extract_boxed_answer(response) or ""
        rm_type = rm_type[len("boxed_") :]

    if rm_type == "gemma_math":
        from orbit.rewards.gemma_math import get_gemma_math_reward

        return get_gemma_math_reward(response, label)
    if rm_type == "math_alignment":
        from orbit.rewards.math_alignment import grade_math_alignment

        return 1 if grade_math_alignment(response, label, metadata) else 0
    return _NOT_ORBITS


@patch_function(
    "miles.rollout.rm_hub",
    "async_rm",
    upstream_sha="42645f44bbc88724aad10e887f5d6ad02bbf3d7c8ce19afad82a3821fb715065",
    reason=_REASON,
)
async def async_rm(args, sample, **kwargs):
    # `args.custom_rm_path` unguarded, as upstream reads it: a namespace without
    # the attribute must still fail here the way it fails upstream, and a custom
    # rm still wins over every rule-based type including orbit's two.
    if args.custom_rm_path is None:
        reward = _orbit_rm_types(args, sample)
        if reward is not _NOT_ORBITS:
            return reward
    return await original(_RM_HUB, "async_rm")(args, sample, **kwargs)


async def default_async_rm(args, sample):
    """The rule-based/remote RM dispatch, bypassing any --custom-rm-path.

    Exposed so custom rms that hijack the reward slot for non-reward transports
    (e.g. OPD teacher scoring) can still hand evaluation samples to the real
    task reward.

    Goes back through ``async_rm`` -- looked up on the module, so it is orbit's
    patched one -- rather than reimplementing the dispatch, which keeps this a
    ten-line lift instead of a second copy of the reward table.
    """
    from miles.rollout import rm_hub

    return await rm_hub.async_rm(_NoCustomRM(args), sample)
