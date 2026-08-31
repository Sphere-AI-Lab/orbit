"""Orbit's reward-hub additions, expressed from orbit's side.

``miles/rollout/rm_hub/__init__.py`` used to carry two edits. Both are here now,
so that file is byte-pristine:

* one extra ``rm_type`` -- ``math_alignment`` (dataset-specific aligned grading
  used by the eval suites), which upstream has no spelling for. A DELEGATING
  patch: upstream still owns every other type and both NotImplementedError
  messages.
* ``default_async_rm`` -- a LIFT. It is the rule-based dispatch with any
  ``--custom-rm-path`` bypassed, so a custom rm that hijacks the reward slot for
  a non-reward transport (OPD teacher scoring) can still reach the task reward.
  Its only callers are orbit's.

Orbit's OTHER rm_type is gone from here, and that is the mechanism working:
miles @ dbbab1566 ships ``get_gemma_math_reward`` in ``rm_hub/deepscaler.py``
and dispatches ``gemma_math`` itself -- with the same "grade everything after
the last ``<channel|>``" semantics orbit had. orbit-main-isolated, on the older
base, still carries both the reward and the branch.

Nothing here imports torch or miles at module scope: ``import orbit`` executes
this module and must stay cheap (see orbit/patch/runtime.py).
"""

from __future__ import annotations

from orbit.patch import original, patch_function

_RM_HUB = "miles.rollout.rm_hub"

_REASON = (
    "orbit grades one reward type upstream has no spelling for: math_alignment "
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


class _NoCustomRMSpec:
    """A view of a ``reward_spec`` whose ``custom_rm_path`` is ``None``."""

    __slots__ = ("_spec",)

    custom_rm_path = None

    def __init__(self, spec):
        self._spec = spec

    def __getattr__(self, name):
        return getattr(self._spec, name)


class _NoCustomRMSample:
    """A view of a sample whose reward spec names no custom rm.

    Nulling ``args.custom_rm_path`` is not enough on this miles base: upstream's
    ``_resolve_reward_config`` takes the custom rm from the sample's
    ``reward_spec`` FIRST and only then falls back to args. A per-sample spec
    would therefore route straight back into the hijacking rm this function
    exists to bypass -- an infinite regress, not just a wrong reward.
    """

    __slots__ = ("_sample",)

    def __init__(self, sample):
        self._sample = sample

    @property
    def reward_spec(self):
        spec = self._sample.reward_spec
        return None if spec is None else _NoCustomRMSpec(spec)

    def __getattr__(self, name):
        return getattr(self._sample, name)


def _orbit_rm_types(args, sample):
    """Grade the sample if its rm_type is orbit's; else ``_NOT_ORBITS``.

    Resolution goes through upstream's own ``_resolve_reward_config`` rather than
    a reproduction of it, so orbit's type sees the same precedence (reward spec
    over sample metadata over args) as every type next to it, and inherits any
    later change to that order for free. Only the ``boxed_`` prefix handling is
    repeated, because upstream applies it after the custom-rm branch it owns.
    """
    from miles.rollout.rm_hub import _resolve_reward_config
    from miles.rollout.rm_hub.math_utils import extract_answer as extract_boxed_answer

    _, rm_type = _resolve_reward_config(args, sample)
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    response = sample.response
    label = sample.label
    if rm_type.startswith("boxed_"):
        response = extract_boxed_answer(response) or ""
        rm_type = rm_type[len("boxed_") :]

    if rm_type == "math_alignment":
        from orbit.rewards.math_alignment import grade_math_alignment

        return 1 if grade_math_alignment(response, label, metadata) else 0
    return _NOT_ORBITS


@patch_function(
    "miles.rollout.rm_hub",
    "async_rm",
    upstream_sha="fad83203e01d0da675493ccb79107bafc93392b1219f3340466f6db18fc36b32",
    reason=_REASON,
)
async def async_rm(args, sample, **kwargs):
    from miles.rollout.rm_hub import _resolve_reward_config

    # A custom rm still wins over every rule-based type, orbit's included --
    # which is upstream's order, read through upstream's own resolver so a reward
    # spec that names a custom rm is honoured here too.
    custom_rm_path, _ = _resolve_reward_config(args, sample)
    if custom_rm_path is None:
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

    return await rm_hub.async_rm(_NoCustomRM(args), _NoCustomRMSample(sample))
