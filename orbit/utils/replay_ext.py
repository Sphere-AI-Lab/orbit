"""Orbit's replayed-routing sanitiser, moved out of ``miles/utils/replay_base.py``.

Two pieces used to live inside that vendored file, and both are expressed from
here now, which is what makes it byte-pristine again:

* ``_sanitize_replay_top_indices`` -- a whole function orbit added. LIFTED here
  verbatim.
* ``BaseReplayManager.get_topk_fn`` -- six upstream lines inside it were swapped
  for a call to that function. Orbit carries the method body and installs it on
  the vendored class through an import-time seam (orbit/patch/on_import.py).

Why the method is REPLACED rather than delegated to
--------------------------------------------------
The change is six lines deep inside ``_get_replay_result``, a closure built by
``get_topk_fn`` and never exposed. Upstream repairs a padded (-1) routing slot
IN PLACE and then does ``scores.gather(1, top_indices)`` with the repaired ids,
so there is no seam either side of it: sanitising the returned indices
afterwards would leave the probabilities gathered against upstream's ids. Orbit
therefore owns the whole 46-line body -- the one case the mechanism rules call
"delegation genuinely cannot express it".

Why a seam on the CLASS rather than ``patch_function``
------------------------------------------------------
``patch_function`` swaps a module-level attribute; this is a method on a
vendored class, and a mixin cannot be used either because that would mean
editing the vendored class's bases (the file would not be pristine). So the seam
sets the attribute on the class at the moment the vendored module is first
imported -- the same moment, and the same arming rule (``import orbit``), as
every other orbit seam.

That means the pin has to be carried here rather than by
tools/check_patch_pins.py, which only reads ``@patch_function`` declarations.
``install_replay_sanitizer`` verifies the upstream body's hash before it swaps,
so an upstream rewrite of ``get_topk_fn`` aborts loudly instead of silently
running orbit's stale copy of a body that has moved on; the same hash is
re-derived statically from the vendored source in
tests/fast/test_replay_seams.py, which is what puts the check in the CPU gate.

Nothing here imports torch or miles at module scope -- ``import orbit`` executes
this module and must stay cheap (see orbit/patch/runtime.py).
"""

from __future__ import annotations

import hashlib
import inspect

from orbit.patch import UpstreamDrift
from orbit.patch.on_import import on_import
from orbit.patch.runtime import normalize

_REPLAY_BASE = "miles.utils.replay_base"

# Hash of upstream ``BaseReplayManager.get_topk_fn``, the body orbit replaces.
# Re-derive it with tests/fast/test_replay_seams.py, which prints the current
# value when it disagrees.
GET_TOPK_FN_UPSTREAM_SHA = "b0cbb1548f5335c1c5cd50eede476d6000e0c96c7e0ebf5559127d55d70532a7"

_REASON = (
    "upstream repairs a padded (-1) replay slot only when the token's WHOLE topk "
    "row is padded, so a partially padded row keeps its -1 and a negative expert "
    "id reaches Megatron's sparse dispatch map; and the repair it does apply, "
    "`arange(n) % num_experts`, can hand a token the same expert twice. Either "
    "way the dispatch map builds a route the model never produced. Orbit "
    "validates the replayed ids and repairs each padded slot with an expert that "
    "does not collide with the rest of that token's row"
)


def method_sha(source: str) -> str:
    """Hash of a method body, agreeing across ``inspect`` and ``ast``.

    ``orbit.patch.runtime.normalize`` slices from the ``def`` line onward but
    keeps that line's indentation, which is empty for the module-level functions
    ``patch_function`` handles and four spaces for a method read with
    ``inspect.getsource``. ``ast.get_source_segment`` -- what the static check
    uses -- dedents only the first line. Stripping the first line here is what
    makes the runtime and the static check produce the same digest, exactly as
    check_patch_pins.py's decorator exclusion does for the function patches.
    """
    lines = normalize(source).split("\n")
    if lines:
        lines[0] = lines[0].lstrip()
    return hashlib.sha256(
        "\n".join(lines).encode("utf-8", "surrogateescape")
    ).hexdigest()


def _sanitize_replay_top_indices(top_indices, num_experts: int):
    """Keep replayed MoE routes valid for Megatron's sparse dispatch map."""
    import torch

    if top_indices.numel() == 0:
        return top_indices
    num_tokens, topk = top_indices.shape
    if topk > num_experts:
        raise ValueError(
            f"replay topk ({topk}) cannot be represented with {num_experts} experts"
        )

    out_of_range = top_indices >= num_experts
    if bool(out_of_range.any().item()):
        raise ValueError("replay top_indices contain expert ids out of range")

    valid = top_indices >= 0
    if topk > 1:
        duplicate_pairs = top_indices.unsqueeze(2) == top_indices.unsqueeze(1)
        duplicate_pairs &= valid.unsqueeze(2) & valid.unsqueeze(1)
        duplicate_pairs = duplicate_pairs.triu(diagonal=1)
        if bool(duplicate_pairs.any().item()):
            raise ValueError("replay top_indices contain duplicate expert ids")

    padding_mask = top_indices < 0
    if not bool(padding_mask.any().item()):
        return top_indices

    sanitized = top_indices.clone()
    device = sanitized.device
    dtype = sanitized.dtype
    expert_ids = torch.arange(num_experts, device=device, dtype=dtype).unsqueeze(0)
    row_offsets = torch.arange(num_tokens, device=device, dtype=dtype).unsqueeze(1) * topk

    for col in range(topk):
        current = sanitized[:, col]
        repair = current < 0
        if not bool(repair.any().item()):
            continue

        candidates = (expert_ids + row_offsets + col) % num_experts
        conflicts = ((candidates.unsqueeze(1) == top_indices.unsqueeze(2)) & valid.unsqueeze(2)).any(dim=1)
        if col > 0:
            conflicts |= (candidates.unsqueeze(1) == sanitized[:, :col].unsqueeze(2)).any(dim=1)
        replacement_pos = (~conflicts).int().argmax(dim=1)
        replacements = candidates.gather(1, replacement_pos.unsqueeze(1)).squeeze(1)
        sanitized[:, col] = torch.where(repair, replacements, current)

    return sanitized


def get_topk_fn(self, old_topk_fn, return_probs):
    """Upstream's ``BaseReplayManager.get_topk_fn`` with orbit's repair.

    Byte-for-byte upstream's body apart from the marked line: everything about
    which stage records, replays or falls through stays upstream's, so the pin
    above is the thing that has to be reviewed when upstream touches it.
    """
    from miles.utils.replay_base import _get_rank

    manager = self

    def _get_replay_result(top_indices, scores, topk, *args, **kwargs):
        assert (
            top_indices.shape[0] == scores.shape[0]
        ), f"rank {_get_rank()}: replay n_tokens {top_indices.shape[0]} does not match scores n_tokens {scores.shape[0]}"

        assert (
            top_indices.shape[1] == topk
        ), f"replay topk does not match expected topk, replay topk {top_indices.shape[1]}, topk {topk}"

        if self.enable_check_replay_result:
            self.check_replay_result(old_topk_fn, scores, topk, top_indices, *args, **kwargs)

        # ORBIT: replaces upstream's inline repair -- on this miles base that is
        # "rows that are ENTIRELY -1 get arange % num_experts", which leaves a
        # partially padded row still holding -1. See _REASON above.
        top_indices = _sanitize_replay_top_indices(top_indices, scores.shape[1])

        if return_probs:
            return scores.gather(1, top_indices), top_indices
        else:
            return top_indices

    def new_topk_fn(scores, topk, *args, **kwargs):
        if not manager.enabled:
            return old_topk_fn(scores, topk, *args, **kwargs)

        stage = manager.stage
        replay = manager.get_current()

        if stage == "fallthrough":
            return old_topk_fn(scores, topk, *args, **kwargs)

        elif stage == "record":
            result = old_topk_fn(scores, topk, *args, **kwargs)
            if return_probs:
                probs, top_indices = result
            else:
                top_indices = result
            replay.record(top_indices)
            return result

        elif stage == "replay_forward":
            return _get_replay_result(replay.pop_forward(), scores, topk, *args, **kwargs)

        elif stage == "replay_backward":
            return _get_replay_result(replay.pop_backward(), scores, topk, *args, **kwargs)

        else:
            return old_topk_fn(scores, topk, *args, **kwargs)

    return new_topk_fn


def install_replay_sanitizer() -> bool:
    """Put orbit's ``get_topk_fn`` on the vendored class. Returns whether it swapped.

    Idempotent, and pinned: the upstream body is hashed before the swap so an
    upstream rewrite stops the process instead of leaving orbit's stale copy
    quietly in charge of MoE routing replay.
    """
    from miles.utils.replay_base import BaseReplayManager

    current = BaseReplayManager.__dict__.get("get_topk_fn")
    if current is None:
        raise UpstreamDrift(
            f"{_REPLAY_BASE}.BaseReplayManager.get_topk_fn: orbit replaces this "
            f"method but the class no longer defines it. Orbit's replacement "
            f"exists because: {_REASON}. Find where the behaviour moved, then "
            f"re-point or retire the seam."
        )
    if getattr(current, "__module__", None) == __name__:
        return False  # already installed in this process

    actual = method_sha(inspect.getsource(current))
    if actual != GET_TOPK_FN_UPSTREAM_SHA:
        raise UpstreamDrift(
            f"{_REPLAY_BASE}.BaseReplayManager.get_topk_fn: upstream's body "
            f"changed (pinned {GET_TOPK_FN_UPSTREAM_SHA[:12]}, found "
            f"{actual[:12]}). Orbit replaces it because: {_REASON}. Review "
            f"whether the new upstream body changes that reasoning, then update "
            f"GET_TOPK_FN_UPSTREAM_SHA in {__name__} together with orbit's copy "
            f"of the body."
        )

    BaseReplayManager._orbit_unpatched_get_topk_fn = current
    BaseReplayManager.get_topk_fn = get_topk_fn
    return True


# The vendored class used to carry orbit's version directly; install it at the
# moment that module is first executed, which is when the edited body used to
# come into existence.
on_import(_REPLAY_BASE, install_replay_sanitizer)
