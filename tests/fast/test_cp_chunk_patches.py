"""orbit/megatron/training_utils_patches.py: padded-batch CP chunk sizing.

Both patches say the same thing -- "a thd batch given an explicit max_seq_len is
chunked against that padded length" -- by delegating through upstream's own
padded branch. So every test comes in a pair: orbit's chunking happens, AND the
untouched cases still come out of `_orbit_unpatched_*`. The second half is what
proves nothing was copied; if it ever fails because someone inlined upstream's
arithmetic here, upstream's fixes to it stop reaching us silently.
"""

import pytest

torch = pytest.importorskip("torch")

import orbit  # noqa: F401  -- importing orbit installs the patches
from miles.backends.training_utils import cp_utils, parallel


def _state(cp_size, cp_rank=0):
    trivial = parallel.GroupInfo(rank=0, size=1, group=None)
    cp = parallel.GroupInfo(rank=cp_rank, size=cp_size, group=None)
    return parallel.ParallelState(intra_dp=trivial, intra_dp_cp=cp, cp=cp, tp=trivial)


@pytest.fixture
def cp(request):
    """Install a ParallelState for this test and put the old one back."""
    previous = parallel._parallel_state
    parallel.set_parallel_state(_state(*request.param))
    yield
    parallel.set_parallel_state(previous) if previous is not None else None


def _cp2(fn):
    return pytest.mark.parametrize("cp", [(2,)], indirect=True)(fn)


def _cp1(fn):
    return pytest.mark.parametrize("cp", [(1,)], indirect=True)(fn)


def test_both_patches_are_actually_installed():
    for name in ("get_logits_and_tokens_offset_with_cp", "slice_with_cp"):
        assert getattr(cp_utils, name).__module__ == "orbit.megatron.training_utils_patches"
        assert hasattr(cp_utils, f"_orbit_unpatched_{name}"), (
            "the pristine upstream function must be kept so the patch can delegate"
        )


# ---------------------------------------------------------------------------
# get_logits_and_tokens_offset_with_cp
# ---------------------------------------------------------------------------


@_cp2
def test_thd_chunks_follow_the_padded_length(cp):
    """11 real tokens padded to 16 across 2 CP ranks: chunks of 4, not of 3."""
    chunk_size, _, _, _ = cp_utils.get_logits_and_tokens_offset_with_cp(
        11, 7, "thd", max_seq_len=16
    )
    assert chunk_size == 4

    # ...and prove the patch is what did it: upstream alone chunks by 11.
    upstream, *_ = cp_utils._orbit_unpatched_get_logits_and_tokens_offset_with_cp(
        11, 7, "thd", max_seq_len=16
    )
    assert upstream == 3


@_cp2
def test_thd_without_a_max_seq_len_is_exactly_upstreams_result(cp):
    """The delegation property for the unpadded case."""
    patched = cp_utils.get_logits_and_tokens_offset_with_cp(11, 7, "thd")
    upstream = cp_utils._orbit_unpatched_get_logits_and_tokens_offset_with_cp(11, 7, "thd")
    assert patched == upstream
    assert patched[0] == 3


@_cp2
def test_bshd_is_exactly_upstreams_result(cp):
    patched = cp_utils.get_logits_and_tokens_offset_with_cp(11, 7, "bshd", max_seq_len=16)
    upstream = cp_utils._orbit_unpatched_get_logits_and_tokens_offset_with_cp(
        11, 7, "bshd", max_seq_len=16
    )
    assert patched == upstream


@_cp2
def test_padded_thd_and_bshd_agree(cp):
    """Restating the mechanism as a property: for a padded batch the two layouts
    must produce the same offsets, because they are the same slicing."""
    assert cp_utils.get_logits_and_tokens_offset_with_cp(
        11, 7, "thd", max_seq_len=16
    ) == cp_utils.get_logits_and_tokens_offset_with_cp(11, 7, "bshd", max_seq_len=16)


@_cp2
def test_a_max_seq_len_shorter_than_the_sequence_is_rejected(cp):
    """A too-small max_seq_len yields a negative pad and a silently truncated
    slice downstream, so it must be loud here."""
    with pytest.raises(AssertionError, match="max_seq_len must be >= total_length"):
        cp_utils.get_logits_and_tokens_offset_with_cp(11, 7, "thd", max_seq_len=8)


# ---------------------------------------------------------------------------
# slice_with_cp
# ---------------------------------------------------------------------------


@_cp2
def test_thd_slices_against_the_padded_length(cp):
    tokens = torch.arange(11)

    out = cp_utils.slice_with_cp(tokens, 0, "thd", max_seq_len=16)

    # chunk 4: rank 0 takes [0:4] and [12:16] of the 16-long padded sequence.
    assert out.tolist() == [0, 1, 2, 3, 0, 0, 0, 0]
    # ...and prove the patch is what did it: upstream chunks by 11 -> 6 tokens.
    upstream = cp_utils._orbit_unpatched_slice_with_cp(tokens, 0, "thd", max_seq_len=16)
    assert upstream.tolist() == [0, 1, 2, 9, 10, 0]


@_cp2
def test_thd_without_a_max_seq_len_is_exactly_upstreams_slice(cp):
    tokens = torch.arange(11)
    assert torch.equal(
        cp_utils.slice_with_cp(tokens, 0, "thd"),
        cp_utils._orbit_unpatched_slice_with_cp(tokens, 0, "thd"),
    )


@_cp2
def test_bshd_is_exactly_upstreams_slice(cp):
    tokens = torch.arange(11)
    assert torch.equal(
        cp_utils.slice_with_cp(tokens, 0, "bshd", max_seq_len=16),
        cp_utils._orbit_unpatched_slice_with_cp(tokens, 0, "bshd", max_seq_len=16),
    )


@_cp1
def test_with_cp_off_a_padded_thd_batch_is_still_not_padded(cp):
    """The carve-out. Upstream's bshd path pads up to max_seq_len when cp_size
    is 1; thd does not, so the patch must not re-route there. Routing through
    bshd unconditionally would return 16 tokens instead of 11."""
    tokens = torch.arange(11)

    out = cp_utils.slice_with_cp(tokens, 0, "thd", max_seq_len=16)

    assert torch.equal(out, tokens)
    assert len(cp_utils._orbit_unpatched_slice_with_cp(tokens, 0, "bshd", max_seq_len=16)) == 16


@_cp1
def test_a_max_seq_len_shorter_than_the_tokens_is_rejected(cp):
    with pytest.raises(AssertionError, match="max_seq_len must be >= token length"):
        cp_utils.slice_with_cp(torch.arange(11), 0, "thd", max_seq_len=8)


# ---------------------------------------------------------------------------
# Import order
# ---------------------------------------------------------------------------


def test_patch_survives_cp_utils_being_imported_before_orbit():
    """Import ORDER must not decide whether the patch takes effect.

    `miles/backends/training_utils/data.py` does `from .cp_utils import
    slice_with_cp` and imports nothing from orbit, so a process that reaches
    data.py first bound upstream's function into data's namespace before the
    hook was ever armed. Patching the cp_utils attribute afterwards leaves
    data.py calling upstream -- silently, with no error and no visible sign,
    which is the exact failure orbit/patch/runtime.py exists to remove.
    Measured before the sweep in `_repoint_reexports` was widened past parent
    packages: this printed the vendored module.

    Run in a subprocess so the import order is real rather than simulated.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    probe = (
        "import miles.backends.training_utils.data as data\n"  # BEFORE orbit
        "import orbit\n"
        "print(data.slice_with_cp.__module__)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repo,
        capture_output=True,
        text=True,
        # Inherit the environment: torch needs the CUDA and loader paths the
        # activated env sets. Only the two knobs this probe cares about are forced.
        env={**os.environ, "PYTHONPATH": str(repo), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("orbit.megatron.training_utils_patches"), (
        f"module-level re-export was not re-pointed: {out.stdout.strip()!r}"
    )
