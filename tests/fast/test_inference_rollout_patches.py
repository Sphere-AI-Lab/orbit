"""``compute_sampling_params`` takes per-call stop / min_new_tokens overrides.

Upstream builds the dict from ``args`` alone. Orbit's eval path configures stop
strings, stop token ids and a minimum response length PER EVAL DATASET, so the
signature GROWS -- and the patch must add exactly that and nothing else: the
args-derived defaults, the fixed keys and their order still come from upstream's
body.
"""

import argparse

import pytest

import orbit  # noqa: F401  -- importing orbit installs the patches
from miles.rollout.inference_rollout import inference_rollout_common as irc


def _args():
    return argparse.Namespace(
        rollout_stop=["</s>"],
        rollout_stop_token_ids=[2],
        rollout_skip_special_tokens=False,
    )


def _base_kwargs():
    return dict(temperature=0.7, top_p=0.9, top_k=20, max_new_tokens=128)


def test_the_patch_is_actually_installed():
    assert irc.compute_sampling_params.__module__ == "orbit.rollout.inference_rollout_patches"
    assert hasattr(irc, "_orbit_unpatched_compute_sampling_params"), (
        "the pristine upstream function must be kept so the patch can delegate"
    )


def test_without_overrides_the_result_is_byte_for_byte_upstreams():
    """The delegation property, including key ORDER: a caller that passes no
    override must not be able to tell the patch is there at all."""
    patched = irc.compute_sampling_params(_args(), **_base_kwargs())
    upstream = irc._orbit_unpatched_compute_sampling_params(_args(), **_base_kwargs())
    assert patched == upstream
    assert list(patched) == list(upstream)
    assert "min_new_tokens" not in patched


def test_the_overrides_replace_only_what_they_were_given():
    params = irc.compute_sampling_params(
        _args(),
        **_base_kwargs(),
        stop=["<|im_end|>"],
        stop_token_ids=[151645],
        min_new_tokens=16,
    )
    assert params["stop"] == ["<|im_end|>"]
    assert params["stop_token_ids"] == [151645]
    assert params["min_new_tokens"] == 16
    # Untouched keys still come from upstream's args-derived body.
    assert params["skip_special_tokens"] is False
    assert params["no_stop_trim"] is True
    assert params["temperature"] == 0.7


def test_overriding_one_key_leaves_the_others_on_the_args_defaults():
    params = irc.compute_sampling_params(_args(), **_base_kwargs(), min_new_tokens=1)
    assert params["stop"] == ["</s>"]
    assert params["stop_token_ids"] == [2]


def test_the_extra_parameters_are_orbits_not_upstreams():
    """Prove the signature growth is what the patch contributes."""
    with pytest.raises(TypeError):
        irc._orbit_unpatched_compute_sampling_params(_args(), **_base_kwargs(), min_new_tokens=1)


def test_the_eval_modules_re_export_is_re_pointed():
    """The "installed but bypassed" failure mode, for the one caller that needs
    the grown signature.

    miles/rollout/inference_rollout/inference_rollout_eval.py does
    `from ...inference_rollout_common import compute_sampling_params` at import
    time. If that binding is left pointing at upstream's function -- which is
    what happens without orbit/patch/runtime.py::_repoint_reexports when the eval
    module is imported before orbit -- every eval run raises TypeError on the
    per-dataset `stop=` it passes. Run in a subprocess so the import order is
    real rather than simulated.
    """
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    probe = (
        "import miles.rollout.inference_rollout.inference_rollout_eval as ev\n"  # BEFORE orbit
        "import orbit\n"
        "print(ev.compute_sampling_params.__module__)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=repo,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(repo), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("orbit.rollout.inference_rollout_patches"), (
        f"re-export was not re-pointed: {out.stdout.strip()!r}"
    )
