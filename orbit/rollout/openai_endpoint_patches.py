"""Orbit's policy-version recording on the OpenAI-endpoint rollout path.

Upstream records the rollout's policy version as whatever
``meta_info["weight_version"]`` happened to hold. Orbit's true-on-policy contract
needs the NORMALIZED version instead (miles/utils/types.py::
``_extract_policy_version``): adapter-only rollouts report ``adapter_version``
and no ``weight_version`` at all, the two must agree when both are present, and
the value is compared as a string. ``Sample.update_from_meta_info`` already reads
it that way; this is the same read on the endpoint path.

Mechanism: a LIFT plus a DELEGATING patch. ``_record_policy_version`` did not
exist upstream, so it simply moved here. The caller is upstream's 40-line
``_compute_sample_from_openai_record``, of which orbit owned two lines -- far too
little to justify copying the body -- so orbit lets upstream build the whole
sample and then corrects the one field it disagrees about.

Why the correction is exact rather than approximate: upstream's ``sample`` is a
``deepcopy(input_sample)``, and the ONLY statement in its body that touches
``weight_versions`` is the single append at the end. So truncating the list back
to the length the input carried removes precisely upstream's entry and nothing a
caller put there earlier. If upstream ever appends twice, the truncation still
holds (it is length-based, not pop-based) and the pin catches the body change
regardless.

Nothing here imports miles at module scope: ``import orbit`` executes this module
and must stay cheap (see orbit/patch/runtime.py).
"""

from __future__ import annotations

from orbit.patch import original, patch_function

_OPENAI_ENDPOINT = "miles.rollout.generate_utils.openai_endpoint_utils"

_REASON = (
    "orbit's true-on-policy contract records the normalized policy version "
    "(adapter_version-or-weight_version, string-compared, and an error when the "
    "two disagree); upstream appends the raw meta_info['weight_version'] and has "
    "no spelling for an adapter-only rollout, which reports no weight_version"
)


def record_policy_version(sample, meta_info: dict) -> None:
    """Append the normalized policy version, if the rollout reported one.

    Lifted verbatim out of the vendored module, where it had no upstream
    counterpart. ``_extract_policy_version`` is imported at call time so a
    rename upstream fails loudly here instead of at ``import orbit``.
    """
    from miles.utils.types import _extract_policy_version

    version = _extract_policy_version(meta_info)
    if version is not None:
        sample.weight_versions.append(version)


@patch_function(
    "miles.rollout.generate_utils.openai_endpoint_utils",
    "_compute_sample_from_openai_record",
    upstream_sha="e80378fd6a22126c1c639fc2f844e8531776dc1b96ae703fa481e6a93062f609",
    reason=_REASON,
)
def _compute_sample_from_openai_record(args, input_sample, record, tokenizer, trim_count=0):
    """Upstream's sample, with the policy version re-read orbit's way."""
    sample = original(_OPENAI_ENDPOINT, "_compute_sample_from_openai_record")(
        args, input_sample, record, tokenizer, trim_count
    )
    meta_info = record.response["choices"][0]["meta_info"]
    if sample.weight_versions:
        # Drop whatever upstream appended (see module docstring: at most one
        # entry, on top of the input's own list) before recording orbit's.
        del sample.weight_versions[len(input_sample.weight_versions or ()) :]
    record_policy_version(sample, meta_info)
    return sample
