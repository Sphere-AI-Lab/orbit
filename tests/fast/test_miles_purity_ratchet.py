"""Ratchet: orbit's entanglement with its miles fork base only shrinks.

Every file orbit shares with radixark/miles at the fork base is recorded in
miles_purity_manifest.json as either pristine (identical modulo the mechanical
miles->orbit rename) or budgeted (carries orbit modifications, hash-pinned).
Editing a pristine file, or editing a budgeted file without regenerating the
manifest, fails here — so growing the fork delta is always a deliberate,
reviewed act, never drift. See tools/miles_purity.py for regeneration.
"""

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "miles_purity.py"
MANIFEST_PATH = REPO_ROOT / "tests" / "fast" / "miles_purity_manifest.json"

SPEC = importlib.util.spec_from_file_location("orbit_miles_purity", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MILES_PURITY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MILES_PURITY
SPEC.loader.exec_module(MILES_PURITY)


def test_manifest_exists_and_is_populated():
    manifest = json.loads(MANIFEST_PATH.read_text())
    assert manifest["miles_base"] == MILES_PURITY.MILES_BASE
    assert len(manifest["pristine"]) >= 100, "pristine set shrank suspiciously"
    assert manifest["budgeted"], "budgeted set empty; manifest looks truncated"


def test_no_new_entanglement_with_miles_base():
    manifest = json.loads(MANIFEST_PATH.read_text())
    errors = MILES_PURITY.check(manifest)
    assert not errors, (
        "miles-shared files drifted from the recorded state:\n  "
        + "\n  ".join(errors)
    )


def test_no_orbit_code_outside_the_home_layer():
    """New orbit files inside the vendored miles tree belong under orbit/."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    violations = MILES_PURITY.home_violations(manifest)
    assert not violations, "\n  ".join(violations)


# Seam files whose entire delta is non-functional (comment-policy rewording,
# one-line identity docstrings, CI shard lists) — no ORBIT-SEAM mark required.
COSMETIC_ONLY = {
    ".github/workflows/pr-test.yml",
    ".github/workflows/pr-test.yml.j2",
    "examples/__init__.py",
    "miles/backends/training_utils/ci_utils.py",
    "miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_fp8.py",
    "miles/backends/megatron_utils/megatron_to_hf/processors/quantizer_mxfp8.py",
    "miles/rollout/base_types.py",
    "miles/rollout/generate_hub/__init__.py",
    "miles/rollout/generate_hub/benchmarkers.py",
    "miles/rollout/session/linear_trajectory.py",
    "miles/utils/env_report.py",
    "miles/utils/eval_config.py",
    "miles/utils/iter_utils.py",
    "miles/utils/profile_utils.py",
    "miles/utils/tracking_utils.py",
    "miles_plugins/__init__.py",
    "miles_plugins/models/glm5/glm5.py",
    "miles_plugins/models/hf_attention.py",
    "tests/__init__.py",
    "tools/__init__.py",
}


def test_every_functional_seam_is_stamped():
    """Every budgeted code file must carry at least one ORBIT-SEAM mark naming
    why the miles file is modified, unless its delta is purely cosmetic
    (allowlisted above). As of Phase 3 the whole fork delta is annotated, so
    there is no size threshold: `git grep ORBIT-SEAM` is the complete seam
    inventory."""
    manifest = json.loads(MANIFEST_PATH.read_text())
    unstamped = []
    for path, entry in manifest["budgeted"].items():
        if path in COSMETIC_ONLY:
            continue
        # Adapted upstream TEST files document divergences with `# orbit:` comments;
        # ORBIT-SEAM stamps are the contract for vendored SOURCE only.
        if path.startswith("tests/"):
            continue
        p = REPO_ROOT / path
        if p.suffix not in (".py", ".yml", ".yaml", ".j2", ".sh"):
            continue
        if "ORBIT-SEAM" not in p.read_text(errors="surrogateescape"):
            unstamped.append(path)
    assert not unstamped, (
        "seam files without an ORBIT-SEAM mark (stamp the hunk or add to "
        "COSMETIC_ONLY):\n  " + "\n  ".join(sorted(unstamped))
    )
