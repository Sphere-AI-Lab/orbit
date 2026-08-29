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
