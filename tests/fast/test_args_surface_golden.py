"""Argument-surface golden (Phase 2 arguments.py registration refactor, gate 1).

`tools/dump_args_surface.py` builds the full training-argument parser --
megatron's own arguments plus every orbit-added argument from
`miles.utils.arguments.get_orbit_extra_args_provider` -- exactly the way
`miles.utils.arguments.parse_args()` does, and dumps every `argparse` action
to a record. Pinning that surface here, before
`docs/superpowers/plans/2026-08-29-phase2-arguments-registration.md` moves the
orbit-added argument definitions out of `miles/utils/arguments.py` into
`miles/orbit/arguments.py`, gives that refactor a total-surface equivalence check:
group/help ordering inside the Python source may shift freely, but a
moved/renamed/retyped/re-defaulted argument must show up here.
"""

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "dump_args_surface.py"
GOLDEN_PATH = REPO_ROOT / "tests" / "fast" / "args_surface_golden.json"

SPEC = importlib.util.spec_from_file_location("orbit_dump_args_surface", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
DUMP_ARGS_SURFACE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DUMP_ARGS_SURFACE
SPEC.loader.exec_module(DUMP_ARGS_SURFACE)


def test_golden_is_populated():
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert len(golden) > 500, "argument-surface golden looks truncated"


def test_live_argument_surface_matches_the_golden():
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    live = DUMP_ARGS_SURFACE.dump_records()
    diff = DUMP_ARGS_SURFACE.diff_records(golden, live)
    assert not diff, (
        f"live argument surface ({len(live)} actions) differs from "
        f"{GOLDEN_PATH} ({len(golden)} actions):\n" + diff
    )
