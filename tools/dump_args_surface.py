"""Dump (or check) the total training-argument-parser surface, as JSON.

Gate 1 of docs/superpowers/plans/2026-08-29-phase2-arguments-registration.md:
a golden snapshot of every `argparse` action the production parser ends up
with -- megatron's own arguments plus every orbit-added argument from
`miles.utils.arguments.get_orbit_extra_args_provider` -- built the same way
`miles.utils.arguments.parse_args()` builds it (via
`megatron.training.arguments.parse_args(extra_args_provider=add_orbit_arguments)`),
minus the final `parser.parse_args()` call against real argv. The refactor this
gate exists to protect (arguments.py registration split, `miles/orbit/arguments.py`)
must reproduce this surface exactly; group/help-string reflow inside the
Python source is fine, a moved/renamed/retyped/re-defaulted argument is not.

Usage:
    dump_args_surface.py --write GOLDEN_PATH   # (re)generate the golden file
    dump_args_surface.py --check GOLDEN_PATH   # compare live surface to it; exit 1 on any diff
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Fields recorded per parser action, in the order the plan specifies them.
_FIELDS = ("option_strings", "dest", "default", "type", "choices", "nargs", "required", "help")


def _type_name(action: argparse.Action) -> str | None:
    """`repr()` of a bound method / lambda / type object carries a memory
    address, which would make the golden non-reproducible across runs. Use
    `__name__` (stable, source-derived) instead, falling back to the class
    name for callable instances (e.g. `argparse.FileType(...)`) that lack one.
    """
    t = getattr(action, "type", None)
    if t is None:
        return None
    name = getattr(t, "__name__", None)
    if name:
        return name
    return type(t).__name__


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _sort_key(value):
    return (type(value).__name__, str(value))


def _record_for_action(action: argparse.Action) -> dict:
    choices = action.choices
    if choices is not None:
        # `choices` is sometimes a plain `set` literal in upstream code (e.g.
        # sglang's `server_args.py` SAMPLING_BACKEND_CHOICES). Set iteration
        # order is randomized per-process (PYTHONHASHSEED), so without
        # sorting here the same source produces a different `choices` list
        # order on every run -- noise, not a real surface change. What's
        # semantically meaningful is the *set* of legal choices, so sort it.
        choices = sorted((_json_safe(c) for c in choices), key=_sort_key)
    return {
        "option_strings": list(action.option_strings),
        "dest": action.dest,
        "default": repr(action.default),
        "type": _type_name(action),
        "choices": choices,
        "nargs": action.nargs,
        "required": bool(action.required),
        "help": action.help,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the full training-argument parser exactly the way
    `miles.utils.arguments.parse_args()` does when it calls
    `megatron.training.arguments.parse_args(extra_args_provider=add_orbit_arguments)`
    -- i.e. megatron's own `add_megatron_arguments(parser)` followed by orbit's
    extra-args provider -- but stop short of `parser.parse_args()` itself, since
    this script has no real training-launch argv to parse.

    Two ambient toggles that gate *conditional* argument registration inside
    the orbit provider are pinned for determinism, independent of the
    invoking shell:

    - `sys.argv`: `add_sglang_tp_size()` (always) and, if the experimental
      rollout refactor is on, `add_user_provided_function_arguments()` both
      call `parser.parse_known_args()` with no explicit args, i.e. against
      `sys.argv` -- which would otherwise be pytest's argv or this script's
      own `--write/--check` flags. Neutralized to just the program name.
    - `MILES_EXPERIMENTAL_ROLLOUT_REFACTOR`: forced off (its documented
      default). On, `add_user_provided_function_arguments()` calls
      `load_function()` on `--rollout-function-path` /
      `--custom-generate-function-path` and imports whatever module that
      path names, which is not part of the stable default surface this gate
      pins (mirrors the fixture in tests/fast/utils/test_peft_arguments.py).
    """
    # Heavy (torch/megatron) imports live inside the function so merely
    # importing this module (e.g. from the pytest golden test) stays light
    # until a surface is actually requested.
    from megatron.training.arguments import add_megatron_arguments

    from miles.utils.arguments import get_orbit_extra_args_provider

    parser = argparse.ArgumentParser(description="Megatron-LM Arguments", allow_abbrev=False)
    parser = add_megatron_arguments(parser)

    add_orbit_arguments = get_orbit_extra_args_provider()

    old_argv = sys.argv
    old_env = os.environ.get("MILES_EXPERIMENTAL_ROLLOUT_REFACTOR")
    try:
        sys.argv = [old_argv[0] if old_argv else "dump_args_surface"]
        os.environ["MILES_EXPERIMENTAL_ROLLOUT_REFACTOR"] = "0"
        parser = add_orbit_arguments(parser)
    finally:
        sys.argv = old_argv
        if old_env is None:
            os.environ.pop("MILES_EXPERIMENTAL_ROLLOUT_REFACTOR", None)
        else:
            os.environ["MILES_EXPERIMENTAL_ROLLOUT_REFACTOR"] = old_env

    return parser


def dump_records(parser: argparse.ArgumentParser | None = None) -> list[dict]:
    """Every action on `parser` (built via `build_parser()` if not given), as
    JSON-safe records, sorted by dest then option_strings."""
    if parser is None:
        parser = build_parser()
    records = [_record_for_action(action) for action in parser._actions]
    records.sort(key=lambda r: (r["dest"], r["option_strings"]))
    # Normalize through JSON so a freshly-built record and one read back from
    # a written golden file always compare equal for equal content -- e.g. a
    # `help=(...)` tuple (present on at least one upstream action) round-trips
    # to a list from `json.loads`, and would otherwise diff false-positive
    # against a live record where it is still a tuple.
    return json.loads(json.dumps(records, sort_keys=True))


def write_golden(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, sort_keys=True, indent=1) + "\n", encoding="utf-8")


def _record_key(record: dict):
    return (record["dest"], tuple(record["option_strings"]))


def diff_records(old_records: list[dict], new_records: list[dict]) -> str:
    """Readable per-dest diff of added/removed/changed entries; empty string
    means no difference."""
    old_map = {_record_key(r): r for r in old_records}
    new_map = {_record_key(r): r for r in new_records}
    all_keys = sorted(set(old_map) | set(new_map))

    lines: list[str] = []
    for dest, option_strings in all_keys:
        old_r = old_map.get((dest, option_strings))
        new_r = new_map.get((dest, option_strings))
        if old_r is None:
            lines.append(f"+ added   dest={dest} option_strings={list(option_strings)}")
            for field in _FIELDS:
                if field in ("dest", "option_strings"):
                    continue
                lines.append(f"    {field}: {new_r[field]!r}")
        elif new_r is None:
            lines.append(f"- removed dest={dest} option_strings={list(option_strings)}")
            for field in _FIELDS:
                if field in ("dest", "option_strings"):
                    continue
                lines.append(f"    {field}: {old_r[field]!r}")
        elif old_r != new_r:
            lines.append(f"~ changed dest={dest} option_strings={list(option_strings)}")
            for field in _FIELDS:
                old_v = old_r.get(field)
                new_v = new_r.get(field)
                if old_v != new_v:
                    lines.append(f"    {field}: {old_v!r} -> {new_v!r}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", metavar="GOLDEN_PATH", help="(Re)generate the golden JSON file at this path.")
    group.add_argument(
        "--check", metavar="GOLDEN_PATH", help="Compare the live surface to this golden file; exit 1 on any diff."
    )
    args = parser.parse_args(argv)

    records = dump_records()

    if args.write:
        path = Path(args.write)
        write_golden(path, records)
        print(f"wrote {len(records)} argument records to {path}")
        return 0

    path = Path(args.check)
    golden = json.loads(path.read_text(encoding="utf-8"))
    diff = diff_records(golden, records)
    if diff:
        print(f"argument surface differs from {path} ({len(golden)} golden vs {len(records)} live records):")
        print(diff)
        return 1
    print(f"argument surface matches {path} ({len(records)} records)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
