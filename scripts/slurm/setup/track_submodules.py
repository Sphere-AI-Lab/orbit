#!/usr/bin/env python3
"""Show pin state of vendored submodules vs upstream branch HEAD, and bump on demand.

orbit' Dockerfile resolves Megatron-LM / sglang / Megatron-Bridge by *branch*
name (not commit), so every docker build picks up whatever's at branch HEAD at
build time. Our fork promotes these to git submodules so the parent repo's
tree locks each one to a specific commit — but that means we have to bump
manually to follow upstream.

Usage:
  # Status (uses cached `origin/<branch>` refs — fast, no network):
  python scripts/slurm/setup/track_submodules.py

  # Refresh remote refs first, then show status:
  python scripts/slurm/setup/track_submodules.py --fetch

  # Bump one submodule to its tracking branch HEAD (stages submodule pointer):
  python scripts/slurm/setup/track_submodules.py --bump thirdparty/Megatron-LM

  # Bump all of them:
  python scripts/slurm/setup/track_submodules.py --bump --all
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SUBMODULES = [
    "thirdparty/Megatron-LM",
    "thirdparty/sglang",
    "thirdparty/Megatron-Bridge",
]


def run(args: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        cwd=cwd or REPO_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def gitmodules_get(path: str, field: str) -> str | None:
    res = run(
        ["git", "config", "-f", ".gitmodules", "--get", f"submodule.{path}.{field}"],
        check=False,
    )
    return res.stdout.strip() or None


def commit_info(sub: Path, ref: str = "HEAD") -> tuple[str, str, str, str]:
    """Return (full_sha, short_sha, iso_date, subject)."""
    sha = run(["git", "rev-parse", ref], cwd=sub).stdout.strip()
    short = run(["git", "rev-parse", "--short", ref], cwd=sub).stdout.strip()
    date = run(["git", "log", "-1", "--format=%cs", ref], cwd=sub).stdout.strip()
    subj = run(["git", "log", "-1", "--format=%s", ref], cwd=sub).stdout.strip()
    return sha, short, date, subj


def count_between(sub: Path, base_ref: str, head_ref: str) -> int:
    out = run(["git", "rev-list", "--count", f"{base_ref}..{head_ref}"], cwd=sub).stdout.strip()
    return int(out)


def recent_log(sub: Path, base_ref: str, head_ref: str, limit: int = 5) -> list[str]:
    out = run(
        ["git", "log", f"{base_ref}..{head_ref}", "--oneline", "--no-merges", f"-{limit}"],
        cwd=sub,
    ).stdout
    return [line.rstrip() for line in out.splitlines() if line.strip()]


def fetch_remote(path: str) -> None:
    sub = REPO_ROOT / path
    print(f"[fetch] {path}", file=sys.stderr)
    run(["git", "fetch", "origin", "--quiet"], cwd=sub)


def status(do_fetch: bool) -> int:
    if do_fetch:
        for path in SUBMODULES:
            fetch_remote(path)

    any_behind = False
    for path in SUBMODULES:
        sub = REPO_ROOT / path
        branch = gitmodules_get(path, "branch")
        url = gitmodules_get(path, "url") or "?"

        print(path)
        print(f"  url     {url}")
        print(f"  branch  {branch or '(missing in .gitmodules)'}")

        if not (sub / ".git").exists():
            print("  state   NOT INITIALISED — run: git submodule update --init " + path)
            print()
            continue

        try:
            _, p_short, p_date, p_subj = commit_info(sub)
        except subprocess.CalledProcessError as e:
            print(f"  state   failed to read pinned commit: {e.stderr}")
            print()
            continue
        print(f"  pinned  {p_short}  {p_date}  {p_subj[:70]}")

        if not branch:
            print()
            continue

        remote_ref = f"origin/{branch}"
        try:
            r_sha, r_short, r_date, r_subj = commit_info(sub, remote_ref)
        except subprocess.CalledProcessError:
            print(f"  remote  no `{remote_ref}` ref locally — run with --fetch first")
            print()
            continue

        behind = count_between(sub, "HEAD", r_sha)
        ahead = count_between(sub, r_sha, "HEAD")
        if behind == 0 and ahead == 0:
            print(f"  remote  UP TO DATE ({r_short})")
        elif behind > 0:
            any_behind = True
            print(f"  remote  {r_short}  {r_date}  → {behind} commits behind")
            for line in recent_log(sub, "HEAD", r_sha):
                print(f"            {line}")
        else:
            print(f"  remote  {r_short} — local pin is {ahead} commits AHEAD of remote branch")
        print()

    if any_behind:
        print("To bump:")
        print("  python scripts/slurm/setup/track_submodules.py --bump --all")
        print("Inspect the resulting `git diff --submodule=log` before committing.")
    return 0


def bump(targets: list[str]) -> int:
    for t in targets:
        if t not in SUBMODULES:
            print(
                f"FATAL: {t!r} is not a tracked submodule. Options: {', '.join(SUBMODULES)}",
                file=sys.stderr,
            )
            return 2
    for t in targets:
        sub = REPO_ROOT / t
        before, _, _, _ = commit_info(sub)
        print(f"[bump] {t}: git submodule update --remote --recursive")
        run(["git", "submodule", "update", "--remote", "--recursive", t])
        after, after_short, after_date, after_subj = commit_info(sub)
        if before == after:
            print(f"  unchanged ({after_short})")
        else:
            print(f"  {before[:8]} → {after_short}  ({after_date})  {after_subj[:70]}")
    print()
    print("Stage + commit when ready:")
    print("  git diff --submodule=log    # review the bump")
    print("  git add " + " ".join(targets))
    print("  git commit -m 'bump <submodule> to ...'")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="(with --bump) submodule paths to bump; omit and pass --all to bump every tracked one",
    )
    parser.add_argument(
        "--fetch", action="store_true", help="run `git fetch origin` inside each submodule before reading status"
    )
    parser.add_argument(
        "--bump", action="store_true", help="bump given path(s) (or all with --all) to their tracking branch HEAD"
    )
    parser.add_argument("--all", action="store_true", help="(with --bump) bump every tracked submodule")
    args = parser.parse_args()

    if args.bump:
        if args.all:
            return bump(SUBMODULES)
        if not args.paths:
            print("FATAL: --bump needs a path arg, or --bump --all", file=sys.stderr)
            return 2
        return bump(args.paths)

    return status(do_fetch=args.fetch)


if __name__ == "__main__":
    sys.exit(main())
