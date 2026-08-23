"""Replace symlinks that point into the uv cache with real copies.

uv's symlink link mode (often inherited via UV_LINK_MODE=symlink) leaves
site-packages entries pointing into the cache directory, so the prefix dies
with the cache and cannot move between nodes. Copy mode avoids that but runs at
a few files per second on Lustre because uv copies file-by-file. Building with
a node-local cache and then materializing in parallel keeps both properties:
fast unpack, self-contained prefix.

Only symlinks whose resolved target lies under --cache-dir are replaced;
symlinks internal to a package (for example versioned .so aliases) are kept.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def find_cache_links(root: Path, cache: Path) -> list[Path]:
    cache = cache.resolve()
    links: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            if not path.is_symlink():
                continue
            try:
                target = path.resolve(strict=True)
            except OSError:
                continue
            if cache == target or cache in target.parents:
                links.append(path)
        # Do not descend into symlinked directories: they are replaced whole.
        dirnames[:] = [d for d in dirnames if not (Path(dirpath) / d).is_symlink()]
    return links


def materialize(link: Path) -> int:
    target = link.resolve(strict=True)
    tmp = link.with_name(link.name + ".materialize-tmp")
    if tmp.exists() or tmp.is_symlink():
        shutil.rmtree(tmp) if tmp.is_dir() and not tmp.is_symlink() else tmp.unlink()
    if target.is_dir():
        shutil.copytree(target, tmp, symlinks=True)
        count = sum(len(files) for _, _, files in os.walk(tmp))
    else:
        shutil.copy2(target, tmp, follow_symlinks=True)
        count = 1
    link.unlink()
    os.rename(tmp, link)
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--prefix", required=True, type=Path, help="Conda/venv prefix to materialize")
    parser.add_argument("--cache-dir", required=True, type=Path, help="uv cache directory the links point into")
    parser.add_argument("--jobs", type=int, default=16, help="parallel copy streams")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.prefix.resolve()
    if not root.is_dir():
        print(f"FATAL: prefix is not a directory: {root}", file=sys.stderr)
        return 2
    links = find_cache_links(root, args.cache_dir)
    print(f"[materialize] {len(links)} symlinks into {args.cache_dir} under {root}")
    if args.dry_run:
        for link in links:
            print(f"  {link.relative_to(root)} -> {os.readlink(link)}")
        return 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        counts = list(pool.map(materialize, links))
    remaining = find_cache_links(root, args.cache_dir)
    elapsed = time.time() - started
    print(f"[materialize] copied {sum(counts)} files in {elapsed:.0f}s; {len(remaining)} cache links remain")
    return 1 if remaining else 0


if __name__ == "__main__":
    sys.exit(main())
