import os
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[6] / "scripts" / "slurm" / "setup" / "cu130" / "materialize_env.py"


def _build(tmp_path: Path):
    cache = tmp_path / "cache" / "uv" / "archive-v0"
    pkg = cache / "abc" / "torch"
    (pkg / "lib").mkdir(parents=True)
    (pkg / "__init__.py").write_text("x = 1\n")
    (pkg / "lib" / "libfoo.so.1").write_bytes(b"\x7fELF")
    (pkg / "lib" / "libfoo.so").symlink_to("libfoo.so.1")  # internal alias stays a symlink
    (cache / "def" / "finder.py").parent.mkdir(parents=True)
    (cache / "def" / "finder.py").write_text("print('finder')\n")
    sp = tmp_path / "prefix" / "lib" / "python3.12" / "site-packages"
    sp.mkdir(parents=True)
    (sp / "torch").symlink_to(pkg)
    (sp / "__editable___x_finder.py").symlink_to(cache / "def" / "finder.py")
    (sp / "local").mkdir()
    (sp / "local" / "alias.so").symlink_to(sp / "torch" / "lib" / "libfoo.so.1")  # resolves into cache via torch
    return tmp_path / "prefix", tmp_path / "cache" / "uv", sp


def test_dry_run_lists_only_cache_links(tmp_path):
    prefix, cache, sp = _build(tmp_path)
    out = subprocess.run([sys.executable, SCRIPT, "--prefix", prefix, "--cache-dir", cache, "--dry-run"],
                         check=True, capture_output=True, text=True).stdout
    assert "3 symlinks into" in out
    assert (sp / "torch").is_symlink()


def test_materialize_replaces_cache_links_and_keeps_internal_ones(tmp_path):
    prefix, cache, sp = _build(tmp_path)
    subprocess.run([sys.executable, SCRIPT, "--prefix", prefix, "--cache-dir", cache, "--jobs", "2"], check=True)
    assert not (sp / "torch").is_symlink() and (sp / "torch" / "__init__.py").read_text() == "x = 1\n"
    assert not (sp / "__editable___x_finder.py").is_symlink()
    assert (sp / "torch" / "lib" / "libfoo.so").is_symlink()  # package-internal alias preserved
    assert os.readlink(sp / "torch" / "lib" / "libfoo.so") == "libfoo.so.1"
    assert not (sp / "local" / "alias.so").is_symlink()  # resolved into the cache, so copied
    import shutil
    shutil.rmtree(cache)
    assert (sp / "torch" / "lib" / "libfoo.so.1").read_bytes() == b"\x7fELF"  # survives cache deletion
