#!/usr/bin/env python3
"""CPU-only end-to-end check of the disk-delta wire format. No GPU, no cluster, no Ray.

Reproduces exactly what the trainer publishes (``UpdateWeightFromDiskDelta._encode_delta`` /
``._write_delta_files``), then hands the published directory to the *real* receiver —
``sglang.srt.weight_sync.local_checkpoint.pull``, the same code path ``/pull_weights`` runs on
every rollout host — and verifies the patched checkpoint is byte-identical to what the trainer
intended.

That makes it a genuine cross-implementation check rather than a self-consistency one: the
publisher here is a transcription of miles' encoder, the applier is sglang's own. It catches the
three things that actually break a first bring-up — a missing codec/checksum dependency in the
runtime env, an encoding the two sides disagree about, and a shared directory the engine can't
read — without burning a multi-GPU allocation to find out.

Usage:

    # zero setup: synthesizes a small checkpoint, runs the full round trip
    python examples/disk_delta_weight_sync/check_delta_roundtrip.py

    # against real weights, both encodings
    python examples/disk_delta_weight_sync/check_delta_roundtrip.py \\
        --hf-checkpoint /data/shared/hf_cache/models/Qwen3-4B --encoding overwrite

Run it in the same environment the training job uses — proving the deps import on a login node
says nothing about the compute nodes.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import sys
import tempfile
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from miles.utils.disk_delta import checksum, make_tensor_reader, overwrite_encode  # noqa: E402

# The trainer's ~3% density operating point: after one RL step almost every bf16 element is
# byte-identical to the previous step, which is the entire premise of shipping deltas.
DEFAULT_DENSITY = 0.03


def check_dependencies(algorithm: str) -> None:
    """The checksum and codec deps are pinned in requirements.txt but are easy to miss in an env
    built before disk-delta landed, and the failure otherwise surfaces mid-run."""
    hashers = {"xxh3-128": "xxhash", "blake3": "blake3", "adler32": "zlib"}
    modules = ["numpy", "zstandard", "safetensors", hashers[algorithm]]

    missing = []
    for module in modules:
        try:
            mod = __import__(module)
        except ImportError as e:
            missing.append(f"{module} ({e})")
        else:
            attrs = (getattr(mod, a, None) for a in ("__version__", "VERSION", "ZLIB_VERSION"))
            print(f"  ok   {module:<12} {next((v for v in attrs if v), '?')}")
    if missing:
        raise SystemExit("missing dependencies: " + ", ".join(missing) + "\n  pip install -r requirements.txt")


def synth_checkpoint(path: str, num_tensors: int, tensor_mb: int) -> None:
    """A small stand-in checkpoint. fp16 rather than bf16 only because safetensors.numpy has no
    bf16; both delta encodings are byte-level and dtype-blind, so the path exercised is the same."""
    import safetensors.numpy

    os.makedirs(path, exist_ok=True)
    rng = np.random.default_rng(0)
    elems = tensor_mb * 1024 * 1024 // 2
    tensors = {
        f"model.layers.{i}.weight": rng.integers(0, 2**16, size=elems, dtype="<u2").view(np.float16)
        for i in range(num_tensors)
    }
    safetensors.numpy.save_file(tensors, os.path.join(path, "model.safetensors"))


def tensor_names(ckpt_dir: str) -> list[str]:
    names = []
    for entry in sorted(os.listdir(ckpt_dir)):
        if not entry.endswith(".safetensors"):
            continue
        with open(os.path.join(ckpt_dir, entry), "rb") as f:
            (header_len,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(header_len))
        names.extend(name for name in header if name != "__metadata__")
    return names


def mutate(old: np.ndarray, density: float, rng: np.random.Generator, nudge: int) -> np.ndarray:
    """Perturb ``density`` of the 2-byte elements, standing in for one optimizer step.

    Nudges the low mantissa bits rather than randomizing the element, because the difference
    decides whether the reported numbers mean anything. A real step moves a weight slightly, so
    most changed bf16 elements differ only in their low byte: the XOR residue keeps its
    high-order zeros, byte-level density lands below element-level density, and zstd has
    something to work with. Randomized replacements destroy all three effects and understate
    the win by roughly an order of magnitude.
    """
    new = old.copy()
    if new.nbytes >= 2:
        view = new[: new.nbytes // 2 * 2].view("<u2")
        k = max(1, int(view.size * density))
        idx = rng.integers(0, view.size, size=k)
        view[idx] += rng.integers(1, nudge + 1, size=k, dtype="<u2")
    return new


def publish_version(
    version_dir: str,
    deltas: dict[str, np.ndarray],
    checksums: dict[str, str],
    version: int,
    encoding: str,
    algorithm: str,
) -> int:
    """Write one version exactly as ``_write_delta_files`` does: the changed tensors as a single
    canonical safetensors shard carrying the new-state checksums as metadata, plus the HF index
    whose metadata is the contract the receiver reads the encoding and base version from."""
    import safetensors.numpy

    os.makedirs(version_dir, exist_ok=True)
    wire_bytes = 0
    fname = None
    if deltas:
        fname = "model-00000-of-00001.safetensors"
        blob = safetensors.numpy.save(deltas, metadata=checksums)
        wire_bytes = len(blob)
        with open(os.path.join(version_dir, fname), "wb") as f:
            f.write(blob)
    index = {
        "metadata": {
            "version": f"{version:06d}",
            "base_version": f"{version - 1:06d}",
            "delta_encoding": encoding,
            "compression_format": "zstd",
            "checksum_format": algorithm,
        },
        "weight_map": {name: fname for name in deltas},
    }
    with open(os.path.join(version_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)
    return wire_bytes


def encode_version(
    snapshot: dict[str, np.ndarray],
    truth: dict[str, np.ndarray],
    encoding: str,
    algorithm: str,
) -> tuple[dict[str, np.ndarray], dict[str, str], int, int]:
    """Transcription of ``_encode_delta``'s per-tensor work: diff against the snapshot, encode,
    zstd level 1, and checksum the *new* state (what the receiver verifies it reconstructed)."""
    import zstandard

    deltas: dict[str, np.ndarray] = {}
    checksums: dict[str, str] = {}
    changed_bytes = total_bytes = 0

    compressor = zstandard.ZstdCompressor(level=1)
    for name, new in truth.items():
        old = snapshot[name]
        total_bytes += new.nbytes
        if encoding == "xor":
            diff = new ^ old
            changed = int(np.count_nonzero(diff))
        else:
            mask = new != old
            changed = int(np.count_nonzero(mask))
            diff = overwrite_encode(new, mask)
        if not changed:
            continue
        changed_bytes += changed
        deltas[name] = np.frombuffer(compressor.compress(diff), dtype=np.uint8)
        checksums[name] = checksum(algorithm, new)
    return deltas, checksums, changed_bytes, total_bytes


def corrupt_last_delta(version_dir: str) -> None:
    """Falsify one tensor's recorded checksum.

    Targets the integrity layer specific to this feature rather than something zstd would catch
    anyway: the delta decompresses cleanly and applies cleanly, and the only thing standing
    between it and the engine serving weights nobody verified is the per-tensor checksum
    comparison. If that comparison is absent or unreachable, this run applies without complaint.
    """
    import safetensors.numpy

    shard = next(f for f in sorted(os.listdir(version_dir)) if f.endswith(".safetensors"))
    path = os.path.join(version_dir, shard)
    with open(path, "rb") as f:
        blob = f.read()
    (header_len,) = struct.unpack("<Q", blob[:8])
    metadata = dict(json.loads(blob[8 : 8 + header_len]).get("__metadata__", {}))
    victim = sorted(metadata)[0]
    metadata[victim] = "0" * len(metadata[victim])
    with open(path, "wb") as f:
        f.write(safetensors.numpy.save(dict(safetensors.numpy.load(blob)), metadata=metadata))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hf-checkpoint", help="real HF checkpoint to diff against; a small one is synthesized if omitted")
    p.add_argument("--encoding", default="xor", choices=["xor", "overwrite"])
    p.add_argument("--checksum", default="xxh3-128", choices=["xxh3-128", "blake3", "adler32"])
    p.add_argument(
        "--density",
        type=float,
        default=DEFAULT_DENSITY,
        help=f"fraction of elements changed per sync (default {DEFAULT_DENSITY})",
    )
    p.add_argument("--versions", type=int, default=2, help="how many syncs to publish and apply in sequence")
    p.add_argument(
        "--nudge",
        type=int,
        default=8,
        help="max increment applied to a changed element's bit pattern; larger = coarser step (default 8)",
    )
    p.add_argument("--workdir", help="scratch dir for the published + local checkpoints (a temp dir by default)")
    p.add_argument("--synth-tensors", type=int, default=8)
    p.add_argument("--synth-tensor-mb", type=int, default=8)
    p.add_argument(
        "--corrupt",
        action="store_true",
        help="flip a byte in the last published delta; the run PASSES only if the receiver refuses it",
    )
    args = p.parse_args()

    print("dependencies:")
    check_dependencies(args.checksum)

    # The receiver is sglang's own; without it this script can only test miles against itself,
    # which is not the question worth answering.
    try:
        from sglang.srt.weight_sync.local_checkpoint import pull
    except ImportError as e:
        raise SystemExit(
            f"\ncannot import sglang's receiver ({e}).\n"
            "Run this in the environment the training job uses — the point of the check is that\n"
            "miles' publisher and sglang's applier agree on the format."
        )
    print("  ok   sglang.srt.weight_sync.local_checkpoint.pull")

    workdir = args.workdir or tempfile.mkdtemp(prefix="delta-roundtrip-")
    owns_workdir = args.workdir is None
    published = os.path.join(workdir, "published")
    local = os.path.join(workdir, "local")
    shutil.rmtree(published, ignore_errors=True)
    shutil.rmtree(local, ignore_errors=True)

    try:
        if args.hf_checkpoint:
            base = args.hf_checkpoint
        else:
            base = os.path.join(workdir, "base")
            shutil.rmtree(base, ignore_errors=True)
            print(f"\nsynthesizing base checkpoint: {args.synth_tensors} x {args.synth_tensor_mb} MiB")
            synth_checkpoint(base, args.synth_tensors, args.synth_tensor_mb)

        print(f"\nbase       : {base}")
        print(f"published  : {published}")
        print(f"local      : {local}")
        # Element churn is the knob; byte density is what the trainer reports and what the wire
        # size follows from. They differ because a nudged element usually changes only one byte.
        print(f"encoding   : {args.encoding}   checksum: {args.checksum}   element churn: {args.density:.2%}\n")

        # The trainer seeds its snapshot from --hf-checkpoint, not from current GPU weights, so
        # the invariant `snapshot == engine base` holds even where the Megatron->HF round trip
        # trims vocab padding. Same thing here.
        read = make_tensor_reader(base)
        names = tensor_names(base)
        snapshot = {name: read(name) for name in names}
        truth = dict(snapshot)
        rng = np.random.default_rng(1234)

        for version in range(1, args.versions + 1):
            truth = {name: mutate(buf, args.density, rng, args.nudge) for name, buf in truth.items()}

            t0 = time.perf_counter()
            deltas, checksums, changed, total = encode_version(snapshot, truth, args.encoding, args.checksum)
            version_dir = os.path.join(published, f"weight_v{version:06d}")
            wire = publish_version(version_dir, deltas, checksums, version, args.encoding, args.checksum)
            t_publish = time.perf_counter() - t0
            snapshot = dict(truth)  # the applied state becomes the next diff's base

            if args.corrupt and version == args.versions:
                corrupt_last_delta(version_dir)
                try:
                    pull(local_checkpoint_dir=local, base_dir=base, source_dir=published, target_version=version)
                except Exception as e:
                    print(f"v{version}: corrupted delta rejected — {type(e).__name__}: {e}")
                    print("\nPASS — the receiver refused a corrupted delta instead of serving bad weights.")
                    return 0
                raise SystemExit(f"FAIL v{version}: a corrupted delta was applied without complaint")

            t0 = time.perf_counter()
            pull(local_checkpoint_dir=local, base_dir=base, source_dir=published, target_version=version)
            t_apply = time.perf_counter() - t0

            # pull() raises on a checksum mismatch, but a checksum only proves the receiver
            # reconstructed what the trainer *said* it should. Compare the actual bytes.
            applied = make_tensor_reader(local)
            for name in names:
                got = applied(name)
                if not np.array_equal(got, truth[name]):
                    bad = int(np.count_nonzero(got != truth[name]))
                    raise SystemExit(f"FAIL v{version}: {name} differs in {bad} of {got.nbytes} bytes")

            print(
                f"v{version}: byte density {100.0 * changed / max(total, 1):5.2f}%  "
                f"wire {wire / 1e6:8.2f} MB of {total / 1e6:8.2f} MB  "
                f"({total / max(wire, 1):5.1f}x)  publish {t_publish:5.2f}s  apply {t_apply:5.2f}s  OK"
            )

        print(
            f"\nPASS — {args.versions} version(s) published by miles' encoder "
            "and applied byte-exactly by sglang's receiver."
        )
        return 0
    finally:
        if owns_workdir:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
