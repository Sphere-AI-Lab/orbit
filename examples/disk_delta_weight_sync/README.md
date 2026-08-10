# Disk-Delta Weight Sync

Testing recipes for `--update-weight-transfer-mode disk-delta`: ship only the bytes that changed
between two syncs instead of the whole actor.

After one RL step almost every bf16 element is byte-identical to the previous step, so the trainer
diffs each gathered HF tensor against a CPU snapshot, publishes the changed bytes to a shared
directory, and each engine's `/pull_weights` patches a host-local checkpoint in place and reloads
from it through the ordinary `update_weights_from_disk` path. miles only ever talks to one endpoint
per engine — the engine fans the apply out to every host it spans — so multi-node serving and
external rollout engines need nothing extra.

Compare with [`../p2p_weight_transfer/`](../p2p_weight_transfer/), which moves the *full* weights
over RDMA. disk-delta trades interconnect bandwidth for CPU: it wins where the fabric is the
constraint (cross-rack, cross-datacenter, plain Ethernet) and loses where it isn't.

## Files

```
examples/disk_delta_weight_sync/
├── check_delta_roundtrip.py                  # CPU-only format check — run this first
├── _common.sh                                # shared workload; wrappers vary one axis each
├── _model-qwen3-30B-A3B.sh                   # MoE model block (arch, parallelism, layout)
│
├── 01-qwen3-4B-1node-delta-smoke.sh          # dense, 1 node 4+4, --check-weight-update-equal
├── 02-qwen3-4B-2node-delta.sh                # dense, deltas cross a real host boundary
├── 03-qwen3-4B-2node-broadcast.sh            # control arm for 02
│
├── 04-qwen3-30B-A3B-4node-delta-smoke.sh     # MoE correctness — first run of the expert path
├── 05-qwen3-30B-A3B-4node-delta.sh           # MoE measurement arm
├── 06-qwen3-30B-A3B-4node-broadcast.sh       # control arm for 05
└── 07-qwen3-30B-A3B-1node-delta-smoke.sh     # same expert coverage, one node
```

`_common.sh` fixes the dataset and the RL hyperparameters; a model block overrides the
architecture, parallelism, and layout. Within one model, the delta and broadcast arms differ in
the transport block and nothing else — `diff` the resolved `MILES_ARGS` if you want to confirm it.

## Start here: the format check

No GPU, no cluster, no Ray. It transcribes miles' encoder, then hands the published directory to
sglang's *real* receiver (`sglang.srt.weight_sync.local_checkpoint.pull`, the code `/pull_weights`
runs) and verifies the patched checkpoint byte-for-byte:

```bash
python examples/disk_delta_weight_sync/check_delta_roundtrip.py
```

Run it **in the environment the training job uses**. Its main job is catching the three things that
break a first bring-up — a codec or checksum dependency missing from the runtime env
(`blake3`, `xxhash`, `zstandard` are pinned in `requirements.txt` but easy to miss in an env built
before disk-delta landed), an encoding the two sides disagree about, and a shared directory the
engine can't read. A login-node pass says nothing about the compute nodes.

Useful variants:

```bash
# both encodings, all three checksums, longer delta chains
python .../check_delta_roundtrip.py --encoding overwrite --checksum blake3 --versions 5

# against real weights instead of a synthesized checkpoint
python .../check_delta_roundtrip.py --hf-checkpoint /data/shared/hf_cache/models/Qwen3-4B

# prove the integrity layer actually fires: falsifies a checksum, passes only if it is refused
python .../check_delta_roundtrip.py --corrupt

# coarser simulated step: --nudge bounds how far a changed element's bit pattern moves
python .../check_delta_roundtrip.py --density 0.01 --nudge 64
```

**What the compression ratio does and does not tell you.** It is governed by the mutation model,
not by the base weights: against real Qwen3-4B and against the synthesized checkpoint, the same
`--density` and `--nudge` produce the same ratio. The script perturbs low mantissa bits to imitate
an optimizer step — that shape matters, and randomizing elements instead understates the win by
roughly an order of magnitude — but the changed *positions* here are uniformly random where a real
step's are not. Read the ratio as a format and throughput check. The number worth quoting comes
from `perf/update_weights_wire_bytes` on an actual run.

## Cluster runs

The wrappers satisfy the `scripts/slurm/submit.sh` recipe contract, but `submit.sh` resolves
recipes by name under `scripts/experiments/`. Add a three-line shim:

```bash
# scripts/experiments/dd-01-smoke.sh
source "$MILES_REPO/examples/disk_delta_weight_sync/01-qwen3-4B-1node-delta-smoke.sh"
```

```bash
bash scripts/slurm/submit.sh dd-01-smoke
```

Run `01` first. It is short and turns on `--check-weight-update-equal`, which compares engine
tensors against the trainer's after each sync — the only direct proof that an applied delta
reconstructed the right bytes on real weights. Then run `02` and `03` back to back; `_common.sh`
holds the workload fixed so the difference between them is the transport and nothing else.

### The MoE arm

`04`–`06` run Qwen3-30B-A3B: 48 layers, 128 experts, 8 active per token, 57 GB in bf16. Both
checkpoints are already on slinky, outside `HF_CACHE_DIR`, so `submit.sh` skips the download:

| | |
|---|---|
| `--hf-checkpoint` | `/data/shared/models/Qwen3-30B-A3B` (16 shards, 57 GB) |
| `--ref-load` | `/data/home/zeju/models/Qwen3-30B-A3B_torch_dist` — a user-specific path; override `DD_HF_TORCHDIST_DIR` elsewhere |

Two reasons the MoE arm matters, and the second is the one that justifies running it before any
measurement:

1. **It is where the mechanism should pay.** Broadcast moves all 57 GB every sync no matter how
   little the step changed, against ~3B active parameters per token.
2. **It covers code the dense arm cannot reach.** `_for_each_hf_bucket` runs a TP pass and then a
   separate EP pass, and `_drop_duplicate_names` exists because an expert tensor can be gathered
   by more than one rank. At `--expert-model-parallel-size 1` the Qwen3-4B run entered neither.
   Run `04` first — it is the first exercise of the expert path, and it fails loudly.

Parallelism (TP4, EP8, 2 training + 2 rollout nodes, sglang `--sglang-ep-size 8` with DP
attention) mirrors [`../p2p_weight_transfer/run-qwen3-30B-A3B-4node-profile.sh`](../p2p_weight_transfer/run-qwen3-30B-A3B-4node-profile.sh),
the validated shape for this model on 4 nodes. Note the host-local checkpoint is now ~57 GB per
rollout host.

**`07` gets the same expert coverage on one node.** The nodes are H200s (141 GB), so 8 GPUs hold
1128 GB: 57 GB of weights shards to ~14 GB per GPU on both sides of a 4+4 disaggregated split,
and the 360 GB of Adam state lives in the node's 1.8 TB of host RAM via `--optimizer-cpu-offload`.
The forced change is EP: eight-way expert parallelism does not fit across 4 training GPUs, so `07`
runs EP4 — 32 experts per rank instead of 16. The EP gather and the duplicate-name drop still run,
so it covers the same code. What it does *not* cover is the fabric: with one node the publish dir
and the host-local checkpoint sit on the same box, so use `04`/`05` when the cross-host path is
the question.

### Adding another model

Write a `_model-<name>.sh` block setting `DD_MODEL_CONFIG` (a file in `scripts/models/`),
the checkpoint paths, parallelism, and layout, then a wrapper per transport. Other MoE
checkpoints already on the cluster: `Qwen3-VL-30B-A3B-Thinking` (58 GB) and
`Qwen3-Omni-30B-A3B-Instruct` (66 GB) under `/data/shared/models/`, though neither has a
converted `torch_dist` yet.

### The two directories

| Flag | What it is |
|---|---|
| `--update-weight-disk-dir` | Shared filesystem. Trainer writes one `weight_v{N:06d}/` per sync; every rollout host reads it. |
| `--update-weight-local-checkpoint-dir` | **Host-local** (NVMe if you have it). A full HF checkpoint each engine patches in place. |

Two things to get right:

- **The publish dir is wiped.** The trainer `rmtree`s it when it captures the baseline — a stale
  version stream from a previous run would apply against the wrong base. Point it at per-run
  scratch, never at anything you want to keep. `_common.sh` defaults to
  `$MILES_REPO/checkpoints/$RUN_NAME/delta-updates`.
- **The local dir holds a whole checkpoint per host**, ~8 GB for Qwen3-4B in bf16. `_common.sh`
  defaults to `/tmp/miles-delta-local-ckpt/$RUN_NAME`; override `DD_LOCAL_CKPT_DIR` if `/tmp` is
  small or is not node-local on your nodes. Putting it on the *shared* filesystem defeats the
  entire mechanism.

### Knobs

All read by `_common.sh` from the environment or the wrapper:

| Variable | Default | Notes |
|---|---|---|
| `DD_TRANSFER_MODE` | `disk-delta` | `broadcast` for the control arm |
| `DD_ENCODING` | `xor` | `new ^ old`; smallest and fastest, but an involution — it must land exactly once on the right base. `overwrite` stores positions + absolute values: larger, idempotent. |
| `DD_CHECKSUM` | `xxh3-128` | `blake3` for untrusted storage, `adler32` for interop |
| `DD_DISK_DIR` | `$MILES_REPO/checkpoints/$RUN_NAME/delta-updates` | shared; wiped at baseline |
| `DD_LOCAL_CKPT_DIR` | `/tmp/miles-delta-local-ckpt/$RUN_NAME` | host-local |
| `DD_CHECK_EQUAL` | `0` | `1` adds `--check-weight-update-equal`; on for `01` and `04` |
| `DD_NUM_ROLLOUT` | `3000` | `01` and `04` use 5 |
| `DD_MODEL_CONFIG` | `qwen3-4B` | names a file in `scripts/models/` |
| `DD_HF_MODEL_DIR` / `DD_HF_TORCHDIST_DIR` | Qwen3-4B under `HF_CACHE_DIR` | checkpoint paths |
| `DD_TP` / `DD_PP` / `DD_CP` / `DD_EP` / `DD_ETP` | `2/1/1/1/1` | training parallelism |
| `DD_GPUS_PER_ENGINE`, `DD_MEM_FRACTION` | `2`, `0.85` | rollout engine shape |
| `DD_EXTRA_SGLANG_ARGS`, `DD_EXTRA_OPTIMIZER_ARGS` | unset | arrays appended verbatim |

## Reading the results

The trainer logs per sync and records two metrics the actor drains onto the step log:

```
[disk delta v=3] density=2.87% wire=0.19 GB
```

| Metric | Meaning |
|---|---|
| `perf/update_weights_density` | fraction of weight bytes that changed — the premise. ~1–3% is the expected operating point. |
| `perf/update_weights_wire_bytes` | compressed bytes actually written |

A density near 100% means the diff is finding no reuse and something is wrong with the baseline —
most likely the snapshot and the engine base disagree. The snapshot is seeded from
`--hf-checkpoint` rather than from current GPU weights precisely so they agree even where the
Megatron→HF round trip trims vocab padding on `embed`/`lm_head`.

**Sync #1 publishes nothing.** The first `update_weights()` call only captures the baseline; deltas
start at v1, the second sync. A run of fewer than three syncs measures almost nothing.

## Constraints

Asserted at startup in [`miles/utils/arguments.py`](../../miles/utils/arguments.py):

- **No `--colocate`.** Colocated weights cross via CUDA IPC — only a handle moves — so snapshot +
  diff + encode would be pure overhead. Every recipe here is disaggregated.
- **No PD disaggregation** (`--prefill-num-servers`) — untested.
- **No LoRA** (`--lora-rank > 0`).
- **`--hf-checkpoint` must be a local directory**; the baseline is seeded from its safetensors bytes.

## Object-store filesystems

If your shared storage has no cross-host read-after-write consistency, the writes are not visible to
the engines without an explicit refresh. Two hooks, loaded by import path, keep that vendor logic
out of miles:

- `--custom-update-weight-post-write-path` (trainer) — after a version's files are written, before
  the engines are told to read them.
- `--sglang-custom-pull-weights-pre-read-hook` (engine) — on each host before it reads the version
  directory.

POSIX shared filesystems (NFS, Lustre, Weka) need neither.

## Upstream documentation

slime's [`docs/en/advanced/delta-weight-sync.md`](https://github.com/THUDM/slime/blob/main/docs/en/advanced/delta-weight-sync.md).
Flag names differ — slime splits mode and transport (`--update-weight-mode delta`
`--update-weight-transport disk`) where miles folds both into
`--update-weight-transfer-mode disk-delta` — but the on-disk format and the encodings are the same.
