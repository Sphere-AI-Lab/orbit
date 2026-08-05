# PEFT Allocator Memory Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `available_memory()` report the three allocator counters that separate
"memory trapped by fragmentation" from "memory the allocator declined to release", so the
49.9 GB the PEFT train actor holds at SGLang resume becomes readable in the logs we already
emit.

**Architecture:** Three fields are added to the dict returned by `available_memory()` in
`orbit/utils/memory_utils.py`, read from `torch.cuda.memory_stats()`. No new call sites and
no new flag: every existing `print_memory()` picks them up, including the three in
`MegatronTrainRayActor.sleep()` that bracket the offload. A second, CUDA-gated test exists
solely to defeat a failure mode this design creates for itself — see Task 2.

**Tech Stack:** Python 3.12, PyTorch 2.11.0+cu130, pytest.

## Global Constraints

- **Instrumentation only. No fix is applied by this plan.** Unchanged:
  `SGLANG_MEM_FRACTION_STATIC`, `SGLANG_MAX_TOTAL_TOKENS`, `SGLANG_MAX_RUNNING_REQUESTS`,
  `GLOBAL_BATCH_SIZE`, the `offload_train_*` flag defaults, and every protocol constant in
  `scripts/lora_regret/e4_protocol.sh`.
- **Byte-valued fields** go through the existing `_byte_to_gb` helper and carry the existing
  `_GB` suffix. **Count-valued fields** are dimensionless integers and take no suffix.
- **Every `memory_stats()` lookup uses `.get(key, 0)`.** `torch.cuda.memory_stats()` returns
  `{}` for a device the allocator has never served, and `print_memory` is called during
  setup before the first allocation.
- **No new flag, no new call site, no gating on an env var.** The counters are always on.
- Exact stat keys, verified present in torch 2.11.0+cu130:
  `inactive_split_bytes.all.current`, `segment.all.current`, `num_alloc_retries`.

---

## File Structure

| File | Responsibility |
|---|---|
| `orbit/utils/memory_utils.py` (modify) | The only production change. `available_memory()` gains three fields; `_byte_to_gb`, `clear_memory` and `print_memory` are untouched. |
| `tests/fast/utils/test_memory_utils_allocator_counters.py` (create) | Both tests. Lives beside `test_full_model_train_offload.py`, which covers the sibling offload behaviour and is the closest analogue in the suite. |

There is no existing test for `orbit/utils/memory_utils.py`, so the test file is new. It is
one file rather than two because both tests interrogate a single function and would share
the same fixture.

---

### Task 1: Report the three allocator counters

**Files:**
- Modify: `orbit/utils/memory_utils.py:18-28` (the `available_memory` function)
- Test: `tests/fast/utils/test_memory_utils_allocator_counters.py` (create)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `available_memory()` returns its existing six keys — `gpu`, `total_GB`,
  `free_GB`, `used_GB`, `allocated_GB`, `reserved_GB` — plus exactly three more:
  `inactive_split_GB: float`, `segments: int`, `alloc_retries: int`. Task 2 relies on the
  three stat key strings being the ones listed in Global Constraints.

- [ ] **Step 1: Write the failing tests**

Create `tests/fast/utils/test_memory_utils_allocator_counters.py`:

```python
"""`reserved - allocated` is not one number, and the difference decides the fix.

On 2026-08-05 the three LoRA arms of E4 gsm8k column 4 died at rollout 2 on
8xH100 in `torch_memory_saver ... func=resume`, while the FullFT arm of the same
column completed 149/149 rollouts on the same node. At the failure rank 0 held
`allocated 0.12 GB` against `reserved 50.01 GB` -- 49.9 GB free in PyTorch's
eyes and unavailable to SGLang's `cuMemCreate` all the same.

`offload_megatron_frozen_base_to_cpu` already calls `gc.collect()` then
`torch.cuda.empty_cache()` every rollout, and the 50.01 GB survives it. Two
things explain that equally well and imply different fixes: the segments are
partially occupied and therefore non-releasable (fragmentation ->
`expandable_segments:True`), or they are fully free and were skipped because
their blocks still carry recorded stream uses (-> a synchronising clear).

`inactive_split_bytes` is exactly the first quantity. Torch's own memory summary
labels it "Non-releasable memory": bytes that are free but sit inside a segment
still holding a live block. Reading it costs a host-side counter lookup on a log
line already being emitted, and it tells the two hypotheses apart.
"""

from __future__ import annotations

import pytest
import torch

from orbit.utils import memory_utils


def _patch_cuda(monkeypatch, stats):
    """A plausible 80 GB device, so only the stats dict varies between tests."""
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda device: (13 * 1024**3, 80 * 1024**3)
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 1024**3 // 8)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda device: 50 * 1024**3)
    monkeypatch.setattr(torch.cuda, "memory_stats", lambda device: stats)


def test_reports_the_non_releasable_bytes_empty_cache_cannot_return(monkeypatch):
    _patch_cuda(
        monkeypatch,
        {
            "inactive_split_bytes.all.current": 49 * 1024**3,
            "segment.all.current": 812,
            "num_alloc_retries": 17,
        },
    )

    info = memory_utils.available_memory()

    assert info["inactive_split_GB"] == 49.0
    assert info["segments"] == 812
    assert info["alloc_retries"] == 17


def test_the_existing_fields_are_not_disturbed(monkeypatch):
    _patch_cuda(monkeypatch, {})

    info = memory_utils.available_memory()

    assert info["total_GB"] == 80.0
    assert info["free_GB"] == 13.0
    assert info["used_GB"] == 67.0
    assert info["reserved_GB"] == 50.0


def test_survives_a_device_the_allocator_has_never_served(monkeypatch):
    """`print_memory` runs during setup, before the first allocation, and
    `memory_stats()` returns {} for such a device. A bare subscript would raise
    KeyError at startup, so every lookup must default."""
    _patch_cuda(monkeypatch, {})

    info = memory_utils.available_memory()

    assert info["inactive_split_GB"] == 0.0
    assert info["segments"] == 0
    assert info["alloc_retries"] == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
/fast/zqiu/orbit-iclr/orbit_env/bin/python -m pytest \
    tests/fast/utils/test_memory_utils_allocator_counters.py -v
```

Expected: `test_reports_the_non_releasable_bytes_empty_cache_cannot_return` and
`test_survives_a_device_the_allocator_has_never_served` both FAIL with
`KeyError: 'inactive_split_GB'`. `test_the_existing_fields_are_not_disturbed` PASSES already
— it guards against regression, so passing now is correct.

Collection alone takes ~65 s on this machine: importing `orbit.utils.memory_utils` pulls in
torch and megatron. That is normal, not a hang.

- [ ] **Step 3: Add the three fields**

In `orbit/utils/memory_utils.py`, replace the body of `available_memory()`:

```python
def available_memory():
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    # Returns {} for a device the allocator has never served, and print_memory
    # runs during setup before the first allocation -- hence .get on every key.
    stats = torch.cuda.memory_stats(device)
    return {
        "gpu": str(device),
        "total_GB": _byte_to_gb(total),
        "free_GB": _byte_to_gb(free),
        "used_GB": _byte_to_gb(total - free),
        "allocated_GB": _byte_to_gb(torch.cuda.memory_allocated(device)),
        "reserved_GB": _byte_to_gb(torch.cuda.memory_reserved(device)),
        # Torch calls this "Non-releasable memory": free bytes trapped inside a
        # segment that still holds a live block, which empty_cache() cannot
        # return. A large value here against a small allocated_GB is
        # fragmentation, not a leak.
        "inactive_split_GB": _byte_to_gb(stats.get("inactive_split_bytes.all.current", 0)),
        "segments": stats.get("segment.all.current", 0),
        "alloc_retries": stats.get("num_alloc_retries", 0),
    }
```

Leave `clear_memory`, `_byte_to_gb` and `print_memory` untouched.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
/fast/zqiu/orbit-iclr/orbit_env/bin/python -m pytest \
    tests/fast/utils/test_memory_utils_allocator_counters.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Confirm nothing else reads this dict positionally**

```bash
cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
grep -rn "available_memory\|print_memory" --include=*.py orbit/ tools/ train.py | grep -v "def "
```

Expected: every hit either calls `print_memory(...)` for its log side effect or binds the
result to a name and subscripts it by key. If any caller unpacks the dict positionally or
asserts its length, stop and report — adding keys would break it.

- [ ] **Step 6: Commit**

```bash
cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
git add orbit/utils/memory_utils.py tests/fast/utils/test_memory_utils_allocator_counters.py
git commit -m "feat(peft): report allocator fragmentation counters in memory logs"
```

---

### Task 2: Guard against the silent zero

**Files:**
- Modify: `tests/fast/utils/test_memory_utils_allocator_counters.py` (append one test)

**Interfaces:**
- Consumes: the three stat key strings established in Task 1.
- Produces: nothing consumed downstream.

**Why this task exists.** Task 1's `.get(key, 0)` is required — without it `print_memory`
raises `KeyError` at startup — but it also means a *misspelled* key silently returns `0`.
The mocked tests cannot catch that, because they feed the dict the keys they expect. A typo
in `inactive_split_bytes.all.current` would therefore make the diagnostic report
`inactive_split_GB: 0.0` on a live run, which is not a null result: per the spec's decision
rule, near-zero is read as evidence *for* the pending-stream hypothesis. The instrument
would confidently point at the wrong fix. This test asserts the keys exist in the real
allocator's output.

- [ ] **Step 1: Write the failing test**

Append to `tests/fast/utils/test_memory_utils_allocator_counters.py`:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a real CUDA allocator")
def test_the_stat_keys_this_module_reads_exist_in_this_torch():
    """The .get(key, 0) defaults above make a typo indistinguishable from a
    genuine zero -- and a genuine zero is what the spec reads as evidence
    against fragmentation. Pin the key names against the installed torch so a
    rename upstream fails here loudly instead of in a campaign's log."""
    torch.zeros(1, device="cuda")  # force the allocator to serve this device
    stats = torch.cuda.memory_stats(torch.cuda.current_device())

    for key in (
        "inactive_split_bytes.all.current",
        "segment.all.current",
        "num_alloc_retries",
    ):
        assert key in stats, f"{key} missing from torch {torch.__version__} memory_stats"
```

- [ ] **Step 2: Run it**

```bash
cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
/fast/zqiu/orbit-iclr/orbit_env/bin/python -m pytest \
    tests/fast/utils/test_memory_utils_allocator_counters.py -v
```

Expected on a CUDA machine: 4 passed. Expected on a CPU-only machine: 3 passed, 1 skipped.

**This test initialises a CUDA context** (a few hundred MB, transient) because
`memory_stats()` requires one. On a shared login node, ask before running it rather than
launching unprompted; the other three tests need no GPU and can always be run.

This test is expected to pass on first run — it pins existing behaviour rather than driving
new code. If it fails, a key name in Task 1 is wrong and that is exactly the bug this task
exists to catch: fix the key in `memory_utils.py`, do not relax the assertion.

- [ ] **Step 3: Commit**

```bash
cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
git add tests/fast/utils/test_memory_utils_allocator_counters.py
git commit -m "test(peft): pin the allocator stat key names against installed torch"
```

---

## Measurement Handoff

Not a task — this is the GPU run that the instrumentation exists to serve, and it is the
user's to launch.

Do **not** go through `run_e4_gsm8k_lr4_8gpu.sh`: it hardcodes `METHOD_RE` and
`EXPECT_ARMS=4` through `exec env`, so an outer override is silently ignored and all three
LoRA arms run. Call `campaign.sh` directly after sourcing the protocol, which is the
documented path (`campaign.sh:10`). Sourcing `e4_protocol.sh` is required, not optional —
it is what sets `GLOBAL_BATCH_SIZE=1024`, and that batch is the memory condition under test.

```bash
source /fast/zqiu/orbit-iclr/orbit_env/bin/activate
cd /lustre/fast/fast/zqiu/orbit-iclr/orbit
source scripts/lora_regret/e4_protocol.sh

# 1. confirm the selection is exactly one arm, running nothing
MATRIX=e4 METHOD_RE='^lora-r1-all-gsm8k-lr7e-05-s' RESULTS=results/mem_diag_lora_r1.jsonl \
    EXPECT_ARMS=1 DRY_RUN=1 bash scripts/lora_regret/campaign.sh

# 2. run it
MATRIX=e4 METHOD_RE='^lora-r1-all-gsm8k-lr7e-05-s' RESULTS=results/mem_diag_lora_r1.jsonl \
    EXPECT_ARMS=1 bash scripts/lora_regret/campaign.sh
```

The scratch `RESULTS` path keeps a diagnostic row out of `results/e4_gsm8k_lr4.jsonl`; the
per-arm log path is unaffected by it. The r1 arm reaches the failure at rollout 2 in roughly
14 minutes, so this needs a node but not a campaign. Then read the counter at the last
`before update_weights` line:

```bash
grep "Rank 0\] Memory-Usage before update_weights" \
    logs/lora_regret/lora-r1-all-gsm8k-lr7e-05-s0.log | tail -3
```

Decision rule, from spec §6, at the probe where `reserved - allocated ≈ 49.9 GB`:

| reading | conclusion | indicated fix |
|---|---|---|
| `inactive_split_GB` ≈ 49 | fragmentation confirmed | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` for the train actors |
| `inactive_split_GB` ≈ 0 | fully free segments went unreleased | synchronising clear on the PEFT offload path |
| in between | mixed pool | escalate to `torch.cuda.memory_snapshot()` per-segment capture |

Either fix is a separate change with its own smoke run, and both are expected to be
hardware-independent — nothing in either hypothesis is specific to an 80 GB card.
