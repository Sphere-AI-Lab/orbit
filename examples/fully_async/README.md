# Fully Asynchronous Rollout Example

This example shows a simple way to make rollout generation **fully asynchronous**: a single global worker is created once and then keeps running in the background, continuously pulling prompts and launching generation tasks. Training only needs to fetch already finished results. This removes the per‑step wait that happens in the normal synchronous style.

## Files
* `fully_async_rollout.py`: global async worker + `generate_rollout_fully_async` entry.
* `run-qwen3-4b-fully_async.sh`: example launch script with Qwen3‑4B.

## Prerequisite
First set up model & environment following the Qwen3-4B example.

## Quick Start
```bash
cd miles
bash examples/fully_async/run-qwen3-4b-fully_async.sh
```
You should see log lines like:
```
Creating new global async worker...
Continuous async rollout worker started
```

## How It Works (Very Short)
* First call: create `AsyncRolloutWorker` (thread + asyncio loop).
* Loop keeps up to `--rollout-batch-size * --fully-async-prefetch-batches` prompt groups actively generating with `generate_and_rm_group`.
* Completed groups are pushed into a queue; caller drains until it has enough samples.
* Worker is stopped automatically at process exit.

## Prefetch and Staleness

`--fully-async-prefetch-batches` controls how aggressively the background worker
produces data ahead of the trainer. The training side still consumes exactly
`--rollout-batch-size` prompt groups per rollout step; this option only changes
the number of prompt groups allowed to run concurrently in the background worker.
Completed groups waiting in the worker output queue do not count against this
active generation window.

For example, with:

```bash
--rollout-batch-size 64
--n-samples-per-prompt 8
--fully-async-prefetch-batches 2
```

the worker may keep up to `128` prompt groups actively generating, or up to
`1024` sample generations before the SGLang client semaphore applies
backpressure.

`--fully-async-max-completed-queue-groups` is a soft safety cap for completed
groups waiting to be consumed by the trainer. When the queue reaches the cap,
the worker stops launching new generation tasks until the trainer drains queued
groups. Already-running tasks can still finish and enter the queue, so this is
a memory-growth guard rather than a second concurrency target. The default is
`2048` prompt groups.

`--max-weight-staleness` is a separate acceptance filter. Completed groups whose
oldest recorded rollout weight version is too far behind the current engine
weight version are reset and returned to the data buffer, so their prompts are
sampled again instead of being dropped. A useful starting point is:

```bash
--fully-async-prefetch-batches 2
--max-weight-staleness 2
```

Large prefetch windows with a small staleness limit can waste sampler work,
because many completed groups may be recycled before training. The worker emits
a warning when `fully_async_prefetch_batches > max_weight_staleness + 1`.

## Limitations
* No evaluation mode.
* Ordering is best effort (sorted at the end by index).
* Minimal error handling.

## Scaling limits & roadmap

This example is intentionally simple — a **single** global worker living inside the
one `RolloutManager` Ray actor. That has a structural ceiling worth knowing before
scaling sampler nodes:

* **Single-node funnel.** The completed-group `output_queue` is plain Python CPU
  memory in that one actor process. No matter how many sampler nodes generate in
  parallel, every finished group is pulled back into a single node's heap. For VLM
  this is heavier than text — the queued samples carry processed `pixel_values`
  (computed in-rollout), duplicated across `n_samples_per_prompt`.
* **Single-process compute.** Sample assembly, image preprocessing
  (`call_processor`), logprob postprocessing, and reward calls all run as asyncio
  tasks in that one process, so it is also a GIL/CPU choke point. For VLM at many
  sampler nodes, CPU/GIL tends to be the wall before raw RAM.
* `--fully-async-max-completed-queue-groups` is therefore effectively a
  **hard-coded single-node memory budget**, not something you can grow your way out
  of by raising it.

Planned direction (deferred — current setup is for testing / small node counts):

1. **Memory-aware dynamic cap.** Replace the hard-coded queue cap with a cap derived
   from a live memory estimate (queued sample bytes vs. a budget), instead of a fixed
   group count.
2. **External durable queue / artifact store.** Move completed groups off the actor
   heap (Ray `ObjectRef`s with spill, or an external durable queue / artifact store),
   which also buys producer/consumer decoupling and restart/replay.
3. **Formal distributed scheduling.** Shard the worker into per-node rollout actors
   that materialize samples locally and hand the trainer references, distributing both
   memory and CPU.

To know which wall you hit first, watch the manager process RSS, the queued-sample
byte total, and the manager-node CPU% as sampler nodes are added.

## Config Differences (2 Key Points)
To enable the fully async pattern there are only two changes compared to a normal run:

1. Use the async training driver: `train_async.py` (not `train.py`).
2. Set the rollout function path:
	```bash
	--rollout-function-path fully_async_rollout.generate_rollout_fully_async
	```

Why is it still "fully" async although `train_async.py` itself schedules rollouts step‑by‑step?

Because the real generation work is done by a **persistent background worker** created in `generate_rollout_fully_async`. Each call from `train_async.py` only drains already completed samples from the worker's output queue; the worker has been continuously generating since the first call. Thus rollout production (model inference) and training consume happen in parallel with minimal waiting.
