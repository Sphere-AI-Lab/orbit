# WekaFS `/data` read hangs on slinky GPU nodes

Slack summary:

> Could an admin check WekaFS client/backend health for `/data` on slinky GPU nodes?
> On 2026-07-01 UTC we confirmed real data reads from `/data` hanging on
> `slinky-2` and `slinky-50`. Metadata (`stat`/`ls`) still returned, but large
> reads hung in the Weka client / uninterruptible D-state, which wedged a
> multi-node GPU job during model weight load. Repro below.

## Ask

Please check WekaFS client/agent health on `slinky-2` and `slinky-50`, and any
backend-side telemetry for the same window. We suspect this may be broader than
a fixed bad-node list, but the confirmed repro evidence is on `slinky-2/50`.

## Confirmed impact

- Job: `21623`, run dir:
  `/data/home/xiuyul/workspace/miles-imp/runs/geo3k-async-mt-pf2-8b/260630_232759`
- Nodes: `slinky-2`, `slinky-50`, `slinky-19`
- Time window: `2026-07-01 00:00-00:06 UTC`
- GPU/IB/Ray were healthy before the stall:
  - healthcheck OK on all nodes
  - NCCL all-reduce PASS, `ranks=24`, `max_busbw=254.9 GB/s`
  - Ray cluster assembled
- Failure point: SGLang engines reached `Load weight begin` while reading model
  weights from `/data`; weight load did not complete before the job was killed.

Post-hoc read probe result on the same `/data` file:

| Node | Result |
|---|---|
| `slinky-2` | `timeout 15 cat .../libtorch_cuda.so > /dev/null` returned `rc=124` |
| `slinky-50` | same probe returned `rc=124` |
| `slinky-19` | same probe returned `rc=0` at that sample time |

## Minimal repro

Run on a suspect node, for example:

```bash
srun --overlap -w slinky-2 bash
```

Then:

```bash
# 1) Confirm /data is WekaFS.
findmnt -T /data

# 2) Look for tasks stuck in the Weka client.
ps -eo stat,pid,etime,comm,wchan:64,args \
  | awk '$1 ~ /^D/ && ($5 ~ /weka|commit/ || $0 ~ /\/data/)'

# 3) Time a real data read from /data. Metadata-only checks are not enough.
time timeout 20 dd \
  if=/data/shared/conda/miniconda3/envs/miles/lib/python3.12/site-packages/torch/lib/libtorch_cuda.so \
  of=/dev/null bs=1M count=64 iflag=direct status=none
echo "rc=$?"
```

Expected healthy result:

- Step 2: no persistent `D`-state rows in Weka-related wait channels
- Step 3: `rc=0`, normally well under a few seconds for 64 MiB

Bad result seen:

- Step 2: persistent `D` rows in `wekafs_*` / `commit_blocking_request`
- Step 3: `rc=124` from `timeout` (read did not return), or very slow throughput

## Notes

- `stat` and `ls` can look fine because they hit cached metadata; the issue is
  real file data reads.
- A wedged read leaves jobs stuck in `COMPLETING` because D-state processes
  cannot be killed until the kernel I/O returns.
- We are not asking admin to debug the training application here. The app-level
  run only provides a reproducible symptom: model weight reads from `/data`
  stopped progressing while GPU/IB/Ray checks had already passed.
