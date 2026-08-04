# Scaling V-JEPA 2.1 CPT to 256 nodes — status

Working notes from the 2026-08-03/04 session. Written so the next person does not
re-derive the dead ends. Job IDs are given for everything so claims can be checked.

## The original blocker was misdiagnosed

The record said 256n was "corpus-blocked: 0/16 sources clear the 3072-shard
threshold." That was an artifact of one arithmetic choice, not a property of the
data path.

`scripts/stage_node_shards.py::_shards_for_node` partitioned by the **global**
world size (`nodes*12`) and replicated any smaller source onto every node. At
256n that meant `ws=3072` > every source's shard count, so the whole 3.1 TB
corpus targeted each node's ~503 GB `/tmp` and the preflight aborted.

But the loader never needed 3072 shards per source: every launcher sets
`WDS_LOCAL_SLICING=1`, and `webdataset.py::_make_stream` then slices by **local**
rank/world (12). `_load_or_build_metadata` re-lists whatever the node actually
holds. The staging partition was always a free parameter.

`--partition-mode nodes` fixes it: 130 GB/node at floor 24, 264 GB at floor 48.

## What actually blocked it, in order of discovery

| # | problem | measurement | fix |
|---|---|---|---|
| 1 | staging partition math | 66 TB/node-set at 256n | `--partition-mode nodes` |
| 2 | Lustre staging bandwidth | **0.43 GB/s/node**, ~6.9 GB/s aggregate; 66 TB = up to 2.7 h before iter 1 | move corpus to DAOS |
| 3 | Lustre *read* bandwidth | **2.71 GB/s** at 192 ranks — *below* the 3 GB/s streaming demand | DAOS: **25.07 GB/s** (9.3x) |
| 4 | init checkpoint still on Lustre | 3072 ranks x 28.2 GB = up to 85 TB; ranks cleared load at ~69/min | second DAOS container |
| 5 | stdout funnel | 442,649 lines through MASTER_ADDR, which also served rendezvous + DAOS keepalives → 120 s ping timeout killed job 8730678 | `mpiexec --outfile-pattern` → **68 lines** |

Each was invisible at the scale below. (2)-(3) only bite when 3072 ranks pull at
once; (4) is fine at 192 ranks; (5) produced ~24k lines at 16n and nobody noticed.

## Current state

**Data path: done and verified.**
- Corpus: `AuroraGPT/vjepa_surg_wds`, 16/16 sources, 13,215 tars, >=99% bytes,
  zero symlinks. Ingest job 8730316 (~55 min, 8 nodes).
- Weights: `AuroraGPT/vjepa_models`, `torch.load` in 29 s, 590 encoder tensors.
- Read benchmark job 8730476: DAOS 25.07 GB/s vs matched-file Lustre 2.71 GB/s.

**Training verified at 64 nodes** (job 8730919): rc=0, 30 iters, **0 DAOS ping
timeouts**, 78-line PBS log.

**256 nodes reached training once** (job 8730678): loss 0.330/0.339 at ~8 s/iter —
matching the 16n baseline — before the log funnel killed the head node. That
funnel is fixed and the fix is measured at both 16n and 64n.

**Throughput: accum=2 is +36%** (job 8731439, paired same-nodes, n=25 each,
**IQRs disjoint**): 35.4 → 48.2 clips/s. Lower bound for 256n, since each avoided
allreduce costs 63 ring hops at 64n vs 255 at 256n.

## Open

- **CCL knob sweep at 256n** (`scripts/ccl_knob_sweep.sh`) — ring vs double_tree
  vs 64 MB chunk. Every CCL decision in this repo was made at 16n, and
  `app/main_dist_aurora.py:154-157` says so about itself: rabenseifner was
  rejected because its "intra-node fanout overhead exceeds inter-node savings **at
  only 16 nodes** — ALCF's large scale recs target 64+ nodes." `CCL_CHUNK_SIZE`
  has never been A/B'd anywhere.
- **Sustained 256n run** — `VJEPA_SUSTAINED=1` in `scripts/vitG384_256n_daos.sh`.
- **Does the large batch train well** — separate from throughput; the
  `lbA8`/`lbB8` capacity arms.

## Two things that are NOT technical blockers but govern the schedule

**256-node blocks are scarce.** Of four requests, two were dequeued with "Not
enough free nodes available" (8730185, 8731534). Budget queue time independently
of anything in the code.

**HSDP did not fix the runtime hang.** It fixed the DDP L0-headroom wedge (57.6 →
22.0 GB) and the *startup* collective hang (transport). The silent fabric hang
persists **under** HSDP — root-caused from 384 stacks (job 8645821) as an FSDP
collective desync, which `docs/vitG_2B_HSDP_findings.md` calls "the HSDP ZERO2
grad-AllReduce over the 16-node replicate dim." It is handled by watchdog →
forensics → resubmit, never eliminated, and it is straggler-driven on the
**replicate dim** — the dimension that grows 16 → 64 → 256. Survivability is
therefore a requirement at 256n, not hygiene. That layer is now in the launcher.

## The ceiling nobody has hit yet

At `sampling_temperature: 0.5` the inverse-collision effective corpus is
**386,159 clips = 32% of the nominal 1,196,365**. A 256n-scale budget therefore
runs **~9.9 effective epochs against a 4x-fresh rule**, and because effective size
is a property of the sampling distribution rather than the budget, **more compute
makes this worse**. `T~=0.75` fits (4.1x) but pushes `pe_video` to 65% of every
batch. This caps what any scale-up can deliver and is being measured by the 1B
T-sweep. See `scripts/gen_large_batch_configs.py`, which now prints the check.
