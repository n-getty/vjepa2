# Scaling V-JEPA 2.1 CPT to 256 nodes — status

See also `THROUGHPUT_RECIPE_AURORA.md` for the settings themselves, what each
is worth, and the list of knobs already tested to nothing.

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

## Weak-scaling efficiency

Reproduce with `python scripts/scaling_efficiency.py --preset`. Window: epoch 1,
iters 1-3, **max over ranks** (a synchronous step costs what its slowest rank
costs), **only iterations where every rank logged**.

| run | ranks | median s | clips/s | clips/s/tile |
|---|---|---|---|---|
| 16n fixedshape (bs2, ckpt **ON**, 2B) | 192 | 9.94 | 38.6 | 0.201 |
| **64n lbA8** (bs1, ckpt off, 2B) | 768 | 7.73 | 99.3 | **0.129** |
| **256n lbA8** (bs1, ckpt off, 2B) | 3072 | 9.02 | 340.5 | **0.111** |

**64n → 256n weak-scaling efficiency: 86% per-tile, 3.43x aggregate for 4x the
nodes.** These two rows are the only valid scaling pair here — `diff` of their
`params-pretrain.yaml` is empty apart from topology.

**The 16n row is NOT comparable** and must not be read as "16n is faster per
tile": different per-rank batch (2 vs 1), activation checkpointing ON, and a
different config lineage. Per-tile clips is exactly the metric bs inflates.

**Confidence: low-to-moderate — n=3 iterations.** Full-rank coverage exists only
for iters 1-3 at 256n (job 8730678 died to the log funnel at iter 11, and from
iter 4 on only 192 of 3072 ranks logged). Within-run CV is ~81%, so 86% ±
a lot. It is the honest number available today; the sustained run replaces it.

Three traps, each of which produced a *wrong and plausible* number first:

- **Key on `(epoch, itr)`, not `itr`.** `itr` restarts per epoch; pooling on it
  mixes cold and steady-state iterations. This alone reported the 16n run at
  171 s/iter instead of 10.3 s.
- **Sample a fixed FRACTION of ranks, or all of them — never a fixed count.**
  Max-over-192-of-3072 misses stragglers that max-over-768-of-768 catches. This
  made 256n look *superlinear* (0.069 vs 0.052 clips/s/tile), which is not
  physical for a comms-bound run and is what exposed the bug.
- **Match the window.** A long run is mostly steady state; a 30-iter shakeout is
  mostly warmup. 16n epoch 1 medians 18.0 s against 8.8-10 s in later epochs, so
  whole-run medians favour whichever run is longer, independent of scale.

Sanity check that the corrected script is right: it recovers 21.40 s for the
accum A/B's accum=1 arm, matching the independently-recorded `ring_16M` median
from a different job to 3 significant figures.

## Open

- ~~CCL knob sweep~~ — **DONE at 64n** (job 8732160), all three arms
  indistinguishable: double_tree 1.00x, 64 MB chunk 0.97x, IQRs overlap
  everywhere. Keep ring / 16 MB. This tested the real concern — every CCL
  decision here was made at 16n, and `app/main_dist_aurora.py:154-157` says so
  about itself ("intra-node fanout overhead exceeds inter-node savings **at only
  16 nodes**") — and the answer is that ring-vs-tree does NOT flip at 63 hops.
  `CCL_CHUNK_SIZE` had never been A/B'd anywhere; 16 MB is now validated.
  Re-testing at 255 hops is optional, not blocking: backward is ~57% of iter time
  and accum=2 already halves the collective count, so Amdahl caps any algorithm
  win at <=9%. One loose thread if a block ever opens — double_tree's range was
  tighter (6.4-31.7 vs ring 5.7-46.5), hinting at better TAIL behaviour, which is
  what matters for the straggler-driven desync. n=20, overlapping IQRs, so not a
  claim.
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
