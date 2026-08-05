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

**Throughput: get 2 clips/rank/step through ONE collective — and do it with
`batch_size: 2`, not accum.** Two results, in order:

- accum=2 vs accum=1 is **+36%** (job 8731439, paired, n=25, IQRs disjoint):
  35.4 → 48.2 clips/s. But that compares against *half* the global batch, so it
  shows "2 clips per collective beats 1", not "accum beats bs=2".
- At **matched** gb=1536 (job 8735877, 64n, n=19/arm), bs=2 median 24.92 s /
  61.6 clips/s / **4.6 GiB** min l0-free vs accum 32.36 s / 47.5 clips/s /
  **0.8 GiB**. 1.30x on the median with overlapping IQRs — throughput suggestive,
  headroom decisive, nothing favoring accum.

Still a lower bound for 256n either way, since each avoided allreduce costs 63
ring hops at 64n vs 255 at 256n. The bs-vs-accum *ranking*, though, is not
scale-free: accum's per-step overhead is fixed while the collective saving grows
with node count, so re-measure rather than extrapolate.

**Taken, 2026-08-05.** The 256n path used to take *neither* lever (`lbA8` is
`batch_size: 1`, launcher default `VJEPA_TRUE_ACCUM=1`) — gb=3072 with one
collective per clip, the worst compute-to-comms ratio available.
`scripts/vitG384_256n_daos.sh` now defaults to **`vitG384_lbA`**: `batch_size: 2`
with `lr`/`ema`/`warmup`/`lambda` all derived for the gb=6144 that 3072 ranks x 2
produces. Samples seen is unchanged (624 x 6144 == 1248 x 3072 == 3.83 M), so
this is a throughput-and-schedule change, not a corpus-budget change. The
launcher asserts `world x bs x accum` matches the config's derived gb and exits
on mismatch. See `THROUGHPUT_RECIPE_AURORA.md` for the table and the OOM
fallback (`VJEPA_PER_RANK_BS=1 VJEPA_TRUE_ACCUM=2`, same gb).

**Validated on hardware at 2 nodes** (job 8736104, `debug`, rc=0, 20 iters in
646 s, 24 ranks): the shared env fragment sources on a compute node with no
oneCCL enum rejection on any of the 24 ranks (the empty-`CCL_KVS_MODE` form
killed 8731004 at iter 0), both DAOS containers mount, `lbA` is picked up with
`bs=2 / lr 2.1e-4 / ema 0.988`, loss falls 0.343 → 0.329, and `l0-free` sits flat
at 14.5 GiB. The gb assertion also fired correctly here and was overridden on
purpose: 24 ranks x 2 = 48 against a schedule derived for 6148, so the run was
launched with `VJEPA_SKIP_GB_CHECK=1`. That makes it a **mechanics** test — it
says nothing about throughput (2 replicate hops, not 255) or about the schedule.

**Throughput confirmed at 64 nodes** (job 8736153, `debug-scaling`, rc=0, 30
iters, 768 ranks): 61.0 clips/s and 7.82 GiB min `l0-free` against the reference
bs=2 arm's 61.6 / 4.62 on the common window, ratio 0.989x with overlapping IQRs.
Indistinguishable, which is what a refactor that preserves the recipe should
look like. See `THROUGHPUT_RECIPE_AURORA.md` for the table and the note that the
headroom difference is tail, not level.

Whether gb=6144 *trains* as well as gb=3072 is a separate, still-open question —
the `lbA`/`lbB` capacity A/B. This change only makes the schedule self-consistent
at whatever batch is run.

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

For the bs / checkpointing interaction — they are independent knobs, bs2+ckpt-off
runs and is the fastest config measured (0.233 clips/s/tile), and 256n uses bs1
for **global-batch** reasons rather than memory — see the dedicated section in
`THROUGHPUT_RECIPE_AURORA.md`.

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
