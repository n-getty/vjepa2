# V-JEPA 2.1 ViT-G throughput recipe on Aurora (HSDP / DAOS)

What to set, what it is worth, and what was tested and found NOT to matter.
Everything here carries a job ID so it can be re-checked rather than trusted.

Companion docs: `SCALEOUT_256N_STATUS.md` (the 256-node path and its blockers),
`vitG_2B_HSDP_findings.md` (the 16n campaign this builds on).

---

## The recipe

```bash
# data path — the single largest win
--local_data_root /tmp/AuroraGPT/vjepa_surg_wds   # DAOS, not /tmp staging
meta.pretrain_checkpoint: /tmp/AuroraGPT/vjepa_models/vjepa2_1_vitG_384.pt
export WDS_LOCAL_SLICING=0        # MUST flip to 0 on DAOS. See "silent traps".

# comms — get 2 clips/rank/step through ONE collective. Two ways to do it;
# at matched global batch bs=2 wins 1.31x AND leaves 5.5x more headroom (8735877).
batch_size: 2                     # PREFERRED. Not VJEPA_TRUE_ACCUM=2.
# export VJEPA_TRUE_ACCUM=2       # same comms saving, strictly worse. Use only
                                  # when bs=2 does not fit (it does, under HSDP).

# compute
use_activation_checkpointing: false   # +22% at 16n, only affordable under HSDP
VJEPA_USE_XPU_FLASH=1                 # memory lever (-3 GB), not a speed lever

# transport (HSDP; DDP needs the opposite — see traps)
export CCL_PROCESS_LAUNCHER=none CCL_ATL_TRANSPORT=ofi CCL_KVS_IFACE=hsn0
unset CCL_KVS_MODE CCL_KVS_USE_MPI_RANKS
export CCL_ALLREDUCE=ring CCL_CHUNK_SIZE=16777216   # both re-validated at 64n
export CCL_OP_SYNC=1 CCL_WORKER_COUNT=1
export VJEPA_NUM_WORKERS=0
mpiexec ... --no-vni -o "$DIR/rank.%r.out" -e "$DIR/rank.%r.err"   # NO --pmi=pmix
```

## What each lever is worth

| lever | effect | evidence |
|---|---|---|
| **DAOS instead of Lustre reads** | **9.3x** (25.07 vs 2.71 GB/s) | job 8730476, same files/job/hour |
| **DAOS instead of /tmp staging** | removes 66 TB/job, up to **2.7 h** before iter 1 | staging measured at 0.43 GB/s/node (8729208) |
| **2 clips/rank/step via `batch_size: 2`** | 61.6 clips/s and **4.6 GiB** min free | job 8735877, paired, n=19/arm |
| ~~`VJEPA_TRUE_ACCUM=2`~~ | same goal, 1.30x slower on the median (47.5 clips/s) and **0.8 GiB** min free — IQRs overlap, so decided on headroom | job 8735877 — superseded, see below |
| activation checkpointing OFF | **+22% at 16n** (+9% at 1n) | 1n: job 8659973; 16n: fixedshape vs v2, n=5750/8937 |
| HSDP instead of DDP | 57.6 -> 22.0 GB/tile; makes ckpt-off affordable | findings §1-2 |
| per-rank stdout files | 442,649 -> **68** lines through the head node | 16n and 64n |

Lustre's 2.71 GB/s at 192 ranks is *below* the ~3 GB/s a 256n run streams, so
DAOS is not merely faster — Lustre could not have sustained the read load.

## Tested and found NOT to matter (do not re-run)

| knob | result | where |
|---|---|---|
| `CCL_ALLREDUCE` double_tree vs ring | **1.00x at 64n**, IQRs overlap | job 8732160 |
| `CCL_CHUNK_SIZE` 64M vs 16M | 0.97x, IQRs overlap | job 8732160 |
| `CCL_ALLREDUCE` rabenseifner | ~4% slower | 16n, 2026-06-05 |
| 3 "insurance" env flags | falsified; l0-free/l0-ext flat ±5 MiB over 74 iters | job 8643398 |
| `CCL_WORKER_COUNT=4` | first-batch stall, even at 1n | job 8643434 |
| `CCL_OP_SYNC=0` | fatal — wedge from iter 0 | job 8641553 |
| DDP bucket 200 MB | no-op at 8x default | 1n ViT-L |
| predictor `static_graph` | +80 ms = noise | job 8570116 |
| pluggable allocator / `expandable_segments` | falsified / broken on Aurora | job 8643336 |

**Why the CCL knobs cannot be worth much more.** Backward is ~57% of iteration
time, and `TRUE_ACCUM=2` already halves the collective COUNT. Amdahl then caps any
allreduce-algorithm win at **<=9%** on top of accum. That is why the sweep was run
at 64n rather than holding out for a scarce 256-node block.

## Silent traps (each cost a run)

These fail without an error, which is what makes them expensive.

- **`WDS_LOCAL_SLICING` must be 0 on DAOS.** It exists because each node's `/tmp`
  held a *different* shard subset. On DAOS every node sees the whole corpus, so
  local slicing makes all N nodes compute the same 12 slices — **N x data
  duplication, no error message.**
- **`CCL_KVS_MODE` must be UNSET, never empty.** oneCCL validates the enum and
  rejects `''`. findings §5a literally recommends the empty string; it is wrong
  for this build (killed job 8731004 at iter 0).
- **The init checkpoint must be on DAOS too.** 3072 ranks x 28.2 GB off Lustre =
  up to 85 TB; job 8730487 spent its whole allocation in `load_pretrained`.
  Invisible at 192 ranks.
- **Never funnel rank stdout to the PBS log at scale.** 442,649 lines through
  MASTER_ADDR — which also serves rendezvous and DAOS keepalives — killed job
  8730678 via a 120 s DAOS ping timeout.
- **`--no-vni` or DAOS RPCs fail** with NA_HOSTUNREACH.
- **`libpil4dfs` OFF** — hangs FSDP AllGather (DAOS-17499).
- **`glob.glob()` hangs on dfuse** — use `os.listdir()`.
- **Transport is strategy-dependent.** HSDP needs `launcher=none`+`ofi` and NO
  `--pmi=pmix`; DDP needs the opposite. The same ofi block under DDP hung at
  iter 0 (job 8641936).

## Activation checkpointing and per-rank batch are INDEPENDENT

Easy to conflate, because the production 16n config happened to pair bs2 with
ckpt ON. They are separate knobs and ckpt-off is not the price of bs2.

| run | bs | ckpt | med s | clips/s/tile | min l0-free |
|---|---|---|---|---|---|
| 16n fixedshape | 2 | **ON** | 10.44 | 0.192 | 7.0 GiB |
| **16n v2** | **2** | **OFF** | **8.57** | **0.233** | **1.8 GiB** |
| 64n lbA8 | 1 | OFF | — | 0.129 | 10.5 GiB |
| 256n lbA8 | 1 | OFF | — | 0.111 | 11.3 GiB |

**bs2 + ckpt-off runs, and it is the fastest config measured.** `v2` sustained
331 epochs that way. So "bs2 needs checkpointing" is false — the 2B bs2 memory
problem was a **DDP** problem (57.6 GB/tile), and HSDP fixed it by sharding
optimizer state to a 22 GB baseline. Under HSDP both knobs are affordable at once.

**ckpt-off is worth +22% at 16n, not the +9% measured at 1 node** (job 8659973,
1 node, bs2 — 7022 -> 6440 ms). Steady state, epoch>=3, full 192-rank coverage,
n=5750 vs 8937 iterations: **10.44 -> 8.57 s, 1.22x, bootstrap 95% CI
1.17-1.26x.** The multi-node gap is expected in direction: ckpt-off removes a
recompute-forward from the backward phase, and backward is where the inter-node
allreduce also lands, so the saving is worth more where backward is a larger
share of the step. Confounded by data-source list (15 vs 16 sources) and by being
different jobs; compute-relevant config (model, 384px, fpcs 16, mask, pred_depth,
bs) is identical, and n is large.

**The real constraint is headroom, and bs2+ckpt-off has almost none: 1.8 GiB
free** against 7.0 (bs2+ckpt) and ~11 (bs1+ckpt-off). bs3 OOMs. Do not read
1.8 GiB as safe — `torch mem:` undercounts true L0 peak by ~12 GB, so size
headroom off `l0-free`/`l0-ext` only.

**Why the 256n config uses bs1 anyway: it is a GLOBAL BATCH decision, not a
memory one.** At 3072 ranks bs1 gives gb=3072 and bs2 gives 6144; bs1 is the only
lever holding the recipe near a batch the schedule was derived for. The per-tile
throughput cost of bs1 is real and is being paid deliberately, with
`VJEPA_TRUE_ACCUM` buying the comms amortization back.

**But accumulation is not the free-in-memory option this doc used to claim.**
Its *activations* are flat; `no_sync` nonetheless holds the unreduced gradient
across sub-batches. Measured on job 8731439 (64n, 48 ranks, 1200 rows/arm,
median l0-free):

| arm | l0-free | min |
|---|---|---|
| bs=1 accum=1 | 10.73 GiB | 9.92 |
| bs=1 accum=2 | **7.10 GiB** | 6.29 |

That is a **3.6 GiB** charge, matching `app/vjepa_2_1/train.py:1169`'s own "2B
bf16 grad ~4GB" estimate, so it is the mechanism and not a measurement artifact.
Both levers cost L0; the question is which costs less at matched global batch,
and that had never been measured — see the next section.

## bs=2 vs bs=1+accum=2 at matched global batch — job 8735877

The +36% accum result compares accum=2 against accum=1, i.e. against **half** the
global batch, so it conflates "amortize the collective" with "do more work per
step". It does not establish that accumulation beats simply raising `bs`.

Job 8735877 (64n, `scripts/bs_vs_accum_ab_64n.sh`) runs both at gb=1536: each arm
moves 2 clips/rank/step and issues one collective per step, so comms volume and
global batch match and the median ratio *is* the throughput ratio. The arms are
gradient-equivalent — `train.py:1189` divides each sub-batch loss by `true_accum`,
giving the same mean reduction `bs=2` does internally — so the only difference is
where the memory goes. Verdict via `scripts/ab_verdict.py`, which is standalone so
the number survives a walltime kill.

### RESULT: bs=2 is 1.30x faster on the median, but the IQRs OVERLAP

Full common window itr 3-21 (n=19), max over ranks, both arms at gb=1536:

| arm | median | IQR | throughput | min l0-free |
|---|---|---|---|---|
| **bs=2 accum=1** | **24.92 s** | 20.2-29.5 | **61.6 clips/s** | 4.6 GiB |
| bs=1 accum=2 | 32.36 s | 28.1-44.4 | 47.5 clips/s | 0.8 GiB |

**By the pre-registered rule this does NOT resolve on throughput alone** — arm 2's
IQR widens to 28.1-44.4 because iters 19-21 hit a tail both arms felt (arm 1
36-37 s, arm 2 44-51 s). That tail is part of the distribution, not an artifact to
trim, so the full window is the honest one. An earlier truncated window (3-19,
before arm 2 finished) gave disjoint IQRs and 1.31x; **do not cite it** — the
extra two iterations are what make the arms overlap.

Three independent things nonetheless point the same way:
- median ratio **1.30x** to bs=2, replicated at 1.31x on the shorter window;
- **wall clock**: arm 1 finished 975 s **cold** (first DAOS touch + 28 GB ckpt
  read) vs arm 2's 1009 s **warm**, same 22 iters — the challenger carried the
  handicap and still won;
- **headroom**: 4.6 vs 0.8 GiB min free (below), which is not close.

Both mechanisms deliver the identical comms saving (one collective per 2 clips),
so any remaining gap is pure overhead: accum pays a second loader fetch and a
second forward/backward per step where bs=2 batches the work into one.

**Consequence: prefer `batch_size: 2`; `VJEPA_TRUE_ACCUM` is the fallback.** The
throughput case is suggestive rather than resolved, but it never favors accum, and
the memory case is decisive. Use accum only where per-rank `bs` cannot rise —
under HSDP with ckpt off, nowhere we have measured. The earlier +36% stands as
measured but does not mean what the recipe used it to mean.

To resolve the throughput half properly, rerun with more iterations (the ipe=22
budget was set by the 1 h wall, and n=19 is thin against 81% within-run CV).

**Memory half — same conclusion.** All 768 ranks, same
job, same nodes, same hour, iters ≥3:

| arm | l0-free median | min over ranks |
|---|---|---|
| bs=2 accum=1 | **12.66 GiB** | 4.62 |
| bs=1 accum=2 | 8.89 GiB | **0.84** |

At matched global batch **bs=2 is the roomier config**, not the tighter one. The
accum arm's tightest rank has 0.84 GiB free — at the OOM edge at 64 nodes, and
`no_sync` holds the unreduced gradient whose replicate dim only grows toward 256n.
So the headroom reasoning that motivated preferring accum over `bs` points the
other way once the comparison is fair. Compare l0-free only *within* a job;
absolute MiB across jobs differ by corpus slice and allocator state.

Both arms confirmed to have actually engaged their knob: arm 2's `rank.0.out`
carries `TRUE gradient accumulation ON: true_accum=2, per-rank batch=1`. Commit
`8317abe` now logs the resolved knobs unconditionally so the control arm is
checkable too, rather than inferred from timing.

## Caveats — they apply to `bs: 2` now, not to accum

The global-batch caveat is a property of **2 clips/rank/step**, not of the
mechanism, so switching from accum to `bs: 2` does not escape it. Either way
global batch doubles (3072 -> 6144 at 256n), which shifts EMA/warmup/lambda —
raw STEP counts with no batch awareness, so they must be re-derived
(`scripts/gen_large_batch_configs.py --batch-mult 16`). This is **throughput
only**; whether the larger batch *trains* as well is the separate `lbA8`/`lbB8`
question.

The **lower bound at 256n** argument also survives the switch, and for the same
reason: both mechanisms halve the allreduce count, and each avoided allreduce
costs 63 ring hops at 64n vs 255 at 256n. What does NOT carry is the ranking
between them — accum's extra per-step overhead is fixed work while the collective
saving grows with node count, so re-measure the 1.31x at 256n rather than
assuming it holds ([[scale-dependent-results-dont-transfer]]).

## Measurement notes (these bit repeatedly)

- **Use wall-clock `iter-time(ms)`. Never `backward-ms`.** PhaseTimer takes XPU
  event deltas and they go **negative** on stalled collectives (-139/-326/-277 s
  at itrs 0/12/24 of job 8730919) — i.e. it breaks precisely on the iterations a
  comms experiment is about.
- **Pair the arms in one allocation.** Within-run CV is ~81%. Across jobs the
  *median* reproduces to ~1% (21.68 s vs 21.40 s for the same config in jobs
  8731439 and 8732160), but pairing removes the fabric-hour confound for free.
- **Require disjoint IQRs before claiming a winner**, and register the threshold
  before the second arm reports. Under this rule accum=2 resolved and every CCL
  knob did not.
- **Equal-length arms.** ipe=60 let arm 1 eat the walltime and truncated arm 2
  (job 8731332).
- **Judge on CSV rows, not exit codes.** A `| tail` pipeline returns tail's
  status; runs have reported rc=0 with zero iterations.

## Survivability is a throughput lever at scale

HSDP fixed the DDP memory wedge and the *startup* collective hang. It did **not**
fix the runtime FSDP collective desync — findings calls that "the HSDP ZERO2
grad-AllReduce over the 16-node replicate dim", handled by watchdog -> forensics
-> resubmit, never eliminated. It is straggler-driven on the **replicate dim**,
the dimension that grows 16 -> 64 -> 256.

At 16n it cost ~1 event per 3.5-4 h, all self-healed. Scaled by node count that is
~14 min at 256n (an extrapolation, not a measurement — the one 64n run saw zero).
Either way, a 1.36x throughput win is worth nothing if the first hang ends the
run, so `scripts/vitG384_256n_daos.sh` carries `VJEPA_SUSTAINED=1`: stall watchdog
(1800 s, above the ~544 s worst recoverable spike), SIGUSR1 stack forensics,
`ipe=30` so each epoch banks in ~5 min, and a guarded self-resubmit that refuses
to relaunch when no epoch was banked.
