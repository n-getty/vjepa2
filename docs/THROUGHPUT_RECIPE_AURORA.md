# V-JEPA 2.1 ViT-G throughput recipe on Aurora (HSDP / DAOS)

What to set, what it is worth, and what was tested and found NOT to matter.
Everything here carries a job ID so it can be re-checked rather than trusted.

Companion docs: `SCALEOUT_256N_STATUS.md` (the 256-node path and its blockers),
`vitG_2B_HSDP_findings.md` (the 16n campaign this builds on).

> **READ THIS FIRST (2026-08-06).** Most of the levers below are comms levers, and
> comms is **not** where the time goes. The ladder measured `iter −
> max_over_ranks(dataload)` at **2.88 s (1n) / 3.74 s (2n) / 3.31 s (64n)** — the
> compute-plus-comms floor is FLAT, with a 3.29–3.35 s IQR over 29 iterations at
> 768 ranks. At 1 node, compute is 2.92 s of a 16.99 s iteration: **17%
> utilization with no inter-node fabric in the picture at all.** The remaining 83%
> is the intra-node decode tail (p10 0.48 s vs p90 13.88 s across 12 ranks on ONE
> node), and it does not get worse with scale — the dataload p90 *falls* from
> 13.88 s at 1n to 6.54 s at 256n, which rules out DAOS bandwidth saturation.
> The stall is **overlappable I/O latency**: `num_workers=2` is **1.95× total
> wall** at 2n (reproduced on a second allocation), and the barrier column proves
> it — at nw0 ranks wait 11.35 s in the pre-step barrier against an 11.71 s
> dataload. Still **not a default**: 2 of 24 ranks throw at teardown (0 of 24 at
> nw0), and on short rungs the worker-fork overhead cuts the win to 1.27×.
> Before tuning another CCL knob, read "Where the time actually goes".
>
> **AND (2026-08-06, job 8740093, first valid 1n baseline):** the 1n→8n loss to
> 57% per-tile is an **order statistic, not a degradation**. The per-rank dataload
> distribution is *identical* at every rung (p50 1.12/1.21/1.18/1.20 s,
> p(>10 s) ≈ 0.098 throughout); only max-over-ranks grows, because a synchronous
> step waits for the slowest of N draws from a heavy tail. `max dataload + 3.2 s`
> predicts the wall at every rung to within 3%. **Adding nodes buys more lottery
> tickets on the same tail; it makes nothing slower.** Reducing the p99 decode is
> the only lever that changes the asymptote — nw=2 lowers the stall *rate*
> (0.098 → 0.030) but not the shape. See "L1a ladder, 1n→8n".

---

## The recipe

**As of 2026-08-05 this is no longer something you copy — it is the default.**
The env half lives in `scripts/lib/aurora_hsdp_env.sh`; source it and put your
overrides after. The config half is `vitG384_lbA` (bs=2), which
`scripts/vitG384_256n_daos.sh` now defaults to.

```bash
source $ROOT/scripts/lib/aurora_hsdp_env.sh   # HSDP ONLY -- see the traps below
```

Validated end-to-end on hardware, twice:

- **Mechanics, job 8736104** (2 nodes, `debug`, rc=0, 20 iters, 24 ranks):
  fragment sources clean, no oneCCL enum rejection on any rank, both DAOS
  containers mount, `lbA` picked up at `bs=2`, loss 0.343 → 0.329. Not a
  throughput datapoint — at 2 nodes the replicate dim is 2 hops.
- **Throughput, job 8736153** (64 nodes, `debug-scaling`, rc=0, 30 iters, 768
  ranks, `avg. loss 0.326`). Against the bs=2 arm of job 8735877 on the **common
  window 3..21**, both at gb=1536 with 768 ranks:

  | | median max-over-ranks | IQR | clips/s | min `l0-free` |
  |---|---|---|---|---|
  | 8736153 (promoted defaults) | 25.19 s | 20.61–32.16 | **61.0** | **7.82 GiB** |
  | 8735877 bs2 arm (reference) | 24.92 s | 20.17–29.47 | 61.6 | 4.62 GiB |

  0.989x with heavily overlapping IQRs — **indistinguishable**, which is the
  intended result: the refactor was meant to preserve the recipe, not improve it.
  Both hit the plan's ~61 clips/s and ≥4 GiB targets. Over 8736153's own full
  window (3..29) the median is 23.37 s / 65.7 clips/s; the shorter common window
  is the number to quote, since a truncated tail flatters whoever is behind
  ([[ab-window-truncation-trap]]).

  The min-`l0-free` gap (7.82 vs 4.62 GiB) is **tail, not level**: median
  headroom over the same window is 12.63 vs 12.66 GiB and p1 is 8.29 vs 5.36, so
  it is one rank's transient dip rather than a systematic difference. The two
  configs differ only in schedule constants (lr/ema/warmup/lambda/ipe, the
  lbA-vs-lbA8 derivation) — nothing compute- or memory-relevant. Do not read the
  gap as the refactor buying headroom.

That fragment is exactly the block below. It is written out here because the
*reasoning* is the useful part; the file is the thing that actually runs.

```bash
# data path — the single largest win
--local_data_root /tmp/AuroraGPT/vjepa_surg_wds   # DAOS, not /tmp staging
meta.pretrain_checkpoint: /tmp/AuroraGPT/vjepa_models/vjepa2_1_vitG_384.pt
export WDS_LOCAL_SLICING=0        # MUST flip to 0 on DAOS. See "silent traps".

# comms — get 2 clips/rank/step through ONE collective. Two ways to do it;
# at matched global batch bs=2 wins 1.30x AND leaves 5.5x more headroom (8735877).
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

### Two rules for editing the fragment

- **`${VAR:-default}` is the right guard for most things and WRONG for the FI
  variables.** Aurora's system profile already exports `FI_PROVIDER`,
  `FI_CXI_RX_MATCH_MODE` and `FI_CXI_OFLOW_BUF_SIZE`, so a `:-` guard on those
  inherits the system value and silently discards the recipe — the exact opposite
  of the guard's purpose. They are hard-set. Before adding a `:-` to anything
  new, check `env | grep <VAR>` in a clean login shell.
- **Never `set -u`** — Lmod's init trips it.

### Taking the throughput without breaking the schedule

bs=2 doubles the global batch, and `lr`/`ema`/`warmup`/`lambda_*_iter` are all
DERIVED for one specific global batch. So the default config moved too:

| | `vitG384_lbA8` (old default) | **`vitG384_lbA` (now default)** |
|---|---|---|
| per-rank bs / gb @ 3072 ranks | 1 / 3072 | **2 / 6144** |
| lr | 1.5e-04 | **2.1e-04** |
| ema | 0.994 | **0.988** |
| warmup | 3.2 ep | **1.6 ep** (62 steps either way) |
| lambda ramp | 62 / 188 | **31 / 94** |
| ipe x epochs | 39 x 32 = 1248 | **39 x 16 = 624** |
| samples seen | 3.83 M | **3.83 M** (invariant) |

Samples-seen is identical, so the replay / 4x-fresh position does not move. At
3072 ranks `lbA` is a *native* gb=6144 config, not an emulation of one — the two
files are byte-identical outside `batch_size` and those constants.

`scripts/vitG384_256n_daos.sh` asserts
`world_size x batch_size x true_accum == the gb the config was derived for` and
exits on mismatch (override: `VJEPA_SKIP_GB_CHECK=1`). It recovers the derived
value as `3,836,160 / (ipe x epochs)`, since the whole family is built to hold
samples-seen at the 16n baseline. What it catches:
`scripts/submit_large_batch_arm.sh` pairs `lbA` with `ACCUM=16`, which is right
at 16 nodes (accum *emulating* gb=6144) and gives **gb=98304** if the same
pairing is pointed at 256.

**If bs=2 ever OOMs at 256n**, the fallback keeps gb=6144 rather than switching
config: `VJEPA_PER_RANK_BS=1 VJEPA_TRUE_ACCUM=2`. Falling back to `lbA8` instead
would run gb=6144 against gb=3072 constants — the assertion rejects it.

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
  iter 0 (job 8641936). So **do not source `scripts/lib/aurora_hsdp_env.sh` into
  a DDP launcher.** The `vitG1B_capwd_*.sh` family sets
  `VJEPA_DIST_STRATEGY=ddp` and its `pmix`/`mpi` transport is correct as written
  — an audit flagged it as off-recipe and was wrong.
- **The resolved `num_workers` used to be unrecoverable from a run's own
  artifacts.** `scripts/lib/aurora_hsdp_env.sh:85` exports
  `VJEPA_NUM_WORKERS=0`, which overrides config `num_workers: 2`, but
  `params-pretrain.yaml` snapshots only the *config* value. So a finished run's
  worker count could only be recovered from its launcher **at its submit-time
  git revision**. Fixed 2026-08-06: `train.py` now prints the resolved
  `num_workers=` and `pin_mem=` in the `THROUGHPUT KNOBS` line. Runs before that
  date must have their launcher checked out at the submit commit.
- **A launcher that rewrites `ipe` must preserve TOTAL STEPS.**
  `VJEPA_SUSTAINED=1` shortens the epoch so each one banks durably; it used to
  keep `epochs` while overwriting `ipe`, turning `lbA`'s 39 x 16 = 624 into
  30 x 16 = 480 — 2.95 M samples instead of 3.83 M, run against an EMA/warmup/
  lambda schedule derived for 624. Fixed to rescale `epochs`, and to rescale
  `warmup` too (it is specified in epochs but consumed as
  `int(warmup * ipe)`, `app/vjepa_2_1/utils.py:493`, so a shorter epoch silently
  shortens warmup in steps).

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

### Why bs=2 is the ceiling, and why free memory at small scale does not change that

Asked 2026-08-05, after a 2-node run showed ~14 GiB free and looked like room for
a bigger batch. It is not. Three separate things have to be true to raise bs, and
none of them is:

**1. bs=3 was measured and OOMs.** Job leg L4 in
`xpu_flash_attention_porting_guide.md` (1 node, ckpt-off, flash-ON): bs=2 is
43.2 GB torch, bs=3 is 52.1 GB torch → `UR_RESULT_ERROR_OUT_OF_RESOURCES` at FSDP
`_mp_shard.copy_`. With the ~12 GB undercount that is ~54 GB real vs >64 GB on a
64 GB tile. XPU flash frees 3 GB — an order of magnitude short. That test ran at
**1 node, where headroom is most generous**; every larger topology is tighter.

**2. `l0-free` falls with node count, so a small-scale reading does not transfer.**
Compute-identical configs (`lbA` and `fixedshape_v2` are byte-identical in model /
crop / fpcs / mask / loss / ckpt), min over ranks:

| nodes | bs | ckpt | min `l0-free` | job |
|---|---|---|---|---|
| 2 | 2 | off | **11.9 GiB** | 8736104 |
| 16 | 2 | off | **1.8 GiB** | v2 |
| 64 | 2 | off | **4.6 GiB** | 8735877 |
| 64 | 2 | off | **7.8 GiB** | 8736153 |
| 256 | 2 | off | **7.1 GiB** | 8736390 |

HSDP shards optimizer state intra-node (12 tiles) and replicates inter-node, so
what grows with node count is the replicate dim and its fabric transients. A
2-node run is the *most* headroom that will ever be observed — sizing a batch off
it is [[scale-dependent-results-dont-transfer]] in its most expensive form.

**What 256n actually showed (job 8736390), which is narrower than "headroom
shrinks":** the floor did *not* keep falling — 7.1 GiB at 3072 ranks against
7.8 GiB at 768 in the run right before it. The pessimistic reading of the table
above would have predicted worse. What the 256n data adds is the *shape* of the
distribution: the per-rank median is 14.2 GiB and 3064 of 3072 ranks sit near it,
while **8 ranks sit at 7-9 GiB for all 50 iterations** — a stable per-rank offset,
not a transient dip (rank 2735 was at 7276 MiB on every single iteration). So the
spread is ~7 GiB wide and one-sided, and it is those 8 ranks, not the median, that
a bigger batch has to fit. The 2n → 16n → 64n column is best read as "the minimum
is set by a few persistently-tight ranks whose number grows with scale", which is
why min-over-ranks is the only safe statistic. bs=3 needs ~9 GiB more than bs=2
per point 1; the tight ranks do not have it at any measured scale.

**3. Read the MIN over ranks, not rank 0.** On job 8736104 rank 0 reported 14,564
MiB while the tightest rank (12) held 11,943 MiB — a 2.6 GiB spread across 24
ranks. The rank that OOMs is the tightest one, and the spread widens with scale.

**Even with memory to spare, the return is small and the risk is not.** The lever
is clips-per-collective: 1→2 halves collectives per clip, 2→3 removes only another
1/6 of the original, against backward at ~57% of step time. And bs is also the
global-batch knob — at 3072 ranks bs=3 means gb=9216 with no derived schedule,
when gb=6144 itself is not yet shown to train well (the `lbA`/`lbB` A/B). The
spare headroom's job is absorbing fabric transients on the replicate dim, which
is exactly what grows on the way to 256 nodes.

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

- **Use wall-clock `iter-time(ms)` for headline numbers** — but the phase columns
  are usable, and an earlier version of this note wrongly said they were not.
  **Negative XPU event deltas are a 32-bit counter wrap, not a broken timer**
  (verified 2026-08-06). The counter ticks at 80 ns, so it rolls over every
  `2**32 * 80e-9 = 343.597 s`; a delta spanning the rollover comes out short by
  exactly one period. Add `WRAP_MS = 343597.38368` to recover it.
  The evidence: across 176,640 rank-rows of the 64n and 256n runs, 6.3–6.6% of
  rows had at least one negative phase, and after adding one period **zero
  remained negative**, while `sum(phases) − gpu-time` kept the same ~0.5 ms
  residual as the never-negative control rows (that residual is just the `"%d"`
  truncation of `gpu-time`). Random garbage does not reconcile to half a
  millisecond. Note the old note's own counter-example fits: −139/−326/−277 s all
  lie inside the single-period range (−343.6, 0) s, which is what a bounded
  artifact looks like.
  `scripts/scaling_efficiency.py` now unwraps at parse time. **Do not oversell
  the fix**: at 64n it moved `backward-ms` p50 12.19 → 12.30 s and the
  max-over-ranks median not at all, because whether a phase wraps depends on
  where the free-running counter sits, not on how long the phase took — so the
  dropped samples were spread through the distribution, not concentrated in the
  slow tail. The gain is closing a 4% silent coverage hole with a systematic low
  bias, not recovering a hidden tail.
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
- **Count rank CSVs against the rank count you expected, from the topology —
  not from the number of files present.** `arm_B_lr6e5` was read as a 1-node
  baseline for a scaling table. It was `scripts/vitG384_v2_lr_ab_32n.sh`
  (job 8681025): `-l select=32`, `WORLD_SIZE=192 mpiexec -n 192 -ppn 12` — a
  **16-node/192-rank** run, of which only `log_r0..log_r11` exist (its mpiexec
  was `Killed`). "Max over ranks" was therefore taken over 6% of ranks, which
  makes a run look fast in exactly the way a straggler study cannot afford.
  `scripts/scaling_efficiency.py --ladder` now derives the expected rank count
  from the rung **directory name** (`n<NODES>_nw<N>`) and reports
  `partial rank coverage` instead of a number.
- **`fwd-target-ms` changed meaning on 2026-08-06.**
  `phase_timer.mark("fwd_target_done")` was immediately followed by
  `mark("fwd_context_done")` at all three call sites, so `fwd-context-ms` was
  always ~0 and `fwd-target-ms` silently held the **entire** forward. The marks
  now bracket real work. **For any CSV written before 2026-08-06, read
  "forward" as `col6 + col7`**; after it the two columns are genuinely the
  target encoder vs the context encoder + predictor.

## Where the time actually goes — ladder result, 2026-08-06

Job 8739625 (1n + 2n hold, `VJEPA_SCALE_PROBE=1`) plus re-analysis of the
pre-existing 64n (8736153) and 256n (`daos_256n/vitG384_lbA8`) runs. Read
`[[scaling-loss-is-decode-tail-not-fabric]]` for the short version.

**1. The compute+comms floor does not grow with node count.**

| | 1n | 2n | 64n |
|---|---|---|---|
| median `iter − max(dataload)` | 2.88 s | 3.74 s | 3.31 s |
| IQR | 2.88–2.90 | 3.08–3.82 | **3.29–3.35** |

A 60 ms IQR over 29 iterations at 768 ranks. Whatever scales with node count is
already inside `max(dataload)`. Do not read a trend into the 2n value being
highest — 0.9 s of spread across three points is not a trend; the finding is the
*absence* of growth.

**2. Only 17% of a single-node iteration is compute.**

| | barrier | dataload | fwd-tgt | fwd-ctx | backward | opt | ema | **iter** |
|---|---|---|---|---|---|---|---|---|
| 1n | 12.60 | 14.11 | 0.67 | 1.00 | 1.22 | 0.03 | 0.00 | **16.99** |
| 2n | 17.25 | 17.60 | 0.69 | 1.40 | 1.72 | 0.03 | 0.00 | **20.68** |

1n→2n per-tile efficiency is 82%, but compute only grows 2.92→3.84 s against a
+3.69 s iteration, so **four fifths of that regression is not compute**. The
backward part of it (1.22→1.72 s) is the HSDP replicate-dim allreduce appearing
for the first time — real, and small.

⚠️ **Do not sum `barrier` and `dataload`.** Both are wall clock and both measure
largely the *same* wait from opposite ends, which is why `unacct` is −12.6 s at
1n. They are two views of one stall, not two additive costs.
⚠️ The 2n rung was still warming (trend 1.16), so its 20.68 s is a lower bound
and 82% is an optimistic ceiling.

**3. The dataload tail does not degrade with scale — it improves.**
Per-rank-sample, iters > 0:

| | 1n | 2n | 64n | 256n |
|---|---|---|---|---|
| p10 | 0.48 | 0.46 | 0.47 | 0.30 |
| p50 | 1.79 | 2.03 | 1.26 | 4.08 |
| p90 | 13.88 | 14.35 | 11.51 | **6.54** |
| p99 | 24.18 | 28.48 | 22.84 | **11.52** |
| mean | 5.29 | 5.28 | 3.62 | **3.52** |

256× the concurrent readers, a *lower* tail than one node. DAOS bandwidth
saturation cannot produce that.

**4. The stalls are whole-node correlated, and present at 1 node.** Per-iteration
at 1n: itr 2 = all 12 ranks 10.4–16.4 s together; itr 4 = 9 of 12 at ~12 s while
ranks 3/6/11 sit at 1.0 s; itrs 3/9/10 = nobody over 10 s. Per-rank means are
3.6–7.5 s with **no persistently slow tile**, so it is not a bad device, and the
correlation rules out independent per-rank shard-content variation.

**5. RESOLVED — it is overlappable I/O latency, not CPU starvation** (job
8739712, rungs `2 2:nw2`, 38 commonly-covered iters, `scripts/nw_arm_report.py`):

| arm | median iter | **total wall** | dataload | iters at 3.6 s floor |
|---|---|---|---|---|
| n2_nw0 | 14.76 s | **624 s** | 11.71 s | **0/38** |
| n2_nw2 | 3.43 s | **312 s** | 0.00 s | **21/38** |

Adding worker processes to a node already running 192 threads on 104 physical
cores made it **2.00× faster in total wall**. If cores were the binding
constraint that could not happen, so hypothesis (a), CPU oversubscription, is
dead. `iter − dataload` is unmoved (3.09 → 3.18 s): compute untouched, prefetch
simply hides the wait.

**Quote total wall (2.00×), not the median (4.30×).** `nw2` is bimodal — a hard
3.2 s floor with dataload exactly 0.00 on 26 of 38 iterations, spiking when the
prefetch queue runs dry. Consequences, all of which bit here:
- the median sits on the floor and overstates the win;
- the **IQR rule reports "not separated"** (9.63–22.19 vs 3.19–11.88) even though
  the arm is twice as fast — that rule assumes unimodal arms. The floor counts
  (21/38 vs 0/38) are the statistic that is not fooled;
- a 21-iteration window said 4.59× with *disjoint* IQRs. The full window said
  2.00× with overlapping ones ([[ab-window-truncation-trap]]).
A synchronous run pays the **sum** of its iterations, so total wall governs.

🛑 **`num_workers=2` is NOT a default — the arm did not exit.** Read the full
timeline before theorising; a first pass over these artifacts got it wrong twice.

| time | event |
|---|---|
| 15:46:53 | all 24 ranks write CSV row 39 — **training completed everywhere** |
| 15:47:02-03 | ranks 12 and 15 (node 2) throw and die |
| 15:47:12 | rank 0 finishes `latest.pth.tar`, 15.0 GB — **checkpoint completed too**, 9 s after those ranks were already gone |
| 16:25:45 | PBS SIGTERM at the walltime cap; rank 0 throws the **same** exception |

The exception in every case is
`terminate called after throwing an instance of 'std::system_error' / what(): No
such file or directory`.

Two corrections to the obvious reading:
- **Three ranks threw, not two.** Rank 0 — which finished the checkpoint and
  looked like a clean survivor — threw the identical exception when PBS killed it.
  So that text is what an ordinary signal-kill produces on this stack, **not a
  signature unique to forked DataLoader workers.**
- **Nothing was blocked by the deaths.** Rank 0 wrote a complete, correctly-sized
  checkpoint after ranks 12/15 were dead.

What is actually specific to `nw2`: **two ranks exited early, on their own, after
their work was done.** Zero `nw0` ranks did. State it that way. Whether the throw
is causal or is itself a symptom of the shutdown path is **not established** — it
was never traced to a frame.

**Do not propose the two obvious mitigations — they were already on.**
`app/main_dist_aurora.py:39` sets `MP_SOCKET_DIR=/tmp` at module load under
`--train_mode`, and `:357` calls `set_sharing_strategy("file_system")` inside
`run_training()`, the ladder's path. The crash happens despite both.

**Fix shipped: `VJEPA_HARD_EXIT`, default on.** `run_training()`'s `finally`
calls `os._exit()` after `destroy_process_group()`. The exit code comes from
`sys.exc_info()`, never a literal 0: this sits in a `finally`, so a hardcoded
`os._exit(0)` would report every real crash as rc=0. Set `VJEPA_HARD_EXIT=0` for
a normal exit when debugging the throw.

### Hardware validation, job 8739914 (2n, `2:nw2 2 2:probe0`) — partial

The rung **returns** now (`rc=0 in 554s`, 24/24 rank CSVs; previously it never
returned). But the throw is **still there**, and the hard exit is **not proven to
be the reason** it now returns.

- 2 of 24 ranks (0 and 22) threw the identical `std::system_error`.
- Rank exits: 21 at 16:52:11, then :13, :24, **:42** — a 31 s straggle.
- Rank 0's stdout logs `clip-diag` at 16:52:16-17, i.e. **DataLoader workers were
  still decoding after `avg. loss` printed.**

That contradicts the mechanism claimed above this section. If the throw came from
a destructor at interpreter shutdown, `os._exit()` would skip it and rank 0 would
have left with the other 21; it left 32 s later. The throw and the straggle occur
during worker-pool teardown *inside* `app_main`, **before** the `finally` is
reached — so the hard exit cannot be what clears them, and may be doing nothing
here. One job cannot separate "the fix worked" from "this run differed"; do not
write it up as validated.

What this run *does* establish, controlled (same allocation, same hour): nw2 threw
on **2/24** ranks, nw0 on **0/24**; exit spread 31 s vs 3 s. The earlier version
of that claim compared across a job boundary.

### Root cause found, fix landed (98ca1ce) — not yet hardware-validated

The hard exit could never have reached this. `unsupervised_loader`
(`train.py:572`) and `loader` (`:861`) are **locals of `main()`**, so
`_MultiProcessingDataLoaderIter.__del__ → _shutdown_workers` fires at `main()`'s
return — inside `app_main`, **upstream of `run_training`'s `finally`**. The exit
is not misplaced, it is *unreachable*. That is consistent with every observation
above: the throw and the 31 s straggle happen while `avg. loss` has printed but
workers are still decoding.

Two changes that are **one fix**:

- **A.** `_LOADER_KEEPALIVE` (module-level list in `train.py`) retains the loader
  past `main()`'s return, so the destructor never runs there and the process
  leaves through `VJEPA_HARD_EXIT` instead. Daemonic workers are then reaped by
  the kernel.
- **B.** `persistent_workers` is now passed to `init_data`, default `True`,
  matching `app/vjepa/train.py:116` and `app/vjepa_droid/train.py:117` — 2.1 was
  the only trainer dropping it, so `data_manager.py`'s `False` default silently
  won and the pool was re-forked at every epoch boundary.

⚠️ **A without B is a no-op, and that is not a style claim.** The worker pool
does not hang off the DataLoader, it hangs off `DataLoader._iterator`, and
`__iter__` only stores it there when `persistent_workers and num_workers > 0`
(`torch/utils/data/dataloader.py:485`). At `persistent_workers=False` every
`iter()` returns a fresh unreferenced iterator, so retaining the loader retains
nothing that owns a worker. Verified empirically through the real
`_LenWrapper → WebLoader → DataLoader` chain, and pinned as a test so B cannot be
reverted as "unrelated tuning" while silently un-fixing A.

Inert on the `nw=0` path — `webdataset.py:787` ANDs with `num_workers > 0`.
`tests/test_loader_teardown_survival.py`: 6 tests, mutation-verified 3/3.

**Still unvalidated on hardware.** In flight: job 8740311 rung `8:nw2`, the first
nw>0 run of the fixed code, at 4× the node count the failure was seen at. And
still needs 64n before nw>0 becomes a default — the failure appeared on node 2 of
2, so it is not obviously scale-free
([[scale-dependent-results-dont-transfer]]).

⚠️ **The ladder's watchdog never armed for either rung** — `pid=$(start_watchdog)`
blocks until the backgrounded subshell exits, so it returned a corpse's pid. That
is why nothing reaped this hang and why the job's third rung was lost; an armed
watchdog would have killed it at 15:53:53. Fixed (`RUNG_WD_PID=$!` + a start grace
period + a `kill -0` arming check). Both fixes are pinned by
`tests/test_aurora_teardown_and_watchdog.py`, mutation-verified.
**Watchdog fix is hardware-validated**: job 8739914 printed `watchdog armed
pid=... (grace 180s, stall 420s, first-iter 900s)` on all three rungs and
completed its whole rung list — the first ladder job to do so. The arming line is
part of the fix: this failure mode is silence, so it has to announce itself.

### Controlled nw2-vs-nw0 numbers (job 8739914, 39 common iters, max-over-ranks)

| arm | med iter | **total wall** | med dataload | dl==0 | med barrier | throws |
|---|---|---|---|---|---|---|
| n2_nw2 | 3.45 s | **302 s** | 0.00 s | 21/39 | 0.10 s | **2/24** |
| n2_nw0 | 14.89 s | **589 s** | 11.71 s | 0/39 | 11.35 s | 0/24 |

**Quote both ratios.** Iteration total wall is **1.95×**, but the *rung* wall was
554 s vs 704 s = **1.27×**: nw2 pays ~252 s of non-iteration overhead against
nw0's ~115 s (worker fork + the 31 s teardown straggle). That fixed cost eats a
third of the win over 40 iterations and amortizes away over production lengths.
Citing only 1.95× oversells short jobs.

**The barrier column settles H3.** At nw0 the median barrier (11.35 s) is
essentially the whole dataload (11.71 s) — ranks sit in the pre-step barrier for
the entire decode. At nw2 it is 0.10 s. So `num_workers=0` decode is *genuine
serialized wait*, not a cost merely relocated into a visible column, which was
the open question in "Where the time actually goes".

### Probe-overhead gate: PASS

`VJEPA_SCALE_PROBE`'s barrier is free, so ladder rungs measure the workload and
not the instrument. Probe ON vs OFF, nw0, same allocation, 39 common iters:

| arm | med | mean | IQR | total wall |
|---|---|---|---|---|
| n2_nw0 (ON) | 14.89 s | 15.11 s | 8.50–21.30 | 589 s |
| n2_nw0_probe0 | 10.63 s | 14.50 s | 8.25–22.08 | 566 s |

IQRs overlap heavily; means differ 4%; total wall 4.2%. **Ignore the median here**
— both arms are bimodal (min 6.1 s, max 33.4 s), so it lands wherever the mode
split falls and shows a 4.3 s "gap" the mean and sum both deny. This was the
blocking gate for the L1/L2/L3 debug-scaling slots; it is cleared.

**Why this is not a probe artifact.** The 64n and 256n CSVs are **16 columns** —
they predate the barrier probe entirely — yet show the same flat floor and the
same non-degrading tail. Findings 1 and 3 reproduce in probe-free data.

**What it means for the levers above.** The comms levers in this doc are real but
they are optimizing 17% of the iteration. Reporting dataload **p50** (~1–2 s at
every scale) hides the cost completely: a synchronous step pays
**max-over-ranks**, not the median. The "64n forward blowup" and "backward
blowup" that motivated this study were skew absorbed by each phase's first
collective — `target_encoder` is HSDP-wrapped, so forward's first FSDP all-gather
eats every millisecond of skew the dataloader created.

## L1a ladder, 1n→8n — the scaling loss is an order statistic

Job 8740093 (`-q debug-scaling -l select=8`, rungs `1 2 4 8`, ipe=40, nw=0,
DAOS, probe on). **Full rank coverage at every rung** (12/24/48/96), 39 common
iterations, iteration 0 dropped. This is the **first valid 1n baseline** — the
historical one was a 16n run with 6% rank coverage.

| rung | ranks | med iter | mean | max dload | max barrier | fwd | bwd | clips/s/tile | eff |
|---|---|---|---|---|---|---|---|---|---|
| n1 | 12 | 10.84 s | 13.05 s | 7.91 s | 7.48 s | 1.70 s | 1.26 s | 0.1846 | 100% |
| n2 | 24 | 14.01 s | 14.53 s | 10.89 s | 8.99 s | 1.70 s | 1.40 s | 0.1427 | 77% |
| n4 | 48 | 16.04 s | 16.88 s | 12.90 s | 12.57 s | 1.71 s | 1.48 s | 0.1247 | 68% |
| n8 | 96 | 19.01 s | 20.43 s | 15.64 s | 15.08 s | 1.72 s | 1.56 s | 0.1052 | 57% |

**The per-rank dataload distribution is identical at every rung.** Pooling every
(rank, iteration) sample:

| rung | n | p10 | p50 | p90 | p99 | mean | p(>10 s) |
|---|---|---|---|---|---|---|---|
| n1 | 468 | 0.47 | **1.12** | 9.93 | 21.58 | 2.97 | 0.098 |
| n2 | 936 | 0.45 | **1.21** | 9.72 | 25.25 | 3.19 | 0.098 |
| n4 | 1872 | 0.46 | **1.18** | 9.84 | 22.32 | 3.13 | 0.099 |
| n8 | 3744 | 0.47 | **1.20** | 9.61 | 22.12 | 3.08 | 0.096 |

Nothing per-rank degrades. What grows is **max-over-ranks**, because a
synchronous step waits for the slowest of N draws from a heavy tail. The
fraction of iterations containing at least one >10 s rank goes
0.46 → 0.51 → 0.67 → **0.79**, and `max dataload + 3.2 s floor` reproduces the
measured wall at every rung to within 3% (11.11/14.09/16.10/18.84 predicted vs
10.84/14.01/16.04/19.01 measured).

**Adding nodes is not making anything slower — it is buying more lottery tickets
on the same tail.** The compute+comms floor (`iter − dataload − barrier`, per
rank) moves 2.93 → 3.07 → 3.15 → **3.22 s** across an 8× node jump. Fabric
tuning cannot address a 0.29 s term, which is why
[[ccl-knobs-settled-at-64n]] found nothing left to tune.

**H6 (warmup transient) is REFUTED.** The per-iteration trace is bursty end to
end, not decaying: n8 hits 47.1 s at iter 9, 3.1 s at iter 38, 20.9 s at iter 39.
Quarter-binned max-dataload does not descend monotonically at any rung. **The
`trend` column is measuring burstiness, not warmup** — it reads 2.89× at *1n*,
where no fabric is involved. Do not read it as "not yet steady state."

⚠️ **Do not sum max-over-ranks columns.** `dataload` and `barrier` are
anti-correlated by construction — the rank that loads fast is the one that waits
longest — so taking each column's max independently and subtracting gives a
nonsense −7.5 s residual at 1n. Decompose **per rank**, then aggregate.

⚠️ **Stalls are node-clustered, so the iid order-statistic model over-predicts**
by 26–40%. At n8 a stall event hits a median of 6 ranks spread over only 4 nodes;
iid would spread 6 over ~6. Consistent with a shared per-node DAOS client, but
that is a hypothesis — not traced. Use the model for the shape of the argument,
not as a quantitative predictor.

**What this means for `num_workers`.** nw=2 does not remove the tail (p99 still
19.1 s vs nw=0's 26.1 s) — it makes it **rarer**, p(>10 s) 0.098 → 0.030. Per
iteration at 2n:

| arm | iters with a >10 s rank | their median wall | others' median wall |
|---|---|---|---|
| nw2 | **5/39** | 30.79 s | **3.39 s** |
| nw0 | **23/39** | 17.79 s | 7.56 s |

When nw=2 does stall it is *worse*, but it stalls one sixth as often and clean
iterations run at the 3.4 s floor instead of 7.6 s. Since the loss is
`1−(1−p)^N` saturating, the nw=0 curve should flatten once nearly every iteration
contains a stall — which is exactly the 64n≈256n flatness already on record.
**nw=2 lowers p but does not change the shape**: it buys a constant factor and
pushes the knee out. Reducing the *tail itself* (p99 ≈ 22 s per-rank decode) is
the only lever that changes the asymptote.

### Chasing the tail: payload size is REFUTED, keyframe spacing is the lead

Since the tail is the only asymptote lever, its owner has to be identified.
Measured offline against the real shards (shard 0 of each source, 4 clips,
`decord` `num_threads=1`, 16-frame `linspace` — the access pattern training
actually uses), so this costs no scaling slot:

| source | MB/clip | frames | resolution | GOP | open | seek-16 |
|---|---|---|---|---|---|---|
| surgtoolloc2022 | 8.9 | 1806 | 640×360 | 12 | 0.01 | **0.05 s** |
| multibypass140 | 32.6 | 1500 | 720×576 | 12 | 0.04 | **0.15 s** |
| heichole_512 | 12.9 | 1319 | 910×512 | 16 | 0.01 | **0.36 s** |
| openh | 7.5 | 406 | 682×512 | 14 | 0.01 | 0.38 s |
| grasp_noleak | **96.6** | 2730 | 1280×1024 | 10 | 0.16 | **0.56 s** |
| pe_video | 2.3 | 699 | 608×342 | 66 | 0.01 | 0.44 s |
| sitl | 4.9 | 1688 | 640×360 | 218 | 0.01 | 0.94 s |
| cholec80 | 22.5 | 1632 | 854×480 | 249 | 0.03 | 3.16 s |
| surgvu24_clean | 9.8 | 3514 | 1280×720 | 250 | 0.02 | 3.10 s |
| lemon | 21.1 | 1626 | 916×660 | 96 | 0.03 | 3.76 s |
| sitl_2026 | 75.4 | 1800 | 1920×1080 | 30 | 0.13 | **4.85 s** |
| lapgyn6_events | 23.7 | 908 | 1920×1080 | — | 0.05 | **5.37 s** |

**Payload size is refuted as the predictor.** `grasp_noleak` is the largest
source on disk (96.6 MB/clip) and decodes in 0.56 s; `surgvu24_clean` is 10×
smaller and takes 3.10 s. Pearson r vs decode time: MB/clip **+0.20**, frame
count +0.50, GOP **+0.62**, and `min(16·GOP, frames)` — the frames a 16-seek
scatter actually forces the decoder through — **+0.70**.

⚠️ **The arithmetic near-match that started this was a coincidence.** Under the
realized T=0.5 mixture, `p(clip from a >30 MB/clip source) = 0.104` against a
measured `p(dataload > 10 s) = 0.098`. Two numbers agreeing to 6% is not a
mechanism, and here the mechanism it suggested is wrong. It is recorded because
it was persuasive and false.

**The lead is keyframe spacing.** Seek dominates open by 10–100× everywhere, and
every fast source is a *dense-GOP* one. That is consistent with the already-
measured heichole re-encode (1080p high-bitrate sparse-keyframe → 512p/crf23/
**g16**: decode 1765 → 182 ms, `scripts/reencode_source_reshard.py`) — the same
intervention, on a different source, for the same stated reason.

The intervention arm confirms GOP, at unchanged pixels — re-encode the *same
clip*, decode it again:

| source | orig GOP | orig | **g16 @native** | g16 @512 |
|---|---|---|---|---|
| surgvu24_clean | 250 | 3.38 s | **0.41 s (8.2×)** | 0.26 s (13.1×) |
| cholec80 | 249 | 3.04 s | **0.35 s (8.8×)** | 0.40 s (7.6×) |
| lemon | 96 | 2.77 s | **0.63 s (4.4×)** | 0.41 s (6.8×) |
| sitl_2026 | 30 | 3.55 s | 2.35 s (1.5×) | **0.61 s (5.8×)** |

GOP alone buys 4.4–8.8× with resolution untouched. `sitl_2026` is the
informative exception — already GOP-30, so its cost is pixels and it needs the
512p arm. Both levers are real and separable.

### …but keyframe spacing does not own the TAIL

Take that per-source table as a mixture, weight it by the run's own realized
T=0.5 fractions, and draw bs=2 clips per batch. It reproduces the body of the
per-rank dataload distribution and misses the tail entirely:

| quantile | predicted from mixture | observed on-node |
|---|---|---|
| p50 | 1.38 s | 1.20 s |
| p90 | 5.75 s | **9.58 s** |
| p99 | 8.61 s | **22.99 s** |
| mean | 2.66 s | 3.10 s |
| p(>3 s) | 0.449 | 0.232 |
| p(>10 s) | **0.002** | **0.095** |

**A ceiling argument makes this decisive.** At bs=2 no pure-decode story can
exceed twice the slowest per-clip decode ever measured — `lapgyn6_events` at
5.37 s, so 10.74 s. **8.7% of 18,252 observed samples are above that ceiling**,
median 15.1 s and max 67.8 s = 6.3× it. No mixture of measured decode costs can
produce those draws.

What the excess is *not*: it is not fabric (it is present at **one node**), not
a bad node (the argmax rank rotates), and not payload. Offline benchmarks
structurally cannot see it, because they do not read through DAOS. It is
node-clustered — at 16n a stall hits a median of 11 ranks across 7 nodes where
iid over the same rank count predicts 8.1 — which is consistent with a shared
per-node DAOS client but **not traced, and therefore a hypothesis**.

So GOP re-encoding is worth doing on its own terms (it moves p50, mean, and the
whole body, permanently and offline) but **must not be sold as the fix for the
scaling asymptote**. The projected "1.18 s → 0.42 s" mixture-expected decode is
a body statistic; the asymptote is set by the tail above the ceiling, which the
re-encode does not address because the re-encode cannot make a clip decode
faster than a clip decodes.

**Instrument for the live path** — `VJEPA_DECODE_PROFILE=1` (default OFF) logs
per source the decode call's own time and the gap since the previous sample, so
a codec tail (lives in `decode`, sorts by source) is distinguishable from a
storage stall (lives in `gap`, does not). Gated to the first
`VJEPA_DECODE_PROFILE_RANKS` ranks (default 12 = one node) rather than rank 0:
a >ceiling event hits a median of 2 of 12 ranks, so rank-0-only gating would
sit out roughly five iterations in six. Run it as a ladder rung: `1:prof1`.

## Scaling ladder (`scripts/scaling_ladder.sh`)

Measures per-tile efficiency across node counts in **one allocation**, so every
rung shares a fabric hour, and separates straggler wait from compute.

- Rungs are `<nodes>[:nw<N>]`, run serially, each in its own sub-world: private
  nodefile + explicit `WORLD_SIZE` + offset `MASTER_PORT`. `WORLD_SIZE` takes
  precedence over PMI `SIZE` (`src/utils/distributed.py:146-158`) and
  `hsdp.py` derives `num_nodes = world_size // local_world_size`, so each rung
  builds a correctly sized mesh from its own world.
  ```
  L1: qsub -l select=16 -v VJEPA_LADDER_RUNGS="1 2 4 8 16"   scripts/scaling_ladder.sh
  L2: qsub -l select=32 -v VJEPA_LADDER_RUNGS="16 32"        scripts/scaling_ladder.sh
  L3: qsub -l select=64 -v VJEPA_LADDER_RUNGS="16 64 16:nw2" scripts/scaling_ladder.sh
  ```
  16n repeats in every job as a **cross-job anchor**. If it moves by more than
  its IQR between jobs, cross-job comparisons are void.
- `VJEPA_SCALE_PROBE=1` adds an explicit pre-step `torch.distributed.barrier()`
  (after a device sync) and logs it as **`barrier-ms`, CSV column 16**, appended
  so pre-existing readers that index 0-15 are unaffected. Per rank it is how
  much *earlier* that rank arrived than the last one: min-over-ranks ≈ 0,
  **max-over-ranks is the skew**. Without it, skew hides inside forward's first
  FSDP all-gather and manufactures an apparent forward blowup. Default OFF — it
  is a real collective and must not be left on for production throughput runs.
- Rungs skip the checkpoint load (`load_checkpoint: false`). Every rank reads
  the 22.8 GB `.pt` independently: 5m26s to iter 0 at 64n, 8m57s at 256n, which
  would eat the 1 h cap in startup. Shapes and FLOPs are identical from random
  init. **Loss from a ladder run is meaningless and must never be reported as a
  training signal.**
- The nw arm is a **hazard arm, run last**: `train.py` forces `num_workers=0`
  under HSDP because forking persistent workers after `init_device_mesh` can
  inherit broken xccl state and deadlock on the first batch. Its output dir is
  `n16_nw2/`, so a hang cannot be mistaken for a clean rung.
- Analyse with `scripts/scaling_efficiency.py --ladder <root> --clips-per-rank 2`
  (`lbA` is bs=2; the default 1 makes every throughput number 2x wrong). It
  prints per-tile efficiency vs the smallest measured rung, a phase breakdown
  taking max-over-ranks **per phase independently** (stragglers rotate), an
  `unacct` residual, and a first-vs-last-quartile **trend**. A rung still
  descending at the end of its window is a lower bound, not a median.
- **Discard iteration 0** — at 256n it logged `gpu: -137918.2 ms`.

`tests/test_phase_csv_contract.py` pins the 17-column order, that `barrier-ms`
stays last, that `scaling_efficiency.py`'s indices agree, that the forward marks
are not re-adjoined, and that the probe stays opt-in.

### Ladder result: the curve, and what `num_workers=2` does to it

L1a (job 8740093, rungs `1 2 4 8`) and L1b (job 8740311, rungs `8 16 8:nw2`),
full rank coverage at every rung, 39 common iters, iteration 0 dropped. The
**cross-job anchor passed** — 8n reads 19.01 s in L1a and 20.49 s in L1b, 7.8%
apart with each median inside the other's IQR — so the two jobs compose into one
curve.

| rung | ranks | med iter | max dl | floor | p(>10 s) | stall iters | clips/s/tile | eff |
|---|---|---|---|---|---|---|---|---|
| 1n | 12 | 10.84 s | 7.91 s | 2.93 s | 0.098 | 18/39 | 0.1846 | 100% |
| 2n | 24 | 14.01 s | 10.89 s | 3.05 s | 0.098 | 20/39 | 0.1427 | 77% |
| 4n | 48 | 16.04 s | 12.90 s | 3.14 s | 0.099 | 26/39 | 0.1247 | 68% |
| 8n | 96 | 19.01 s | 15.64 s | 3.22 s | 0.096 | 31/39 | 0.1052 | 57% |
| 8n\* | 96 | 20.49 s | 17.24 s | 3.23 s | 0.094 | 32/39 | 0.0976 | 53% |
| 16n | 192 | 24.79 s | 21.46 s | 3.26 s | 0.095 | 33/39 | 0.0807 | 44% |

(\* = L1b's anchor rung.) The per-rank *marginal* is unchanged out to 192 ranks
— p50 1.19 s, p99 24.0 s, p(>10 s) 0.095, statistically identical to 1n's. The
compute+comms **floor moves only 2.93 → 3.26 s across a 16× node jump**: 11%.
Everything else is the order statistic.

**`num_workers=2` is the single largest throughput lever measured on this
model.** Same 8 nodes, same allocation, same fabric hour:

| arm | med iter | IQR | total wall | max dl | floor | clips/s/tile |
|---|---|---|---|---|---|---|
| 8n nw0 | 20.49 s | [15.91, 24.24] | 785 s | 17.24 s | 3.23 s | 0.0976 |
| 8n **nw2** | **3.83 s** | [3.50, 8.83] | **401 s** | **0.00 s** | 3.40 s | **0.5222** |

**5.35× on the median, 1.96× on total wall**, and the 30 clean iterations run at
3.57 s against a 3.40 s floor — dataload fully hidden. The gap between the two
multipliers is the whole story: prefetch does not remove the tail, it makes it
**rarer**. nw2's per-rank max is 44.2 s, *higher* than nw0's 35.1 s; what falls
is the rate, p(>10 s) 0.094 → 0.014 (6.7×). Median-only reporting would claim
5.35× for a lever that delivers 1.96× — quote total wall.

Consequence for scale-out: nw2 does not escape the order statistic, it moves the
knee. p(≥1 stalling rank) reaches 1.00 by 32 nodes at nw2 versus by 4 nodes at
nw0. **Shrinking the tail remains the only thing that changes the asymptote**,
and 8.7% of draws are still above what any measured decode cost can explain.

⚠️ **nw2 is not yet production-safe.** The rung completed rc=0 with 96/96 rank
CSVs and a full 15 GB checkpoint written, but **85 of 96 ranks then aborted at
exit** with `terminate called after throwing an instance of 'std::system_error'
— No such file or directory`, after `avg. loss` and after the checkpoint. The 11
survivors are exactly ranks 1–11: node 0's non-zero ranks. Nothing is lost at a
rung boundary; under `VJEPA_SUSTAINED` self-resubmit, an aborting exit path is a
different matter and needs to be closed before nw>0 becomes a default. It also
has no 64n hazard arm yet.

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
