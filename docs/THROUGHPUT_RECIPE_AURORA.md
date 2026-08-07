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
export VJEPA_NUM_WORKERS=2        # 3.45x at 64n; was 0 here until 2026-08-07
export OMP_NUM_THREADS=${VJEPA_OMP_NUM_THREADS:-16}   # NOT ${OMP_NUM_THREADS:-16}
mpiexec ... --no-vni -o "$DIR/rank.%r.out" -e "$DIR/rank.%r.err"   # NO --pmi=pmix
```

### Two rules for editing the fragment

- **`${VAR:-default}` is the right guard for most things and WRONG for the FI
  variables.** Aurora's system profile already exports `FI_PROVIDER`,
  `FI_CXI_RX_MATCH_MODE` and `FI_CXI_OFLOW_BUF_SIZE`, so a `:-` guard on those
  inherits the system value and silently discards the recipe — the exact opposite
  of the guard's purpose. They are hard-set. Before adding a `:-` to anything
  new, check `env | grep <VAR>` in a clean login shell.
- **`OMP_NUM_THREADS` is the same trap, and it bit us.** PBS exports it into
  *every* Aurora job script, set to the node's **logical** CPU count = **208**
  (104 cores × 2 HT). So `export OMP_NUM_THREADS=${OMP_NUM_THREADS:-16}` — which
  is what `aurora_hsdp_env.sh:64` said until 2026-08-07 (`2a3820a`) — **never
  fired inside a job**, and every launcher sourcing the lib believing it set 16
  actually set 208. The fix is a distinct override name,
  `${VJEPA_OMP_NUM_THREADS:-16}`, so a deliberate choice cannot be confused with
  PBS's inherited value.

  *Scope, stated precisely.* mpiexec passed a literal `--depth 16`, so ranks were
  CPU-**bound** to 16-HT spans regardless; the 2496 threads were never
  simultaneously runnable node-wide. Each rank oversubscribed its own 16-HT span
  13×. Wrong, but **not** a node-wide 24×, and **no past throughput number in
  this doc is invalidated by this alone.** What IS invalidated: *cross-launcher*
  comparisons. `vitG384_capacity.sh:170` and `vitG384_256n_shakeout.sh` set 16
  unconditionally and were never affected; the six that source the lib without an
  explicit export (`accum_ab_64n`, `bs_vs_accum_ab_64n`, `ccl_knob_sweep`,
  `probe_tmpdir_pals`, `scaling_ladder`, `vitG384_256n_daos`) got 208. Ladder
  rungs and capacity runs have therefore been on **different thread counts all
  along** — any number carried between the two families is confounded.

  *Corollary for `--depth`:* at 12 ranks/node on 208 HT only depth ≤ 16 is
  launchable. `--depth 208` needs 2496 HT and gives `rc=139` in 0 s.

  *How it was found, and the general lesson.* Only because the ladder's new
  `:omp<N>` rung field echoed the resolved value for the first time and printed
  `omp=208`. It was invisible before precisely because **nothing printed it** —
  the same failure mode as the unlogged `num_workers` that once made a whole
  scaling comparison unrecoverable. **A knob that is never echoed is a knob
  nobody is actually setting.** Echo resolved values, not intended ones.
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

**Executed for cholec80** (job 8740763, 8 nodes × 32 workers, ~6 min wall).
`cholec80_g16` is the whole source re-encoded at `--short-side 0 --gop 16
--crf 23`, and it holds up on the corpus rather than on one hand-picked clip:

| check | original | `_g16` |
|---|---|---|
| shards | 256 | 256 |
| samples | 2916 | **2916** |
| on disk | 70 G | 43 G |
| clip `video02_clip_0027` frames | 1622 | **1622** |
| …resolution | 480×854 | **480×854** (untouched) |
| …decode, 16-frame scatter | 3.19 s | **0.45 s (7.1×)** |

7.1× measured against 8.8× predicted, on a different clip than the prediction
was drawn from. Sample count and frame count are preserved exactly — this
changes the *cost* of the corpus, not its content.

#### ⚠️ At the corpus level the win is 1.6–3×, not 7–8×

Every ratio above this line — the 8.2×/8.8×/4.4× intervention arm, the 3.16 s
per-source table, and the 7.1× directly above — is **one clip** decoded with a
**16-frame scatter across the whole video**. The trainer does not do that. It
seeks once into a *single 64-frame window* and `linspace`s 16 frames inside it
(`src/datasets/webdataset.py:375-383`). A scatter forces 16 independent seeks
and is the access pattern GOP helps most; the predictor that fit best,
`min(16·GOP, frames)` at r=0.70, only *means* anything for 16 independent seeks.

Measured over n=200 **paired** clips per source (job 8741066, harness
`scripts/decode_per_source_profile.py`, which mirrors the trainer's window;
`resampled=False` so each row is the same 200 clips before and after):

| source | p50 base | p50 g16 | **p50 ratio** | max base | max g16 | max ratio |
|---|---|---|---|---|---|---|
| cholec80 | 355 ms | 154 ms | **2.31×** | 2887 ms | 787 ms | 3.67× |
| surgvu24_clean | 266 ms | 168 ms | **1.58×** | 534 ms | 232 ms | 2.30× |
| lemon | 356 ms | 229 ms | **1.55×** | 910 ms | 399 ms | 2.28× |
| sitl_2026 → `_g16_512` | 612 ms | 156 ms | *3.92×* | 2129 ms | 249 ms | 8.55× |

**The sitl_2026 row is not a GOP measurement** and must not be averaged with the
other three. Its twin is `_g16_512`: `reencode_gop_pbs.sh:106` sets `SHORT=512`
for that source alone, so it is GOP **and** a downscale, and the resolution line
shows it — 1080×1920/720×1280 mixed → 512×910. Most of its 3.92× is pixels, not
keyframes. Report it as "GOP+downscale" or not at all.

The three pure-GOP sources land at **1.55–2.31× on p50**, dead centre of the
1.6–3× predicted here and nowhere near the 7–8× per-clip figure.

Pairing is verified in the output itself, not assumed: within every pair the
`frames/video` triple and the resolution histogram are **identical** (cholec80
1580/1703/1747 both sides; lemon 1823/1971/2039 both sides), and `frac<1.0(DROP)`
is 0.000 on all eight arms. Sample count, frame count and drop rate are
preserved exactly — the re-encode changed the *cost* of the corpus, not its
content, on all four sources.

A backfill model accounts for it: cost ≈ decode(64-frame window) +
decode(≈GOP/2 frames of backfill from the previous keyframe). Solving the two
pairs gives a 64-frame window at 145/161 ms (2.26 and 2.52 ms/frame, two
different resolutions landing in the same place) and backfill at 0.8–1.8
ms/frame. GOP-249 → GOP-16 removes ~116 backfill frames, worth ~100–200 ms —
which is the whole observed delta. **The re-encode did exactly what the
mechanism predicts; the 7–8× expectation was measuring a different workload.**

Take the corpus numbers, not the clip numbers, and note the tail improves more
than the median (3.0–4.6× vs 1.6–2.3×) — consistent with GOP mainly buying back
the unlucky-seek draws.

Two process notes worth more than the numbers:

- The first manifest recorded `short_side: 512` for an encode that ran at 0.
  `--finalize` is a separate invocation, callers do not repeat the encode flags,
  and it stamped argparse's default. The pixels were correct; only the
  provenance lied — the worse failure, because bad pixels get noticed and a bad
  manifest gets believed. Finalize now merges the settings the workers actually
  recorded in `_partial_*.json` and warns if partials disagree.
- Blanket downscaling would have been an *upscale* here: cholec80 is 854×480,
  below the 512 target. Check the source resolution before reaching for the
  pixel lever.
- **The first corpus benchmark was not a paired comparison**, and it announced
  itself as a content change rather than as a measurement bug.
  `WebDataset(resampled=True)` reaches `ResampledShardList(urls)` through
  `create_url_iterator()` **with no seed forwarded**, and `ResampledShards`
  then mixes `time_ns()`, `getpid()` and `os.urandom(4)` into its own — so
  neither `seed=` nor `np.random.seed()` pins it, and a source and its `_g16`
  twin streamed *different videos*. The visible symptom was `frac<1.0` going
  0.015 → 0.000 and clip-std min 0.00 → 28.72 across the surgvu24 pair, i.e.
  exactly the signature of "the re-encode altered corpus content" — the one
  thing this intervention promised not to do — and in fact 3 different clips
  out of 200. Now `resampled=False` plus a seeded frame window (`520e0ce`);
  the paired run proves itself by printing identical `frames/video` and
  identical resolution histograms within each pair.
  **The unpaired ratios turned out to be close to right anyway** — cholec80
  2.30× unpaired vs 2.31× paired, surgvu24 1.58× both — so the defect cost
  nothing in the headline number. It is recorded because of what it *looked*
  like: an unpaired draw manufactured a fake corpus-content change on a
  measurement whose entire promise was that content did not change. The error
  was silent in the quantity being reported and loud in an unrelated one.
  (An earlier note here claimed pairing moved cholec80 to ~3.0×; that came from
  a partial read of the pairing fix and the completed paired run does not
  support it.)

**Scope.** This moves the body of the dataload distribution — p50, mean, p90 —
permanently and offline. It does not touch the tail, for the reason the next
section gives: a re-encode cannot make a clip decode faster than a clip decodes.
Do not report it as the scaling fix.

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
sit out roughly five iterations in six.

Run it as a **pair** of one-node rungs, and read them in order:

```
qsub -q debug -l select=1 \
     -v VJEPA_LADDER_RUNGS="1:prof1 1:nw2:prof1" scripts/scaling_ladder.sh
python scripts/decode_prof_report.py <outroot>/n1_nw0_prof <outroot>/n1_nw2_prof
```

One node is enough — the >ceiling draws are present at 1n, so node count buys
nothing and the cheap queue is the right place for it.

**The two columns are not equally trustworthy at both worker settings, and the
verdicts are asymmetric because of it.** `decode` is one call in one process
either way, so it is clean at any `num_workers`. `gap` is only clean at nw>0,
where the profiled process does nothing but decode in a loop; at nw=0 decode runs
inline in the training process, so every batch-boundary gap also contains the
whole training step (~3.4 s here) and is bimodal by construction at bs=2. So:

| observation | rung needed | verdict |
|---|---|---|
| live `decode` max > 10.74 s ceiling | nw=0 alone suffices | tail is codec cost the offline bench under-measured; re-encode is the direct fix |
| tail not in `decode`, large `gap` | **nw=2 required** | tail is upstream of decode; storage remains a hypothesis until traced DAOS-side |
| tail not in `decode`, large `gap`, nw=0 | — | **inconclusive**, not a storage result: gap contains compute |

⚠️ **Set `VJEPA_DECODE_PROFILE_EVERY` against the rung's length.** A rung emits
`ipe × batch_size` samples per rank — 80 × 2 = 160 — and the module default
period is 200, so the obvious invocation produces an *empty log* that looks
exactly like "the tail did not happen". The ladder now passes 40, and the module
logs a one-time `[decode-prof] ENABLED` banner on the first sample so ON-and-
silent is distinguishable from OFF. `decode_prof_report.py` reports those two
cases differently and refuses to call either one a null result.

#### The pair ran (job 8741045): the tail is upstream of decode

Both rungs, 1 node, 80 iters, 12 profiled ranks, `EVERY=20`. Worst values across
all sources, in seconds, against the 10.74 s bs=2 pure-decode ceiling:

| rung | worst `decode` | worst `gap` | verdict |
|---|---|---|---|
| `n1_nw0_prof` | 3.17 | 50.85 | INCONCLUSIVE by construction — gap holds the step |
| `n1_nw2_prof` | **4.22** | **53.84** | **tail is upstream of decode** |

**Decode is exonerated.** The worst single decode observed live, over 16
sources, is 4.22 s — **39% of the ceiling**, and that on `cholec80` whose
offline max is 3.04 s. No source's live decode came within 2.5× of the ceiling.
Whatever produces the >10 s dataload draws, it is not the codec, and it is not
the offline benchmark having under-measured the codec.

**And at nw=2 the `gap` column is finally a clean storage measurement** — the
profiled process is a worker that does nothing but decode in a loop, so a gap is
time spent waiting on everything upstream: tar read, DAOS, shuffle-buffer
refill. It reads 53.84 s worst, 5× the ceiling, on `grasp_noleak` (offline max
0.56 s). Several sources show gap p50 in the 10–27 s range while their decode
p50 is under 1.2 s.

Two cautions on those numbers, neither of which changes the verdict:

- **`gap` at nw=2 is upstream wait, not necessarily *storage* wait.** A worker
  also blocks when the loader's output queue is full — i.e. when *training* is
  the bottleneck and prefetch is doing its job. Since nw=2 shows `dataload-time`
  = 0 ms at the median, that backpressure is present by design and inflates gap
  p50. It cannot explain the >ceiling draws (the step is ~3 s), but read gap p50
  as "worker was idle", not "DAOS was slow".
- The per-source `n` counts are small (1–26 per source over the window), so
  per-source ordering within the table is noise. The claim rests on the
  aggregate: max decode 4.22 s vs max gap 53.84 s, an order of magnitude apart.

This closes the "is it the codec?" branch that motivated the GOP re-encode as a
*scaling* fix. The re-encode remains worth its cost for the body of the
distribution (1.55–2.31× on p50, measured paired) — but the asymptote is set by
the tail, the tail is upstream of decode, and **a per-node DAOS client stall
remains the leading hypothesis and is still untraced on the DAOS side.**

#### The re-encode is shipped, and what it is still owed

All four twins are ingested into DAOS `AuroraGPT/vjepa_surg_wds` and verified
**byte-exact** against Lustre: job 8741105, 8 nodes, **7,484 tars / 1.744 TiB in
69 s at 27.8 GiB/s**, source and destination both 1,917,412,353,556 bytes, diff
**0**.

⚠️ The verifier prints **"99% bytes" and that is expected, not a shortfall.**
`du -sbL` counts directory entries, which Lustre and DAOS size differently. Only
a file-bytes-only comparison closes it exactly:

```
find -L <src> -type f -printf '%s\n' | awk '{t+=$1} END{print t}'
```

Keep the 99% gate — it catches real truncation, and it exists because a previous
ingest passed on *tar counts* with 413 bytes of dangling symlinks behind it — but
do not chase the last percent without running the line above first.

Batch-mode ingest also retires the "cost tracks file count, not bytes" rule from
the per-source-mpiexec era: one `dsync --dereference` over a symlink farm moved
1.74 TiB in 69 s, where the old per-source loop budgeted ~0.17 s/tar (≈21 min for
the same 7,484 tars). The old rule described the launch loop, not DAOS.

Point a run at the twins with
`configs/vitg16_surg_vid_webdataset_single4/vitG384_lbA_g16.yaml` — `lbA` with
exactly four dataset basenames changed, proven equal by parsing both configs,
substituting the datasets list, and comparing the remainder. `lbA` itself is
untouched: `scripts/vitG384_256n_daos.sh` defaults to it and the swap has no live
measurement yet.

**Two things it is still owed, and neither should be skipped:**

1. **A live s/iter measurement.** Everything above is offline per-clip decode. The
   tail is upstream of decode, so the *body* win is not entitled to become an
   iteration-time win — that is a prediction, not a result. A corpus A/B must run
   serially **inside one allocation**; across two jobs it measures the fabric
   hour, not the corpus. The ladder now takes a per-rung `:cfg<NAME>` field for
   exactly this.
2. **Reading sitl_2026's arm correctly.** Its twin is GOP *and* a short-side-512
   downscale, so `vitG384_lbA_g16` genuinely feeds lower-resolution sitl_2026
   frames than `lbA` does. `crop_size` is 384 so 512 still exceeds the crop, but
   it is a real input change and the only one in the file. Any quality delta
   attributed to "the re-encode" has to account for it.

#### The live A/B ran (job 8741594): NULL, as predicted

Debt #1 is paid. Three arms serial in one allocation, 16 nodes / 192 ranks,
nw=2, `ipe=60`, 53 usable iterations each, A/B/A so the baseline brackets the
treatment. Read with `scripts/tail_arm_compare.py`:

| arm | iter_mean | iter_med | gap | dl mean-of-max | gap/dl | excess | p99 |
|---|---|---|---|---|---|---|---|
| `n16_nw2` | 9.87 | 3.82 | 6.05 | 5.91 | 1.02 | 0.087 | 3.85 |
| `n16_nw2_rep2` | 9.02 | 4.13 | 4.90 | 5.11 | 0.96 | 0.084 | 3.72 |
| `..._vitG384_lbA_g16` | 8.98 | 3.68 | 5.30 | 5.44 | 0.97 | 0.105 | 5.02 |

**Verdict: the GOP re-encode does not move live s/iter at 16n.** Tail +23.8%
(MEASURED, wrong direction); wall −4.9% (**below floor**, no claim in either
direction). This is the outcome the section above predicted — g16 buys the
*body*, and the body is not what a synchronous step pays for.

**Read the floor before the delta.** The two anchor arms are identical
configuration and differ by 1.03× on `excess` and **1.09× on `iter_mean`** — so
this study's own noise floor is ±3.3% on the tail and **±8.9% on wall**. A
−4.9% wall delta is inside it. An anchor spread of 1.09× passes any reasonable
"is the baseline tight" gate and *still* swamps a 5% effect, which is why the
gate is not the test: the test is delta-vs-spread, and `tail_arm_compare.py`
now computes and prints it (`39a86b9`). A two-repeat range also *understates*
the true floor. A delta labelled "below floor" supports no claim in either
direction — report it as a null with the floor attached, never as a small win.

**Do not sum the `excess` column with an nw=0 run's.** Under nw=2 the per-rank
dataload distribution is zero-inflated (median 0.00), so `excess` degenerates
to the plain mean and the BODY column carries no information. Still absolute
and arm-comparable *within* this table; not the same statistic as on nw=0.

Debt #2 (sitl_2026's resolution confound) is unaffected and still open — it is
a quality question, and this was a throughput measurement.

## 64n efficiency is 63%, and the dataload tail is only HALF of the loss

Job 8741170, 2026-08-07. `vitG384_lbA`, DAOS, nw=2, `VJEPA_SCALE_PROBE=1`, both
rungs in ONE allocation. 30 iters each, **full rank coverage** (12/12 and
768/768), window itr>=7.

This is the first ladder with a **valid 1n reference** — the historical 4.19 s
"1n" was a 192-rank run with 12 CSVs written, and is void.

| | 1n (12 tiles) | 64n (768 tiles) |
|---|---|---|
| compute floor (`iter − max dataload`) | 2.99 s | 4.21 s |
| stall tax (mean − floor) | 1.49 s | 2.94 s |
| **mean s/iter** | **4.48 s** | **7.15 s** |
| median s/iter | 3.08 s | 4.56 s |

**Per-tile efficiency 64n vs 1n = 63%** (means). Against the >80% bar.

**Report the MEAN, not the median.** Dataload is bimodal — exactly 0.00 s on
most iterations and multi-second on the rest — so the median silently prices the
stall at zero. Median gives a flattering 68%. Wall-clock is the mean.

**The loss splits almost evenly, and neither half alone is enough:**

- **+1.22 s compute-floor growth, 94% of it backward.** Median-*over-ranks*
  backward goes 1.23 s → 2.38 s. Stripping the straggler does not touch it, so
  **every rank pays** — this is the HSDP replicate-dim all-reduce growing with
  node count. **H1 confirmed, for backward only.**

  > **QUALIFIED 2026-08-07 (job 8741386).** "Every rank pays" stands; "growing
  > with node count" does not follow from these two points. Backward is **not
  > stationary within a run**: at 16n it drifts 1.68 → 7.43 s across 50
  > iterations while 1n holds 1.24 s flat, so a window median measures partly
  > where the window was cut. The 2.38 s above came from 23 iterations and mixes
  > a clean-phase cost with an unknown amount of drift. Early 64n iterations in
  > 8741386 sit at ~1.9 s clean-phase, close to 16n's ~1.7 s — the *small*
  > increment a saturating bandwidth term predicts, with the rest of the gap
  > being drift rather than node count. Treat the 63/71% split as still directionally
  > right (both halves are real and both matter) but do not quote 2.38 s as
  > "the 64n all-reduce cost". See `[[backward-growth-is-drift-not-per-n-cost]]`
  > and use `scripts/backward_vs_nodes.py`, which withholds a verdict when the
  > first-vs-last-quarter drift exceeds 1.25x.

- **+1.45 s stall tax**, from the dataload tail.

Killing the tail outright would leave 64n at its 4.21 s floor = **71%**. The
all-reduce growth alone holds it under the bar. Both must be fixed.

**The forward "blowup" is a straggler artifact — H1 refuted there.** Max-over-
ranks `fwd-context` rises 1.07 → 1.83 s, but median-over-ranks is FLAT
(1.03 → 1.01 s). No rank's forward got slower; the max is sampling skew that
arrives *inside* forward's first FSDP all-gather. Any future claim of a "forward
blowup" must be checked against the median-over-ranks column before it is
believed. `barrier-ms` is only 0.14 s at the median, so ranks are well
synchronized on ordinary iterations — **H2 is not the median story**; skew
appears only on the 8/23 iterations where someone stalls.

**Caveat:** both rungs ran ipe=**30**, not the budgeted 40, from the `local IFS`
leak (fixed in `7d3666f`, after this job was submitted). 23 post-warmup
iterations is thin for a mean that the spikes dominate. The split is large
enough to survive that; the exact percentages are not precise to a point.

### The tail is fully present at ONE node

`n1_nw2` is the decisive rung, and it kills every node-count explanation: with
prefetch on, one node, no inter-node fabric at all, dataload still shows p90
6.32 s / max 15.16 s. Structure:

- **itr 0-6: all 12 ranks spike together** (itr 0 = 12/12 at ~34 s). Prefetch
  warmup. A window starting at itr>=5 still eats two of these.
- **itr >=7: isolated single-rank events** — 5 of 23 iterations, 1-2 ranks each.
  9 of 12 ranks show nothing at all past itr 6.

At 64n the same shape scales as an order statistic: 8/23 iterations affected,
max 12 of 768 ranks. Per-rank incidence is roughly constant; what grows is the
chance that *somebody* is stalled, and the barrier makes all 768 pay it.

### GOPEN_BUFFER / DAOS read granularity: REFUTED as a lever

Job 8741245 measured **36-44x** on DAOS between a 4 KB and a 1 MB tar read
buffer, with a FLAT curve on Lustre. Read granularity is genuinely a first-order
cost on the DAOS client. It is **not** a win available to us:

`buffering=-1` resolves to `st_blksize` **exactly** — verified by reading one
byte through webdataset's own `gopen` and checking `f.raw.tell()`: unset gives
4194304 against an st_blksize of 4194304, and `GOPEN_BUFFER=1048576` gives
1048576. DAOS reports st_blksize = 2 MB, and 1 MB/4 MB are within 5% of each
other. **Training already sits on the flat fast end.** `GOPEN_BUFFER` works, but
on DAOS it can only make the buffer smaller.

The 42x was the gap between a setting nothing uses and the one already in force.
The first run of the probe reported it as a lever because the sweep omitted -1,
so it could not locate the default on its own curve.

What this *does* establish is a fragility: anything that shrinks the effective
buffer — a mount reporting a small `st_blksize`, an explicit `GOPEN_BUFFER`, a
library re-opening the shard — costs ~40x on DAOS, silently. That is why
`train.py` now logs the resolved value in `THROUGHPUT KNOBS`.

**The tail's owner remains unidentified.** Decode is exonerated (worst decode
4.22 s vs worst gap 53.84 s), read granularity is exonerated, and it is not
node-count. Do not write a cause into this doc without direct evidence.

### The 68× is an order statistic, and coupling makes it BETTER not worse

`scripts/order_statistic_curve.py` on job 8741594 `n16_nw2` (192 ranks, 53
iterations, nw=2, DAOS):

    per-rank mean dataload   0.087 s/iter
    mean of max-over-ranks   5.91  s/iter    <- what the synchronous step pays

Both are correct. The factor of **68** is what taking a maximum over 192
heavy-tailed draws does, and it is the single largest source of confusion in
this document's history: a per-rank mean of 87 ms and a "1 second dataload at
scale" are the same measurement read two ways.

**Coupling suppresses the max by 47%.** Independently permuting each rank's
series across iterations preserves every marginal *exactly* and destroys only
the cross-rank alignment. observed/shuffled falls monotonically **1.00 → 0.526**
from k=1 to k=192. Ranks stall *together*, so extra ranks mostly join an
iteration already paying; under independence nearly every iteration would catch
somebody's spike. **So the node-clustering measured above is not the thing to
remove** — de-clustering the stalls, if it were free, would make this worse.
The levers that remain are per-rank stall *probability* and *magnitude*.

The curve is still climbing at the largest k measured — β = 0.684 on random
rank subsets, **0.776 node-contiguous** (the real deployment shape) — so no
ceiling is in sight at 192 ranks and neither lever is spent.

⚠️ **Do not extrapolate β.** It is fitted within one run's rank range. Production
64n and 256n came in at 23.19 and 22.92 s/iter — a 4× rank jump at ~zero
marginal cost, which β > 0 does not predict. The within-run subset curve and the
across-run node curve **disagree, and that disagreement is unexplained.**

Two further facts recorded as open, not explained:

1. Per-rank mean dataload *falls* with node count: 0.247 s at 1n → 0.056 at 16n
   → 0.009 at 64n, with both the stall rate (0.0291 → 0.0178 → 0.0066) and the
   magnitude (8.51 → 3.17 → 1.34 s) falling. No order-statistic account predicts
   this.
2. **Prefetch masking does not explain (1), and is refuted.** Marginal
   r(compute, stall rate) = −0.683 across 15 nw2 runs reads like "a slower step
   hides more prefetch" — but r(ranks, compute) = +0.545, so it is confounded by
   rank count. Hold rank count fixed and the sign **flips positive**: 12r
   −0.030, 24r **+0.979**, 192r **+0.707**. The rank effect is real and
   independent of step time.

**The prize is bounded, and now measured.** `gap = iter_mean − iter_med` divided
by the dataload mean-of-max is **0.97–1.02** across all three arms of 8741594:
the dataload order statistic accounts for the entire hidden part of wall clock,
with no other phase contributing a tail. Removing the tail moves `iter_mean`
down to `iter_med` and no further. (This is near-tautological under
zero-inflation — treat it as a bound on the prize, not as evidence for a cause.)

### `num_workers=2` is worth 3.45x at 64n — and it survives 768 ranks

The paired arm, same allocation, same 30 iterations, only `num_workers` differs:

| 64n rung | mean s/iter | median | floor | dataload (mean / med) | iters stalled |
|---|---|---|---|---|---|
| `nw=0` | **24.65** | 23.59 | 3.99 | 20.84 / 20.18 | **23/23** |
| `nw=2` | **7.15** | 4.56 | 4.21 | 2.77 / 0.00 | 8/23 |

**3.45x per-iteration.** The compute floor moves only +0.22 s (3.99 → 4.21 s),
confirming `num_workers` touches loading and nothing else. Per-tile efficiency
vs the 1n anchor: **nw=0 gives 18%, nw=2 gives 63%.**

**H3 answered: with nw=0 the cost is real, not merely relocated.** Median
dataload is 20.18 s and *all 23 of 23* iterations pay it — the bimodal
0.00-or-spike structure that nw=2 shows is gone, because with inline decode
every rank decodes every batch on the critical path. `barrier-ms` at 19.93 s
says the ranks then wait on each other for essentially that whole time.

**Do not quote the rung wall-clock ratio (933 s / 524 s = 1.78x) as the win.**
30 iterations do not amortize startup, so wall understates it by half. The
per-iteration figure is what scales to a real run.

**The nw>0 hazard is closed at the node count that matters.** The documented
failure mode — forking persistent DataLoader workers after `init_device_mesh`
builds the inter-node xccl subgroups — is O(ranks), so 1n and 8n passes never
transferred. `n64_nw2` completed rc=0, 30 rows, **768/768 rank CSVs, no
`terminate called`**. Combined with the `TMPDIR=/tmp` fix for the AF_UNIX
`sun_path` limit (`a31e1f1`), nw=2 is safe to default at 64n.

## Scaling ladder (`scripts/scaling_ladder.sh`)

Measures per-tile efficiency across node counts in **one allocation**, so every
rung shares a fabric hour, and separates straggler wait from compute.

- Rungs are `<nodes>[:nw<N>][:probe<0|1>][:prof<0|1>][:cfg<NAME>]`, fields in any
  order, run serially, each in its own sub-world: private nodefile + explicit
  `WORLD_SIZE` + offset `MASTER_PORT`. `WORLD_SIZE` takes precedence over PMI
  `SIZE` (`src/utils/distributed.py:146-158`) and `hsdp.py` derives
  `num_nodes = world_size // local_world_size`, so each rung builds a correctly
  sized mesh from its own world.
  ```
  L1: qsub -l select=16 -v VJEPA_LADDER_RUNGS="1 2 4 8 16"   scripts/scaling_ladder.sh
  L2: qsub -l select=32 -v VJEPA_LADDER_RUNGS="16 32"        scripts/scaling_ladder.sh
  L3: qsub -l select=64 -v VJEPA_LADDER_RUNGS="16 64 16:nw2" scripts/scaling_ladder.sh
  L5: qsub -l select=64 -v VJEPA_LADDER_RUNGS="1:nw2 64:nw2 64" scripts/scaling_ladder.sh
  ```
  16n repeats in every job as a **cross-job anchor**. If it moves by more than
  its IQR between jobs, cross-job comparisons are void.
- **`:cfg<NAME>` exists for CORPUS arms** (the g16 re-encode), which cannot be
  expressed as an env knob the way `nw`/`probe`/`prof` can. A corpus A/B across
  two jobs measures the fabric hour, not the corpus, so it has to sit serially in
  one allocation like every other arm. The rung dir is tagged only when the
  config differs from the job's, so existing rung names and
  `scaling_efficiency.py --ladder` discovery are unchanged. Unknown fields
  hard-reject: a typo'd `:cfg` that silently ran the default config would be
  indistinguishable from a real null result.
- **Rung ordering is a budget decision, not cosmetic.** Put the cheap
  irreplaceable rung first, the hazard arm next, its control after, and anything
  optional last. A hang burns `FIRST_ITER_DEADLINE` (900 s) of a 60 min cap, so
  whatever follows a hazard arm is what you are willing to lose. Budget from
  *measured* rung walls, which include 350–500 s of per-rung startup — not from
  `ipe × s/iter`.
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

⚠️ **The `eff` column is a ratio to a 1n rung that does not reproduce.** A
second 1n nw=0 rung (job 8741045, same config, same recipe, a different node six
hours later) came back **~2× faster than the L1a anchor**, and the gap survives
window-matching:

| | iter p50 | max-dl p50 | floor | per-rank dl p50 | p(>10 s) |
|---|---|---|---|---|---|
| L1a 8740093 `n1_nw0` (the 100% anchor) | 10.19 s | 7.26 s | 2.94 s | 1.03 s | 0.050 |
| L4 8741045 `n1_nw0_prof` | 5.40 s | 2.43 s | 2.92 s | 0.92 s | 0.015 |
| same window, iters 20–39 | 10.19 → **6.35 s** | 7.26 → **3.42 s** | | | |

What it is **not**: not compute (the floor is identical, 2.94 vs 2.92 s), not
`num_workers` (both nw=0), and not the body of the dataload distribution (per-
rank p25/p50/p75 agree within 11%: 572/1028/1901 vs 635/1144/1976 ms). The
difference is entirely in **how the per-rank tails line up across ranks**. Both
rungs have slow ranks; L1a's co-occur less. Shuffling each rank's samples
independently — which destroys within-iteration co-occurrence while preserving
every rank's marginal exactly — predicts a median max-over-ranks of 9.9 s for
L1a and 8.2 s for L4, against observed 7.26 and 3.42. **Both are clustered
relative to iid; L4 is far more so** (0.73× and 0.42× of the shuffled
prediction). So the two rungs differ in the *correlation structure* of their
stalls, not in their per-rank cost.

No cause is assigned. The two rungs ran on different nodes (`x4115c6s0b0n0` vs
`x4310c3s6b0n0`) at different times (17:54 vs 23:41) with different DAOS
neighbours, and either is a candidate; neither is evidence. **Until a 1n rung
reproduces, treat the `eff` column as showing the *shape* of the curve and not
its level** — the shape (per-rank marginal flat, floor flat, loss all in the
order statistic) is independently supported and does not depend on the anchor's
absolute value. The multi-node rungs are unaffected relative to each other:
they share an allocation, and the L1a↔L1b 8n anchor agreed to 7.8%.

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

⚠️ **nw2 was not production-safe when this rung ran — the exit abort is now
fixed; the 64n hazard arm is still owed.** The rung completed rc=0 with 96/96
rank CSVs and a full 15 GB checkpoint written, but **85 of 96 ranks then aborted
at exit** with `terminate called after throwing an instance of
'std::system_error' — No such file or directory`, after `avg. loss` and after
the checkpoint. The 11 survivors are exactly ranks 1–11: node 0's non-zero
ranks. Nothing is lost at a rung boundary; under `VJEPA_SUSTAINED`
self-resubmit, an aborting exit path is a different matter.

**This job predates the `TMPDIR=/tmp` fix** (`a31e1f1`, 22:16; this rung
submitted 19:48). With the fix live, job 8741045's nw=2 rung threw on **0 of 12
ranks** — see "The decisive test ran" below for the before/after table. The
remaining gate on nw>0 as a default is the 64n hazard arm, for the separate
xccl-fork deadlock, which is `O(ranks)` and does not transfer from 8n.

### Two nw>0 failures, not one — and only the second is fixed

Tracing the above turned up a **second, unrelated** failure that had been
folded in with it. They are separable by a single observable: whether any
batches were delivered.

| | (A) exit throw | (B) startup hang |
|---|---|---|
| when | after `avg. loss`, after the checkpoint | before iteration 0 |
| iterations | all of them | **zero** |
| symptom | `std::system_error` abort at exit | silence |
| status | **open** | **fixed** |

**(B) is an AF_UNIX path-length overflow, and it is not our port.** PBS sets
`TMPDIR=/var/tmp/pbs.<jobid>.<server>` (68 chars); PALS then splices a
per-mpiexec-launch UUID into it, `.../<uuid>/tmp`, reaching **109**. Python
multiprocessing builds `pymp-XXXXXXXX/listener-XXXXXXXX` on top — **141**
against an `AF_UNIX` `sun_path` cap of 107 usable bytes. The bind fails, and it
fails **on the queue feeder thread**, where the exception is non-fatal. So the
loader never delivers a batch and the run **hangs instead of erroring** — which
is why this read as a mysterious stall rather than as the plain `OSError` it is.

Bisected at 12 ranks (job 8740830), six arms, per-arm timeout:

| arm | rc | af_unix | batches |
|---|---|---|---|
| bare (no XPU, no dist, 512-elem tensors) | 1 | 12/12 | 0/12 |
| + XPU | 143 | 12/12 | 0/12 |
| + xccl init | 143 | 12/12 | 0/12 |
| + `worker_init_fn` → `file_system` | 143 | 0/12 | **0/12** |
| + `TMPDIR=/tmp` | **0** | 0/12 | **12/12** |

Identical failure at bare/xpu/dist puts the defect **below XPU and below
xccl** — it reproduces with no GPU, no collective, and no video decode.

The `file_system` arm is the informative one: it *clears the AF_UNIX error and
still delivers nothing*, failing at `libshm/core.cpp:62` instead. It trades the
`resource_sharer` socket for `torch_shm_manager`'s, and both are built from the
same over-long `TMPDIR`. **The cause is path length, not sharing strategy** —
which retires two mitigations this file used to carry. `MP_SOCKET_DIR=/tmp`
has zero occurrences anywhere in the torch install, and
`set_sharing_strategy("file_system")` is process-local and does not survive the
`spawn` into a worker (the workers print `strategy=file_descriptor` regardless).
Both were set for years and neither ever reached a worker.

Fix: `scripts/lib/aurora_hsdp_env.sh` now **hard-sets** `TMPDIR=/tmp`. The line
previously read `${TMPDIR:-/tmp}` — and PBS *always* sets `TMPDIR`, so the guard
never fired and the intended `/tmp` was never applied. That is the identical
`:-` trap this file already documents for the `FI_*` block, in a file that
documents it. Headroom is `107 − 32 = 75` chars; `/tmp` leaves 71 spare even
after PALS.

Two process notes:

- **This could not be settled retrospectively.** No historical rung ever logged
  its resolved `TMPDIR` — the same "resolved value never logged" gap that made
  `num_workers` unrecoverable from a finished run's own artifacts. `train.py`'s
  `THROUGHPUT KNOBS` line now prints it and flags `len > 75`.
- **Do not re-derive these paths off-node.** A login-shell calculation gives a
  comfortable 36 and a job-shell one gives ~68, because neither has the UUID
  yet, and `tempfile.gettempdir()` silently falls back to `/tmp` when `TMPDIR`
  does not exist. An off-node probe reported PASS while all 12 ranks were
  failing. Only the value printed from inside a rank is evidence.

Corroboration: ALCF documents this (`user-guides/aurora/known-issues.md` #7),
and all four BaseMM_PRISM Aurora launchers already export `TMPDIR=/tmp`.

**Scope — what is still owed.** (B) is validated in a 12-rank toy, not in the
trainer, and the fix was landed one layer *above* where it was validated: the
bisect set `TMPDIR` from inside python, while `aurora_hsdp_env.sh` sets it in
the job shell above `mpiexec` — and PALS *rewrites* `TMPDIR` rather than
inheriting it. If PALS composes from `PBS_TMPDIR` instead, the one-liner is
inert and looks exactly like a fix. `scripts/probe_tmpdir_pals.sh` settles that
in under a minute on one node. (A) remains untraced; a `sigterm` arm returned
`exit-throw=0/12`, so it is **not** generic signal-kill. And the 64n hazard arm
is still owed regardless — the xccl-fork deadlock is `O(ranks)` and 8n does not
transfer.

**Update — the probe passed, and (A) now has a candidate mechanism.** Job
8740990 ran all three arms on one node with an untouched control:

| arm | rank `TMPDIR` | listener path | verdict |
|---|---|---|---|
| A — job-shell `export TMPDIR=/tmp` (the landed fix) | `/tmp/<uuid>/tmp` (45) | 77/107 | **ok**, 30 spare |
| B — `mpiexec --env TMPDIR=/tmp` | `/tmp` (4) | 36/107 | ok |
| C — untouched control | `/var/tmp/pbs.…/<uuid>/tmp` (109) | **141/107** | OVER |

A passes and C reproduces, so the landed one-liner **is** effective and no
per-`mpiexec` `--env` is needed. `PBS_TMPDIR` is empty on Aurora, so PALS
composes from `TMPDIR` — the "if PALS composes from `PBS_TMPDIR`" branch above
is closed. **(B) is fully fixed, in the layer where it ships.**

#### A candidate mechanism for (A) — same cap, different socket

This is a **hypothesis with arithmetic behind it, not a diagnosis.** The throw
in job 8739712's `n2_nw2` rung lands with no Python frame on the stack:

```
terminate called after throwing an instance of 'std::system_error'
  what():  No such file or directory
```

`std::system_error` from ENOENT with no Python frame is the signature of
`SYSCHECK_ERR_RETURN_NEG1` in `torch/lib/libshm/err.h`, which wraps `errno` in
exactly that type. Under `set_sharing_strategy("file_system")` — active at
`app/main_dist_aurora.py:385` — every shared storage goes through
`torch_shm_manager`, whose socket is `<c10::TempDir>/manager.sock`. `c10`'s
`try_make_tempdir` reads `TMPDIR`/`TMP`/`TEMP`/`TEMPDIR` and falls back to
`/tmp` (`c10/util/tempfile.h`), and the manager's own prefix is
`torch-shm-dir-` (in the `torch_shm_manager` string table). So the path is
`$TMPDIR/torch-shm-dir-XXXXXX/manager.sock` — a **34**-byte suffix against
python multiprocessing's 32:

| `TMPDIR` at the rank | `manager.sock` | |
|---|---|---|
| PBS default, 109 | **143**/107 | OVER |
| with the fix, 45 | 79/107 | ok, 28 spare |

Job 8739712 **predates the `TMPDIR` fix** and *did* deliver batches, so it was
not the bind that hangs (B) — but its shm segments were created under the
143-byte path, which makes an ENOENT at teardown consistent with the same cap by
a second route. What this does **not** establish: that the throw is in fact in
libshm rather than another ENOENT-raising `SYSCHECK` in the same family, and
that the 3-of-24 rank distribution follows from it. Both need the decisive test:
**nw=2 in the real trainer with the fix live**. If the throw is gone, (A) was
downstream of (B); if it survives, the libshm story is wrong and the next step
is `catchsegv`/`gdb` on the aborting rank, not more arithmetic.

#### The decisive test ran: (A) is gone, and it was downstream of (B)

Job **8741045**, 1 node, rungs `n1_nw0_prof` then `n1_nw2_prof`, 80 iters each,
the `TMPDIR` fix live (`THROUGHPUT KNOBS … num_workers=2 pin_mem=True
persistent_workers=True tmpdir='/tmp/af511f5e-…/tmp'(45)`). The nw=2 rung exited
`rc=0` with 80 rows and 12/12 rank CSVs, and **zero of 12 ranks** threw.

The throw rate tracks the fix and nothing else:

| job | submitted | vs. `TMPDIR` fix (`a31e1f1`, 22:16) | ranks throwing `what(): No such file or directory` |
|---|---|---|---|
| 8739914 `n2_nw2` | 17:04 | before | 2 / 24 |
| 8740311 `n8_nw2` | 19:48 | before | **85 / 96** |
| 8741045 `n1_nw2_prof` | 23:54 | **after** | **0 / 12** |

Note 8740311: **85 of 96 ranks**, which retires a side-worry the 2-of-24 rate
invited — that this was a rare race on a few unlucky ranks. It is the common
case, and the earlier 2/24 was the low draw. Under the fix, zero.

So both symptoms of nw>0 have one cause, the `sun_path` cap, reached by two
routes: multiprocessing's `listener-` socket (141/107 → hang) and libshm's
`manager.sock` (143/107 → exit throw). One `export TMPDIR=/tmp` closes both.
This is consistent with the libshm mechanism above but does not prove it — the
fix removes every over-length path at once, so it cannot distinguish libshm from
another ENOENT-raising `SYSCHECK` on the same cap. The distinction no longer
blocks anything.

**What this still does not license.** Node count is not controlled: the clean
rung is 1n and the two throwing rungs were 2n and 8n. The fix is
node-count-independent by construction — path length does not depend on world
size — but per `[[scale-dependent-results-dont-transfer]]` that is an argument,
not a measurement. The 64n hazard arm is still owed for the *separate*
xccl-fork-deadlock risk, which genuinely is `O(ranks)`. Read this as "the throw
is closed at 1n and its mechanism is node-independent", not "nw=2 is cleared at
scale".

#### What nw=2 buys, measured (1 node, same job, back-to-back)

Same allocation, same node, same 80 iters. Window iters ≥ 20, max-over-12-ranks:

| | iter p50 | iter **mean** | gpu p50 | max-dl p50 | per-rank dl p50 | dl > 1 s | dl > 10 s |
|---|---|---|---|---|---|---|---|
| `nw=0` | 5403 ms | 6717 ms | 2921 ms | 2433 ms | 919 ms | 46.5% | 1.5% |
| `nw=2` | **3040 ms** | **5033 ms** | 2994 ms | **0 ms** | **0 ms** | 3.2% | 0.8% |
| ratio | **1.78×** | **1.33×** | 0.98× (control) | | | | |

The `gpu` column is the control and it is flat (2921 vs 2994 ms): same compute,
same work, and the entire difference is dataload being *overlapped* instead of
*serialized*. At the median prefetch hides it completely — `dataload-time` is
**0 ms** on every rank.

**Judge this on the mean, not the p50.** 1.78× at p50 is the best case; summed
wall over the window improves **1.33×** (403 s → 302 s), because the tail is not
prefetchable — 0.8% of samples still exceed 10 s at nw=2, and a stall deeper
than the prefetch queue stalls the step no matter who is reading. Overhead over
pure compute falls 130% → 66%: nw=2 removes about half the non-compute time and
leaves the other half for the tail work to take.

Same shape as the 8n result recorded above (5.35× median, 1.96× total wall),
reproduced at a node count where fabric cannot be the explanation.

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

## Backward degrades WITHIN a run — job 8741386, 2026-08-07

The "backward all-reduce grows with node count" reading is **not what the data
shows**, and the bandwidth-vs-latency question this ladder was built to answer is
**mis-framed rather than answered**. Rungs `1:nw2 16:nw2 64:nw2`, ipe=50, one
allocation, identical config, all rc=0 with full rank coverage (12 / 192 / 768).

Backward **median-over-ranks**, first vs last quarter of the itr 7-49 window:

| rung | Q1 | Q4 | drift | min-over-ranks @ Q4 | window median |
|---|---|---|---|---|---|
| 1n | 1.24 | 1.38 | 1.11x | 1.36 | 1.24 |
| 16n | 1.68 | 7.43 | **4.43x** | 2.23 | 2.46 |
| 64n | 1.93 | 8.83 | **4.59x** | 4.72 | 8.06 |

**A window median is not a per-step cost here.** 16n reads 2.46 s over the window
while actually travelling 1.68 → 7.43 s across it; the median describes where the
run was cut off. Both the ring model and every efficiency number assume a
*stationary* per-step cost, so fitting either to these medians fits the drift.
`scripts/backward_vs_nodes.py` now refuses a verdict above 1.25x Q1→Q4 drift.

**What is ruled out, by columns in the same rows:**

- **Not dataload skew.** `dataload` is 0.00 s on every rank from itr 1 (nw=2
  prefetch working) and `barrier-ms` — the scale probe, which measures skew
  arriving at the top of the step — is ~0.1 s on the degraded iterations. Ranks
  enter the step synchronized with data in hand. This is the *opposite* regime
  from the nw=0 tail.
- **Not forward absorbing skew.** `fwd-ctx` stays flat at ~1.0 s while backward
  goes to 12 s.
- **Not memory.** `l0-free-mib` flat at 12147, `l0-ext-mib` flat at 46788.
- **Not the model or schedule.** 1n runs the identical config for the identical
  50 iterations and holds 1.24 s. The only thing 16n/64n add is an inter-node
  all-reduce (at 1n the HSDP replicate group is size 1, so only the intra-node
  ReduceScatter runs).

**Mostly wait, partly work.** Per-iteration, min-over-ranks often stays at ~1.5 s
while p50 ≈ max hits 12.8 — ranks blocked on peers. But reduced over the last
quarter, min does rise (1.68 → 2.23 at 16n), so a smaller real per-rank component
sits underneath. Quote both; "min stays at the floor" is true per-iteration and
false per-quarter.

**Onset is not a fixed iteration:** itr 26 at 16n (357 s in-loop), itr 17 at 64n
(271 s).

### It does not reproduce — the drift is allocation-specific (job 8741490)

The follow-up ran `16:nw2` twice at ipe=60, same config, same node count, *longer*
than the rung that degraded. Backward held **1.65–1.95 s across all 60
iterations** — no degradation whatever, where 8741386's 16n rung was at 7.43 s by
itr 40-49. The two jobs drew **disjoint node sets** (8741386 on x4201/x4202,
8741490 on x4116).

So the numbers above are **one allocation's behaviour, not a law of the run.**
Consequences for how to read all of this:

- **The "after a few minutes" onset timing does not generalize.** It described
  two rungs in one job.
- **"Restart clears it" is withdrawn.** The 64n rung starting clean after the 16n
  rung ended degraded is equally consistent with the degradation having simply
  stopped on its own. Both readings survive; neither is established.
- **Not one sick node.** Inside the degraded rung, all 16 nodes slowed together —
  per-node median backward at itr>=40 spans 4.61–7.96 s with no outlier. A single
  straggler dragging the collective would show one node far out.
- **What this means practically:** a run can hit a regime where the inter-node
  all-reduce costs 4-6x its clean-phase value, lasting at least tens of
  iterations, and another allocation running the identical job never sees it. Any
  A/B that puts its two arms in different jobs can be swamped by this
  (`ccl_knob_sweep.sh` already runs arms in one allocation for exactly this class
  of reason). Treat a single-allocation throughput number as a draw from a
  distribution, not a measurement of the config.

The restart test itself came back **inconclusive by construction**: both `16:nw2`
rungs of 8741490 were clean (backward 1.76 / 1.68 s, drift 1.08 / 1.05 over 53
fully-covered iterations each). You cannot test whether a restart clears a state
that never accumulated.

### How much does this cost production? Mostly it isn't this at all

Rather than buy more slots hoping to catch an episode, scan the `backward-ms`
every production run has already written — `scripts/backward_drift_scan.py`,
46 runs, 195k rank-0 iterations, zero node-hours. Degraded blocks come to ~7% of
wall clock, but split by loader era:

| loader | runs | rank-0 iters | weighted wall% | runs with episodes |
|---|---|---|---|---|
| LIVE (DAOS) | 26 | 77,348 | **17.2** | 22/26 |
| staged (/tmp) | 20 | 117,349 | **0.3** | 1/20 |

Every clean-era run reports `dataload` **identically 0.00 at max-over-ranks
across 11-14k iterations**; every degraded-era run has a live tail. Config is not
the split — `abl_laponly` (0.0%) and `abl_full` (27.6%) match on model, bs,
activation checkpointing and workers, three days apart.

**So most "backward degradation" in production is the dataload tail wearing the
comms column's clothes.** One late rank blocks every other rank inside the
gradient all-reduce, and that wait is charged to `backward`. Practical rule:
never read a `backward` blowup as a comms result without checking `dataload`
**max**-over-ranks in the same block — the median stays ~1 s while the max goes
to 12 s, so the median will not warn you.

It also revises the "two halves" split: the halves are not independent, because
part of the backward half *is* the tail half. At 17.2% of wall clock on the
current DAOS path the tail outranks any remaining CCL knob.

Conditioning `surg_2_1_vitG384_fixedshape` on a quiet loader still leaves 11
degraded blocks (vs 4 with a tail) at 10.5 s against a 4.31 s floor, min-over-ranks
up 2.00 → 3.87 — a small residual, and the only production evidence bearing on
the 8741386 drift. Segmenting that run by allocation (on the repeated CSV
**header**, not on `itr`, which cycles every `ipe`=30) gives 14 segments spanning
**4.78-14.34 s** in backward median: the allocation spread, visible in production.

**What survives of the original framing:** clean-phase cost is 1.24 / ~1.7 /
~1.9 s at 1 / 16 / 64 nodes. The 16→64 increment is *small*, which is what a
saturating bandwidth term predicts. The large numbers are drift, not node count.

**Consequences.** Short shakeout rungs spend most of their iterations in the clean
phase, so they *understate* what a long production run pays — the opposite of the
usual warmup bias, and a reason not to extrapolate a 30-50 iteration rung to a
multi-hour epoch. Cause is UNKNOWN and nothing here narrows it: process/CCL-context
lifetime (oneCCL or Level-Zero resource accumulation across successive collectives)
was the natural family *while* reset-on-restart looked real, but with that
withdrawn, an allocation-level or fabric-neighbour explanation fits the evidence
equally well. No column identifies which, so do not label it
([[no-lazy-cause-labels]]). `[[vitG-2b-allreduce-spikes]]`'s 9 s → 150 s 16n spike may be the same
phenomenon at larger amplitude and makes `CCL_ZE_CACHE_OPEN_IPC_*` worth a paired
arm — but that was a hypothesis, and the two have not been shown to be the same.
