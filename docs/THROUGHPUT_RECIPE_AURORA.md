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

# comms — the measured throughput lever
export VJEPA_TRUE_ACCUM=2         # +36% at 64n. Raises global batch; see caveat.

# compute
use_activation_checkpointing: false   # +9%, only affordable under HSDP
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
| **`VJEPA_TRUE_ACCUM=2`** | **+36%** (35.4 -> 48.2 clips/s) | job 8731439, paired, n=25/arm, **IQRs disjoint** |
| activation checkpointing OFF | +9% | job 8659973, same-node A/B |
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

## Caveats on the accum win

`TRUE_ACCUM=2` doubles global batch (3072 -> 6144 at 256n), which shifts
EMA/warmup/lambda — those are raw STEP counts with no batch awareness, so they
must be re-derived (`scripts/gen_large_batch_configs.py --batch-mult 16`). The
+36% is **throughput only**; whether the larger batch *trains* as well is the
separate `lbA8`/`lbB8` question.

It is also a **lower bound at 256n**: accum halves the allreduce count, and each
avoided allreduce costs 63 ring hops at 64n vs 255 at 256n.

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
