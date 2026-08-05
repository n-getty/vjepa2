# ALCF Ticket Draft — FSDP gradient-AllReduce deadlock at 192 ranks on Aurora (XPU/xccl)

**Date:** 2026-07-05 (updated 2026-07-06)
**System:** Aurora (Intel Max 1550 / PVC, 12 tiles/node)
**Scale:** 16 nodes × 12 tiles = 192 ranks
**Stack:** torch 2.13.0.dev (xpu), native `xccl` backend, `module load frameworks` (2025.3.1) base;
CCL_PROCESS_LAUNCHER=none, CCL_ATL_TRANSPORT=ofi, FI_PROVIDER=cxi
**Workload:** V-JEPA 2.1 ViT-G **2B** (vit_gigantic, depth 48, embed 1664) continued-pretrain,
PyTorch FSDP1 HYBRID_SHARD_ZERO2 (shard within a node's 12 tiles, replicate across the 16 nodes).

## Summary

At 192 ranks, the training job intermittently **deadlocks in the FSDP gradient AllReduce** during
the backward pass. It is **not a crash and not a single dead rank** — it is a collective
desynchronization: the cohort splits across an FSDP collective boundary and blocks forever. Frequency
is roughly **one deadlock per ~3h43m** of wall-clock at this scale (measured MTTF). The identical software stack,
launcher, and config running the **1B model (vit_giant, depth 40) is clean** for >11k iterations —
the 2B (depth 48) has ~2× the per-rank gradient-AllReduce payload and correspondingly longer
collective windows. **Note:** this 1B-vs-2B discriminator is consistent with *either* "more bytes →
more fabric stress" *or* "longer window → more opportunity for an iteration-schedule skew to open";
it does not by itself favor a fabric cause (see Ask #2), and we do not present it as such.

This appears to be the same silent-hang class other Aurora large-scale jobs report (e.g. the AGPT/
torchtitan runs at 256N document "silent hangs, no traceback, blind-rotate a node"). **We captured the
traceback** via a per-rank `faulthandler.dump_traceback_later` watchdog, which is attached below.

## Captured evidence (REPRODUCED — 2 instrumented hangs, same signature)

**Both captured hangs split the cohort across TWO different collectives on TWO different process
groups** — the majority in the cross-node gradient AllReduce, a minority one full iteration ahead in
the intra-node forward all-gather. Counting each rank's **main-thread** stack (raw frame counts
double-count because both the 600 s timer and the shell's SIGUSR1 fire, producing 228 dump headers
for 192 ranks — we de-dup to the main-thread program stack):
- job 8645821 (e98): **395 → 24** normalized to 192 ranks = **~181 backward / ~11 forward**.
- job 8646258 (e145, ~04:00 UTC 2026-07-06): **371 → 24** frames = the **same 24 in forward**.
(A third hang, job 8645921, blocked in an application-level all_reduce we have since removed; not
relevant to the FSDP-collective signature.)

**The forward-unshard count is 24 in BOTH independent captures, and 24 is node-quantized**
(2 × 12 tiles/node, or 1 × 12 if those ranks double-dumped). A random per-rank straggler would not
land on an exact node multiple twice; this already leans toward *whole-node(s) stalled in the shard PG*
rather than scattered ~1/node. **Full node attribution is pending the next capture** — see the "Two
readings" box below and the forensics upgrade in "Mitigation".

## Captured evidence detail (job 8645821, ~19:22 UTC, epoch 98 iter 15)

A per-rank 600 s stall watchdog dumped stacks (de-duplicated to main-thread program state per rank):
- **~181 ranks** blocked in the **backward gradient AllReduce** — the **cross-node replicate PG**:
  ```
  torch/distributed/distributed_c10d.py:3239 in all_reduce
  torch/distributed/c10d_logger.py:83 in wrapper
  torch/distributed/fsdp/_runtime_utils.py:872 in _reduce_grad
  torch/distributed/fsdp/_runtime_utils.py:766 in _post_backward_hook
  (autograd engine: torch/autograd/graph.py:913 _engine_run_backward)
  ```
- **24 ranks** one full iteration ahead, in the **forward all-gather (unshard)** — the **intra-node
  shard PG** (`all_gather_into_tensor`):
  ```
  torch/distributed/fsdp/_runtime_utils.py:421 in _pre_forward_unshard
  torch/distributed/fsdp/_runtime_utils.py:305 in _unshard
  torch/distributed/fsdp/_flat_param.py:1382 in unshard
  torch/distributed/fsdp/_flat_param.py:1476 in _all_gather_flat_param
  torch/distributed/distributed_c10d.py:4381 in all_gather_into_tensor
  ```

### Why this is a permanent deadlock, not a drained transient (two readings, one open datum)

A purely transient slowdown on a *matched* collective cannot hang permanently — peers wait, the slow
rank arrives, the collective drains. A permanent deadlock requires the delay to be converted into an
**ordering/PG mismatch**. Our config is **`_HYBRID_SHARD_ZERO2` (= SHARD_GRAD_OP)**: params stay
gathered from forward through backward, so there is **no backward re-unshard**. Therefore the 24 ranks
in `_pre_forward_unshard` are in a *genuine forward pass* — a full iteration skewed from the ~181 ranks
in backward. Two mechanisms fit this, and they imply **opposite answers to Ask #1**:

1. **Same-PG collective-order divergence** (if the stalled ranks share a PG with peers issuing a
   different collective): a shard-PG member issuing forward `all_gather` while its PG-peers issue
   backward `reduce` can never rendezvous. **No CCL timeout+retry can fix an ordering mismatch** — the
   fix is FSDP-schedule-side.
2. **Cross-PG straggler cascade** (if a whole node's shard-PG `all_gather` stalls → that node emits no
   rank into the replicate `all_reduce` → the other 15 nodes block there): here a **per-collective
   timeout on the replicate AllReduce is a plausible mitigation** — the waiters could bail out.

**The distinguishing datum is node-locality of the 24 forward ranks:** concentrated on 1–2 whole nodes
⇒ reading (2), cross-PG cascade, timeout-mitigable; scattered ~1/node ⇒ reading (1), same-PG order
divergence, timeout won't help. The 24 = node-multiple already hints at (2), but our current dumps are
**not rank/host-tagged** (faulthandler writes unprefixed stacks interleaved into shared job stdout), so
we cannot yet map the 24 to hosts. This is fixed for the next capture (per-rank dump files, see
Mitigation) and the result will be filled in here before filing.

**Interpretation:** the majority reach the backward `all_reduce` over the 16-node replicate group and
block waiting for the minority that are a full iteration behind in the forward all-gather. Those
lagging ranks cannot arrive because they are waiting on a collective the majority already left →
mutual deadlock. **This deadlock mechanism (an FSDP collective desync) is independent of *why* a rank
lags** — that is the substance of this ticket. Whether it "cannot tolerate a transiently-slow rank"
in a way a CCL timeout could fix depends on which of the two readings above holds — that is decided by
the node-locality of the 24 forward ranks (pending next capture), NOT presumed here.

**On the lag origin (cause still under investigation — stated as hypotheses, not settled):** at 2B
payload the AllReduce window is long enough that a straggler occasionally never rejoins. Candidate lag
sources, both present in the captured-hang jobs:
- **Transient fabric (CXI/dragonfly) contention** on some ranks' links — the leading hypothesis, but
  not yet isolated.
- **Decode-latency outliers.** The captured-hang jobs (8645821, 8646258) ran `num_workers=2` with a
  known high-bitrate source (heichole, ~1765 ms/clip decode, ~4× the other sources) in the catalog —
  a demonstrable per-rank lag source. In the live `num_workers=0` run, `dataload-time` is p50≈1.0 s /
  p99≈10 s with rare multi-second outliers (decode is on the critical path at nw=0), i.e. decode jitter
  of the same order as the backward spikes. We have since re-encoded heichole to remove this outlier.
  We have **not** confirmed `dataload-time` was flat at the deadlock iters of 8645821/8646258
  specifically, so decode is not excluded for those captures.

We are running a clean isolation: the live run is `num_workers=0` (removes the decode-worker path) with
the heichole bitrate outlier removed. If deadlocks persist at the same MTTF with decode thus ruled out,
that isolates the fabric hypothesis; that result will be appended here before/when this is filed.

Supporting data from per-rank CSV timing (steady-state, 192 ranks):
- Backward-phase time is bimodal: floor ~2 s (flat over the whole run — rules out a memory/IPC leak,
  which would raise the floor), with cohort-wide spikes where ~87 % of the 192 ranks are simultaneously
  elevated (not a single straggler node). The deadlock is the tail of this same distribution.
- `CCL_ALLREDUCE=ring` vs `double_tree` A/B: **no difference** (identical ~2 % spike rate) — the
  contention is not the AllReduce topology/hop-count.

## Reproducer

1. 16 nodes × 12 tiles, torch 2.13 xpu, native xccl, OFI transport (env above).
2. FSDP1 HYBRID_SHARD_ZERO2, a ~2B ViT (depth 48, embed 1664), bf16 autocast, bs=2/rank.
3. Run continued-pretrain; within ~3-4 h a backward `_reduce_grad` AllReduce deadlocks with the split
   shown above. 1B (depth 40) under the identical setup does not.

## What we've ruled out (with evidence)

- **Not a dead rank / not our watchdog / not walltime** — job Exit_status=0 paths; the stall watchdog
  (no-progress 1800 s) is what kills it; stacks show live threads blocked in a collective.
- **Not the AllReduce algorithm** — ring == double_tree.
- **Not L0-IPC-handle exhaustion / memory leak** — flat backward floor; stack is a clean torch
  `all_reduce` blocked on a peer, not an allocator/IPC path.
- **Not `libpil4dfs`/DAOS-17499** — we are on Lustre (`/lus/flare`), no pil4dfs.
- **Not a specific bad node** — different hosts each incident; nodefile attached per incident.

## Ask

1. **Which reading holds** (see "Two readings" above), and does it admit a CCL/xccl mitigation? If the
   24 forward-unshard ranks are concentrated on whole node(s) → cross-PG straggler cascade → is there a
   per-collective timeout on the replicate-group AllReduce that lets the 15 waiting nodes bail out
   instead of hanging? If scattered ~1/node → same-PG collective-order divergence → this is an
   FSDP-schedule issue no CCL timeout fixes; is there xccl guidance either way? (We will attach the
   node-locality breakdown from the next capture — forensics upgrade below.)
2. Is there fabric-level (CXI/dragonfly) guidance for 192-rank FSDP collective patterns at ~2B payload?
   (We are separately isolating whether the lag origin is fabric vs. host-side — see lag-origin note.)
3. Can the collective be made to surface a timeout error (so the framework can recover) instead of
   hanging silently?

## Attachments
- Full per-rank stack dump: job 8645821 `.OU` (search `Timeout (0:10:00)!`) — NOTE: stacks are
  interleaved and un-prefixed in this capture (see forensics upgrade); node attribution pending.
- Per-incident nodefiles (16 hosts each): `checkpoints/surg_2_1_vitG384_fixedshape/vitG384_n16g12_weak/hang_diag/`.

## Our mitigation meanwhile (working)
Per-rank + shell watchdog captures forensics then kills; a self-healing PBS resubmit auto-resumes from
the last checkpoint (every ~epoch). Net: ~1 deadlock/3h43m (measured MTTF), all recovered, training
progresses (e15→e214+, ~Epoch 313, across many self-healing jobs; walltime set to 4 h in-script since
MTTF < walltime means no job reaches the wall). We accept the throughput tax; the ask is whether the
deadlock itself can be avoided or made recoverable in-framework.

**Forensics upgrade (for the next capture, so the two-readings question is answerable).** The current
watchdog writes faulthandler stacks un-prefixed into shared job stdout, so the stalled ranks cannot be
mapped to hosts — exactly the datum Ask #1 turns on. Fixed: each rank now dumps to its own
`hang_diag/stack_rank<NNNN>_<host>.txt` (verified: `file=` kwarg works on this build; SIGUSR1 self-dump
writes to the per-rank file). On the next hang, `grep -l _pre_forward_unshard hang_diag/stack_rank*.txt`
gives the exact hosts of the 24 forward ranks → resolves cross-PG-cascade vs same-PG-order-divergence
directly, and that breakdown will be added to "Captured evidence detail" before this ticket is filed.
