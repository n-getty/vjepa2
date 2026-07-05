# ALCF Ticket Draft — FSDP gradient-AllReduce deadlock at 192 ranks on Aurora (XPU/xccl)

**Date:** 2026-07-05
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
is roughly **one deadlock per ~3.5 hours** of wall-clock at this scale. The identical software stack,
launcher, and config running the **1B model (vit_giant, depth 40) is clean** for >11k iterations —
the 2B has ~2× the per-rank gradient-AllReduce payload.

This appears to be the same silent-hang class other Aurora large-scale jobs report (e.g. the AGPT/
torchtitan runs at 256N document "silent hangs, no traceback, blind-rotate a node"). **We captured the
traceback** via a per-rank `faulthandler.dump_traceback_later` watchdog, which is attached below.

## Captured evidence (job 8645821, ~19:22 UTC, epoch 98 iter 15)

A per-rank 600 s stall watchdog dumped stacks. Of **145 per-rank dumps** captured before the kill:
- **114 ranks** blocked in the **backward gradient AllReduce**:
  ```
  torch/distributed/distributed_c10d.py:3239 in all_reduce
  torch/distributed/c10d_logger.py:83 in wrapper
  torch/distributed/fsdp/_runtime_utils.py:872 in _reduce_grad
  torch/distributed/fsdp/_runtime_utils.py:766 in _post_backward_hook
  (autograd engine: torch/autograd/graph.py:913 _engine_run_backward)
  ```
- **10 ranks** one collective ahead/behind, in the **forward all-gather (unshard)**:
  ```
  torch/distributed/fsdp/_runtime_utils.py:421 in _pre_forward_unshard
  torch/distributed/fsdp/_runtime_utils.py:305 in _unshard
  ```
- 21 in neither / transitional.

**Interpretation:** the majority reach the backward `all_reduce` over the 16-node replicate group and
block waiting for the minority that are still in the previous/next collective (forward unshard). Those
lagging ranks cannot arrive because they are waiting on a collective the majority already left →
mutual deadlock. The lag origin is consistent with **transient fabric (CXI/dragonfly) contention** on
some ranks' links: at 2B payload the AllReduce window is long enough that a straggler occasionally
never rejoins.

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

1. Is there a known xccl/CCL setting to make the 16-node replicate-group AllReduce tolerant of a
   transiently-slow rank (timeout+retry, or a progress-thread/kvs tuning) rather than deadlocking?
2. Is there fabric-level (CXI/dragonfly) guidance for 192-rank FSDP collective patterns at ~2B payload?
3. Can the collective be made to surface a timeout error (so the framework can recover) instead of
   hanging silently?

## Attachments
- Full per-rank stack dump: job 8645821 `.OU` (search `Timeout (0:10:00)!`).
- Per-incident nodefiles: `checkpoints/surg_2_1_vitG384_fixedshape/vitG384_n16g12_weak/hang_diag/`.

## Our mitigation meanwhile (working)
Per-rank + shell watchdog captures forensics then kills; a self-healing PBS resubmit auto-resumes from
the last checkpoint (every ~epoch). Net: ~1 deadlock/3.5 h, all recovered, training progresses
(e15→e106+ across several 6 h jobs). We accept the throughput tax; the ask is whether the deadlock
itself can be avoided or made recoverable in-framework.
