# ViT-G 2B @ 16n — allreduce spike investigation (2026-07-03, overnight)

## TL;DR
The 2B run at 16 nodes is **not launchable with any config-only change.** Two separable
problems found; one is fixed, one needs code or an ALCF fabric fix.

1. **bf16 comm hook → L0/CCL resource exhaustion** — FIXED by turning the hook off
   (`VJEPA_BF16_COMM=0`). With it on: UR40 OOM (fpcs16) or GPU page-fault `banned:1`
   (fpcs8) at iter 2.
2. **Inter-node allreduce WEDGE** — NOT fixed by any no-code lever. One rank arrives
   ~300s late at the gradient allreduce (compute uniform across all ranks, different
   rank each spike, worsens over iters). Intrinsic to 2B's ~1.8× per-rank gradient-AR
   volume on Aurora's CCL/CXI stack. The 1B ran clean 11,290 iters on the identical
   launcher because its AR volume is ~half.

## What was tested (7 configs, all ≤45min diagnostic jobs, no long/wasteful runs)
| # | Config | Result |
|---|--------|--------|
| baseline | 2.10 + hook + OP_SYNC=1 | wedge spikes 9→150s (orig symptom) |
| op0 | 2.10 + hook + OP_SYNC=0 | wedge from iter 0 (worse) |
| pt213 | 2.13 + hook + fpcs16 | no wedge onset, but UR40 OOM @ iter 2 |
| C | 2.13 + **no-hook** + fpcs16 | **no OOM (44 iters)**, but still wedges (runaway iter 35+) |
| D | 2.13 + hook + fpcs8 | GPU page-fault banned:1 @ iter 2 (fpcs8 ≠ the lever) |
| E | 2.13 + no-hook + **8 nodes** | wedge persists (not hop-count-driven), walltime-killed |
| F | 2.13 + no-hook + **OFI/launcher=none** | HUNG at iter 0 (worse than pmix/mpi) |

Evidence for the wedge being fabric/collective (not compute/straggler-node): at a 330s
spike, `fwd-context` was 742–892ms uniform across all 192 ranks; `backward-ms` showed
one rank +328759ms and every other rank negative (timer overflow from waiting). A
*different* rank each spike. Reproduced across two allocations → not a bad node.

## Key references
- Root-cause lead: `BaseMM_PRISM/.../scaling-study/investigation/REPORT.md:548-551`
  (torch 2.10 backward-collapse fixed in 2.13) — confirmed torch 2.13 removes the
  *onset speed* but not the wedge itself at our AR volume.
- UR40 / IPC-handle: `torchtune/CLAUDE.md:48` ("accumulates IPC handle memory by
  step 1 BWD → OOM at step 2") — matched our iter-2 OOM exactly; our bf16_compress_hook
  is the per-step handle churner.
- torch 2.13 XPU venv: `/flare/ModCon/ngetty/venvs/torchtune-pt213-xpu` (native xccl,
  no ipex). Installed missing deps: cv2/scipy/decord. Runs vjepa2 fine.

## Paths forward (need your decision — none are overnight-safe)
- **(a) Grad accumulation** in `app/vjepa_2_1/train.py` — fewer, larger allreduces.
  torchtune's single biggest win (+68%). The most promising real fix. Scope: accumulate
  N microbatches, `model.no_sync()` on non-final, divide loss by N, keep EMA/LR
  denominator = ipe*epochs (do NOT divide ipe), preserve the d_weights bs≥2 path.
  Correctness-critical trainer change → must be reviewed, not shipped blind.
- **(b) ALCF/Intel ticket** — reproducer in hand: "2B DDP gradient allreduce at 192
  ranks, one rank intermittently ~300s late, compute uniform, worsens over iters;
  1B identical launcher fine."
- **(c) Run degraded NOW** if you need a model before (a)/(b): torch 2.13 +
  `VJEPA_BF16_COMM=0` + pmix/mpi + fpcs16 (Test C config). It progresses (44 iters,
  no crash) but at ~3-5× wall due to spikes. Wasteful but produces a checkpoint.

## Recommendation
Pursue (a) grad-accum with review this morning — it directly attacks the AR-frequency
that scales the wedge, and it's a durable capability. File (b) in parallel. Avoid (c)
unless a model is urgently needed.

Full blow-by-blow in memory: `vitG-2b-allreduce-spikes.md`.
