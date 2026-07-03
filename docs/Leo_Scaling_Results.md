# Results

Live table of pretraining runs and their downstream ASFormer probe metrics.
Kept in this project so all measurements land in one place. See `GOAL.md` for
what we're trying to answer.

## Pretraining runs

**Setup common to every row below**: V-JEPA 2.1 ViT-L, 256px × 16f × tubelet 2,
data = 4 surgical webdatasets (CRCD, SITL, surgenet_robotic, surgtoolloc2022,
sqrt-temperature sampling), block-3D masks, bf16, `epochs: 5`, ipe 300,
LR schedule and EMA identical across rows (only `nodes` differs).

**Strong scaling**: `prepare_runtime_config.py` halves per-rank batch as nodes
double, so **global batch stays 192** and total FLOPs are constant across rows.

| topology | ranks | per-rank bs | global bs | epoch-5 loss | walltime spent |
|---|---:|---:|---:|---:|---|
| 1 nodes × 12 tiles | 12 | 16 | 192 | **0.3157** | ~3 h wall (3 walltime-cut jobs + 2 preempts) |
| 2 nodes × 12 tiles | 24 | 8 | 192 | **0.3375** | ~2 h wall (2 walltime-cut jobs) |
| 4 nodes × 12 tiles | 48 | 4 | 192 | **0.3430** | — |
| 8 nodes × 12 tiles | 96 | 2 | 192 | **0.3024** | — |
| 16 nodes × 12 tiles (weak-scale) | 192 | 2 | 384 | **0.3503** | — |

Same total FLOPs, same recipe → **2n loss is +6.9% above 1n**. First observable
scaling trend: at fixed budget, halving per-rank batch and doubling grad-sync
cost quality. Under the study's frame, the 2n config needs a different knob
setting (probably LR or EMA) to catch up.

**Note on 16n:** strong-scaling to 192 ranks would require per-rank bs=1,
which crashes the trainer (`d_ij.unsqueeze(2)` IndexError in the multi-source
weight path). So 16n uses **weak scaling with bs=2** — global batch = 384,
2× the FLOPs of 1n/2n/4n/8n. That row is included as a data point but is
NOT under the fixed-FLOPs constraint of the rest of the study; interpret it
as "same wall-clock budget, more compute" rather than a direct scaling
comparison.

Checkpoints snapshotted for probing:
- `probes/checkpoints/1node_baseline_e3.pth.tar` (loss 0.310, taken during chain)
- `probes/checkpoints/1node_baseline_e5.pth.tar` (final 1n; loss 0.316)
- `probes/checkpoints/2node_baseline_e5.pth.tar` (final 2n; loss 0.338)

## ASFormer probes (SAR-RARP50 action seg, 8 classes)

Head: ASFormer (10 layers, 8 heads), 3 clips × 16 frames @ 384px, 20 epochs
configured, early-stop patience 6. Each probe is 2 nodes × 12 tiles.

Reported: **best val across the epochs the probe completed** (all runs so far
were walltime-cut at 9/20 probe epochs and were still climbing, so numbers
are lower bounds on what each encoder can support).

| encoder                                       | probe epochs run | best val_acc | best val_macro_F1 |
|-----------------------------------------------|-----------------:|-------------:|------------------:|
| Meta V-JEPA 2.1 (ViT-L distilled from ViT-G)  |                9 |       81.21% |         **73.43** |
| Our 1n pretrain @ epoch 3                     |                9 |       41.88% |             21.33 |
| Our 1n pretrain @ epoch 5                     |                9 |       48.58% |         **33.04** |
| Our 2n pretrain @ epoch 5 | 9 | 50.87% | **34.24** |
| Our 4n pretrain @ epoch 5 | 9 | 50.01% | **33.28** |
| Our 8n pretrain @ epoch 5 | 10 | 47.83% | **31.23** |
| Our 16n pretrain @ epoch 5 (weak-scale, 2× FLOPs) | 9 | 47.95% | **30.79** |

## Observations so far

- **E3 → E5 lifted probe val_F1 by 21.3 → 33.0 (+55% relative).** Our pretrain
  is far from saturation at 5 epochs (~7.5k iterations); the "5-epoch" budget
  is a starting point for the scaling study, not an endpoint. Whatever gap
  the 2n and 4n runs open vs 1n is exaggerated by how early we're measuring.
- **Meta is far ahead** — expected. Meta's ckpt has orders of magnitude more
  pretraining behind it. It's the ceiling reference, not something we're
  trying to beat at 5 epochs.
- **All three completed probes were still climbing on val_F1 at walltime cut.**
  Follow-up: chain-resubmit the probes so early-stop actually fires. Numbers
  above under-report each encoder's downstream ceiling by an unknown amount.

## Aurora reliability notes (context for pretraining walltime spent)

- Roughly 1 in 3 pretraining jobs on `debug` / `debug-scaling` dies mid-epoch
  with `Exit_status=143` (SIGTERM) triggered by a rank-level `signal 6`
  (`SIGABRT with core dump`) inside `torch.autograd` backward — i.e. a native
  XPU/oneCCL abort, not a code bug. Live PBS stdout via a `tee` patch was
  needed to see this; PBS's `.e`/`.o` files were never flushed on those runs.
- Mitigation used: chain-watcher (`scripts/watch_chain_v2.sh`) resubmits from
  `latest.pth.tar` until the target epoch is saved. Checkpoints are saved
  only at epoch boundaries by the trainer, so any mid-epoch death loses the
  in-progress epoch's work.

## Effective training time per topology

The wall-time budget PBS records includes queue overhead, checkpoint reload,
distributed init, and dataloader warm-up for every resubmit — plus the entire
runtime of any job that died mid-epoch. What we actually want is **the time
the trainer spent doing training iterations**. This comes straight from the
per-iter timings the trainer emits in log_r0.csv.

- **median iter (s)**: median iter-time(ms) across all logged iters, after
  trimming the top 10% (crash-recovery warm-ups, first-of-run outliers,
  checkpoint-save iters).
- **productive 5-epoch time**: median iter x 1500 (5 epochs x ipe=300). This
  is the "if the job never died and never queued, how long would 5 epochs
  take end-to-end at steady-state throughput" number.
- **actually logged iter time**: sum of iter-time(ms) across every logged
  iter, including in-progress epochs from aborted jobs. Larger than productive
  when aborts caused repeated work.

| topology | ranks | iters logged | median iter (s) | productive 5-epoch time | actually spent iterating |
|---|---:|---:|---:|---:|---:|
| 1n x 12 tiles | 12 | 2680 | 3.56 | 1h29m | 2h56m (aborts redid work) |
| 2n x 12 tiles | 24 | 1534 | 2.87 | 1h12m | 1h17m |
| 4n x 12 tiles | 48 | 1709 | 2.47 | 1h02m | 1h12m |
| 8n x 12 tiles | 96 | 1779 | 2.29 | 0h57m | 1h10m |
| 16n x 12 tiles (weak-scale) | 192 | 1360 | 2.53 | 1h03m | 1h01m |

**Reading the productive-time column:** this is the fair "how long does 5
epochs of pretraining take on N nodes" number, decoupled from Aurora
reliability and queue variance.

- Fixed-FLOPs rows (1n -> 8n): productive time falls monotonically from
  1h29m to 57m as we scale -- the extra ranks *are* buying wall-time speedup,
  even though the pretraining loss got worse at 2n/4n and only recovered at
  8n. Speedup 1n -> 8n: **~1.55x** with 8x the hardware (parallel efficiency
  ~19%). Most of the "missing" 6.5x is dataloader-bound: the surgical
  webdatasets don't scale I/O linearly with rank count, and per-rank batch
  shrinks from 16 to 2 so each rank does less compute per iter but the same
  synchronization.
- 16n (weak-scale, 2x FLOPs): productive time stays at ~1h03m even with 2x
  the global batch -- so at 16 nodes we're spending double the FLOPs for
  the same wall-time. That's a positive scaling signal *for weak scaling*
  (you get 2x the samples per wall-clock hour), but it's not comparable to
  the fixed-FLOPs rows.

## v3 series (aligned with ngetty's recipe)

All previous rows used a stale June-23 snapshot of the trainer with a
different recipe. The v3 rerun aligns with ngetty's current
`v3_lambdaoff_cleandata.yaml` (git bbe97d6):

- **Init from Meta ckpt** `vjepa2_1_vitl_dist_vitG_384.pt` (previous runs
  trained from scratch — different starting point, so the pretraining
  losses aren't directly comparable to the v1 series above).
- **11-dataset mix** (was 4): + endovis15, jigsaw, kinetics400,
  miccai_2017, miccai_endoseg, surgvisdom, surgvu24.
- **`min_clip_std: 1.0`** filter drops surgvu24's ~22% pure-black clips.
- **LR = 7.5e-5, warmup = 2 epochs** (was 6e-4 / 40 warmup — ~10x lower).
- **`ipe: 500`, epochs = 5** → **2500 optimizer steps** on every topology
  (the scaling variable — matched-steps across N, not matched-FLOPs).
- **`lambda_value_vid: 0.0`** (context-loss branch off, matches ngetty
  isolation).
- Per-rank **batch = 2 on every N**, weak-scale. Global batch grows with
  nodes: 1n=24, 2n=48, 4n=96, 8n=192, 16n=384.

| topology | ranks | per-rank bs | global bs | steps | epoch-5 loss |
|---|---:|---:|---:|---:|---:|
| 1n x 12 tiles | 12 | 2 | 24 | 2500 | pending |
| 2n x 12 tiles | 24 | 2 | 48 | 2500 | pending |
| 4n x 12 tiles | 48 | 2 | 96 | 2500 | pending |
| 8n x 12 tiles | 96 | 2 | 192 | 2500 | pending |
| 16n x 12 tiles | 192 | 2 | 384 | 2500 | pending |

### v3 ASFormer probes (SAR-RARP50 macro-F1)

Same probe recipe as the v1 series (20 epochs configured, 9 completed at
walltime cut).

| encoder | probe epochs | best val_acc | best val_macro_F1 |
|---|---:|---:|---:|
| v3 1n pretrain @ epoch 5 | pending | - | - |
| v3 2n pretrain @ epoch 5 | pending | - | - |
| v3 4n pretrain @ epoch 5 | pending | - | - |
| v3 8n pretrain @ epoch 5 | pending | - | - |
| v3 16n pretrain @ epoch 5 | pending | - | - |

## Files

- Pretrain runs: `runs/{1node,2node,4node}/baseline*/`
- Probe configs: `probes/configs/`
- Probe runs: `probes/runs/`
- Snapshot ckpts: `probes/checkpoints/`
- Legacy per-probe notes: `probes/results.md` (superseded by this file)
