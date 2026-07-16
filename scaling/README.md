# JEPA scaling-law pipeline (`scaling/`)

Compute-optimal (IsoFLOP) scaling-law tooling for V-JEPA 2.1. See `docs/JEPA_SCALING_LAWS_DESIGN.md`
for the why (loss-based law is ill-posed; fit a downstream/common-space metric instead) — but note that
doc is a SUPERSEDED design spec.

> **Results source of truth (2026-07-16):** `scaling/FINDINGS.md` (technical) and
> `scaling/scaling_law_report.html` (figures + assessment). The "current status" section near the
> bottom of THIS readme and the design doc's results are **stale**: the corpus is now PE-Video (not
> K400), Metric B has the std/intercept/CV-λ fix, we tried 6 metrics (CKA/kNN/Procrustes/RankMe), and
> the headline is a **null** — α consistent with 0 over 1e17–1e19; a 1e20 tier at gb=3072 is running.

**Everything runs under `module load frameworks`.** The FLOP/param/plan/gen/collect/fit/plot stages
are CPU/meta-device only; the two evaluators and training are GPU/MPI.

## Pipeline

```
flops.py ─► plan.py ─► gen_configs.py ─► [ TRAIN sweep ] ─► collect.py ─► fit.py ─► plot.py
              (manifest)   (per-cell YAMLs)                    (experiments.csv)  (α,β)  (PNG)
                                              │
                              eval_metric_a.py (SSv2 probe)  ─► metric_A.json ─┐
                              eval_metric_b.py (frozen-T*)   ─► metric_B.json ─┴─► joined by collect.py
```

### 1. Plan the IsoFLOP grid
```
python -m scaling.plan --budgets logspace:3e17:1e19:4 --global-batch 72 --ipe 500 \
    --frames 16 --res 384 --keep-enc 1536 --keep-pred 2560 --out scaling/manifest.json
```
`--global-batch` is HELD FIXED across the whole sweep (decouples the data axis D from N). Cells too
big for a budget are flagged SKIP. `--budgets` accepts `a,b,c` or `logspace:LO:HI:N`.

### 2. Generate per-cell training configs
```
python -m scaling.gen_configs --base configs/.../vitG384_fixedshape.yaml \
    --manifest scaling/manifest.json --out-dir configs/scaling/run1 \
    --world-size 12 --data-root /flare/ModCon/ngetty/data/kinetics400_full
```
Overrides ONLY model_name/epochs/ipe/warmup/batch_size; inherits LR/wd/mask/aug verbatim. From-scratch
by default (`--cpt` to continue-pretrain for the stretch goal). A **geometry guard** aborts if the
manifest's res/frames/patch/tubelet ≠ the base config (prevents silent mis-scaling). global_batch must
be divisible by world_size.

### 3. Train the sweep
Launch each generated YAML with the Aurora launcher (`app.main_dist_aurora`). Each run writes a rank-0
`scaling.json` sidecar (identity + measured params) next to its `log_r0.csv`.

### 4. Score checkpoints — the REAL y-axis (do NOT fit raw pretraining loss)
**Metric A — SSv2 downstream probe error:**
```
python -m scaling.eval_metric_a gen --base configs/eval_2_1/vitg-384/ssv2.yaml \
    --runs 'experiments/run1/*' --out-dir configs/scaling/run1_ssv2 \
    --ssv2-train <ssv2_train.csv> --ssv2-val <ssv2_val.csv> --num-classes 174
# launch the emitted *_ssv2.yaml with the Aurora eval launcher, then:
python -m scaling.eval_metric_a read --configs 'configs/scaling/run1_ssv2/*_ssv2.yaml'
```
**Metric B — frozen common-space linear-predictivity error** (cross-scale comparable; per run):
```
python -m scaling.eval_metric_b --run-dir experiments/run1/vit_large_C1e18 \
    --t-star-ckpt /flare/ModCon/ngetty/checkpoints/vjepa2_1_vitG_384.pt \
    --t-star-model vit_giant_xformers --data-glob '<held_out_clips>/*.mp4'
```

### 5. Collect → fit → plot
```
python -m scaling.collect --runs 'experiments/run1/*' --csv scaling/experiments.csv
python -m scaling.fit  --csv scaling/experiments.csv --metric metric_a_error_acc --maximize? 
python -m scaling.plot --csv scaling/experiments.csv --metric metric_a_error_acc --out scaling/run1.png
```
`--metric` selects the y-axis column: `loss_main` (diagnostic ONLY — not comparable across N),
`metric_a_error_f1`/`metric_a_error_acc` (Metric A), or `metric_b_error` (Metric B). The fit uses only
`status==done` cells unless `--all-status`; it rejects inverted parabolas and edge/extrapolated optima.

## Files
| file | stage | notes |
|---|---|---|
| `flops.py` | FLOPs+params | term-by-term (not 6ND); exact params via meta-device |
| `plan.py` | IsoFLOP plan | fixed global batch; steps=C/(tf·gbatch); `--max-corpus-epochs` parallelogram cap |
| `topology.py` | per-size launch | hold gbatch fixed, factor into (tiles,per_rank_bs,accum,ddp/hsdp); cell packing |
| `gen_configs.py` | configs | drift-safe overrides + geometry guard; `--use-topology` emits `<slug>_launch.json` |
| `gen_calib_configs.py` / `read_calib.py` | calibration | measure per-size max batch + clips/s (throughput) |
| `collect.py` | runs→CSV | last-K-mean loss + status gate + metric-sidecar join |
| `fit.py` | α,β | PRISM parabola/power-law/bootstrap; measured D_opt; configurable metric |
| `plot.py` | figure | 2×2 IsoFLOP; reuses fit.py |
| `eval_metric_a.py` | Metric A | SSv2 frozen-probe driver (gen/read); GPU |
| `eval_metric_b.py` | Metric B | frozen-T* linear-predictivity (dimension-agnostic); self-contained decord loader; runs CPU or XPU |
| `overnight_chain.py` | run driver | **self-resubmitting** 1h debug-scaling chain; resumes from latest.pth.tar; crash-loop/walltime guards |
| `sweep_launch.py` | run driver | topology-aware hold-node launcher (node blocks per cell, HSDP env) |
| `hold_node_job.py` / `hold_run.py` | run driver | generic hold-node control loop (iterate on bugs w/o requeue) |
| `fanout_pbs.py` | run driver | one multi-node PBS, N cells concurrent one-per-node |

Trainer touchpoint (NOT in `scaling/`): `app/vjepa_2_1/train.py` writes a rank-0 `scaling.json`
sidecar when a config carries a `scaling:` stamp, and resolves `embed_dim_encoder` for any ladder
size. Corpus ingest: `scripts/ingest_k400full_{to_wds.py,pbs.sh}` (raw K400 → WebDataset, lossless).

## Status (2026-07-10) — ⚠️ STALE, see FINDINGS.md
> The block below is the initial (mistaken) read. The clean 1e18 vertex did NOT survive the full study
> (wrong corpus + un-fixed metric). **Current status: `scaling/FINDINGS.md`.**

Pipeline validated end-to-end on synthetic data (fit recovers planted α=β=0.5) AND now on **real
Aurora runs**. Calibration complete (sweep ≈202 node-h; strong batch-amortization). Full-K400 corpus
ingested (241,258 clips). The DDP ladder is training via `overnight_chain` (self-resubmitting). First
real result: a clean **1e18 IsoFLOP parabola on Metric B** with N_opt ≈ vit_base (134M) — and Metric B
disagrees with raw loss (confirming the design thesis). Remaining: more budgets to complete for the
α/β exponent fit; Metric A (SSv2, staged at `/flare/ModCon/ngetty/data/ssv2_eval/`) needs a GPU slot;
HSDP giant/gigantic pending a smoke gate.

## The actual study as run (2026-07-16) — supersedes "The real study" below
- corpus: **PE-Video** `/flare/ModCon/ngetty/data/pe_video_wds/pe_video` (984K), NOT K400.
- configs: `configs/scaling/pe/*.yaml`; outputs `/flare/ModCon/ngetty/experiments/scaling_pe/<slug>/`.
- metric: Metric B with std+intercept+CV-λ fix, plus `metrics_zoo.py` (CKA/kNN/Procrustes/RankMe),
  fit + pre-registered vertex test in `fit_metrics.py`, joint-α fit in `joint_fit.py`.
- budgets 1e17–1e19 (gb=96) done; **1e20 tier at gb=3072** (large/giant/gigantic) running as a
  standalone anchor. Result: α consistent with 0 — see FINDINGS.md / scaling_law_report.html.

## The original K400 invocation (STALE — for reference only)
- corpus: `/flare/ModCon/ngetty/data/kinetics400_full_wds/kinetics400` (241K clips)
- configs: `configs/scaling/real/*.yaml` (+ `_launch.json` per cell)
- outputs: `/flare/ModCon/ngetty/experiments/scaling_real/<slug>/`
- run: `python -m scaling.overnight_chain start --ctrl <ctrl> --configs 'configs/scaling/real/*.yaml' --exclude giant,gigantic --nodes 16`
- check: `python -m scaling.overnight_chain status --ctrl <ctrl>`
- T*: `/flare/ModCon/ngetty/checkpoints/vjepa2_1_vitg_384.pt` (Meta ViT-g e40, model `vit_giant`)
