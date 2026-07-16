# Neural Scaling Laws for V-JEPA 2.1 — Experimental Design Spec

> ## ⚠️ SUPERSEDED (2026-07-16) — DESIGN/MOTIVATION ONLY
> This is the **pre-sweep design spec** (frozen 2026-07-10). Its *motivation and design* are still
> valid (§1 why loss-based law is ill-posed, §0a code map, §10 data audit), but **all results,
> decisions, and status below are OUT OF DATE** and in several cases refuted. Do NOT cite results from
> this file.
>
> **Current sources of truth:**
> - **`scaling/scaling_law_report.html`** — polished report with figures + paper-readiness assessment.
> - **`scaling/FINDINGS.md`** — detailed running lab notebook (the authoritative technical record).
>
> **What changed since this spec (see FINDINGS.md for detail):**
> - **Corpus:** K400 (this doc) → **PE-Video (984K)** — K400 too small (replay saturated the vertices).
> - **Metric B:** raw fixed-λ ridge (this doc) → **standardized X + intercept + CV-λ** (fixed a
>   dimension confound where score crept with encoder width independent of quality).
> - **More metrics tried (all fail to resolve a vertex):** linear CKA, RBF CKA, mutual-kNN (Platonic),
>   orthogonal Procrustes, and **RankMe** (reference-free) — not just Metric A/B.
> - **T\* ceiling:** the open question in §7 was tested — re-scoring against a **2B ViT-G** reference
>   (above the whole ladder) barely moved the vertices, **refuting** T\*-size saturation as the cause.
> - **The "first real result" below (N_opt ≈ base @1e18) does NOT hold** across budgets — the vertex
>   floats within metric noise; per-budget vertices do not order with compute.
> - **Result:** across 6 metrics × 2 references × per-budget and joint IsoFLOP fits, the compute-optimal
>   exponent **α is statistically consistent with 0** over our 1e17–1e19 range (best: mutual-kNN
>   α=+0.11, CI [−0.08, +0.33]); dim-invariant metrics exclude the LLM value 0.5. Not resolvable at
>   this compute — needs ≥3 decades. A **1e20 tier at gb=3072** (large/giant/gigantic) is now running
>   as a standalone high-fidelity anchor; **3e20 is corpus-blocked** (replay = C/corpus, GB-independent).
> - **Global batch:** this doc's gb=96 is 32× below the real V-JEPA 2.1 recipe (gb=3072).

**Status:** SUPERSEDED design spec — see banner above. (Original: LIVE, pipeline built + validated,
sweep training on Aurora, updated 2026-07-10.)
**Date:** 2026-07-09 (rev 2026-07-10; superseded 2026-07-16)
**Owner:** Neil Getty
**Code:** all tooling under `scaling/` (see `scaling/README.md`); configs `configs/scaling/real/`;
outputs `/flare/ModCon/ngetty/experiments/scaling_real/`.
**Decision context:** New parallel research line. Fit compute-optimal (Chinchilla-style) scaling
relationships for JEPA-style video SSL: at fixed FLOP budgets, sweep model size to find the
param/data-optimal recipe, then fit the optimal-frontier exponents.

**Progress at a glance (2026-07-10):** full pipeline validated end-to-end on synthetic (recovers
planted α=β=0.5) and on real Aurora runs. Calibration done. Full-K400 corpus (241K clips) ingested.
DDP ladder training via a self-resubmitting chain. **First real result: a clean 1e18 IsoFLOP parabola
on Metric B, N_opt ≈ vit_base (134M); Metric B and raw loss disagree (confirming the core thesis).**
Pending: more budgets for the α/β fit, Metric A (SSv2, staged), HSDP giant/gigantic smoke gate.

**Scope (IMPORTANT):** This line is **general first, NOT domain-specific.** The scaling law is fit
on general video SSL (standard corpora + standard downstream benchmarks), so the result is a
statement about JEPA scaling in general — not about surgery. Surgical data/probes are relegated to a
**stretch goal** (§9): does a domain-specific from-scratch pretrain follow the same law, and how does
it compare to continued-pretraining (CPT) from a general checkpoint?

---

## 0. One-paragraph thesis

No compute-optimal scaling law has been published for **any** JEPA variant (I-JEPA / V-JEPA /
V-JEPA 2). The obvious approach — fit the pretraining loss vs (N, D) — is **ill-posed** for JEPA
because the loss is an L1 distance in a *moving, per-model-size, learned* embedding space, with no
fixed data-entropy floor and only architectural collapse protection. We therefore fit scaling laws
to **two** well-posed y-axes in parallel: (1) **downstream probe error** (what the field does), and
(2) a **frozen common-space prediction loss** (intrinsic, novel). Running both on one sweep lets us
report whether an intrinsic JEPA loss and downstream quality track each other under scaling — itself
a publishable result.

---

## 0a. Code map (for collaborators — start here)

All scaling-analysis code is isolated under **`scaling/`** (a self-contained Python package; run
everything with `module load frameworks`). It does NOT fork the trainer — the only trainer touch is an
additive `scaling.json` sidecar in `app/vjepa_2_1/train.py`. Read `scaling/README.md` first.

| you want to… | look at |
|---|---|
| the whole pipeline + current run commands | `scaling/README.md` |
| FLOPs/params per model (analytic, meta-device) | `scaling/flops.py` |
| IsoFLOP planner (fixed global batch, parallelogram cap) | `scaling/plan.py` |
| per-size launch topology (DDP/HSDP, batch, packing) | `scaling/topology.py` |
| generate per-cell training configs | `scaling/gen_configs.py` → `configs/scaling/real/` |
| the two y-axis evaluators | `scaling/eval_metric_a.py` (SSv2 probe), `scaling/eval_metric_b.py` (frozen-T*) |
| collect → fit exponents → plot | `scaling/collect.py`, `scaling/fit.py`, `scaling/plot.py` |
| how the sweep actually runs unattended | `scaling/overnight_chain.py` (self-resubmitting chain) |
| calibration (per-size batch/throughput) | `scaling/gen_calib_configs.py`, `scaling/read_calib.py` |

**Data/artifacts (on flare, not in git):** corpus `data/kinetics400_full_wds/kinetics400` (241K clips);
SSv2 eval `data/ssv2_eval/{webm,labels}`; T\* `checkpoints/vjepa2_1_vitg_384.pt`; run outputs
`experiments/scaling_real/<slug>/`.

## 0b. First real result (2026-07-10) — ⚠️ SUPERSEDED, DOES NOT HOLD

> **This result did not survive the full study.** The N_opt≈base vertex shown here is from a single
> budget (1e18) on the old K400 corpus with the un-fixed Metric B. With the corrected metric, the
> PE-Video corpus, and all budgets, the per-budget vertices do NOT order with compute and α is
> consistent with 0. See `scaling/FINDINGS.md`. Kept below only as a record of the initial (mistaken)
> read.

The 1e18-FLOP IsoFLOP row, scored on **Metric B** (frozen-T\* linear-predictivity error, lower=better),
over the complete DDP ladder:

| model | params | Metric B (↓) | raw loss |
|---|---|---|---|
| vit_tiny | 49M | 0.856 | 0.568 |
| vit_small | 67M | 0.797 | 0.510 |
| **vit_base** | **134M** | **0.774** ← N_opt | 0.416 |
| vit_large | 355M | 0.835 | 0.444 |

Two takeaways: (1) a **clean U-shaped IsoFLOP parabola** with a fitted optimum at ~vit_base — real
compute-optimal behavior, first shown for JEPA video SSL. (2) **Metric B and raw loss DISAGREE** (loss
ranks large ≈ base as good; Metric B shows large is clearly worse at this budget) — empirical
confirmation of §1's thesis that raw JEPA loss is the wrong y-axis. Plot:
`experiments/scaling_metricb_first.png`. NOT yet the finished law — α/β exponents need ≥2 complete
budgets (only 1e18 is complete so far).

---

## 1. Why the naive loss-based law is ill-posed (the motivating argument)

The V-JEPA 2.1 training loss (`app/vjepa_2_1/train.py:1000-1055`) is

```
loss = mean( |z_ij - h_ij|^p ) / p      # p = loss_exp = 1.0 in prod → pure L1
```

where the target `h = LayerNorm(EMA_target_encoder(x))` under stop-gradient
(`train.py:962-982`, EMA update `train.py:1282-1292`), plus a 2.1 dense-context term
`+ λ·loss_context` (`train.py:1074-1091`) and an optional `1/d_ij` spatial reweight
(`train.py:1038-1041`).

Chinchilla (Hoffmann 2022) fits `L(N,D)=E+A/N^α+B/D^β` to cross-entropy, which has three
properties JEPA's loss lacks:

| Property CE has | JEPA loss |
| --- | --- |
| Fixed floor `E` = data entropy, same for all N | Floor is set by each model's *own* target space → **every N has its own E** |
| Absolute comparability across N (one ruler = nats) | Different N ⇒ different embedding dim/geometry (1024 vs 1408) → **incommensurable rulers** |
| Lower loss ⇒ strictly better model | Anti-collapse is architectural only (no VICReg term in loss path); LayerNorm blocks constant- but not **dimensional**-collapse → **lower loss can be worse** |

Empirical corroboration: Ma et al. 2024 (arXiv:2408.11243) show for graph SSL that pretraining
loss falls monotonically with scale while downstream does **not** scale — they abandon loss as the
y-axis. Xie et al. 2022 (arXiv:2206.04664) show MIM loss *does* track downstream — but only because
MIM's reconstruction loss lives in a **fixed** input space, which JEPA's does not. This repo's own
`docs/Leo_Scaling_Results.md:139` already notes JEPA losses "aren't directly comparable" across
initializations. **Conclusion: do not put the raw training loss on the scaling y-axis.**

---

## 2. The two y-axis metrics (DECISION: run both in parallel)

### Metric A — Downstream probe error (field-standard, defensible)
Fit `Error(N,D) = E + A/N^α + B/D^β` to a **frozen-backbone** probe on **general** video benchmarks
(not surgical).

- **Benchmark choice (general):** standard V-JEPA eval tasks — attentive/linear probe on
  **Kinetics-400** (action classification) and **SSv2** (Something-Something-v2, temporal/motion —
  the discriminating one, less saturated than K400), and optionally **ImageNet-1k** linear probe via
  the image branch. These are exactly the frozen-backbone evals in `evals/video_classification_frozen/`
  and the V-JEPA 2 paper's own scaling axis. Use probe **cross-entropy** (unbounded, floored at Bayes
  error) as primary, top-1 error as secondary.
- **Why not the surgical probes:** SAR-RARP50 F1@10 is saturated (memory `sar-f1at10-backbone-saturated`)
  AND domain-specific. Surgical probes move to the stretch goal (§9), not the main law.
- **Ceiling gate (mandatory):** before fitting, confirm the metric is *not* at ceiling for the
  **largest** N. K400 saturates earlier than SSv2 — prefer SSv2 for headroom at the top of the ladder.
- **Aggregate** across ≥2 tasks (as the V-JEPA 2 paper does) to reduce per-task ceiling artifacts.
- Harnesses already exist under `evals/video_classification_frozen/` — reuse (Aurora XPU port already
  in-tree per the asformer/probe work).

### Metric B — Frozen common-space prediction loss (intrinsic, novel)
Score every run's held-out prediction loss **in ONE fixed embedding space**, so the ruler stops
moving:

- Freeze a single reference target encoder `T*` — a general-video reference (candidate: Meta ViT-g
  V-JEPA 2, general pretrain, loadable via `p_file`/`load_pretrained`, `train.py:559-569`). Keep `T*`
  general so Metric B is not biased toward any domain.
- After each scaled run, on a fixed held-out set with a **fixed mask seed**, compute
  `|predictor_N(context) - LayerNorm(T*(x))|` — every N scored against the same target.
- **Dim-mismatch fix:** when the run's encoder dim ≠ `T*` dim, the predictor's output head must map
  into `T*`-space. Cleanest: fix the predictor output dim = `dim(T*)` for all runs (predictor is
  small relative to encoder, so this barely perturbs the N-axis). Document the exact convention.
- This is **eval-only** — swap `target_encoder → T*` in a copy of `forward_target`; **no change to
  training**.

### Metric C — collapse diagnostic (guard, not a y-axis)
Log **effective rank / RankMe** of the target patch-embedding covariance for every run. LayerNorm
hides dimensional collapse; this catches "low loss = collapsed, not good." Report alongside A and B.

**Cross-metric deliverable:** do Metric-A exponents and Metric-B exponents agree? If yes, the
intrinsic frozen-space loss is a cheap scaling proxy for JEPA (useful result). If no, it quantifies
exactly how JEPA loss and downstream quality decouple under scale.

---

## 3. Axes of the sweep

### N — model size (capacity ladder)
Use the **existing** ViT defs in `src/models/vision_transformer.py` — no new architectures needed:

| name | embed_dim | depth | heads | ~encoder params |
| --- | --- | --- | --- | --- |
| vit_tiny | 192 | 12 | 3 | ~5M |
| vit_small | 384 | 12 | 6 | ~22M |
| vit_base | 768 | 12 | 12 | ~86M |
| vit_large | 1024 | 24 | 16 | ~300M |
| vit_huge | 1280 | 32 | 16 | ~630M |
| vit_giant | 1408 | 40 | 16 | ~1.0B |
| vit_gigantic | 1664 | 48 | 16 | ~1.8B |

That's **7 points spanning ~5M→1.8B** — enough for ≥5 usable points per budget after edge-minima
trimming (PLAN.md rule).

**Param counts VERIFIED (2026-07-09) via `scaling/flops.py`** (meta-device instantiation through the
real `init_video_model` — exact, zero-memory): tiny 5.6M · small 21.9M · base 86.2M · large 303.9M
(≈ published V-JEPA2 ViT-L 300M ✓) · giant 1012M (≈ ViT-g 1B ✓) · gigantic 1844M (the "2B" ✓).
Predictor adds ~22–57M on top depending on encoder dim (pred in/out projections scale with enc dim).

**CONSTRAINT DISCOVERED — the 2.1 encoder only supports depths {12, 24, 40, 48}.**
`app/vjepa_2_1/models/vision_transformer.py:147-174` hardcodes `hierarchical_layers` (for the
dense-prediction distillation taps) only for those depths; any other depth hits
`else: print("Check the code! ;)")` and fails to construct (`vit_huge`, depth 32, is currently NOT
buildable in the 2.1 trainer). Implications for the ladder:
  - Usable off-the-shelf 2.1 sizes: tiny/small/base/large (12/12/12/24) + giant (40) + gigantic (48).
    `vit_huge` needs a `depth==32` branch added to that table before it can join the sweep.
  - Any *intermediate* capacity point we add (e.g. a width-scaled variant) must use one of {12,24,40,48}
    or extend the table. Prefer varying **width** at a fixed allowed depth to add points cheaply.
**Open question:** ladder mixes width+depth simultaneously (like the real ViT family). Chinchilla
purists vary along one shape family; decide whether to (a) accept the standard ViT ladder as-is
(pragmatic, matches published models) or (b) also add pure-width intermediate points. Recommend (a)
for v1, note the confound.

### D — data / compute-seen axis
- **Define D = number of clips seen** (samples), i.e. `global_batch × steps`. (Alternative units —
  frames, patch-tokens-seen — are monotone transforms; pick clips for interpretability, record the
  conversion factor to tokens for the FLOP model.)
- **Decouple D from N (critical).** Our current weak-scaling grows global batch with N
  (`docs/Leo_Scaling_Results.md`: 24→384), which **confounds D with N**. For the scaling sweep,
  hold **global batch fixed** across all N and vary **steps** to move along D. This is the single
  most important methodology fix.

### C — FLOP budget (the IsoFLOP grid)
- `C ≈ 6·N·D` is the LLM shortcut; **we will compute C from a real video FLOP model** (§6), not
  assume 6ND. The factor-6 also appears hardcoded in PRISM's `isoflop_fit.py`/`plot.py`
  (`D_opt = C/(6·N_opt)`) — must be replaced with our measured FLOPs/clip.
- Choose ~4–5 budgets spaced ~0.5 dex apart, sized to Aurora debug/prod walltime. Exact values set
  after calibration measures FLOPs/clip per model.
- Per PLAN.md: ≥5 capacity points per budget, no budget's optimum at a ladder edge, and never
  confound a backbone/arch change with a capacity change inside one budget.

---

## 4. Confounds to control (hard-won, from our own record)

1. **Data pipeline cleanliness — verify BEFORE trusting any curve.** Past losses were corrupted by
   the XPU SDPA bug (`use_sdpa:true` numerically broken — memory `xpu-sdpa-bug-confounds-pretraining`)
   and dataset-mixing oversampling (memory `dataset-mixing-oversamples-tiny-sets`). Confirm both are
   fixed in the exact configs used for the sweep. Run the `ml-data-pipeline-correctness` checks.
2. **Matched compute, not matched steps.** Our prior runs matched *steps* (2500) across N; that is
   **not** matched D or matched C. The IsoFLOP design fixes this by construction.
3. **Fix everything that isn't N or D:** data mix, `λ` schedule (or `λ=0` isolation as in v3 runs),
   mask config, `fixedshape` mask counts, fps/frames, optimizer/LR-warmup recipe. LR is the known
   JEPA collapse lever (memory `v2-lr-sweep-validation`) — use a per-N LR rule that is validated
   collapse-free, or the fits ingest collapsed cells.
4. **Convergence gate before a cell enters a fit.** Reuse PRISM `isoflop_collect.py`'s last-K-mean +
   `loss_stability` (stdev) flag; drop unconverged/collapsed cells from parabola fits (a collapsed
   cell poisons `N_opt`).
5. **No `device_id` on XPU multi-node, `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, no torch.compile** — standard
   Aurora-port constraints (CLAUDE.md). Sweep launcher must respect them.

---

## 5. Code reuse map

Reuse from `/flare/ModCon/ngetty/BaseMM_PRISM/tools/` and `nanochat/notebooks/`:

- **Verbatim (loss-agnostic):** `isoflop_fit.py` (parabola→vertex `N_opt`, log-log α/β power laws,
  bootstrap CIs), `isoflop_plot.py` (2×2 IsoFLOP figure). Only change: swap the hardcoded `6` in
  `D_opt = C/(6·N_opt)` for measured FLOPs/clip.
- **Adapt:** `isoflop_calibrate.py` (structure: instantiate → time warmup+timed steps → count params
  → emit JSON; replace transformer-LLM FLOP formulas with video ones), `isoflop_plan.py` (budget→
  steps rescale is generic; replace projector-variant axis with our ViT ladder), `isoflop_collect.py`
  (perf.jsonl→CSV pattern; rewrite column set + loss-source map for one video loss), and
  `scaling-study/reference/scaling_aurora.sh` (XPU/PBS sweep driver — already debugged on Aurora).
- **Methodology:** `scaling-study/PLAN.md` (budget ladders, ≥5-points/budget, variance-floor via
  seed replication, no-edge-minima).
- **Do NOT reuse:** any FLOP/param math that assumes vocab/lm_head/`bpb`/causal attention
  (`nanochat_flop_accounting.py`, `compute_effective_params`).

---

## 6b. Build status (updated 2026-07-09)

Implemented and tested under `scaling/`:
- **`scaling/flops.py`** ✓ — analytic FLOPs/clip (term-by-term, target-enc + context-enc + predictor
  per mask view) + exact params via meta-device `init_video_model`. Param counts verified vs published
  (L=303.9M, g=1012M, gg=1844M). Handles the mask keep-counts.
- **`scaling/plan.py`** ✓ — IsoFLOP planner. `budgets × ladder → {epochs, ipe, steps, D=clips_seen,
  actual_C}` at FIXED global batch (decouples D from N). `steps = C/(train_flops_per_clip·gbatch)`,
  rounds to whole epochs, flags cells too big for a budget (`min-steps`). Emits manifest JSON.
- **`scaling/gen_configs.py`** ✓ — manifest + base YAML → per-cell training YAMLs. Overrides ONLY
  model_name/epochs/ipe/warmup/batch_size; inherits LR/wd/ema/mask/aug/crop verbatim. From-scratch by
  default (`load_checkpoint:false`), optional `--data-root` corpus repoint, `--cpt` for the stretch
  goal, `scaling:` provenance stamp per config. **Geometry guard** refuses to emit if manifest
  (res/frames/patch/tubelet) ≠ base config — catches the silent mis-scaling drift class.

Pilot dry-run (6-model ladder × 4 budgets 3e17–3e19, gbatch=72, 384px) generates 22 runnable configs.
It also makes the **corpus-size ceiling concrete**: at 3e19, vit_tiny needs D=13M clips ≈ 54 epochs
over full K400 (~240K) — beyond that, budgets loop the corpus and D collapses into epochs-over-data.
Practical top budget for the general law on K400 is ~1e19; going higher needs a bigger corpus.

Also implemented and tested (2026-07-09):
- **trainer scaling logger** ✓ — `app/vjepa_2_1/train.py` writes a rank-0 `scaling.json` sidecar
  (run identity + measured param counts) when the config carries a `scaling:` stamp. Static, off the
  hot path (zero per-iter cost); collector joins it with the existing `log_r0.csv`.
- **`scaling/collect.py`** ✓ — runs → one `experiments.csv`. Last-K-mean loss + stdev stability
  (mirrors PRISM `_last_eval_losses`); status gate {done, collapsed, unconverged, incomplete, nan,
  no_loss}. Tested on synthetic runs — all 5 status classes classified correctly.
- **`scaling/fit.py`** ✓ — reuses PRISM parabola→vertex + power-law + bootstrap CI math, with two
  JEPA changes: (a) **configurable y-metric** (`--metric`; default loss_main for testing, point at a
  probe-error column for the real law; `--maximize` for accuracy); (b) **D_opt from measured per-clip
  FLOPs** (`D_opt = C / tf_per_clip(N_opt)` via log-log interp of the ladder), NOT the `6ND` shortcut.
  Convergence gate: only `status==done` cells fit unless `--all-status`. **Edge-minimum guard**:
  rejects a vertex outside the sampled ladder (extrapolated N_opt is unreliable — PLAN.md no-edge-minima).
  Validated: recovers planted α=0.5, β=0.5 (α+β=1.0) exactly from synthetic data.
- **`scaling/plot.py`** ✓ — 2×2 IsoFLOP figure (parabolas + N_opt stars, N_opt-vs-C, D_opt-vs-C,
  per-model spread), reuses `scaling/fit.py` so plot and numbers agree. Renders correctly.

**Full pipeline proven end-to-end on synthetic data with known exponents.**

Evaluators built and tested (2026-07-09):
- **`scaling/eval_metric_a.py`** ✓ — SSv2 frozen-probe driver. `gen` emits a per-checkpoint SSv2 eval
  YAML from a base config (overrides checkpoint/model_name/resolution/num_classes/folder, matched to
  each run's own size via its `scaling.json`); `read` harvests best_val_f1/acc from the eval's
  `log_r0.csv` into a `metric_A.json` sidecar with error columns. Probe hyperparams untouched (probe =
  constant function of the representation). Tested gen+read on fake runs.
- **`scaling/eval_metric_b.py`** ✓ — frozen common-space metric, **reformulated** to dodge the confirmed
  blocker: the run's predictor emits `4×encoder_embed_dim` (hierarchical levels: 4096 for large, 6656
  for gigantic), so you can't score different-sized runs against one fixed T* through their predictors
  without adding trained params. Instead we measure **linear-predictivity**: fit a ridge map from the
  run's frozen encoder features to the fixed T*'s LayerNorm'd features on a train split, report
  normalized residual (1−R²) on a held-out split. Cross-scale comparable (one fixed target space),
  dimension-agnostic (the map absorbs d_run→d_T*), encoder-only (no predictor). Alignment math
  unit-tested: self≈0.02, full-latent≈0.02, impoverished≈0.77, and d_run 72 vs 28 → d_T*=64 both map.
- **`scaling/collect.py`** extended ✓ — joins `metric_A.json`/`metric_B.json` sidecars into columns
  `metric_a_error_f1/acc`, `metric_b_error`. **Full-integration test passed**: metric_A sidecar →
  collector column → `fit.py --metric metric_a_error_acc` recovers planted α=0.5, β=0.5.
- **`scaling/README.md`** ✓ — end-to-end usage.

The whole pipeline (flops→plan→gen→train→[metric A/B]→collect→fit→plot) is wired and validated.
Remaining for real numbers: settle budget ladder + global batch (§7), stage SSv2 to flare, pick T*.
RankMe collapse diagnostic (Metric C) still a nice-to-have, not blocking.

## 6. Build list (the missing ~20%) — for the NEXT milestone after this doc

1. **Video FLOP model** — encoder + predictor FLOPs/clip from (frames, tubelet size, spatial
   patches, depth, dim, heads, mlp_ratio); no vocab/lm_head. Feeds calibrate/plan/fit. **Gating task.**
2. **Encoder+predictor param counter** — `N_total/N_active/N_trainable` convention; exclude EMA
   target (no-grad). Note: JEPA CPT often *starts from* a pretrained backbone, so "params trained"
   vs "params present" matters.
3. **Data-axis (D) plumbing** — clips-seen logging + conversion to patch-tokens for FLOPs.
4. **Metric-B eval harness** — frozen-`T*` held-out prediction-loss scorer (eval-only fork of
   `forward_target`), with the fixed predictor-output-dim convention.
5. **RankMe/effective-rank logger** (Metric C).
6. **Trainer-side scaling logger** — emit per-eval loss + param counts + clips-seen in a schema the
   adapted `isoflop_collect.py` reads.
7. **Config generator** — (ladder × budget × seed) → per-run YAMLs, fixed global batch, per-N LR
   rule, matching an existing clean config (e.g. `vitG384_fixedshape.yaml` lineage minus the
   confounds).
8. **(Optional) parametric fit** `L(N,D)=E+A/N^α+B/D^β` via scipy + Huber (Chinchilla "Approach 3").
   Neither repo has it; existing code gives Approach 1 (power laws) + 2 (IsoFLOP parabola) for free.

---

## 7b. Decisions RESOLVED (2026-07-10, from pilot throughput analysis)

Grounded in measured pilot clips/s + the FLOP model (not estimates):

- **Budgets & corpus:** 4 budgets `[1e18, 3e18, 1e19, 3e19]` on **full K400** (240K, downloaded,
  needs reshard). Run a **parallelogram, not a rectangle**: each budget uses only the sizes where
  `steps ≥ 200` AND `corpus_epochs ≤ ~12` (drops cells that loop the corpus so hard D collapses into
  epochs-over-data). ~**641 node-hours** total (vs 1036 for the naive rectangle). This keeps ≥5
  points/budget centered on the vertex. 3 budgets is the bare min for the outer `N_opt~C^α` fit; 4 is
  chosen for a usable slope+CI.
- **Global batch = 96, HELD FIXED** across the whole sweep (the D-vs-N invariant). Chosen as ~the
  MAX batch that still keeps all 6 sizes present at the 1e18 budget (gigantic@1e18 = 213 steps, just
  clears the 200-step floor; doubling gb drops the big models out of the low budget). 1/2/4 nodes →
  8/4/2 clips/tile, all ≥2 (weight_distance_loss constraint). Caveat: small vs Meta's ~3072 prod
  batch — big models are consistently under-batched, but the alternative (batch∝N) reintroduces the
  confound. Keep fixed, note it.
- **T\* (Metric B) = Meta V-JEPA2 ViT-g/16 @384 (1B, general)** — already on disk, general (no
  domain bias). MANDATORY ceiling gate: verify linear-predictivity `1−R²` still DECREASES
  giant→gigantic on held-out clips; if it ceilings at the top (our gigantic 1.9B > any Meta target),
  report Metric B tiny→giant and use Metric A (SSv2) for the gigantic point.
- **Per-size setup differentiation (efficiency, user directive 2026-07-10):** pilot proved small
  models are **overhead/latency-bound, not FLOP/memory-bound** at 256px (`fwd-target` flat ~520ms
  tiny→base, clips/s 20→15.3 over a 15× param range). Global batch stays fixed, but `gb = tiles ×
  per_rank_bs` is free to reshape per size:
  - tiny/small/base → **1–3 tiles, plain DDP, large per-rank bs (32–96)**; PACK multiple small cells
    per node (ZE_AFFINITY_MASK tile partition). Fewer ranks = less fixed comm; big per-rank bs
    amortizes the ~520ms per-iter floor. HSDP sharding is pure overhead here.
  - large → 6–12 tiles, DDP (transitions compute-bound).
  - giant/gigantic → 12+ tiles, **HSDP** (only here does param/opt-state sharding pay its comm cost;
    matches proven 2B recipe).
  - CALIBRATE first (debug-scaling smoke): confirm raising per-rank bs raises small-model clips/s
    (pilot ran bs=2 everywhere so this is unconfirmed and directly sets small-model cost).
- **Execution:** debug-scaling (1h) is smoke-only now — every real cell is multi-hour (1e19 =
  33–45 node-h, 3e19 = 108–135). Real study runs on **prod** with **trainer checkpoint-resume across
  2–3 prod jobs** for the big cells (proven tooling), NOT 4-node data-parallel (unvalidated at gb=96,
  16n allreduce-spike risk per memory `vitG-2b-allreduce-spikes`).

## 7. Open questions for review

1. **Ladder shape:** accept the standard ViT width+depth ladder (§3), or add pure-width points to
   isolate the exponent? (Recommend accept for v1.)
2. **Reference `T*` for Metric B:** Meta ViT-g e40, or a fixed *large* (not gigantic) so the biggest
   sweep models aren't scored against a smaller target? Trade-off: if `T*` is smaller than the run,
   its space may not resolve the bigger model's quality.
3. **Budgets & walltime:** how much Aurora compute is this line allotted? Sets the number/size of
   budgets and whether the top of the ladder (giant/gigantic) is in-scope for v1 or a later phase.
4. **Data corpus (general):** RESOLVED for v1 — see §10. Use the **Kinetics-400 WebDataset already on
   flare** (~136K mp4 clips, Aurora-reachable) as the general pretrain corpus; SSv2 on Polaris-eagle
   as the discriminating eval. Meta's full VM22M is not reproducible (YT-Temporal-1B + HowTo100M =
   >90% of it, YouTube-ID-only, heavy link rot). Open sub-question: is 136K clips enough D-range for
   the biggest budgets, or do we cap the top budget so we don't loop the corpus too many times (which
   confounds D with epochs-over-data)?
5. **Downstream probe for Metric A:** confirm SSv2 (and/or K400) has headroom at gigantic-N before
   committing (ceiling gate); K400 likely saturates first.

---

## 10. Data availability — what's actually reachable (audited 2026-07-09)

**Meta's V-JEPA 2 corpus = VideoMix22M (VM22M), NOT reproducible.** Composition (paper Table 1):
YT-Temporal-1B (19M samples, YouTube-IDs only), HowTo100M (1.1M, YouTube-IDs only), Kinetics
400/600/700 (733K, YouTube-IDs; K600 has no video mirror), SSv2 (168K, gated archive), ImageNet-1K
(1M images). The two dominant sources (YT1B + HT100M, >90% of samples and ~all the video-hours) are
YouTube-ID lists with years of link rot → **cannot rebuild VM22M**. This is exactly why practitioners
CPT from Meta's released checkpoints instead. Original V-JEPA used the smaller VideoMix2M (K710 + SSv2
+ HowTo100M).

**What we actually have (Aurora-reachable, on flare `/lus/flare/projects/ModCon/ngetty/data/`):**
- **Kinetics-400 WebDataset** — `surg_vid_webdataset/kinetics400/` (171 tars, mp4+json+cls, ~136K
  clips, 15 GB) and a resharded copy `surg_vid_webdataset_resharded/kinetics400/` (500 shards). This
  is a **subset** (full K400 is ~240K train clips / ~450 GB) but it's clean, local, and in the exact
  WebDataset format the trainer already consumes. **→ v1 general pretrain corpus.**

**Polaris-only (eagle, NOT mounted on Aurora — needs staging or run there) — VERIFIED via SSH 2026-07-09:**
- `/eagle/argonne_tpc/ngetty/data/20bn-something-something-v2` — **SSv2 REAL video archive**:
  **82,338 `.webm` clips, 7.0 GB**, with labels/splits in `.../data/labels/` (train/validation/test
  .json + labels.json) and a path CSV `.../data/ssv2_train_paths.csv` (82,337 rows). Also
  `20bn-something-something-v2-00/-01` (9.4G + 8.8G, likely additional splits/copies) and a
  `pretrain_neil/webdataset` dir on the same eagle root. → **Metric-A eval, fully reproducible.**
  Note 82K < full SSv2 (~220K all-splits) — this is the train-path subset; confirm the validation
  split clips are present before using as eval, or pull the ~25K val webms.
- `/eagle/tpc/ngetty/data/surgery/kinetics700_2020` — **METADATA ONLY, no videos.** Contains
  train/validate/test `.csv` (label,youtube_id,time_start,time_end,split) + matching `.json` with
  YouTube URLs. Confirms the general Kinetics link-rot problem — this is an ID list, not a corpus.
  **Does NOT extend the D-axis.** The local flare K400 WebDataset (§ above) remains the only real
  general video corpus we have on Aurora.

**Implication for design:**
- v1 pretrain on the local flare K400 WebDataset (~136K clips, Aurora, no staging blocker). K700-2020
  is metadata-only (verified) so it's NOT an option without a fresh YouTube scrape — deprioritize.
  Corpus-size ceiling: the biggest FLOP budget must not loop 136K clips so many times that D collapses
  into "epochs-over-data." If we need a bigger D-range, options are (a) re-download full K400 (~240K,
  ~450 GB) or (b) accept 136K and cap the top budget.
- SSv2 eval is the ceiling-safe Metric-A benchmark and IS a real archive on eagle (82K webms, 7 GB) —
  decide: **stage SSv2 webms to flare (~7 GB, one-time)** vs. run the eval leg on Polaris. Staging to
  flare keeps the whole loop on Aurora and is recommended (7 GB is trivial). Verify the validation
  split clips are staged, not just the 82K train subset.
- Metric-B reference `T*` = Meta's released V-JEPA 2 ViT-g checkpoint (we already load it) — general,
  no corpus reproduction needed.

---

## 9. Stretch goal — domain-specific from-scratch vs. CPT

Once the **general** law is fit, use it as the baseline for a domain question we're uniquely equipped
to answer (surgical data + probes already in this repo):

- **Q1 — does a domain follow the general law?** Fit the same IsoFLOP sweep on a domain corpus
  (surgical) from scratch. Compare exponents (α, β) and the compute-optimal N(C) frontier to the
  general law. Does surgical video scale like general video, or does the smaller/narrower distribution
  bend the curve (earlier data saturation, different optimal N)?
- **Q2 — from-scratch vs. CPT at matched compute.** For a fixed total budget C, compare:
  (a) domain pretrain **from scratch** with C, vs (b) **CPT**: general checkpoint (cost C_gen already
  spent) + domain continuation with C_dom. At what domain-data scale does from-scratch overtake CPT,
  if ever? This is the practical "should I CPT or train fresh" curve — directly useful and, combined
  with the general law, novel. Our record already has CPT-vs-raw signals (memory
  `triplet-2b-scale-vs-cpt`, `sar-f1at10-backbone-saturated`) but no *scaling-law* framing of it.
- **Metrics here:** the domain probes (SAR-RARP50, triplet IVT) are appropriate — with the ceiling
  caveat (F1@10 saturated; prefer triplet IVT / probe CE).
- **Reuse:** identical harness as the general sweep; only corpus + probe swap. This is why the general
  design keeps corpus and probe as pluggable axes (§3, §2).

This is explicitly a **later phase (P5+)**, gated on the general law landing first.

---

## 8. Suggested phase plan (after this doc is approved)

- **P0 (this doc):** design review, resolve §7.
- **P1:** build §6.1–6.2 (FLOP + param), validate against 2–3 known configs (sanity vs measured
  wall-clock FLOPs). No sweeps.
- **P2:** tiny pilot IsoFLOP (2 budgets × 3 sizes) at debug scale end-to-end; shake out
  logger→collect→fit→plot on real (tiny) data.
- **P3:** full **general** sweep at prod scale; fit Metric A + B; RankMe guard; cross-metric comparison.
- **P4:** write-up of the general JEPA scaling law.
- **P5+ (stretch, §9):** domain from-scratch sweep + from-scratch-vs-CPT curve, using the general law
  as baseline.
