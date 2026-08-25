# Probe results summary — headline tables only

Concise companion to `docs/PROBE_RESULTS_LEDGER.md` (the full ledger — protocol
detail, root-cause investigations, retracted reads, job IDs). This file keeps
only the **settled/current** numbers per probe. When a ledger entry has been
superseded, only the superseding number appears here.

_Last updated: 2026-08-24._

## Checkpoint legend

| tag | model | what it is |
|---|---|---|
| metaraw / meta1b | ViT-g (1B) | raw Meta V-JEPA 2.1, no surgical CPT |
| metaraw2b / meta2b | ViT-G (2B) | raw Meta V-JEPA 2.1, no surgical CPT |
| e19 / ours1b_e19 | ViT-g (1B) | our surgical CPT (`vitg384_cleandata` e19) |
| fs_e159 / ours2b_e159 | ViT-G (2B) | our surgical CPT (`vitG384_fixedshape` e159) |
| v2_e324 | ViT-G (2B) | our surgical CPT, v2/leak-free lineage |
| prod9M_e29 | ViT-G (2B) | our surgical CPT, prod run (9.22M samples, e29 final, natural completion) |
| SNX / SurgeNetXL | CAFormer | published supervised surgical baseline |

Common settings unless noted: res 384, 16 frames, global batch 16 (frozen probes).

---

## 1. Triplet recognition — IVT mAP (SAR-RARP50 tool×verb×target)

**Frozen, 3-seed:**

| encoder | IVT mAP |
|---|---|
| meta2b (raw 2B) | **29.97 ± 0.56** |
| prod18M_e59 (CPT 2B, 18.43M) | **30.01 ± 0.67** † |
| ours1b_e19 (CPT 1B) | 29.61 ± 0.51 |
| ours2b_e159 (CPT 2B) | 29.53 ± 0.36 |
| meta1b (raw 1B) | 28.54 ± 0.31 |
| SurgeNetXL | 26.03 ± 0.16 |

Verdict: ties at both scales frozen; all V-JEPA ≫ SurgeNetXL.

† prod18M_e59 added 2026-08-18 from the TPN=4 re-probe (job 8762936), **final
ep25/25, 3 seeds: 30.34 / 29.23 / 30.45**. Supersedes the void 21.27/19.75 TPN=12
pair — a **+9.50** correction. Seed spread (0.67) is normal for this probe
(0.24–0.90), unlike the ~10-pt swings the confound produced. Note it **ties** the
band rather than beating it (meta2b 29.97, ours1b_e19 29.61, ours2b_e159 29.53):
2× the pretraining budget buys no frozen-Triplet gain, but neither does it cost
anything — the apparent regression was entirely the probe launcher.

**Fine-tune (last-4 blocks unfrozen), 3-seed:**

| | raw 1B | CPT 1B | raw 2B | CPT 2B (e159) |
|---|---|---|---|---|
| IVT mAP | 33.35 ± 0.25 | 35.49 ± 0.72 | 34.89 ± 0.46 | 36.09 ± 0.25 |
| Δ (CPT−raw) | | **+2.14** | | **+1.20** |

FT lifts every encoder +4–7 mAP over frozen and is where CPT's edge becomes visible.

**★★★ Production-run FT arms (added 2026-08-18, job 8763223, 3-seed, same protocol):**

| encoder | samples | IVT mAP (FT) | vs raw 2B |
|---|---|---:|---:|
| **prod18M_e59** | 18.43M | **37.17 ± 0.17** | **+2.28** |
| prod9M_e29 | 9.22M | 35.92 ± 0.70 | +1.03 |
| CPT 2B e159 (prior best) | — | 36.09 ± 0.25 | +1.20 |
| raw 2B (meta2b) | 0 | 34.89 ± 0.46 | — |

**prod18M_e59 is the best FT number in the project** — it clears the prior CPT-2B
best (e159) by +1.08 and raw Meta by +2.28. Crucially, **prod18M beats prod9M by
+1.25 (Welch t = 3.00, n=3/arm)**: doubling surgical pretraining 9.22M→18.43M
samples produces a real, measurable downstream gain. prod18M also has the tightest
seed spread of any FT arm (±0.17).

This is the monotonic-improvement-with-more-surgery-data signal that the frozen
probes could not resolve (all frozen arms sit in a saturated 28.5–30.0 band). It
also directly reverses the retracted "2× budget bought nothing / made Triplet worse"
claim, which was an artifact of the TPN=12 probe launcher (see § below).

**★★★ Unfreeze-depth sweep (added 2026-08-24, `prod37M_e199`, `token_aggregation` head,
3 seeds/arm) — CURRENT BEST TRIPLET RESULT:**

| unfreeze depth | IVT mAP (mean±std) | ensemble | Δ vs last-4 (ensemble) |
|---|---:|---:|---:|
| last-4 (baseline) | 37.68 ± 0.23 | 38.69 | — |
| **last-8** | **39.92 ± 0.91** | **41.24** | **+2.55** |
| full (48/48 blocks) | in flight, interrupted by 2026-08-24 maintenance | — | — |

Last-8 is a clean win, well outside both arms' seed spreads, and tracks the same
direction as the GraSP precedent (Leonardo's fs_e159: last-4=83.66→last-8=85.34 mAP).
A separate test found A1's frozen attentive-pooling win (37.06 ensemble, above) does
**not** stack with FT — combined FT+A1-head lands at 38.03, slightly *below* FT with
the original head — the two levers interfere rather than compose. Full-unfreeze
(48 blocks) requires `batch_size=2` (OOMs at bs=4) and was mid-run when interrupted;
2 of 3 seeds had 2 clean epochs (val_micro_f1 ~82.1-82.6%) before a system maintenance
window paused the chain — result not yet in.

**Seed-ensemble (mean-of-probabilities across 3 seeds):**

| | best-of-seed Δ (CPT−raw) | ensemble Δ (CPT−raw) |
|---|---|---|
| frozen 1B | +1.44 | +0.60 |
| frozen 2B | −0.55 | +0.02 |
| FT 1B | +2.83 | +2.30 |
| FT 2B | +0.81 | +1.17 |

Ensembling shrinks the frozen deltas toward a tie (closes out seed-luck); FT deltas hold.

**★ Training-budget control (same arm, only epochs differ; job 8751525/8751781):**

| checkpoint | budget | IVT mAP |
|---|---|---|
| samp_t050 e79 | 80ep | 28.31 ± 0.06 (≈ raw meta1b, t=−1.26) |
| samp_t050 e159 | 160ep-equiv | 29.12 ± 0.56 |
| samp_t050 e239 | 240ep | **30.15 ± 0.30** |

Budget delta e79→e239 = **+1.84 IVT, t=10.4** — the largest confirmed single-lever
effect in the project. At ≤80 epochs, frozen CPT shows nothing over raw Meta; the
whole margin opens up between epoch 80 and 240, roughly linearly (no saturation
detected). Every composition/objective ablation in §5 of the full ledger was
screened inside this pre-onset window — treat those nulls as budget-scoped.

---

## 2. SAR-RARP50 action segmentation — F1@10 (official TEST split)

**Frozen, 3-seed, identical protocol both scales:**

| scale | encoder | F1@10 |
|---|---|---|
| 2B | **CPT (prod9M_e29)** | **88.06 ± 0.31** |
| 2B | CPT (fs_e159) | 86.93 ± 0.38 |
| 2B | raw (meta2b) | 85.88 ± 0.90 |
| 1B | CPT (e19) | 86.27 ± 0.39 |
| 1B | raw (meta1b) | 85.75 ± 0.45 |

Verdict: **prod9M_e29 clears every other frozen 2B arm** — +1.13 over fs_e159 (previous 2B CPT
best), well outside fs_e159's own ±0.38 spread. fs_e159/meta2b/e19/meta1b remain a **statistical
tie among themselves** (85.6–87.0, backbone-saturated), all clearing published SOTA 84.10 by
≥1.5. prod9M_e29 is the first frozen-probe checkpoint in this ledger to separate cleanly from
that saturated band — worth checking whether it's the larger/longer prod-run budget (9.22M
samples, openh/pe_video dropped) rather than scale driving this, since fs_e159/e19 are smaller
or differently-composed CPT runs.

**Fine-tune (last-4 blocks), 3-seed:**

| | raw 1B | CPT 1B | raw 2B | CPT 2B |
|---|---|---|---|---|
| F1@10 | 88.58 ± 0.22 | 89.98 ± 0.05 | 88.32 ± 0.36 | 90.25 ± 0.50 |
| Δ (CPT−raw) | | **+1.40** | | **+1.93** |

Same pattern as triplet: FT opens a real CPT gap that frozen probing cannot see.

**`prod37M_e199` frozen + FT (added 2026-08-24, 2026-08-21/22 jobs):**

| | frozen (3-seed) | FT-last4 (1 seed, full 25-epoch ceiling sweep) |
|---|---:|---:|
| F1@10 | 86.95 / 87.85 / 87.50 → **87.43 ± 0.38** | **90.81** (true ceiling @ ep20) |

Frozen ties the backbone-saturated band above. FT gain is **+3.38 F1@10**, consistent
with the general FT-beats-frozen pattern. The FT ceiling sweep rescored every epoch
individually on TEST — the `val_macro_f1`-selected `best.pt` (ep14, 89.55) undershoots
the true per-epoch max by 1.26 points, the same selection-metric gap documented
elsewhere in this project; always cite the per-epoch max, not `best.pt`, for this
probe's FT ceiling.

**Ensembling (mean-of-3-softmax before argmax, no test-set peeking):**

| axis | scale | baseline (head0) | ensemble | Δ CPT−raw shifts |
|---|---|---|---|---|
| head-ensemble (3 LR heads) | 2B | 1.33σ | 1.55σ | modest gain |
| head-ensemble | 1B | 1.05σ | **2.80σ** | strongest signal in the ledger |
| seed-ensemble (3 independent seeds) | 2B | +1.06 | **+1.69** | grows |
| seed-ensemble | 1B | +0.52 | +0.44 | shrinks slightly |

Ensembling axis (head vs seed) does not predict a consistent direction for the
CPT-vs-raw delta — treat as a free variance-reduction lever, not a guaranteed CPT win.

**prod9M_e29 seed-ensemble (self-contained, not an existing arm's axis):**

| scale | arm | baseline (3-seed mean) | seed-ensemble |
|---|---|---:|---:|
| 2B | CPT (prod9M_e29) | 88.06 | **89.76** (Δ +1.70) |

Same free lever as the other arms — ensembling grows prod9M_e29's already-leading number
further, widening its margin over fs_e159's own ensemble (89.04).

---

## 2b. SAR-RARP50 dense segmentation — val mIoU (separate spatial probe)

Single seed each, 60-epoch runs:

| scale | CPT | raw | Δ |
|---|---|---|---|
| 1B | 65.31 | 63.78 | +1.53 |
| 2B | 65.52 | 63.60 | +1.92 |

Small, consistent CPT edge at both scales — not seed-replicated.

---

## 3. GraSP phase recognition — mAP (official TEST split, 11-class)

**★ Official-build result (fixed `build_grasp_ctx_official.py`, supersedes all
prior ctx3-builder numbers, jobs 8751526/8751981/8753458, 2026-08-13):**

| encoder | best-head mAP | ensemble-of-3 mAP |
|---|---:|---:|
| raw meta2b | 75.65 | 79.77 |
| **CPT v2_e324 (2B)** | **79.68** | **82.00** |
| Δ (CPT−raw) | **+4.03** | +2.23 |
| TAPIS SOTA (end-to-end) | 76.72 | — |

Both CPT numbers and the raw-ensemble clear published SOTA. This is the first
unconfounded probe where CPT clearly separates from raw Meta (vs. ties on
triplet/SAR frozen). The old ctx3-builder table (69.31 CPT / 68.04 raw) is
**superseded** — that builder had 6.6% wrong labels, a burned-in-UI shortcut, and
non-keyframe-centered windows; fixing it lifted both encoders ~7–11 mAP.

Caveats: single seed each; `best.pt` selected on val_macro_f1 (a proxy); noise
floor not yet re-measured on this build (was ~3 mAP on the old build).

**`prod37M_e199`, 3 seeds — frozen + FT-last4 (added 2026-08-24):**

| arm | best-head mAP | ensemble mAP |
|---|---:|---:|
| frozen (block 47, default head) | **82.64** | **84.84** |
| FT-last4 | 83.41 ± 0.63 (s0=82.95/s1=84.27/s2=83.00) | 84.17 ± 0.39 |

**GraSP-FT is essentially flat vs. frozen** (+0.77 best-head, −0.67 ensemble) — unlike
Triplet-FT (+6.33 IVT ensemble) and SAR-FT (+3.38 F1@10), which both show clean FT
gains on this same checkpoint. Root cause under investigation: our FT *training*
config was found to still be on the pre-fix `ctx3` CSV builder (wrong geometry too —
`num_segments=3` vs the corrected `4`, `num_views_per_segment=2` vs `1`) while every
other GraSP config in the repo already migrated to the corrected `grasp_ctx_official`
builder — so this FT training ran on stale/mismatched data relative to what it was
then scored against. Leonardo's own fs_e159 lineage (kinetics-only, no openh) gets
83.66 (last-4) → 85.34 (last-8) on the corrected builder — a real, uninvestigated
possibility that our flat FT result is a config bug, not a checkpoint property.
**Not yet resolved** — re-run pending on the corrected geometry/CSVs before treating
prod37M_e199's GraSP-FT as settled.

**GraSP Step (21-class) probe, `prod37M_e199` — head-width A/B (added 2026-08-24):**

3 seeds/arm, frozen probe, varying ASFormer head width and dropout:

| arm | best-head mAP | ensemble mAP |
|---|---:|---:|
| d1664 (default width, no dropout) | 54.04 ± 0.50 | 57.12 ± 0.25 |
| **d768 (narrow head), no dropout — SHIPPED DEFAULT** | **54.46 ± 0.63** | **57.85 ± 0.55** |
| d768 (narrow head) + dropout 0.3 | 54.20 ± 0.95 | 56.32 ± 0.43 |

Narrowing the head (1664→768) wins cleanly on mAP at 3.7× lower compute (backward
dominates iter time at full width: 94-97.5% of it, per profiling). **Adding dropout
0.3 on top looked like a further win on val_macro_f1** (the training-loop selection
metric: 53.57% vs 52.89% vs 52.02% baseline) **but is actually the WORST arm on mAP**
(the metric of record, matching TAPIS's paper metric) — dropout smooths the confident
predictions AP's ranking depends on while helping macro-F1's threshold-based
precision/recall balance. **Do not select head configuration from val_macro_f1 alone
on this probe** — same selection-metric trap documented elsewhere for epoch/checkpoint
choice, now shown to also apply to architecture choice. d768-no-dropout beats the
TAPIS canonical SOTA (52.01 mAP) by +5.84 ensemble and is the new shipped default
(`configs/heads/grasp/official_ctx16/grasp_off16_step_prod37M_e199_probe.yaml`).
**Untested whether this transfers to GraSP Phase** — in progress as of this writing
(early single-seed reads land close to the Phase default-head baseline: one seed
above on best-head, one below, one seed pending — too early to call).

---

## 4. SARAS-ESAD double-head (presence + class-conditioned box) — detection AP_mean

Scored on the paper's full denominator (`gt_full=11207`, fixed 2026-08-05 —
earlier numbers covering only 32% of GT are void and not reproduced here).
Paper best submission: **AP_mean = 0.1928**.

**Frozen vs fine-tune (last-4 blocks), 3-seed:**

| encoder | frozen | fine-tune |
|---|---|---|
| CPT 2B (prod37M_e54, mid-run e54/120) | 0.1571 ± 0.0208 | — |
| CPT 2B (prod9M_e29) | 0.1420 ± 0.0075 | — |
| CPT 2B (e159) | 0.1483 ± 0.0149 | **0.2005 ± 0.0191** |
| CPT 1B (e19) | 0.1302 ± 0.0196 | 0.1832 ± 0.0088 |
| raw 2B (meta2b) | 0.1271 ± 0.0264 | 0.1936 ± 0.0727 |
| raw 1B (meta1b) | 0.1441 ± 0.0462 | 0.1855 ± 0.0194 |
| SurgeNetXL (frozen) | 0.1162 ± 0.0012 | — |
| LemonFM (frozen) | 0.1537 ± 0.0359 | — |
| EndoViT (frozen) | 0.0853 ± 0.0020 | — |

Fine-tuning clears frozen for every V-JEPA checkpoint (CPT 2B beats the paper's
SOTA outright: 0.2005 vs 0.1928); no frozen CPT-vs-raw delta clears its own seed
spread at either scale — read frozen as a tie, FT as the only real gain. prod9M_e29's
frozen mean (0.1420, per-seed 0.1355/0.1502/0.1403) sits inside fs_e159's own spread
(0.1483 ± 0.0149) — **a tie with the existing 2B CPT champion, not a win** (unlike SAR,
where prod9M_e29 does separate — see §2). prod37M_e54 (a mid-run e54/120 checkpoint, 2x
prod_18M's budget by schedule, per-seed 0.1367/0.1565/0.1782) nominally moves higher still
(0.1571 ± 0.0208) but its own spread is the widest of the three CPT-2B rows and fully
overlaps both — still no resolved win, see ledger §4j. No FT run for prod9M_e29 or
prod37M_e54 yet. **Uses the same
pre-fix (`union`-mode) presence labeling as every other row in this table** — the
2026-08-17 presence-label defect fix (§ below) has not yet been applied to ANY arm's
manifests, so this comparison is apples-to-apples on the current protocol, not stale
relative to a fixed baseline that doesn't exist yet for any checkpoint.

**★★ SUPERSEDED IN PART BY §4i (2026-08-19): the external arms now HAVE fine-tuned
numbers.** The frozen-only comparison below is no longer the whole picture. Complete
FT table, n=3 everywhere, same population (`gt_full=11207`, `frames=5903`):

| arm | frozen | FT (last-4) | FT gain |
|---|---|---|---|
| **e159 (CPT 2B)** | 0.1483 | **0.2005 ± 0.0098** | +0.0521 |
| meta2b (raw 2B) | 0.1271 | 0.1936 ± 0.0409 | +0.0665 |
| *paper SOTA* | — | *0.1928* | — |
| meta1b (raw 1B) | 0.1441 | 0.1855 ± 0.0099 | +0.0414 |
| ours1b_e19 (CPT 1B) | 0.1302 | 0.1832 ± 0.0048 | +0.0530 |
| **LemonFM** | 0.1538 | **0.1585 ± 0.0048** | **+0.0047 (t=0.42, NULL)** |
| **SurgeNetXL** | 0.1162 | **0.1257 ± 0.0039** | +0.0094 |

**All four V-JEPA arms beat both externals fine-tuned**, and e159/meta2b clear the paper's
SOTA. But only `e159 − SurgeNetXL` (+0.0748) clears the ~0.06 noise floor; every
margin over LemonFM (+0.025 to +0.042) sits BELOW it and is directional only — the same
floor that dissolves LemonFM's frozen lead applies here too, and must not be invoked
selectively. **The clean result is the within-backbone gain: LemonFM gains +0.0047
(t=0.42, a null) from fine-tuning while every V-JEPA arm gains +0.041–0.067.** Mechanism
NOT established (baselines differ; ceiling effect is a live alternative). Note SurgeNetXL
*does* gain significantly, so "image backbones don't fine-tune" is too broad.

**Per-threshold + oracle columns (2026-08-19, full breakdown in ledger §4i).** AP_mean above
macro-averages IoU {0.10,0.30,0.50}; per-threshold FT values: e159 0.2981/0.2245/0.0788,
meta2b 0.2828/0.2243/0.0736, meta1b 0.2699/0.2145/0.0721, ours1b_e19 0.2639/0.2128/0.0726,
**LemonFM 0.2921/0.1509/0.0325**, **SurgeNetXL 0.2346/0.1224/0.0199**. LemonFM's near-null
aggregate gain hides a **trade** (AP10 up, AP30/AP50 down), unlike SurgeNetXL and every
V-JEPA arm, which gain at all three thresholds together. Oracle columns (presence mAP /
mean IoU / [email protected], n=3 unless noted): e159 FT 0.3158/0.4536/0.4497; **LemonFM FT
0.3118/0.3488/0.2242 (n=3)**; **SurgeNetXL FT 0.2711/0.3318/0.2082**.
e159 clearly leads on localization (IoU/map50); presence mAP is closer and not yet
noise-floor-tested for this metric family, so treat as directional only.

**★ Read the LemonFM row correctly (added 2026-08-17, see full ledger §4g).** LemonFM's
0.1537 vs our 0.1483 is **NOT a loss — it is a tie**: Welch t = **+0.46**, and LemonFM's
seed range [0.1320, 0.1679] overlaps every V-JEPA arm. A power analysis (pooled σ=0.0146)
shows resolving that 0.005 gap would need **~133 seeds/arm**, so the frozen column of this
probe **cannot rank encoders at any feasible cost** — report it as a tie band (0.127–0.154),
never as an ordering. The comparison that IS resolved: **e159 FT 0.2005 vs LemonFM frozen
0.1538 = +0.047 at t=3.76** (n=2 would suffice), which also clears the paper's 0.1928 —
but note **no external arm has an FT number**, so state that asymmetry when citing it.

Three cross-arm asymmetries were also found, all favoring the image arms: a **presence-label
defect affecting ONLY the V-JEPA arms** (8.1% train / **17.1% val** mislabeled rows; 0% at
tubelet=1 by construction — **fixed 2026-08-17**, reruns pending), **2× supervised token-rows
per step** for image arms, and **unnormalized head width** (LemonFM's head is 19% larger than
the 1B V-JEPA arms'; SNX/EndoViT's are 7–9× *smaller*, which undercuts the "all externals sit
below V-JEPA" reading). A LemonFM **leakage hypothesis was tested directly and REFUTED**
(min Hamming 6, zero matches; ESAD has no video form online) — do not make that claim.
Separately, **SurgeNetXL has documented ESAD in its pretraining and scores lowest.**

**★★★ `prod37M_e199` frozen + FT + augmentation — CURRENT BEST ESAD RESULT (added
2026-08-24).** `train_esad_unfreeze.py` never actually passed augmentation into the
FT dataset despite the dataset class supporting it — every FT run on record before
this fix trained on identical frames every epoch. One-line fix, 3 seeds, one
population (`gt_full=11207`, `gt_cov=10619`, `frames=5903`):

| arm | AP_mean | AP50 (strictest) | presence mAP† | box mAP@50† |
|---|---:|---:|---:|---:|
| frozen | 0.1793 ± 0.010 | 0.0801 | 0.2507 | 0.4810 |
| FT-last4 | 0.2035 ± 0.016 | 0.0680 (↓) | 0.3159 | 0.4209 (↓) |
| **FT-last4 + augmentation** | **0.2319 ± 0.024** | **0.1127** | **0.3248** | **0.5101** |

**Augmentation reverses the FT localisation penalty.** Bare FT trades box quality for
presence detection (box mAP@50†/AP50 both decline vs frozen); adding augmentation
recovers both — box mAP@50† ends up *above* frozen while keeping the presence gain.
Retracts an earlier "box head is capacity-limited" reading: it was overfitting on
2,468 windows from only 2 training surgeries, and the regularizer that fixes it had
simply never been wired in (`train_bce` at ep19 is 3.4× higher with augmentation —
it stops memorizing). **+0.053 over frozen clears this probe's ~0.06 noise bar** — the
only ESAD arm in the whole campaign to do so. This is the best AP_mean on record for
any checkpoint on this probe, ahead of both the paper's best submission (0.1928) and
every prior V-JEPA arm.

**Seed-ensemble, FINE-TUNED arms (added 2026-08-25, jobs 176554/176559).** FT seeds each
tune their own encoder, so they don't share a feature cache — fusion happens in label
space (`score_esad_seed_ensemble_ft.py`). Against the **seed mean**, the comparison that
corresponds to a real choice ("train 3 and fuse" vs "train 1"), the ensemble wins **6/6**
fusions across both arms, +0.0155 average:

| arm | best fusion | seed mean ± σ | **ensemble** | Δ |
|---|---|---:|---:|---:|
| FT-last4 | `wbf_meanconf` | 0.2048 ± 0.0147 | **0.2274** | **+0.0226** |
| FT-last4 + aug | `wbf_meanconf` | 0.2323 ± 0.0249 | **0.2498** | **+0.0175** |

**★ The gain is localisation, not detection.** Split by IoU threshold, every
coordinate-*averaging* fusion improves more at 0.5 than at 0.1 (up to +25.8%), while
`maxpick` — the only fusion that copies one seed's box verbatim — is the only one that
goes **negative** at 0.5 (−1.1%). It cannot improve a box it merely selects. Use
`wbf_meanconf`. Against *best-of-3* the ensemble is 2/6, but best-of-3 is chosen on the
test set and is not an achievable policy; the seed spreads (0.030–0.048) exceed every
effect here. Ledger §4b-ens.

**Is the frozen "+aug" gain augmentation, or just more windows? — SEED 0 ONLY, do not cite
(added 2026-08-25, jobs 176587/176588).** `--train-cache` is `action="append"`, so the frozen
"+aug" arm passed clean **+** augmented and thereby **doubled** windows/epoch (2,468 → 4,936),
confounding augmentation with 2× the gradient steps. `frozen_augonly` is the missing control:
the augmented cache **alone**, at the baseline's 2,468 windows. Seed 0 of 3, probe-internal
metrics only (the deciding full-denominator detection AP is scored after all three seeds and
does not exist yet):

| metric† | baseline s0 | `augonly` s0 | `frozen_aug` s0 |
|---|---:|---:|---:|
| macro mAP | 0.3187 | **0.3506** | 0.3345 |
| box mAP@50 | 0.5087 | 0.5573 | **0.5641** |

Leaning **"it's the augmentation"** — `augonly` matches or beats `frozen_aug` on mAP using half
the windows, and beats the baseline on all four columns. But the baseline's own 3-seed spread
is 0.0112 and one seed cannot carry a 0.03 gap. Note the two columns rank the arms *oppositely*
(the same presence-vs-localisation split as FT) — verdict deferred to the 3-seed detection AP.
Ledger §4b-frozctl.

**Seed-ensemble (frozen only):**

| scale | arm | baseline | ensemble |
|---|---|---|---|
| 2B | CPT (prod9M_e29) | 0.1420 | 0.1508 |
| 2B | CPT (e159) | 0.1483 | 0.1569 |
| 2B | raw (meta2b) | 0.1271 | 0.1362 |
| 1B | CPT (e19) | 0.1302 | 0.1348 |
| 1B | raw (meta1b) | 0.1441 | 0.1658 |

Δ(CPT−raw) stays ~+0.02 at 2B under ensembling but **worsens** for CPT at 1B
(−0.014 → −0.031) — ensembling favors whichever arm has the noisiest seeds
(here, raw meta1b), not CPT specifically.

---

## 5. SSv2 forgetting — frozen val_acc, 174-class action recognition, 10 epochs

Tests catastrophic forgetting of general (non-surgical) video capability. Invisible
to every other probe in this doc — none of them can detect loss of a capability
they don't test for.

| checkpoint | corpus | ema | val_acc @ ep10 | Δ vs raw meta2b |
|---|---|---|---:|---:|
| raw meta2b | — | — | 60.20% | — |
| CPT 2B (fs_e159, old lineage) | 15 src, **includes kinetics400** | 0.99925, 9990 steps | 59.43% | −0.77 |
| **CPT 2B (prod9M_e29, 9.22M samples)** | 14 src, **no kinetics400** | 0.988, 1500 steps | **52.57%** | **−7.63** |
| **CPT 2B (prod18M_e59, 18.43M samples)** | 14 src, **no kinetics400** | 0.988, 3000 steps | **49.35%** | **−10.85** |

**★★★ RESOLVED 2026-08-19 (supersedes an earlier "small, −0.84" reading, and corrects
an intermediate draft of this entry that wrongly stated fs_e159 had no rehearsal — it
does, verified against its config).** fs_e159's corpus **includes kinetics400**;
production (prod9M/prod18M) dropped it. The −0.84 number was measured on a run that
already had general-video rehearsal, so it was never a fair "no-rehearsal" baseline.
On the checkpoints that actually shipped (no rehearsal), forgetting is ~10× larger and
INCREASES with more CPT budget — the opposite direction of the budget story on
triplet/SAR/GraSP. **Primary, well-supported fix: restore kinetics400 (and consider
pe_video) to the corpus** — both already staged. A secondary, unconfirmed contributor
is EMA speed (fs_e159 also used a much slower EMA, 0.99925 vs 0.988) — see full
ledger §7 for why this is worth a separate, corpus-held-fixed test before assuming
rehearsal alone fully explains the gap.

---

## Cross-cutting findings

- **Frozen readouts understate CPT everywhere measured; fine-tuning (last-4
  blocks) reveals it** — +1.2 to +7 points depending on probe/scale.
- **Training budget is the dominant lever found so far**: at ≤80 epochs, frozen
  CPT ≈ raw Meta on triplet; nearly all of the CPT margin appears between epoch
  80 and 240 (§1). Any composition/objective ablation run at ≤80 epochs should be
  read as budget-scoped, not conclusive.
- **Ensembling (head- or seed-diversity) is a free lever but not a reliable CPT
  booster** — it shrinks noise on whichever arm disagrees most across seeds/heads,
  which is sometimes CPT, sometimes raw.
- **Single-seed cached probes (GraSP, ESAD) carry a ~3 mAP / ~0.03 AP noise
  floor** — deltas below that need 3-seed replication before citing.
- **Eval-harness bugs can dominate model differences**: the GraSP ctx3 clip
  builder (label drift, UI-shortcut leakage, non-keyframe windows) suppressed
  both encoders by ~7–11 mAP; fixing the harness moved the headline number more
  than 3× the current best pretraining lever (budget, +1.84 IVT).
- **Selection-metric traps recur across levels**: `val_macro_f1`/`best.pt` selection
  undershoots the true per-epoch F1@10/mAP ceiling on SAR (by up to 1.3pts) and, most
  strikingly, **inverts the ranking on GraSP head-architecture choice** — dropout 0.3
  looked like a win on val_macro_f1 but was the worst arm on mAP. Never select
  epochs, checkpoints, or architectures from a training-loop proxy metric without
  checking the actual target metric.
- **Unfreezing more of the encoder keeps paying off past last-4**: Triplet last-8
  beats last-4 by +2.55 ensemble IVT (2026-08-24), mirroring Leonardo's GraSP
  last-4→last-8 precedent (83.66→85.34 mAP) — worth testing last-8 (and full-48,
  memory permitting) on every probe with an established last-4 FT result, not just
  assuming last-4 is the ceiling.
- **`mpiexec`/PALS can mask real crashes as `rc=0`** (confirmed twice independently:
  the GraSP export `WORLD_SIZE` bug and a Triplet full-unfreeze OOM) — grep the
  actual log for `Traceback` before trusting any run's success, and bake that check
  into launcher scripts rather than trusting the exit code.
- **Missing augmentation silently caps FT ceilings**: the ESAD FT trainer never
  wired in the dataset's own augmentation support, and adding it produced the
  largest single lever in that probe's campaign (+0.028 over bare FT, the only arm
  to clear the noise bar) — worth auditing every other FT trainer for the same gap.

## Open (unresolved as of 2026-08-24)

- **GraSP-FT on `prod37M_e199` looks flat vs. frozen (+0.77/-0.67) but the training
  config was found to still be on the stale pre-fix CSV builder/geometry** — every
  other GraSP config in the repo migrated to the corrected builder, only the FT
  training config didn't. Re-run pending on corrected geometry before concluding
  anything about whether this checkpoint underperforms Leonardo's fs_e159 lineage
  on GraSP-FT (83.66/85.34 last-4/last-8, on kinetics-only-no-openh data).
- ~~GraSP Step's winning head architecture (d768, no dropout) transfer-to-Phase
  test is in progress~~ — **RESOLVED 2026-08-24, NULL.** 3-seed mAP: best-head
  82.58±1.25 (vs. frozen default-head baseline 82.64, a wash), ensemble
  83.25±0.73 (vs. 84.84, a real −1.59 loss, consistent direction all 3 seeds).
  The narrow-head win does NOT transfer from Step to Phase — keep d1664_dr00
  (default width) for Phase, d768_dr00 stays Step-only. See ledger §11c.
- **Triplet full-unfreeze (48/48 blocks) result is pending** — training was
  interrupted by a system maintenance window with 2 of 3 seeds at epoch 2/25;
  the 3-way depth verdict (does IVT keep climbing past last-8, plateau, or reverse)
  isn't settled yet.

## Open (unresolved as of 2026-08-17)

- GraSP official-build result is single-seed for both raw and CPT — needs seed
  replication before the +4.03 best-head delta is citable as settled.
- ESAD: only e159 has 3-seed FT; the other three checkpoints (e19, meta1b, meta2b)
  need the same FT run to make the CPT-vs-raw FT comparison complete.
- **★★★ RETRACTED (2026-08-18): the prod_9M/prod_18M Triplet "mid-run collapse"
  is a PROBE-LAUNCHER CONFOUND, not a checkpoint property. Every claim in the
  entry below is void; see the ledger's "Triplet mid-run dip" entry for the
  correction.** The healthy and collapsed checkpoints were probed by two
  different launchers with different hardcoded world sizes:
  `triplet_seed_sweep.sh` (`TPN:-4` → world_size 4, **222 iters/epoch**, wrote
  `*_s<seed>` dirs) ran e4/e9/e29; `triplet_diag_chain.sh` (`WORLD_SIZE=12`,
  `mpiexec -n 12` hardcoded, no override → world_size 12, **74 iters/epoch** at
  the same unscaled LR, wrote unsuffixed dirs) ran e14/e19/e24 and prod_18M.
  The split is perfect — no checkpoint was ever measured under both. Note
  `tasks_per_node: 12` in the YAML is INERT; the launcher's `-n` wins.
  **Controlled proof on a FROZEN SurgeNetXL encoder** (weights cannot change),
  3 seeds each: TPN=4 → 25.86/26.24/26.00 (mean **26.03**); TPN=12 →
  21.95/22.48/21.74 (mean **22.06**). **−3.97 IVT from protocol alone**, seed
  spread ~0.3. The "clipper-class signature" reproduces on that frozen encoder
  too (clipper AP 66.71%→40.80%, "clip" verb 66.61%→39.07%), so it is a
  low-support-class sensitivity to head undertraining, not a representation
  deficit. This also explains why 2 seeds agreed (seeds don't change the
  launcher), why GraSP showed no dip (cached path, unaffected), and why global
  feature statistics were normal (the encoder was always fine).
  **★★★ CONFIRMED ON IVT (job 8762936, 3 seeds, converged):** re-probing
  prod_18M **e59** at TPN=4 gives **IVT 30.30 / 29.12 / 30.09 (mean ~29.8)**
  vs ~20.5 at TPN=12 — **+9.4 points**. Stable across epochs (ep17 vs ep19
  differ by ≤0.16/seed) and the seed spread (~0.6) is back in this probe's
  normal 0.24–0.90 range. The rare-class signature is gone: clipper AP
  61.23%→**81.98%**, "clip" verb 31.62%→**81.61%**, both at/above the healthy
  baseline. e59 lands at the **top of the all-encoder band** (raw meta2b 29.97,
  CPT arms 29.5–29.6) — a completely normal checkpoint.
  **So "doubling the budget made Triplet worse" is FALSE and retracted.** The
  earlier −3.97 frozen-SNX estimate UNDERSTATED the effect for V-JEPA
  checkpoints; there is no unexplained residual for e59. (The GraSP half of the
  old claim — a genuine 71.05/72.82 vs 70.98/73.52 tie — used a different code
  path and still stands.)
  Still in flight: prod9M **e14** and **v2_e199** at TPN=4 (chain 8763075→).
  Until they land, treat prod9M e19/e24 and the rest of the v2 lineage as
  **unmeasured**, not collapsed.
  The §1 headline arms are unaffected (all TPN=4, train_loss ~1.09-1.17).

- <details><summary>VOID — original entry, retained for provenance</summary>

  **★★★ PARTIALLY RESOLVED (2026-08-17, corrected): the prod_9M/prod_18M Triplet
  mid-run collapse is REAL and REPRODUCIBLE ACROSS SEEDS, not probe-training
  noise — but the underlying mechanism is still open.** prod_9M's OWN e19
  (never before probed) showed the identical collapse to prod_18M's e19 (IVT
  19.84 vs 19.06), and a dense trajectory (e14/e19/e24, IVT 18.0-20.0) proved
  this is a real, sustained, ~500-step-wide dip that fully recovers by e29
  (28.45) — not a single noisy checkpoint. A first pass concluded this was
  single-seed linear-probe-training noise for rare classes; **a second seed at
  e14 refuted that (seed0=18.02, seed1=21.25 IVT, both firmly collapsed, both
  showing the same clipper-class signature)** — every other Triplet arm ever
  measured in this project has shown 0.24–0.90 point seed std, so a ~10-point
  drop reproducing across 2 seeds is a real property of the checkpoint, not
  noise. Ruled out via direct evidence: sampling/corpus (byte-identical logs,
  both runs), LR/lambda schedule (smooth + monotonic through the dip,
  non-monotonic IVT can't follow monotonic LR), pretraining loss (no anomaly),
  and global representation collapse (rank/cosine-sim/variance/token-norm-
  outliers all normal on exported GraSP-cache features). **GraSP at the same
  e14 checkpoint shows NO dip** (68.13/70.86, right on the smooth e9→e29
  trend) — a different task/probe/label-space on the identical encoder state,
  proving the frozen encoder is not globally degraded. The collapse is
  concentrated in minority classes (clipper 80%→32%↔29%[2 seeds]→78%, its
  "clip" verb 80%→26%↔34%→81%; common classes flat) — **real, reproducible,
  but mechanistically unexplained**: something about this checkpoint state
  specifically degrades rare-instrument discriminability in a way invisible
  to global feature statistics and to GraSP's phase-recognition task.
  **★★★ CONFIRMED (2 seeds): prod_18M's final checkpoint (e59, 18.43M samples)
  is STABLY collapsed on Triplet and did NOT recover the way prod_9M's e29
  did** — seed0=21.27, seed1=19.75 (within 1.5pts, mean ~20.5), both firmly
  collapsed, both showing the same clipper-class signature. **Doubling the
  pretraining budget did not fix Triplet.** GraSP tells the same "no gain"
  story from the other side: prod_18M e59 = 71.05 best-head / 72.82 ensemble,
  a statistical tie with prod_9M e29 (70.98/73.52, both diffs inside the ~3
  mAP noise floor) — **doubling the budget bought no clear GraSP improvement
  either.** Net: the 2x-budget run produced a checkpoint that is flat-to-tied
  on GraSP and measurably worse on Triplet than the 1x-budget run's own final
  checkpoint. Open question this raises: prod_9M's own e29 Triplet "recovery"
  (28.45) has only ever been measured with 1 seed — a 3-seed re-probe of
  prod_9M's e29 is now the natural next step, to rule out the recovery itself
  being a lucky single seed before concluding prod_18M specifically failed to
  reach it. No dataset-weighting/sampling change is implicated (both runs
  verified byte-identical); the mechanism remains unresolved — GraSP proves
  the frozen encoder isn't globally degraded at any collapsed checkpoint, so
  this is a Triplet/rare-class-specific phenomenon whose cause is still
  unknown. Full writeup in the ledger's "Triplet mid-run dip" entry.

  </details>

- A held-out 4th seed on the key triplet/SAR/ESAD ensembling comparisons, to
  confirm the ensemble generalizes rather than just fitting the 3 existing draws.
- `vitG384_prod_9M` GraSP trajectory now has a 4th point: e14 (68.13 best-head /
  70.86 ensemble, 4.61M samples) sits smoothly between e9 (68.42/71.66) and e29
  (70.98/73.52) — confirms the rise is gradual/monotonic on GraSP, no evidence
  of the mid-run dip seen on Triplet at the same sample range (see the resolved
  Triplet-collapse item above). `vitG384_prod_18M` (2x budget, now COMPLETE,
  e59 final @ 18.43M samples) has a GraSP read at e59 (71.05 best-head / 72.82
  ensemble — a tie with prod_9M e29, still valid). ~~only Triplet e59 = 21.27~~
  — **that 21.27 is a VOID TPN=12 number; the correct TPN=4 value is ~29.8
  (3 seeds), see the retraction above.** GraSP e19/e29/e39/e49 configs exist via
  `scripts/gen_prod18m_grasp_triplet_configs.py` and remain unprobed, so whether
  the GraSP rise continues past 9.22M or flattens is still open — but note the
  Triplet side no longer shows a 2x-budget regression to explain.
- **`vitG384_prod_9M`'s final checkpoint (e29) now has SAR + ESAD 3-seed reads
  (2026-08-17), completing its probe suite alongside the existing GraSP/Triplet
  trajectory.** SAR is the headline: **F1@10 88.06 ± 0.31 (ensemble 89.76)** clears
  every other frozen 2B arm, including fs_e159 (86.93 ± 0.38) by +1.13 — the first
  frozen-probe result in this ledger to separate from the previously-saturated
  85.6–87.0 band. ESAD frozen (0.1420 ± 0.0075) ties fs_e159 (0.1483 ± 0.0149) —
  no ESAD win. Worth investigating **why SAR moves and ESAD/triplet don't** for
  this specific checkpoint — candidates: prod_9M's larger/differently-composed
  corpus (openh/pe_video dropped) interacting specifically with SAR's action-
  segmentation objective, or an SAR-specific noise/protocol artifact needing a
  4th seed to rule out before treating +1.13 as settled.
- Data-composition ablations (§5 of the full ledger) were all screened at
  40–80 epochs, which the budget-control result above shows is pre-onset; not
  wrong, but not yet re-tested at a budget where CPT effects are visible.
- **§5 (new) production dropped kinetics400 rehearsal that the old fs_e159 lineage
  had** — this is now a confirmed corpus difference, not a hypothesis, and the
  primary fix (restore kinetics400/pe_video) is well-supported. Whether EMA speed
  ALSO contributes independently of corpus is still untested — a corpus-held-fixed
  EMA A/B is recommended before assuming rehearsal alone closes the gap; see full
  ledger §7.

## Cross-references

- Full ledger (protocol detail, retractions, job IDs): `docs/PROBE_RESULTS_LEDGER.md`
- Fairness/how-to: `docs/PROBING_GUIDE_FOR_LEO.md`
- Throughput/topology: `docs/PROBE_THROUGHPUT_GUIDE.md`
- Triplet deep-dive: `docs/TRIPLET_PARITY_2026-07-09.md`
