# Probe results ledger — surgical V-JEPA 2.1 downstream evals

Living reference for all downstream probe results. **Append, don't rewrite.** Every number
here is copied from an on-disk artifact (JSON / CSV), not from memory. When you add a row,
record the **protocol** (checkpoint selection, #seeds, split) — mismatched protocols are the
#1 source of false conclusions on these tasks (see `docs/PROBING_GUIDE_FOR_LEO.md`).

_Last updated: 2026-07-28._

---

## Checkpoint legend

| tag | model | what it is |
|---|---|---|
| **metaraw** / meta1b | ViT-g (1B), `vjepa2_1_vitg_384.pt` | raw Meta V-JEPA 2.1, **no** surgical CPT |
| **metaraw2b** / meta2b | ViT-G (2B), `vjepa2_1_vitG_384.pt` | raw Meta V-JEPA 2.1, **no** surgical CPT |
| **e19** / ours1b_e19 | ViT-g (1B) | our surgical CPT, `vitg384_cleandata` epoch 19 |
| **fs_e159** / ours2b_e159 | ViT-G (2B) | our surgical CPT, `vitG384_fixedshape` epoch 159 |
| fs_e{N} | ViT-G (2B) | our 2B CPT trajectory, epoch N (e39…e214 available) |
| **SNX** / SurgeNetXL | CAFormer (supervised) | published supervised surgical baseline |

Common probe settings unless noted: res 384, 16 frames, `token_pool: topk_mean` k=8 (triplet)
/ asformer head (SAR), global batch 16, weighted-BCE (triplet) / CE (SAR).

---

## 1. Triplet recognition — IVT mAP (tool5 × verb6 × target12)

3 seeds, global batch 16, 25 epochs, topk_mean k=8, live-encoder probe. IVT = macro-AP over
278 supported triplets (product-of-per-task-sigmoid proxy). Scored with
`scripts/compute_triplet_map.py`; aggregated by `scripts/aggregate_triplet_seeds.py`.

### Aurora (XPU) — 3-seed mean ± std

| encoder | IVT | mean(t/v/t) | tool | verb | target |
|---|---|---|---|---|---|
| ours2b_e159 (CPT 2B) | 29.53 ± 0.36 | 73.48 | 90.63 | 77.92 | 51.89 |
| meta2b (raw 2B) | **29.97 ± 0.56** | 73.02 | 91.17 | 77.22 | 50.66 |
| ours1b_e19 (CPT 1B) | 29.61 ± 0.51 | 72.04 | 89.02 | 75.28 | 51.81 |
| meta1b (raw 1B) | 28.54 ± 0.31 | 71.39 | 88.91 | 76.23 | 49.04 |
| SurgeNetXL | 26.03 ± 0.16 | 70.62 | 88.91 | 73.22 | 49.75 |

### ★ 1B data-ablation e8-equiv SCREEN — triplet IVT (2026-07-30, autonomous driver)

Screening the 1B data-composition arms at **e39 = 40 epochs trained = e8-equivalent** (2000
iters, 768K clips; ipe=50 recast). 3-seed, gb16, TPN=4. **These are HALF-budget screens** —
compare against `robonly@e8 = 28.31 ± 0.51` (the robotic-only 1B arm at the same budget) and
the raw `meta1b 28.54`, NOT the full-budget e19 (29.61). Fired/scored automatically by
`scripts/autonomous_ablation_probe.sh`.

| arm | e8-equiv IVT | seeds | read |
|---|---|---|---|
| robonly (ipe250 e8) | 28.31 ± 0.51 | 27.73/28.69/28.51 | robotic-only ceiling; ≈ raw meta1b |
| **lemonout** (e39) | **28.41 ± 1.40** | 27.59/27.61/30.02 | full − lemon; ≈ raw meta1b / robonly |
| **gynout** (e39) | **28.69 ± 0.49** | 28.47/28.34/29.25 | full − lap-gyn (H2 control); ≈ raw meta1b, tight σ |
| **croppedopenh** (e39) | **28.38 ± 0.77** | 28.31/27.65/29.19 | full + center-cropped openh; ≈ healthy, NOT poisoned |
| **lapout** (e39) | **28.61 ± 0.46** | 28.20/28.51/29.11 | full − lap (H2); ≈ gynout/others, tight σ |
| **openhin** (e39) | **29.13 ± 1.48** | 28.13/28.42/30.83 | full + RAW openh (positive control); ≈ band — NOT regressed |
| **★★ openhin** (e79, FULL budget) | **28.52 ± 0.68** | 27.90/28.41/29.24 | full + RAW openh at FULL 1B budget (e16-equiv); ≈ meta1b — STILL not regressed, tight σ. **Scale/budget test: openh poison is SCALE-driven not budget-driven.** |
| **★ full** (e39) | **29.14 ± 1.93** | 27.87/28.18/31.36 | the ANCHOR (R+L+G+M, 15 sources); ≈ band (median 28.18 ≈ meta1b, mean pulled by s3=31.36) |

**e8-equiv picture (6 of 6 arms scored — COMPLETE — ALL in the HEALTHY band ≈ meta1b 28.54):** robonly 28.31,
croppedopenh 28.38, lemonout 28.41, lapout 28.61, gynout 28.69, openhin 29.13, full 29.14 — every arm lands
within noise of raw meta1b (spread 28.3–29.1, σ all overlapping). **No data composition change — group
removal, lemon, lap, gyn, or raw/cropped openh, nor the full 15-source anchor itself — moves the robotic
triplet needle at half-budget.** The anchor `full@e39 = 29.14` closes the study: every ablation sits within
±0.6 of it, so no group is a net help or harm at 1B/e8-equiv.

**★ H2 ANSWER (lapout 28.61 vs gynout 28.69, both ≈ band): laparoscopic data does NOT hurt the
robotic triplet eval at e8-equiv.** H2 (does lap dilute robotic performance) = REJECTED. lapout ≈
gynout ≈ lemonout ≈ robonly ≈ openhin — no modality-specific effect at this budget.

**★ openh mechanism — ALREADY SETTLED AT 2B; this 1B screen establishes the SCALE/BUDGET THRESHOLD
(openhin 29.13, croppedopenh 28.38):** the openh poison was root-caused at 2B/full-budget by a clean
3-seed leave-one-out (`surgonly_fresh` 21.99 WITH openh vs `noopenh_fresh` 28.92 WITHOUT — see the
"v2 triplet REGRESSION root-caused to openh" section below and [[v2-triplet-regression-diagnosis]],
[[openh-overlay-root-cause]]). openh is a proven −6.9-IVT poison at 2B and is ALREADY DROPPED from
production. What this 1B/e8 screen adds is a BOUNDARY on that poison: the raw-openh positive control
does NOT regress at 1B/e8-equiv — openhin (full + RAW openh) sits at the TOP of the band (~29), not
~22. So **the openh shortcut only bites at 2B and/or full budget; at 1B/half-budget the model does not
lock onto the overlay.** This is consistent with the capacity/budget hypothesis (more params + more
training → more room and time to exploit the fixed-overlay shortcut). The overlay-vs-content question
(croppedopenh) is MOOT *here* because there is no 1B regression to dissect — it can only be adjudicated
at 2B, where the poison manifests. Note openhin's wider σ (±1.48, seed s2=30.83) — 3 seeds agree it's
≥28 (healthy), but noisier than the tight arms; an e16 (e79) rescreen would firm it. **CAVEAT: this is
the TRIPLET probe only — openh being neutral-to-poisonous for triplet does NOT establish its value for
other probes (GraSP/segmentation); those need their own openh-in/out reads before a global corpus call.**

**★★ SCALE/BUDGET THRESHOLD RESOLVED (2026-08-02): openhin@e79 (FULL 1B budget) = 28.52 ± 0.68.**
Trained the openhin arm from e39 (e8-equiv) all the way to e79 (full 1B budget, e16-equiv, 2× the training
of the e8 screen) to answer: is the 1B-neutral / 2B-poison divergence SCALE-driven or BUDGET-driven?
**Answer: SCALE-driven.** At full 1B budget, raw-openh openhin = 28.52 ± 0.68 — still square in the clean
band (≈ meta1b 28.54, ≈ openhin@e39 29.13, ≈ full@e39 29.14), and **nowhere near the 2B openh poison
level (~22, the −6.9 regression).** The tight σ (0.68, vs e39's noisy ±1.48) firms the earlier read: doubling
the training budget did NOT unlock the openh overlay shortcut at 1B. **The openh poison is therefore a
LARGE-MODEL (2B) phenomenon, not a training-duration effect** — a 1B model has insufficient capacity to
lock onto the fixed CMR-console overlay even at full budget, whereas 2B does. Practical implication: **1B is
robust to openh contamination; 2B is not.** If 1B is the production scale, openh is safe (though still adds
no value); if scaling to 2B, openh must be dropped (as it already is). This CLOSES the 1B data-ablation
program. Same TRIPLET-only caveat applies. See [[openh-overlay-root-cause]], [[v2-triplet-regression-diagnosis]].

**★ DECISIVE comparison RESOLVED — `full@e39 = 29.14 ± 1.93` (the anchor):** full sits at the TOP of the
band alongside openhin (29.13), confirming that ADDING every group (the 15-source full mix) neither helps
nor hurts vs. any leave-one-out. All six deltas vs full are within noise: lemonout −0.73, croppedopenh
−0.76, robonly −0.83, lapout −0.53, gynout −0.45, openhin −0.01 — every one inside the overlapping σ. **The
Lemon lever (lemonout − full = −0.73) is noise, not a regression.** No composition change moves the needle.

**lemonout 28.41 read:** at e8-equiv, `full − lemon` sits in the **healthy band** (≈ raw meta1b
28.54, robonly@e8 28.31, full 29.14) — removing lap-heavy lemon does NOT hurt (nor help) the robotic
eval at half budget. The Lemon-lever delta (lemonout − full@e39 = −0.73) is within noise. The alarming
"regression" first-read (23.30) was a SCORING ARTIFACT, not real (see below).

**CAVEATS:**
1. **★ First score (23.30 ± 1.65) was WRONG — a premature read.** The driver scored at 09:14
   from partial/early-epoch val_probs dumps while the probe head was still training; re-scored
   from the converged dumps (~10:12) it is **28.41 ± 1.40**. Driver since fixed to require the
   probe job to COMPLETE (job gone from qstat) before scoring. Lesson: the triplet probe writes
   dumps progressively per head-epoch — never score mid-run.
2. **`full` baseline RESOLVED (2026-07-31): full@e39 = 29.14 ± 1.93.** The Lemon-lever delta is
   `lemonout − full = 28.41 − 29.14 = −0.73` (noise). full was the last arm; 6/6 now complete.
3. `e39.pth.tar` for lemonout/full was a snapshot of `latest.pth.tar` (epoch 39) — the numbered e39
   save was skipped by a chain-resubmit landing on the boundary (driver fixed to fire on first
   numbered ckpt ≥ e39). For full specifically, its ~65-min epoch could not complete inside a 1h
   debug-scaling slice (deadlock — [[full-arm-epoch40-deadlock]]); latest.pth.tar (epoch field 39 =
   e8-equiv) was hand-copied to e39.pth.tar (verified: loads clean, 494 enc keys) to close the arm.
4. **Two arms (openhin ±1.48, full ±1.93) carry a wide σ from one high seed each** (openhin s2=30.83,
   full s3=31.36); medians are ≈ meta1b. Point estimates noisier than the tight arms — an e16 (e79)
   rescreen would firm them, but every arm's median lands in-band, so the null conclusion is robust.

### ★ 1B data-ablation e8-equiv — GraSP phase mAP (cross-probe breadth, 2026-08-01)

Second probe on the SAME 6 arm e39 checkpoints (cached-head GraSP, paper mAP = macro AP over the
GraSP official TEST split, best-of-3-LR-heads via `scripts/eval_grasp_map.py`). Purpose: does the
triplet composition-null (every arm ≈ meta1b) also hold on a DIFFERENT downstream task, and does
openh add value on GraSP even though it's neutral on triplet? Ran via one-arm-per-node PARALLEL
fanouts (`grasp_abl_fanout.sh` head-train, `grasp_abl_score_fanout.sh` scoring; job 8724376).

| arm | GraSP mAP | head ckpt | vs full |
|---|---|---|---|
| **croppedopenh** | **66.59** | ep6 | +0.16 |
| **openhin** | **66.55** | ep4 | +0.12 |
| **lemonout** | **66.51** | ep5 | +0.08 |
| **full** | **66.43** | ep6 | — |
| **lapout** | **66.37** | ep5 | −0.06 |
| **gynout** | **66.35** | ep5 | −0.08 |

**★ RESULT: dead-flat tie — spread 0.24 mAP (66.35–66.59), every arm statistically identical.** The
triplet composition-null REPRODUCES on GraSP: no group removal (lap/gyn/lemon), no openh add (raw or
cropped), and not the full 15-source anchor moves GraSP phase mAP at 1B/e8-equiv. **openh is neutral
on GraSP too** (openhin 66.55 ≈ full 66.43), consistent with the 2B openh-in/out GraSP tie (68.23 in /
68.56 out) — openh's poison is TRIPLET-specific AND 2B/full-budget-specific; it does not appear on
GraSP at any scale measured, and does not appear on triplet at 1B/e8. **No evidence openh (or any
group) adds value on GraSP.**

CAVEATS: (1) heads walltime-capped ~4-6 epochs on the 1h queue (the head-train fanout hit the wall);
absolutes read LOW vs fully-trained refs (2B GraSP 68-72, 15-20 ep heads) but the cap is IDENTICAL
across arms so the cross-arm comparison is fair. (2) single-seed heads (no 3-seed CI here) — the 0.24
spread is within plausible single-seed head noise, so read as "tie," not a ranking. A longer-head +
3-seed rerun would firm absolutes but is unlikely to separate arms given this flatness. (3) SAR NOT
run: F1@10 is backbone-saturated (85.6-87.0, does not discriminate CPT from raw); SAR-FT is the only
SAR variant that separates encoders and is the right 3rd axis if one is wanted.

### Polaris (CUDA) — 3-seed mean ± std (cross-platform check)

| encoder | IVT | note |
|---|---|---|
| ours2b_e159 | 29.62 ± 0.26 | matches Aurora ✓ |
| meta2b | 28.48 ± 0.89 | ~1σ of Aurora ✓ |
| ours1b_e19 | 29.54 ± 0.53 | matches Aurora ✓ |
| meta1b | 27.11 ± 0.28 | ⚠ ~1.4 below Aurora — real XPU/CUDA quirk on this raw ckpt (ckpt verified bit-identical, clean load); baseline-only |
| SurgeNetXL | 26.00 ± 0.18 | matches Aurora ✓ |

**Reading (triplet):** all V-JEPA ≫ SurgeNetXL (+3.5). **CPT vs raw Meta: 2B = TIE** (Aurora
−0.44; Polaris +1.14, both <1.2σ), **1B = marginal CPT win** (+1.07 Aurora / +2.4 Polaris).
Scale ≥ CPT (both 2B > both 1B on mean). Full write-up: `docs/TRIPLET_PARITY_2026-07-09.md`.

### ★ v2 triplet REGRESSION root-caused to `openh` (leave-one-out, 3-seed, 2026-07-27)

The v2 lineage cratered triplet IVT to ~19-20 (worse than raw Meta 28.81). PE-Video removal
did NOT recover it. Ran a clean single-variable leave-one-out at e79 (fresh Meta-2B, 80ep,
gb=16, 3 seeds, job 8705354):

| arm (fresh Meta-2B, 80ep) | openh? | IVT mAP (3-seed) | vs raw Meta 28.81 |
|---|---|---:|---|
| `surgonly_fresh` (all 15 src) | **yes** | 21.99 | −6.8 (REGRESSED) |
| **`noopenh_fresh`** (14 src, openh dropped) | **no** | **28.92 ± 0.72** | **+0.11 (RECOVERED)** |
| `sitlonly_fresh` (sitl+sitl_2026 only) | no | 28.65 ± 0.71 | −0.16 (recovered) |
| raw meta2b | — | 28.81 | baseline |
| CPT fs_e159 (old lineage, never had openh) | no | 29.53 | +0.72 |

**VERDICT: `openh` (Open-H-Embodiment) is the cause of the v2 triplet regression.**
`surgonly_fresh` and `noopenh_fresh` are **byte-identical configs except openh is removed**
(verified: same init/LR/EMA/schedule/14 other sources) → +6.9 IVT swing is attributable to
openh alone. sitl-only corroborates (no openh → recovered). The old fs_e159 lineage never
regressed precisely because it predates openh. **This REFUTES the earlier "catastrophic
forgetting / needs general-video rehearsal" hypothesis** — rehearsal is not required; just
exclude openh. **Action: drop openh from the production CPT corpus.**
Scored via `scripts/compute_triplet_map.py` on each seed's val_probs_dump.

**ROOT CAUSE (data forensics, 2026-07-27): burned-in surgical-console UI overlay.**
openh (nvidia HF Open-H-Embodiment, 36,693 clips, 910×512 @ 8fps) is **~80% CMR
Surgical / Versius robot footage**, and ≥70% of those clips have a **persistent
synthetic console overlay burned into every frame**: "CMR SURGICAL" corner watermarks,
high-saturation colored tool-status icons (pink/blue/orange discs in all 4 corners that
CHANGE STATE with tool activity), and menu/camera buttons (bottom corners). Verified by
decoding + saving frames (`/tmp/openh_probe/frames/`) and a corner blue-pixel detector
(cmr: 70% of clips flagged, undercounts since it only catches the blue icon state).
**Why this poisons the encoder when unrelated general video (Kinetics) does not:** the
overlay is (1) IN-DOMAIN — it sits on real surgical tissue, so the JEPA objective can't
dismiss it as a different distribution the way it does a soccer clip; (2) SPATIALLY FIXED
and HIGH-CONTRAST — trivially predictable masked-token targets at fixed corner positions,
so the predictor learns to exploit the overlay instead of tissue/tool features (shortcut
learning); (3) DYNAMIC and TOOL-CORRELATED — the icons change with tool state, so they
act as a spurious proxy for exactly the tool/verb signal the triplet task needs, actively
displacing genuine tool-appearance features. Kinetics has none of these — it's out-of-domain
and carries no surgical-tool shortcut, so it dilutes but doesn't poison. Minor sub-sources
add a second issue (extreme color-cast: cuhk R/(GB)=1.98, ut_austin 2.64; near-toolless
cavity-navigation views) but CMR-overlay is the dominant mechanism (80% of openh).
This explains the triplet-specificity: the shortcut most directly corrupts TOOL
discrimination (triplet's tool axis dropped most, −17pt per [[v2-triplet-regression-diagnosis]]).
**CONTROL — nearly EVERY surgical source has an overlay; presence is NOT the discriminator
(2026-07-27, audited all 9 sources).** grasp (top/bottom borders + SIDE tool-name panels +
active markers — heavy, AND a probe set), surgenet (bottom toolbar + mid-frame "da Vinci"
watermark), surgtoolloc (bottom tool toolbar), sitl (bottom toolbar), heichole (endoscope
circle + v1.0) — ALL have overlays, ALL benign, ALL in the fs_e159 lineage. Only openh
poisons. **An earlier "crop removes the bottom strip → that's why they're benign" claim was
WRONG** — grasp's side panels and surgenet's mid-frame watermark are not croppable yet both
are benign. Crop-survivability is not the mechanism.

**The one metric that cleanly separates openh: overlay SATURATION/COLOR.** vivid non-red
colored-pixel fraction: openh 0.0134 vs all benign sources 0.0000-0.0010 (13-60x). Every
benign overlay is dark/desaturated text-on-black; openh alone has high-saturation vivid
colored discs (pink/blue/orange) that change state with tool activity. Discriminator =
overlay color/saturation + dynamics, NOT position. Diagnostic: `docs/openh_source_diagnostic.html`.

**HONEST CAVEAT: overlay-causality is NOT proven — and is now WEAKENED.** Leave-one-out proved
openh-the-DATASET regresses triplet (airtight). The vivid-overlay was a *mechanism hypothesis*
(visual forensics, correlational). **Leonardo's independent bucket-attribution
(/flare/ModCon/leonardo_borgioli/openh_attrib/HANDOFF.md, same 2 checkpoints) argues AGAINST
it:** per-openH-bucket Δmargin (control vs collapsed head) is DIFFUSE across all 21 buckets
(0.50–1.50, no tier), and regrouping his npz by overlay-presence gives CMR-overlay buckets
mean Δ=1.01 vs non-overlay Δ=0.95 — essentially equal, with big overlay buckets
(cmr_prostatectomy 0.59, inguinal_hernia 0.50) near the BOTTOM and non-overlay hamlyn/jhu
(1.47/1.43) at the TOP. If the overlay were the shortcut, overlay clips would degrade MOST;
they don't. **Leading explanation is now GLOBAL: domain gap (robotic-lab/phantom/embodiment
vs real-endoscopy eval) / mixture / openh's full sampling weight — NOT the overlay or any one
sub-bucket.** (The two methods differ: my leave-one-out = dataset causality; his Δmargin =
symptom localization. They don't strictly contradict, but his overlay-tie undercuts my
mechanism.) **Decisive test = the A6 croppedopenh ablation arm: still regresses → overlay
exonerated, global/content cause confirmed (matches Leonardo); recovers → overlay causal
after all. Do NOT report "overlay is THE cause."** Metrics: his micro-F1 79.6→74.1 (−5.5) and
our IVT mAP 28.9→22.0 (−6.9) are the same collapse in different triplet metrics.

**Center-crop removes the overlay (measured); whether it FIXES training is the open causal
test.** openh corner icons ~10-15% in from each edge; central-70% crop removes ~97% of
overlay pixels (full 0.013 → c80 0.004 → c70 0.0004; c70 frame verified clean). 910×512→638×360.

**Fix options: (a) DROP openh [shipped, clean — noopenh recovered triplet to 28.92];
(b) test SALVAGE via center-crop [also the causal experiment above]; (c) overlay-free
sub-sources only. Do NOT crop the OTHER (benign) sources — their dark-text overlays are
harmless and cropping breaks fs_e159 lineage comparability + removes real content.**
PROBE-dataset overlays (grasp etc.) are a SEPARATE tolerable concern: train/val-consistent
dark overlays don't leak and the frozen encoder isn't shortcut-captured by them; only vivid
overlays in PRE-TRAINING corrupt the representation. GraSP/SAR openh-in-vs-out probes pending
to confirm triplet-specific vs general corruption.

**GraSP openh-in/out breadth check (2026-07-27): the openh corruption is TRIPLET-SPECIFIC.**
GraSP phase mAP is essentially a tie regardless of openh (68.23 openh-in vs 68.56 openh-out,
Δ0.33) — nowhere near triplet's −6.9 collapse. This weakens the "overlay poisons the general
representation" reading and points toward a triplet-specific (tool/verb-discrimination) failure
mode, consistent with Leonardo's diffuse-Δmargin finding above. See `[[openh-overlay-root-cause]]`.

### ★ Fine-tune vs frozen readout (2026-07-18, job 8679530)

The dominant downstream lever on triplet is the **readout** (fine-tune vs frozen), not CPT.
Partial fine-tune = last-4 encoder blocks unfrozen, encoder LR 1e-5, single head, IVT mAP:

| triplet IVT mAP | raw 1B | CPT 1B | raw 2B | CPT 2B |
|---|---|---|---|---|
| Frozen | 28.84 | 29.49 | 29.18 | 29.04 |
| Fine-tune | 33.26 | 34.47 | 34.88 | 35.92 |
| **FT − Frozen** | **+4.4** | **+5.0** | **+5.7** | **+6.9** |

**Reading:** fine-tuning beats frozen probing by **+4 to +7 IVT mAP for every encoder** — frozen
probes massively understate what the representation carries. Under FT, **CPT > raw is small but
consistently positive** (+1.0 @1B, +1.2 @2B) — a real signal invisible frozen (frozen CPT-vs-raw
deltas were ≤0.3, inconsistent sign). **Conclusion: judge encoders by fine-tuning, not frozen
probes** — frozen protocols were hiding both the true representation quality AND the (small) CPT
edge. Caveats: single seed (frozen triplet seed spread ~1-2pt, so the +1 CPT gap wants seed
replication); FT dumps are ep25 (final), not necessarily best-epoch. This predates and is
unaffected by the later v2/openh regression work above (uses the fs_e159-lineage CPT checkpoints).
See `[[fine-tune-beats-frozen-readout]]`.

---

## 2. SAR-RARP50 action segmentation — TEST set (videos 41-50)

asformer head on cached features. **Two metrics that DISAGREE — report both:**
- **F1@10** = segmental (community-standard SAR-RARP50 metric; over-seg tolerant, rewards
  segment boundaries/order).
- **frame-macro-F1** = per-frame macro across 8 classes (rewards rare-class per-frame accuracy).

Scored with `scripts/eval_segmental_f1.py`. ⚠ **Protocol matters:** best.pt vs latest.pt can be
worth ~1 F1@10 point per seed; only compare like-for-like.

### ★ FULLY-CONTROLLED TABLE (both scales, 3 seeds, early-stop OFF/ep20, identical cache+head+split)

All four encoders trained with the SAME protocol: 3 seeds, `early_stop_patience` removed (full 20
epochs), shared per-encoder cache, identical asformer head recipe, scored on the official TEST split
(videos 41-50). This is the apples-to-apples set — everything else below is superseded/exploratory.

| scale | encoder | F1@10 (latest.pt) | F1@10 (best.pt) | frame-macro (latest) |
|---|---|---|---|---|
| **2B** | ours fs_e159 (CPT) | **86.93 ± 0.38** | 87.02 ± 0.28 | 72.43 ± 0.94 |
| **2B** | metaraw2b (raw) | 85.88 ± 0.90 | 86.01 ± 0.51 | 69.70 ± 3.42 |
| **1B** | ours e19 (CPT) | **86.27 ± 0.39** | 85.64 ± 1.32 | 69.74 ± 5.09 |
| **1B** | metaraw (raw) | 85.75 ± 0.45 | 85.81 ± 0.70 | 72.22 ± 1.55 |
| — | Published SOTA | 84.10 | — | — |

Per-seed F1@10 latest.pt: 2B ours 86.50/87.10/87.20 · 2B meta 84.89/86.65/86.09 · 1B ours
86.72/86.04/86.04 · 1B meta 86.07/85.23/85.95.

**CPT-vs-raw-Meta head-to-head (Δ = CPT − Meta, σ = combined seed std):**

| scale | policy | our | meta | Δ | σ | verdict |
|---|---|---|---|---|---|---|
| 2B | latest-vs-latest | 86.93 | 85.88 | +1.06 | 0.98 | 1.08σ TIE |
| 2B | best-vs-best | 87.02 | 86.01 | +1.01 | 0.58 | 1.72σ TIE |
| 1B | latest-vs-latest | 86.27 | 85.75 | +0.52 | 0.60 | 0.86σ TIE |
| 1B | best-vs-best | 85.64 | 85.81 | −0.17 | 1.49 | 0.11σ TIE |

**Verdict (SAR F1@10, both scales): STATISTICAL TIE everywhere.** CPT is nominally ahead in 3 of 4
cells (+0.5 to +1.1) and dead-even in the 4th, but NO comparison reaches 2σ. Surgical CPT does not
significantly beat raw Meta on the community-standard SAR metric at either 1B or 2B — F1@10 is
**backbone-saturated** (every V-JEPA encoder 85.6–87.0, all ≥ +1.5 over published SOTA 84.10).
This matches the triplet 2B tie and confirms the user's worry. frame-macro (non-standard) is mixed:
2B favors CPT (+2.7), 1B favors Meta (−2.5), both within their (large) noise. Checkpoint policy does
not systematically favor either side (best≈latest ±~0.6, no consistent sign). Superseded single-seed
rows and the best.pt-selection deep-dive are retained below for provenance.

<details><summary>Superseded single-seed exploratory rows (do not cite — replaced by the ★ table)</summary>

| encoder | F1@10 | frame-macro | ckpt / seeds |
|---|---|---|---|
| metaraw (raw 1B) | 86.32 | 72.39 | best.pt, 1 seed |
| metaraw (raw 1B) | 85.36 | 72.13 | latest.pt, 1 seed |
| ours1b e19 (CPT 1B) | 85.41 | 69.19 | best.pt, 1 seed |
| metaraw2b (raw 2B) | 86.38 | 67.44 | latest.pt, 1 seed |
| metaraw2b (raw 2B) | 83.92 | 64.97 | best.pt (ep6), 1 seed |

These mixed n=1 numbers (esp. Meta-2B best.pt 83.92 = bad-epoch outlier) produced the earlier
best-vs-latest confusion; the ★ 3-seed table supersedes them.
</details>

**★ latest-vs-latest 3-seed — THE FAIR 2B VERDICT (settled 2026-07-11, jobs 8662940/8662945/8663337):**
best.pt is an unreliable F1@10 selector (Meta-2B best 83.92 < its own latest 86.38), so the clean
comparison is **latest.pt (fully-trained ep20), 3 seeds each, early-stop disabled on both** (the
old single Meta-2B head was early-stopped at ep12 — unfair; retrained to ep20 to match fs_e159).
Result: **our-2B 86.93 ± 0.38 vs Meta-2B 85.88 ± 0.90 → Δ +1.06, combined σ 0.98 = 1.08σ.**

**Verdict: STATISTICAL TIE on F1@10** — CPT is *nominally* ahead (+1.06) but inside the seed noise
(1.08σ, need ≥2σ). Frame-macro: our-2B 72.43 vs Meta 69.70 (Δ +2.72 but only 0.77σ — Meta's
frame-macro is very noisy, σ 3.42, one bad seed s0). So even on the fair protocol, **surgical CPT
does not produce a statistically significant win over raw Meta at 2B on the community-standard
F1@10.** This confirms the user's worry and matches the triplet 2B tie: at 2B, both community
benchmarks are backbone-saturated and CPT's effect is within noise. NOTE the sign did flip vs the
best.pt confusion (CPT now nominally ahead, not behind) — because latest.pt removed Meta's
lucky-epoch inflation; but nominal ≠ significant. The productive question is unchanged: **why does
2B CPT not add signal** (recipe / objective / data), and does a discriminating harder benchmark exist.

**★★ Checkpoint-policy robustness check (2026-07-11, job 8663452 — is the table unfair to Meta?):**
Concern: our-2B was scored best.pt-3seed but Meta only best.pt-1seed (83.92, a bad-epoch outlier),
so latest-vs-latest might understate Meta. Resolved by scoring Meta-2B **best.pt at 3 seeds**. Both
encoders converge — best.pt is ~+0.1 over latest.pt for EACH (our +0.09, Meta +0.14), so ckpt
policy does NOT favor either side:

| policy | our-2B | Meta-2B | Δ | σ |
|---|---|---|---|---|
| best-vs-best | 87.02 ± 0.28 | 86.01 ± 0.51 | +1.01 | 1.72σ |
| latest-vs-latest | 86.93 ± 0.38 | 85.88 ± 0.90 | +1.06 | 1.08σ |
| most generous to Meta (our latest vs Meta best) | 86.93 | 86.01 | +0.92 | 1.44σ |

**Every fair policy gives CPT +0.9 to +1.0, none ≥ 2σ → robust TIE.** Even handing Meta its best
ckpt and ours its latest, CPT stays +0.92 ahead — the table does not understate Meta. best-vs-best
is the tightest (1.72σ) only because Meta's latest had one noisy seed (s0=84.89) inflating its std.
The single-seed rows above (Meta best 83.92, latest 86.38) are superseded by these 3-seed rows.

**Reading (SAR) — HONEST, the matched number changes the story to "no clean signal":** The
matched Meta-2B **best.pt = 83.92** came in *below* its own **latest.pt = 86.38** — the opposite
direction from Meta-1B (best 86.32 > latest 85.36). So **`best.pt` is not a reliable F1@10
selector**: it is chosen by a val-macro-F1 proxy that does not track segmental F1@10, and the
epoch it lands on swings F1@10 by ~2.5 points (83.9 ↔ 86.4 for the *same* Meta-2B encoder). That
epoch-selection variance **exceeds any plausible CPT effect**, so no honest CPT-vs-Meta F1@10
claim can be made from these single-selection numbers:
- Comparing best-vs-best (87.02 vs 83.92, "+3.1 CPT") would now be **cherry-picking in OUR favour**
  — our best.pt caught a good epoch, Meta's best.pt caught a bad one.
- The most stable Meta-2B anchor is **latest.pt = 86.38** (fully-trained). Against that, our-2B
  best.pt 87.02 is a ~0.6 nominal edge — but it is *still* protocol-mismatched (our best vs Meta
  latest) and inside the ~2.5-pt selection swing. **Not a real win.**

**Verdict at 2B on F1@10: no clean CPT signal — NOW SETTLED via latest-vs-latest 3-seed (see the
★ block above).** The fair protocol gives our-2B 86.93 ± 0.38 vs Meta-2B 85.88 ± 0.90 = 1.08σ, a
statistical tie. F1@10 is backbone-saturated (everything 84–87). This confirms the user's worry:
thousands of hours of surgical CPT does **not** significantly beat raw Meta on the community-standard
SAR metric at 2B. (The best.pt-selection confusion below is retained for the record; the ★ block
supersedes it.) **Do not report a 2B F1@10 CPT win.**

### SAR 2B CPT trajectory — cached VAL best val_macro_f1 (screening metric only)

Coarse screen (cached val macro-F1, NOT TEST F1@10). Meta-2B cached-val anchor = 72.14.

| epoch | e39 | e59 | e79 | e99 | e119 | e139 | e159 | e174 | e189 |
|---|---|---|---|---|---|---|---|---|---|
| val macro-F1 | 71.75 | 72.00 | 74.67 | 75.69 | 73.54 | **76.55** | 74.39 | 75.10 | 74.54 |

Note: e159 (74.39) is a local **dip** on this curve; e139 (76.55) is the peak. Triplet was
only ever probed at e159 — a 2B-trajectory triplet sweep would test whether the triplet "tie"
is partly an epoch-selection artifact. (This is a val-macro screen; TEST F1@10 is the arbiter.)

### ★ 3-seed TEST F1@10 reverify — fs_e139 / fs_e214 / v2_e224 (RESOLVED, 2026-07-25)

`[[v2-run-beats-old-lineage-sar]]` and `[[fs-e159-checkpoint-was-suboptimal]]` flagged their
single-seed epoch-plateau numbers as PROVISIONAL because a single-seed rescore of fs_e159
(83.93) contradicted its own TRUSTED 3-seed anchor (87.02±0.28) by >3 points — meaning
single-seed head-training noise could fully explain (or reverse) every ranking those notes
claimed. The deferred 3-seed reverification (job `scripts/sar_reverify_3seed_score.sh`, 18
checkpoints = {fs_e139, fs_e214, v2_e224} × {best.pt, latest.pt} × 3 seeds, scored
2026-07-25) **was actually run and completed but never propagated here.** Final means:

| checkpoint | best.pt F1@10 | latest.pt F1@10 | frame-macro (latest) |
|---|---|---|---|
| fs_e139 | 85.67 ± 0.87 | 86.93 ± 0.46 | 72.77 ± 2.78 |
| fs_e159 (★ table anchor, unchanged) | 87.02 ± 0.28 | 86.93 ± 0.38 | 72.43 ± 0.94 |
| fs_e214 | 86.15 ± 1.57 | 87.40 ± 0.38 | 72.89 ± 1.83 |
| v2_e224 | 86.95 ± 0.42 | 86.88 ± 0.46 | 73.36 ± 3.14 |

**Verdict: no real ranking — everything clusters within noise (85.67–87.40).** This resolves the
standing "was e159 cherry-picked / does v2 beat the old lineage?" question raised by the
provisional single-seed data: **NO on both counts.** fs_e159's 3-seed anchor sits comfortably
inside the spread of fs_e139/fs_e214/v2_e224, none of the pairwise deltas approach 2σ, and v2_e224
does not beat the old fixedshape lineage — it ties it, same as everything else in §2's
backbone-saturation finding. `[[v2-run-beats-old-lineage-sar]]` and
`[[fs-e159-checkpoint-was-suboptimal]]` are hereby **superseded/closed** by this table — do not
re-open the "e159 was a bad pick" question without new evidence beyond seed noise.

### ★ SAR-RARP50 partial fine-tune F1@10 ceiling (2026-07-23)

Same lever as the triplet FT-vs-frozen result above, applied to SAR. Partial fine-tune (last 4
encoder blocks, encoder LR 1e-5, single ASFormer head), full 25-epoch per-epoch F1@10 sweep,
CPT-only (no raw-Meta FT arm run at this granularity):

| model | ceiling epoch | F1@10 (per-epoch max) | best.pt (val_macro_f1-selected) | latest.pt (ep25) |
|---|---|---|---|---|
| CPT 1B (e19) | ep21 | **89.96** | 89.41 (−0.55) | 89.82 (−0.14) |
| CPT 2B (fs_e159) | ep22 | **90.21** | 89.77 (−0.44) | 89.85 (−0.36) |

**Reading:** fine-tuning lifts the F1@10 ceiling to **~90**, +3–5 points above the frozen-probe
saturation plateau (85–87, §2 ★ table) — consistent with the triplet FT finding: the readout is a
bigger lever than CPT-vs-raw. Both models peak ep19–22 then wobble/decline slightly (mild
overfitting past peak, not monotonic). **`best.pt`/`latest.pt` (selected by the noisy
val_macro_f1 proxy) undershoot the true per-epoch ceiling by 0.4–0.8 pt** — always rescore every
saved epoch checkpoint post-hoc rather than trusting the auto-selected checkpoint, same lesson as
`[[sar-metric-f1at10-not-macrof1]]`. **Caveat: CPT-only** — this establishes the achievable
ceiling for our checkpoints, not a CPT-vs-raw delta under FT (a raw-Meta FT arm at the same
per-epoch granularity would be needed for that comparison). See `[[sarft-ceiling-sweep-result]]`.

---

## 2b. SAR-RARP50 dense tool **segmentation** — TEST val_mIoU (separate probe from §2)

**Different probe from §2 above** — this is per-pixel instrument mask prediction (mean IoU over
a light conv decoder on the frozen ViT's 24×24 token grid), not the ASFormer action-segmentation
head. Ported from Leonardo's Polaris harness 2026-07-22 (`[[sarrarp50-segmentation-probe-port]]`),
first genuinely **spatial** probe in the suite (every other probe measures temporal/global
quality). `eval_name: video_segmentation_frozen`, configs under
`configs/heads/sarrarp50/segmentation/`, 60-epoch runs, `resume_checkpoint: true`.

| encoder | best val_mIoU | epoch | status |
|---|---|---|---|
| ours1b_e19 (CPT 1B) | **65.31** | 23 | complete (60/60 ep), job 8700778 |
| meta1b (raw 1B) | 63.78 | 33 | complete (60/60 ep), job 8700778 |
| ours2b_fs_e159 (CPT 2B) | **65.52** | 44 | complete (60/60 ep), job 8700779 |
| meta2b (raw 2B) | 63.60 | 47 | complete (60/60 ep), job 8700779 |
| ours2b_v2_e329 (CPT 2B, v2/leak-free lineage) | 65.00 (so far) | 43 | **INCOMPLETE — stalled at epoch 46/60**, see below |

**Reading:** CPT beats raw Meta by **+1.5 to +1.9 mIoU** at both scales (1B: 65.31 vs 63.78;
2B: 65.52 vs 63.60) — small but consistent, matching the same-magnitude small-CPT-win pattern
seen on frozen SAR F1@10 and triplet (nominal CPT edge, not dramatic). Single-seed each (no
seed-replication run yet for this probe — unlike §2's 3-seed protocol). `ours2b_v2_e329` (the
GraSP-leak-free v2-lineage 2B checkpoint, e329/333, still mid-pretraining at probe time) tracks
closely with `ours2b_fs_e159` (65.00 vs 65.52 at its last completed epoch) — consistent with the
§2/§3 pattern that different 2B CPT lineages land in the same narrow band, not with any
lineage-specific regression (this checkpoint is unaffected by the openh triplet-regression bug,
which is v2-fresh-init-specific; v2_e329 here is the earlier v2/leak-free run, see caveat below).

**`ours2b_v2_e329` run stalled, not finished.** 4 consecutive attempts on the single-node capacity
launcher died to an intermittent Aurora node/fabric fault (job 8701930, 4th attempt, killed by
SIGTERM mid-epoch 2026-07-25 21:46 — consistent with the prior 3 failures, not a walltime/OOM
issue). Currently at epoch 46/60 (best val_mIoU 65.00 @ epoch 43); `resume_checkpoint: true` +
`latest.pt` mean it can resume cheaply once relaunched. **A debug-scaling self-chaining resume
script for this config is still pending** (deferred — capacity slot was needed for pretraining).

---

## 3. GraSP phase recognition — mAP on official TEST split (Leonardo's probe)

**Source (not ours):** Leonardo Borgioli's GraSP probe. Status doc lives on Eagle,
`/eagle/tpc/leonardo_borgioli/surg_vid/grasp/GRASP_BENCHMARK_STATUS.md` (Aurora
mirror `/flare/ModCon/leonardo_borgioli/probes/configs/grasp_v1/`, byte-verified in
sync 2026-07-10). Runnable configs + runs on Flare under `probes/{configs,csv,runs}/grasp_v1/`.

**Task:** GraSP (Ayobi et al., TAPIS, arXiv:2401.11174) phase recognition, 11 classes,
robot-assisted radical prostatectomy, official test = CASE 041/047/050/051/053. Frozen
V-JEPA-2.1 encoder + ASFormer sequence head (the SAR-RARP50 head, only num_classes /
class_weights / CSVs swapped). ctx3 clips: 12-s windows @ 4 fps → 48 frames.
**Metric = macro mAP (sklearn AP, per-token softmax) on the TEST split** — the paper's
metric, directly SOTA-comparable.

**⚠ EVAL-LEAK CAVEAT — applies to every `fs_e{N}` (fixedshape lineage) row in this section.**
The `fixedshape` 2B pretraining corpus's `grasp` webdataset source contained GraSP's own TEST
cases (CASE041/047/050/051/053) — i.e. the checkpoints scored below (fs_e159 and the whole
e99–e214 trajectory) saw the probe's test frames self-supervised during CPT. Verified from the
per-checkpoint `params-pretrain.yaml` snapshot (the 1B `cleandata` lineage predates the leak and
is CLEAN; only the 2B `fixedshape` lineage is confounded). **Every "+3.47 CPT win" / trajectory
number below should be read as CPT-vs-raw-Meta on a LEAKED benchmark, not a clean result.** Fixed
2026-07-20 via a re-sharded `grasp_noleak` source (0 leaked cases), wired into the newer
`vitG384_fixedshape_v2`/v2-lineage configs. **The leak-free re-probe is now DONE (2026-07-29) —
see the CLEAN table immediately below.** Treat the entire fs_e* GraSP table + trajectory sweep
further below as **superseded by the clean numbers** — kept for provenance, not as a settled
finding. See `[[grasp-eval-leaked-into-cpt-corpus]]` and `[[grasp-eval-leak-fixed]]`.

**⚠⚠⚠ EVERY GraSP NUMBER IN THIS SECTION IS BUILD-CONFOUNDED — audit 2026-08-03.**
Applying the `[[tapis-campaign-lessons-for-probes]]` checklist to the "gap to TAPIS is
structural" conclusion found **three defects in the clip builder itself**, upstream of
every result below (clean AND leaky tables, trajectory sweep, ablation arms, unfreeze
ladder). Do not cite any absolute here, and do not cite the CPT-vs-raw delta, until the
rebuild is re-probed.

1. **Labels were wrong 6.60% of the time (val).** `build_grasp_asformer_ctx3_30fps.py`
   mapped annotation `frame_num` → second via `int(round(frame_num / ffprobe_fps))` with
   ffprobe returning `30000/1001` = 29.97, while GraSP keyframes are spaced **exactly 30**.
   The 1.001 factor accumulates: **+1 s at case start growing to +14 s (28 tokens — more
   than one whole 24-token window) by the end of CASE050.** Measured directly: 5,631/85,304
   tokens carry a label that disagrees with the phase their pixels show, and the per-case
   rate tracks case length (r=0.73: CASE051 2.94% → CASE050 7.92%). Anchor: CASE041's video
   is 9450.34 s with 9451 keyframes, so keyframe rank IS real second; the builder's key
   space overruns the video by +10 s.
2. **Wrong pixels / shortcut leakage.** Clips were cut from the RAW per-CASE da Vinci
   video (1280x1024) with the surgeon-console UI burned in — including **legible instrument
   name-plates that correlate with the phase label**. TAPIS is denied this (their released
   frames are debranded 1280x800). Our 384 center crop excludes the top/bottom bars but
   **cuts through the side name-plates**. Same failure mode as `[[openh-overlay-root-cause]]`.
   Direction matters: this **inflates** our score, so the true representation gap is
   probably WIDER than the headline, not narrower.
3. **Windows were never keyframe-centered.** Non-overlapping 12 s blocks meant token 0 had
   **zero lookbehind and 11.5 s lookahead** (token 23 the reverse); only the middle tokens
   approximated TAPIS's view, which centers 16 s on every keyframe.

**Also measured in the same audit — the probe's noise floor is ~3 mAP, not ±1.4.** Two
encoders whose exported features are **cos = 0.9998** score **69.31 vs 66.40** (2.91 apart),
and best.pt-vs-latest.pt on ONE encoder swings **3.17**. **The headline "+1.27 clean
CPT-vs-raw" is ~0.4x this noise and is NOT a measurable result.** See
`[[grasp-probe-noise-floor-3map]]`. Compounding it, the two arms were not even scored on
identical data: meta2b's cache has 3,576 val samples vs the v2 pair's 3,600 (exported at 24
vs 48 ranks; DistributedSampler round-up padding differs with world size).

**What the audit CLEARED** (positive evidence, not absence): no init/checkpoint mismatch
(all export logs report `<All keys matched successfully>`, zero missing-key warnings across
all 12 ranks); no LR-scaling error (our gb=24 matches the reference gb=24 exactly, and our
winning head's 1.5e-4 IS the reference LR); no multi-rank file race (every write in
`eval.py` is rank-gated + atomic; there is no `os.remove`/`unlink`/`rmtree` in the file).
Metric convention is also exonerated — TAPIS's own epoch-30 predictions reproduce to
**75.2507 under our scorer's convention**, identical to theirs to 4 decimals.

**Heads are OVERFITTING, not undertrained** (train_acc 44.5%→84.5% while val_loss rises
monotonically 1.07→1.52; all runs early-stop with val-F1 flat from ~ep6), so "train longer"
is not a lever and this part genuinely supports the structural reading.

**FIX SHIPPED, RE-PROBE PENDING:** `scripts/build_grasp_ctx_official.py` (commit 6649109)
rebuilds from GraSP's official debranded frames, whose dirs are indexed by the SAME
`frame_num` the annotation uses — so label lookup is a direct index and the
fps/rounding/seek chain is **structurally removed**, not patched. TAPIS-parity geometry
(16 s centered, 64 frames, 32 tokens). Verified CPU-only: all keyframes resolve incl.
CASE001 (45.45 fps, sparse) at 10973/10973; residual label mismatch **0.57%, all at phase
transitions, ZERO in phase interiors** (irreducible ±0.5 s quantization vs the old
builder's systematic interior error). Configs `configs/heads/grasp/official_ctx16/`
(commit ac64a03) export BOTH arms at 4 nodes, fixing the val-set padding mismatch.
Scorer fixes in commit d9b004a. **Expect the rebuilt number to come in LOWER** — the UI
shortcut is gone. That is the correct trade: a number comparable to SOTA beats a higher
number that is not. See `[[grasp-label-fps-drift-bug]]`, `[[grasp-probe-input-mismatch]]`.

### ★ CLEAN (leak-free) 2B GraSP result — supersedes the fs_e* table (2026-07-29)

Probed the **v2 lineage** (`surg_2_1_vitG384_v2`, pretrained on `grasp_noleak` — GraSP's 5 TEST
cases held out of CPT). Head-trains had completed 2026-07-25 but were **never scored to mAP** until
now; scored via `scripts/eval_grasp_map_cached.py` on the exported val cache (jobs 8714429 / 8714491),
same cached protocol / 3-LR-head sweep / gb=24 as the leaky table.

| encoder | best-head mAP | ensemble-of-3 mAP | leak status | ckpt / job |
|---|---:|---:|---|---|
| **ours2b v2_e324 (CPT 2B)** | **69.31** | 70.38 | **clean ✓** | best.pt ep7, job 8714429 |
| ours2b v2_final / e333 (CPT 2B) | 66.40 | 69.07 | clean ✓ | best.pt ep18, job 8714491 |
| meta2b (raw 2B) | 68.04 | — | clean baseline | job 8663718 |
| ~~ours2b fs_e159 (CPT 2B)~~ | ~~71.51~~ | — | **LEAKED — do not cite** | (retracted) |
| TAPIS SOTA (end-to-end) | 76.72 (76.07 v2) | — | — | dataset-paper, NOT frozen-probe |

**Reading (CLEAN) — the "+3.47 GraSP CPT win" DOES NOT SURVIVE leak removal:**
- **Leak was worth ~+2.2 mAP of inflation:** leaky fs_e159 71.51 → clean v2_e324 69.31 (−2.2 best-head).
- **Clean CPT-vs-raw = +1.27 best-head** (v2_e324 69.31 vs meta2b 68.04) — and the later v2_final is
  *below* raw at best-head (66.40 < 68.04). The old headline "+3.47, first benchmark where 2B CPT
  clearly beats raw Meta" was ~65% leak. **GraSP now joins SAR F1@10 (§2) and triplet IVT (§1): CPT ≈
  raw, a marginal +1-ish within this probe's ±1.4 single-seed noise.** Not a clean, large CPT win.
- **v2_final < v2_e324** mirrors the leaky lineage's late-CPT decay ([[triplet-2b-scale-vs-cpt]]) —
  best clean representative is the mid-lineage **v2_e324 (69.31 best-head / 70.38 ensemble)**.
- **SOTA gap is STRUCTURAL and now WIDER:** clean gap to TAPIS = 76.72 − 69.31 = **~7.4 mAP** (vs the
  leaky ~5.2). The leak was *helping* us, so it can never explain the gap. Dominant cause is
  **frozen-probe vs end-to-end fine-tune** (the +4–7 mAP FT lever, §1) — not insufficient surgical
  CPT (worth ≤~+1 here). No basis to suspect TAPIS themselves leaked (they own the split).
- **Caveats:** single-seed each (like the originals; ±1.4 seed noise on this probe → the +1.27 is
  noise-adjacent, needs 3 seeds before any "CPT helps GraSP" claim). Ensemble-of-3 beats best-head by
  a large +1.07/+2.67 here — consistent with weaker/shorter-trained heads decorrelating more; use
  best-head-vs-best-head (69.31 vs 68.04) for the apples-to-apples CPT comparison.

**⚠⚠ TEST-SET MODEL SELECTION — applies to EVERY absolute GraSP number in this section (clean AND
leaky), 2026-07-29 code audit.** The probe's `dataset_val` is
`grasp_phase_asformer_ctx3_seq_val.csv`, whose only cases are **CASE041/047/050/051/053 — the
official GraSP TEST split** (verified: `grep CASE` on the CSV). Selection runs on that same test
data: `eval.py:1083-1091` picks the best-of-3-LR-heads by **val macro-F1 every epoch**, and `best.pt`
is chosen by best val-F1 epoch — then we report mAP on it. So every number here is
**best-epoch × best-head selected on the test set**, an optimistic bias.
- **Effect on CPT-vs-raw delta:** ~fair — CPT and raw Meta are selected identically, so the +1.27 is
  a roughly symmetric comparison (the headline internal finding survives).
- **Effect on vs-TAPIS gap:** the absolute numbers (69.31, 71.51, 68.04) are **inflated** by this
  selection, so the true frozen-probe-vs-end-to-end gap to TAPIS 76.72 is **even larger** than the
  ~7.4 stated. Do not present these as blind test scores.
- **Also non-TAPIS-comparable (protocol):** our scorer flattens all 24 tokens × every overlapping
  12-s window (`eval_grasp_map_cached.py`), scoring each ~1-s keyframe many times, vs TAPIS's one
  prediction per official keyframe; and we train with inverse-freq class weights + temporal smoothing
  vs TAPIS plain CE. Selection by hard-label F1 but reporting by ranking mAP also explains the
  "F1 moved, mAP didn't" pattern seen in the unfreeze runs below.

**★ OVERLAP-DISPARITY QUANTIFIED (2026-07-29, `scripts/eval_grasp_map_aggr.py`, job 8718213) — it is
NOT a meaningful confound.** Re-scored the SAME cache + heads (no retrain) under 4 aggregations;
token→keyframe mapping validated by the dedup collapse ratio (full 24-tok = 3.90× ≈ expected 4×;
center-clip = 1.34× ≈ expected 1.33×).

| aggregation | v2_e324 (CPT) | meta2b (raw) | CPT−raw |
|---|---:|---:|---:|
| A. flatten-all (the §-top table) | 69.31 | 68.04 | +1.27 |
| B. center-clip only (drop 2 ctx clips) | 69.17 | 67.79 | +1.38 |
| C. time-dedup, all tokens→keyframe | **72.90** | 71.09 | +1.81 |
| D. center-clip + dedup | 69.70 | 68.26 | +1.44 |

  - **Center-clip ≈ flatten-all (B−A ≈ −0.2):** context/edge tokens do NOT drag mAP down (refutes the
    "edge tokens depress it" hypothesis). Removing the non-uniform double-counting is nearly neutral
    (D−B = +0.5). **So 69.31 is a fair single-prediction-per-keyframe number; the all-token flattening
    is not distorting it.**
  - **The +3.6 jump in C is multi-window TTA, not a bug fix** — it comes from averaging the softmax of
    the ~4 overlapping windows per keyframe. Real and available, but it gives us an advantage TAPIS
    likely didn't use, so it must be labelled "with multi-window TTA," not cited as the base number.
    The single-window TAPIS-comparable value is D ≈ 69.7 → gap to 76.72 stays ~7, **structural**.
  - **CPT-vs-raw delta is robust across all 4 aggregations (+1.27/+1.38/+1.81/+1.44)** — same sign,
    same small magnitude. Aggregation choice does NOT change the "CPT ≈ raw + ~1–2, noise-adjacent"
    conclusion.

**★ WEIGHTED-vs-UNWEIGHTED CE QUANTIFIED (2026-07-29, head-only retrain from existing cache, jobs
8718340/8718529/8718745/8718894 train + 8719092 score) — small, noisy, not a clean lever.** Removed
`class_weights` (single-variable; smoothing/LRs/caches identical), retrained the head for both encoders,
scored mAP under the 4 aggregations:

| aggregation | v2_e324 wCE | v2_e324 unwCE | meta2b wCE | meta2b unwCE |
|---|---:|---:|---:|---:|
| A. flatten-all | 69.31 | **70.29** | 68.04 | 67.56 |
| B. center-clip | 69.17 | 70.31 | 67.79 | 67.13 |
| C. time-dedup  | 72.90 | 73.84 | 71.09 | 70.23 |
| D. center+dedup| 69.70 | 70.84 | 68.26 | 67.37 |

  - **Asymmetric & small:** unweighted HELPS the CPT head (~+1.0 everywhere) but slightly HURTS raw
    (~−0.5). Both magnitudes sit inside this probe's ±1.4 single-seed noise, and both arms are still
    test-set-selected (best-epoch × best-head) — the winning head even flipped (unwCE CPT=head0 vs
    wCE=head1). So the apparent gap-widening (CPT−raw +1.27→+2.73) is **not a bankable finding**
    without seed replication.
  - **Clean takeaway:** unweighted CE is **not worse** and is marginally better for the reported
    ranking-mAP — consistent with inverse-freq weights being redundant-to-slightly-harmful when the
    metric (macro-mAP) already weights classes equally. Does NOT rescue a large CPT win; best
    single-window number D≈70.8 still ~6 short of TAPIS 76.72. **Both protocol disparities (overlap +
    weighted-CE) are now measured: neither is a material confound for the frozen result.**
- **FIX for a clean number (not yet done):** carve a val split from the 8 TRAIN cases (prefer official
  fold1/fold2), select on THAT, keep the 5 test cases blind; select by val **mAP** not test F1.

**⚠ THE TABLE + READING BELOW USE THE LEAKED fs_e159 CHECKPOINT — SUPERSEDED by the CLEAN
table above (2026-07-29). Retained for provenance only; do not cite 71.51 or "+3.47".**

**★ Ported into OUR repo on the hardened CACHED path (2026-07-11).** Configs
`configs/heads/grasp/full_cached_384/grasp_phase_{fs_e159,meta2b}_{export,probe}.yaml`;
scorers `scripts/eval_grasp_map.py` (live) + `scripts/eval_grasp_map_cached.py` (fast,
reads the exported val cache — no encoder). Topology: wide 2-node export (compute-bound) →
1-node head-train (comm-bound), gb=24 (bs2×12) matching Leonardo's live 2n×bs1. Cache is
**full-token (`pooled=none`)** so the head's learnable SpatialAttentionPool runs = numerically
equivalent to his live probe. This is the same pattern as SAR §2's cached path.

| encoder | phase mAP (TEST) | val macro-F1 @best | ckpt / heads | source |
|---|---|---|---|---|
| ours2b fs_e159 (CPT 2B) — **our cached** | **71.51** | 67.40 @ep5 | best.pt, best of 3 LR-heads (63.55 / **71.51** / 67.79) | job 8663717 (cached) |
| meta2b (raw 2B) — **our cached** | **68.04** | 62.56 @ep5 | best.pt, best of 3 LR-heads (63.15 / **68.04** / 66.76) | job 8663718 (cached) |
| ours2b fs_e159 (CPT 2B) — Leo live | 71.01 | 67.23 @ep9 | best.pt, best of 3 LR-heads (65.13/71.01/70.85) | Leo job chain 8659861→…→8660300 |
| TAPIS SOTA (end-to-end) | 76.72 | — | — | dataset-paper model, NOT frozen-probe |

**Reading (GraSP) — SETTLED, fair 3-head-vs-3-head (2026-07-11):**
- **Parity ✓ (cached reproduces live):** our cached fs_e159 = **71.51 mAP** vs Leonardo's live
  **71.01** (Δ +0.50, within head-init noise; val-F1 67.40 vs 67.23). The cached port is
  faithful — confirms the full-token cache preserves the learnable spatial pool.
- **CPT beats raw Meta at 2B on GraSP: +3.47 mAP** (71.51 vs 68.04), both best-of-3-heads,
  identical LR sweep / CSVs / gb=24 / cache protocol → the ledger's old best-of-3-vs-1
  fairness gap is CLOSED. val-F1 agrees: +4.84 (67.40 vs 62.56). Per-head, CPT wins on the
  matched winning LR too (head1 71.51 vs 68.04, +3.47).
- **This is the FIRST community benchmark where 2B CPT shows a real edge.** SAR F1@10 (§2) and
  triplet IVT (§1) are backbone-saturated → CPT ties at 2B. GraSP phase mAP is less saturated
  (frozen probe sits ~5–9 mAP below end-to-end TAPIS 76.72, lots of headroom) and DOES
  discriminate CPT from raw Meta. Per-class, CPT's gain concentrates in the harder/rarer phases
  (c9 Severing 68.8 vs 54.2, c7 Denonvilliers 44.7 vs 30.2) — consistent with surgical CPT
  helping exactly the fine-grained-phase discrimination the saturated metrics can't see.
- Still **~4.5 mAP behind end-to-end TAPIS SOTA** — expected for a frozen-encoder probe.
  (SOTA figure: the GraSP status doc / earlier note used **76.72**; the ar5iv paper Table 4 reads
  **76.07** for TAPIS Phases. Using 76.07 as the reference below; both are cited pending a
  definitive read of the published table.)

### GraSP SOTA-chase — SUMMARY SCOREBOARD (2026-07-11/12)

**⚠⚠ THE UNFREEZE (Stage C1/C2) RESULTS BELOW ARE INVALID — encoder-resume bug (found 2026-07-29).**
`load_checkpoint` (eval.py) never restored `checkpoint["encoder"]` on resume; it only reloaded the
heads + optimizer + scaler. The C1/C2 unfreeze runs **requeued mid-training** (C1 log has 3 header
lines, C2 has 5 → 3 and 5 job segments), so **every requeue reset the encoder to pretrained weights
while keeping the trained head, epoch counter, and stale Adam moments.** The C1/C2 runs were therefore
NOT continuous fine-tunes — the "partial unfreeze gives ~identical mAP / doesn't help / the gap is
unreachable by last-N" conclusion **cannot be trusted** and must be re-run after the fix.
Compounding confound: the final output norm `norms_block[-1]` (which produces the features,
vision_transformer.py:339) stays **frozen** during unfreeze — `models.py:55-77` only re-enables grad on
the last-N transformer *blocks*, not the output norm. **FIXED 2026-07-29** in `load_checkpoint`
(restores `checkpoint["encoder"]` when present, before optimizer state); regression test
`tests/evals/test_ft_resume_encoder.py` (1-step+resume+1-step == 2-step encoder parity; verified to
fail without the fix). The **frozen** rows are UNAFFECTED (frozen runs write no "encoder" key and a
reloaded frozen encoder is bit-identical; the v2 clean head-trains didn't requeue anyway).

Goal: push our GraSP frozen probe from **71.51** toward TAPIS SOTA **76.72** (v3; 76.07 v2).
**Every FROZEN lever was tried and NONE beats 72.12** — the frozen ceiling is real and robust,
confirming the ~4.5 mAP gap is **structural (frozen-vs-end-to-end)**, exactly as the TAPIS-recipe
diagnosis predicted (TAPIS fine-tunes MViT end-to-end; our encoder is frozen).

| lever | mAP | vs 71.51 base | verdict |
|---|---|---|---|
| baseline (final layer, best-of-3 head) | 71.51 | — | anchor |
| Stage A: temporal smoothing 0.25 | 72.12 (s0) → **70.19±1.4** (3-seed) | ~0 | ❌ single-seed luck; not real (see below) |
| Stage B: live encoder + RandAugment | (val-F1 66.1, below) | − | ❌ aug on frozen = noise |
| 3-head ensemble (mean) | 71.12 | −0.39 | ❌ weak heads poison mean |
| TTA (2-view spatial crop) | 72.08 | +0.57 | ❌ ≈smoothing, no add'l gain |
| Hierarchical 4-layer fusion | 68.99 | −2.52 | ❌ dilutes semantic layer |
| Stage C1: unfreeze last-2 (LR1e-5) | 71.64 | +0.13 | ❌ forgetting, val-F1 lied |
| Stage C2: unfreeze last-8 (safe sched) | 71.63 | +0.12 | ❌ 4× capacity, IDENTICAL mAP |

**Frozen best = smoothing 0.25, but 3-SEED VALIDATION DEFLATES IT (2026-07-12):** seeds {0,1,2} mAP =
{72.12, 69.36, 69.09} → **mean 70.19 ± 1.37**. seed0's 72.12 was a **+1.4σ high outlier**; seeds 1&2
land ~69.2, *below* the 71.51 baseline anchor. **So smoothing=0.25 is NOT a reliable win — the single-seed
+0.61 was seed luck.** The honest frozen number is ~70–72 with ~±1.4 seed noise, i.e. statistically
indistinguishable from the 71.51 baseline. This validates the ledger's own "3 seeds before any claim"
rule (single-seed probe deltas <~2 mAP are noise here). **Reportable GraSP frozen-probe result: fs_e159
CPT ≈ 71–72 mAP (seed-robust), CPT still clearly > raw Meta (68.04); the smoothing tweak is not a real
lever.** Detail sections below. **Frozen and unfrozen numbers kept DISTINCT.**

**CONCLUSION (2026-07-12): the frozen probe is at its ceiling (~72.1 mAP) and NONE of the tested
levers — frozen OR partial-unfreeze — closes the ~4.6 mAP gap to TAPIS 76.72.** Partial unfreeze
(C1 last-2 → 71.64; C2 last-8 → 71.63) gives IDENTICAL mAP despite 4× the trainable capacity and a
forgetting-safe schedule, and both *underperform* the frozen best. The val-F1 metric misleads under
fine-tuning (C1/C2 both ~68.6 val-F1 > frozen 68.0, yet lower mAP) — rare-class AP (c7/c9) is
consistently suppressed by fine-tuning. **This is strong evidence the gap is not reachable by
last-N unfreezing at safe LRs; matching TAPIS would require FULL end-to-end fine-tuning** (all 48
blocks, higher LR, their 30-epoch SGD recipe) — which abandons the frozen-CPT-probe framing entirely
and risks the rare-class collapse we already see amplifying. Recommendation: **report 72.12 as our
GraSP frozen-probe result** (CPT +0.6 over frozen baseline, +3.5 over raw Meta), validate it at 3
seeds, and frame the TAPIS gap honestly as frozen-vs-fully-finetuned rather than chasing it with
half-measures. Full end-to-end is a separate, larger project (see Stage C3 note if pursued).

### GraSP 2B CPT TRAJECTORY sweep — was e159 a cherry-picked epoch? (2026-07-27)

Answers the standing "was e159 a bad/lucky representative?" question (open item, and
the §2 note that e159 is a val-macro DIP while e139 is the screen peak). Probed the
fixed-shape (fs) 2B CPT lineage at **e99/e139/e159/e174/e214** on the SAME cached GraSP
port (configs `grasp_phase_fs_e{N}_{export,probe}.yaml` via
`scripts/gen_fs_trajectory_grasp_configs.py`; packed sweep
`scripts/grasp_trajectory_sweep.sh`; scored `scripts/eval_grasp_map_cached.py` via
`scripts/grasp_score_trajectory.sh`; jobs 8703972 head-train / 8704079+8704117 score).

| ckpt | best-head mAP | ensemble-of-3 mAP | head-train ep (best.pt) |
|---|---:|---:|---|
| fs_e99  | 70.18 | 70.85 | ep9 |
| fs_e139 | 70.58 | 71.72 | ep11 |
| **fs_e159** (anchor) | **71.51** | — | ep5 |
| fs_e174 | 68.71 | 71.83 | ep8 |
| fs_e214 | 67.01 | 70.28 | ep9 |
| meta2b (raw) | 68.04 | — | (ledger §3) |

**Reading — e159 was NOT cherry-picked; it is at/near the trajectory PEAK.** Best-head
mAP traces a clean rise→peak→decay centered on e159: e99 70.18 → e139 70.58 →
**e159 71.51 (peak)** → e174 68.71 → e214 67.01. So the headline **+3.47 CPT over raw
meta2b (68.04) is not epoch-luck** — every trajectory point ≥ e214's 67.01 is within ~1
of or above raw Meta, and the peak region (e139–e159) is clearly above it. Late CPT
(e214) DECAYS back toward raw Meta on best-head (67.01), consistent with the
[[triplet-2b-scale-vs-cpt]] "CPT erodes past ~e159" pattern seen on triplet — i.e. the
lineage has a genuine sweet spot around e139–e159, not a monotonic gain.

**Caveats (do not over-read):** (1) these trajectory head-trains ran only ~8–11 epochs
(debug-scaling 1h walltime) vs the anchor's clean best.pt@ep5 — best.pt selection is not
fully protocol-matched, so treat 70.18/70.58/68.71/67.01 as ~±0.5–1 vs the 71.51 anchor,
not exact. A fair rerun would train all to 20ep. The rise→peak→decay SHAPE is robust to
this; the exact gaps are not. (2) Single-seed each (like the original 71.51). (3)
Ensemble-of-3 > best-head here (Δ +0.7 to +3.3), OPPOSITE the strong-ckpt ensemble
penalty in the SOTA-chase below — because these shorter-trained heads are more
decorrelated (the LR5e-4 head isn't yet strictly worse). Don't mix ensemble and best-head
numbers across sections. **Bottom line: e159 is a fair, near-optimal representative of the
fs 2B lineage on GraSP; the CPT-beats-Meta result stands.**

### GraSP SOTA-chase — Stage A (frozen-cached recipe/head sweep), 2026-07-11

Goal: push the frozen probe toward TAPIS (~76.07). **Diagnosis (ar5iv):** TAPIS is MViT
**fine-tuned END-TO-END** + a *linear* head (no MS-TCN) → the gap is **frozen-vs-end-to-end**,
NOT the temporal head (our ASFormer is stronger). So recipe tuning is expected to yield <1 mAP;
the structural levers are Stage B (live aug) and Stage C (partial unfreeze). Ladder plan in
`~/.claude/plans/peppy-baking-wreath.md`. **Frozen and unfrozen numbers kept DISTINCT.**

Stage-A results (fs_e159, cached, single-seed=0, mAP via `eval_grasp_map_cached.py`):

| variant | change | best val-F1 | phase mAP | vs 71.51 |
|---|---|---|---|---|
| baseline | smoothing 0.15 | 67.40 @ep5 | 71.51 | — |
| **A6 sm0.25** | smoothing 0.15→**0.25** | 68.02 @ep4 | **72.12** | **+0.61** |
| A6 sm0.5 | smoothing 0.5 | 68.21 @ep7 (later peak) | (not scored; ≈plateau) | ~0 |
| A2 L14 | ASFormer num_layers 10→14 | 65.60 @ep4 | (dead end) | −1.8 val-F1 |
| A0 | selection audit: latest.pt(ep11)=67.25 mAP < best.pt(ep5)=71.51 | — | — | val-F1 selection is SOUND |

**Reading (Stage A):** best cheap lever = **temporal smoothing 0.25 → 72.12 mAP (+0.61)**; val-F1
and mAP agree. Deeper/heavier-smoothing don't help (frozen-cached ceiling ≈ 68 val-F1 / ~72 mAP,
near-saturated). **Caveat: single-seed** — +0.61 is within plausible seed noise; NOT 3-seed-validated
(deferred per plan — recipe gains are sub-1-mAP; compute prioritized to Stage B/C which target the
~4 mAP structural gap). smoothing=0.25 locked as provisional best head to carry forward. Cheap
class-weight/dropout variants staged but not run (ceiling already ~saturated). **Transfer:** the
smoothing-0.25 win likely applies to SAR (same head) — see open-items ★.

### GraSP SOTA-chase — Stage B (frozen encoder + LIVE features + augmentation), 2026-07-12

Still-frozen encoder, but run on the LIVE data path (`training=True` → RandAugment m7-n4 + RRC +
hflip + erase 0.25) with val-time TTA (`num_views_per_segment: 2`), smoothing 0.25. Config
`configs/heads/grasp/live_384/grasp_phase_fs_e159_live.yaml`; job 8667861 (debug 2n). Tests
whether augmentation ALONE closes part of the frozen-vs-end-to-end gap (encoder `init_opt`
heads-only, forward under `no_grad`).

| variant | best val-F1 | vs frozen-cached 67.40 | reading |
|---|---|---|---|
| Stage B (live + aug + TTA) | **66.14 @ep8** | **−1.26** | aug alone does NOT help; peaked ep8 then declined (ep9 64.30, ep10 61.38) |

**Reading (Stage B):** augmentation alone **underperforms** the deterministic cached probe on val-F1
(66.14 vs 67.40). Per-epoch aug injects noise the frozen 2B features can't exploit — the encoder is
fixed, so RandAugment only perturbs inputs to a frozen feature extractor. This **confirms the Stage-A
diagnosis**: the gap is structural (frozen-vs-end-to-end), not an augmentation/regularization deficit.
Not carried forward; mAP not scored (val-F1 already below anchor). Stage B verdict: **negative lever.**

### GraSP SOTA-chase — HIERARCHICAL multi-level features (4-layer fusion), 2026-07-12 — NEGATIVE

Biggest frozen lever per research (+1-2 mAP est). The V-JEPA-2.1 gigantic ViT natively concatenates
its 4 distillation layers [11,23,37,47] (each via its own TRAINED norms_block) along the feature axis
→ 6656-dim multi-scale tokens (`return_hierarchical` flag, `vit_encoder_multiclip_v21.py`; CPU-validated
all-keys-match, forward=[1,3,4608,6656]). Re-exported train(1.1TB)/val(613G) caches (jobs 8668190/8668224).
Head OOM'd twice (6656-dim ASFormer body is O(D²) everywhere = 16× params) → added `head_embed_dim` knob:
a learned `input_proj` Linear(6656→1664) fuses the 4 scales at head entry, body runs at 1664 (seams in
`asformer_head.py`/`eval.py`/scorer, default None = bit-identical). Probe 8668303, score 8668463.

| variant | best val-F1 | phase mAP (TEST) | vs frozen sm0.25 72.12 |
|---|---|---|---|
| sm0.25 (single final layer) | 68.02 @ep4 | **72.12** | — |
| **hierarchical (4-layer, proj→1664)** | 68.01 @ep4 | **68.99** | **−3.13** |

**Reading:** multi-level fusion **HURTS** (−3.13 mAP) despite matching val-F1. Per-class, the damage is
in the rare/hard phases: **c7 31.8** (vs ~44), c8 44.3, c9 55.5 (vs ~68) — the exact classes where the
final-layer CPT features carry the phase signal. Two compounding causes: (1) the random-init 6656→1664
projection is a bottleneck that must relearn fusion from scratch in ~4 epochs and loses fine-grained
discrimination; (2) the added shallow layers (11,23,37) are lower-level (texture/motion) and dilute the
semantic final-layer(47) signal that phase recognition needs. The distillation norm_blocks were trained
for a distillation objective, not phase probing. **Not a lever. sm0.25 72.12 stands as the frozen best.**
(A learned-weighted-sum fusion or final-2-layer-only concat might do less harm, but the direction is
clearly wrong — deeper≠richer for this task; deferred.)

### GraSP SOTA-chase — TEST-TIME AUGMENTATION (2-view spatial crops), 2026-07-12 — NEGATIVE

Re-exported the val/TEST split through frozen fs_e159 with `num_views_per_segment: 2`
(`EvalVideoTransform` = genuine sliding-window spatial crops, left+right along the long axis;
cache `feature_shape=[2,3,4608,1664]`, 307G — double the 1-view 154G, confirming 2 real views).
Scored sm0.25 best.pt with view-averaging (jobs 8668112 export / 8668153 score).

| scoring | head1 mAP (the winner) | vs 1-view 72.12 |
|---|---|---|
| 1-view (baseline) | 72.12 | — |
| 2-view TTA (avg) | **72.08** | **−0.04 (noise)** |

**Reading:** spatial-crop TTA gives **nothing**. GraSP frames are already center-framed surgical
views at 384px; the sliding-window crops only shift the field slightly and the frozen encoder's
features are crop-position-robust, so averaging two near-identical views adds no information. Not a
lever here. (Temporal-jitter TTA might differ, but would need a new export with temporal offsets and
is unlikely to beat the frozen ceiling given this result.) **sm0.25 72.12 still stands.**

### GraSP SOTA-chase — Head ENSEMBLE (free lever), 2026-07-12 — NEGATIVE for strong ckpts

Research flagged 3-head ensembling as a free win (heads already trained, scorer already computes
per-head outputs). Added mean-of-heads + top-2 to `eval_grasp_map_cached.py` and scanned all frozen
checkpoints on existing caches (job 8668124). **It HURTS the strong checkpoints:**

| checkpoint | head0 (LR5e-4) | head1 (LR1.5e-4) | head2 (LR5e-5) | best-of-3 | ensemble-of-3 | Δ |
|---|---|---|---|---|---|---|
| **fs_e159 sm0.25** (our best) | 60.43 | **72.12** | 67.72 | 72.12 | 71.12 | **−1.00** |
| fs_e159 baseline | 63.55 | 71.51 | 67.79 | 71.51 | 71.69 | +0.18 |
| meta2b (raw) | 63.15 | 68.04 | 66.76 | 68.04 | 70.32 | +2.28 |

**Reading:** the 3 "heads" are NOT decorrelated experts — they're the *same* head at 3 LRs, and the
hottest (head0, LR5e-4) is simply *worse* (60–63), not a diverse view. Averaging a 60 into a 72 drags
the mean down. Ensemble only helps where the best head is itself weak (meta2b +2.28, baseline +0.18) —
i.e. it narrows the *spread* but can't exceed a strong singular winner. **Dead end for pushing 72.12.**
top-2 (drop head0) also can't win: mean(72.12, 67.72) < 72.12 since head2 is 4.4 below head1. Real
ensembling would need *independently-seeded* heads (different init, not just LR) — deferred as it needs
3× the training, and the frozen ceiling (~72) makes the upside small. **sm0.25 best-head 72.12 stands.**

### GraSP SOTA-chase — Stage C (partial encoder UNFREEZE, end-to-end), 2026-07-12 — UNFROZEN

⚠ **These are UNFROZEN numbers — a SEPARATE question from the §3 frozen CPT-vs-Meta comparison.**
The 71.51 frozen anchor is NOT same-family comparable to unfrozen rows. Stage C is the SOTA-chase.

Behind `optimization.encoder_unfreeze_last_n: N` (flag-gated, default 0 = frozen bit-identical;
5 additive seams in `models.py`/`eval.py`). **C1 = last-2 blocks** (blocks 46+47 of 48), encoder LR
1e-5 (15× below head 1.5e-4), single head (fine-tune can't share a trainable encoder across the
3-head loss-sum), act-ckpt on, bf16, live path. Config `configs/heads/grasp/unfreeze_384/grasp_c1_last2.yaml`;
job 8667862 (capacity 2n). Verified post-hoc: C1 best.pt changed **exactly 24 keys = blocks 46+47**,
all else bit-identical to fs_e159 — the partial unfreeze is surgically correct.

| variant | best val-F1 | vs frozen 67.40 | phase mAP (TEST) | vs frozen head1 71.51 | reading |
|---|---|---|---|---|---|
| C1 last-2 unfreeze (s0) | 68.56 @ep8 | +1.16 | **71.64** | **+0.13** | val-F1 gain did NOT translate to mAP; class-7 rare phase DEGRADED |

**FAIR COMPARISON NOTE:** C1 trained a **single head @ LR 1.5e-4**. The frozen anchor's *winning* head
is that SAME head1 @ LR 1.5e-4 = **71.51** (the best-of-3 winner). So C1 71.64 vs frozen 71.51 is a
clean single-head-to-single-head comparison: **+0.13 mAP = noise.** (Scored via re-exported C1 TEST
cache — jobs 8667955 export / 8667983 score; the frozen fs_e159 cache is stale since C1 changed blocks
46+47.)

**Reading (Stage C1 — SETTLED, NEGATIVE):** last-2 unfreeze at encoder-LR 1e-5 does **NOT** improve
mAP (+0.13, noise) despite a +1.16 val-F1 bump. Three signals it's a genuine miss, not just seed luck:
1. **val-F1 and mAP DISAGREE here** (unlike Stage A where they agreed) — the val-F1 selector is not
   tracking mAP under fine-tuning; +1.16 val-F1 → +0.13 mAP.
2. **ep9 collapse** (68.56 → 59.49 val-F1 in one epoch) — classic small-LR encoder instability.
3. **Rare-class forgetting:** class-7 (Denonvilliers) AP **crashed to 27.12** vs frozen ~44.7 — the
   fine-tune ate exactly the fine-grained rare-phase discrimination that frozen CPT provided. Per-class
   C1 AP: c0 72.3 / c1 91.7 / c2 94.6 / c3 82.1 / c4 74.5 / c5 74.3 / c6 84.8 / **c7 27.1** / c8 56.3 /
   c9 68.3 / c10 62.0.

**Interpretation:** last-2 blocks is too small a capacity change to close the ~4.5 mAP structural gap,
AND the naive LR-1e-5 schedule causes forgetting of CPT's rare-phase edge. TAPIS fine-tunes the WHOLE
MViT end-to-end — matching that needs far more unfrozen capacity (last-8+, act-ckpt mandatory) with a
forgetting-safe schedule (warmup + lower LR / discriminative LR / norm-freeze). The C1 evidence does
**not clear the plan's C2 gate** ("C1 helping monotonically") on the current recipe. Frozen-cached
smoothing-0.25 (**72.12**) remains the best GraSP number to date — the frozen probe sits near its
ceiling and the cheap Stage-A lever still leads the unfrozen last-2 attempt.

**Stage C2 — last-8 unfreeze + forgetting-safe schedule (enc_lr 5e-6, warmup 0.15, act-ckpt), 2026-07-12:**
Escalated capacity 4× (blocks 40-47, verified) with a gentler schedule to fix C1's forgetting.
val-F1 peaked **68.66 @ep8** (breakout at ep8 like C1; ep9 declined). mAP-scored via repack→val-export
(job 8668575, blocks 40-47 confirmed changed) → **mAP = 71.63** (job 8668594).

| unfreeze variant | val-F1 @best | phase mAP | rare c7 / c9 AP | vs frozen 72.12 |
|---|---|---|---|---|
| C1 last-2 (LR1e-5) | 68.56 @ep8 | 71.64 | 27.1 / 68.3 | −0.48 |
| C2 last-8 (LR5e-6, warmup0.15) | 68.66 @ep8 | 71.63 | 36.3 / 64.2 | −0.49 |
| frozen sm0.25 | 68.02 @ep4 | **72.12** | ~44 / ~69 | — |

**Reading (Stage C2 — SETTLED NEGATIVE, closes the unfreeze ladder):** 4× the trainable capacity gives
**identical mAP** (71.63 vs C1's 71.64) — last-N unfreezing at safe LRs simply does not move GraSP mAP.
The forgetting-safe schedule did partly rescue c7 (36.3 vs C1's 27.1) but at the cost of c9 (64.2 vs
68.3) and still nets below frozen. The val-F1-vs-mAP divergence repeats (both C1/C2 beat frozen on
val-F1, lose on mAP). **The ladder does not clear its own gate — C2 does not beat C1, neither beats
frozen.** Stopping the unfreeze line here: matching TAPIS 76.72 would require FULL end-to-end fine-tune
(all 48 blocks + their high-LR 30-ep SGD recipe), a separate project that abandons the frozen-probe
framing. **Frozen smoothing-0.25 @ 72.12 is our GraSP result; next step is 3-seed validation of it.**

**Validation of Leonardo's setup vs our hardened protocol** (checked 2026-07-10; SDPA note
corrected 2026-07-11 after tracing the ACTUAL import path during our own cached port):
- ✅ **Runs OUR code, not a fork.** Launcher `run_asformer_probe_aurora.sh` `cd`s into
  `/lus/flare/projects/ModCon/ngetty/vjepa2` — so the encoder path is our repo, not his
  `vjepa2_probeloop` copy.
- ✅ **`use_sdpa: true` — the ACTIVE SDPA path is validated flash, CORRECTION to an earlier
  note here.** The eval encoder (`vit_encoder_multiclip_v21`) imports
  `app.vjepa_2_1.models.vision_transformer` → uses **`app/vjepa_2_1/models/utils/modules.py`**,
  NOT `src/models/utils/modules.py` (my first note cited the wrong file). That 2.1 `_sdpa`
  has `VJEPA_USE_XPU_FLASH` **default "1" (ON)** — the native SYCL-TLA fused flash kernel
  engages by default for the ViT-gigantic encoder (head_dim 64 ∈ eligible set; predictor
  head_dim 32 excluded). Confirmed live: our fs_e159 export logs `[vjepa2] xpu_flash=engaged`.
  This is **NOT the broken BSHD-scrambling bug** (`[[xpu-sdpa-bug-confounds-pretraining]]`,
  cos~0.02); it is a stride-only coerce, **HW-validated** by `scripts/gate_flash_sdpa_xpu.py`
  at exactly this shape (bf16 flash-vs-math cos=0.999988, max|Δ|=1.95e-3; see
  `[[xpu-flash-attention-port]]`). So the GraSP probe (ours AND Leonardo's live 71.01) runs on
  the validated fused kernel — correct, but note it is flash, not the "default dispatch" the
  first draft claimed.
- ✅ **Right checkpoint key** (`target_encoder`, not `ema_encoder`) — matches our loader.
- ✅ **mAP scorer is sound**: per-token softmax scores (not argmax) → one-vs-rest AP →
  macro; evaluates the true GraSP test JSON (builder maps "val"→test split). Classes
  absent from labels are skipped per sklearn convention.
- ✅ **Protocol asymmetry RESOLVED (2026-07-11).** Leo's original Meta config trained only 1
  head vs fs_e159's best-of-3. Our cached port ran **both encoders with the identical 3-head LR
  sweep** → fair best-of-3-vs-best-of-3 = **71.51 vs 68.04, +3.47 CPT**. Even matched-LR
  head-to-head (both head1, lr 1.5e-4) = same +3.47. The head-selection advantage is gone and
  CPT still wins — this was the ledger's #1 open GraSP item. (`docs/PROBING_GUIDE_FOR_LEO.md`
  parity rule applied to the head axis.)
- ⚠️ **Selection metric ≠ report metric.** Head/epoch are selected by val macro-F1, but the
  headline is mAP — same best.pt-selection risk we hit on SAR F1@10 (best.pt caught a good
  epoch for one encoder, a bad one for another). Low risk here (single split, mAP and F1
  co-move on this curve) but worth a per-epoch-mAP check if the CPT-vs-Meta gap is small.
- ℹ️ **Topology:** config says `tasks_per_node: 4` but the launcher **ignores it** and
  derives PPN=12 (full node) from `PBS_NODEFILE`; doc's "24 ranks, 12/node" is what actually
  runs. Global batch = world_size × bs(1). As long as CPT and Meta use the **same** node
  count (both 2n → gb 24) the comparison is batch-matched; verify this if either is re-run
  at a different node count.

---

## Open items / pending measurements

- [x] **Meta-2B best.pt F1@10 on TEST** (job 8662164) — DONE: 83.92 (< its own latest.pt 86.38;
      best.pt is not a reliable F1@10 selector — see reading above).
- [x] **our-2B vs Meta-2B latest.pt F1@10, 3 seeds** (the fair latest-vs-latest) — DONE
      2026-07-11: 86.93 ± 0.38 vs 85.88 ± 0.90 = 1.08σ TIE. Frame-macro 3-seed also captured
      (72.43 ± 0.94 vs 69.70 ± 3.42 = 0.77σ). See ★ block in §2. **SAR 2B loop CLOSED.**
- [ ] 1B latest-vs-latest F1@10 3 seeds (e19 vs meta1b) — 1B is the scale where triplet showed a
      marginal CPT win; SAR 1B still only single-seed (e19 85.41 vs meta1b 85.36 latest).
- [ ] 1B SAR F1@10 with 3 seeds (e19 vs meta1b both single-seed: 85.41 vs 86.32 best.pt).
- [x] **3-seed reverify of fs_e139/fs_e214/v2_e224 TEST F1@10** — DONE 2026-07-25 (never
      propagated until 2026-07-28): all cluster 85.67–87.40, no real ranking. Closes/supersedes
      `[[v2-run-beats-old-lineage-sar]]` and `[[fs-e159-checkpoint-was-suboptimal]]` — e159 was
      NOT cherry-picked and v2 does NOT beat the old lineage; see §2 ★ block.
- [x] **GraSP leak-free re-probe** — DONE 2026-07-29 (jobs 8714429/8714491). Probed the v2
      (`grasp_noleak`) lineage: v2_e324 = **69.31 best-head** / 70.38 ensemble; v2_final = 66.40 /
      69.07; vs raw meta2b 68.04. **The leaky "+3.47 CPT win" collapses to +1.27 (noise-adjacent);
      GraSP now ties like SAR/triplet.** See the ★ CLEAN table at top of §3. Remaining: 3-seed
      replication of v2_e324 (single-seed, +1.27 is within ±1.4 seed noise) before any "CPT helps
      GraSP" claim.
- [ ] **SNX (SurgeNetXL) SAR baseline on the official F1@10 metric** — never completed. Only
      ever scored on a cached macro-F1 proxy (`scripts/snx_score_compare.py`), which
      `[[cached-stability-doesnt-transfer]]` shows does not reliably track the official
      segmental F1@10. Do not cite any existing SNX-vs-VJEPA SAR comparison as resolved.
- [ ] **1B data-composition ablation arms** (`abl_{full,laponly,lapout,robonly,robplusgyn,
      gynout,openhin,croppedopenh,lemonout}`, checkpoints exist on disk as of 2026-07-28, mid-training) —
      no downstream probe (GraSP/SAR/triplet) has been run against any of them yet. Tests
      lap-vs-robotic composition + openh overlay-vs-content; see `[[cpt-ablation-next-steps]]`
      for the design and current training status.
    - **★ Corpus sampling composition (T=0.5 weights, computed 2026-07-28).** The `full` mix is
      NOT robotic-dominated at the batch level. Per-batch source share: **lemon 29.7%** (single
      largest source), sitl 14.9%, surgvu24 9.1%, surgenet_lap 7.8%, ... . **Group share: R 40.2%,
      L 19.3%, G 9.5%, M(lemon+small_surg) 31.1%.** Crucially, `lemon` is **60.3% laparoscopic**
      (2527 lap / 1667 robotic videos, from `LEMON/labels.json`), so **effective lap-family
      exposure = L + G + 60% lemon ≈ 46.5%** of every batch — nearly half, despite a ROBOTIC eval.
    - **★ CAVEAT — the `lapout` arm is a WEAK H2 test.** `lapout` drops only the dedicated L group
      (19.3%) but STILL contains lemon-lap (17.8%) + gyn (9.5%) = ~27% lap-family. So `lapout ≈ full`
      would NOT clear laparoscopic data — the dominant lap contributor (lemon) rides along in BOTH
      arms. The clean "no lap anywhere" arm is **`robonly`** (R only; the closest 1B analog of the
      ours1b_e19 win, which was robotic-only, no lemon, no openh, and beat raw Meta 29.61 vs 28.81).
    - **★ `lemonout` arm ADDED 2026-07-28** (job 8711455/vGwdlm, capacity, config diff = full − lemon
      only, byte-identical else). Single-source LOGO on the 29.7%-of-batch mixed set. `lemonout > full`
      ⇒ lemon net-hurts regardless of the L group; combined with `robonly` it separates "lemon
      specifically" from "all non-robotic data." Lemon's robotic/lap split is NOT recoverable from the
      resharded shards (keys flattened to `r{shard}_{idx}`) — a `lemon_robotic` subset would need a full
      re-segment from raw mp4s (~900G) using `segment_videos_to_wds.py --labels-json`; DEFERRED until
      lemonout/robonly justify it.
- [ ] **`ours2b_v2_e329` segmentation probe (§2b) is incomplete** (epoch 46/60, 4 crashes on an
      intermittent Aurora fabric fault) — needs a debug-scaling resume-chain script, not yet built.
- [x] **GraSP across the 2B trajectory (e99/e139/e159/e174/e214)** — DONE 2026-07-27 (jobs
      8703972/8704079/8704117): rise→peak→decay centered on e159 (best-head 70.18/70.58/**71.51**/
      68.71/67.01; raw meta2b 68.04). **e159 is NOT a cherry-picked epoch — it's at the trajectory
      peak; the +3.47 CPT win is not epoch-luck.** See §3 "GraSP 2B CPT TRAJECTORY sweep". Late CPT
      (e214) decays toward raw Meta. Caveat: trajectory head-trains only ran ~8-11ep (walltime).
- [ ] Triplet across the 2B trajectory (e99/e139/e174) — is the tie an e159-epoch artifact? GraSP
      trajectory (above) shows e159 IS near-peak; triplet sweep would confirm the tie isn't an
      e159 artifact either. ~3.7h/probe → capacity queue, not debug-scaling. Deferred.
- [ ] Why 2B CPT under-delivers on the reported metrics (recipe vs objective vs data).
- [x] **GraSP Meta-2B baseline, fair 3-head** — DONE 2026-07-11 via our cached port (jobs
      8663717/8663718): fs_e159 **71.51** vs meta2b **68.04** = **+3.47 CPT** (best-of-3 both).
      GraSP is the FIRST community benchmark where 2B CPT shows a real (non-tie) edge. See §3.
- [ ] **GraSP with seeds** — the +3.47 is single-seed each; run 3 seeds to put an error bar on it
      (SAR/triplet needed 3 seeds to call ties; GraSP's gap is larger but unquantified).
- [ ] GraSP **1B ViT-g** phase probe (cached port: clone the grasp configs, swap encoder to
      e19/meta1b) for the 2B-vs-1B point; GraSP **step** (21-cls) 2B config not yet built.
- [ ] **Why does GraSP discriminate CPT when SAR/triplet don't?** Hypothesis: GraSP phase mAP is
      unsaturated (frozen probe ~5 mAP under end-to-end SOTA) and CPT's gain concentrates in
      hard/rare phases (c7/c9). If it holds under seeds, GraSP becomes the go-to CPT-sensitive
      benchmark. Feeds the standing "why 2B CPT under-delivers on SAR/triplet" question.
- [ ] **★ Temporal-smoothing sweep on SAR (transfer from GraSP, 2026-07-11).** The ASFormer head's
      MS-TCN truncated-MSE consistency term (`loss_smoothing_weight`) was set to **0.15 in ALL 138
      SAR cached configs AND the original GraSP config**. On GraSP, raising it **0.15→0.25 gave
      +0.61 mAP** (71.51→72.12, single seed; val-F1 and mAP agree; see §3 / Stage-A sweep). SAR
      uses the **identical head + `sequence_labels: true`**, so the same knob applies directly —
      and SAR's community metric is **F1@10, which explicitly rewards low over-segmentation**, the
      exact thing temporal smoothing targets, so it could help SAR *more* than it helped GraSP's
      per-frame mAP. **Action:** sweep `loss_smoothing_weight ∈ {0.25, 0.5}` (± `threshold`) on the
      SAR cached probe, 3 seeds, score F1@10 (not just val-macro-F1). Cheap (reuses SAR caches).
      **NOT applicable to triplet** — it's not sequence-labeled (`token_pool: topk_mean`,
      `weighted_bce`, no temporal token axis); the smoothing code path (`use_smoothing =
      sequence_labels and loss_smoothing_weight>0`, `eval.py:1156`) never fires there.
      NOTE: if a SAR F1@10 gain lands, it is a **head-recipe** improvement (applies to CPT and
      Meta equally) — re-run BOTH encoders before any CPT-vs-Meta reread; it doesn't change the
      settled 2B tie unless it lifts one encoder more than the other.

## Cross-references
- Fairness/how-to: `docs/PROBING_GUIDE_FOR_LEO.md`
- Throughput/topology: `docs/PROBE_THROUGHPUT_GUIDE.md`
- Triplet deep-dive: `docs/TRIPLET_PARITY_2026-07-09.md`
- GraSP status (Leonardo, external): `/eagle/tpc/leonardo_borgioli/surg_vid/grasp/GRASP_BENCHMARK_STATUS.md`
  (Aurora mirror: `/flare/ModCon/leonardo_borgioli/probes/configs/grasp_v1/`)
- Scorers: `scripts/compute_triplet_map.py`, `scripts/aggregate_triplet_seeds.py`,
  `scripts/eval_segmental_f1.py`, `probes/scripts/eval_grasp_map.py` (Leonardo's, GraSP mAP),
  `scripts/eval_grasp_map_cached.py` (fast cached GraSP mAP), `scripts/sar_reverify_3seed_score.sh`
- Segmentation (§2b) probe code: `evals/video_segmentation_frozen/{eval.py,models.py}`,
  `src/models/segmentation_head.py`, `src/datasets/video_seg_dataset.py`,
  `scripts/build_sarrarp50_seg_csv.py`
