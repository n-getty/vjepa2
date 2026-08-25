# Probe results ledger — surgical V-JEPA 2.1 downstream evals

Living reference for all downstream probe results. **Append, don't rewrite.** Every number
here is copied from an on-disk artifact (JSON / CSV), not from memory. When you add a row,
record the **protocol** (checkpoint selection, #seeds, split) — mismatched protocols are the
#1 source of false conclusions on these tasks (see `docs/PROBING_GUIDE_FOR_LEO.md`).

_Last updated: 2026-08-19._

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

### ★★ P0 ft_gate: 3-seed fine-tune replication (2026-08-06/07, job 8730254)

3-seed re-run of the FT-vs-frozen result above, same recipe (last-4 encoder blocks unfrozen,
encoder LR 1e-5, single head), scored on latest.pt (ep25) with the canonical
`scripts/compute_triplet_map.py` / `scripts/aggregate_triplet_seeds.py`:

| triplet, 3-seed mean ± std | raw 1B | CPT 1B | raw 2B | CPT 2B |
|---|---|---|---|---|
| tool mAP | 92.88 ± 0.24 | 93.34 ± 0.34 | 93.41 ± 0.23 | 93.49 ± 0.54 |
| verb mAP | 81.89 ± 0.44 | 83.05 ± 0.09 | 82.82 ± 0.26 | 83.71 ± 0.34 |
| target mAP | 54.98 ± 0.09 | 57.00 ± 0.18 | 55.19 ± 0.53 | 58.18 ± 0.41 |
| mean per-task mAP | 76.58 ± 0.24 | 77.79 ± 0.20 | 77.14 ± 0.26 | 78.46 ± 0.42 |
| **IVT mAP** | **33.35 ± 0.25** | **35.49 ± 0.72** | **34.89 ± 0.46** | **36.09 ± 0.25** |
| **Δ IVT (CPT − raw)** | | **+2.14** | | **+1.20** |

**Reading:** raw-Meta 3-seed means land almost exactly on the single-seed job 8679530 values
(33.26→33.35 @1B, 34.88→34.89 @2B) — confirming that earlier single run was effectively one of
these three seeds, not a different regime. The CPT delta is larger with 3 seeds than the
single-seed read (+2.14 vs +1.21 @1B; +1.20 vs +1.04 @2B), i.e. this **confirms and modestly
strengthens** the original finding rather than overturning it — seed std (0.25–0.72) means it
still isn't a tight multi-sigma claim, but the sign and rough magnitude both hold up under
replication. See the matching SAR entry `[[sarft-ceiling-sweep-result]]` in §2, which this same P0
gate also closes (adds the raw-Meta FT arm that section was missing).

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

### ★★ P0 ft_gate: raw-Meta FT arm added, 3-seed, F1@10 on TEST (2026-08-06/07, job 8730254 + 8740150/8740416)

Supplies the raw-Meta FT arm the ceiling-sweep above explicitly flagged as missing, using the
same recipe, scored on the canonical TEST split (`scripts/eval_segmental_f1.py`), **latest.pt
only**, 3 seeds per arm:

| F1@10 | raw 1B | CPT 1B (e19) | raw 2B | CPT 2B (fs_e159) |
|---|---|---|---|---|
| 3-seed mean ± std | 88.58 ± 0.22 | 89.98 ± 0.05 | 88.32 ± 0.36 | 90.25 ± 0.50 |
| **Δ (CPT − raw)** | | **+1.40** | | **+1.93** |

**Reading:** two things cross-validate here. First, the CPT columns closely match the ceiling
sweep's CPT-only per-epoch-max (89.96 @1B, 90.21 @2B vs this table's 3-seed latest.pt means 89.98,
90.25) — a single-seed exhaustive per-epoch search and a 3-seed latest.pt-only protocol land
within 0.05–0.3 pt of each other, so both studies corroborate one another. Second, and new: with
the raw-Meta arm now measured at the same FT granularity, **CPT shows a real edge under
fine-tuning (+1.4 @1B, +1.9 @2B)** — larger than the frozen-probe F1@10 deltas in the §2 ★ table
(+0.5 to +1.1, all statistical ties) and, unlike those, not competing against a saturated
85–87 plateau; FT lifts both raw and CPT to ~88–90, and the gap between them opens up rather than
closing. This mirrors the triplet FT-vs-frozen pattern directly above: **fine-tuning is the
condition under which CPT's benefit becomes visible** — frozen readouts understate it for both
triplet and SAR. Same seed/epoch caveats as the triplet P0 result apply (seed std 0.05–0.5,
latest.pt/ep25 or ep30 dump, not best-epoch).

### ★★ Head ENSEMBLE (free lever), 3-seed, F1@10 on TEST (2026-08-15, job 8758553)

Extends [[seed-ensembling-generalizes-to-triplet]]'s free-lever family to SAR, closing the item
flagged in §6's open list. `eval_segmental_f1.py` already trained 3 LR-heads per checkpoint
(`multihead_kwargs`, same shape as GraSP's original ensemble) but the scorer only ever evaluated
`head_idx=0`, discarding the other two heads' forward passes for free. Added `--ensemble` (mean of
the 3 heads' per-clip **softmax probabilities**, averaged **before** argmax and before
`stitch_per_video`, not a post-hoc label-sequence average — segmental F1 is boundary-sensitive, so
averaging after argmax would not be equivalent). No retraining: same 12 checkpoints as the §2 ★
FULLY-CONTROLLED TABLE (4 arms × 3 seeds, `latest.pt`, official TEST split), same encoder forward
reused across all 3 heads.

**Sanity check before trusting any new number:** re-scoring `head_idx=0` (the scorer's old default)
through the new code path reproduced the ★ table's cited per-seed values **exactly** (e.g. 2B ours
86.50/87.10/87.20, 2B meta 84.89/86.65/86.09, 1B ours 86.72/86.04/86.04, 1B meta 86.07/85.23/85.95)
— the inference/stitching path is unchanged, only `--ensemble` is new code.

| F1@10, 3-seed mean ± std | best-head | ensemble | Δ |
|---|---|---|---|
| 2B CPT (fs_e159) | 88.93 ± 0.37 | 89.09 ± 0.17 | +0.16 |
| 2B raw (metaraw2b) | 87.99 ± 0.05 | 88.56 ± 0.29 | **+0.57** |
| 1B CPT (e19c) | 88.49 ± 0.07 | 89.14 ± 0.25 | +0.65 |
| 1B raw (metaraw) | 87.74 ± 0.11 | 88.41 ± 0.09 | **+0.67** |

Per-seed head F1@10 (head0/head1/head2) and ensemble column:
- 2B CPT: [86.50/87.44/88.89], [87.10/87.14/88.50], [87.20/87.61/89.40] → ens 89.02/88.93/89.31
- 2B raw: [84.89/86.69/87.97], [86.65/87.34/87.95], [86.09/87.20/88.06] → ens 88.15/88.72/88.82
- 1B CPT: [86.72/88.59/87.98], [86.04/87.56/88.47], [86.04/87.70/88.42] → ens 89.49/88.97/88.96
- 1B raw: [86.07/86.71/87.72], [85.23/86.86/87.89], [85.95/87.34/87.61] → ens 88.53/88.37/88.33

**⚠ "Best-of-3-head" was computed and is deliberately EXCLUDED from this table — it is not a
valid statistic and should not be cited.** An earlier pass of this section reported it (2B
88.93±0.37, 1B 88.49±0.07) and used it as the headline "revises the tie verdict" number. That
was wrong: "best head" was selected **by looking at F1@10 on the TEST set itself**, the exact
metric being reported — textbook test-set peeking / multiple-comparisons bias, not a real
pre-registered decision rule (nothing before this scan ever selected a head this way). It is not
"cherry-picking a seed" in the sense the user was worried about (seeds are still averaged, not
cherry-picked), but picking the max of 3 noisy draws *using the held-out metric* inflates the
number for exactly the reason a held-out test set exists to prevent. Retracted; do not resurrect
without a selection rule that does not look at TEST F1@10 (e.g. select by val_macro_f1 the way
`best.pt` already does).

**The legitimate comparison is head0 (fixed, no-peeking baseline) vs. the mean-probability
ensemble (a fixed combination rule that also never looks at TEST F1@10 to decide anything):**

| scale | protocol | CPT | raw | Δ | σ | n·σ |
|---|---|---|---|---|---|---|
| 2B | head0 (baseline) | 86.94 ± 0.31 | 85.88 ± 0.74 | +1.06 | 0.80 | 1.33σ |
| 2B | **ensemble** | 89.09 ± 0.17 | 88.56 ± 0.29 | +0.52 | 0.34 | **1.55σ** |
| 1B | head0 (baseline) | 86.27 ± 0.32 | 85.75 ± 0.37 | +0.52 | 0.49 | 1.05σ |
| 1B | **ensemble** | 89.14 ± 0.25 | 88.41 ± 0.09 | +0.73 | 0.26 | **2.80σ** |

**Reading: this IS the "combine heads to reduce noise and get a better read" result the ensemble
lever is supposed to produce** — at 1B the ensemble both raises the absolute F1@10 (86–89 range)
and tightens the CPT-vs-raw significance (1.05σ→2.80σ), a genuine improvement over the head0
baseline obtained with a fixed, no-peeking combination rule (mean-of-3-softmax). 2B improves less
(1.33σ→1.55σ) — still a real gain, just smaller; 2B's raw arm has more head-to-head spread
(σ jumps 0.31→0.74 on head0) so there's more noise for the ensemble to average out on the CPT
side too, partially offsetting. Neither result overturns the §2 ★ table's "statistical tie"
verdict outright (1.55σ/2.80σ are both short of a clean ≥2σ bar for at least 2B), but 1B's 2.80σ
is a meaningfully stronger signal than anything in the existing ★ table and is worth flagging as
a candidate revision, pending the caveats below.

**Caveats:**
1. Only `latest.pt`, matching this section's own P0 protocol — not cross-checked against `best.pt`
   selection, which the ★ table showed can swing F1@10 by up to ~2.5 points per checkpoint.
2. n=3 seeds per arm is small; σ estimates from 3 points are themselves noisy. A held-out 4th
   seed per arm would tighten this (§6's caveat 1, unresolved everywhere it's been raised).
3. Frame-macro-F1 (the non-standard secondary metric in the ★ table) was not re-scored here —
   `--ensemble` computes it per-head/per-ensemble internally (see the JSON dumps) but this section
   only reports F1@10, the community-standard metric.
4. The seed-diversity axis (independently-trained seeds, not LR-heads within one seed) is a
   DIFFERENT, still-untested ensembling lever for SAR — see the open item below.

Reproduce: `scripts/eval_segmental_f1.py --ensemble` (new flag, this session); scan driver
`scripts/sar_ensemble_scan.sh`; raw per-checkpoint JSONs at
`/flare/ModCon/ngetty/logs/sar_ensemble_scan/*_latest_TEST_ens.json`.

### ★★ SEED ensemble (independent axis from the head-ensemble above), F1@10 on TEST (2026-08-15, job 8758825)

Closes the open item this section flagged: extends seed-ensembling to the SEED axis
(independently-trained seeds — different init/data order, SAME `head_idx=0`), distinct from the
head-ensemble above (same seed, 3 different LR heads). Added `--dump-probs` to
`eval_segmental_f1.py` (mirrors triplet's `dump_probs_path` mechanism) and a new
`scripts/aggregate_sar_seeds.py`. Same averaging rule as every ensemble in this ledger:
probabilities merged **before** argmax/stitching, never post-hoc label sequences. Uses the same
12 checkpoints as the ★ table and the head-ensemble above (4 arms × 3 seeds, `latest.pt`, TEST
split), `head_idx=0` fixed throughout (no cross-contamination with the head-ensemble's axis).

**Only two protocols reported, matching this section's corrected convention — no best-of-N
selection by the held-out metric:**

| scale | protocol | CPT | raw | Δ |
|---|---|---|---|---|
| 2B | head0 baseline (mean±std, no peeking) | 86.94 ± 0.31 | 85.88 ± 0.74 | +1.06 (1.32σ) |
| 2B | **seed-ensemble** | **89.04** | **87.35** | **+1.69** |
| 1B | head0 baseline (mean±std, no peeking) | 86.27 ± 0.32 | 85.75 ± 0.37 | +0.52 (1.06σ) |
| 1B | **seed-ensemble** | **88.10** | **87.66** | **+0.44** |

(σ combined from the two arms' per-seed stds; the ensemble row has no std — it is a single fused
prediction, not 3 independent draws, same as every ensemble number in this ledger.)

Per-arm ensemble gain over its own baseline mean (all 4 positive, same direction as the
head-ensemble): fs_e159 +2.10, metaraw2b +1.47, e19c +1.83, metaraw +1.91.

**Reading — a DIFFERENT pattern from the head-ensemble axis, not a repeat of it:**
1. **2B's CPT-vs-raw delta GROWS under seed-ensembling** (+1.06 → +1.69), the opposite direction
   from the head-ensemble result immediately above (which grew CPT-vs-raw significance at 1B but
   barely moved 2B). Seed-diversity ensembling is closing more noise on the RAW arm's larger
   seed-to-seed spread (σ 0.74, the widest of any arm in this whole ledger's SAR tables) than on
   CPT's tighter spread (σ 0.31) — asymmetric noise reduction, not uniform, so it widens rather
   than closes the gap here.
2. **1B's delta slightly SHRINKS** (+0.52 → +0.44) — the more familiar shrink-toward-parity
   pattern from [[seed-ensembling-generalizes-to-triplet]] and the SAR head-ensemble result.
3. **Net: seed-ensembling on SAR is genuinely mixed, sign-dependent on scale** — unlike the head
   axis (consistent gain at both scales) or triplet's frozen result (consistent shrink at both
   scales). This is itself informative: it confirms ensembling's effect on a CPT-vs-raw delta is
   not a fixed direction to expect a priori — it depends on which arm's seed noise the specific
   axis happens to average out more.

**Caveats:** no std on the ensemble side (single fused prediction, not re-derivable without a
4th held-out seed); n=3 seeds/arm is the same small-sample caveat as every ensembling result in
this ledger; only `latest.pt`, matching every other number in this section.

Reproduce: `python scripts/eval_segmental_f1.py --head-idx 0 --dump-probs <path>` per checkpoint,
then `python scripts/aggregate_sar_seeds.py --root <dir-of-arm_sN-subdirs>`; scan driver
`scripts/sar_seed_ensemble_scan.sh`; raw dumps and per-arm JSONs at
`/flare/ModCon/ngetty/logs/sar_seed_ensemble_{dumps,scan}/`.

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

## 4. SARAS-ESAD double-head probe (presence + class-conditioned box) — TEST split

**Source:** Leonardo Borgioli's new probe (`/eagle/tpc/leonardo_borgioli/esad_probe/`), ported
to Polaris 2026-08-05 (this repo's Aurora launcher `run_esad_double_probe_aurora.sh` was never
run; all results below are from the **Polaris port**, `run_esad_double_probe_polaris.sh`).
21-class surgeon action detection (SARAS-ESAD, RARP2+RARP4 train / RARP1 val / RARP3 test),
48-frame windows (stride 16), frozen encoder → two heads trained **jointly**: ASFormer presence
head (BCE, 21-class multi-label per temporal token) + class-slot cross-attention box head
(L1+GIoU, masked by GT presence). Metric = presence frame-mAP (`macro` over all 20 valid
classes with ≥1 test positive; `well_supported` restricted to classes with **≥25 test
positives**, per `metrics.py:presence_frame_map(min_pos=25)`) + class-conditioned box IoU /
[email protected]. **This is NOT the SARAS-ESAD challenge's own detection AP** — the challenge scores
confidence-ranked, non-oracle boxes at 3 IoU thresholds (Frame-mAP = mean of AP@{0.10,0.30,0.50}
averaged over 21 classes; best published challenge result: an ensemble Faster R-CNN + weighted-box-
fusion submission at **AP_mean=19.28** — AP10=27.63, AP30=22.05, AP50=8.16 — arXiv 2104.03178;
organizers' own single-model baseline sat ~12–15 test AP_mean). Our probe removes both hard parts
of that task: the presence head only classifies (no confidence-ranked FPs penalized against the
full image), and the box head is **oracle-conditioned** — it is only scored where GT presence is
already given for free, so it never has to discover whether a class is present, only where. Our
map50 (41–48%) vs. the challenge's real AP50 (8.16) reflects that task simplification, not a
SOTA-beating result — **there is no valid apples-to-apples number here**; this probe's box metric
is oracle-conditioned on GT presence, so the two numbers are not comparable to each other, only
checkpoint-to-checkpoint within this probe.

**⚠ `min_pos=25` well-supported filter was BROKEN until 2026-08-05 and is a hard prerequisite for
every number below.** The library default `min_pos=1` only excludes classes with literally zero
test positives — already excluded by the separate NaN filter — so `well_supported_map` was
bit-identical to `macro_map` in the first (now-superseded) e159 run. Fixed by passing
`min_pos=25` explicitly in `train_esad_double_head.py:evaluate()`. **Any e159 number reading
`well_map == macro_map == 0.2817` is the pre-fix run — do not cite it.**

**UPDATE 2026-08-05 — a real, comparable detection-AP number now exists (§4b below).** The
paragraph above is left as originally written for history, but is now partially superseded:
`detection_ap.py`/`score_esad_detection_ap.py` (new, additive, no retraining) convert the
existing box-head output into a genuine confidence-ranked, non-oracle detection AP at the
paper's own IoU thresholds, giving an actual (if partial-coverage) comparison point. See §4b.

### Results (min_pos=25, fixed metric; all 4 checkpoints, single seed, 20 epochs each)

| checkpoint | scale | CPT? | well_map | macro_map | mean_iou | map50 | run order |
|---|---|---|---|---|---|---|---|
| **e159 (ours)** | 2B | yes | **0.3785** | **0.2951** | **0.4666** | **0.4791** | 1st (jobs 7340817→7341570) |
| meta1B (raw) | 1B | no | **0.3806** | 0.2892 | 0.4384 | 0.4137 | 2nd (job 7341780) |
| meta2B (raw) | 2B | no | 0.3431 | 0.2680 | 0.4548 | 0.4358 | 3rd (jobs 7342194→7342899) |
| ours1b_e19 (CPT) | 1B | yes | 0.3544 | 0.2697 | 0.4654 | 0.4669 | 4th (job 7343207) |

**Same-scale CPT-vs-raw deltas (Δ = CPT − raw, the only apples-to-apples reads):**

| scale | CPT | raw | Δ well_map | Δ mean_iou | Δ map50 |
|---|---|---|---|---|---|
| 2B | e159 0.3785 | meta2B 0.3431 | **+0.035** | +0.012 | +0.043 |
| 1B | ours1b_e19 0.3544 | meta1B 0.3806 | **−0.026** | +0.027 | +0.053 |

**Reading — matches this ledger's standing pattern, does NOT establish a clean CPT win.** The
well_map delta **flips sign between scales** (2B: CPT ahead; 1B: CPT behind), and both deltas
(~0.03, ~2.6–3.5 mAP points) sit inside/at the edge of this project's measured ~3 mAP
single-seed noise floor for cached probes (§3's `[[grasp-probe-noise-floor-3map]]`: two
cos=0.9998-identical encoders scored 2.91 apart on a structurally similar cached protocol).
**Consistent with §1/§2/§3's repeated finding that surgical CPT does not reliably beat raw
Meta checkpoints on this project's downstream probes.** The one signal that holds at BOTH
scales: CPT checkpoints have better box localization (mean_iou, map50) than their same-scale
raw counterpart — small but same-direction at 1B and 2B, unlike presence.

**⚠→✅ RUN-ORDER CONCERN, flagged 2026-08-05, RESOLVED BY ANALYSIS 2026-08-05 (false alarm).**
Original worry: e159, probed **first**, scored best on 3 of 4 metrics, suggesting run/submission
order correlates with score. Checking all 4 metrics against run order (1st→4th = e159, meta1B,
meta2B, ours1b_e19) refutes a monotonic or order-driven effect:
  - `well_map`: 0.3785, **0.3806**, 0.3431, 0.3544 — 2nd (meta1B) is highest, not 1st.
  - `macro_map`: **0.2951**, 0.2892, 0.2680, 0.2697 — 1st is highest, but not a decaying chain
    (4th > 3rd).
  - `mean_iou`: **0.4666**, 0.4384, 0.4548, 0.4654 — U-shaped: 2nd is the *worst*, 4th recovers
    to near-1st.
  - `map50`: **0.4791**, 0.4137, 0.4358, 0.4669 — same U-shape as mean_iou.
A real order/pipeline artifact would show a monotonic trend across all four metrics; instead
only `macro_map` favors 1st, and 2nd/meta1B is simultaneously the *best* on well_map and the
*worst* on mean_iou/map50 — inconsistent with any single confound tied to submission order.
All 4 runs share bit-identical `test_pos_per_class` (same 7274-instance test split, ruling out
a data/export drift). The spread that remains (well_map range 0.343–0.381, i.e. ~0.038) is
consistent with this project's already-documented **~0.03 (3 mAP point) single-seed noise
floor** (`[[grasp-probe-noise-floor-3map]]`), and several of the swingiest per-class APs sit on
classes with only 26–96 test positives — small-N class noise, not a systematic effect.
Independent corroboration that e159 isn't an order-fluke: §3's GraSP 2B trajectory sweep already
showed e159 sits at a genuine trajectory **peak** (best-head 71.51 vs e99 70.18 vs e214 68.71),
so e159 scoring well here is consistent with checkpoint quality, not launch order.
**Conclusion: no protocol artifact found; treat this as ordinary single-seed noise, same caveat
as every other single-seed comparison in this ledger.** Does not by itself validate the
CPT-vs-raw deltas above as real — that still needs the 3-seed replication called out below —
but the order-correlation is no longer a reason to distrust them beyond the noise floor already
disclosed.

**Caveats:**
- Single seed per checkpoint (no seed replication yet — every other probe in this ledger
  needed 3 seeds to distinguish signal from noise at similar effect sizes).
- All 4 share the identical harness/cache/head-training recipe (fair on that axis).
- e159 and meta2B both needed a walltime-kill-and-resume mid-run (PBS 1h debug-queue limit);
  meta1B and ours1b_e19 completed in a single submission. Resume correctness was verified
  (epoch/best_metric/optimizer state all continued correctly, per-epoch curve is smooth across
  the resume boundary for both) but is listed as a candidate cause above out of caution.
- 6 of 20 classes have <25 test positives (down to 2–3) and are correctly excluded from
  `well_supported_map` by the fix above, but still drag `macro_map` down for all 4 checkpoints
  — expect `macro_map < well_map` always; this is by design, not a bug.

See `esad-probe-leo-eagle-locations.md` (session memory) for the full launcher/port history,
including 4 distinct bugs fixed to get this pipeline running on Polaris (missing modelcustom
module, `ClipAggregation` version mismatch, a shard-cache/shuffle bug in the training script,
and a launcher rankwrap-path race) — none of which are expected to explain the order-correlation
above (all fixes were applied identically before ANY of the 4 checkpoints ran), but listed for
completeness when investigating.

## 4b. SARAS-ESAD real confidence-ranked detection AP (comparable to the paper) — TEST split

> **🛑 CORRECTION 2026-08-05 (supersedes this section's original headline claim).** Everything
> below was computed over a **partial ground-truth denominator**: only **3,587 of the test
> set's 11,207 real GT boxes (32.0%)** ever entered the AP recall denominator, because only
> 1,816 of 6,088 test frames are covered by a window. AP is recall-weighted, so scoring a
> 32% subset does **not** yield a number comparable to the paper's — it inflates it ~3.4×.
> The original reading ("within ~0.01–0.02 of the challenge's best submission", "comfortably
> above the organizers' baseline") is **WITHDRAWN — it was wrong.** The corrected,
> apples-to-apples numbers are in **§4b-corrected** below: **e159 = 0.0530 vs the paper's
> 0.1928, a ~3.6× gap, not a near-tie.** The original text is preserved unedited for history;
> read it only as "AP over the covered 32% subset", never as a SOTA comparison.

**New 2026-08-05, purely additive — no retraining, no re-export.** The box head already emits
one box per class per temporal token regardless of ground truth, and the presence head already
emits a per-class confidence for that same token — exactly the shape a real detector needs.
Added `detection_ap.py` (greedy IoU-matched, VOC-style AP per class, shared PR-curve
integration with `metrics.py:average_precision`) and `score_esad_detection_ap.py` (runs the
existing `best.pt` over the existing test cache, ranks every predicted box by the presence
head's own sigmoid confidence — **no oracle GT-presence gating** — and scores it against real
ground truth). Both files live in `esad_probe/` alongside Leo's code; unit-tested against 6
hand-computed cases (perfect/zero overlap, no-preds, no-GT, duplicate-prediction-same-GT,
2-GT-1-TP-1-FN) with the real torch backend before running on real data. Run via
`run_esad_detection_ap_polaris.sh` (PBS `debug`, job **7348633**, 1m29s, `Exit_status=0` — CPU-
light, no encoder re-inference needed since it scores from the existing feature cache).

Metric = per-class AP averaged over IoU={0.10, 0.30, 0.50} (`AP_mean`), macro-averaged over
classes with test-set ground truth — the SARAS-ESAD challenge's own headline metric (arXiv
2104.03178). Best published challenge result: AP_mean=**19.28** (AP10=27.63, AP30=22.05,
AP50=8.16, ensemble Faster R-CNN + weighted-box-fusion); organizers' own baseline ~12–15.

### Results (job 7348633, all 4 checkpoints, single seed)

| checkpoint | scale | CPT? | AP_mean | AP10 | AP30 | AP50 |
|---|---|---|---|---|---|---|
| e159 (ours) | 2B | yes | **0.1796** | 0.2607 | 0.2059 | 0.0723 |
| meta1B (raw) | 1B | no | 0.1704 | 0.2521 | 0.1936 | 0.0656 |
| meta2B (raw) | 2B | no | 0.1715 | 0.2433 | 0.1985 | 0.0727 |
| ours1b_e19 (CPT) | 1B | yes | 0.1752 | 0.2491 | 0.2082 | 0.0681 |
| *SARAS-ESAD baseline* | — | — | *~0.12–0.15* | — | — | — |
| *SARAS-ESAD best submission* | — | — | *0.1928* | *0.2763* | *0.2205* | *0.0816* |

**⚠ Frame coverage is 1816/6223 test frames (29.2%), identical across all 4 checkpoints (same
window manifest/test split).** This is a real, measured limitation, not an estimate — it comes
from two compounding, disclosed causes: (1) tubelet=2 means every 48-frame window scores only
the LATER frame of each 2-frame pair (structural, ~50% by construction — the encoder has
already fused the pair by the time the box head sees it, so the earlier frame's box is not
separately recoverable from this architecture); (2) `build_esad_windows.py`'s gap-aware
windowing additionally drops 226 windows at RARP3's 1599 frame-index gaps, so only frames
inside sufficiently-long contiguous runs ever enter a window at all. **These 4 numbers are
each averaged over the identical 1816-frame subset, so the CPT-vs-raw comparison within this
table is still apples-to-apples — the coverage caveat affects comparability to the paper's own
6223-frame numbers, not comparability across our own 4 checkpoints.**

**Reading.** All 4 checkpoints land in a tight band (AP_mean 0.170–0.180, a 0.010 spread) —
tighter than the well_map spread in §4's oracle-conditioned table (0.038) and well inside the
~0.03 single-seed noise floor. e159 is nominally highest but by a margin (+0.0044 to +0.0092
over the other three) that is noise, not signal — do not read this as a CPT win. ~~**All 4
checkpoints sit comfortably above the SARAS-ESAD organizers' baseline (~0.12–0.15) and within
~0.01–0.02 of the challenge's best submission (0.1928)** — a genuinely strong result for a
frozen-encoder, class-slot box head with no NMS, no anchor tuning, and no confidence
calibration beyond a plain sigmoid, though the coverage caveat above means this is not a
strictly apples-to-apples beat of 19.28.~~ **← STRUCK 2026-08-05: WRONG.** These are AP over
the covered 32% of GT; comparing them to the paper's full-set 0.1928 is invalid. On the
paper's own denominator e159 is **0.0530**, i.e. ~3.6× BELOW the best submission and below the
organizers' baseline too. See the correction banner at the top of §4b and §4b-corrected.
Per-class ordering sanity-checked against §4's
oracle-conditioned `test_per_class_mean_iou` for e159: classes 0 and 14 (near-perfect presence
AP, best oracle IoU in §4) are also the strongest here (AP_mean 0.70 and 0.62), and rare
classes with 1–21 test GT instances collapse toward zero in both metrics — consistent, no sign
of a scoring bug.

**Caveats:**
- Single seed per checkpoint, same as §4.
- 29.2% frame coverage (measured, not estimated) — see above; full-coverage would need a
  second, phase-shifted export pass so the *earlier* frame of each tubelet pair also becomes
  some other window's representative frame (roughly doubles export cost) — not done, deferred
  as a follow-up only if these partial-coverage numbers justify the investment.
- 1 box per class per token cap (documented in STATUS.md) still applies — same-class duplicate
  GT instances (2–4% of non-empty frames) are undercounted, same limitation as §4.
- No cross-window NMS beyond per-`(frame,class)` best-confidence dedup (overlapping test
  windows, stride=16/window=48, mean up to 3 windows independently predict a box for the same
  real frame) — sufficient here since the box head already caps at 1 box/class/token.

See `esad-probe-leo-eagle-locations.md` (session memory) for implementation notes and the raw
per-checkpoint JSON paths (`runs/esad_double_*/test_detection_ap.json`).

### 4b-corrected. The same runs scored on the paper's denominator — the REAL gap to SOTA

**2026-08-05. No new runs, no re-export, no retraining — this is a denominator fix applied to
the exact same `test_detection_ap.json` artifacts.**

**The error.** `score_esad_detection_ap.py` builds its GT pool *only from the feature cache*,
i.e. only from frames some window actually covers. Uncovered frames contribute neither
predictions nor ground truth, so they silently vanish from the recall denominator instead of
counting as missed detections. The paper scores all 6,088 test frames / 11,207 GT boxes; we
scored 1,816 frames / **3,587 GT boxes = 32.0%**. AP is recall-weighted, so this is not a
small bias — it is a ~3.4× inflation.

**The fix is exact, not an approximation.** Predictions exist only on covered frames, so adding
the uncovered GT back changes *nothing* about the ranked TP/FP sequence — it only enlarges
`n_gt`. Every recall value therefore scales by `gt_cov_c / gt_full_c`, and since VOC AP is a
sum of `Δrecall × precision` terms, per-class AP scales by the same factor:

```
AP_full_c = AP_cov_c × (gt_cov_c / gt_full_c)
```

**Verified empirically, not just algebraically:** feeding the real `detection_ap.py` a fixed
prediction set plus 3/10/37 extra unpredicted GT instances reproduces the predicted AP to
≤2e-8 in all three cases. Classes with GT on disk but none in the covered subset (class 12,
37 real boxes) correctly become AP=0 rather than NaN-dropped.

| checkpoint | scale | CPT? | AP_mean **(full)** | AP10 | AP30 | AP50 | AP_mean (covered-32%, as first reported) |
|---|---|---|---|---|---|---|---|
| e159 (ours) | 2B | yes | **0.0530** | 0.0773 | 0.0609 | 0.0208 | *0.1796* |
| meta1B (raw) | 1B | no | 0.0495 | 0.0734 | 0.0563 | 0.0190 | *0.1704* |
| meta2B (raw) | 2B | no | 0.0502 | 0.0721 | 0.0580 | 0.0204 | *0.1715* |
| ours1b_e19 (CPT) | 1B | yes | 0.0506 | 0.0723 | 0.0596 | 0.0200 | *0.1752* |
| *SARAS-ESAD baseline* | — | — | *~0.12–0.15* | — | — | — | — |
| *SARAS-ESAD best submission* | — | — | ***0.1928*** | *0.2763* | *0.2205* | *0.0816* | — |

**Reading — the honest picture.** Our best frozen probe reaches **0.053 vs the paper's 0.1928
(~3.6× gap)**, and sits **below** the organizers' own ~0.12–0.15 baseline. That is a very
different conclusion from the original section. What survives unchanged: the CPT-vs-raw
comparison *within* our 4 checkpoints (all rescaled by the identical per-class factors, so the
ordering and the "all inside the noise floor" verdict are untouched — e159 nominally top, spread
still ~0.004).

**The gap is dominated by coverage, which is a fixable pipeline property, not a model ceiling.**
68% of GT is unreachable today purely because no window covers those frames. See §4b-followup's
corrected verdict below: recovering coverage is the single largest measured lever found so far.

**Also corrected: `total_test_frames` was 6223, but the test split has 6,088 frames** (verified:
6,088 `.jpg` in `test/`, 6,088 `.txt` in `test_labels/`). The 6223 constant is a launcher default
that was never checked against the data, so every `frame_coverage_frac` in these JSONs is
slightly *optimistic* (29.2% reported vs 29.8% true). Minor next to the GT-denominator error, but
fix the constant before quoting coverage again. **The GT-based fraction (32.0%) is the one that
matters for AP** — frame coverage and GT coverage differ because covered frames are not a random
sample of GT density.

### 4b-coverage. Padded/anchored windowing — honest AP 0.0531 → 0.1403 (job 7353227, 2026-08-05)

**Test-side only: no retraining, no encoder change, same `best.pt`.** Only the test window
manifests and caches change, so this cannot alter any trained model — it changes what the
metric can *see*, not what the model *is*.

**Root cause of the missing GT, measured (not assumed).** Of the 34.8% of test GT the old
manifests never reached:

| where the missing GT was | GT boxes | % of test GT |
|---|---|---|
| 187 runs SHORTER than one 48-frame window (unreachable under any phase) | 3,379 | **30.2%** |
| tail of long runs, past the last stride-aligned window | 523 | 4.7% |

So the dominant loss was **not** tubelet parity — it was runs too short to hold a window.
Fixed by two opt-in flags in `build_esad_windows.py` (`--pad-short-runs`, edge-replicating a
short run up to 48 frames; `--tail-anchor`, one final window flush with the run end), plus
`--full-gt-label-dir` in `score_esad_detection_ap.py` so GT is read from the label files and
uncovered GT counts as missed detections instead of vanishing.

Measured coverage: **33.0% → 97.1% of test GT** (147 → 734 windows). Verified before spending
export compute: phase-0 default output byte-identical to the cited baseline manifest (slot-0
comparison — the live builder emits M=2 boxes while the cited caches are single-box), every
window exactly 48 frames, 187 padded windows per phase introducing **zero** foreign frames,
box coords in range. 44 unit assertions on the windowing, 9 on the scorer.

| condition | GT scored | AP (covered subset) | **AP (full denominator)** |
|---|---|---|---|
| baseline (the published §4b run) | 3,587 / 11,207 | 0.1796 | **0.0531** |
| **coverage (padded p0+p1)** | 10,619 / 11,207 | 0.1464 | **0.1403** |
| *SARAS-ESAD best submission* | 11,207 | — | *0.1928* |

Per-threshold: AP10 0.0774 → **0.2061**, AP30 0.0611 → **0.1600**, AP50 0.0207 → **0.0547**.

**+0.0872 (2.6×). Gap to the paper closes from 3.63× to 1.37×.**

**Two independent validations landed in this run:**
1. **The baseline re-scored at 0.0531 against the 0.0530 predicted by §4b-corrected's rescale
   identity** — the ledger correction is now confirmed by the code path, not just by algebra.
2. **The denominator trap is visible in the table.** On the covered-subset metric the coverage
   arm looks *worse* (0.1464 vs 0.1796) — the exact signature that produced §4b-followup's
   wrong "coverage made it worse" verdict. Same runs, opposite conclusions, decided purely by
   whether the denominator is held fixed.

**Cost of edge-replicated padding, quantified.** If the newly-covered GT scored at the same
per-GT rate as the old, AP would be 0.1572; actual is 0.1403, so the recovered frames yield
**89% of the old rate**. Padding is mildly OOD (the encoder sees duplicated frames it never saw
in training) and that costs ~11% — real, modest, and far outweighed by reaching the frames at
all. This also means the earlier linear extrapolation to ~0.17 was optimistic; measuring beat
extrapolating.

WBF remains null on both denominators (0.1403 → 0.1405 `wbf_meanconf`; `iou_clustered` 0.1410
is nominally best and still noise-sized).

**Caveats:** single seed; v1/e159 only (the other 3 checkpoints are NOT yet re-scored on this
denominator, so §4b-corrected's cross-checkpoint table is still the covered-subset one);
94.8% GT coverage, not 100%.

### 4b-rare. Rare-class sweep (pos_weight cap × selection metric) — NULL, and it measured the noise floor

**2026-08-05/06, jobs 7353990 + 7354651.** Motivated by a real, verified pair of code defects
(both still true, independent of the result below):
1. `pos_weight_cap: 50.0` **binds on 7 of the 8 rare classes and on ZERO well-supported
   classes** — measured true neg/pos: cls12=1096, cls2=986, cls3=493, cls17=352, cls6=308,
   cls16=237, cls4=157, all clipped to 50; every class with nGT≥50 sits below the cap
   untouched. The cap is a pure rare-class suppressor.
2. `best.pt` is selected on `well_supported_map`, which **excludes classes with <25 val
   positives by construction** — i.e. selection is blind to the 8 classes carrying 8/21 = 38%
   of the reported macro AP.

Swept cap ∈ {50,150,300,600,1200} × selection ∈ {well_supported_map, macro_map}, 10 arms, one
GPU each (fan-out driver `run_esad_sweep.sh`), scored on the §4b-coverage full-denominator
protocol.

| arm | AP_mean | vs base | rare | well |
|---|---|---|---|---|
| cap1200_macro | 0.1839 | +0.0436 | 0.0674 | 0.2555 |
| cap50_macro | 0.1596 | +0.0193 | 0.0308 | 0.2389 |
| **cap50_well (= baseline config, re-run)** | **0.1579** | **+0.0176** | 0.0147 | 0.2460 |
| cap300_well | 0.1465 | +0.0062 | 0.0115 | 0.2295 |
| cap1200_well | 0.1428 | +0.0025 | 0.0300 | 0.2121 |
| baseline (original v1 run) | 0.1403 | — | 0.0163 | 0.2166 |
| cap600_macro | 0.1370 | −0.0033 | 0.0118 | 0.2140 |
| cap600_well | 0.1327 | −0.0076 | 0.0105 | 0.2080 |
| cap300_macro | 0.1289 | −0.0114 | 0.0093 | 0.2026 |
| cap150_well | 0.1262 | −0.0141 | 0.0081 | 0.1989 |
| cap150_macro | 0.1200 | −0.0203 | 0.0132 | 0.1857 |

**⚠ THIS TABLE IS NOT RANKABLE. There is no seed set anywhere in
`train_esad_double_head.py` or the probe YAMLs**, and the sweep accidentally produced two
same-config replicate pairs that measure the run-to-run spread directly:

| config | run A | run B | **spread** |
|---|---|---|---|
| cap1200 + macro | 0.1522 (`rareboth`, job 7353990) | 0.1839 (`sw2`) | **0.0316** |
| cap50 + well | 0.1403 (`v1`, the published baseline) | 0.1579 (`sw2`) | **0.0176** |

**Every between-arm difference in the table is smaller than the identical-config spread.** Per
class the swing is larger still — between the two cap1200+macro replicates, cls1 moved
0.1175→0.4424 and cls12 0.2292→0.3824; between the two cap50+well replicates, cls0 moved
0.4293→0.1846, cls14 0.5415→0.7378, cls5 0.1820→0.3725. Note cls0/cls5/cls14 are
**well-supported** (188/436/1828 instances), so this is not just small-N rare-class jitter —
the whole probe is high-variance, presumably driven by n=2 training surgeries.

**`cap1200_macro`'s apparent +0.0436 win is one class.** cls12 (37 test instances) moved
0.0107→0.3824, worth +0.0177 of macro on its own; of the other 7 rare classes 3 improved
slightly and 4 got worse. The same config's other replicate scored 0.1522. Lottery ticket, not
a lever.

**Verdict: NULL.** No cap or selection setting is distinguishable from noise. The two code
defects above are real and worth fixing on correctness grounds, but there is **no evidence
they improve test AP**. An earlier reading of the `rareboth` arm as "a real +0.0120 gain with
the mechanism confirmed" is **WITHDRAWN** — that delta is inside the 0.0316 same-config spread
for its own config, and against the re-run baseline (0.1579) it is negative.

**This retroactively asterisks every single-run number in §4/§4b/§4c/§4d**, all of which were
unseeded single draws. Same-coverage A/B deltas of ≲0.03 in those sections should be read as
ties. It does NOT touch §4b-coverage's +0.0872 — that is ~3–5× this spread, and structural
(a denominator/coverage change) rather than stochastic.

**Required before any further head-side tuning on this probe:** set a seed, then run
3 seeds × 2 configs rather than 1 seed × N configs. Same GPU cost, interpretable output.
Cross-reference `[[grasp-probe-noise-floor-3map]]` — this project already documented a ~3 mAP
noise floor on a structurally similar cached probe; this is that same floor, re-measured on
ESAD, and it was under-applied here before the replicates surfaced by accident.

### 4b-prod37M-e199. The finished 200-epoch CPT run, 3 SEEDS on the §4b-coverage protocol (2026-08-23)

**First ESAD result for `prod37M_e199`** — the completed 200-epoch / 6000-step / 36.86M-sample
CPT run (`[[prod37m-run-complete]]`, finished 2026-08-21). Frozen encoder, `presence_mode=rep`,
padded/anchored windowing, full-denominator scoring — i.e. the **§4b-coverage protocol**, so
these are directly comparable to that section's 0.1403 and NOT to §4b's covered-subset numbers.

Identical denominators on all three seeds: `gt_full=11207`, `gt_cov=10619` (94.8%),
`frames_scored=5903`.

| variant | s0 | s1 | s2 | **mean** | sd | range |
|---|---|---|---|---|---|---|
| `maxpick` | 0.1707 | 0.1905 | 0.1767 | **0.1793** | 0.0101 | 0.0198 |
| `iou_clustered` | 0.1705 | 0.1908 | 0.1771 | **0.1795** | 0.0103 | 0.0203 |
| `wbf_meanconf` | 0.1717 | 0.1922 | 0.1796 | **0.1812** | 0.0104 | 0.0205 |
| *SARAS-ESAD best submission* | — | — | — | *0.1928* | — | — |
| *v1/e159 frozen, §4e 3-seed retrofit* | — | — | — | *0.1483* | *±0.0149* | — |
| *v1/e159 **FT last-4**, §4b-unfreeze 3-seed* | — | — | — | *0.2005* | *±0.0191* | — |

**Variance context — this is the tightest 3-seed set measured on this probe so far, and that
is itself worth distrusting slightly.** Prior seeded work (§4b-seeded, 2026-08-06) got spreads
of **0.022–0.063** across four configs; §4b-rare's accidental unseeded replicates spread
0.018–0.032. This run's range is **0.0198** — at the low end of both. Two readings are open and
n=3 cannot separate them: either the encoder genuinely produces a more stable head, or these
three draws were lucky. **Assume the pooled ~0.06 worst-case spread from §4b-seeded when
powering any comparison against this number**, not this run's own 0.0101 sd.

**Reading — nominally the best frozen checkpoint on record, but inside the noise.** Against
§4e's 3-seed frozen table this ranks **1st of 8**: 0.1793 vs e159 0.1483 (+0.031), LemonFM
0.1537, meta1b 0.1441, ours1b_e19 0.1302, meta2b 0.1271, SurgeNetXL 0.1162, EndoViT 0.0853.
⚠ The correct comparator is the §4e **3-seed retrofit** value 0.1483 ± 0.0149, NOT the older
single-draw 0.1403 — an earlier version of this entry used 0.1403 and overstated the delta.
+0.031 against combined spreads of 0.0149 and 0.0198 is roughly one spread: **suggestive,
not established.** §4b-seeded's standing rule (*"n=3 cannot resolve a ~0.02 effect against a
~0.06 spread; need ~8–10 seeds or an effect >0.06"*) means this does not yet support a
CPT-progress claim.

**What is notable regardless of the ranking:** frozen 0.1793 already sits close to e159's
**fine-tuned** 0.2005 and to the challenge best submission (0.1928) — i.e. this checkpoint
frozen is approaching what the previous best checkpoint needed unfreezing to reach.

**Expected FT.** §4e shows FT last-4 gains of +0.041 to +0.067 on all four V-JEPA checkpoints
(e159 +0.052, meta2b +0.067, ours1b_e19 +0.053, meta1b +0.041) at ~0% compute overhead. Naively
that puts prod37M_e199 FT at **~0.22–0.25**, which would clear the paper's 0.1928 — but treat
that as a hypothesis to test, not a projection to quote: this probe has burned an extrapolation
before (§4b-coverage's linear estimate said ~0.17, measurement said 0.1403). **Running FT last-4
on this checkpoint is the recommended next step** — highest expected value, essentially free.

Relative to the SARAS-ESAD best submission (0.1928), the 3-seed mean sits ~0.013 below — inside
one seed range. Note the standing caveat from §4b-corrected: comparability to the challenge
number is protocol-dependent and has not been fully verified; treat proximity as a sanity
signal, not a head-to-head.

**WBF remains nominally best and still null** (+0.0019 over maxpick, ~5× smaller than the seed sd)
— consistent with §4b-coverage.

**Provenance.** Run on **Sophia**, not Polaris — polaris-login-01's Lustre client was degraded
that day (`[[eagle-metadata-latency-not-striping]]`) and `module load conda` was broken
site-wide (`[[polaris-conda-module-broken-cray-pe-drift]]`). Sophia runs this probe with zero
env workarounds (`[[esad-probe-on-sophia]]`). Cov caches re-exported there (jobs 176298/176299,
16 GB per phase, ~5 min/phase on one A100); s0/s1 scored from pre-existing `best.pt` (jobs
176302/176303); s2 trained fresh + scored (jobs 176304/176306). s2 landing between s0 and s1
rules out a train-host confound from s2 being the only seed trained on Sophia.

Artifacts: `runs/esad_double_prod37m_e199_frozen_presrep_s{0,1,2}/test_detection_ap_cov_fulldenom.json`.

### 4b-prod37M-e199-FT. Last-4 fine-tuning, 3 seeds — and the presence/box trade that caps it (2026-08-24)

**Full metric set, 3 seeds, one population** (`gt_full=11207`, `gt_cov=10619`,
`frames=5903`, 29,854 box instances — asserted identical on every seed and both arms):

| metric | frozen | **FT last-4** | Δ | sd (fz / ft) |
|---|---|---|---|---|
| **detection AP_mean** (maxpick, full-denom) | 0.1793 | **0.2035** | **+0.024** | 0.010 / 0.016 |
| presence mAP† | 0.2507 | **0.3159** | **+0.065** | 0.005 / 0.002 |
| mean IoU† | 0.4664 | 0.4426 | −0.024 | 0.016 / 0.012 |
| box mAP@50† | 0.4810 | 0.4209 | **−0.060** | 0.035 / 0.031 |

† oracle-gated / probe-internal — never compare these to a challenge AP (see §4b-unfreeze).
Per-seed detection AP: frozen 0.1707/0.1905/0.1767; FT 0.1849/0.2146/0.2108.

**★ Fine-tuning buys FINDING and costs LOCALISING.** Presence +0.065 is 13-30× its
seed sd — unambiguous. Box mAP@50 −0.060 is ~2× its sd — a real decline. Detection
AP needs both, which is why a **1.5× val_well_map gain** (0.272 → 0.419) converts to
only **+0.024** test detection AP. This reproduces e159's published FT split
(presence +0.093, detection +0.060, IoU −0.014, box map50 −0.047) on a second
checkpoint, so it is a property of the method rather than a fluke draw.

**~~Consequence: the BOX HEAD is the binding constraint, not the encoder.~~ ← RETRACTED
2026-08-24, see §4b-aug.** The reading was that `train_bce` collapses to ~0.002 by ep19 while
`train_box` stalls at 0.24-0.27, so the box head must be capacity-limited. **Wrong: it was
OVERFITTING.** Adding train-time augmentation (never wired into the FT trainer until 2026-08-24)
lifts boxAP50† from 0.4209 back to **0.5101, above the frozen 0.4810**, and AP50 from 0.0680 to
0.1127. The presence/box trade documented in the table above is real and reproduces on two
checkpoints — but it is a symptom of an unregularised box head, not a structural ceiling.

**Cross-checkpoint: FT gain is inversely proportional to frozen score.**

| checkpoint | frozen | FT last-4 | gain |
|---|---|---|---|
| **prod37M_e199** | **0.1793** | **0.2035** | **+0.024** |
| surgical 2B e159 | 0.1483 | 0.2005 | +0.052 |
| raw Meta 2B | 0.1271 | 0.1936 | +0.067 |
| raw Meta 1B | 0.1441 | 0.1855 | +0.041 |
| surgical 1B e19 | 0.1302 | 0.1832 | +0.053 |
| *SARAS-ESAD best submission* | — | *0.1928* | — |

All five land in **0.18-0.21** regardless of where they started. That band reads as a
**task ceiling** (n=2 training surgeries, 2,468 windows), not a checkpoint ranking.
prod37M_e199 FT is nominally the best number on record — above e159 FT and the paper's
0.1928 — but **+0.030 over e159 does not clear the n=3 noise bar** (§4b-seeded: spreads
0.022-0.063; need ~8-10 seeds or an effect >0.06). Report it as a nominal lead.

**Provenance.** Sophia, 4 GPUs/seed, **global batch held at 8** (bs=2 × accum=1 ×
4 ranks, vs the baseline's bs=2 × accum=2 × 2 ranks) so step count and LRs are
unchanged and the arms stay comparable. Measured unfreeze-depth feasibility on
A100-40GB (1 epoch, grad-ckpt on): N=4 12.8 GB, N=8 15.0 GB, N=16 19.5 GB, N=48
**37.8 GB** — memory is not the limit, wall time is (N=48 is 2.1× and hits the
walltime cap). See `[[esad-ft-trades-boxes-for-presence]]`.

### 4b-aug. Train-time augmentation on the FT path — the largest lever since coverage (2026-08-24)

**`train_esad_unfreeze.py` never passed augmentation to its dataset.** `ESADWindowDataset`
has accepted `augment=`/`aug_seed=` all along (fixed per-window 90-100% area crop + colour
jitter, boxes remapped into the crop frame, horizontal flip deliberately excluded because box
semantics under flip were never verified), but the FT trainer constructed it with defaults — so
**every FT run on record before this date trained on identical frames every epoch**. Adding
`--augment` (TRAIN only; val/test stay deterministic so the metric is unchanged) is a one-line
change with the largest measured effect since the coverage fix.

Verified sound BEFORE judging the result, so a null would have been a real finding rather than a
broken transform: deterministic-vs-augmented pixel mean|Δ| = 0.073, seed0-vs-seed1 = 0.207, and
box targets ARE remapped (mean|Δ| 0.0068 on real boxes).

**prod37M_e199, 3 seeds, one population (`gt_full=11207`, `gt_cov=10619`, `frames=5903`):**

| arm | AP_mean | AP10 | AP30 | AP50 | presence† | meanIoU† | boxAP50† |
|---|---|---|---|---|---|---|---|
| frozen | 0.1793±0.010 | 0.2496 | 0.2082 | 0.0801 | 0.2507 | 0.4664 | 0.4810 |
| FT last-4 | 0.2035±0.016 | 0.3185 | 0.2239 | 0.0680 | 0.3159 | 0.4426 | 0.4209 |
| **FT last-4 +aug** | **0.2319±0.024** | **0.3262** | **0.2568** | **0.1127** | **0.3248** | **0.4765** | **0.5101** |
| *SARAS-ESAD best submission* | *0.1928* | *0.2763* | *0.2205* | *0.0816* | — | — | — |

**★ Augmentation REVERSES the FT localisation penalty.** §4b-prod37M-e199-FT showed FT trading
box quality for presence (boxAP50† 0.4810 → 0.4209). With augmentation, boxAP50† recovers to
**0.5101 — above the frozen baseline** — while the presence gain is kept (0.3248). The same
signal appears independently in the per-threshold detection columns: **AP50 goes 0.0801 frozen
→ 0.0680 FT (a real decline) → 0.1127 +aug**, i.e. the strictest, most localisation-sensitive
threshold is exactly where FT hurt and augmentation helps most. Two independent metric families,
same conclusion.

**This retracts "the box head is the ceiling"** (stated earlier in §4b-prod37M-e199-FT's reading).
The box head was not capacity-limited — it was **overfitting**, and the regulariser to fix it had
simply never been wired in. `train_bce` at ep19 is **3.4× higher** with augmentation (0.0125 vs
0.0036), i.e. it demonstrably prevents the memorisation collapse on 2,468 windows from n=2
surgeries.

⚠ **Do not read the early epochs.** At ep0-2 the augmented and baseline train curves are
indistinguishable (0.821/0.498/0.247 vs 0.874/0.486/0.255); the divergence only appears late, as
the un-augmented model collapses. An epoch-2 reading of this arm would have called it null.

**+0.053 over frozen clears the §4b-seeded noise bar** (spreads 0.022-0.063, need >0.06 or 8-10
seeds) — marginally, and it is the only arm all campaign to do so. The +0.028 over FT alone does
NOT clear it.

⚠ **frozen+aug is NOT epoch-matched to frozen — do not read its curve epoch-wise.** The frozen
trainer reads *cached* features, so augmentation cannot be a runtime flag the way it is on the FT
path; it needs a **second, augmented cache**, which `--train-cache` (`action=append`) then
**combines with the original**. Manifests confirm it: `cache_prod37m_e199/train` `num_samples=309`
per rank ×8 = 2,468 windows, `cache_prod37m_e199_augtrain/train` 617 ×4 = 2,468 **more**. So
frozen+aug sees **4,936 windows/epoch, 2× the baseline**, and "epoch N" is 2N epochs of gradient
steps. The naive epoch-matched read is **+0.104 val_macro_map at ep16** — mostly steps.

On a **samples-seen** axis, over the range where both arms have data:

| samples | frozen+aug | frozen | Δ |
|---|---|---|---|
| 24,680 | ep4 0.178 | ep9 0.212 | −0.034 |
| 34,552 | ep6 0.167 | ep13 0.251 | −0.084 |
| 44,424 | ep8 0.250 | ep17 0.252 | −0.002 |
| 49,360 | ep9 0.277 | ep19 **0.236** | **+0.042** |

Augmentation is *behind* for most of the run and only crosses over at the end — the same
late-divergence shape §4d flagged, and another reason not to judge a regulariser early. The
crossover is not purely a step effect: at 49,360 samples the **baseline has already peaked
(ep15 0.264) and is declining**, while frozen+aug is still climbing (and reaches 0.328 by ep16,
past the baseline's data). But the honest step-matched margin inside the overlap is **+0.014**,
not +0.104.

**These are two different treatments sharing one name.** FT `--augment` transforms the *same*
2,468 windows in place — steps/epoch unchanged, a pure regulariser. Frozen "+aug" *doubles the
dataset* with a second fixed augmented copy — a data-quantity change with one extra view per
window. Their numbers are not interchangeable, and a combined "augmentation helps" claim across
both paths would be comparing different interventions.

**Significance, quantified (2026-08-24, `collect_esad_arms.py`).** The sentence above compares a
*mean* delta to a raw *per-seed spread*, which is not a test — the mean's error shrinks with n, the
spread does not. Redone as two-sided t-tests (Welch, and paired-by-seed since seed N is the same
init + data order in every arm):

| comparison | Δ AP_mean | Welch p | paired p | seeds improved |
|---|---|---|---|---|
| FT+aug vs frozen | +0.0526 | 0.049 | **0.034** | 3/3 |
| FT last-4 vs frozen | +0.0241 | 0.107 | 0.052 | 3/3 |
| FT+aug vs FT last-4 | +0.0284 | 0.178 | 0.185 | **2/3** |

So the two verdicts survive but with better-calibrated language: **aug-over-frozen is significant
(p=0.034 paired), not "marginal"**, and **aug-over-FT remains a non-result** — and the per-seed view
shows *why*. The paired deltas are **+0.0416 / +0.0438 / −0.0001**: two seeds move together by
~+0.042 and the third does not move at all. That is not a uniform shift with noise on top, so the
mean 0.0284 is a poor summary of it either way. Whatever augmentation is doing here, one of three
initialisations is immune to it — which is itself a lead the mean would have hidden, and a reason
the follow-up arms (per-epoch, strong, pw200) are worth running even though the headline is a tie.

⚠ **A t without its df is not a verdict.** At n=3 the paired df is 2, where t=2.0 is p=0.18. An
earlier draft of this block thresholded on |t| ≥ 2 and would have called the aug-over-FT tie a win.

**Results are now collected, not transcribed** (`collect_esad_arms.py`, on Sophia in the probe dir).
It discovers arms by globbing `runs/esad_double_*_s{N}`, reads AP from the scorer JSON, **asserts
`gt_full=11207 / frames=5903` on every cell and drops mismatches loudly** rather than averaging them
in, and writes `results_index.json` so the next reader parses results instead of re-deriving them.
This replaces `collect_ft.py`, which carried its numbers as a hardcoded literal dict — every new arm
needed a collector edit, and any stale entry silently produced a wrong table.

### 4b-augext. Does the augmentation gain hold across checkpoints? — V-JEPA ARMS RESOLVED: NULL (2026-08-25)

§4b-frozctl measured `frozen_augonly` on **one** checkpoint (prod37m_e199) and found +0.0415 AP
= 4.0 baseline sd. If that is a property of *augmentation* it should reproduce on other
backbones; if it is a property of *this checkpoint* it will not. Until it is measured on more
than one backbone, every "ours vs theirs" frozen comparison in this ledger is unfair in our
favour — our arm would have the augmentation and the externals would not.

> 🛑 **Superseded premise (2026-08-25 ~04:10, after launch).** That +0.0415 was **one seed**.
> Seed 1 scored 0.1631, dropping the arm to 0.1893 at n=2 and the delta to **+0.0100** — inside
> the baseline's own sd. See the third correction in §4b-frozctl. This campaign was launched to
> ask whether a large effect *generalises*; it is now asking whether the effect *exists*. The
> design is unchanged and is the right one either way — 3 seeds × 6 backbones is exactly the
> power that was missing — but do not treat +0.0415 as a target these arms must reproduce.

#### RESULT (V-JEPA family, 2026-08-25 ~04:45): the effect does not exist

The three V-JEPA campaign jobs finished. Comparing each `frozen_augonly` arm against **its own**
`frozen_presrep` control — which is the comparison this campaign was designed to make, and *not*
the collector's `vs base` column (that is against `ft_last4` and answers a different question):

| backbone | `frozen_augonly` | `frozen_presrep` | Δ |
|---|---:|---:|---:|
| `prod37m_e199` | 0.1893 (n=2) | 0.1793 ± 0.0101 (n=3) | +0.0100 |
| `meta2b` | 0.1362 ± 0.041 (n=3) | 0.1278 ± 0.012 (n=3) | +0.0084 |
| `meta1b` | 0.1432 (n=2) | 0.1459 ± 0.022 (n=3) | −0.0027 |
| `ours1b_e19` | 0.1545 (n=2) | 0.1339 ± 0.011 (n=3) | +0.0206 |

Mean **+0.0091**, one of four negative, and between-seed spread on these arms is ~0.05 — **five
times the effect**. **No effect is established for frozen augmentation.** The retracted +0.0415
does not reproduce on any backbone including the one it came from. `ft_aug` (0.2319 ± 0.024,
n=3) is unaffected and remains the best arm on record; the FT augmentation result in
§4b stands.

**The treatment is doing something — it just isn't helping.** Sorting all 27 frozen seeds by
final train `bce` puts `augonly` at the high-loss end and `presrep` at the low end (8 of the 12
worst are augonly; 7 of the 9 best are presrep). That ranking is not an artifact of unequal run
lengths: read at a **fixed epoch 5**, augonly's train BCE exceeds its own presrep control for
all four backbones (meta1b 0.92/0.89/0.94 vs 0.64/0.62/0.54; meta2b 0.77/0.80/0.94 vs
0.56/0.73/0.64; ours1b 0.82/0.86/0.80 vs 0.67/0.58/0.62; prod37m 0.49/0.69/0.75 vs
0.49/0.55/0.52). Higher train loss under augmentation is the regulariser working as intended. It
does not convert to test AP here.

**Two seeds failed to converge outright, both in the augonly arm, with one signature: the
presence head collapses while the box head is fine.** `meta2b_s2` — final `bce` 0.820 vs s0's
0.407, `val_wellmap` 0.103 vs 0.256, yet IoU 0.458 vs 0.466. That is the same shape as
`prod37m_s1` in the §4b-frozctl retraction. `meta1b_s0` is worse — `bce` 0.913, `wellmap` 0.078,
**IoU 0.210** (both heads failed) — and early-stopped at epoch 6. So the wide frozen-arm spread
is **not measurement noise**: it is a real bimodality in whether the presence head trains at all,
and augmentation raises the rate of that failure. Any future frozen arm should report final
`bce`/`val_wellmap` next to AP so a collapsed head is visible rather than averaged in.

**Scope correction — the instability is the augonly arm's, not frozen probes' in general.** The
third correction in §4b-frozctl warned that "any frozen delta under ~0.05 is unresolved at n=3."
Measured across all 27 frozen seeds on disk, that is too broad: **4 of 12 `augonly` seeds end
with train BCE above 0.48 (max 0.913), while all 15 `presrep` seeds fall in 0.207–0.437** — a
tight, bounded distribution with no collapses at all. The un-augmented frozen columns elsewhere
in this ledger (notably §4i, sample sd 0.0007–0.0258 over 3 seeds) are therefore **not** put in
doubt by this campaign and stand as published. It is specifically the *augmented* frozen arm
that requires n≥3 plus a per-seed convergence check before any delta is read from it.

**Still pending:** the three ext arms (`lemonfm`/`snx`/`endovit`, jobs 7555011/14/15) were queued
on Polaris `preemptable` when this was written. They extend the table but cannot change the
V-JEPA verdict above.

**Six checkpoints, 3 seeds each, all `augonly`.** Launched on Polaris `preemptable` as jobs
7555007/9/10/11/14/15.

| checkpoint | family | run tag | baseline it mirrors |
|---|---|---|---|
| `meta2b` | vjepa | `frozen_augonly` | `meta2b_frozen_presrep` |
| `meta1b` | vjepa | `frozen_augonly` | `meta1b_frozen_presrep` |
| `ours1b_e19` | vjepa | `frozen_augonly` | `ours1b_e19_frozen_presrep` |
| `lemonfm` | ext | `t1_augonly` | `lemonfm_t1` |
| `snx` | ext | `t1_augonly` | `snx_t1` |
| `endovit` | ext | `t1_augonly` | `endovit_t1` |

**`augonly`, not `aug` — the arm choice is the whole point.** `aug` passes `--train-cache`
twice (deterministic + augmented). Since that flag is `action="append"`, it **doubles**
windows/epoch from 2,468 to 4,936, so its delta conflates augmentation with 2× data
([[frozen-aug-is-a-different-treatment]]). `augonly` passes only the augmented cache: identical
window count, identical optimiser steps, the sole difference is that the pixels were
crop/jitter-perturbed at export. That is the arm whose delta is attributable, so it is the one
replicated across all six.

**The two families are NOT scored the same way, and must not be.** They inherit different
baselines and mixing them would produce a number comparable to nothing:

* **vjepa** (meta2b / meta1b / ours1b_e19) — mirrors `frozen_presrep`: rep-mode presence
  overrides at load time, features from `cache_<n>`, scored on `cov_p0`+`cov_p1` and read from
  `combined_not_isolated` (5,903 frames, `gt_full=11207`).
* **ext** (lemonfm / snx / endovit) — mirrors `<n>_t1`: tubelet-1 manifests, `cache_<n>_t1`,
  **no** presence override, scored fairness-masked to V-JEPA's covered frame stems (the image
  arms reach ~100% test coverage at tubelet=1, so the unmasked number is a secondary column and
  never the comparison column).

**Config fairness was verified before spending the compute**, not asserted. Flattening all seven
probe YAMLs and diffing against prod37m leaves exactly three differing keys —
`model_kwargs.embed_dim`, `head_kwargs.tokens_per_clip`, `head_kwargs.temporal_tokens` — and all
three are **backbone-determined, not free choices**: `embed_dim` is the encoder's output width,
and the token counts follow each backbone's patch grid (16/48 for the image arms, 8/24 for the
V-JEPA tubelet-2 arms). Every tunable knob already matches across all seven: `w_presence 1.0`,
`w_box 1.0`, `w_l1 5.0`, `w_giou 2.0`, `pos_weight_cap 50.0`, `batch_size 8`, `num_epochs 20`,
`warmup_epochs 2.0`, `early_stop_patience 6`, `lr 1.0e-3`, `weight_decay 0.05`,
`presence_num_layers 4`, `presence_dropout 0.2`, `num_segments 3`. The export YAMLs differ only
in checkpoint path and encoder arch.

**Why Polaris and not Sophia.** Sophia's `by-gpu` allows `max_run=5` / `max_queued=20` per
*project* and the FT campaign already held 17 of those slots — six more jobs would have been
refused by the server, not queued. Polaris `preemptable` allows 10 concurrent single-node jobs
at Priority 155 with a 72 h walltime, and Polaris is `force_exclhost` so each job gets all 4
A100s: 4-way parallel export, then 3 seeds in parallel, then 3 scores in parallel. Note `debug`
is **not** usable here — its 1 h cap already walltime-killed two of these runs at epoch 18/20
(7551194, 7551483). See [[polaris-preemptable-is-the-capacity-queue]].

**Two gates the launcher enforces, both from failures already on record:**

1. **Export completeness is summed across shards, not per-rank.** `export_esad_cache.py`
   resolves world size from `(PMI_SIZE, PMIX_SIZE, OMPI_COMM_WORLD_SIZE, PALS_SIZE, WORLD_SIZE)`
   *in that order*, and each rank writes its own manifest marked `completed`. So a stray
   `PMI_SIZE` — or one dead rank — yields a cache that passes every per-rank check while holding
   a fraction of the training set, under a full-size name. The gate sums `num_samples` over all
   `rank_*/manifest.json` and demands it equal the window count (2,468).
2. **Epoch count, not `best.pt`, is the completion signal.** `best.pt` is written from epoch 0
   onward, so gating on it publishes a killed 17/20-epoch run as finished — which is exactly how
   a "20 epochs in 37 minutes" run got past review earlier ([[wall-clock-hides-a-resume]]).
   Training gates at `>=20` epochs in `log_r0.csv`; scoring gates at `>=19`.

The vjepa arms additionally assert `presence override ACTIVE` in the train log — without it a run
trains on *union* labels under a `presrep` name, a plausible answer to the wrong question.
(Confirmed present in all nine vjepa seeds by reading `stdout.log` directly, since the six
in-flight jobs carry a spooled copy predating the guard fix in `dab1de7`. Each line also reports
`2468 windows`, which is the `augonly` signature — `aug` would read 4,936.)

⚠ **Gate 2 as written skips converged seeds, and the in-flight six carry it.** `>=19` conflates
"finished" with "ran long enough". With `early_stop_patience: 6` against `num_epochs: 20`, a seed
that converges can legitimately stop at epoch 14 — and **8 of the 21 completed frozen runs on
record ended at ep 14–18** (`meta2b_presrep_s0`=14, `v1_presrep_s1`=16, `prod37m_presrep_s1`=17).
Those were scored only because the older presrep launcher had no epoch gate. Here the gate would
skip them via `return 0`: no failure, no result, roughly a third of the campaign silently absent.

Fixed in `2903465` — the gate now accepts `>=19 epochs` **or** the trainer's
`[TRAIN] early stop at epoch N` line (`train_esad_double_head.py:732`), which a walltime kill
never prints, so convergence passes and a kill still does not. Falsified over four cases before
use: converged@14 → SCORE, killed@14 → SKIP, full@20 → SCORE, missing log → SKIP (fails *closed*).
Because PBS spooled the pre-patch script, **the six in-flight jobs are unaffected by the fix** —
any early-stopped seed among them trains and goes unscored. This is recoverable rather than lost:
`best.pt` is written regardless, and `~/.esad_catchup_score.sh` rescores exactly those runs
(`best.pt` present, 19 epochs *or* an early-stop line, no result JSON, `latest.pt` cold ≥10 min).
So **a seed missing from the table below means "not yet scored", not "failed"** — check the
catch-up pass before reading anything into it. See
[[early-stop-is-completion-not-truncation]].

**This played out exactly as predicted, and cost three seeds — all recovered.** When the jobs
ended, `meta1b_s0` (early stop @6) and `ours1b_e19_s2` (early stop @17) were unscored by the
spooled pre-patch gate, as expected. A **third**, `prod37m_e199_s2`, was missed for a different
reason worth recording: it ran all 20 epochs — every row present in `log_r0.csv`, best at epoch
17 — but its `stdout.log` was **never written at all**. The gate greps that file for the
early-stop line, and `grep -q ... 2>/dev/null` on a missing file fails **closed**: safe, but it
discards a finished seed ([[grep-guard-on-a-file-nobody-writes]]). Fixed in `8f9a8d9` with a
second witness derived from the CSV alone — the trainer breaks when
`(last_epoch − best_epoch) >= patience`, so that arithmetic is itself proof it stopped on its
own rather than by kill. A seed now scores if it ran ≥19 epochs, **or** printed the early-stop
line, **or** shows the converged signature in its own CSV. Falsified before use: accepts
`meta1b_s0` (6−0) and `ours1b_e19_s2` (17−11), rejects a synthetic 10-epoch kill with best at 9.
The catch-up scorer also omitted `prod37m_e199` from its backbone list entirely (it was written
for the six campaign checkpoints); patched, dry-run to confirm it selected those three and no
live run, and submitted as job 7555181.

### 4b-frozctl. Is the frozen "+aug" gain augmentation, or just more windows? — SEED 0 ONLY, preliminary (2026-08-25)

**Status: 1 of 3 seeds. Do not cite these deltas as a final result** — but the seed-0 detection
AP is now measured, and it is much larger than the probe-internal metrics suggested.

**★ UPDATE 2026-08-25 02:05 — the deciding metric now exists for `frozen_augonly` s0, via an
early 1-GPU score pass (job 176612).** The original text below said the deciding metric "does not
exist for either arm yet," which was true of the *chained* score passes: those run only after all
three seeds finish, and the 4-GPU arms are queued behind a Thu Aug 27 estimate. But scoring needs
**one** GPU, and 1-GPU jobs place on Sophia in seconds
([[one-gpu-jobs-place-while-quads-wait]]). Gated at `>=19` epochs so it cannot score a
mid-training seed.

**Full-denominator detection AP, canonical population** (`gt_full=11207`, `frames=5903`,
node `combined_not_isolated`, fusion `wbf_meanconf`):

| arm | seed | AP_mean | AP10 | AP30 | AP50 |
|---|---|---:|---:|---:|---:|
| `frozen_presrep` (baseline) | s0 | 0.1717 | 0.2417 | 0.2007 | 0.0727 |
| | s1 | 0.1922 | 0.2635 | 0.2211 | 0.0921 |
| | s2 | 0.1796 | 0.2534 | 0.2099 | 0.0757 |
| | **mean** | **0.1812 ± 0.0103** | | | |
| **`frozen_augonly`** | **s0** | **0.2227** | **0.2901** | **0.2510** | **0.1270** |

**+0.0415 over the baseline mean = 4.0 baseline sd**, and +0.0305 over the *best* baseline seed.
It also clears the **un-augmented** fine-tuned last-4 mean (0.2035). And it gains at **all three
IoU thresholds together** (+0.048 / +0.045 / +0.049 vs the baseline mean), the signature of a
genuine improvement rather than the presence-vs-localisation trade that FT shows. One seed is
still one seed, but 4 sd is well outside where a single draw normally lands.

⚠ **Correction (same night, 02:45).** The line above originally read "a frozen probe beating
fine-tuning, which no other arm in this campaign has done." That is **wrong**, and the error is
worth keeping visible: I compared the new frozen arm against `ft_last4` (0.2035) without checking
whether the *augmented* FT arm had finished. It had. `ft_aug` scores **0.2319 ± 0.024 over three
seeds** (paired p = 0.034), which is **above** `frozen_augonly`. The correct statement is that
augmentation helps both paths, and FT+aug is still the best arm on record:

| arm | n | maxpick AP | wbf AP |
|---|---:|---:|---:|
| `frozen_presrep` (baseline) | 3 | 0.1793 ± 0.0101 | 0.1812 ± 0.0104 |
| `frozen_aug` | 1 | 0.2084 | 0.2129 |
| `ft_last4` | 3 | 0.2035 ± 0.0162 | 0.2048 ± 0.0147 |
| `frozen_augonly` | 1 | 0.2155 | 0.2227 |
| **`ft_aug`** | **3** | **0.2319 ± 0.0243** | **0.2323 ± 0.0249** |

What survives: frozen+augonly beats the frozen baseline by ~4 sd, and beats *un-augmented* FT —
i.e. augmentation buys more on this probe than unfreezing four blocks does. That is still a real
and useful finding; it is just not "frozen beats fine-tuning."

---

🛑 **THIRD CORRECTION, 2026-08-25 ~04:10 — seed 1 lands and the effect does not survive it.**
Everything above this line is an **n=1** result. `frozen_augonly` **s1 = 0.1631 maxpick**
(0.1674 wbf), against s0's 0.2155. Two seeds of the same arm, 0.0524 apart:

| arm | n | maxpick AP | delta vs frozen baseline |
|---|---:|---:|---:|
| `frozen_presrep` (baseline) | 3 | 0.1793 ± 0.0101 | — |
| `frozen_augonly` **s0 only** | 1 | 0.2155 | **+0.0362** ← what was published |
| `frozen_augonly` **s0+s1** | 2 | **0.1893** | **+0.0100** |

**The "+0.0415 = 4.0 baseline sd" headline was a single lucky draw.** At n=2 the delta is
+0.0100 — inside the baseline's own sd (0.0101) — and the between-seed spread (0.0524) is
**five times the effect**. Nothing here is significant, and with n=2 nothing here is testable.

**This is not a scoring artifact; s1 genuinely trained worse.** Both seeds ran the full 20
epochs, but the presence head never converged on s1: final-epoch `bce = 0.4515` vs s0's
`0.2163`, and `val_wellmap = 0.2024` vs `0.3403`. The AP gap is downstream of a worse head, so
the variance is in the *training*, not the measurement. That also means seed variance on this
arm is far larger than the ±0.0101 the 3-seed baseline suggested, so **any frozen-arm delta
below ~0.05 needs 3 seeds before it means anything.**

**What this does to the campaign in flight:** §4b-augext (six checkpoints × 3 seeds) was
launched to test whether a +0.0362 effect replicates across backbones. The effect it was
chasing may not exist. The campaign is still the right experiment — it is now *powered*
(3 seeds/arm, 6 backbones = the replication this needed from the start) and it will answer the
question either way. But its framing changes from "does the gain generalise?" to "is there a
gain at all?" Do not read the earlier sections as an established baseline it must beat.

`ft_aug` (0.2319 ± 0.0243, n=3) is unaffected and remains the best arm on record.

**The lesson, again and more expensively:** this is the *second* correction to the same claim in
one night. The first ([[run-the-collector-before-the-verdict]]) was comparing against a
hand-picked subset; this one is publishing a delta from a single seed against a 3-seed baseline.
A 1-seed arm has no error bar — "4 sd above baseline" measures the *baseline's* spread and says
nothing about the arm's own. The correct move at n=1 was to state the number and withhold the
comparison until n=3. See [[one-seed-has-no-error-bar]].

---

**Second correction, methodological.** The table above reports `wbf_meanconf`, but
`collect_esad_arms.py` — the repo's own collector, and therefore every other AP number in this
ledger — uses **`maxpick`** as the headline fusion. Both fusions rank all five arms identically
here (see the table), so nothing downstream changes, but quoting a non-default fusion for one
section invites exactly the apples-to-oranges comparison that produced the error above. Prefer
`maxpick` unless there is a stated reason not to.

**The general lesson:** run the collector before writing the verdict. It reads every arm on
record and prints the significance columns; I had the numbers for `ft_aug` on disk while writing
a claim that contradicted them. A comparison against a hand-picked subset of arms is not a
ranking. See [[verify-the-probe-before-the-hypothesis]].

⚠ **Read the right node.** These JSONs carry four population nodes. `source0_only` scores only
**3,467** frames and gives ~0.10 for the same checkpoints; the canonical
`combined_not_isolated` / `coverage_isolated` nodes score **5,903** and reproduce the ledger's
published 0.1707/0.1905/0.1767 → 0.1793 exactly. I first read `source0_only` here and got numbers
that looked catastrophically low. Always check `num_frames_scored == 5903` before comparing.
Also note `ap_by_iou_threshold` is the **threshold list** `[0.1,0.3,0.5]`, not per-threshold AP —
the AP breakdown is `nanmean(per_class_ap_by_iou, axis=0)`.

The probe-internal `test_summary.json` table further below is retained because it shows why this
needed the real metric: on those columns the two aug arms looked nearly tied and split
oppositely across metrics. On detection AP the picture is much cleaner.

**The design.** §4b-aug showed that adding an augmented cache to the *frozen* probe helps. But
`--train-cache` in `train_esad_double_head.py` is `action="append"`, so passing clean + augmented
does **not** swap the data — it **doubles** it (2,468 → 4,936 windows/epoch). The "+aug" arm
therefore confounds two treatments: augmentation, and 2× the gradient steps per epoch. Job
**176587** (`frozen_augonly`) is the missing control: the augmented cache **alone**, 2,468
windows/epoch, matched to the baseline's window count.

| arm | job | train cache | windows/ep |
|---|---|---|---|
| `frozen_presrep` (baseline) | — | clean | 2,468 |
| `frozen_augonly` (control) | 176587 | **aug only** | 2,468 |
| `frozen_aug` | 176588 | clean **+** aug (appended) | **4,936** |

**Seed-0 test metrics** (`test_summary.json`, oracle-gated / probe-internal — these are *not*
challenge AP and must never be compared to one, per §4b-unfreeze):

| metric | baseline s0 | `augonly` s0 | `frozen_aug` s0 |
|---|---|---|---|
| macro mAP† | 0.3187 | **0.3506** | 0.3345 |
| well-supported mAP† | 0.4135 | **0.4144** | 0.3912 |
| mean IoU† | 0.4786 | 0.4935 | **0.5016** |
| box mAP@50† | 0.5087 | 0.5573 | **0.5641** |
| best epoch | 15 | 17 | 16 |

**Reading — leaning "it is the augmentation, not the extra windows," and inside the noise.**
`augonly` matches or beats `frozen_aug` on both mAP columns while training on **half** the
windows, and beats the baseline on all four. If the §4b-aug gain had come from the extra 2,468
windows, `augonly` should have sat at the baseline; it does not. But the margin is small and
**one seed cannot carry it**: the baseline's own 3-seed detection-AP spread is
0.1014 / 0.1126 / 0.1041 (range **0.0112**), and §4b-seeded's standing rule is that n=3 cannot
resolve a ~0.02 effect against a ~0.06 pooled spread. A one-seed gap of 0.03 macro mAP is not
evidence. **Verdict deferred to the 3-seed detection AP.**

**A trap this table avoids.** `frozen_aug` wins IoU and box mAP@50 while `augonly` wins macro
mAP — reporting either column alone would have produced opposite conclusions. This is the same
presence-vs-localisation split §4b-prod37M-e199-FT documented, and the same reason
[[val-macro-f1-not-a-map-proxy]] exists. The arms are ranked on detection AP, which needs both,
or they are not ranked.

**Timing, measured not estimated** (from the `[TRAIN] ep{N} … ({secs}s)` line that survives the
launchers' `| tail -3`):

| arm | s/epoch | note |
|---|---|---|
| 176587 `frozen_augonly` | **213.6** | one cache |
| 176588 `frozen_aug` | **611.9** | 2× windows accounts for most of the 2.9×; co-tenancy on gpu-01 the rest |

⚠ **`frozen_aug` s0 finished in 37 min, which does *not* contradict 611.9 s/epoch — it resumed.**
Its `best.pt` is dated Aug 24 13:30, from the run the maintenance window killed; 176588 picked it
up near epoch 16 and ran ~3.6 epochs. `augonly` s0 was fresh: 20 epochs in 87 min at 213.6 s/ep.
Both readings are consistent once the resume is accounted for. Anyone costing these arms from
wall-clock alone would have concluded `frozen_aug` was 5× faster than it is.

**Continuations are held and correctly sized.** Neither arm fits three seeds in its own wall, so
each has an `afterany` successor: **176598** (4 h, on 176587) and **176609** (7 h, on 176588,
resubmitted from an undersized 176599 because `qalter` is broken site-wide,
[[qalter-broken-on-sophia]]). Resuming is safe: the trainer persists `epochs_no_improve` /
`best_metric` in `latest.pt` and rewrites `best.pt` only on strict improvement, so an
early-stopped seed re-enters and re-stops without damaging its checkpoint.

Artifacts: `runs/esad_double_prod37m_e199_frozen_{augonly,aug}_s{0,1,2}/`.

### 4b-ens. Cross-seed ensembling on the FINE-TUNED arms — beats the seed MEAN 6/6, and the gain is localization (2026-08-25)

Extends §4f's frozen seed-ensemble to the fine-tuned path. The reason this needed a separate
scorer (`score_esad_seed_ensemble_ft.py`, driver `ens_ft.sh`) is that **FT seeds do not share a
feature cache**: each seed tuned its own encoder, so each has its own test cache. Fusion is
therefore done in *label* space — the same `fusion.py` machinery that already consolidates
overlapping-window votes, since a cross-seed vote is structurally identical to a cross-window one.
Jobs **176554** (`ft_aug`, 7827.9 s) and **176559** (`ft_last4`, 7755.9 s), both rc=0. Population
asserted identical on every cell: `gt_full=11207 / gt_cov=10619 / frames=5903`.

| arm | fusion | seeds (s0/s1/s2) | mean ± σ | best seed | **ensemble** | vs mean | vs best |
|---|---|---|---|---|---|---|---|
| `ft_aug` | maxpick | 0.2265 / 0.2585 / 0.2107 | 0.2319 ± 0.0243 | 0.2585 | 0.2460 | **+0.0141** | −0.0125 |
| `ft_aug` | wbf_meanconf | 0.2286 / 0.2588 / 0.2094 | 0.2323 ± 0.0249 | 0.2588 | **0.2498** | **+0.0175** | −0.0090 |
| `ft_aug` | iou_clustered | 0.2264 / 0.2587 / 0.2093 | 0.2314 ± 0.0251 | 0.2587 | 0.2492 | **+0.0177** | −0.0095 |
| `ft_last4` | maxpick | 0.1849 / 0.2146 / 0.2108 | 0.2035 ± 0.0162 | 0.2146 | 0.2106 | **+0.0072** | −0.0040 |
| `ft_last4` | wbf_meanconf | 0.1879 / 0.2146 / 0.2120 | 0.2048 ± 0.0147 | 0.2146 | **0.2274** | **+0.0226** | **+0.0128** |
| `ft_last4` | iou_clustered | 0.1852 / 0.2152 / 0.2116 | 0.2040 ± 0.0164 | 0.2152 | 0.2176 | **+0.0136** | +0.0024 |

**Against the seed mean: 6/6 positive, +0.0155 average (range +0.0072…+0.0226).** That is the
comparison that corresponds to a real decision — "train three seeds and fuse them" versus "train
one seed and take what you get." It reproduces §4f (frozen, 4/4 positive) and
[[sar-seed-ensemble-result]] on a third training regime.

**Against the best seed: 2/6.** This is the number that looks like a reversal, and it is not one.
Best-of-3 is selected *on the TEST set* — it is not an achievable policy, it is the same
test-set-peeking estimator §4f's convention was written to avoid (see the head-ensemble retraction
in §"GraSP SOTA-chase — Head ENSEMBLE"). Both arms' seed spreads (0.048 and 0.030 range) exceed
every ensemble effect here, so a best-seed delta of ±0.01 is inside the draw.

**★ The mechanism: ensembling buys localization, not detection.** Splitting the AP by IoU
threshold makes the pattern sharp, and it explains the one fusion that fails:

| arm | fusion | rel. gain @ IoU 0.1 | @ 0.3 | @ 0.5 |
|---|---|---|---|---|
| `ft_aug` | maxpick | +7.0% | +3.8% | +8.8% |
| `ft_aug` | wbf_meanconf | +5.5% | +7.2% | **+14.4%** |
| `ft_aug` | iou_clustered | +5.2% | +6.6% | **+17.5%** |
| `ft_last4` | maxpick | +4.1% | +4.1% | **−1.1%** |
| `ft_last4` | wbf_meanconf | +8.0% | +10.7% | **+25.8%** |
| `ft_last4` | iou_clustered | +4.4% | +7.8% | **+13.8%** |

The gain grows with the IoU threshold for every coordinate-averaging fusion, and **`maxpick` — the
only fusion that selects one seed's box verbatim rather than averaging coordinates — is the only
one that goes negative at IoU 0.5.** It cannot improve a box it merely copies; at loose IoU it
still gains from better confidence ranking, but at 0.5 it has nothing to offer and drifts.
That is a mechanistic prediction the data satisfies, not a post-hoc label. **Practical consequence:
use `wbf_meanconf` for seed ensembling on this probe** — it wins on both arms and is the only
configuration that also beats the (unachievable) best seed.

**Why FT has more to average than frozen.** The bucket diagnostic shows the cross-seed prediction
sets genuinely disagree: median pairwise IoU falls **0.91 → 0.76** and the fraction of buckets with
a low minimum IoU rises **0.011 → 0.162** (per-seed cross-window fusion vs the 3-seed fuse;
multi-entry buckets 60,228 → 123,963). Three independently tuned encoders localize differently in
a way three heads on one frozen cache do not — which is exactly the diversity that makes
coordinate averaging pay, and why the effect here is larger at IoU 0.5 than §4f's frozen result.

**Caveats.** No σ on the ensemble side — it is a single fused prediction, not three draws, so no
significance test is available on the ensemble column (same limitation as every ensemble row in
this ledger; a 4th held-out seed per arm is the only way to get one). n=3 seeds/arm. Both jobs ran
~7,800 s against the ~4,550 s §4f budgeted for the frozen equivalent — the O(n²) IoU clustering is
unchanged, but these landed on a co-tenanted gpu-07; **budget ~2.2 h/arm, not 80 min**, when
scheduling FT ensembles onto a shared node.

Reproduce: `qsub -v CKPT=prod37m_e199,RUN_TAG=ft_aug ens_ft.sh` (and `RUN_TAG=ft_last4
CACHE_TAG=ft` — the baseline arm's caches predate `RUN_TAG` and sit under the default tag).
Raw JSONs: `/eagle/projects/ModCon/ngetty/esad_probe/seed_ensemble_scores/prod37m_e199_ft_{aug,last4}_seed_ensemble.json`.

### 4e-full. All checkpoints, all seven columns, ONE population (2026-08-24)

Merges both result-file conventions — V-JEPA arms write `test_detection_ap_cov_fulldenom.json`,
external arms write `test_detection_ap_masked_fulldenom.json` (frame-masked to the same 5,903
frames via `--restrict-frame-stems`). All rows verified at `gt_full=11207 / gt_cov=10619 /
frames=5903` before comparison. Reproduces §4e exactly where they overlap (SNX 0.1162, LemonFM
0.1537 frozen). Generated by `~/full_table.py` on Sophia.

| checkpoint / variant | AP_mean | AP10 | AP30 | AP50 | presence† | meanIoU† | boxAP50† |
|---|---|---|---|---|---|---|---|
| **prod37M e199 FT+aug** | **0.2319** | **0.3262** | **0.2568** | **0.1127** | 0.3248 | 0.4765 | 0.5101 |
| prod37M e199 FT last-4 | 0.2035 | 0.3185 | 0.2239 | 0.0680 | 0.3159 | 0.4426 | 0.4209 |
| surgical 2B e159 FT | 0.2005 | 0.2981 | 0.2245 | 0.0788 | 0.3158 | 0.4536 | 0.4497 |
| raw Meta 2B FT | 0.1935 | 0.2828 | 0.2243 | 0.0736 | — | — | — |
| raw Meta 1B FT | 0.1855 | 0.2699 | 0.2145 | 0.0721 | — | — | — |
| surgical 1B e19 FT | 0.1831 | 0.2639 | 0.2128 | 0.0726 | — | — | — |
| prod37M e199 frozen | 0.1793 | 0.2496 | 0.2082 | 0.0801 | 0.2507 | 0.4664 | 0.4810 |
| LemonFM FT | 0.1585 | 0.2921 | 0.1509 | 0.0325 | 0.3118 | 0.3488 | 0.2242 |
| LemonFM frozen | 0.1537 | 0.2739 | 0.1460 | 0.0412 | — | — | — |
| raw Meta 1B frozen | 0.1459 | 0.2188 | 0.1637 | 0.0553 | 0.2358 | 0.4346 | 0.4146 |
| surgical 2B e159 frozen | 0.1395 | 0.1897 | 0.1546 | 0.0742 | 0.2103 | 0.4822 | **0.5156** |
| surgical 1B e19 frozen | 0.1339 | 0.2004 | 0.1547 | 0.0466 | 0.2139 | 0.4371 | 0.3883 |
| raw Meta 2B frozen | 0.1278 | 0.1815 | 0.1425 | 0.0593 | 0.2035 | 0.4627 | 0.4728 |
| SurgeNetXL FT | 0.1256 | 0.2346 | 0.1224 | 0.0199 | 0.2710 | 0.3318 | 0.2082 |
| SurgeNetXL frozen | 0.1162 | 0.2309 | 0.0997 | 0.0181 | — | — | — |
| EndoViT frozen | 0.0853 | 0.1341 | 0.0788 | 0.0430 | — | — | — |
| *SARAS-ESAD best submission* | *0.1928* | *0.2763* | *0.2205* | *0.0816* | — | — | — |
| *SARAS-ESAD organiser baseline* | *~0.12-0.15* | — | — | — | — | — | — |

n=3 on every row. Oracle backfill for the four frozen V-JEPA rows landed 2026-08-24 (jobs
176428–31 + 176552); the remaining blanks are FT rows of the external/older checkpoints whose
per-seed caches were never exported.

**★ Our margin over the external baselines is a LOCALISATION margin.** LemonFM FT reaches
presence 0.3118 — statistically tied with our 0.3159 — but **boxAP50† 0.2242 vs our 0.4209**.
SurgeNetXL likewise (presence 0.2710, box 0.2082). They find the actions and cannot localise
them, which is why they collapse at the strict threshold: **AP50 0.0325 / 0.0199 vs our 0.1127**.
The gap to LemonFM FT widens from +0.034 at AP10 to +0.080 at AP50.

**★★ The oracle backfill shows that localisation margin is a PRETRAINING-FAMILY property, not
something our CPT bought.** Every V-JEPA row — including *raw Meta 2B frozen*, the weakest
detector in the table at AP_mean 0.1278 — has boxAP50† 0.41–0.52, while both externals sit at
0.21–0.22 whether frozen or fine-tuned. Strikingly, the single best box localiser on record is
**surgical 2B e159 frozen at 0.5156**, above even prod37M FT+aug's 0.5101, despite ranking 11th
of 16 on detection AP. So boxAP50† does **not** track detection AP across checkpoints
(Spearman ρ = **0.19** over the 8 V-JEPA rows with oracle columns — weak and well inside the
noise at that n) — **presence is what separates our checkpoints from
each other, and box quality is what separates the V-JEPA family from the image-encoder
externals.** Two consequences: (a) do not use boxAP50† to rank our own checkpoints; (b) an
external's poor detection AP should be attributed to localisation, but our own CPT gains should
not be attributed to it.

**Scoring is no longer a two-step afterthought.** `score_ft.sh` now runs the oracle pass in the
same job (all seven columns per arm), its summary block no longer hardcodes `ft_last4` (it read
the BASELINE's numbers for every non-default arm), and `full_table.py` prints per-arm `n`/`o`
counts and refuses to compare across populations.

### 4b-unfreeze. Partial encoder fine-tuning (last-4 blocks) — PIPELINE BUILT + VERIFIED, RESULT PENDING

**2026-08-06.** After §4b-seeded showed every head-side lever inside the noise floor, the
remaining lever with a precedent effect size above it is encoder unfreezing
(`[[fine-tune-beats-frozen-readout]]`: FT beats frozen readout by +4–7 IVT mAP on triplet,
~100× this probe's 0.06 spread). Note the in-project precedent is genuinely SPLIT: the GraSP
unfreeze ladder came back null (C1 +0.13, C2 identical) — but that study was later
**invalidated** by the encoder-resume bug (`[[ft-encoder-resume-bug]]`), so it is unknown
rather than negative.

**This required building a live-pixel trainer, not a config change.**
`train_esad_double_head.py` trains on a CACHED feature tensor and contains **no encoder at
all**, so it structurally cannot fine-tune one. New file `train_esad_unfreeze.py` puts the
pixel→encoder→head path in the training loop, reusing the exporter's dataset, the existing
`init_module(..., unfreeze_last_n=N)` seam from the GraSP work, and the cached trainer's
loss/metrics. The cached trainer is untouched and remains the frozen protocol of record.

**Platform:** Sophia (8× A100-40GB/node, 24 h walltime, `-A tpc` — **ModCon has no Sophia
allocation**). Polaris's 1 h debug cap cannot hold a 2.8 h/seed run without chunking. See
`[[sophia-shares-gpus]]` — Sophia shares GPUs and does not set `CUDA_VISIBLE_DEVICES`, so jobs
must select free devices themselves.

**Measured throughput (job 171233), correcting a 4.6× overestimate:**

| config | s/window | peak mem |
|---|---|---|
| frozen fwd only (= export) | 0.749 | 11.1 GB |
| unfreeze last-4, act-ckpt ON | 0.773 | 12.4 GB |
| **unfreeze last-4, act-ckpt OFF** | **0.718** | 15.5 GB |
| unfreeze last-4, bs=2 | 0.720 | 20.9 GB |

Fine-tuning costs **~0% over frozen forward** — backward through 4 of 48 blocks is negligible.
**Activation checkpointing is a net LOSS here** (+8% time to save memory a 40 GB card doesn't
need). An earlier 3.20 s/window estimate was an artifact of amortizing encoder load and
dataloader spin-up over a 24-window smoke epoch — do not size production runs from smoke
slices.

**Six bugs found before a single valid epoch ran. Five were in the new code, and four share
one root cause: the cached trainer's COMPONENTS were reused without porting its TRAINING
PROCEDURE.** Recording them because two would have produced confident wrong numbers rather
than crashing:

| # | bug | failure mode |
|---|---|---|
| 1 | wrong `ClipAggregation` call convention (needs nested list + `clip_indices`, returns a list, may be 5-D) | silent wrong features |
| 2 | manifests store `boxes [24,21,4]` with no M axis; `ESADCacheDataset` normalizes this, the live path did not | crash in eval |
| 3 | `evaluate()` called the **DDP-wrapped** modules — a DDP forward is a COLLECTIVE, so rank 0 all-reduced alone while others sat in a barrier | NCCL watchdog timeout at epoch 1 |
| 4 | **no LR schedule** — the cached trainer's `lr_at()` (2-ep warmup 1e-4→1e-3, cosine to 1e-5) was never ported, so AdamW ran flat at 1e-3 from step 0 | **box head saturated its output sigmoid: cx[0.0000,0.9999], w[0.0000,1.0000], IoU 0.003 — WORSE than the 0.137 of an untrained head.** BCE trained normally, so the run looked healthy |
| 5 | Sophia GPU placement | OOM on another user's card |
| 6 | **stale-checkpoint resumption** — the trainer resumes from `latest.pt`, which is right for a requeue but wrong across a code change; a relaunch was about to continue seeds 0/1 from pre-LR-fix checkpoints | **would have silently blended broken and fixed training into one plausible 20-epoch curve** |

Guards added: `--full-gt`-style pre-flight `mem_get_info` assert; `CODE_STAMP` (md5 of the
trainer) stored beside each run, wiping any checkpoint written by different code; and **no
`| tail` on training output** — piping it is what discarded the traceback in job 171266.

**Verification (job 171305), pass bar committed in advance (IoU > 0.137 = untrained head):**

| epoch | IoU | map50 | box loss |
|---|---|---|---|
| 0 | **0.153** | 0.044 | 3.27 |
| 2 | 0.435 | 0.426 | 1.35 |
| 3 | **0.444** | 0.455 | 1.03 |

Matches the cached path's trajectory (0.449 @ ep2). Resume verified live (48 encoder tensors
restored, best-metric carried). DDP verified at 100% util on 4 ranks.

### RESULT (jobs 171347/8/9 train, 171444 score, 2026-08-07) — **POSITIVE, first real gain on this probe**

3 seeds × 20 epochs, last-4 unfreeze, all `rc=0`. Scored on the **identical** full-denominator
protocol as the frozen baseline (verified: `gt_full=11207`, `gt_cov=10619`, `frames=5903`
byte-identical across all four rows — the denominator error that opened this investigation
cannot be repeating here).

**Table in the challenge's own reporting format (job 171475 adds the oracle columns).
Every row is on ONE population** — the coverage test split, 10,619 GT instances /
5,903 frames for detection AP, 29,854 (token,class,slot) entries for the oracle
metrics. The frozen run's pre-existing `test_summary.json` was NOT reused: it is on the
original 7,274-box population and mixing it in would repeat the denominator error this
investigation opened with. All rows recomputed by `score_oracle.py`.

| method | AP_mean | AP10 | AP30 | AP50 | presence mAP† | mean IoU† | box mAP@50† |
|---|---|---|---|---|---|---|---|
| *SARAS-ESAD baseline* | *~0.12–0.15* | — | — | — | n/a† | n/a† | n/a† |
| *SARAS-ESAD best submission* | *0.1928* | *0.2763* | *0.2205* | *0.0816* | n/a† | n/a† | n/a† |
| frozen probe (v1/e159) | 0.1403 | 0.2061 | 0.1600 | 0.0547 | 0.2228 | **0.4677** | **0.4966** |
| FT last-4, seed 0 | 0.1924 | 0.2812 | 0.2181 | 0.0778 | 0.3106 | 0.4573 | 0.4680 |
| FT last-4, seed 1 | **0.2114** | **0.3278** | **0.2295** | 0.0771 | **0.3286** | 0.4452 | 0.4099 |
| FT last-4, seed 2 | 0.1976 | 0.2853 | 0.2259 | **0.0815** | 0.3084 | 0.4584 | 0.4711 |
| **FT mean (n=3)** | **0.2005** | **0.2981** | **0.2245** | **0.0788** | **0.3158** | 0.4536 | 0.4497 |
| *FT seed spread* | *0.0191* | *0.0465* | *0.0114* | *0.0045* | *0.0202* | *0.0132* | *0.0612* |

**† The last three columns are PROBE-INTERNAL and do not exist for the paper — the `n/a`
is structural, not missing data.** The challenge scored one metric: confidence-ranked
detection AP at IoU {0.10, 0.30, 0.50}. `presence mAP` comes from our probe's separate
presence head (a detector has no such head — it emits ranked boxes, not per-token
multi-label logits). `mean IoU` and `box mAP@50` are **oracle-gated**: the box head is
scored only where GT presence is supplied for free, which removes the hard half of
detection. **Never set an oracle column against a challenge AP number** — §4 records
exactly that error, where our oracle map50 of 0.479 was informally read against the
challenge's AP50 of 8.16 and looked like a 6× win while measuring an easier problem.
Use these columns only to compare our own runs to each other.

**Reading the split.** Fine-tuning improves everything that requires *finding* things —
detection AP +0.060 and presence mAP +0.093, both far outside their seed spreads — while
slightly degrading the oracle box columns (mean IoU −0.014, box mAP@50 −0.047). So the
unfrozen blocks bought detection and classification, not coordinate refinement once the
class is given. Note box mAP@50's own spread (0.0612) EXCEEDS its delta, so that decline
is not individually claimable; mean IoU's (−0.014 vs spread 0.0132) is marginal at best.
The headline gain does not rest on either.

**Also note `presence mAP` macro == well-supported exactly** on this population: the
`min_pos=25` rare-class filter excludes nothing here, because every class clears 25
positives among 29,854 entries (vs 7,274 originally). Same symptom as the original
`min_pos=1` bug in §4, different and benign cause — the filter simply has nothing to
drop. Do not read the equality as the bug recurring.

**Δ = +0.0602 vs frozen, with a seed spread of 0.0191 — the delta is 3.2× the spread, so this
CLEARS the pre-committed bar** (a delta counts only if it exceeds the seed spread; the bar was
set before any result was seen). It also clears the ~0.06 noise floor from
`[[esad-probe-unseeded-noise-floor]]` — the only lever in this whole investigation that does.

**The gain is broad, not a single-threshold artifact:** AP10 0.2061→0.2981, AP30
0.1600→0.2245, AP50 0.0547→0.0788. All three IoU thresholds improve by a similar proportion,
and 2 of 3 seeds individually exceed the paper's 0.1928.

**Reading.** This is consistent with `[[fine-tune-beats-frozen-readout]]` (+4–7 IVT mAP on
triplet) and INconsistent with the GraSP unfreeze null — but that null was invalidated by
`[[ft-encoder-resume-bug]]`, the same bug class explicitly guarded against here, so it was
never evidence to begin with. **Every head-side lever tried on this probe (WBF, multi-box,
augmentation, pos_weight cap, selection metric, +RARP1 data) landed inside noise; unfreezing
the last 4 of 48 encoder blocks did not.** The frozen probe was understating the encoder, not
measuring a task ceiling.

**Caveats — read before citing:**
- n=3. The delta is 3.2× the observed spread, but 3 seeds is a small sample for a spread
  estimate; the honest statement is "clears the bar by a comfortable margin," not a p-value.
- Comparison is FT (this section) vs FROZEN (§4b-coverage), **1 checkpoint (v1/e159) only**.
  The other three checkpoints are still frozen-only, so this says nothing about CPT-vs-raw.
- The paper's 0.1928 used a fully-supervised detector trained end-to-end; ours is a frozen-
  trunk + last-4-unfrozen probe. Beating it on this protocol is a real result for the
  *readout*, and is NOT a claim of a better detector architecture.
- Selection was by val `well_supported_map`; best epochs landed at 10/8/8, matching the GraSP
  ladder's ~ep8 peak. Late epochs were flat-to-declining — worth a shorter schedule next time.

**Reproduce:** `train_esad_unfreeze.py` (`--unfreeze-last-n 4 --batch-size 2 --accum 2
--no-grad-ckpt --ddp --seed N`), then `score_ft.sh` (merges the 48 tuned tensors onto the
pretrained encoder, re-exports per-seed test caches — **the frozen caches are stale once the
encoder changes** — and scores). ~5.9 h/seed on 2× A100.

### 4b-seeded. Seeded 3-seed campaign (rare-class fix + more training data) — NULL

**2026-08-06, job 7355901.** First ESAD run with a seed: `--seed` added to
`train_esad_double_head.py` (seeds torch/random/numpy/CUDA; the batch sampler's
`torch.randperm` is the thing that was actually varying — verified same-seed → identical
permutation, different-seed → different). Design deliberately **few configs × 3 seeds** rather
than the previous many configs × 1 seed. 12 arms, one GPU each, 3 nodes, ~35 min.

Arms (all scored on the §4b-coverage full-denominator protocol):
`base` cap50+well_map · `cap` cap1200+macro_map (the §4b-rare defect fix) ·
`data` +RARP1 folded into train (n=2→3 surgeries, 2468→2848 windows) ·
`both`. Arms training on RARP1 use a new `selection_metric: last_epoch` (val is now training
data, so val-based selection would leak; `last_epoch` is leak-free and deterministic).

| config | n | AP mean | per-seed (s0/s1/s2) | **spread** | rare | well |
|---|---|---|---|---|---|---|
| base | 3 | 0.1470 | 0.1340 / 0.1511 / 0.1560 | 0.0220 | 0.0131 | 0.2294 |
| **data** | 3 | **0.1659** | 0.1251 / 0.1850 / 0.1877 | **0.0626** | 0.0471 | 0.2390 |
| cap | 3 | 0.1452 | 0.1498 / 0.1614 / 0.1245 | 0.0369 | 0.0313 | 0.2153 |
| both | 3 | 0.1444 | 0.1192 / 0.1455 / 0.1685 | 0.0493 | 0.0209 | 0.2204 |
| *PAPER* | | *0.1928* | | | | |

**Verdict: NULL — no config clears its own seed spread.** `data` +0.0189, `cap` −0.0018,
`both` −0.0026, against a pooled worst-case spread of **0.0626**. `data` has the best mean and
a plausible mechanism, but its three draws were 0.1251 / 0.1850 / 0.1877 — one bad draw and two
good ones, not a shifted distribution demonstrable at n=3.

**The important methodological finding: seeding did NOT reduce the variance — it exposed more
of it.** Seeded spreads (0.022–0.063) are *wider* than the accidental unseeded replicate
spreads measured in §4b-rare (0.018–0.032). Seeding makes a run reproducible; it does not make
training stable. With n=2–3 training surgeries this probe has a genuinely wide outcome
distribution, and **n=3 seeds cannot resolve a ~0.02 effect against a ~0.06 spread** — any
future ESAD head experiment needs either ~8–10 seeds or an effect size >0.06 to be worth
running.

**Directional (not conclusive):** every intervention raised `rare` (0.0131 → 0.021–0.047),
consistent with §4b-rare's two verified code defects being real; `data` was the only arm to
raise `rare` AND `well` together. Neither survives the spread test.

**Cumulative position: best mean 0.1659 vs paper 0.1928 = 1.16×** (from 3.63× at the start of
this investigation). Essentially all of that closure came from §4b-coverage, which was
structural. **Three independent head-side attempts have now landed inside noise**
(§4b-followup WBF/coverage-fusion, §4c multi-box, §4d augmentation; then §4b-rare's 10-arm
sweep; then this campaign). Head-side tuning on this probe appears exhausted at this data
scale. The remaining levers with effect sizes plausibly above the noise floor are more
training surgeries, or **encoder unfreezing** — `[[fine-tune-beats-frozen-readout]]` records
FT beating frozen readout by +4–7 mAP elsewhere in this project, which is ~10× this spread.

### 4b-followup. WBF and coverage-expansion attempt — both tried, neither beats the baseline

**2026-08-05, in response to a request to close the remaining gap to SOTA via non-unfreezing
levers.** Sent the implementation plan through an independent adversarial review
(codex/gpt56sol) before building — the first draft overclaimed coverage recovery (~70%) and
under-tested the WBF fusion assumption; both were corrected before implementation (full record
in session memory). What was actually built and run:

- **`build_esad_windows.py`** refactored to segment each surgery into maximal contiguous runs
  explicitly, then window each run independently at `--phase {0,1}` (phase 1 starts one frame
  later, flipping representative-frame parity). **Phase=0 output verified byte-identical to the
  pre-refactor manifest** (147/147 windows exact match — tensors, frame lists, window IDs).
  Phase=1 test manifest: 141 windows, **1768 newly-covered frames, exactly 0 overlap with
  phase=0** (parity-complementary as designed) → **union coverage 57.6%** (3584/6223), close to
  the corrected ~55–60% estimate from the review, not the original draft's overclaimed ~100%.
- **`fusion.py`**: 3 box-fusion variants over overlapping-window predictions for the same real
  frame — `maxpick` (original method), `wbf_meanconf` (unconditional confidence-weighted box
  average), `iou_clustered` (single-link cluster by mutual IoU first, keep the highest-
  confidence cluster — restores WBF's actual defining correspondence step). A pairwise-IoU
  diagnostic checks which fusion mode the data supports **before** trusting one as default,
  per the review's concern that "one class-slot = one object" was asserted, not measured.
- **`score_esad_detection_ap.py`** rewritten for multiple `(cache, windows)` sources, computing
  all 3 fusion variants plus two **explicitly isolated** comparisons so "more coverage" and
  "more fused votes on already-covered frames" are never conflated: `coverage_isolated`
  (phase0 + phase1's newly-covered frames only, unfused) and `ensembling_isolated` (phase0+
  phase1 fused, restricted to phase0's original 1816 frames). **Regression-verified**: the new
  scorer's `maxpick` on phase0-only data reproduces job 7348633's `ap_mean` to 0.0 abs diff.

**Diagnostic result — the "one class-slot = one object" assumption HOLDS on real detections,**
once correctly restricted to GT-positive buckets (the earlier all-bucket diagnostic was
dominated by the ~90% of buckets with no ground truth, where the box head still emits *some*
box regardless — checked and excluded before trusting the number): of 3587 GT-positive
`(frame,class)` pairs, 2295 had ≥2 overlapping-window predictions, and among those, **median
pairwise IoU ≈ 1.0, 0% with any pairwise IoU below 0.3 or even 0.5** (job 7349650). This is why
`maxpick` and `iou_clustered` are bit-identical in every result below (clusters never split) —
overlapping windows really do just re-estimate the same object, at least for this probe.

**Results (job 7349429, all 4 checkpoints):**

| checkpoint | native (29.2% cov) maxpick | native WBF | full-cov (57.6%) maxpick | full-cov WBF |
|---|---|---|---|---|
| e159 (ours, 2B) | 0.1796 | 0.1819 (+0.0023) | 0.1753 (**−0.0043**) | 0.1761 |
| meta1B (raw) | 0.1704 | 0.1729 (+0.0025) | 0.1554 (**−0.0150**) | 0.1576 |
| meta2B (raw) | 0.1715 | 0.1698 (−0.0017) | 0.1581 (**−0.0134**) | 0.1581 |
| ours1b_e19 (CPT) | 0.1752 | 0.1746 (−0.0006) | 0.1690 (**−0.0062**) | 0.1670 |

> **🛑 CORRECTION 2026-08-05: the coverage-expansion verdict below is WRONG and is reversed.**
> The table above compares two conditions scored on **different denominators** (3,587 GT vs
> 7,085 GT), so "AP went down" measures *the denominator growing*, not the lever failing.
> Re-scored on the paper's fixed 11,207-GT denominator (`AP_full_c = AP_cov_c × gt_cov_c /
> gt_full_c`, the exact identity verified in §4b-corrected), coverage expansion is the
> **largest single improvement measured in this whole probe**:
>
> | condition | scored GT | AP (own covered subset) | **AP (fixed full denominator)** |
> |---|---|---|---|
> | phase0 only | 3,587 | 0.1796 | 0.0530 |
> | phase0 + phase1 | 7,085 | 0.1753 *(looks −0.004)* | **0.0999 (+89%)** |
>
> Same for WBF, which is genuinely null either way (0.0530 → 0.0540, still noise-sized).
> **The "newly-recovered frames are genuinely harder" hypothesis is unnecessary** — it was
> invented to explain an artifact. The new frames perform *comparably*; the reported drop was
> the denominator moving underneath the metric. Do not cite that hypothesis.
>
> The caveats at the bottom of this section did correctly disclose the moving population — but
> the conclusion was drawn as if it hadn't. **Lesson: when a lever changes the evaluated
> population, a fixed denominator is not a caveat, it is a precondition for the comparison.**
> Cross-reference `[[ab-window-truncation-trap]]`: same failure family — a verdict read off a
> window/population that one arm happens to define.

**Neither lever closes the gap to the paper's SOTA — both were tried honestly and reported as-is:**
*(⚠ superseded for the coverage lever by the correction banner above — text kept for history.)*
- **WBF at native coverage**: effect is tiny (±0.002–0.003 AP) and **sign-inconsistent across
  checkpoints** (helps e159/meta1B, hurts meta2B/ours1b_e19) — consistent with noise, not a
  real effect, and expected given the diagnostic above: when overlapping windows already agree
  almost perfectly, there's nothing substantive for fusion to fix.
- **Coverage expansion made AP *worse* for all 4 checkpoints**, by −0.004 to −0.015 — the
  opposite of the hoped-for direction. Reading: this is not evidence the phase-1 windows are
  buggy (the manifest diff-check and 0%-overlap confirmation rule that out) — it means the
  **newly-recovered 1768 frames are genuinely harder** for this probe than the original
  1816. A plausible mechanism, not yet verified: phase-1 windows' representative frames sit at
  different token positions within their window than phase-0's, so if the ASFormer head's
  learned temporal structure (positional encoding, dilated conv receptive field) has any
  position-dependent quality variation, the newly-covered frames could be drawn from
  systematically weaker positions. **Flagged as a hypothesis, not confirmed** — would need a
  per-token-position AP breakdown to test, not done here.
- **The `coverage_isolated` and `ensembling_isolated` numbers are numerically identical to
  `source0_only`/`combined_not_isolated`** in this run because phase0/phase1 have exactly 0%
  frame overlap (confirmed above) — with no shared frames, "coverage-isolated" and "combined"
  collapse to the same computation. The isolation machinery is real and will matter the moment
  a future window-generation scheme produces overlapping sources; it was inert here by
  construction, not broken.

~~**Conclusion:** both requested levers (WBF, coverage recovery) were implemented carefully,
independently reviewed before building, unit-tested, and regression-verified — and neither
improves on the 0.170–0.180 AP_mean band already reported in §4b. The remaining gap to the
paper's 0.1928 does not appear to be closable by post-hoc fusion or coverage tricks on top of
this frozen-head architecture; the honest next lever is the head-training side (more capacity,
augmentation, multi-box-per-class, or eventually encoder unfreezing — the last explicitly
scoped as a separate future experiment per the user's original framing).~~

**CORRECTED CONCLUSION (2026-08-05).** WBF is null (confirmed on both denominators). **Coverage
recovery is NOT null — it is the biggest measured win in this probe (+89% honest AP, 0.0530 →
0.0999)** and was mis-scored as a small loss. The gap to 0.1928 is therefore *primarily a
coverage gap*, not an architectural ceiling: 68% of test GT is currently unreachable because no
window covers those frames. Extrapolating the same per-GT rate to full coverage projects
~0.17 — near the paper's 0.1928 — though that extrapolation assumes the remaining uncovered
frames behave like the recovered ones and must be measured, not assumed. **Priority order
revised: finish coverage first, then head-side levers.**

**Caveats:**
- Single seed per checkpoint, unchanged from §4/§4b.
- All numbers above are computed over the covered-subset population for their own condition
  (29.2% vs 57.6%), not a fixed 6223-frame denominator — see `note_population_caveat` in every
  output JSON. The coverage-expansion result is a fair AP-over-covered-subset comparison
  *within* each checkpoint's own two conditions, but is not directly comparable to the paper's
  full-test-set AP either way.
- The "harder new frames" hypothesis above is unconfirmed; do not cite it as an established
  cause without the per-position breakdown.

See `esad-probe-leo-eagle-locations.md` for the full implementation/review history and exact
job IDs (7349337 window-verify, 7349406 regression-check, 7349429 full-coverage run, 7349650
GT-restricted diagnostic).

### 4c. Multi-box-per-class (M=2) — NEGATIVE RESULT, confirmed root cause, abandoned

**2026-08-05, continuing the search for non-unfreezing SOTA-closing levers after §4b-followup
found WBF/coverage exhausted.** Extended `ClassSlotBoxHead` from 1 query per class to 2
(`max_boxes_per_class=2`), with a 2-permutation assignment (`box_matching.py`, brute-forced
since M=2 is fixed by design — same-class duplicates are only 2–4% of frames per STATUS.md, so
a general Hungarian solver would be overkill) feeding both the training loss and the oracle
`conditioned_localization` metric. Box targets extended to top-2-by-area per class
(`build_esad_windows.py`); patched directly into existing feature-cache shards (box targets
don't depend on cached features, verified all 4 checkpoints share one manifest per split) so no
re-export was needed. **M=1 regression-verified exact match (abs diff 0.0) against job
7348633** before trusting any M=2 result — the refactor is a true no-op at M=1.

**Real M=2 result (v1/e159 smoke test, job 7350689 train + 7351132 score, full 20 epochs):**

*(AP rows are covered-subset AP over 32% of test GT — see §4b-corrected. Both arms share the
identical coverage, so the **delta** is valid; the absolute values are not full-test AP.)*

| metric | M=1 (e159, job 7348633) | M=2 (job 7351132) | delta |
|---|---|---|---|
| detection AP_mean (maxpick) | 0.1796 | **0.0913** | **−0.0883** |
| oracle well_map | 0.3785 | 0.3500 | −0.0285 |
| oracle mean_iou | 0.4666 | 0.4869 | +0.0203 |
| oracle map50 | 0.4791 | 0.5334 | +0.0543 |

**Detection AP roughly halved** — far outside the ~3 mAP single-seed noise floor, and the
oracle box-localization metrics actually *improved*, which ruled out "the box head just got
worse" as the explanation and pointed at the detection-AP pipeline itself (oracle metrics are
GT-presence-gated; detection AP is not).

**Root cause CONFIRMED, not just hypothesized** (job 7351194, direct inspection of raw
model output on 38,136 real `(frame, class)` pairs from the test set): **the presence head has
no notion of "slot"** — it emits exactly one confidence logit per class per token, shared
identically by both box query slots. Result: **100% of pairs (38136/38136) had bit-identical
confidence scores across slot 0 and slot 1, and the two slots' boxes were also near-identical**
(crude L1-proxy IoU > 0.5 on 100% of pairs). Mechanism, once seen: the box-matching loss only
gives slot 1 a distinct training target on the 2–4% of frames with a genuine 2nd same-class
instance; on the other ~96–98%, slot 1 has no assignment pressure at all and simply converges
toward copying slot 0. The net effect is that **every real detection now spawns a same-
confidence, same-box duplicate** — the AP scorer's greedy matcher assigns one copy to the real
GT (TP) and is forced to count the twin as a false positive at the *identical* confidence rank,
diluting precision at every threshold for every well-supported class uniformly (confirmed:
class-level breakdown shows broad degradation concentrated in classes with real GT support —
well-supported classes each dropped 0.03–0.33 AP — not concentrated in the handful of classes
that actually have duplicate instances, which is exactly what the "phantom twin on every
detection" mechanism predicts and a genuine "does the model find 2nd instances" story would
not).

**Decision: abandoned, not fixed.** A real fix would need a genuine architecture change (e.g.
per-slot confidence — `num_classes*M` presence logits instead of `num_classes` — so each slot
can independently signal "I have nothing to contribute here"), which is new, unproven design
work with its own retrain-and-verify cycle, for a mechanism (recovering 2–4%-of-frames same-
class duplicates) whose maximum plausible upside is small relative to the demonstrated
downside. User confirmed abandoning multi-box as a dead end rather than iterating on the fix.

**What stays reusable:** `box_matching.py`'s 2-permutation assignment logic and its unit tests,
`build_esad_windows.py`'s top-M box-target extraction, and `patch_cache_boxes_m2.py`'s no-
re-export cache-patching pattern are all independently correct and unit-tested — the flaw is
specifically in `ClassSlotBoxHead`/`ESADDoubleHead`'s shared-confidence design, not in any of
the supporting machinery. If per-slot confidence is ever built, these pieces do not need
rework.

**Caveats:**
- Only tested on 1 of 4 checkpoints (v1/e159) — the root cause is architectural (presence-head
  output shape), not checkpoint-specific, so there was no reason to expect the other 3 to
  behave differently, and the negative result was severe enough not to warrant spending the
  compute to confirm on all 4.
- `--config`/YAML support for `max_boxes_per_class` remains in the codebase (default 1, inert
  for every existing checkpoint config) — this is dead-but-harmless surface area, not cleaned
  up, since removing it would require re-touching files already verified byte-identical at M=1.

**2026-08-24 — the deferred per-slot-confidence fix is now CLOSED as not worth building, on a
measured ceiling rather than a judgement call.** §4c above proposed per-slot confidence
(`num_classes*M` presence logits) as the "real fix" and left it open. Before building it, the
upside was measured directly from the test GT — a slot can only ever recover the 2nd..Nth
instance of a `(frame, class)`, so counting those bounds *any* M=2 head, oracle-perfect or not:

| boxes in a (frame,class) cell | cells | GT boxes | % of all GT |
|---|---|---|---|
| 1 | 10,691 | 10,691 | 95.40% |
| 2 | 231 | 462 | 4.12% |
| 3 | 18 | 54 | 0.48% |

Recall ceiling with perfect slots: **M=1 97.62% → M=2 99.84%** (+2.22 pts global). But AP_mean
is **macro**, so the per-class version is what binds: **macro ceiling 98.89% → 99.96%, +1.07
pts.** Since AP ≤ recall, a *flawless* per-slot head is worth **≤ ~0.0025 absolute AP** on a
0.23 base — **10–25× below the 0.022–0.063 seed noise floor** (§4b-seeded). It is not
measurable with any seed count we would run, and it is paid for by doubling the false-positive
opportunity on the 95.4% of cells holding exactly one box. Train-manifest side agrees: only
**1.38%** of positive `(t,class)` cells (570/41,352) have a 2nd box, so slot 1 gets an
assignment on ~1 in 72 positives — the "no assignment pressure → copy slot 0" mechanism §4c
diagnosed is structural, not a tuning failure.

**The headroom is somewhere else, and the same computation says where.** On the best FT arm
(last-4 + aug, s0): **macro AP 0.2265 vs GT-weighted AP 0.3484** — the metric's loss is
concentrated in rare classes, and **7 of the 8 worst-AP classes have exactly ZERO slot-2
headroom**:

| worst classes | AP | GT | slot-2 headroom |
|---|---|---|---|
| 20 | 0.0123 | 105 | +0.019 |
| 19 | 0.0356 | 48 | +0.000 |
| 3 | 0.0391 | 36 | +0.000 |
| 16 | 0.0924 | 18 | +0.000 |
| 2 | 0.1150 | 48 | +0.000 |
| 1 | 0.1223 | 113 | +0.000 |
| 7 | 0.1226 | 307 | +0.000 |

Task #6 was therefore **redirected to rare-class rebalancing**, which targets exactly those
classes. `pos_weight_cap: 50.0` clips **12 of 21 classes** — the majority of a macro metric —
distorting the true `neg/pos` weight by **21.9× for class 12** (1095.9 → 50) and 19.7× for
class 2. Caps of 100/200/500 clip 10/6/2 classes respectively. Arm running: `pos_weight_cap:
200.0` on the best config, 3 seeds (jobs 176569–176571). ⚠ This is a *lever*, not a result —
raising the cap trades precision on frequent classes for recall on rare ones, and the §4b-rare
arms are the reminder that rare-class interventions often move `val_macro_f1` without moving
mAP ([[val-macro-f1-not-a-map-proxy]]).

**Independent support: the §4d augmentation win is disproportionately a rare-class win.**
Splitting the per-class AP of `ft_last4` vs `ft_last4+aug` (s0) by whether a class clips the
`pos_weight` cap — a split defined purely by train-set frequency, with no knowledge of the
results — gives:

| group | n classes | ft_last4 | +aug | delta |
|---|---|---|---|---|
| clipped at cap=50 (rare) | 12 | 0.1470 | 0.2053 | **+0.0583** |
| unclipped (frequent) | 9 | 0.2355 | 0.2547 | +0.0192 |
| macro | 21 | 0.1849 | 0.2265 | +0.0416 |

The gain on the clipped classes is **3.0× the gain on unclipped ones**. Two things follow:
augmentation's mechanism (preventing memorisation of 2,468 windows from n=2 surgeries) bites
hardest exactly where support is thinnest, and the rare classes are demonstrably *movable* —
they are not a hard ceiling imposed by label noise. Biggest single movers are all rare:
`CuttingMesocolon` 0.1380→0.3871, `ClippingTissue` 0.3073→0.4971, `BaggingProstate`
0.1648→0.2693. `rare_class_report.py` (new, in the probe dir) prints this table with class
names for any set of run dirs.

⚠ Single seed, so the per-class numbers are indicative only — the seed noise floor is quoted on
the *macro* number and per-class values are noisier still. The clipped-vs-unclipped *split* is
the robust part (12 vs 9 classes averaged), not any individual row.

### 4d. Pixel-space augmentation (crop + color jitter) — NULL RESULT, single checkpoint

**2026-08-05, the last of the SOTA-closing levers tried this session** (STATUS.md flags this
probe as deliberately "augmentation-free by construction" — the anisotropic-squash
preprocessing keeps normalized box coordinates invariant with zero augmentation, which this
experiment deliberately breaks). Since the frozen encoder must actually SEE perturbed pixels
for augmentation to mean anything (feature-space noise on the cache is not equivalent to what
the paper's winning submission credits — real "bounding box jitter" applied before the
backbone), this required genuinely new export compute, unlike WBF/coverage which reused
existing caches. Added `--augment` to `export_esad_cache.py`: a per-window random crop (90–100%
scale) + brightness/contrast/saturation jitter, applied identically to all 48 frames in a
window (not per-frame flicker), with GT boxes remapped into the crop's coordinate frame and
zeroed out if their remapped area drops below 20% of original (unit-tested: identity crop,
centered-scale crop, fully-outside-crop masking, already-absent-slot preservation, batched
input — all pass). **Horizontal flip and rotation deliberately excluded** — ESAD's 21 action
classes' left/right-handedness semantics were not confirmed safe under flip, and there was no
STATUS.md class-name list available to check; scoped to geometry- and appearance-preserving
transforms only until that's resolved. Augmented cache written to a separate dir
(`cache_v1_aug/train`), train-only (val/test stay deterministic); `train_esad_double_head.py`
extended to accept multiple `--train-cache` dirs (concatenates shard indices, same pattern the
multi-source scorer already used) so training ran on original+augmented combined (2468→4936
windows).

**v1/e159 smoke test (jobs 7351268 export+train-to-ep5 walltime-killed, 7351899 resume-to-
completion+score):** augmented export doubled the train set, which roughly doubled per-epoch
time (~470–510s vs. the unaugmented ~66–113s) — underestimated when budgeting the first job's
1h walltime; needed one resume cycle (`resume: true` picked up cleanly from `latest.pt` at
epoch 5, confirmed by the log's "resumed from epoch 5 (best 0.2587 @ 5)" line) to reach the
full 20 epochs.

*(AP rows are covered-subset AP over 32% of test GT — see §4b-corrected. Both arms share the
identical coverage, so the **delta** is valid; the absolute values are not full-test AP.)*

| metric | baseline (no aug) | augmented | delta |
|---|---|---|---|
| detection AP_mean (maxpick) | 0.1796 | 0.1686 | **−0.0110** |
| oracle well_map | 0.3785 | 0.3508 | −0.0277 |
| oracle mean_iou | 0.4666 | 0.4713 | +0.0047 |
| oracle map50 | 0.4791 | 0.4922 | +0.0131 |

**Reading: a null result, not a clear win or loss.** The detection-AP delta (−0.011) and the
oracle-metric deltas are all smaller than or comparable to this project's established ~3 mAP
(~0.03) single-seed noise floor — none of the four numbers moved decisively in either
direction. **User decision (2026-08-05): stop after 1 checkpoint rather than confirm on all
4** — the single-checkpoint result is honest but inconclusive; a real effect at this magnitude
would need seed replication (not just more checkpoints) to separate from noise, and the cost
(2 walltime-slot cycles per checkpoint due to the doubled training-set size) wasn't judged
worth it for a result already this close to null.

**Caveats:**
- Single checkpoint (v1/e159), single seed — do not generalize to the other 3 checkpoints or
  treat this as a confirmed absence of effect; it's "no effect large enough to see on 1 run,"
  not "proven no effect."
- Only the geometry/appearance-preserving augmentation subset was tried; flip/rotation (once
  confirmed class-semantics-safe) and stronger jitter ranges remain untested.
- Augmentation was applied only to the TRAIN cache; a from-scratch retrain each time, not an
  incremental fine-tune — this is the fair comparison for "does more training diversity help,"
  but does mean the ~2x per-run compute cost is real and recurring for every future checkpoint
  tested this way, unlike WBF/coverage which were free re-scores of existing artifacts.

~~**Overall conclusion across §4b-followup/§4c/§4d: every non-unfreezing lever tried this session
(WBF, coverage expansion, multi-box-per-class, pixel-space augmentation) either showed no real
effect or, in multi-box's case, a confirmed and severe negative effect.** The ~0.170–0.180
AP_mean band from §4b appears to be a genuine ceiling for this frozen-encoder, single-query,
un-augmented-by-default architecture — closing the remaining ~0.01–0.02 gap to the paper's
0.1928 likely requires either encoder unfreezing (already scoped separately, per the user's
original framing) or a more substantial head-architecture change than what was tried here.~~

**CORRECTED OVERALL CONCLUSION (2026-08-05).** Two claims above are wrong:
1. **Coverage expansion was NOT null — it is +89% honest AP** (see §4b-followup's correction
   banner). It was mis-scored on a moving denominator.
2. **The "~0.01–0.02 gap" and the "0.170–0.180 ceiling" are artifacts of the 32%-GT
   denominator.** On the paper's denominator the band is **0.050–0.053** and the gap to 0.1928
   is **~3.6×** (§4b-corrected).

What still stands unchanged: WBF null, multi-box severely negative with a confirmed
architectural root cause, augmentation null on 1 checkpoint. Those three were all measured as
same-coverage A/B deltas, so the denominator error cancels within each comparison — but their
*absolute* AP columns are covered-subset numbers and should be read as such. **The revised
picture: the dominant deficit is coverage (68% of GT unreachable), which is a pipeline
property and fixable, not a frozen-encoder ceiling.**

See `esad-probe-leo-eagle-locations.md` for job IDs and full implementation notes.

**2026-08-24 — two augmentation follow-ups wired and verified (arms pending queue).** §4d's
recipe had two properties worth testing that were not reachable without a code edit:

1. **The crop was resampled once per RUN, not per epoch.** `export_esad_cache.py` seeded the
   RNG as `Random(aug_seed*1_000_003 + window_id)` — no epoch term — so `--augment` gave each
   window ONE fixed crop for all 20 epochs: a static dataset perturbation, not augmentation.
   The published +0.028 AP therefore came from the *weaker* form, and genuine per-epoch
   resampling should be at least as strong. Added `set_aug_epoch()` + an `aug_epoch*8_388_617`
   term, called once per epoch before any rank iterates the loader (workers are non-persistent,
   so they fork after the call and inherit it). Opt-in via `--aug-per-epoch` so the published
   arm stays reproducible. Verified: crops differ across epochs and are reproducible within one.
   Arm: `ft_augpe_last4`, 3 seeds, jobs 176562–176564.
2. **Strength is now env-tunable** (`ESAD_AUG_MIN_SCALE`/`MAX_SCALE`/`ASPECT`/`COLOR`) instead
   of hardcoded default args, so testing "a stronger recipe may have more headroom" no longer
   needs a source edit (and a `CODE_STAMP` bump) per arm. **Defaults reproduce the published
   recipe bit-for-bit: 0/20000 RNG draws differ from the pre-patch function** — every arm on
   record stays comparable. Horizontal flip deliberately still NOT added: box semantics under
   flip were never verified for this class set (left/right-handed instrument actions), and an
   unverified flip would silently corrupt box targets.

⚠ **Two plumbing traps checked rather than assumed**, either of which would have produced a
false null: the knob is read inside `__getitem__`, i.e. in a forked DataLoader worker (verified
workers see the parent's value and the crop actually changes, area 0.913 → 0.740), and `qsub
-v` populates the job shell without necessarily exporting to `torchrun`'s children (added an
explicit export loop in `run_ft_seed.sh`, which logs `[aug] VAR=...` when set and is silent
when unset).

**"Train longer" is RULED OUT as a lever — checked from logs already on disk, zero GPU-hours.**
Since augmentation raised `train_bce` 3.4× at ep19, the obvious next question was whether 20
epochs is simply too short for the augmented recipe. It is not: every arm's val selection
metric peaks early and *declines*, so `best.pt` is never the last epoch.

| arm | best `val_well_map` @ epoch | last 3 epochs | `train_bce` ep0→ep19 |
|---|---|---|---|
| ft_last4 s0/s1/s2 | 0.4041 @8 / 0.4290 @6 / 0.4251 @10 | 0.386→0.383 / 0.398→0.397 / 0.408→0.408 | →0.0036 / 0.0020 / 0.0065 |
| ft_aug s0/s1/s2 | 0.4378 @10 / 0.4415 @9 / 0.4145 @5 | 0.430→0.428 / 0.424→0.422 / 0.395→0.394 | →0.0125 / 0.0075 / 0.0097 |

Best epoch is 5–10 of 20 in all six runs, with a flat-to-declining tail — adding epochs would
only add overfitting. **But `train_bce` still collapses to 0.0075–0.0125 even with
augmentation**, i.e. the augmented model *also* ends up memorising, just later. Those two facts
together point the same way: the remaining regularisation headroom is in augmentation
*strength*, not training *duration*. Hence the strong arm (`ESAD_AUG_MIN_SCALE=0.70`,
`ASPECT=0.15`, `COLOR=0.30`, with per-epoch resampling): jobs 176575–176577.

⚠ **A `CODE_STAMP` hazard was created and repaired in the process.** `run_ft_seed.sh` stores
`md5sum train_esad_unfreeze.py` beside each run and WIPES any partial run whose stamp differs.
Patching the trainer while the three `aug_last8` resumes sat in state `Q` invalidated their
stamps, which would have discarded 21 epochs of real work (7 × 3 seeds). Because the diff is
strictly additive and opt-in — an argparse entry, an implication, a print, and a guarded call,
so a run that does not pass `--aug-per-epoch` is behaviourally identical — the stamps were
bumped by hand with a `.code_stamp_history` note recording why. **That is the narrow case where
bumping is legitimate; it is not a general licence.** The lesson is the ordering: verify no
queued job depends on a file *before* editing it, not after.

---

### 4e. The completeness table — 9 requested cells, one population (2026-08-14)

**Goal: fill the requested frozen/FT × {surgical 1B e19, raw Meta 2B, raw Meta 1B, SurgeNetXL,
LemonFM, EndoViT} table on ONE honest denominator.** Before this pass, only e159 (2B, not in
this table) had both a full-denominator frozen number and a 3-seed FT number; the three cells
1/3/5 sat on the stale 32%-GT population from §4b (pre-§4b-coverage), and cells 7/8/9 had never
been run on this probe at all.

**Phase 0 — denominator repair (cells 1/3/5).** Re-scored the existing `best.pt` for meta1b/
meta2b/ours1b_e19 on the §4b-coverage padded/anchored test manifests — no retraining, same
protocol as §4b-coverage. Gate: `gt_full=11207`, `gt_cov=10619`, `frames=5903` on all three,
byte-identical to e159's population.

**Phase 2/3 — tubelet=1 port (cells 7/8/9).** The probe's window builder and scorer were
hardcoded for tubelet=2 (V-JEPA's native geometry: 24 temporal tokens, `MAX_BOXES_PER_CLASS`
baked in). Added `--tubelet {1,2}` and `--max-boxes-per-class {1,2}` flags to
`build_esad_windows.py` (both default to the pre-existing values so cited manifests reproduce
byte-identically), and made `score_esad_detection_ap.py` read tubelet/geometry from each
manifest's own `geometry` block instead of assuming module constants — a tubelet=1 manifest
scored with tubelet=2 constants is **in-range but silently wrong** (misattributes every
prediction to the wrong frame), which is worse than crashing. SurgeNetXL's existing
`video_classification_frozen` adapter (from the SAR port) was reused as-is; LemonFM and EndoViT
adapters were ported from `head_models_jepa/.../triplet_recog_frozen/modelcustom/`, retargeted
to import `vit_encoder_multiclip`'s `preserve_clip_dim=True` variant. All three passed a 4-gate
verification (import; forward-shape; `ESADDoubleHead` forward-shape; 2-window export smoke)
before any real export.

**A real bug caught mid-run, not just a caveat.** The first SNX attempt crashed on a box-head
shape mismatch; setting `max_boxes_per_class=2` in the new probe YAMLs to match the manifest's
native M=2 "fixed" the crash but silently gave the three external backbones **more
box-prediction capacity than every V-JEPA cell**, which — since the manifests' `head_kwargs`
never set `max_boxes_per_class` and therefore default to 1 — inflated their `gt_cov` to 10,861
against the V-JEPA cells' 10,619 on the identical 5,903-frame set. Caught by re-deriving the GT
count from first principles (raw label files, and the manifest's own `box_mask`) and finding it
didn't match the scorer's own output; root-caused to the M mismatch, not a scorer bug. Fixed by
rebuilding all tubelet=1 manifests at `--max-boxes-per-class 1`, matching every V-JEPA cell's
implicit protocol exactly (verified: GT count on the 5,903-frame restriction landed on 10,619,
an exact match).

**Fairness masking (cells 7/8/9).** At tubelet=1 every frame offset is representative, so the
image arms reach ~100% test coverage (6,088/6,088 frames) versus the V-JEPA arms' 94.8%
(5,903/6,088) — the same coverage-inflation failure mode §4d's denominator bug caught, just
running the other direction. Added `--restrict-frame-stems` to `score_esad_detection_ap.py`:
predictions are filtered down to the identical 5,903-frame set the V-JEPA coverage manifests
reach before scoring. The **masked** column below is the comparison column; the unmasked
~100%-coverage number is reported only as a secondary check, never compared across rows.

**The table. Every cell: `gt_full=11207`, `gt_cov=10619`, `frames=5903` (masked column,
verified per-row) — one population, full-denominator AP_mean (maxpick), IoU {0.10, 0.30, 0.50}.
Frozen columns updated 2026-08-14 to 3-seed (see "3-seed frozen retrofit" below) — the
originally-published single-seed frozen numbers (ours1b_e19 0.1463, meta2b 0.1225, meta1b
0.1264, e159 0.1403) are each within their respective 3-seed spreads, so nothing here
overturns an earlier claim, it just adds error bars that didn't exist before.**

| # | cell | frozen (3-seed mean ± spread) | FT last-4 (3-seed mean ± spread) |
|---|---|---|---|
| — | surgical 2B e159 (v1/e159) | **0.1483** ± 0.0149 | **0.2005** ± 0.0191 |
| 1/2 | surgical 1B e19 (ours1b_e19) | **0.1302** ± 0.0196 | **0.1832** ± 0.0088 |
| 3/4 | raw Meta 2B (meta2b) | **0.1271** ± 0.0264 | **0.1936** ± 0.0727 |
| 5/6 | raw Meta 1B (meta1b) | **0.1441** ± 0.0462 | **0.1855** ± 0.0194 |
| 7 | SurgeNetXL, frozen | **0.1162** ± 0.0012 | — |
| 8 | LemonFM, frozen | **0.1537** ± 0.0359 | — |
| 9 | EndoViT, frozen | **0.0853** ± 0.0020 | — |

*(e159 is not one of the 6 checkpoints originally requested for this table, but it's the only
checkpoint with both a published frozen AND FT number prior to this pass — included as a full
row, not a footnote, for direct comparison. Paper best submission: 0.1928.)*

**Per-seed FT values** — seed 0 / seed 1 / seed 2:
- ours1b_e19: 0.1811 / 0.1886 / 0.1798
- meta2b: 0.2152 / 0.1464 / 0.2191
- meta1b: 0.1833 / 0.1769 / 0.1963
- e159: 0.1924 / 0.2114 / 0.1976 (from §4b-unfreeze, unchanged)

**Per-seed frozen values** — seed 0 / seed 1 / seed 2:
- e159/v1: 0.1549 / 0.1400 / 0.1501
- ours1b_e19: 0.1411 / 0.1279 / 0.1215
- meta2b: 0.1119 / 0.1383 / 0.1310
- meta1b: 0.1308 / 0.1277 / 0.1739
- SurgeNetXL: 0.1166 / 0.1167 / 0.1154
- LemonFM: 0.1320 / 0.1614 / 0.1679
- EndoViT: 0.0863 / 0.0854 / 0.0842

**3-seed frozen retrofit (2026-08-14).** The table above originally shipped with single-seed
frozen numbers for ours1b_e19/meta2b/meta1b (inherited from §4b-coverage, which never seeded
the frozen probe) while cells 7/8/9 and every FT cell had 3-seed replication — an inconsistent
error-bar treatment across an otherwise-matched table. Retrofit: re-ran frozen head-training at
seeds {0,1,2} for all four V-JEPA checkpoints (including e159, which also only had a single
frozen number on record) on the EXISTING train/val/test caches — frozen-encoder training reuses
cached features, so this needed no new export, only new head-training + re-scoring on the same
§4e population. `train_esad_double_head.py`'s own `resume: true` (checkpointing every epoch to
`latest.pt`) made every walltime-killed attempt in this retrofit resumable rather than a
restart, confirmed via the `[TRAIN] resumed from epoch N` log line landing exactly where the
prior attempt was cut off.

**Batching lesson, for reuse:** frozen head-training is single-GPU, cheap (~75–200s/epoch
depending on checkpoint D, and node contention), and initially run one-(checkpoint,seed)-per-PBS-job
serially — a bad use of a 4-GPU Polaris node. Batching 4 combos onto one node's 4 GPUs via
`mpiexec -n 4` FAILED SILENTLY: `train_esad_double_head.py` has its own rank-gating (only global
rank 0 trains, `PMI_RANK`/`PALS_RANKID`/etc-detected, by design for its intended single-mpiexec
launch), so `mpiexec -n 4` sets exactly those vars to 0/1/2/3 and 3 of 4 "ranks" silently no-op
before their scoring step crashes looking for a `best.pt` that was never written. Fixed by
launching each combo as an independent background shell process (`... &`) with a clean
environment (`unset PMI_RANK PALS_RANKID ...`) instead of one `mpiexec` world — each process then
correctly sees itself as an unlaunched single run and trains. Verified via direct `ps aux` on the
batch node that all 4 processes were genuinely training (not 1 real + 3 dead) before trusting the
approach for the remaining seeds.

**Reading.**
- **Fine-tuning last-4 blocks helps all four V-JEPA checkpoints at their means**, by margins that
  clear each arm's own FT-seed spread except meta2b (delta 0.066 vs spread 0.073 — within noise,
  see caveat below). e159's delta (0.052) clears its FT spread (0.0190) comfortably.
  ours1b_e19's FT mean (0.1832) is the tightest of the four (spread 0.0088).
- **With 3-seed replication, the frozen CPT-vs-raw comparison flips at 1B and disappears at 2B
  relative to the single-seed read.** At 1B: raw meta1b (0.1441) now edges out CPT ours1b_e19
  (0.1302) — the opposite of the single-seed table's "+0.020 CPT lead" — but both checkpoints'
  spreads (0.0462 and 0.0196) overlap, so this is genuinely a tie, not a raw-Meta win. At 2B: CPT
  e159 (0.1483) vs raw meta2b (0.1271) — the +0.021 delta is SMALLER than meta2b's own spread
  (0.0264), so this does not clear the bar either; there is no confirmed CPT-vs-raw frozen edge
  at either scale. **The single-seed table's apparent "CPT wins at frozen" story does not survive
  seed replication as a clean result.**
- **meta1b's seed 2 (0.1739) is a clear outlier** against its own seeds 0/1 (0.1308, 0.1277) —
  nearly 0.05 higher. This single high draw is what makes meta1b's frozen spread (0.0462) by far
  the widest of the four checkpoints and is largely responsible for the raw-1B-beats-CPT-1B
  read above; a 4th seed would help confirm whether 0.1739 is real variance or a fluke run.
- **All three external image backbones sit below every V-JEPA frozen cell** (0.085–0.154 vs
  0.127–0.148 means). LemonFM is the strongest external arm and the only one whose frozen number
  (0.1537 mean) lands inside the V-JEPA frozen band, though still below every FT mean. EndoViT is
  the weakest arm by a wide margin (0.085, ~33% below the next-lowest V-JEPA frozen cell).

**Caveats — read before citing:**
- **Resolution limit: n=3 seeds resolves only effects >~0.06** (the project's established FT
  noise floor, `[[esad-probe-unseeded-noise-floor]]`). By this bar: **meta2b's FT delta (0.1936 −
  0.1271 = 0.066) and e159's frozen-vs-raw-2B delta (0.1483 − 0.1271 = 0.021) do not clear it —
  both are ties, not wins, and must be reported as such.** ours1b_e19 (FT spread 0.0088) and
  meta1b (FT spread 0.0194) clear their FT bars comfortably. No frozen-vs-frozen CPT comparison
  in this table clears the bar outright once each arm's own spread is accounted for.
- **~~The 48-vs-24 token / dilation-schedule difference...~~ RETRACTED 2026-08-17 — the
  dilation claim is FALSE.** This caveat asserted that at T=48 the image arms "gain a
  32-dilation layer that T=24 never reaches." They do not. `ASFormerHead` uses
  `dilation = 2**(i % max_dilation_exp)` with `max_dilation_exp = ceil(log2(T))` (5 at T=24,
  6 at T=48) — but `presence_num_layers: 4` in **every** probe YAML, so `i` only ever ranges
  0..3 and the modulo **never wraps at either value**. Both geometries produce the identical
  dilation stack **[1, 2, 4, 8]** (verified by direct evaluation of the expression, and by
  reading `asformer_head.py:175-185`). There is no extra layer and no capacity difference from
  dilation. The 48-vs-24 token count itself is still real; only the claimed consequence was
  wrong. **Three OTHER asymmetries, none previously disclosed, are real — see §4g.**
- **Oracle columns (presence mAP / mean IoU / box mAP@50) were not computed for this table** —
  only for e159 in §4b-unfreeze. Do not set any of this table's AP numbers against the paper's
  own oracle-adjacent claims without recomputing them on this same 5,903-frame population first
  (§4b-unfreeze's own caveat about oracle-vs-challenge-AP applies identically here).
- **`meta2b` seed 1's first FT scoring pass was against a 2-epoch checkpoint** (a `score_ft.sh`
  loop outran its own training job on a shared, contended GPU node) and was silently discarded
  once training reached all 20 epochs and the seed was re-scored on the real `best.pt`. The
  numbers in this table are all post-correction. Same failure mode, same fix, hit `meta1b`
  seed 2 once earlier in this campaign.
- Cost: 9 export+train+score pipelines for cells 7/8/9 (~10-40 min each on 2-node Polaris debug,
  cached-feature training is fast) + denominator repair (~1 Polaris debug job) + 3 seeds × 4
  checkpoints × ~5.9h/seed FT training on shared Sophia GPUs for cells 2/4/6/e159 (contention-
  dependent; observed 12-39 min/epoch) + repeated 2h-walltime-limited scoring retries for the FT
  cells (meta1b, meta2b needed 2-3 resubmits each for the final seed's scoring; ours1b_e19 needed
  one) + the 3-seed frozen retrofit: 4 checkpoints × 3 seeds via 3 batched Polaris debug jobs
  (4/4/2-wide), each needing 1-2 resubmits for the train-then-score walltime split.

See `run_esad_ext_probe_polaris.sh` / `run_esad_ext_score.sh` (cells 7/8/9), `run_ft_seed.sh` /
`score_ft.sh` (cells 2/4/6, parameterized via `NAME`/`PROBE_CFG`/`EXPORT_CFG`),
`run_esad_cov_export.sh` (cells 1/3/5 denominator repair), and `run_esad_frozen_batch.sh` (the
3-seed frozen retrofit, batched via independent background processes not `mpiexec`) for the
exact commands.

### 4f. FROZEN seed-ensemble (free lever, no retraining), extending §4e (2026-08-15/16)

Extends [[seed-ensembling-generalizes-to-triplet]]'s free-lever family to ESAD's FROZEN probe
(user directed: frozen before FT, since frozen has no cache-per-seed complication — all 3 seeds
of a frozen checkpoint share the SAME feature cache, only head init/data-order differs, unlike
FT where the encoder itself changes per seed). Same mechanism as SAR's seed-ensemble
([[sar-seed-ensemble-result]]): merges each seed's `(frame, class, slot)` prediction buckets via
the EXISTING cross-window fusion machinery (`fusion.py`'s `fuse_iou_clustered`/`wbf_meanconf`/
`maxpick`, already used to consolidate overlapping-window votes) — a cross-seed vote is
structurally identical to a cross-window vote, so no new fusion math was needed, only a new
driver (`scripts/score_esad_seed_ensemble.py` on eagle) that runs all 3 seeds' checkpoints over
the same cache and merges before scoring. All 4 arms' per-seed baselines (0.1483/0.1271/0.1302/
0.1441 means) are read directly from each seed's existing `test_detection_ap_cov_fulldenom.json`
(§4e's own retrofit output) — no rescoring of numbers already on record, only the new ensemble
number is computed fresh. Scored on the full-denominator (`gt_full=11207`) protocol throughout.

**Correctness check:** every arm's re-derived seed mean matches §4e's published table exactly
(e159 0.1483, meta2b 0.1271, meta1b 0.1441; ours1b_e19 confirmed below) — same cached JSONs, no
new scoring bug introduced.

**Two protocols only, no best-of-N selection by the held-out metric** (same convention as
[[sar-head-ensemble-result]]/[[sar-seed-ensemble-result]] after that session's mid-work
retraction — see those memories for why a "best seed" number is invalid here too):

| scale | arm | baseline mean ± std | ensemble |
|---|---|---|---|
| 2B | e159 (CPT) | 0.1483 ± 0.0062 | **0.1569** |
| 2B | meta2b (raw) | 0.1271 ± 0.0111 | **0.1362** |
| 1B | ours1b_e19 (CPT) | 0.1302 ± 0.0082 | **0.1348** |
| 1B | meta1b (raw) | 0.1441 ± 0.0211 | **0.1658** |

(σ recomputed as population std of the 3 per-seed baseline values read from disk; ours1b_e19's
σ=0.0082 here is the true value from the 3 cited seeds — tighter than §4e's originally-quoted
±0.0196 for the same arm, likely a rounding/spread-statistic convention difference between this
pass and that one; the MEAN, 0.1302, is identical either way and is what every delta below uses.)

**Same-scale CPT-vs-raw deltas:**

| scale | protocol | CPT | raw | Δ (CPT − raw) |
|---|---|---|---|---|
| 2B | baseline | 0.1483 | 0.1271 | **+0.0213** |
| 2B | ensemble | 0.1569 | 0.1362 | **+0.0207** |
| 1B | baseline | 0.1302 | 0.1441 | **−0.0139** |
| 1B | ensemble | 0.1348 | 0.1658 | **−0.0310** |

**Reading — honest, including the uncomfortable part.** At 2B, CPT (e159) clearly and
consistently beats raw (meta2b) at BOTH baseline (+0.0213) and ensemble (+0.0207) — stable
across the ensembling operation, and (per §4e's own noise-floor bar of ~0.06 for this probe) this
delta does NOT clear that bar either, so §4e's standing verdict of "no confirmed CPT-vs-raw
frozen edge at 2B" is not overturned by ensembling — it stays a directional lead inside the noise
floor, not a proven win. **At 1B, raw (meta1b) already beat CPT (ours1b_e19) at baseline
(0.1441 vs 0.1302, per §4e) — the ensemble makes this WORSE for CPT, not better: the gap more
than DOUBLES (−0.0139 → −0.0310).** Both arms' ensembles rise over their own baselines (ours1b_e19
+0.0046, meta1b +0.0217), but meta1b's rise is ~5× larger, so the relative gap widens sharply.
meta1b's ensemble (0.1658) is now also higher than e159's own 2B CPT ensemble (0.1569), i.e. the
smallest, raw, non-surgical 1B checkpoint's ensembled score exceeds the largest
surgically-adapted checkpoint's ensembled score on this probe. This is not a new problem
introduced by ensembling — §4e already documented meta1b beating ours1b_e19 at baseline — but
ensembling makes CPT's position at 1B measurably worse, not better, mirroring
[[sar-seed-ensemble-result]]'s finding that a seed-ensemble's effect is not reliably
pro-CPT — it favors whichever arm has the most seed-to-seed disagreement to average out, and
meta1b (σ=0.0211, includes one outlier seed at 0.1739 already flagged in §4e as "a clear
outlier... largely responsible for the raw-1B-beats-CPT-1B read") has by far the widest spread of
any arm in this table (ours1b_e19 σ=0.0082, e159 σ=0.0062, meta2b σ=0.0111 — meta1b's is 2–3×
every other arm's).

**Mechanism, matching [[sar-seed-ensemble-result]]'s explanation exactly:** ensembling averages
out MORE noise on whichever arm's seeds disagree with each other most. meta1b's σ (0.0211) is
roughly double e159's (0.0062) and meta2b's (0.0111) — it has the most noise to close, and
closing it happens to push its ensembled score up past the 2B CPT ensemble. This is the same
mechanism, not a coincidence: two independent probes (SAR, ESAD) both show a seed-ensemble
widening a raw arm's apparent lead specifically where that raw arm carries outsized seed
variance relative to its CPT counterpart.

**Performance note, not a bug:** the ensemble-scoring step alone (after the cheap ~130–150s
3-seed inference pass) took **4544–4692s (~76–78 minutes)** consistently across all 4 arms
(meta1b 4565s, v1/e159 4578s, meta2b 4544s, ours1b_e19 4692s) — `fusion.py`'s `fuse_iou_clustered`
is an
O(n²) pairwise-IoU union-find per bucket, and seed-ensembling roughly triples the average bucket
size (up to 9 entries vs 3 for a single seed's cross-window-only fusion) across ~124K buckets ×
3 fusion variants × 2 denominators. **Anyone extending this further (e.g. to FT seeds, once the
cross-cache alignment problem is solved) should budget ~80 min/arm for the scoring step alone,
independent of the ~2.5 min inference cost** — a naive walltime estimate from inference time
alone will undershoot by >30×. A first debug-queue attempt at this work (1h cap) never reached
this step; capacity's 4h cap was required.

**Caveats:** all 4 arms complete (2026-08-16); no std on the ensemble side (single fused
prediction, not re-derivable without a 4th held-out seed); n=3 seeds/arm is the same
small-sample caveat as every ensembling result in this ledger; meta1b's σ is dominated by one
outlier seed (0.1739 vs 0.1308/0.1277) — a 4th seed would clarify whether that's real variance
or a fluke draw, same open question §4e already raised.

Reproduce: `scripts/score_esad_seed_ensemble.py` (eagle,
`/eagle/projects/tpc/leonardo_borgioli/esad_probe/`), driver
`scripts/run_esad_seed_ensemble.sh` (`qsub -v ARM=<arm> run_esad_seed_ensemble.sh`, one arm per
job — does not fit a shared job due to the per-arm scoring cost above). Related:
[[sar-seed-ensemble-result]] (same technique, SAR probe), [[sar-head-ensemble-result]] (the
best-of-N retraction this section's convention follows).

---

### 4g. Cross-arm protocol audit + the presence-label fairness bug (2026-08-17)

**Trigger.** The §4e table reads as "LemonFM (0.1537) beats our best frozen cell (0.1483)" —
i.e. our FM losing to a competitor FM. Three independent audits were run: (a) statistical
re-analysis of the published per-seed values, (b) a code audit of the probe for cross-arm
asymmetries, (c) a leakage investigation of LemonFM. Findings below; **the headline is that
the "loss" is not a loss, and the one real defect found favors the image arms.**

#### (a) The LemonFM "win" is a tie, and is statistically UNRESOLVABLE on this probe

Welch t-tests on §4e's own per-seed values (no new runs):

| comparison | Δ | Welch t |
|---|---|---|
| LemonFM − e159 (frozen) | +0.0054 | **+0.46** |
| LemonFM − meta1b | +0.0096 | +0.52 |
| LemonFM − ours1b_e19 | +0.0236 | +1.89 |
| LemonFM − meta2b | +0.0267 | +1.97 |

LemonFM's seed range [0.1320, 0.1679] **overlaps every V-JEPA arm**; its σ (0.0191) is 2.5×
e159's (0.0076). **Power analysis (pooled σ = 0.0146, 80% power, α=0.05):** resolving the
observed 0.005 frozen gap needs **~133 seeds/arm**; 0.010 needs 34; 0.020 needs 9. At ~78 min/arm
for the seed-ensemble scoring step (§4f), the frozen column **cannot rank encoders at any
feasible seed count** — it should be reported as a tie band, not a ranking.

By contrast **e159 FT (0.2005) vs LemonFM frozen (0.1538) = +0.0467 at t=+3.76, needing only
n=2** — already resolved at n=3, and clearing the challenge's best submission (0.1928). Note the
protocol asymmetry when citing this: **no external arm has an FT number at all**, so this is
"our FT vs their frozen," which must be stated, not elided.

#### (b) Code audit — three real asymmetries, all favoring the image arms; and one retraction

Audited resolution, token geometry, head capacity, spatial pooling, augmentation, optimizer
settings, seeds, and scoring population across all 7 frozen cells.

**Clean / genuinely matched:** resolution (all arms exported at 384 via one shared
`export_esad_cache.py:196` squash — LemonFM is in fact *extrapolated* from its native 224, while
V-JEPA runs at its native 384); augmentation (off for every cell in the table, verified from
cache manifests); LR/epochs/warmup/dropout/`pos_weight_cap` (the lemonfm and meta2b probe YAMLs
differ on exactly **3 lines**: `embed_dim`, `tokens_per_clip`, `temporal_tokens`); seeds (3/arm,
same mechanism); scoring population (all 7 cells verified `gt_full=11207`, `gt_cov=10619`,
`frames=5903`, and prediction density 2.961 vs 2.984 votes/frame — neither arm gets more
cross-window ensembling). **V-JEPA also gets 4× the spatial tokens (576 vs 144), which favors
V-JEPA on box regression** and is consistent with its lead on the oracle IoU/map50 columns.

**★★★ REAL DEFECT — the presence label is inconsistent with the frame it is scored on, and
only for the V-JEPA arms.** `build_esad_windows.py` set presence as the **union over both frames
of a tubelet** while boxes come from the **representative (later) frame only**; the scorer then
uses that presence sigmoid as the **detection confidence for the representative frame**
(`score_esad_detection_ap.py:160`, `token_rep_frame():58-63`). So a class in the early frame but
not the scored frame trains the model to fire on a frame whose GT does not contain it. Measured
rate of presence-positive (token,class) rows with no box in the scored frame:

| split | V-JEPA (tubelet=2) | image arms (tubelet=1) |
|---|---|---|
| train | **8.1%** | 0.0% |
| **val (the model-SELECTION split)** | **17.1%** | 0.0% |
| test (cov p0) | 3.3% | 0.0% |

**At tubelet=1 the span is a single frame, so the defect is 0% by construction.** This is
therefore **not a uniform handicap — it is a cross-arm fairness bug that penalizes only the
V-JEPA cells**, worst on the split that picks `best.pt`.

**FIXED 2026-08-17** — `--presence-mode {union,rep}` added to `build_esad_windows.py`
(`union` = legacy default so every cited manifest still reproduces byte-identically; `rep` =
presence from the representative frame only, consistent with both the box target and the scored
frame). `presence_mode` is recorded in each manifest's `geometry` block, and `_pres{mode}` is
appended to the output suffix so fixed manifests cannot be confused with legacy ones. Verified:
synthetic 48-frame window with a class in even frames only → `union` 24 mislabeled rows,
`rep` **0**; `union == rep` at tubelet=1 (image arms provably unaffected); default output
byte-identical to legacy. Backup: `build_esad_windows.py.pre_presmode_backup_20260817`.

**Two further undisclosed asymmetries, both favoring the image arms (NOT fixed — inherent):**
1. **2× the supervision per step.** Same 2,468 windows and identical steps/epoch, but the image
   arm carries 118,464 vs 59,232 labelled train token-rows, so its BCE/box loss averages over
   twice as many targets per gradient step.
2. **Head width is never normalized** — head width == encoder D (`esad_double_head.py:161-183`),
   so LemonFM's head (D=1536, **87.4M** params) is **~19% larger** than the 1B V-JEPA arms'
   (D=1408, 73.5M). It is smaller than the 2B arms' (D=1664, 102.6M). **This also undercuts
   §4e's "all externals sit below V-JEPA" reading**: SNX (512, 9.8M) and EndoViT (768, 21.9M) —
   the two arms that lose badly — carry heads **7–9× smaller** than every other cell.

**RETRACTED:** §4e's "image arms gain a 32-dilation layer" caveat is false — at
`presence_num_layers: 4` both geometries yield dilations **[1,2,4,8]**. See the struck caveat in
§4e.

#### (c) LemonFM leakage — investigated and REFUTED; do not make this claim

LemonFM = LEMON (formerly Surg-3M/SurgFM), Che et al., **arXiv:2503.19740**, CVPR 2026 —
**ConvNeXt-Large**, DINO-style self-distillation, 4,194 YouTube videos, pretrained at 224px.

- **Direct pHash test (run, not inferred):** ESAD test (RARP3, 6,088 frames) vs the 80
  prostatectomy-labeled LEMON videos (52,981 sampled frames) → **min Hamming 6, ZERO matches
  ≤5**; top-8 candidates verified at pixel level, max NCC **0.71** against a calibrated
  positive-control floor of **0.764**. The closest LEMON match is no closer than *a different
  patient in the same operating room*. Uploader channels resolved for all 4,194: zero SARAS /
  San Raffaele / CAMMA affiliations.
- **Structurally near-impossible anyway:** ESAD is 4 San Raffaele RARP videos distributed
  **only as 1-fps JPEG frames** (no video-container form exists online); the SARAS YouTube
  channel has 9 promo clips, and its action-detection clips state they use **3D-printed
  phantoms**, not patients.
- **What IS citable:** LEMON documents **no eval decontamination of any kind** (zero occurrences
  of leak/overlap/dedup/contamin across all 4 arXiv versions and the README). That is a
  disclosure gap, not evidence of contamination.
- **★ SurgeNetXL has DOCUMENTED ESAD in its pretraining** (arXiv:2501.09436 Table 2:
  `SurgeNetRARP` includes ESAD, 4 videos, 47,282 frames) — **and it scores the LOWEST external
  arm (0.1162)**. Direct evidence that ESAD-frame exposure is not worth much on this probe,
  independently undercutting any "leakage explains LemonFM" story.
- **Our own house — one live hazard.** `scripts/pack_images_pbs.sh:48-51` packs ESAD
  **train+val+test** into `esad_img` (53,370 frames), deliberately. **No shipped checkpoint uses
  it** — all 56 `params-pretrain.yaml` under `/flare/ModCon/ngetty/checkpoints/` parsed with a
  YAML parser: zero reference `esad_img`, any `_img` source, or an active `img_data` branch — so
  every ESAD number on record is clean. But it is referenced by
  `configs/.../vitg384_cooldown_64f_imgbranch.yaml:121` (`[DRAFT TEMPLATE]`), and running that
  config would silently void this entire section. Same failure mode as `grasp` → `grasp_noleak`.
  See [[esad-test-frames-packed-into-corpus]].
- **Still open:** our LEMON pHash gate screened only against `yt_robotic_chole` + `surgenet_robotic`
  (`build_phash_ref.py:47`) — **ESAD was never in the reference pool**, so a LEMON→ESAD path into
  *our* corpus is unmeasured. Test specced in [[lemon-phash-gate-never-checked-esad]] (~10 min).
  Note a null there is weak evidence (pHash is crop-blind).

**How to report §4e until the reruns land:** frozen = a tie band spanning 0.127–0.154 with
nothing separable; FT = the only ESAD comparison that clears its noise floor. Do not cite
CPT-vs-Meta ESAD deltas as clean ([[probe-builder-fps-audit-2026-08-13]] lists this builder as
UNVERIFIABLE for the fps-drift class), and do not lead with a leakage argument.

---

### 4h. The presence-mode fix, measured — a clean NULL (2026-08-17)

The §4g presence defect (union-over-tubelet target scored against the representative frame;
8.09% of train / **17.12% of val** presence-positive rows mislabeled, 0% for the tubelet=1
image arms) was fixed and all four V-JEPA arms were re-run at 3 seeds.

**Method.** `build_esad_windows.py --presence-mode rep` (default stays `union`, so every cited
manifest still reproduces byte-identically). The FEATURE caches were NOT rebuilt: presence_mode
does not touch pixels, so the encoder output is bit-identical, and re-exporting would have been
pure waste. Instead `train_esad_double_head.py` gained `--train/--val-presence-override`, which
substitutes the presence tensor at load time keyed by `window_id`
([[label-only-fixes-dont-need-reexport]] — copying the caches would have duplicated ~492 GB to
change ~0.005% of the bytes). `compute_pos_weight()` was fixed to honor the override; it reads
shards directly and would otherwise have weighted the loss by the OLD union labels while
training on the new ones — silent, non-crashing, and biasing every run.

**Verification before any result was trusted** (jobs 7486302 + 7486631, independently
reproduced on two nodes): `features_differ=0`, `boxes_or_mask_differ=0`,
`superset_violations=0`, `cleared=2004/11706 = 17.12%` on val, `VERDICT: PASS`. All 12 result
cells re-verified on one population: `gt_full=11207`, `gt_cov=10619`, `frames=5903` — 12/12.

| arm | union (3-seed) | **rep (3-seed)** | Δ | Welch t |
|---|---|---|---|---|
| v1 / e159 (CPT 2B) | 0.1483 ± 0.0076 | **0.1395 ± 0.0205** | −0.0088 | −0.70 |
| meta2b (raw 2B) | 0.1271 ± 0.0136 | **0.1278 ± 0.0118** | +0.0007 | +0.07 |
| ours1b_e19 (CPT 1B) | 0.1302 ± 0.0100 | **0.1339 ± 0.0115** | +0.0037 | +0.43 |
| meta1b (raw 1B) | 0.1441 ± 0.0258 | **0.1459 ± 0.0222** | +0.0018 | +0.09 |

**★ RESULT: a clean null on all four arms.** Every |Δ| ≤ 0.0088, an order of magnitude inside
this probe's ~0.06 resolution limit, and every |t| ≤ 0.70. **Fixing a defect that mislabeled
17% of the model-selection split did not move the frozen metric.** This was pre-committed as the
expected outcome before any cell was scored.

**A hypothesis raised at n=3-on-one-arm and REFUTED at n=12.** After v1 alone came back with
sd 0.0076 → 0.0205 (2.7×), the session floated "the fix destabilizes checkpoint selection."
The full table kills it: sd rose on **2 of 4** arms and FELL on the other two (meta2b 0.86×,
meta1b 0.86×), mean sd 0.0143 → 0.0165 (1.16×). v1's jump was seed luck, not a mechanism —
a reminder that a variance claim needs the whole table, exactly like [[ab-window-truncation-trap]].

**What it does NOT change.** The CPT-vs-raw picture is unmoved: 2B stays a directional CPT lead
inside the floor (+0.0213 → +0.0117), 1B stays a raw-favoring tie (−0.0140 → −0.0120). Against
LemonFM's frozen 0.1538, every rep arm is still statistically indistinguishable (|t| ≤ 2.00),
and **our FT 0.2005 vs Lemon 0.1538 (+0.0467, t=+3.76) remains the only resolved comparison in
this section.**

**Why the fix ships anyway.** It was a genuine cross-arm fairness defect that penalized only the
V-JEPA cells (0% at tubelet=1 by construction), and leaving a known-wrong label in the builder
to be rediscovered later is worse than a null result. **The honest statement is "we removed a
confound and it changed nothing measurable," not "we fixed the probe."** Consistent with §4g's
power analysis: the frozen column cannot resolve ~0.01-scale effects at any feasible seed count
(~133 seeds/arm for 0.005), so a null here was the *only* outcome this protocol could have
produced short of a large effect.

Reproduce: `run_esad_frozen_presrep.sh` (train+score) / `run_esad_score_presrep.sh`
(score-only), chained by `drive_presrep_v2.sh`. **Batch at most 2 combos per debug job** —
training is ~40 min for 4-on-4-GPUs and scoring is **~28 min/combo SERIALLY** (fusion.py's
O(n²) `fuse_iou_clustered`, the same cost §4f logged at ~78 min/arm). Jobs 7486632 and 7487240
were both killed at the 1 h cap for packing 4.

---

### 4i. FT-vs-FT: the external backbones fine-tuned on our protocol (2026-08-19)

**Why.** §4e compared **our FINE-TUNED** numbers against the external backbones' **FROZEN**
ones — LemonFM and SurgeNetXL had no FT cells at all. That asymmetry is not defensible in
review. This section closes it: both externals were put through the identical last-4 unfreeze
protocol the four V-JEPA arms already used, 3 seeds each, scored on the same population.

**The table. n=3 on every cell; `gt_full=11207`, `gt_cov=10619`, `frames=5903` asserted
per-cell by the collector (it raises rather than print a mixed-population row).**

| arm | frozen | **FT (last-4)** | FT gain | t(gain) |
|---|---|---|---|---|
| **e159 (CPT 2B)** | 0.1483 ± 0.0076 | **0.2005 ± 0.0098** | **+0.0521** | +7.27 |
| meta2b (raw 2B) | 0.1271 ± 0.0136 | 0.1936 ± 0.0409 | +0.0665 | +2.67 |
| *SARAS-ESAD best submission* | — | *0.1928* | — | — |
| meta1b (raw 1B) | 0.1441 ± 0.0258 | 0.1855 ± 0.0099 | +0.0414 | +2.59 |
| ours1b_e19 (CPT 1B) | 0.1302 ± 0.0100 | 0.1832 ± 0.0048 | +0.0530 | +8.30 |
| **LemonFM (ConvNeXt-L)** | 0.1538 ± 0.0191 | **0.1585 ± 0.0048** | **+0.0047** | **+0.42** |
| **SurgeNetXL (CAFormer-S18)** | 0.1162 ± 0.0007 | **0.1257 ± 0.0039** | +0.0094 | +4.11 |

Per-seed FT: LemonFM 0.1549 / 0.1640 / 0.1566; SurgeNetXL 0.1299 / 0.1249 / 0.1222.

**Per-threshold breakdown (2026-08-19).** The AP_mean above is macro-averaged over IoU
{0.10, 0.30, 0.50}; per-threshold values were requested but not originally reported. All from
the same `test_detection_ap_masked_fulldenom.json` outputs as the table above — same
`gt_full`/`gt_cov`/`frames` population, mean ± sample sd over 3 seeds:

| arm | frozen AP10 | frozen AP30 | frozen AP50 | FT AP10 | FT AP30 | FT AP50 |
|---|---|---|---|---|---|---|
| e159 (CPT 2B) | 0.2080 ± 0.0119 | 0.1689 ± 0.0083 | 0.0682 ± 0.0028 | 0.2981 ± 0.0258 | 0.2245 ± 0.0058 | 0.0788 ± 0.0024 |
| meta2b (raw 2B) | 0.1750 ± 0.0151 | 0.1413 ± 0.0176 | 0.0649 ± 0.0132 | 0.2828 ± 0.0431 | 0.2243 ± 0.0482 | 0.0736 ± 0.0320 |
| meta1b (raw 1B) | 0.2278 ± 0.0319 | 0.1587 ± 0.0363 | 0.0458 ± 0.0099 | 0.2699 ± 0.0177 | 0.2145 ± 0.0072 | 0.0721 ± 0.0080 |
| ours1b_e19 (CPT 1B) | 0.1966 ± 0.0122 | 0.1495 ± 0.0132 | 0.0444 ± 0.0055 | 0.2639 ± 0.0033 | 0.2128 ± 0.0043 | 0.0726 ± 0.0068 |
| **LemonFM** | 0.2739 ± 0.0270 | 0.1461 ± 0.0248 | 0.0412 ± 0.0211 | 0.2921 ± 0.0089 | 0.1509 ± 0.0067 | 0.0325 ± 0.0030 |
| **SurgeNetXL** | 0.2309 ± 0.0073 | 0.0997 ± 0.0035 | 0.0181 ± 0.0042 | 0.2346 ± 0.0111 | 0.1224 ± 0.0075 | 0.0199 ± 0.0016 |

*SARAS-ESAD best submission (reference, not a probe row):* AP10=0.2763, AP30=0.2205, AP50=0.0816.

**Reading the per-threshold split.** LemonFM's near-null aggregate FT gain (+0.0047) is not a
wash across thresholds — it is a **trade**: AP10 rises (+0.018) while AP30 (−0.005) and AP50
(−0.009) fall, which macro-averages to roughly zero. That is a genuinely different failure mode
from SurgeNetXL, whose FT gain is small but **positive at every threshold** (+0.004/+0.023/
+0.002), and from every V-JEPA arm, where FT raises AP10/AP30/AP50 together (e.g. e159:
+0.090/+0.056/+0.011; meta2b: +0.108/+0.083/+0.009). Fine-tuning LemonFM's last 4 ConvNeXt
blocks buys easier (loosely-localized) detections at IoU=0.10 but does not tighten
localization — consistent with, but not proof of, the ceiling-vs-adaptability question already
flagged as unresolved above. Per-seed values used for these means are in the raw
`test_detection_ap_masked_fulldenom.json` files under
`/eagle/projects/ModCon/ngetty/esad_probe/runs/esad_double_{arm}_ft_last4_s{0,1,2}/` (externals)
and the corresponding `_frozen_s{seed}` / `esad_double_v1_ft_last4_s{seed}` dirs (V-JEPA arms).

#### Reading it — three claims at three different confidence levels

**1. ONE cross-arm comparison clears the ~0.06 noise floor: `e159 FT − SurgeNetXL FT = +0.0748`
(t=+12.26).** That is claimable.

**2. Everything against LemonFM is DIRECTIONAL, not certified.** Our best beats it by +0.0420
(t=+6.64) and all four V-JEPA arms are ahead (+0.025 to +0.042) — but every one of those deltas
is **below the floor**. §4g used that same floor to dissolve LemonFM's apparent frozen lead;
invoking it there and ignoring it here would be cherry-picking the standard. The honest
statement is **"LemonFM ties us frozen and trails us fine-tuned, directionally, on a probe too
noisy to certify the margin."** The high t-values come from unusually tight 3-seed spreads, not
a large effect; n=3 t-tests are fragile.

**3. ★ The within-backbone FT gain is the substantive result, and it is clean.** LemonFM gains
**+0.0047 (t=+0.42) — a null** — while every V-JEPA arm gains +0.041 to +0.067. Because this is
a WITHIN-arm comparison it does not pay the cross-arm floor. Note SurgeNetXL *does* gain
significantly (+0.0094, t=+4.11), so "image backbones don't respond to fine-tuning" is too
broad — **specifically LemonFM does not move.**

**Mechanism NOT established — do not assert one.** Each gain is measured against that arm's own
frozen baseline, and the baselines differ (0.1162–0.1538). LemonFM starts highest among the
externals and gains least, so a ceiling/regression-to-the-mean effect is a live alternative to
"V-JEPA features are more adaptable." Against that: the ordering is not monotonic in the
baseline (meta1b starts at 0.1441, above meta2b's 0.1271, yet still gains +0.041 vs LemonFM's
+0.005 from a similar 0.1538 start). Report the observation, not the cause
([[no-lazy-cause-labels]]). The discriminating experiment is the param-matched arm below.

#### Disclosures that must travel with this table

1. **`last-4 blocks` is NOT capacity-matched.** Measured: 4 blocks = 8.3% of ViT-G's encoder
   (153.4M), **31.4% of ConvNeXt-L** (61.7M), **45.9% of CAFormer-S18** (10.7M). The merge
   counts differ too — 48 tuned tensors for a V-JEPA ViT, **36** for ConvNeXt, **32** for
   CAFormer. The externals were given 3.8–5.5× MORE relative encoder capacity than we were,
   which cuts *against* a capacity explanation of their flat gains. A param-matched arm
   (~8.3% ≈ last-1 block for both externals) is the natural sensitivity check, ~2.5 h/seed.
2. **`--unfreeze-output-norm` is a no-op for both externals** and was left off. ConvNeXt's final
   norm is `classifier[0]`, which the adapter replaces with `nn.Identity` before reading the
   pre-pool map; CAFormer's `.norm` applies to the POOLED vector while the adapter takes
   `feats[-1]`. Neither is on the head's path. The V-JEPA arms had the same flag off, so the
   protocol matches — but the flag means something different per architecture.
3. **LemonFM's head (87.4M) EXCEEDS its unfrozen encoder capacity (61.7M)**, so "FT" is
   structurally a different operation for that arm than for CAFormer (9.8M head vs 10.7M
   encoder). Inherited from §4g's head-width finding, not introduced here.
4. **Fairness mask applied.** At tubelet=1 the image arms natively reach ~100% of test frames
   (6,088) vs the V-JEPA arms' 94.8% (5,903). All external cells are scored with
   `--restrict-frame-stems` down to the identical 5,903-frame set — the same protocol their
   frozen cells used. The unmasked number is a secondary check, never a comparison column.

#### Port notes (six V-JEPA assumptions, three of them silent)

`train_esad_unfreeze.py` / `models.py` / `score_ft.sh` all assumed a V-JEPA ViT. Full writeup in
[[cross-arch-ft-port-esad]]; the three SILENT ones were: (a) `.train()` re-arms ConvNeXt's
`StochasticDepth` at p=0.47–0.50 — measured dropping **5/6 samples** — which the V-JEPA arms
never had and which would have handicapped the competitor arm in our favour
([[convnext-stochastic-depth-ft-trap]]); (b) no geometry assert on the window manifests, so a
tubelet=2 manifest would mis-attribute every box rather than crash; (c) the merge anchored on
`blocks.`, giving **0/2** matches on both externals and silently exporting a FROZEN encoder as
"fine-tuned." All three are now guarded. **The tubelet assumption turned out to be baked into
THREE separate seams** (trainer, scorer invocation, cache export) — audit all three before
porting another image backbone.

Reproduce: `run_ft_seed.sh` (NAME/PROBE_CFG/EXPORT_CFG/TRAIN_WIN/VAL_WIN/UNFREEZE_N),
`score_ft_external.sh`, `collect_ft.py`. **Do not use `score_ft.sh` for an image backbone** —
its two-phase tubelet=2 test manifests raise `IndexError: rep_frames[j]`. Sophia co-tenancy
killed two jobs at 5 s (`Exit_status=143`); the scorer now picks a free GPU and refuses to start
without one ([[sophia-shares-gpus]]).

---

## 5. P1 objective + P2 sampling-temperature ablations — and the BUDGET reframe (2026-08-08/12)

Canonical machine-readable source for all rows here:
`/flare/ModCon/ngetty/checkpoints/.ablation_probe_state/results.tsv`.
All arms are 1B ViT-g @384, scored on the triplet IVT probe. **Probe protocol is
byte-identical across every arm** — the `configs/heads/triplet/trip_*.yaml` files
differ ONLY in `folder`/`tag`/`dump_probs_path`/`checkpoint` (verified by diff),
so these are directly comparable to each other and to `meta1b`.

### P1 — objective ablation (tempmask vs subtube), 3 seeds

| arm | budget | IVT mean ± std | median |
|---|---|---|---|
| `obj_tempmask` | e39 (40ep) | 28.60 ± 0.33 | 28.72 |
| `obj_subtube` | e39 (40ep) | 27.25 ± 0.35 | 27.07 |
| `obj_tempmask` | **e80 (80ep)** | **28.98 ± 0.12** | 28.95 |
| `obj_subtube` | e80 (80ep) | 27.14 ± 0.28 | 27.22 |

**P1 verdict:** the temporal-masking objective beats sub-tube masking by **~1.7 IVT**,
consistent at both e39 and e80 — not a training-budget artifact. Compute-matched to
within ±5% encoder tokens by design (`scripts/gen_1B_objective_configs.py`); encoder
tokens dominate cost ~22×, so the comparison is not confounded with FLOPs.

**⚠ But tempmask's margin over RAW META is only +0.44** (28.98 vs meta1b 28.54,
t≈2.3). The 1.73 is a within-P1 ranking — it says subtube is worse, not that
tempmask is a large win. **There is no budget-matched control:** `abl_full` was
never trained past e39, so no e80 control arm exists. Do not cite "+1.7" as a
CPT-over-baseline gain.

### P2 — sampling temperature (t050 vs t075), 3 seeds, FULL 240-epoch budget

| arm | budget | IVT mean ± std | median |
|---|---|---|---|
| `samp_t050` | e240 | 30.15 ± 0.30 | 30.04 |
| `samp_t075` | e240 | 29.82 ± 0.15 | 29.77 |

**P2 verdict on its own axis: NULL.** t050 is nominally +0.33 over t075 but the
spreads overlap. Sampling temperature in [0.5, 0.75] is not a lever; if revisited,
use a wider range (0.25 vs 1.0), not incremental steps. See
`p2-sampling-temperature-null-result.md`.

### ★★ THE CROSS-STUDY READ: budget, not temperature, is what moved

P2's two arms are the ONLY arms in the campaign trained to 240 epochs; everything
in P0/P1 ran 40–80. Lined up against the shared `meta1b` anchor:

| arm | budget | IVT mean | sd | Δ vs meta1b 28.54 | t |
|---|---|---|---|---|---|
| `abl_full` e39 | 40ep | 29.14 | 1.93 | +0.60 | 0.5 |
| `openhin` e79 | 80ep | 28.52 | 0.68 | −0.02 | −0.1 |
| `obj_tempmask` e80 | 80ep | 28.98 | 0.12 | +0.44 | 2.3 |
| **`samp_t050` e240** | **240ep** | **30.15** | 0.30 | **+1.61** | **6.5** |
| **`samp_t075` e240** | **240ep** | **29.82** | 0.15 | **+1.28** | **6.5** |

**The 240-epoch arms are the only CPT checkpoints whose FROZEN-probe margin over
raw Meta clears noise decisively**, and they beat the best 80-epoch arm by ~1.2.
This matters because §1/§2's standing conclusion — "CPT ties raw Meta frozen; only
fine-tuning surfaces a gain" — was drawn entirely from 20–80 epoch runs.

### ★★★ P3 BUDGET CONTROL — the within-arm test. CONFIRMED (job 8751525, 2026-08-13)

The §5 read above was cross-study, so it was qualified: `samp_t050` differs from
`abl_full` in `epochs 80→240` AND `warmup 5.0→15.0` AND a 3×-longer LR cosine, and
P2 scored `latest.pth.tar` while P0/P1 scored numbered checkpoints. **All of those
are now held FIXED.** `scripts/p3_budget_e79_capacity.sh` probed `samp_t050`'s own
banked `e79.pth.tar` (verified epoch field 80, 494 encoder keys) against its scored
e239 — same arm, same corpus, same temperature, same seeds, byte-identical probe,
**only training budget differs**:

| checkpoint | budget | IVT mean | sd | seeds |
|---|---|---|---|---|
| `samp_t050` e79 | 80ep | **28.31** | 0.06 | 28.37 / 28.25 / 28.31 |
| `samp_t050` e239 | 240ep | **30.15** | 0.30 | 29.92 / 30.04 / 30.49 |

- **Budget delta = +1.84 IVT, SE 0.177, t = 10.4.** The largest single-arm effect in
  this campaign, and the e79 spread (0.06) is the tightest measured anywhere in it.
- **e79 vs raw meta1b (28.54) = −0.23, t = −1.26 — INDISTINGUISHABLE.** At 80 epochs
  surgical CPT buys nothing on a frozen probe. The entire CPT-over-Meta margin
  appears between epoch 80 and epoch 240.

**BUDGET CONFIRMED as the driver; the P2 temperature null stands unchanged.**

**What this does to the rest of the ledger.** §1/§2's standing conclusion — "CPT ties
raw Meta frozen; only fine-tuning surfaces a gain" — was drawn entirely from 20–80
epoch checkpoints, i.e. from the regime this test shows is *pre-onset*. Undertraining,
not the frozen readout, is the more likely reason every P0/P1 arm tied. It also
asterisks the composition null in §5/§1: all 7 arms were screened at 40–80 epochs, so
they measured composition inside the dead zone. That does not make the null wrong, but
it does mean **"no data composition matters" was established at a budget where nothing
matters yet.** Re-running composition at 240ep is expensive and low-priority, but the
null should no longer be cited as budget-independent.

**Open (cheap):** `e159` is banked — one probe would locate where between e79 and e239
the gain turns on, i.e. whether 240 epochs is necessary or merely sufficient.

See `budget-not-temperature-is-the-p2-signal.md`.

Verification notes: all 6 P2 seeds confirmed at epoch 25 (converged, not the
progressive-dump trap that produced the bogus "lemonout 23.30"); `t050 s0`
independently re-scored and reproduced 29.92 exactly.

---

## 6. Seed ensembling — a free lever, discovered via GraSP and confirmed on triplet (2026-08-13)

**Origin.** GraSP's 3 LR-heads (§3) were found to ensemble to a real win on the
fixed build: raw meta2b best-head 75.65 → 3-head-mean 79.77 (+4.12); CPT v2_e324
79.68 → 82.00 (+2.32). Both clear TAPIS SOTA (76.72) on the ensemble. The
mechanism (`eval_grasp_map_cached.py`): average the SOFTMAX PROBABILITIES across
heads before scoring, not the scores after. This generalizes the 2026-07-12
"Head ENSEMBLE" finding above (§3, "NEGATIVE for strong ckpts") rather than
contradicting it — that study found ensembling hurts when one member is a clear
outlier (fs_e159's head0 was 12pt below its winner) and helps when members
roughly agree. Today's 3 heads sit within ~3pt of each other on the fixed build,
which is exactly the regime the July finding predicted would win.

**Generalized to triplet (job-free — pure post-hoc scoring of existing NPZ
dumps).** Extended `scripts/aggregate_triplet_seeds.py` with `--ensemble`: same
mean-of-probabilities trick, applied across the 3 (or 2) independently-SEEDED
triplet runs every arm already has, via the CANONICAL `compute_triplet_map.py`
functions (imported, not reimplemented). Verified precondition: seed NPZ label
arrays are byte-identical (same val order) before averaging — asserted in code,
not assumed.

**Result: ensemble beats every individual seed's best score in 28/37 scored
arms (76%), mean +0.49 IVT, up to +2.05. Ensemble beats mean-of-seed-scores in
37/37 (100%)** — the universal win is the Jensen's-gap effect (averaging
probabilities before scoring dominates averaging scores after); the partial win
is whether it also beats best-of-seed, which depends on seed agreement exactly
as GraSP predicted:

| arm (frozen, 3 seeds) | best-seed | mean | ensemble | Δ vs best |
|---|---|---|---|---|
| meta1b (raw 1B) | 28.84 | 28.54 | **30.47** | +1.63 |
| ours1b_e19 (CPT 1B) | 30.28 | 29.61 | **31.07** | +0.79 |
| meta2b (raw 2B) | 30.45 | 29.97 | **31.05** | +0.60 |
| ours2b_e159 (CPT 2B) | 29.90 | 29.53 | **31.08** | +1.18 |
| v2_e324 (poisoned by openh) | 19.72 | 19.53 | 21.74 | +2.01 |

**The 9 losses (best-seed beat ensemble) are exactly the tight arms**: `samp_t050`
(σ 0.25), `samp_t050_e79` (σ 0.05), `samp_t050_e159` (σ 0.45), `samp_t075`
(σ 0.12), `obj_subtube_e39` (σ 0.29), `snx` (σ 0.16) — every one has seed std
≤1.6 and most are well under 1. **When seeds already agree, there is nothing to
average away and best-of-3 is as likely to be the lucky high draw; ensembling
only reliably rescues NOISY checkpoints**, not your tightest, most-converged
ones. This is the GraSP mechanism confirmed on an independent probe and an
independent diversity axis (seed, not LR).

**★ CPT-vs-raw delta under ensembling — the one comparison that matters:**

| | best-of-seed Δ (CPT−raw) | ensemble Δ (CPT−raw) |
|---|---|---|
| 1B frozen | +1.44 | **+0.60** |
| 2B frozen | −0.55 | **+0.02** |
| 1B fine-tune | +2.83 | +2.30 |
| 2B fine-tune | +0.81 | +1.17 |

**Frozen deltas SHRINK under ensembling** (2B goes from a small CPT loss to a
dead tie) — because ensembling closed out seed noise that had been randomly
favoring CPT's best-of-3 draw, not because it revealed a hidden CPT advantage.
This is the more trustworthy read precisely because it does not uniformly favor
CPT. **Fine-tune deltas hold** (+2.30, +1.17, both still solidly positive),
independently corroborating [[fine-tune-beats-frozen-readout]]'s central claim
that FT — not the encoder-vs-raw distinction — is where CPT's benefit lives.

**Caveats — do not yet cite ensemble numbers as replacing the ledger's existing
rows:**
1. **n=3 (or n=2) is a very small ensemble.** "Ensemble > best-of-3" partly
   reflects that best-of-3 is a biased-HIGH statistic (the max of 3 noisy
   draws), not proof that mean-of-probabilities is the better estimator of the
   true score. A held-out 4th seed would test whether the ensemble actually
   generalizes better, not just fits these 3 draws.
2. **2-seed arms (`*_f001`, `*_f010`, `v2_e324`, `v2_final`) show the largest
   deltas (up to +2.05)** — consistent with more exploitable diversity at only 2
   seeds, but also the noisiest sample; weight these less than 3-seed arms.
3. This closes out SEED-diversity as an ensembling axis for triplet. LR-diversity
   (GraSP's original axis) has not been tested on triplet, and seed×LR combined
   (new training, not free) remains unexplored on any probe.
4. **Not yet extended to SAR-RARP50** — `eval_segmental_f1.py` has no
   `--dump-probs` path; seed checkpoints already exist
   (`meta1b/meta2b/ours1b_e19/ours2b_e159` × 3 seeds) so this is a small code
   addition away from being equally free. ESAD would need new seeded training
   runs (none exist) plus scorer changes; its adjacent WBF (box-fusion) lever
   was tested and came back NULL (0.1403→0.1405), a caution that not every
   probe's outputs average as cleanly as classification probabilities.

Reproduce: `python scripts/aggregate_triplet_seeds.py --root
/flare/ModCon/ngetty/surg_2_1_v2_final/probes/triplet --ensemble --json-out
<path>`.

---

## Open items / pending measurements

- [x] **★★★ BUDGET CONTROL: `samp_t050` e79 vs e239** — DONE 2026-08-13 (job 8751525),
      **CONFIRMED.** e79 = 28.31 ± 0.06 vs e239 = 30.15 ± 0.30 → **+1.84 IVT, t = 10.4**,
      within-arm with everything but budget held fixed. And e79 ≈ raw meta1b (−0.23,
      t = −1.26): at 80 epochs CPT does nothing frozen. Next production CPT should be
      **~240 epochs, not 80**. See §5's P3 block.
- [x] **Where does the budget gain turn on?** DONE 2026-08-13 (job 8751781): e159
      (3.07M clips) = 29.12 ± 0.56, between e79 (28.31) and e239 (30.15) — NO
      SATURATION, roughly linear in samples. e159 realizes only 44% of the total
      gain. Do not shorten the production run to 160ep-equivalent. See §5's P3b
      block.
- [x] **★★ SAR HEAD-ensemble (mean of the 3 LR-heads within a checkpoint)** — DONE
      2026-08-15 (job 8758553), see §2's new "Head ENSEMBLE" block. Different axis
      from the seed-ensemble asked for below (heads, not independently-trained
      seeds) but the same free-lever family. Result (head0-baseline vs mean-ensemble,
      the only two valid-no-peeking protocols): ensemble improves CPT-vs-raw
      significance at both scales (1B 1.05σ→2.80σ, 2B 1.33σ→1.55σ) — a genuine,
      if partial, revision of the §2 ★ table's "STATISTICAL TIE everywhere" verdict.
      (An earlier draft of this item cited a "best-of-3-head" number — RETRACTED,
      it selected the head using the TEST F1@10 metric being reported, i.e.
      test-set peeking; not a valid statistic, see §2's caveat block.)
- [x] **★★ SAR SEED-ensemble (independent axis from the head-ensemble above)** —
      DONE 2026-08-15 (job 8758825), see §2's "SEED ensemble" block. Result was
      NOT the same shape as predicted: 2B's CPT-vs-raw delta GREW under
      seed-ensembling (+1.06σ1.32 → +1.69), 1B's shrank slightly (+0.52σ1.06 →
      +0.44) — a genuinely mixed, scale-dependent pattern, unlike the head-axis
      result (consistent gain both scales) or triplet's frozen seed-ensemble
      (consistent shrink both scales). Confirms ensembling's effect on a
      CPT-vs-raw delta is not a fixed direction to assume a priori.
- [ ] **A held-out 4th seed on the key ensembling comparisons** (meta1b, meta2b,
      ours1b_e19, ours2b_e159) to test whether the ensemble genuinely predicts
      unseen data better, not just fits the 3 existing draws (§6 caveat 1).
- [ ] **Re-read the composition null at full budget?** All 7 P0 arms were screened at
      40–80 epochs, which P3 now shows is pre-onset. The null is not wrong but is
      budget-scoped; do not cite it as budget-independent. Low priority (expensive),
      but if any composition arm is ever revived it must run to ~240ep.
- [ ] **★★ GraSP official-build re-probe** (job 8751526, launched 2026-08-12,
      debug-scaling 4n, meta2b export phase). The `build_grasp_ctx_official.py`
      rebuild (commit 6649109) fixed all three builder defects that make every number
      in §3 uncitable, and its 65 GB of clips + `configs/heads/grasp/official_ctx16/`
      configs had sat UNRUN since 2026-08-03. Running `meta2b` (raw baseline) first;
      then `v2_e324` (CPT arm) via the same orchestrator invocation. **Expect a LOWER
      absolute than the old 69.31** — the UI shortcut is gone; that is the correct
      trade, a citable number beats a higher confounded one. Until both land, §3
      remains dark and no CPT-vs-raw GraSP claim can be made.
- [ ] **No e80 control for P1.** `abl_full` stopped at e39, so `obj_tempmask` e80 has
      no budget-matched control arm. If the objective lever is to be claimed as a
      production recipe change, train `abl_full` to e80 (or probe tempmask at e39 vs a
      240ep tempmask arm) rather than comparing across budgets.

- [x] **ESAD §4: explain why run order correlates with score (e159 first-run-best pattern).**
      Flagged 2026-08-05, RESOLVED same day by analysis — checked all 4 metrics against run
      order, found no monotonic/order-driven trend (2nd/meta1B is simultaneously best on
      well_map and worst on mean_iou/map50); spread is consistent with the existing ~3 mAP
      single-seed noise floor. Not a protocol artifact. See §4's resolved concern block.
- [ ] **ESAD §4: 3-seed replication of the CPT-vs-raw deltas** (well_map ±0.03, ~2.6–3.5 mAP
      points) — still needed before calling either the 2B (+) or 1B (−) delta real; both sit
      inside/at the edge of the noise floor on a single seed.
- [ ] **ESAD: cross-seed probability-ensemble, scoped but NOT executed 2026-08-15.** Unlike
      SAR/GraSP (a single checkpoint trains multiple heads whose outputs the scorer already
      computes), ESAD's `ESADDoubleHead` trains one presence+box head per run; the only existing
      fusion in `score_esad_detection_ap.py` averages predictions **across overlapping windows
      within one run**, not across independently-trained seeds. The 3-seed FT checkpoints (§4e's
      `esad-unfreeze-beats-frozen` result) each re-export their own test cache (encoder changes
      under FT — see [[esad-probe-leo-eagle-locations]]'s stale-cache trap), so cross-seed
      averaging would need new code to align predictions across differently-exported caches, not
      a free re-score. Also: this probe's one prior ensembling attempt (multi-box-per-class,
      §4c) came back a confirmed structural NEGATIVE (shared per-class confidence made duplicate
      slots collapse into phantom FPs, ~halving AP) — a caution against assuming any new
      ensembling axis here is free money. Deferred as a scoped next-step, not a dead end.
- [x] ~~**ESAD real confidence-ranked detection AP, comparable to SARAS-ESAD challenge AP_mean
      (19.28)** — DONE 2026-08-05 (job 7348633, §4b). All 4 checkpoints score AP_mean
      0.170–0.180 on a 1816/6223-frame (29.2%) subset — within noise of each other, above the
      challenge baseline (~0.12–0.15), within ~0.01–0.02 of the winning submission. Coverage
      gap is structural (tubelet=2 half-frame limit + gap-aware windowing), not a bug.~~
      **↑ WRONG, corrected 2026-08-05 (§4b-corrected).** Those AP values cover only 32% of test
      GT (3587/11207). On the paper's denominator: **0.0530 (e159), a ~3.6× gap to 0.1928, and
      BELOW the organizers' baseline.** The coverage gap is real but *not* fully structural —
      see the coverage item below.
- [x] ~~**ESAD §4b-followup: WBF + full-frame-coverage detection AP** — coverage expansion
      made AP *worse* for all 4 checkpoints (−0.004 to −0.015), not better.~~
      **↑ REVERSED 2026-08-05.** That comparison used a moving denominator (3587 vs 7085 GT).
      On a fixed full-test denominator coverage expansion is **+89% (0.0530 → 0.0999)** — the
      largest measured win in this probe. WBF remains genuinely null on both denominators.
- [x] ~~**ESAD: why do phase-1 (newly-covered) frames score worse?**~~ **VOID** — they don't.
      The apparent drop was the denominator artifact above; no position-dependent-quality
      explanation is needed. Do not spend a per-token-position breakdown on this.
- [x] **ESAD: recover the missing test-frame coverage (the real SOTA lever)** — DONE 2026-08-05
      (job 7353227, §4b-coverage). `--pad-short-runs` + `--tail-anchor` took GT coverage
      33.0% → 97.1% and honest AP **0.0531 → 0.1403** (+0.0872, 2.6×); gap to the paper
      3.63× → 1.37×. Root cause was runs shorter than one window (30.2% of GT), NOT tubelet
      parity. The ~0.17 linear extrapolation was optimistic: recovered frames yield 89% of the
      old per-GT rate (edge-replication is mildly OOD). v1/e159 only, single seed.
- [x] **ESAD: fix `total_test_frames=6223` → 6088** — DONE: default corrected in
      `score_esad_detection_ap.py`. Older JSONs still carry the optimistic fraction.
- [ ] **ESAD: re-score meta1b/meta2b/ours1b_e19 on the full denominator** (~15 min, caches are
      the only cost) so §4b-corrected's cross-checkpoint CPT-vs-raw table sits on the honest
      denominator. Deferred by the user 2026-08-05 in favour of closing the remaining gap on
      the best checkpoint first.
- [ ] **ESAD: last 5.2% of GT** (588 boxes still unreachable — run position 0 and sub-window
      tails). Low priority: worth ≤0.008 AP even if perfectly detected.
- [x] **ESAD: rare-class handling** — DONE 2026-08-06 (§4b-rare), **NULL result.** Both
      suspected defects CONFIRMED by measurement (cap=50 binds on 7/8 rare classes and 0/13
      well-supported; selection metric excludes the rare classes) but a 10-arm cap×selection
      sweep found nothing distinguishable from noise. The sweep's real yield was measuring the
      noise floor: **same-config unseeded replicates spread 0.0176–0.0316 AP**, wider than any
      between-arm gap.
- [x] **ESAD: SET A SEED** — DONE 2026-08-06. `--seed` (CLI) / `optimization.seed` (YAML) in
      `train_esad_double_head.py`, seeding torch/random/numpy/CUDA; determinism verified.
      Unseeded remains the default only so historical runs stay explicable — always pass it.
- [x] **ESAD: seeded multi-config campaign** — DONE 2026-08-06 (§4b-seeded, job 7355901),
      **NULL**. base/cap/data/both × 3 seeds; no config clears its own seed spread. Key
      finding: seeding *widened* the observed spread (0.022–0.063) vs the unseeded replicates
      (0.018–0.032) — reproducibility ≠ stability.
- [ ] **ESAD: is the `data` (+RARP1) arm real?** Best mean (0.1659, +0.0189 vs base) with a
      plausible mechanism (n=2→3 surgeries) but draws of 0.1251/0.1850/0.1877. Needs ~8–10
      seeds to resolve against a 0.063 spread — ~1 job, and the cheapest decisive test left
      before committing to unfreezing.
- [x] **ESAD: encoder unfreezing** — DONE 2026-08-07 (§4b-unfreeze), **POSITIVE and the only
      lever in this investigation to clear the noise floor.** 3 seeds, last-4 unfreeze:
      **AP 0.1403 → 0.2005 (+0.0602, spread 0.0191, 3.2× the spread)**, beating the paper's
      0.1928 on the identical full-denominator protocol. Gain holds at all 3 IoU thresholds.
      Single checkpoint (v1/e159), n=3.
- [ ] **ESAD: run the other 3 checkpoints through FT** (meta1b/meta2b/ours1b_e19). Now that
      the readout is no longer the bottleneck, the CPT-vs-raw comparison may finally be
      measurable — every frozen comparison was inside noise.
- [ ] **ESAD: shorter FT schedule.** Best epochs were 10/8/8 of 20 with flat-to-declining
      tails; ~12 epochs would cut cost ~40% at likely no loss.
- [ ] **ESAD: any future head experiment needs ≥8 seeds or an effect >0.06** — n=3 cannot
      resolve the effect sizes this probe produces. Budget accordingly or don't run it.
- [ ] **ESAD: train on RARP1 too (n=2 → 3 surgeries, +7120 GT boxes, +25%).** STATUS.md names
      n=2 as the top limitation; test is RARP3 so folding val into train is legitimate — hold
      out part of RARP4 for selection. Gains are uneven: cls 20 +269%, cls 9 +62%, cls 10 +45%,
      but ~nil for cls 15 (12 val boxes) and cls 12 (5).
- [ ] **ESAD: select `best.pt` on detection AP, not oracle `well_supported_map`.** Free change;
      the current selector optimizes a different, easier metric than the one reported.
- [x] **ESAD §4c: multi-box-per-class (M=2) — DONE 2026-08-05, NEGATIVE, root cause confirmed.**
      Detection AP roughly halved (0.1796→0.0913) because the presence head has no per-slot
      confidence (100% of 38136 test pairs had bit-identical slot0/slot1 scores) — every
      detection spawns a same-confidence phantom duplicate, penalized as a false positive at
      every threshold. Abandoned rather than fixed (would need a real per-slot-confidence
      architecture change for unproven upside on a 2-4%-of-frames phenomenon).
- [x] **ESAD §4d: pixel-space augmentation (crop+jitter) — DONE 2026-08-05, NULL on 1
      checkpoint (v1/e159).** AP_mean 0.1796→0.1686 (−0.011), inside the ~3 mAP noise floor.
      User decision: stop after 1 checkpoint, don't confirm on the other 3 — cost (2 walltime
      cycles/checkpoint from the doubled training set) not justified by a result already this
      close to null. ~~**All 4 non-unfreezing SOTA-closing levers (WBF, coverage, multi-box,
      augmentation) are now exhausted for this session — remaining gap to 0.1928 likely needs
      encoder unfreezing (separate scoped experiment) or a bigger head redesign.**~~
      **↑ REVISED 2026-08-05: coverage was NOT exhausted — it was mis-scored and is the biggest
      open lever (+89%). §4c/§4d deltas themselves stand (same-coverage A/B, so the denominator
      cancels), but their absolute AP columns are covered-subset values, not full-test AP.**

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
      resharded shards (~~keys flattened to `r{shard}_{idx}`~~ — **FALSE, corrected 2026-08-17: keys are
      `lemon__<youtubeId>_clip_NNNN.mp4` with a `.json` sidecar carrying `source_path`/`robotic`/
      `procedure`; `reshard_lemon_parallel.py:72` groups by source video, it does not rename. All
      4,194 raw .mp4s (925 GB) and `labels.json` also survive at `/flare/ModCon/ngetty/data/LEMON/`.
      A `lemon_robotic` subset IS recoverable from the existing shards** — the deferral below rested
      on a false premise) — a `lemon_robotic` subset would need a full
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

- [ ] **★★ ESAD FT campaign in flight (2026-08-24 night, 15 jobs on Sophia `by-gpu`).**
      All arms are prod37m_e199, ESAD double-head, full-denominator detection AP
      (`gt_full=11207`, `frames=5903`), 3 seeds each unless noted. The reference points
      are §4d's augmented last-4 FT and the §4b-seeded noise floor (spreads 0.022–0.063,
      so **n=3 cannot resolve a delta below ~0.06** — any arm landing inside that band is
      a null, not a small win).

      | jobs | arm (`RUN_TAG`) | what it changes vs the §4d recipe | confirms if | refutes if |
      |---|---|---|---|---|
      | ~~176548~~ **176588** | `frozen_aug` (3 seeds, s0 resumes at 17/20) | augmentation on the **frozen** probe, no unfreezing | frozen AP rises like FT's did → the win is data-side, not adaptation-side | frozen is flat → augmentation only pays once the encoder can move |
      | 176587 | `frozen_augonly` | augmented cache **alone** (2,468/ep), the missing control | matches frozen_aug → the frozen "+aug" gain is augmentation | matches frozen → the gain was the extra 2,468 windows, not the augmentation |
      | ~~176549~~ **176603**, 176550–51 | `ft_aug_last8` | unfreeze **8** blocks instead of 4, augmented | ≥ +0.06 over last-4 → depth and augmentation compose | inside noise → last-4 is the depth plateau (matches `esad-unfreeze-beats-frozen`) |
      | 176562–64 | `ft_augpe_last4` | **per-epoch** aug resampling (RNG gains an `aug_epoch` term) | ≥ +0.06 → the fixed-per-window crop was the binding limit; 20 distinct views beat 1 | inside noise → one crop per window already saturates a 20-epoch budget |
      | 176569–71 | `ft_aug_pw200` | `pos_weight_cap` 50 → 200, augmented | ≥ +0.06 macro AP → clipping (12/21 classes, class 12 off by 21.9×) was suppressing the macro mean | inside noise → the macro/GT-weighted gap (0.2265 vs 0.3484) is capacity, not loss weighting |
      | 176575–77 | `ft_augstrong` | per-epoch aug at **strong** strength (`MIN_SCALE 0.9→0.70`, `ASPECT 0.05→0.15`, `COLOR 0.15→0.30`) | ≥ +0.06 → the published recipe was under-regularised, consistent with `train_bce` still collapsing to 0.0075–0.0125 | inside noise or negative → mild is already the right strength; stop tuning aug |

      **Scoring.** `frozen_aug.sh` originally trained three seeds and stopped — it would have
      landed three run dirs with `best.pt` and no AP, an arm that reads as finished in `qstat`
      but has no number (the exact gap `run_esad_score_presrep.sh` was written to clean up after
      last time). Every arm above now scores inline: detection AP on the full denominator, then
      `score_oracle.py`, then a `collect_esad_arms.py` table in the job log.

      **Why 176548 became 176588.** PBS spools a job's script at submit time, so the inline-score
      patch could not reach the already-queued 176548; a `depend=afterany` scorer (176583) was
      chained instead, to avoid a `qdel`/resubmit that would lose queue position. That became moot
      once the real reason it wasn't starting turned up. The scheduler was explicit —
      `comment: Not Running: Insufficient amount of resource: queue_tags` against
      `select=1:ngpus=4:…:ngpu_quads=1`. Every node was `job-exclusive` except `sophia-gpu-11`,
      which was `free` with 6 quads but tagged `queue_tags = infer-svc-test` (reserved for another
      service), so **no quad request could place at all** while 1-GPU jobs kept packing onto shared
      nodes. And the job did not need a quad: the `WORLD_SIZE=4` loop is the augmented-cache
      *export*, complete on all four rank manifests, while training runs
      `CUDA_VISIBLE_DEVICES=0`. Resubmitted at `ngpus=1` as **176588** — losing queue position was
      the right trade only because the old position was behind an unobtainable resource. 176583
      was then held on a deleted predecessor (it would never release) and was cancelled; 176588
      scores inline. s0 already has 17/20 epochs and a `latest.pt`, and the trainer resumes by
      default, so it costs 3 epochs, not 20.

      **Frozen-arm cost, and a measurement trap.** I first read the frozen head-training as
      ~1 s/epoch and concluded the arms were startup-dominated and comfortably inside walltime.
      That was wrong: **the CSV has no timing column.** `train_esad_double_head.py` writes
      `epoch,train_loss,train_bce,train_box,lr,val_macro_map,val_well_map,val_iou,val_map50,best_metric,best_epoch`
      — the last field is `best_epoch`, a small integer that rises early in a run and so reads
      exactly like a seconds column under `awk -F, '{print $NF}'` ("0s,1s,2s,3s,4s"; the presrep
      "175s total" was final `best_epoch=15`). The real per-epoch duration is printed only to
      stdout as `[TRAIN] ep{N} … ({secs}s)`, which every ESAD launcher discards through
      `| tail -3`.

      Measured from file mtimes instead (`stat -c %y log_r0.csv` against `JOB START`), the truth
      is **~5.3 min/epoch** — 6 epochs in 32 min. That inverts the conclusion.

      **Measured directly, 2026-08-25** — and the mtime estimate was itself a blend of two very
      different arms. `tail -3` keeps the last three lines of the training stream, which are
      exactly `[TRAIN] ep{last}`, `[TRAIN] TEST best@`, `[TRAIN] TEST(by_iou)` — so **one**
      authoritative `(secs)` reading per seed does reach the job log after all:

      | job | arm | measured s/epoch | why |
      |---|---|---|---|
      | 176587 | `frozen_augonly` | **213.6** | one cache (aug only) |
      | 176588 | `frozen_aug` | **611.9** | `--train-cache` is `action="append"` — clean + aug **doubles** windows/epoch; 2× accounts for most of the 2.9×, co-tenancy on gpu-01 the rest |

      Per seed that is ~71 min of training (augonly) and ~204 min (frozen+aug). The score pass is
      **~24 min/seed**, measured end-to-end from four completed single-seed frozen score jobs
      (Polaris 7531253/7531316/7531378 = 25/24/23 min, 7484952 = 25 min) — not the ~39 min I had
      assumed. Against the two walls:

      | job | wall | reaches | |
      |---|---|---|---|
      | 176587 `frozen_augonly` | 3 h | s0 done 00:45, s1 ~01:56, s2 killed mid-run at 02:17 | **s2 overruns** |
      | 176588 `frozen_aug` (s0 resumed at 17) | 4 h | s0 done 00:08, s1 lands 02:55–03:32 | **s2 never starts** |

      The continuations were sized for the wrong cost model. 176598 (augonly, 4 h) is fine: s2
      training 1:11 + 3 × 24 min scoring = **2:23**. 176599 was not — it had to do s1's tail, all of
      s2 (3:24) and the score pass (1:12) inside 4 h, i.e. **4:36**. Requeued as **176609** at 7 h
      (`by-gpu` allows 24 h, so the 4 h in the script was self-imposed). `qalter -l walltime` is
      not usable here — it dies with `Exception in account_check hook encountered` — so this was a
      `qdel` + resubmit with the walltime overridden on the `qsub` command line, which takes
      precedence over the script's `#PBS -l walltime`. Same `depend=afterany:176588`, so no queue
      position was at stake (the job was `H` regardless).

      **The queue is genuinely full, and PBS says so** (2026-08-25 01:10). All twelve FT arms carry
      `comment: Not Running: Insufficient amount of resource: queue_tags`, and the node table
      explains it exactly: every `queue_tags = prod` node is at `ngpu_pairs=0, ngpu_quads=0`, while
      the only two nodes with capacity (gpu-10, gpu-11: 4 pairs / 6 quads each) are tagged
      `infer-svc-test`. `estimated.start_time` on the head of the queue is **Thu Aug 27** — two days
      out. There is no submission that fixes this: a narrower request doesn't help because *pairs*
      are exhausted too, and adding jobs would only consume `max_queued` (12/20 used, 4/5 running).
      Waiting is the correct action.

      **Launcher fix shipped while waiting — `run_ft_seed.sh` no longer polls for GPUs.** The old
      code picked devices by scanning `nvidia-smi` for <15 GB used, waited up to 120 min, then
      `continue`d — **rc=0 having trained nothing**, an arm that reads as finished in `qstat` and has
      no data. Job 176549 died in that loop. Measuring instead of assuming: from inside a job on
      gpu-07, `nvidia-smi` reports exactly 4 devices running exactly that job's 4 PIDs, while
      `pbsnodes` shows two *other users* co-tenant on the same node — **PBS cgroups isolate the
      device view**, so the poll can never see a co-tenant and the "my own jobs starve my quads"
      story in an earlier draft of this ledger was wrong. It also explains a mismatch that looks
      alarming and isn't: 176550 was assigned physical GPUs `2,3,6,7` but correctly ran on
      `CUDA_VISIBLE_DEVICES=0,1,2,3`, because the indices are cgroup-relative.

      The launcher now takes the allocation as given (`CUDA_VISIBLE_DEVICES=0..NGPU-1`), cross-checks
      it against `pbsnodes | grep assigned_gpus` for this `$PBS_JOBID`, and **exits 3** if the cgroup
      exposes fewer devices than requested. The `assigned_gpus` parse was validated against three
      live jobs on gpu-07 (4, 2, 2 — all correct) and an absent job, which reports `unknown` rather
      than a plausible-looking `1`. The same silent `continue` in the legacy `run_ft_3seed.sh` was
      swept to a loud `exit 3`. Backups: `*.bak-gpuwait-2026-08-25`. None of this disturbs the
      twelve queued arms — PBS spooled their script at submit time — and `CODE_STAMP` hashes
      `train_esad_unfreeze.py`, not the launcher, so no queued run will treat its checkpoint as stale.

      Early stopping (`patience=6`) fires too late to save either — the presrep seeds ran 17/19/20
      epochs. So **176598** and **176609** are chained `afterany` on 176587/176588 to resume after
      a walltime kill. Safe to resume: `latest.pt` persists `epochs_no_improve` and `best_metric`,
      and `best.pt` is rewritten only on strict improvement, so a resumed early-stopped seed
      re-enters the loop and exits immediately without damaging its checkpoint. (The skip gate
      keys on `>= 20` epochs while early stopping ends seeds at 17–19; those re-enter and exit
      rather than being skipped — costs one validation pass, harms nothing.)

      Take-away for future arms: get per-epoch cost from **mtimes**, never from a column assumed
      to be time; `head -1` on the CSV costs nothing and would have caught this immediately. See
      `esad-probe-csv-has-no-timing-column`.

      The twelve FT arms **do** need their quads: `torchrun --nproc_per_node=4` holding global
      batch 8, measured at ~750 s/epoch (last-4: ~1070 s) × 20 epochs ≈ 4.2 h. Folded to 1 GPU
      with `accum=4` that is ~17 h against a 5 h walltime, so they wait on genuine contention —
      nothing to fix. See `sophia-quad-requests-cannot-place`.

      Two motivating decompositions, both computed from artifacts already on disk (no GPU):
      **(a)** the §4d augmentation win is **3.0× larger on `pos_weight`-clipped (rare) classes**
      than on unclipped ones — which is why the `pw200` and strength arms target the same
      rare-class axis; **(b)** "train longer" is **ruled out** — all six §4d/last-4 arms peak at
      epoch 5–10 of 20 with declining tails, so the 20-epoch budget is not the limit.

      Also in flight: **176554** (`ft_aug` seed-ensemble) and **176559** (`ft_last4`
      seed-ensemble), both past `merged 123963 buckets across 3 seeds` and inside the
      O(n²) IoU-clustering step. Cross-seed fusion happens in **label space** (per-frame,
      per-class score + box), so it is valid across independently fine-tuned encoders —
      the earlier "frozen-only" restriction was cache plumbing, not a validity limit.

      **Not run, and why:** a per-slot confidence head (the original §4c follow-up) was
      **cancelled without spending GPU-hours** — a label-only ceiling computation showed a
      *perfect* per-slot head could add at most **+1.07 macro recall points (≤~0.0025 AP)**,
      10–25× under the noise floor. See the §4c closure block. Horizontal flip is likewise
      excluded from the strong-aug recipe: box semantics under flip were never verified for
      this class set (left/right-handed instrument actions), and an unverified flip would
      silently corrupt the box targets.


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
