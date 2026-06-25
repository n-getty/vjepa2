# Surgical CPT downstream findings — fs10 cached probe (2026-06-24)

## Setup
- **Goal:** does continued pretraining (CPT) of the Meta distilled ViT-L
  (`vjepa2_1_vitl_dist_vitG_384.pt`) on surgical video improve the SAR-RARP50
  asformer downstream probe over the off-the-shelf Meta init?
- **Probe:** fast cached fewshot-10% asformer probe (Leo's video-stratified 10%
  subset; backbone features exported once per checkpoint, head trained on cache).
  ~15 min/checkpoint, ~14x faster than full-encoder probe. All checkpoints probed
  on byte-identical correct-attention cached features (use_sdpa=false export).
- **Determinism:** probe seed hardcoded `_GLOBAL_SEED=0` → single-seed point
  estimates. Within-lineage deltas share head-init so reflect backbone
  differences; cross-lineage gaps carry ~±2-4 F1 seed noise (no error bars yet).

## Results (best val_macro_f1 @ peak, vs fs10 Meta anchor 65.06)

| Checkpoint | F1 | Δ Meta | Engine | Trajectory |
|---|---|---|---|---|
| Meta-raw | 65.06 | — | — | off-the-shelf |
| v1_e9 | 66.88 | **+1.8** | broken sdpa | phase1-warmup(12ep) → phase2(9ep) |
| v1_e29 | 64.50 | −0.6 | broken sdpa | phase1-warmup → phase2(29ep) |
| v2_e9 | 63.74 | −1.3 | fixed sdpa | hot-from-raw-Meta (9ep), DIRTY data |
| v2_e19 | 61.13 | −3.9 | fixed sdpa | hot-from-raw-Meta (19ep), DIRTY data |
| v3_e4 | 65.79 | +0.7 | fixed sdpa | clean data, e4 (within-version trend) |
| **v3_e9** | **66.27** | **+1.2** | fixed sdpa | hot-from-raw-Meta (9ep), **CLEAN data (black-clip filtered)** |
| v3_e19 | pending | — | fixed sdpa | clean data, e19 (run reaches e19 soon) |

(Full-data anchors, separate validation: full-data Meta = 78.2 ≈ Leo's 79.38,
confirming the harness reproduces published numbers.)

## DATA-FIX ABLATION (2026-06-25): v3_e9 vs v2_e9 — black clips were a real driver
**v3_e9 (66.27) vs v2_e9 (63.74) = +2.5 F1**, SAME epoch / recipe / engine — the
ONLY difference is the degenerate (pure-black) surgvu24 clips are filtered out in
v3 (min_clip_std=1.0). This flips the checkpoint from BELOW Meta (−1.3) to ABOVE
Meta (+1.2) — the first clean (unconfounded) beat of the Meta anchor under the
fixed engine. Strong evidence the surgvu24 ~19-23% byte-identical black-clip
corruption was a real driver of the downstream regression. CAVEAT: v1_e9 also
beat Meta then declined to e29; the open question is whether v3 HOLDS (v3_e4 →
v3_e9 trend + v3_e19) or peaks-early-then-declines like v1/v2. If v3 holds, the
data fix changed the trajectory; if it declines, black clips were one factor but
recipe (EMA/LR/horizon) still drives the late decline.

## WITHIN-VERSION TREND (2026-06-25): v3 RISES where v1/v2 DECLINED
v3_e4 = 65.79 (+0.7) -> v3_e9 = 66.27 (+1.2): trend is **+0.5, UP**. Contrast:
- v1: e9 66.88 -> e29 64.50 = -2.4 (DOWN)
- v2: e9 63.74 -> e19 61.13 = -2.6 (DOWN)
- v3: e4 65.79 -> e9 66.27 = +0.5 (UP), and BOTH points beat Meta (65.06).
This is the first lineage that IMPROVES with more pretraining. The defining
regression ("worse with more surgical pretraining") is ABSENT in v3 so far.
CAVEAT: unequal spans — v3 measured e4->e9 (5 ep) vs v1/v2 over longer horizons;
v3 could still turn down later. **v3_e19 is the decisive test** (training at
e11/20, e19 ckpt lands soon). If v3_e19 >= Meta, data fix robustly fixed the
trajectory; if it drops below Meta like v2, black clips delayed but recipe
(EMA/LR/horizon) still drives a late decline.

## Findings

1. **The downstream regression is REAL and persists under the fixed engine.**
   Both lineages decline with more pretraining (v1: −2.4 over e9→e29; v2: −2.6
   over e9→e19). This is NOT merely the SDPA/probe bug (those are fixed here):
   more JEPA pretraining on this surgical data progressively degrades
   probe-readable structure. Notably v2's decline happens with λ=0 (context loss
   off until e30), so it is NOT the λ/context-loss mechanism either.

2. **Surgical CPT CAN beat Meta — but only briefly/early.** v1_e9 = +1.8 is a
   real win. The "surgical data is too narrow to help" hypothesis is at least
   partially refuted: a CPT checkpoint that improves on the strong init exists.

3. **Trajectory matters more than the engine fix.** v1 sits ~2-3 F1 above v2 at
   comparable points; v1_e9 is the only >Meta point. Most likely cause: v1's
   GENTLE phase1-warmup→phase2 path vs v2's hot-from-raw-Meta recipe. (Confounded:
   v1/v2 differ in engine + init + epochs + LR schedule, so this is a
   recipe-as-whole comparison, not an engine ablation. The engine fix cannot make
   a broken run genuinely better — so the gap is the trajectory, not the bug.)

## Caveats
- Single seed per probe (seed=0). Cross-lineage gaps need error bars (cheap with
  caching; requires adding a seed knob). Within-lineage declines are robust.
- 10% subset → lower absolute numbers than full data; valid for TREND only.
- v1 vs v2 is confounded (4 axes differ simultaneously).

## Implication
The win condition is achievable (v1_e9 beat Meta), but every recipe tried so far
drifts downward with more epochs. Before more pretraining: re-verify no residual
bugs / data corruption / config issues (this audit), THEN the highest-value run is
replicating v1's gentle warmup→continue trajectory UNDER the fixed engine, probing
early epochs (e<9) for the peak, with seed error bars.

---

# Re-audit before more pretraining (2026-06-24)

## CRITICAL DATA BUG — surgvu24 is ~19% byte-identical pure-black duplicate clips
VERIFIED independently: 19.3% of surgvu24 mp4s are the exact same 185137-byte file,
which decodes to 3500 frames of pure black (pixel mean/std/max = 0.000/0.000/0).
- Upstream source corruption (present in pre-reshard source too), faithfully copied.
- surgvu24 is ~23% of the realized mix at T=0.5 -> **~4.4% of EVERY training batch is
  this one identical black clip**. No black/variance filter exists (filter_short_videos
  unset=False). For V-JEPA this is maximally harmful: LayerNorm'd target of a black
  frame is degenerate, masked-prediction loss trivially ~0, identical sample surfaced
  thousands of times -> collapse-pressure that compounds with epochs. **Strong candidate
  mechanism for "worse with more pretraining."**
- FIX: filter near-zero-variance clips at decode (VideoDecoder), or drop the duplicate
  185137-byte file / clean surgvu24 shards + metadata, before any more pretraining.
- Secondary data note: surgical sources highly static (surgvu24 framediff~3.6, jigsaw~2.2
  vs kinetics~23); kinetics CLEAN and genuinely helpful (anti-forgetting). metadata.json
  counts accurate. No other corruption found.

## CPT CONFIG AUDIT — recipe is upstream-from-scratch applied to continue-from-strong-init
Ranked likely contributors to "briefly beats then declines" (beyond the black-clip bug):
1. **Flat-high EMA 0.99925** (constant; equal endpoints => zero ramp). For CPT this is
   co-drift: teacher has no anchor to the strong Meta init, target degrades in lockstep
   with the over-specializing student. Fix: ramp [0.9995->0.9999] or effectively freeze
   teacher (~0.99995).
2. **No LR cooldown / missing re-warmup.** v1-p2 = constant 5.25e-4, warmup 0, no anneal
   over 600ep -> integrates noise into converged init forever (brief e9 sweet spot then
   drift). v2 = correct cold peak 7.5e-5 BUT hot onto raw Meta with only 2ep warmup.
   Fix: low peak + real warmup + cosine cooldown to ~1e-6; consider layer-wise LR decay.
3. **Resolution mismatch.** v2 trains 256px, probe is 384px (RoPE interpolated). v1-p2
   trained 384 = probe res (a second reason v1>v2). Fix: train surgical CPT at 384.
4. **Horizon wildly long** (600-1000ep from-scratch horizon); both peak ~e9. Early-stop
   ~e9-12; cut horizon to ~15-20ep. Symptom not cure.
5. **Keep kinetics ~37%** (do NOT cut) — it's anti-forgetting rehearsal; cutting would
   accelerate decline. Optionally raise temperature to confirm forgetting mechanism.
6. (minor) no temporal masking (max_temporal_keep=1.0) -> easy task on redundant surgical
   video; lambda iters (15k/30k) mistuned for short ipe but lambda=0 at all observed peaks.

## Conclusion
Do NOT launch more pretraining yet. Fix the black-clip data corruption FIRST (highest
leverage, unambiguous bug). Then the recipe direction is: gentle warmup->continue (v1-style)
UNDER fixed engine, at 384px, with anchored/ramped EMA + LR cooldown, short horizon,
keep kinetics, probe early epochs with seed error bars.
