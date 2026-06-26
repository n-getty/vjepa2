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

---

# Full-data CACHED probe (2026-06-26) — speed-optimized cross-check

## Why this exists
Benchmark proved the probe is ENCODER-FLOP-bound, not iteration-bound (batch
size barely helps; sdpa=true ~halves it; caching removes the encoder cost). We
extended the cache exporter to the FULL split (sdpa=true, bs4) to (a) cross-check
the fs10 ranking on full data and (b) do it fast.

## Speed (full-data scale)
- export (encoder once, bs4 sdpa=true): ~no-OOM, 345GB cache/ckpt, minutes.
- cached probe: 41 min / 19 epochs = ~2.2 min/epoch. NOT the ~0.5 min/ep of
  fs10 — at 345GB the probe is now CACHE-I/O-bound (reading features from Lustre
  each epoch), not encoder-bound. Still ~3x vs optimized non-cached (bs4+sdpa
  ~6.2 min/ep), ~8x vs original (bs2+sdpa-off ~16.5 min/ep).

## Anchor (re-anchored on the cached recipe)
| regime | metaraw | note |
|---|---|---|
| full-data augmented (non-cached) | 78.2 | headline path (Leo ~79.38) |
| full-data CACHED (no aug) | **71.69** | new anchor for THIS table; -6.5 = aug-drop |
| fs10 cached | 65.06 | small-subset anchor |

## Results (best val_macro_f1, vs full-data-cached metaraw anchor 71.69)
| ckpt | full-cached F1 | dMeta | (fs10 dMeta for cross-check) |
|---|---|---|---|
| metaraw | 71.69 | — | — |
| v3_e9 | 73.89 | **+2.2** | (+1.2) -> REPRODUCES, stronger |
| v1_e9 | 69.93 | **-1.8** | (+1.8) -> RANKING FLIPPED |
| v1p1_e12 | 70.95 | -0.7 | (new: v1 true epoch/res match to v3) |
| v2_e9 | 72.93 | +1.2 | (-1.3) -> SIGN FLIPPED (fs10 said <Meta) |
(Key question: does the full-data ranking REPRODUCE the fs10 ranking? If yes,
the fast fs10 trend tool is validated. If not, fs10 distorts comparisons.)

## RANKING FLIP (2026-06-26): fs10 over-rated v1_e9 — full data reverses it
| ckpt | fs10 dMeta | full-data dMeta | verdict |
|---|---|---|---|
| v3_e9 (clean) | +1.2 | **+2.2** | beats Meta BOTH ways (data fix real) |
| v1_e9 (dirty/broken) | +1.8 | **-1.8** | fs10 said >Meta; full data says <Meta |
On fs10, v1_e9 (66.88) > v3_e9 (66.27) -- the puzzling result. On FULL DATA it
REVERSES: v3_e9 73.89 >> v1_e9 69.93 (v3 +3.96 ahead). Two causes inflated v1 on
fs10: (1) 10% subset = high-variance head that can't exploit a stronger backbone,
compressing the ranking; (2) v1_e9 = ~22 surg epochs @384px vs v3_e9 = 10 @256px
(epoch+res confound favoring v1). With full probe-training signal, the CLEANER
backbone (v3) wins decisively despite fewer epochs + lower train res.
IMPLICATION: fs10 is reliable for "does X beat its anchor by a clear margin" but
NOT for fine cross-lineage ranking (it got v1-vs-v3 order + v1-vs-Meta sign
wrong). Full data is the arbiter for close calls. The "surgical CPT beats Meta"
claim now rests on v3 (clean data), NOT the confounded v1.

## FULL-DATA CROSS-CHECK COMPLETE (2026-06-26) — engine fix > data fix
Final full-data cached table (vs Meta 71.69):
| ckpt | full dMeta | fs10 dMeta | engine | data |
|---|---|---|---|---|
| v3_e9 | +2.2 | +1.2 | fixed | CLEAN |
| v2_e9 | +1.2 | -1.3 (FLIP) | fixed | dirty |
| v1p1_e12 | -0.7 | — | broken | dirty |
| v1_e9 | -1.8 | +1.8 (FLIP) | broken | dirty |

REVISED CONCLUSION (full data corrects the fs10 story):
1. ENGINE FIX is the dominant factor. Both FIXED-engine surgical runs (v2 dirty
   +1.2, v3 clean +2.2) beat Meta; both BROKEN-engine v1 runs (-0.7, -1.8) do
   not. The sdpa layout fix matters more than the data fix.
2. DATA FIX is real but SMALLER than fs10 implied: v3_e9 (clean) vs v2_e9 (dirty),
   matched epoch/recipe/engine = +0.96 on full data (was +2.5 on fs10).
3. fs10 was SYSTEMATICALLY MISLEADING for cross-lineage sign/order: it flipped
   BOTH v2 (said <Meta, really >Meta) and v1 (said >Meta, really <Meta). fs10 is
   only safe as a coarse "clearly beats its own anchor" screen, NOT for ranking
   or sign near the anchor. FULL-DATA is the arbiter. This retroactively softens
   earlier fs10-based claims (e.g. "v2 dirty data hurts below Meta" was an fs10
   artifact; v2 actually beats Meta once the engine is fixed).
4. STILL OPEN: within-version TREND (does v3 decline with epochs like v1/v2's
   fs10 trend suggested?) -- needs v3_e14/e19 full-data, running now. And whether
   v2's fs10 "decline e9->e19" also reverses on full data (not yet retested).

## v3 WITHIN-VERSION TRAJECTORY (full-data cached, 2026-06-26)
| v3 epoch | full F1 | vs Meta 71.69 |
|---|---|---|
| e9  | 73.89 | +2.2 |
| e14 | 73.06 | +1.4 |
| e19 | pending | (DECISIVE) |
e9->e14 = -0.83: a MILD decline, but stays well ABOVE Meta (+1.4 at e14). This is
NOT the v1/v2-style regression (those fell BELOW their anchor; fs10 slopes were
-2.4/-2.6). v3 reads as PLATEAU-above-Meta with slight drift, not collapse.
Decisive test remains v3_e19: hold >=Meta => fixes changed the trajectory; drop
below => mild late decline persists even in the best recipe. (Caveat: e9 may be a
local high; e14 within ~1 F1 could be seed/early-stop noise -- single seed.)

## fs10 v3_e14 = 64.57 (-0.5 vs fs10 Meta 65.06) -- 3rd fs10 near-anchor sign flip
fs10 v3 trajectory: e4 65.79(+0.7), e9 66.27(+1.2), e14 64.57(-0.5 BELOW Meta).
But FULL-DATA v3_e14 = 73.06 (+1.4 ABOVE Meta). fs10 says v3_e14 dropped below
Meta; full-data says clearly above. Same near-anchor unreliability seen for v1
and v2. CONFIRMED PATTERN: fs10 OK only as a coarse screen, full-data is the
arbiter for sign/order near the anchor. Trust the full-data trajectory
(e9 +2.2 -> e14 +1.4, both >Meta); v3_e19 (full-data) is the decisive endpoint.

# ============================================================
# DECISIVE RESULT (2026-06-26): v3 HOLDS above Meta through full run
# ============================================================
v3 full-data cached trajectory (vs Meta 71.69):
  e9  = 73.89 (+2.2)   <- best
  e14 = 73.06 (+1.4)
  e19 = 71.99 (+0.3)   <- DECISIVE endpoint, still >= Meta

VERDICT (two true halves):
1. The CATASTROPHIC REGRESSION IS GONE. The whole investigation started because
   surgical CPT fell BELOW Meta and worsened with epochs (v2 dirty fs10:
   -1.3 -> -3.9). With BOTH fixes (SDPA engine + black-clip data filter), v3
   stays AT/ABOVE Meta the entire run -- never regresses below baseline. The
   "surgical pretraining hurts on a surgical task" paradox is RESOLVED.
2. But a GENTLE DOWNWARD DRIFT remains: +2.2 -> +0.3 over e9->e19 (-1.9). v3
   peaks early (e9) and decays toward Meta by e19. Not collapse, but not
   improving-with-epochs either. Early checkpoint is best.

TAKEAWAYS:
- BEST surgical model = v3_e9 (+2.2 over Meta on full data). Early-stop ~e9.
- Root causes fixed: (a) XPU SDPA layout bug, (b) surgvu24 ~19% black-clip
  corruption, (c) dataloader oversampling (temperature mixing). Engine fix was
  the dominant factor; data fix additive (+~1 at matched epoch).
- Remaining gentle drift points at the CPT RECIPE (flat-high EMA 0.99925, no LR
  cooldown, over-long horizon) -- the next lever now that data+engine are clean.
  Hypothesis: anchored/ramped EMA + LR cooldown + short horizon could turn the
  e9 peak into a sustained gain.
- METHODOLOGY: fs10 cached probe is only a coarse screen (flipped sign vs full
  data near the anchor for v1, v2, v3_e14). FULL-DATA cached is the arbiter;
  ~30-48 min/ckpt with the optimized recipe (cached + sdpa=true + bs4).

## ROOT-CAUSE HYPOTHESIS (2026-06-26): we are UN-DISTILLING a ViT-G->ViT-L init
The Meta init `vjepa2_1_vitl_dist_vitG_384` is a ViT-L DISTILLED FROM ViT-G. Its
strength is ViT-G-quality features compressed into ViT-L weights -- it punches
above ViT-L's self-SSL weight class (off-the-shelf 71.69 full-data).
Our CPT uses target_encoder_key=ema_encoder = EMA of OUR ViT-L student (verified;
EMA teacher ~= student, gap 0.002 @e19). So CPT REPLACES the ViT-G teacher with a
self-teacher on a WEAKER objective (ViT-L masked-pred) over NARROWER data. Every
step relaxes the ViT-L weights AWAY from the ViT-G-distilled solution toward what
a ViT-L can self-supervise alone = lower capacity. => "un-distillation".

This UNIFIES the evidence:
- loss plateaus e5 but downstream falls: the ViT-L pretext saturates fast while
  the ViT-G-distilled structure keeps eroding.
- downstream degrades even as weight-motion -> 0 (e14->e19 step 0.009): drift is
  DIRECTIONAL (away from ViT-G basin), magnitude-independent.
- monotonic not collapse: smooth reversion ViT-G-quality -> native-ViT-L-SSL.
- EMA no anchor: teacher is the decaying student, not ViT-G.

KEY DISCRIMINATOR (diagnosis running): is the feature decline SURGICAL-SPECIFIC
or GLOBAL (kinetics too)? Global decline => un-distillation (general capability
loss) confirmed; surgical-only => domain overfitting instead.

IMPLIED FIX (different from EMA/LR/temporal-mask tweaks): PRESERVE the distillation
signal. Options: (1) keep original Meta ViT-L FROZEN as a distillation teacher
during CPT (distill-while-adapt) or regularize features/weights toward Meta;
(2) accept ViT-L can't self-improve from a distilled init -> only very-short
heavily-anchored adaptation (e9 peak is the ceiling); (3) anchored/frozen EMA
(teacher~=Meta) as a cheap partial version of (1).


## FEATURE DIAGNOSIS (2026-06-26, @256px) — drift is GLOBAL, not surgical-specific
Token/clip metrics across Meta->e4->e9->e14->e19 (target_encoder, 256px inference):
cos-to-Meta:  surg 1.00->0.961->0.888->0.858->0.838 ; gen(kin) 1.00->0.951->0.863->0.818->0.793
tok_eff_rank: surg 511->484->480->474->465 (-9%)    ; gen 433->412->400->392->386 (-11%)
anisotropy:   surg 0.531->...->0.602 (rising = tokens more similar)
READS:
1. Drift is GLOBAL: kinetics features drift from Meta AS MUCH/more than surgical
   (gen cos 0.793 < surg 0.838 @e19). NOT surgical-domain overfitting -> supports
   UN-DISTILLATION (losing general ViT-G-distilled structure everywhere).
2. Gradual EROSION not collapse: rank -9..11%, anisotropy mild rise. Encoder
   slowly relaxes OUT of the Meta(ViT-G-distilled) basin toward weaker native
   ViT-L self-SSL. cos-to-Meta falls monotonically 1.0->0.84, tracking the
   downstream decline.

## RESOLUTION MISMATCH (flagged 2026-06-26) — SECOND stacked driver
CPT trains at 256 but init is 384 AND probe is 384:
  Meta init 384 -> v3 CPT 256 -> probe 384.  (256 inherited from upstream/v2 configs.)
So every CPT epoch adapts the encoder to 256 statistics (patch content, RoPE scale)
while the probe reads 384 (RoPE interpolated) -> a resolution penalty that GROWS
each epoch, independent of un-distillation. The @256 diagnosis above still shows
global drift, so un-distillation is real on its own; resolution mismatch is an
ADDITIONAL penalty the 384-probe sees that the 256-diagnosis cannot. Likely TWO
stacked monotonic effects. Leo v1 (train 384/probe 384, matched) had no such
penalty -> partly why v1 looked better.
DECISIVE CHEAP TEST: re-probe v3_e9 & v3_e19 at 256px (= CPT res, no retrain). If
regression shrinks at 256 -> resolution mismatch is major (fix: train+probe same
res, ideally 384). If persists -> un-distillation dominates (fix: distill-anchor).

## RESOLUTION-SENSITIVITY RESULT (2026-06-26) — resolution is MINOR, not the driver
cos(ckpt, Meta) on same surgical clips at each res:
| ckpt | @256 | @384 | 384-256 |
| meta | 1.000 | 1.000 | 0 |
| e9   | 0.906 | 0.907 | +0.0005 |
| e19  | 0.864 | 0.859 | -0.005 |
e9->e19 cos-drop: 0.042 @256 vs 0.047 @384 (only ~12% larger at 384).
READ: drift from Meta is OVERWHELMINGLY resolution-INDEPENDENT (e19 cos~0.86 at
both res). Resolution mismatch MILDLY amplifies (~10%) but is NOT the dominant
driver. Points back to UN-DISTILLATION as primary. CAVEAT: cosine proxy, not
probe F1 -> definitive test remains the 256px re-probe (running/next). Predict:
regression mostly PERSISTS at 256 (un-distillation), maybe small recovery.

## 256px RE-PROBE — v3_e9 (2026-06-26, definitive F1 test, in progress)
v3_e9 @256px = 73.73 (vs @384px = 73.89) -> IDENTICAL within noise (Δ0.16).
Confirms the cosine test: resolution barely matters for v3_e9. The encoder
performs the SAME whether probed at its CPT-native 256 or the 384 the probe used.
=> resolution mismatch is NOT meaningfully penalizing v3 downstream.
STILL NEEDED for the regression-slope decision: v3_e19 @256 (+ Meta @256 anchor).
If v3_e19 @256 also ~matches v3_e19 @384 (71.99), the e9->e19 regression slope is
the SAME at native resolution -> resolution definitively ruled out, un-distillation
is THE driver. (next hold queued for v3_e19-256 + Meta-256)
