# V-JEPA 2 Neural Scaling Law — Findings (through 2026-07-15)

## Goal
Establish a compute-optimal (IsoFLOP / Chinchilla-style) scaling law for V-JEPA 2 video SSL:
per-budget parabola of a quality metric vs model size N → vertex N_opt(C); across budgets fit
N_opt ~ C^α, D_opt ~ C^β. Fixed global batch (gb=96) decouples D from N. Never fit raw pretraining
loss (moving EMA target makes it ill-posed across N).

## Metric
**Metric B** = frozen common-space linear predictivity error (1−R²) of a run-encoder against a fixed
reference T* (Meta ViT-g/16 @384), measured on a fixed held-out clip "ruler." Chosen over a downstream
probe (Metric A / SSv2) because at IsoFLOP budgets our encoders are too under-pretrained for SSv2 to
discriminate (converged large_C1e19 = 11% vs Meta ViT-L ~69.5%); Metric A abandoned as an axis (see
[[metric-a-ssv2-status]]).

## Data / corpus (the first big correction)
- Original corpus K400 (241K clips) was **too small**: to give small models a high budget at fixed gb,
  the corpus is replayed 10–90×; past ~4 epochs repeated data saturates → left-arm (vertex-defining)
  points collapse → no clean vertices above ~3e18. Also K400 (241K) < surgical (533K), inverting the
  intended "general ≥ domain" narrative.
- **Pivoted to PE-Video** (facebook/PE-Video, 984K clips, disjoint from the K400 Metric-B ruler so the
  ruler stays uncontaminated). 4×-fresh rule then holds through 1e19. This is the corpus for all results
  below. K400 sweep = superseded reference.

## Sweep executed (PE-Video, 20/20 cells)
Budgets 3e17, 1e18, 3e18, 1e19; sizes tiny(49M)→gigantic(1.9B); gb=96 fixed; per-size DDP/HSDP topology;
self-resubmitting chains across debug-scaling + capacity queues. All 20 cells trained to target.

## Metric-B methodology fixes (the second correction)
The first fits were confounded. Two issues found and fixed (commit ce09617):
1. **Dimension confound.** Ridge fit used un-standardized X with fixed λ=100 → bigger encoders
   (d_run 192→1664) got mechanically worse scores INDEPENDENT of quality (metric–vs–log-N correlation
   +0.82). FIX: standardize X (train-only stats) + intercept → λ comparable across encoder sizes.
   Cut the spurious size-correlation roughly in half (+0.82 → +0.43).
2. **Fixed λ across sizes.** Added per-cell λ cross-validation (`--lam cv`, now default).

IMPORTANT framing correction (user): the RISING right arm of each parabola (bigger N → worse metric at
fixed budget) is NOT contamination — it is the **valid IsoFLOP undertraining signal** (big models are
data-starved at fixed compute). The dimension fix removes the *mechanical* d-penalty while *preserving*
this real capacity-vs-data tradeoff. Goal was never to flatten the right arm.

## The core result — and its limitation

Per-budget parabola vertices under three metric variants (all PE-Video):

| variant | 3e17 | 1e18 | 3e18 | monotonic↑? |
|---|---|---|---|---|
| FIXED (std+cv, 120 clip) | 135M (R²0.97) | 236M (R²0.88) | 188M (R²0.85) | NO |
| 480FIX (std+cv, 480 clip) | 145M (R²0.98) | 208M (R²0.82) | 149M (R²0.89) | NO |

Fitted exponents (FIXED,120): **α=0.147, β=0.890, α+β=1.037** (α 95%CI [−0.21,0.46]).

**The vertices do NOT order monotonically with compute**, which a valid scaling law requires. Root cause,
quantified: at 1e18 and 3e18 the **fit noise (RMSE 0.012–0.030) ≥ the bowl depth (~0.007–0.019)** — the
parabolas are too shallow relative to metric noise, so N_opt is under-determined and its ordering is
unresolvable. Only **3e17 has a bowl genuinely deeper than its noise** (depth 0.023 vs RMSE 0.003) —
which is why it was always the one clean curve.

**Going 120→480 clips (4× tokens) did NOT fix it** (RMSE 0.014→0.012, 0.030→0.019; bowls stayed shallow).
So the resolution floor is not a ruler-sampling problem.

## Honest conclusion (this phase)
- We have **one clean parabola (3e17, N_opt~140M)** and a fitted α/β that is **not defensible** because
  the higher-budget vertices float in noise and don't order with compute.
- The binding problem is **methodological, not (yet) the hypothesis**: Metric B's per-budget bowls are
  intrinsically shallow (~0.01 deep, 0.78–0.97 range, 78% floor) — JEPA linear-predictivity-to-T* is only
  weakly sensitive to model size at fixed compute. Metric noise is comparable to that sensitivity.
- 1e19 is separately unbracketable (base is its leftmost point; tiny/small dropped by the 4× rule) — a
  structural consequence of flat-ish N_opt, same as K400's top budget.

## Next directions (both viable; metric is cheaper than compute)
1. **Try a different / more sensitive metric** (cheap — re-uses existing checkpoints):
   - CKA / centered-kernel similarity to T* (dimension-invariant by construction, may give deeper bowls).
   - A metric with more dynamic range than 1−R² (which floors at 0.78 here).
   - Goal: a per-budget curve whose bowl depth ≫ its noise at ALL budgets, not just 3e17.
2. **Widen / raise the compute range** (expensive): closer-spaced low budgets give short power-law levers;
   the true N_opt spread may only exceed noise over ≥2 decades. Would need larger budgets and (for
   bracketing the top) either a bigger corpus or accepting the top-budget right-arm-only.

## Artifacts (committed)
- `scaling/eval_metric_b.py` — metric + std/intercept/CV fix (ce09617)
- `scaling/experiments_pe_FIXED.csv`, `scaling/fit_pe_FIXED.json`, `scaling/isoflop_pe_FIXED.png`
- Per-cell sidecars: `metric_B.json` (orig), `metric_B_FIXED.json` (120+fix),
  `metric_B_480FIX.json` (480+fix) under `experiments/scaling_pe/<cell>/`
- K400 (superseded): `scaling/experiments*.csv`, `scaling/isoflop_canonical.png`
