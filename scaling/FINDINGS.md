# V-JEPA 2 Neural Scaling Law — Findings (through 2026-07-15)

> **PAPER DELIVERABLE (2026-07-17):** the general-arm result is being folded into
> Leonardo's surgical-arm paper (`/eagle/tpc/leonardo_borgioli/pp_sc/scaling.tex`,
> pulled via `ssh polaris`). Our shareable, drop-in LaTeX fragment (NOT edited into
> his .tex) lives in `scaling/paper_general_arm/`: `general_arm_section.tex` (new
> §, filled `tab:domaincmp`), `abstract_intro_patch.tex`, `general_arm_MERGE_NOTES.md`.
> Framing = **bounding result + suggested slow-growth**: α consistent with 0
> (mutual-kNN +0.11 [−0.08,+0.33], 6 metrics + 2 references agree); pairs with his
> sharp LR-k≈−1 into a two-arm thesis. 1e20 giant/gigantic were compute-blocked
> (256n never scheduled; killed 8678839); large@1e20 anchor is on disk. See the
> merge notes for provenance of every number.

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

## UPDATE 2026-07-16 — multi-metric selection + budget widening (autonomous overnight)

Tested whether a cheaper/sharper metric or a wider compute lever resolves the vertex-ordering failure,
entirely on the EXISTING checkpoints (features cached once per cell → pure-numpy metric sweep) plus a
half-decade downward budget extension. Pre-registered success test: ≥3 budgets with bootstrap vertex
CI half-width <0.08 log10(N) AND monotone compute-ordered vertices across the bracketable budgets.

**What ran:** 6 metrics × 24 cells. Metrics: metric_b_ridge (baseline 1−R²), cka_linear, cka_rbf,
mutual_knn (Platonic), procrustes, rankme (Garrido 2023, reference-free effective rank). Cells: the 20
original + **4 new 1e17 cells** (tiny/small/base/large). NOTE: **3e16 is infeasible at gb=96** — the
per-cell step floor leaves only vit_tiny above min-steps, so 1e17 (ipe=100) is the lowest bracketable
downward extension. giant/gigantic can't bracket at ≤1e17.

**Verdict: NONE of the 6 metrics passed.** But the *robustness checks explain why*, and the reason is
not simply "too little compute":

1. **T\* saturation ceiling (the dominant confound).** T\* = Meta ViT-g = **1B params**; our ladder
   tops at gigantic = **1.9B**. In ALL 3 budgets × ALL 5 T\*-referenced metrics, gigantic scores WORSE
   than giant — a 1.9B encoder structurally cannot align-to a 1B reference better than a 1B encoder can.
   This corrupts the right arm of every high-budget parabola. Re-fitting with gigantic (and giant)
   excluded does NOT restore ordering: N_opt stays pinned at ~100–200M with no compute trend. So the
   ceiling is real but removing it doesn't rescue the law — the whole T\*-alignment axis is size-capped.
2. **RankMe is deep but monotone, not bowl-shaped.** rankme has by far the deepest curves (BDNR 15–98,
   vs ≤13 for the T\* metrics) — but they are CONCAVE (a<0): effective rank rises with N at every budget
   with no undertraining penalty, so it measures capacity but defines no vertex. A reference-free metric
   removes the ceiling but loses the compute-vs-capacity tradeoff that makes a vertex.
3. **Best-resolved single-decade slope is positive but tiny.** 1e17→1e18 (the two best-converged
   budgets, full decade): cka_linear α̂=+0.21, mutual_knn α̂=+0.58, but ridge/rbf/procrustes ≈0. The
   signal exists and points the right way; it is just below the metric's noise over our C-range.

**Honest bottom line (unchanged headline, sharper mechanism):** with a fixed pretrained reference T\*,
the IsoFLOP vertex is not resolvable on this ladder because (a) the alignment metric saturates at the
reference's parameter count, capping the usable N-range at ~T\* size, and (b) reference-free rank has no
vertex. This is a METRIC-STRUCTURE limit, not merely a point-count limit.

A ceiling-free + vertex-forming y-axis would need to be BOTH label-based (no reference to saturate) AND
sensitive at these budgets. Two paths were checked and BOTH are dead as cheap tests:
- **Linear-probe on the ruler: impossible.** The K400 ruler `.cls` members are all class 0 (placeholder
  labels; the ruler is effectively label-less, confirming the earlier finding). No class diversity → no
  probe. A class-diverse labeled eval set would require new GPU feature extraction, not a numpy re-score.
- **Downstream probe (SSv2 / Metric A): already abandoned for the SAME reason.** At IsoFLOP budgets the
  encoders are too under-pretrained for a downstream probe to discriminate (converged large_C1e19 SSv2
  = 11% vs Meta ViT-L ~69.5%). A supervised probe hits the identical under-pretraining wall — CONVERGENT
  evidence that the constraint is genuinely compute, not the metric family.

**Net:** the pure-numpy metric space is exhausted (6 metrics, ceiling-exclusion refits, 1e17 lever). The
two remaining levers both cost real compute and are morning decisions: (1) a **larger T\*** (≥2B) to lift
the alignment ceiling above our ladder, or (2) **more compute per cell** (higher budgets / longer
training) so encoders are pretrained enough for a downstream probe to discriminate AND the vertex drift
exceeds noise. Both point the same way: the study is compute-bound, and the fixed-T\* metric family
cannot substitute for it.

### UPDATE 2026-07-16 (later) — the T\*=2B test REFUTES the ceiling as the binding cause
Found `vjepa2_1_vitG_384.pt` = Meta ViT-**G** 2B (embed_dim 1664) on disk — a reference ABOVE our whole
ladder — and re-scored all 24 cached cells against it (X is T\*-independent, so one GPU extraction of the
new Y + pure-numpy re-score). Result (`scaling/metrics_c256_tstarG_*`):
- **Vertices barely moved** vs T\*=g(1B): cka_linear 8.07→8.01, mutual_knn essentially identical
  ([8.24,8.11,8.76,8.61]). Same zigzag, same non-monotonicity, 0/6 pass.
- **gigantic STILL scores worse than giant** in every budget×metric even under the 2B reference. So that
  gap is NOT a "1.9B exceeds a 1B reference" saturation artifact — it is a genuine representation
  difference (the bigger model is more undertrained at fixed IsoFLOP). The ceiling hypothesis, though
  mechanically real (a same-size target IS harder to beat), is **not the binding constraint**.
- **Therefore reference size is irrelevant to the vertex**, and the flat ~100–200M N_opt across the whole
  2-decade budget range is INTRINSIC, not a metric/reference confound.

**Final, robust conclusion.** Across 6 metrics × 2 references × 24 cells (incl. the 1e17 downward
extension) and every re-fit variant, the fixed-reference linear-alignment IsoFLOP vertex does not order
with compute. This is now demonstrated to be **compute-leverage-bound**: over our 1e17–1e19 range the
true N_opt drift is smaller than the per-budget vertex noise (best single-decade slope α̂≈0.2–0.6 for the
dimension-invariant metrics, but swamped by σ(logN)≈0.1). The metric family is not the culprit — we ruled
out dimension, kernel choice, reference-free rank, AND reference size. The only remaining levers are real
compute: a **wider budget range (≥2.5 decades)** so drift exceeds noise, and/or **longer per-cell
training** to reach a regime where a downstream probe (the one ceiling-free + vertex-forming y-axis)
becomes discriminative. 3e16 is infeasible at gb=96 (step floor); the practical widening is UP, not down.

### Budget estimate (answering "what budgets are necessary")
With measured per-budget vertex noise σ(logN)≈0.10 and a 5:1 SNR target to fit a defensible power law,
the required compute lever is logC_range ≥ 5σ/α:

| true α | decades needed | top budget from 1e17 | within our 2.0-decade lever? |
|---|---|---|---|
| 0.50 | 1.00 | 1e18 | YES |
| 0.40 | 1.25 | 1.8e18 | YES |
| 0.30 | 1.67 | 4.6e18 | YES |
| 0.20 | 2.50 | 3.2e19 | no (need +0.5 dec) |
| 0.15 | 3.33 | 2.2e20 | no (need +1.3 dec) |

**Crucial inference:** our existing 1e17→1e19 lever (2.0 decades) SHOULD resolve any α≥0.25. It does not.
Therefore either (a) V-JEPA's true α<0.25 — i.e. compute-optimal model size grows UNUSUALLY SLOWLY with
compute vs LLMs (α_LLM≈0.5); a genuine finding that V-JEPA prefers spending compute on data/steps over
parameters — or (b) low-budget cells are too undertrained/noisy (σ>0.10). A joint Chinchilla-Approach-3
fit (α estimated from all points at once, bootstrap CI) discriminates these — see `scaling/joint_fit`
output. **Decisive experiment:** add 1e20+3e20 (extend to 3.5 decades); at that lever even α=0.15 clears
5:1. Cost ~10× the 1e19 gigantic cell — a multi-day capacity run, not one night.

### Joint Chinchilla-Approach-3 fit — the definitive statistical closure
Rather than chaining noisy per-budget vertices, fit ONE global α from all cells at once: model each
cell err = c0 + curv·(logN − [logN0 + α·(logC − logC̄)])². For fixed α this is closed-form OLS in
[1,u,u²] with u=logN−α·(logC−logC̄); profile α on a grid, bootstrap over cells for a CI (`joint_fast.py`).

| metric | α̂ (joint) | bootstrap 5–95% CI | excludes 0? |
|---|---|---|---|
| metric_b_ridge | −0.30 | [−0.30, +1.00] | no |
| cka_linear | +0.22 | [−0.30, +0.47] | no |
| cka_rbf | +0.10 | [−0.30, +0.41] | no |
| **mutual_knn** | **+0.11** | **[−0.08, +0.33]** | no |
| procrustes | +0.44 | [−0.30, +0.92] | no |

**Every metric's α is statistically consistent with 0.** The dimension-invariant metrics agree on a
small-POSITIVE point estimate (+0.10 to +0.22) — the right sign for a real but weak dependence of
N_opt on compute — but even the maximally-powerful joint fit cannot separate α from zero over our
1e17–1e19 lever. mutual_knn is best-conditioned (tightest CI, centered +0.11).

**FINAL VERDICT.** A compute-optimal scaling law for V-JEPA-2 on this ladder is **not resolvable with the
current compute budget**, and this is now proven exhaustively — not asserted: 6 metrics, 2 references
(1B + 2B), ceiling-exclusion refits, a downward budget extension, per-budget parabolas AND a joint fit
all agree α is small and noise-dominated. The consistent small-positive α̂ suggests the law is REAL but
that V-JEPA's compute-optimal model size grows slowly with compute (α≈0.1–0.2, well below LLMs' ≈0.5) —
which would mean V-JEPA should spend marginal compute on data/steps rather than parameters. Confirming
this (α̂ CI excluding 0) requires extending the lever to ≥3 decades (add 1e20/3e20), the only remaining
experiment with the statistical power to settle it.

## Artifacts (committed)
- `scaling/eval_metric_b.py` — metric + std/intercept/CV fix (ce09617)
- `scaling/experiments_pe_FIXED.csv`, `scaling/fit_pe_FIXED.json`, `scaling/isoflop_pe_FIXED.png`
- Per-cell sidecars: `metric_B.json` (orig), `metric_B_FIXED.json` (120+fix),
  `metric_B_480FIX.json` (480+fix) under `experiments/scaling_pe/<cell>/`
- K400 (superseded): `scaling/experiments*.csv`, `scaling/isoflop_canonical.png`
