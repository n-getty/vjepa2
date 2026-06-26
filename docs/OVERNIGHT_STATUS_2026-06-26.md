# Overnight autonomous run — status & morning handoff (started 2026-06-26 ~04:00 UTC)

## ★ MORNING SUMMARY — the overnight question is ANSWERED ★
v3 training reached e20 (complete). Decisive full-data trajectory vs Meta 71.69:
  v3_e9 73.89 (+2.2)  ->  e14 73.06 (+1.4)  ->  e19 71.99 (+0.3)

**The regression is RESOLVED.** The investigation began because surgical CPT fell
BELOW Meta and worsened with epochs (v2 dirty: -1.3 -> -3.9). With both fixes
(SDPA engine + black-clip data filter), v3 stays AT/ABOVE Meta the entire run.
The "surgical pretraining hurts on a surgical task" paradox is gone.

**Two caveats kept honest:**
1. A gentle downward DRIFT remains (peaks at e9 +2.2, decays to +0.3 by e19).
   Best surgical model = v3_e9, not the final. Early-stop ~e9.
2. Engine fix (SDPA) was the DOMINANT factor; data fix (black clips) additive
   (~+1 at matched epoch). fs10 v2 ALSO beat Meta on full data once engine fixed.
   Remaining drift points at the CPT recipe (EMA/LR/horizon) = next lever.

Full results + methodology in docs/fs10_probe_findings.md (the DECISIVE section).
Best checkpoint: <ckdir>/e9.pth.tar. In-flight: fs10 v3_e19 (8566725, cross-check
only). All v3 training + capacity backups cleaned up; co-tenant jobs untouched.

---

## The full-data cross-check table (the trustworthy arbiter)
Only fixed-engine runs beat Meta; the data fix is additive on top.

| ckpt | full-data F1 | vs Meta (71.69) | fs10 F1 | data/engine |
|---|---|---|---|---|
| metaraw | 71.69 | — | 65.06 | off-the-shelf |
| **v3_e9** | **73.89** | **+2.2** | 66.27 (+1.2) | CLEAN, fixed sdpa |
| v1p1_e12 | 70.95 | −0.7 | (new) | dirty, broken (v3's epoch/res match) |
| v1_e9 | 69.93 | −1.8 | 66.88 (+1.8) | dirty, broken (~22ep/384px) |
| v2_e9 | running | — | 63.74 (−1.3) | dirty, fixed |

### Two big conclusions
1. **fs10 vs full-data ranking FLIPPED for v1.** fs10 had v1_e9 (+1.8) > v3_e9
   (+1.2); full data reverses to v3_e9 (+2.2) >> v1_e9 (−1.8). The fs10 10%
   subset over-rated v1 (high-variance head + epoch/res confound). fs10 is OK
   for "beats its anchor by a clear margin", NOT for fine cross-lineage order.
   FULL-DATA is the arbiter. The "beats Meta" claim rests on v3 (clean), not v1.
2. **The data fix, not epochs/resolution, gets above Meta.** v1p1_e12 (v1 at
   v3's exact 12ep/256px regime) is still −0.7 below Meta; clean v3_e9 is +2.2.

## What's running overnight (autonomous)
- **v3 training resumed** from e13 -> driving to e20. debug-scaling chain 8564302
  + capacity backup 8559744 (whichever lands; lock-coordinated). Saves e14, e19.
- **full-data campaign** finishing v2_e9 (last cross-check). driver pid 70195,
  log: full_cached_campaign.log
- **overnight v3 trajectory driver** (pid 68142, log: overnight_v3_trajectory.log):
  waits for e14 then e19 checkpoints, runs full-data cached + fs10 cached probe
  for each. THE DECISIVE TEST: does v3 HOLD above Meta at e19, or decline like
  v1/v2 did? If v3_e19 >= ~Meta -> data fix fixed the trajectory (clean win).
  If it drops -> black clips delayed but recipe (EMA/LR/horizon) drives late decline.

## Queue layout (no contention)
- v3 TRAINING -> debug-scaling + capacity
- PROBES -> debug (campaign + trajectory serialize via wait_debug_clear)
- Foreign jobs on the account (bisect_st, sft_compi, etc.) are NOT ours — left alone.

## If something looks wrong in the morning
- Check drivers alive: `kill -0 70195; kill -0 68142`
- v3 epoch: `python -c "import torch;print(torch.load('<ckdir>/latest.pth.tar',map_location='cpu',weights_only=False)['epoch'])"`
- Results land in docs/fs10_probe_findings.md (committed per result) + the driver logs.
- All probe configs committed+pushed (origin/aurora). Cache is delete-after-probe (345GB/ckpt).

## Recipe reference (optimized this session)
- Probe FAST recipe: cached + sdpa=true + bs4 + LR×2 + no sub-epoch saves.
  Full-data cached ~27-48 min/ckpt (export ~6min + probe ~2.1min/ep to early-stop).
- Benchmark verdict: sdpa=true ~halves time; batch plateaus past bs4 (encoder/IO-bound).
- metaraw cached=71.69 vs augmented 78.2 (−6.5 = expected aug-drop; cached valid for TREND).
