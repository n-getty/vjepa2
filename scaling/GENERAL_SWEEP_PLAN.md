# Fresh General Scaling Sweep — K700 + PE-Video (~1.6M clips)

## Why (the fix)
K400 (241K) was too small: budgets ≥1e19 replayed the corpus 10-90×, saturating the left-arm points
that define each parabola's vertex. Result: clean IsoFLOP vertices only at 1e18/3e18. Also the "general"
corpus (241K) was SMALLER than our surgical data (533K), inverting the "general → domain" narrative.

**Fix:** general corpus = K700_2020 (~597K) + PE-Video (~1M) ≈ **1.6M fresh clips**.
- general (1.6M) ≫ surgical (533K) → fairness/narrative restored
- 4×-fresh rule (tiny clips_seen ≤ 4×corpus) now holds through **1e19**:
  - 3e17: 0.07× | 1e18: 0.21× | 3e18: 0.60× | 1e19: 2.06× — ALL CLEAN

## Budget grid (corrected, ≤4×-fresh)
**Phase 1 (fastest full result):** 3e17, 1e18, 3e18 — all 6 sizes each = 18 cells.
- 3 vertices → α/β WITH error bars (headline result). Lowest budgets = fewest steps = finishes fastest.
**Phase 2 (extend range):** add 1e19 (5 sizes: small→gigantic; tiny drops if steps<200).

## Per-cell topology (UNCHANGED — gb=96 fixed, the D-vs-N invariant)
tiny/small=1n(3t,bs32) · base=1n(6t,bs16) · large=2n(24t,bs4) DDP · giant/gigantic=4n(48t,bs2) HSDP.
**4 nodes/cell is the HARD ceiling** (gb=96 / per_rank_bs≥2 = 48 tiles). More data does NOT raise this.

## Anti-slot-wait design (the time-savers, learned the hard way)
1. **PACK all cells into ONE debug-scaling job**, not many small jobs. Phase-1 = 3×13 = **39 nodes**
   concurrent (debug-scaling max=256, so fits easily; capacity's 16-node cap can't). One slot, not 18.
2. **Cap ipe so every cell's epoch < 45 min** → fits the 1h debug-scaling window → NO capacity detours,
   NO epoch-boundary deadlock (the two biggest time-sinks last run). Check: ipe×iter_time(~6s worst) <
   2700s → ipe ≤ 450. Total steps fixed by budget; ipe is free as long as ipe×num_epochs = total.
3. **Self-resubmit chain** (overnight_chain): checkpoint_freq=1, load_checkpoint:true, depth-guard 40,
   40×60s resubmit retry (rides sibling Q-contention). Right-size the link to nodes actually needed.
4. **All budgets clean at 1.6M** → no cell needs >4× replay → no repetition-saturated dead cells.

## Commands (reuse existing tooling — NO rebuild)
```
# 0. reshard/verify corpus dirs (K700 .tar.gz already ok; PE-Video already WebDataset .tar)
#    combined data-root = a dir listing both shard sets (or pass a mix)
# 1. plan the parallelogram (add 3e17; epoch-cap now moot since all clean)
python -m scaling.plan --budgets 3e17,1e18,3e18 --global-batch 96 --out scaling/manifest_gen.json
# 2. gen per-cell configs on the new corpus, per-size topology
python -m scaling.gen_configs --base <base_2_1.yaml> --manifest scaling/manifest_gen.json \
    --out-dir configs/scaling/general --data-root <K700+PE corpus> --use-topology --max-nodes 4 \
    --folder-root /flare/ModCon/ngetty/experiments/scaling_general
# 3. launch PACKED self-resubmitting chain (one ~39-node debug-scaling slot)
python -m scaling.overnight_chain start --ctrl <ctrl> --configs 'configs/scaling/general/*.yaml' \
    --nodes 39 --python python --max-depth 40
```

## Fastest-first option
If 39n schedules slowly: run **3e17 alone first** (13n, 6 cells, tiny job, schedules instantly, gives the
lowest vertex fastest), then 1e18, then 3e18 — each a small quick job. Trades one-slot-efficiency for
lower scheduling latency. Pick based on queue state at launch.

## Metric B ruler
Keep the canonical K400 held-out ruler (120 or 480 clips) — it's a FIXED external probe, independent of
the pretraining corpus, so it stays comparable across the K400 and K700+PE sweeps. (Ruler ≠ training data.)
