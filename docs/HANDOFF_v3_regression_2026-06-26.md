# HANDOFF — v3 surgical-CPT downstream regression investigation (2026-06-26)

## TL;DR
The "monotonic downstream regression" of v3 surgical CPT is **mostly a probe-resolution
artifact**, but a **real, mild decline remains** at native resolution.
- We CPT-train at **256px** but had been probing at **384px**. That mismatch grew
  every epoch and ~5x-inflated the apparent decline.
- Re-probing at native **256px**: surgical CPT **beats Meta at both e9 (+1.33) and
  e19 (+0.95)**, and the win **survives to e19** (at 384 it had collapsed to +0.30).
- BUT the advantage still erodes mildly with epochs (+1.33 -> +0.95, slope -0.38),
  monotonic. So the user's core concern stands: more surgical data still slowly
  degrades the downstream *advantage*, just far less than the 384 probe implied.

## The numbers (full-data cached probe, macro-F1)
| ckpt   | @384  | vs Meta384(71.69) | @256  | vs Meta256(73.44) |
|--------|-------|-------------------|-------|-------------------|
| Meta   | 71.69 | --                | 73.44 | --                |
| v3_e9  | 73.89 | +2.20             | 74.77 | +1.33             |
| v3_e14 | 73.06 | +1.37             | PENDING | --              |
| v3_e19 | 71.99 | +0.30             | 74.39 | +0.95             |

e9->e19 SLOPE: @384 = -1.90 (steep "regression")  vs  @256 = -0.38 (mild, real).

## IMMEDIATE NEXT STEP (the one unfinished probe)
Run **v3_e14 @256** to learn if the residual -0.38 slope is STEADY (ongoing
un-distillation) or FRONT-LOADED-then-flat (one-time settle). Config already
exists: `configs/heads/sarrarp50/full_cached_256/v3_e14_{export,probe}.yaml`.
- Needs export (the e14 384-cache was deleted) + probe. ~5min export + ~12min probe
  on 1 node (12 ranks).
- Driver ready: `scripts/run_256_probe_onnode.sh v3_e14 <port>` (does export->gate->probe).
- IMPORTANT launch caveat (cost me several retries): SSH lands in $HOME with no TTY,
  so use ABSOLUTE script paths; and `run_256_probe_onnode.sh` must NOT use `set -u`
  (module load frameworks trips on unbound ZSH_EVAL_CONTEXT) -- already fixed to
  `set -o pipefail`. Launch pattern that works:
    ssh <node> "nohup bash /lus/flare/projects/ModCon/ngetty/vjepa2/scripts/run_256_probe_onnode.sh v3_e14 29642 > /flare/ModCon/ngetty/logs/run256_v3_e14.log 2>&1 & echo \$!"
  Verify it started: tail the log for "[v3_e14] EXPORT".

## LIVE RESOURCES RIGHT NOW (verify before reuse with `qstat -u $USER`)
- **8568302** debug 1-node hold, node **x4310c4s0b0n0**, ~38/60min used (~20min left,
  may expire soon). Meta-256 probe FINISHED on it (early-stop e11, best 73.44 @ep5).
  It is IDLE -- could run v3_e14 if still alive, else submit a fresh node.
- 8568539 = **foreign** (br_gopred, co-tenant agent) -- DO NOT TOUCH.
- HPC discipline: debug = 1run+1queued PER USER, shared with the co-tenant. Only
  touch jobs you submitted (vjepa_hold / asformer / probe names). Use `qstat -f <id>`
  for the authoritative node, never a shared nodefile.

## REUSABLE ARTIFACTS
- 256 caches persist (154G each, world_size=12 -> must probe at 12 ranks/1 node):
  `/flare/ModCon/ngetty/surg_2_1_v2_final/probes/full_cache_256/{metaraw,v3_e9,v3_e19}`
  (v3_e14 cache NOT yet made). Probe-only re-runs are head-only (~12min) from these.
- All v3 checkpoints: `/flare/ModCon/ngetty/checkpoints/surg_2_1_v3_lambdaoff_cleandata/lr75e6_wu2_256_n16g12_weak/e{4,9,14,19}.pth.tar`
- Full findings: `docs/fs10_probe_findings.md` (read the last ~5 sections, in order:
  DECISIVE -> un-distillation -> resolution-sensitivity -> MAJOR CORRECTION ->
  CORRECTION TO THE CORRECTION -> NATIVE-RES DELTAS).
- Memory: `memory/v3-regression-rootcause.md` (has the corrected conclusion).

## ROOT-CAUSE RANKING (current best understanding)
1. **Resolution mismatch (train256/probe384)** -- PRIMARY driver of the *steep*
   apparent regression. Fix is trivial: probe/report at 256, OR retrain CPT at 384
   to match the strong init. Strongly recommend reporting all probes at 256 going
   forward (or standardize one resolution end-to-end).
2. **Residual un-distillation drift** -- real but MILD at native res (the -0.38).
   Meta init is ViT-L *distilled from ViT-G*; CPT replaces the ViT-G teacher with
   EMA-of-our-own-ViT-L -> slow relaxation out of the ViT-G basin. Feature diagnosis
   confirmed GLOBAL drift (kinetics degrades as much as surgical; cos-to-Meta
   1.0->0.84). This is the genuine remaining lever. Candidate fixes: keep frozen
   Meta as a distillation teacher during CPT / regularize features toward Meta /
   anchor (freeze) the EMA teacher near Meta.
3. EMA momentum / LR / temporal-masking / micro-dataset memorization = second-order.

## METHODOLOGY LESSONS (to not repeat)
- The fs10 10% cached probe flipped sign vs full-data near the anchor for v1, v2,
  AND v3_e14 -> it is only a COARSE screen. Full-data cached is the arbiter.
- The cosine-similarity proxy (res_sensitivity.py) WRONGLY called resolution "minor"
  (e19 cos 0.86 @both res) -- small cosine diffs masked a large F1 effect. Trust the
  real F1 re-probe, not the proxy. (User-suggested 256 re-probe is what cracked it.)
- Report MACRO-F1 (our target), not accuracy (accuracy looked rosier and misled once).
- Probe speed: cached + sdpa=true + bs4; ~30-48min/ckpt full-data (export ~6min +
  probe ~2min/ep, early-stops). Batch barely helps (encoder/IO-bound); caching is
  the lever. Confirmed earlier this session.

## OPEN QUESTIONS / DECISIONS FOR THE USER
1. Standardize probe resolution at 256 (matches CPT) -- or retrain CPT at 384 to
   match the distilled init + 384 probe? (384 retrain may both lift absolute F1 AND
   remove the mismatch, but costs a full CPT run.)
2. The residual mild decline: worth a distillation-anchoring experiment (frozen-Meta
   teacher) to turn the e9 peak into a sustained gain? That is the real remaining
   science once resolution is standardized.
3. Best current checkpoint = **v3_e9** (peak at both resolutions).

## GIT
On branch `aurora`, all analysis committed + pushed through 6ffb784. Working-tree
mods (app/main_dist_aurora.py etc.) are pre-existing Aurora-port edits, not part of
this investigation -- leave them.
