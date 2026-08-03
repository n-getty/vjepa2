# SAR probe throughput & topology guide (Aurora, 2026-07-08)

How to run the SAR-RARP50 asformer downstream probe FAST and CORRECTLY on Aurora,
and why the knobs are set the way they are. Written after a full profiling
investigation of "why does a small head on a frozen 2B backbone take 3+ hours?".

Companion: `docs/VITG384_CPT_SAR_2026-07-01.md` (the science/results),
`docs/fs10_probe_findings.md` (ViT-L lineage). This doc is about **speed**, not F1.

> **Running a GraSP probe? Read `docs/GRASP_REBUILD_2026-08-03.md` first.** The GraSP
> clips were rebuilt on 2026-08-03 (the old ones had 6.6% wrong labels and leaked a
> visual shortcut). New CSVs + 3 changed config numbers; old GraSP results are not
> comparable to new ones. The topology advice below still applies unchanged.

## ★ SUMMARY ★

The probe was never "slow to train a small head." Three separate things stacked up:

1. **The cached head-train is COMM-bound, not compute-bound.** Profiling (job 8654575)
   showed `backward = 97%` of iter time — that is DDP **gradient all-reduce**, not the
   head compute (`head = 2.7%`), not I/O (`data = 0.0%`). The head is also not tiny:
   it was **3 × 155M-param ASFormer heads** (an LR sweep run in parallel).
2. **Wrong default topology.** The probe defaulted to 2 nodes; a comm-bound workload
   scales *negatively* with more ranks. 1 node is faster per-iter for the cached probe.
3. **A pre-existing crash masqueraded as slowness.** The live/uncached path crashed at
   the epoch-end gather (oneCCL all_gather corruption) — it wasn't training slowly, it
   was dying at the epoch boundary. Fixed (gloo side-group).

**What actually makes it fast** (measured, in order of impact):
| lever | speedup | cost | default |
|---|---|---|---|
| single head (drop the LR sweep) | ~2.4× | **−4pt F1** | opt-in (sweep is default) |
| bf16 gradient compression | ~1.7× | small F1 (lossy) | **off** (opt-in) |
| single-DDP over heads + big buckets | ~1.0× | none (result-neutral) | **on** |
| mmap cache reader / staging / logging fixes | correctness/IO | none | on |

The big *free* win is **topology**, not a code trick: **export wide, train narrow**
(next section).

## The workflow that matters: two phases, OPPOSITE topologies

The probe has two phases with opposite bottlenecks. Do NOT run them the same way.

### Phase 1 — EXPORT (cache features): COMPUTE-bound → scale OUT
Run the frozen 2B backbone once over all clips, write features to disk. Forward-only,
under `no_grad`, **zero communication** (each rank forwards its own DistributedSampler
shard, writes its own `rank_N/` dir). Embarrassingly parallel → scales ~linearly.

Measured (fs_e159, 13,043 clips, full train+val):
| nodes | ranks | full export | speedup | parallel eff |
|---|---|---|---|---|
| 1 | 12 | 24.7 min | 1.00× | 100% |
| 4 | 48 | **8.1 min** | 3.05× | 76% |

The 76% (not 100%) at 4n is Amdahl: fixed startup (module load + 2B ckpt load + val
barrier + per-rank final `torch.save`) doesn't shrink with nodes. Sweet spot ≈ 4 nodes;
8+ gives diminishing returns. **Run export on 4 nodes.**

### Phase 2 — HEAD-TRAIN (train probe on cached features): COMM-bound → run on 1 NODE
Reads cached features, trains the head. Backward = gradient all-reduce = ~97% of iter.
More ranks = more all-reduce participants = SLOWER. **Run on 1 node.** ~2 min/epoch.

### End-to-end
- **Recommended (cached):** export 4n (~8 min, once) + head-train 1n (~40 min, 20 ep) ≈ **~48 min**
- All-1-node cached: ~25 min export + 40 min train ≈ 65 min

### If you must run LIVE (augmented) — it ALSO scales with nodes
Live re-runs the frozen 2B backbone every epoch (for augmentation, next section). Its
per-iter cost is 71% **encoder forward = COMPUTE-bound, same as export** — so unlike the
cached head-train, **live scales ~linearly with nodes** (more ranks → fewer clips/rank →
fewer iters/epoch, encoder per-iter cost unchanged):

| nodes | iters/epoch | ~min/epoch | 20-epoch |
|---|---|---|---|
| 1 | 237 | 25.7 | ~8.6 h |
| 2 | 118 | 12.9 | ~4.3 h |
| 4 | 59 | 6.4 | **~2.1 h** |
| 8 | 30 | 3.2 | ~1.1 h |

(1-node measured; 2/4/8-node are projections = 1-node per-iter × fewer iters. The backward
all-reduce (28%) will taper the scaling somewhat by 8n — same oneCCL inter-node cost as the
cached path — so treat ≥4n as approximate until measured.)

**So the earlier "1 node is best" rule is CACHED-ONLY.** It holds because the cached
head-train has NO encoder → backward-comm is 97% → more ranks hurt. Live has the encoder
(71% compute) → scale out. Two regimes, not a contradiction.

**But cached is still ~2–3× faster than even 4-node live** (~48 min vs ~2.1 h) and needs
1/N the nodes — so cached remains the default. Run live only to test whether augmentation
helps (next section).

## Cached vs live — they are DIFFERENT training regimes, not speed variants

The live TRAIN loader applies **per-epoch random augmentation** (VideoTransform
`training=True`: RandAugment `rand-m7-n4`, random-resized-crop scale 0.08–1.0,
random-erasing 0.25). So live features genuinely differ each epoch and **cannot be
cached** without dropping augmentation. The cached path exports ONE deterministic
(eval-transform) view, so cached probing trains without augmentation.

**Implication:** cached is fast but un-augmented; live is augmented but ~12× slower/epoch.
Whether augmentation improves SAR F1 is currently **unmeasured** — if you need the
augmented number, run live (now that it no longer crashes) and compare to the cached
73.9 anchor. Do not assume cached == live in accuracy.

## Config knobs (all in the probe YAML `experiment.optimization`, unless noted)

| knob | default | effect |
|---|---|---|
| `multihead_kwargs` | 3 entries (LR sweep) | **This is the dev-time LR search.** For production speed, use ONE entry (see `ab_e159_singlehead_probe.yaml`). The sweep hedges LR choice AND run-variance, so 1 head loses ~4pt F1 — keep the sweep for reported numbers, single-head for fast iteration. |
| `ddp_bf16_compress` | `false` | `true` = halve gradient bytes on the wire (~1.7× faster backward). LOSSY — small F1 cost. Opt-in. |
| `ddp_bucket_cap_mb` | `100` | DDP grad bucket size. Bigger = fewer/larger oneCCL ops. Result-neutral. |
| `accum_steps` | `1` | Grad accumulation: reduce+step every N microbatches (~N× fewer all-reduces). Changes effective batch → **retune LR**. Implemented + `static_graph` auto-guarded, but **UNTESTED** as of this writing. |
| `export_pool_spatial` (top-level) | `none` | `mean` = mean-pool the spatial token axis at export → ~576× smaller cache. LOSSY (replaces the head's learnable spatial-attention pool with a fixed mean; ~storage win, only ~1.2× speed since head compute isn't the bottleneck). Head auto-detects via manifest `pooled` flag. |

## Launcher

`scripts/run_asformer_probe_aurora.sh` now **derives topology from the PBS allocation**
(`-l select=N`), not hardcoded. So:
```
# export on 4 nodes (compute-bound):
qsub -A <acct> -q debug-scaling -l select=4 -v PROBE_CFG=<..._export.yaml> scripts/run_asformer_probe_aurora.sh
# head-train on 1 node (comm-bound):
qsub -A <acct> -q debug -l select=1 -v PROBE_CFG=<..._probe.yaml> scripts/run_asformer_probe_aurora.sh
```
- `VJEPA_PPN` overrides tiles/node (default 12 = full node).
- `VJEPA_STAGE_CLIPS=1` stages the ~13 GB live-clip corpus to node-local /tmp (only
  helps the LIVE path; cached reads .pt shards via mmap and doesn't need it).
- Aurora queue reminder: `debug` ≤ 2 nodes; use `debug-scaling` for 4-node export.

## Gotchas / bugs fixed this session (so you don't re-hit them)

- **Live epoch-end gather CRASH** (`RuntimeError: Storage size calculation overflowed`):
  oneCCL/xccl `all_gather` corrupts even fixed-shape tensors on the live path. Fixed by
  routing the tiny once-per-epoch gather through a **gloo (CPU) side-group**
  (`_cpu_gather_group` in `eval.py`). Zero throughput cost. See memory
  `probe-int16-gather-crash`, `xpu-allgatherobject-corrupts`.
- **Cached vs live must both be smoke-tested** for hot-path changes — cached (fewer
  iters) survived bugs the live path (more iters) crashed on.
- **SDPA on XPU:** the encoder wrapped SDPA in `torch.backends.cuda.sdp_kernel()` (a
  CUDA no-op on XPU). Replaced with device-agnostic `_sdp_kernel_ctx()` in
  `src/models/utils/modules.py`. Correctness cleanup, NOT a speedup (XPU dispatch was
  already picking the right kernel).

## Where the code lives
- Probe loop / knobs: `evals/video_classification_frozen/eval.py`
  (`run_one_epoch`, `_MultiHeadModule`, `export_feature_cache`, `_gather_1d_int`,
  `_cpu_gather_group`)
- Cached-feature loader: `src/datasets/backbone_feature_cache.py` (mmap reader)
- Encoder SDPA: `src/models/utils/modules.py`
- Launcher: `scripts/run_asformer_probe_aurora.sh`; clip staging: `scripts/stage_probe_clips.py`
- Config generators: `scripts/gen_fulldata_cached_probe_configs.py`, `gen_fs10_probe_configs.py`

## Method note
Every speedup claim here is from a profiled A/B (`profile_timing: true` gives the
data/transfer/encoder/head/loss/backward split). Several plausible-sounding causes were
disproven by measurement (I/O amplification, head compute, int16 dtype). **Profile
before attributing; verify a fix on the path it targets before calling it done.**
