# ViT-G 2B on Aurora — HSDP port findings & open drift problem (2026-07-03)

**Purpose:** hand-off for review. Goal = train ViT-G **2B** V-JEPA 2.1 continued-pretrain
at 16 nodes (192 XPU tiles) on Aurora. The 1B sibling already trains fine at 16n under
DDP. This doc records what's verified, what's still open, and what's been ruled out —
written to be reviewed by a fresh agent, so hypotheses are labelled as such.

---

## 1. The original problem (SOLVED): DDP L0-headroom starvation

**Symptom:** 2B under DDP at 16n — per-rank `backward-ms` climbs monotonically
(≈13s → 328s by iter ~42), job dies <1h. The 1B (bs=1) ran the identical launcher
clean for 4519 iters, flat ~9.2s backward.

**Root cause (confirmed):** L0-headroom starvation. Under DDP each tile holds the full
2B fp32 params + grads + Adam m/v + a full fp32 target-encoder (~43 GB fixed) and sits at
`max_memory_allocated` = **57.6 GB = 90% of the 64 GB tile**. Aurora CCL's per-collective
external memory (IPC handles + OFI fabric registrations) then has no L0 free space to grow
into → the gradient collective stalls every backward, worsening. The 1B had headroom; the
2B did not. (Cross-checked against torchtune `docs/reports/ccl_external_memory_growth_32b.md`
and `docs/features/allocator_strategy.md`.)

**NOTE on a wrong turn:** an earlier overnight conclusion blamed "intrinsic fabric
allreduce volume" and declared it unfixable without an ALCF ticket. That was **wrong** —
a 2B allreduce is smaller than the 7B/32B collectives PRISM/torchtune run fine. The user
corrected this. AR volume is not the wall; L0 occupancy is.

**Fix:** HSDP (FSDP1 HYBRID_SHARD) shards params/grads/optimizer across the 12 intra-node
tiles → **57.6 GB → 22.0 GB/tile (90% → ~34%)**. This removes the starvation root cause.

---

## 2. HSDP implementation (DONE, verified correct at 1n)

Additive, gated behind `VJEPA_DIST_STRATEGY=hsdp` (default `ddp`, unchanged). Commits on
branch `aurora` (f0fb70a … c39fd33 … a92603a …).

- `app/vjepa_2_1/hsdp.py` — 2D device mesh `(replicate=nodes, shard=tiles/node)` from
  `PALS_LOCAL_SIZE`/`WORLD_SIZE`; `wrap_hsdp` with `_HYBRID_SHARD_ZERO2` default,
  `transformer_auto_wrap_policy({Block})` (encoder+predictor share the `Block` class),
  `use_orig_params=True` (required — optimizer builds param groups by named-param WD
  filtering), bf16 `MixedPrecision`, FULL_STATE_DICT context helpers.
- `app/vjepa_2_1/train.py` — HSDP branch: **wrap first, then build optimizer** (use_orig_params
  needs it); DDP path byte-identical in the `else`. Weights loaded **pre-wrap** on the HSDP
  path (every rank reads the `.pt` independently, then FSDP shards) — see bug (b) below.
- `torch 2.13` venv (`/flare/ModCon/ngetty/venvs/torchtune-pt213-xpu`) — native FSDP1 + xccl.

### Bugs found & fixed during bring-up (all confirmed)
- **(a) `sync_module_states=True` hangs FSDP construction on xccl.** Rank-0 broadcast never
  completes (>900s); bare/`device_id` wrap finish in ~0.2s. Removed — safe because encoder
  loads a full ckpt (all ranks identical) and target_encoder is a pre-wrap deepcopy.
- **(b) Post-wrap FULL_STATE_DICT load wedges at 192 ranks.** `rank0_only=False` all-gathers
  the full 22 GB ckpt to every rank with redundant per-rank CPU copies (torch warns of this).
  Fixed by loading into raw modules before wrap.
- **(c) In-job stall watchdog added** — kills training if no first iter in 600s or a >300s
  stall, so a wedge self-terminates in minutes instead of idling nodes to walltime.

### Verified at 1 node (job 8643013, SMOKE_vitG384, real 2B)
- **EMA over sharded params correct:** `tests/test_hsdp_ema.py` (mpiexec -n 2) — sharded
  `_foreach` EMA matches unsharded reference to **max_err 2.98e-08**.
- **Memory 57.6 → 22.0 GB**, loss 0.34 (correct warm-start), ~6.6s/iter.
- HSDP ~10% faster/node than DDP at 1n (7.16 vs 7.93s), same config.

---

## 3. The multinode collective hang (SOLVED): transport, not the collective

At 2n+ (never at 1n, where the replicate mesh dim is size-1 and trivial) HSDP hung at the
**first `train_step`**: all ranks reached "Epoch 1", loader read a few clips, then a hard
freeze, 0 iters.

**Isolation (cheap 2n small-data repros, `SMOKE_vitG384`):**
- `collective_probe.py`: a **bare replicate-subgroup `all_reduce` WORKS** at 2n under our
  `pmix/mpi` transport (sum=2.0, all ranks). So the raw inter-node collective is fine.
- Ruled out: shard strategy (`_HYBRID_SHARD_ZERO2` and `full_shard` both hang), dataloader
  (`num_workers=0` irrelevant), load path.
- **Decisive A/B:** identical 2n code, only the CCL transport swapped —
  `pmix/mpi` **hangs at iter 0**; `launcher=none` + `ofi` + `CCL_KVS_IFACE=hsn0` +
  `FI_CXI_{RX_MATCH_MODE=hybrid,OFLOW_BUF_SIZE=8388608,DEFAULT_CQ_SIZE=131072}` (and dropping
  `--pmi=pmix` from the train mpiexec) → **60 iters clean** (job 8643134).

**Explanation:** a *single* all_reduce passes under mpi ATL, but FSDP's *sustained* rapid
intra+inter-node subgroup collectives deadlock on it. PRISM's launcher documents exactly
this (`launch_aurora_web.py:699`: "pmix/mpi causes MPI re-init errors inside mpiexec; use
launcher=none + ofi"). This is now the transport in both `scripts/vitG384_hsdp_spike_16n.sh`
and the production `scripts/vitG384_capacity.sh`.

---

## 4. ROOT CAUSE OF THE DRIFT: TWO VA-churn sources (both fixed, verifying 16n)

**Empirical result (2026-07-03, post-review).** The reviewer proposed two causes.
Both are real; the first is necessary-but-not-sufficient, the second is dominant.

**Cause 1 — per-layer FSDP wrapping (fixed; verified engaged; NOT sufficient alone).**
`transformer_auto_wrap_policy({Block})` made the 2B ~72 FSDP units → ~72 inter-node
collective triples/step. Fixed in `hsdp.py`: `shard_grad_op` now wraps **top-level**
(one unit, no auto_wrap_policy), mirroring PRISM `distributed.py:458` and torchtune
`CLAUDE.md:15`. **16n spike 8643247 confirmed top-level wrap engaged (all 6 modules
"no auto_wrap_policy") but STILL DRIFTED** — windowed iter-time
19.5→9.5→16.3→35.2→13.0→10.7→16.6→27.0→**54.0**s over 177 iters. So the collective
count was one churn source, not the only one.

**Cause 2 — variable mask keep-lengths (fixed; the dominant driver).**
V-JEPA's mask collator draws a new block size/position **every step**
(`multiseq_multiblock3d.py:__call__`), and `max_keep: null` in every vitg384 config,
so the kept-token counts vary step-to-step. **Measured directly** by driving the real
collator 300 steps at 384px/fpc16/bs2 (grid 8×24×24 = 4608 tokens):

| mask cfg | ENC len range | ENC distinct/300 | PRED len range | PRED distinct/300 |
|----------|---------------|------------------|-----------------|--------------------|
| 0 | 952–2288 | **122** | 1752–3368 | **124** |
| 1 | 96–1344 | **55** | 3128–4224 | **42** |

Nearly every step produces a new tensor length → the caching allocator hands out a
new VA → CCL registers a fresh L0 IPC handle + OFI MR that iteration → external
memory grows outside torch's pool (max_memory_allocated stays flat at 22 GB), and the
collective progressively stalls. This is exactly torchtune `allocator_strategy.md:29-30`
("Stable L0 VAs … No L0 VA change → no stale IPC handles") and the mechanism PRISM
removes with `BucketedMultiWebDatasetWrapper` (fixed seq buckets) and torchtune with
its bucketing allocator. Their bounded ~10 MiB/100-step figure is the *fixed-shape*
regime; V-JEPA was in the *churning* regime.

**Fix (additive, opt-in, default = bit-for-bit unchanged).** `multiseq_multiblock3d.py`
gains `num_keep_enc`/`num_keep_pred` per mask cfg: pin the enc/pred kept counts to an
exact constant every step (truncate longer draws, pad shorter by repeating indices;
`apply_masks` is a pure gather so duplicate indices are valid). Verified: with the
fields set, keep-lengths collapse to **exactly one value each** over 300 steps (from
40–124); with them unset, the path is unchanged (39 distinct/50 steps). Values chosen
at the measured medians (cfg0 enc 1536/pred 2560; cfg1 enc 768/pred 3584). Config
`vitG384_fixedshape.yaml`; A/B spike `vitG384_hsdp_fixedshape_16n.sh` (job 8643289).

**16n A/B result (2026-07-04, job 8643320, clean CSV + 720s watchdog).** Fixed-shape
masks **changed the drift's *character*** — the monotonic *baseline climb* is gone, but
**intermittent recover-spikes remain**. Raw per-iter (rank0, seconds), iters 21–68:

```
5 6 6 11 22 5 6 5 6 5 6 6 8 4 5 5 16 4 7 4 5 13 5 5 6 5 6 5 8 6 5 5
8 8 8 6 24 8 6 15 7 7 8 5 19 6 8 13 9 17 7 15 7 14 9 8 11 11 23 27 245 46
```
- **Baseline is FLAT and LOW: median 7.6 s** post-warmup (plain-wrap climbed 9→16→32→54
  and its *baseline* rose; here the baseline holds ~5–8 s the whole time).
- **But mean = 15.0 s** because of sparse spikes: 5 of 48 iters > 20 s, including a single
  **244.5 s** spike at iter 67 (then 46 s at 68 — partially recovering when this snapshot
  was taken). So spikes are getting *larger/rarer* even as the baseline stays flat.

**Interpretation (hypothesis, unconfirmed).** Fixed-shape masks removed the *dominant,
per-step* VA churn (baseline no longer climbs) — that half of the reviewer's diagnosis is
**confirmed by the flat baseline**. The residual big spikes point to a *third, lower-
frequency* VA/registration event that is NOT per-step:
  - candidate A: a remaining variable-shape tensor *other* than the two mask cfgs — e.g.
    the **image branch** (`img_mask`) or a per-source `dataset_fpcs` path (all 16 here, so
    unlikely), or the `default_collate` of the video buffer itself if clip length varies;
  - candidate B: a **periodic allocator event** — `gc.collect()` every 50 iters
    (`GARBAGE_COLLECT_ITR_FREQ`) can free+reallocate segments → new VAs → a burst of IPC/MR
    re-registration → one huge stall, then recovery. The spike cadence (iters ~43, ~50,
    ~55, ~67) is not obviously period-50 but overlaps it; worth correlating.
  - candidate C: CCL/OFI **cache eviction** when the (now fewer) registrations still cross
    a threshold, causing a periodic rebuild.

**Next diagnostic step (not yet run):** correlate spike iters against (i) the gc cadence
(try `GARBAGE_COLLECT_ITR_FREQ` off or =1 to see if spikes move/vanish), and (ii) whether
`num_workers`/prefetch boundaries align. If gc is the trigger, the fix is to stop freeing
segments (torchtune's `kBucketCap=8GiB` + `garbage_collection_threshold` already partially
addresses this — but our alloc-conf may still gc). This is likely another *fixable* issue,
same class (VA stability), just lower-frequency than the mask churn.

### RESIDUAL SPIKES DIAGNOSED (2026-07-04): allocator segment growth → CCL MR accumulation

The residual spikes are **NOT random and NOT gc** (`sync_gc` defaults False, confirmed not
in config — that code path never fired). Two facts localize the cause exactly:

1. **The spike lives entirely in `backward-ms`** (per-phase CSV, job 8643320 iters 55–69):
   fwd-target/fwd-context/opt/ema/dataload all stay flat (~2–4 s, ~0 s) while backward goes
   `4.6 → 12.3 → 18.7 → 24.1 → [102 desync] → 42.8 → …` and iter-time hits 245 s then 326 s.
   So it is the **gradient collective stalling**, not compute or the loader.
2. **The small spikes are quasi-periodic and rising in frequency** (spike iters 11, 23, 28,
   43, 46, 51, 54, 56, 58, 60, then continuous 64+), i.e. an **accumulator crossing a
   threshold** — not i.i.d. noise. Baseline stays flat ~7 s between them.

**Mechanism (now matched to torchtune's documented 32B signature).**
`static_xccl_buffer_weight_sync.md:5` + `allocator_strategy.md`: with
`CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536` (which we set to avoid banned:1 eviction
crashes), **CCL never evicts IPC handles, so it accumulates one MR/IPC entry for every
allocator *segment* it ever touches.** Fixed-shape masks stopped the *per-step* new-VA
churn (baseline no longer climbs), but the PyTorch caching allocator still **grows/fragments
its pool over the first ~60 iters**, touching new 1 GiB segments; each new segment → new CCL
MR → L0 free memory (~2.5 GiB headroom) is slowly depleted → backward collectives stall,
escalating until continuous. That is precisely the iter-64 onset we see. This is the SAME
external-memory class as the original DDP wedge, one layer deeper: DDP starved L0 immediately
(90 % occupancy); HSDP+fixed-shape delays it to ~iter 64 but doesn't bound it.

**The actual production fix (identified, not yet applied): the pluggable caching allocator.**
torchtune's validated mechanism for **≤4B models** (our 2B qualifies) is
`recipes/dev/usm_caching_alloc.so` — a **power-of-2 bucketing** XPU allocator
(`allocator_strategy.md:133`, "Production for ≤4B, 130 steps validated"). Power-of-2 buckets
mean only a *fixed, small set of segment sizes* is ever allocated, so after warmup the pool
stops touching new segments → CCL stops minting new MRs → external memory **plateaus**. It is
activated in the trainer via `torch.xpu.memory.XPUPluggableAllocator` +
`change_current_allocator(...)` BEFORE any XPU allocation (torchtune `distributed.py:~101`),
with `XPU_USM_ALLOC_SO` pointing at the `.so`. Caveats to handle: (a) these allocators don't
implement `getMemoryInfo`, so our `max_memory_allocated` logging will need a guard; (b) must
run before the first alloc; (c) do NOT use at 32B (documented to fail there — irrelevant to us).
`expandable_segments:True` is NOT an option — it's an Intel oneCCL USM-pointer-rejection bug
on Aurora (torchtune `allocator_strategy.md:71`), a no-op at best.

**Status / clearance.** Two of three churn sources fixed & verified (top-level wrap: baseline;
fixed-shape masks: per-step VA). Third source (segment-growth → MR accumulation) **diagnosed
and mechanism-matched**; fix = pluggable power-of-2 caching allocator, staged next. **NOT yet
cleared for the 12 h `capacity` launch** — a 245 s stall accumulating every ~20 iters would
destroy throughput. The good news: the run is otherwise healthy (loss 0.33 stable, mem 22 GB,
all 192 ranks synchronized between spikes), and the remaining fix is a known, validated,
additive allocator swap rather than an open research question.

**Also fixed in passing:** HSDP resume derefed `None` when a checkpoint contained opt
state (opt is built post-wrap, passed as `None` to `load_checkpoint`). Now guarded on
the local opt object. And the 1n smoke's EMA test hung because it inherited
`WORLD_SIZE=12` from the training stage while launched `-n 2`; fixed with `env -u
WORLD_SIZE`. EMA correctness re-confirmed PASS under **both** wrap modes (max_err 2.98e-08).

---

## 4a. (HISTORICAL) first review pass — per-layer FSDP wrapping only

**Verdict:** the §4-below drift was driven by **per-layer FSDP wrapping** —
`transformer_auto_wrap_policy({Block})` in `hsdp.py`. On ViT-G 2B that is
~48 encoder Blocks + 24 predictor Blocks = **~72 FSDP units → ~72 inter-node
collective triples per step** instead of ~1. This is the single most-condemned
pattern in both sibling KBs, and it is the actual "production mechanism" §6 asked
for — **not an env var, the wrap granularity.**

**Ground truth in the sibling repos (verified by reading the code, not inferred):**
- **PRISM** `src/training/distributed.py:458-469` — for `shard_grad_op` it sets
  `wrap_policy = None` (top-level only), with the comment: *"Per-module wrapping
  causes catastrophic overhead on XPU because each wrapped module does independent
  communication ops. With top-level-only, FSDP does a single ReduceScatter for
  gradients (like DDP AllReduce)."* Their production OLMo-3 E2E (2/4-node) runs this.
- **torchtune** `CLAUDE.md:15` & `:353` — *"FSDP per-module wrapping causes
  catastrophic overhead on XPU — use top-level-only wrapping."*
- **torchtune** `allocator_strategy.md:29-30` — bounded external growth ==
  **stable L0 VAs**; the caching allocator keeps segment VAs pooled so CCL's IPC
  handle cache stays valid. The 10 MiB/100-step figure is that regime.

**Why per-layer explains the *drift* (not just a high-but-flat cost):** each unit's
collective registers L0 IPC handles keyed by buffer VA. 72× the collectives = 72×
the IPC-handle churn rate; with `CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536`
(accumulation mode) handles pile up and lookup cost grows → progressive stall →
drift + spike-and-recover, all with flat torch-mem (growth is external/CCL). Same
signature as `ccl_external_memory_growth_32b.md`, ~36-72× faster than torchtune's
top-level baseline — which is exactly why our drift is so much steeper than their
~10 MiB/100 steps.

**Fix applied (`hsdp.py`):** wrapping is now strategy-aware, mirroring PRISM:
`shard_grad_op` (our default) → **top-level, no auto_wrap_policy** (~72 → ~1
collective/step); per-Block policy kept only for the `full_shard`/HYBRID_SHARD
fallback (which needs it for memory, and which PRISM flags as the slow XPU path).
`HSDP_WRAP={toplevel,perlayer}` overrides. `tests/test_hsdp_ema.py` now validates
EMA-shard alignment under **both** wrap modes. Our 2B at 22 GB (42 GB headroom) is
squarely the regime where top-level `shard_grad_op` is the production default —
PRISM only OOM'd top-level `shard_grad_op` at **7B** (14 GB flat param).

**Second, complementary VA-churn source (NOT yet applied — deferred):** V-JEPA's
mask collator (`src/masks/multiseq_multiblock3d.py:197-235`) draws a new
`min_keep_enc/pred` **every step** (`max_keep: null` in configs), so activation
tensor lengths vary step-to-step → fresh VAs → fresh IPC/MR registrations. This is
the churn source PRISM removes with `BucketedMultiWebDatasetWrapper` (fixed shape)
and torchtune with its bucketing allocator. **Plan: test top-level wrap ALONE
first** (72× is the dominant multiplier; may flatten the drift entirely). Only if a
residual remains, pin the mask keep-lengths (a recipe change — do last, carefully).

---

## 4b. (HISTORICAL) OPEN PROBLEM as originally written: iter-time drift at 16n

With HSDP + OFI applied, the 16n run (job 8643156, `SMOKE`-corpus, ipe120) **trains** — it
is *not* the DDP wedge — but iter-time **drifts upward**:

| iters | mean iter-time |
|-------|----------------|
| 1–20  | 8.9 s |
| 21–40 | 9.9 s |
| 41–60 | 16.9 s |
| 61–80 | 32.3 s |

- Good iters still recover to ~6 s (spike-and-recover), but the baseline + spike frequency
  climb. **Not** DDP's unbounded monotonic runaway, but a real upward trend.
- `torch [mem]` **flat at 22 GB** throughout → the growth is **external** (CCL/OFI/L0
  handles), invisible to PyTorch's allocator — the same *class* as
  `ccl_external_memory_growth_32b.md`, but this is **pure pretraining with NO weight-sync**,
  where torchtune measured only ~10 MiB/100 steps. Ours drifts far faster, so something
  specific to our setup is driving it.

### Environment already applied (did NOT stop the drift)
`PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.95`, `FI_MR_CACHE_MONITOR=disabled`,
`CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536`, `unset XPU_USM_ALLOC_SO`, `CCL_WORKER_COUNT=1`,
`CCL_OP_SYNC=1`. HSDP already shards (22 GB, tons of headroom — so it is **not** L0
starvation this time).

### Trainer facts that may be relevant
- `use_sdpa: true` (config) — and our own memory (`xpu-sdpa-bug-confounds-pretraining`) flags
  XPU FlashAttention-SDPA as *numerically* suspect for us (separate issue). **User constraint:
  SDPA must stay on** (eager cripples throughput; PRISM/torchtune run SDPA in production).
- Trainer does **not** call `torch.xpu.empty_cache()` (good — that's a known UR-leak trigger).
- Trainer calls `gc.collect()` every 50 iters (`GARBAGE_COLLECT_ITR_FREQ`, when `sync_gc`).
- `PhaseTimer` uses xpu events every iter; per-iter timing overflow (negative backward-ms)
  appears on desynced ranks at spikes (cosmetic, but marks the desync).

### What is KNOWN to work elsewhere (the workaround we haven't correctly identified)
The user states PRISM and torchtune run **production multi-node HSDP/FSDP with SDPA** —
0.6B/1B/2B/7B up to 20 nodes (PRISM) and up to 32B (torchtune, incl. GRPO+vLLM) — **without**
being restricted to checkpoint-restart. So a real, documented workaround exists that keeps
external growth bounded on long runs. **We have not yet correctly located it.** Candidates
NOT yet confirmed for our case:
- a specific CCL env (e.g. `CCL_SYCL_*_TMP_BUF=1` persistent temp buffers — flagged in
  torchtune `UPSTREAM_FILING_DRAFT_l0_resource_pool.md` as "most plausible non-structural
  workaround not yet exhausted");
- a torchtune/PRISM **recipe code pattern** for how FSDP is driven (reshard/PG/grad handling);
- a production launcher env block we haven't diffed line-by-line against ours.

### Explicitly rejected (do NOT re-propose)
- "Avoid SDPA / use eager" — rejected by user (perf), and PRISM Issue 10 was RESOLVED with
  SDPA ON.
- "Bad/dirty nodes" — rejected: Aurora validates nodes before release; our runs started clean.
  Do not blame hardware.
- "Checkpoint-restart every N steps" — this is the *stale* torchtune workaround; the user
  states they are no longer restricted to it. Only a last resort, not the answer.

---

## 5. Current state of the tree (branch `aurora`)
- HSDP code (`hsdp.py`, trainer branch, EMA test) committed & 1n-verified.
- OFI transport applied to `scripts/vitG384_hsdp_spike_16n.sh` and
  `scripts/vitG384_capacity.sh` (production 12h launcher, converted DDP→HSDP+OFI, pt213 venv).
- Grad-accum (`VJEPA_GRAD_ACCUM`) + `d_weights` `squeeze(1)` bs≥2 fix committed (secondary lever).
- No long run launched — blocked on resolving the §4 drift.

## 6. The one question for review
**What is PRISM/torchtune's actual production mechanism that keeps CCL/L0 external memory
bounded over long multi-node HSDP+SDPA training runs (no weight-sync)?** Once identified,
apply it, re-run the 16n spike to confirm flat iter-time past ~120 iters, then launch
`vitG384_capacity.sh`.

Full blow-by-blow: memory `vitG-2b-allreduce-spikes.md`. Key repro scripts:
`scripts/vitG384_hsdp_2n_ofi.sh` (2n clean repro), `scripts/collective_probe.py`,
`scripts/vitG384_hsdp_spike_16n.sh`.
