# ViT-G 2B on Aurora — HSDP port findings, failure-mode taxonomy & live-run status (2026-07-03 → 2026-07-05)

**Purpose:** hand-off for review. Goal = train ViT-G **2B** V-JEPA 2.1 continued-pretrain
at 16 nodes (192 XPU tiles) on Aurora. The 1B sibling already trains fine at 16n under
DDP. This doc records what's verified, what's still open, and what's been ruled out —
written to be reviewed by a fresh agent, so hypotheses are labelled as such.
**Read the "CURRENT STATE (2026-07-05)" section first — it supersedes the older verdicts below.**

---

## CURRENT STATE (2026-07-05) — TRAINING LIVE & SELF-HEALING; three failure modes catalogued

**Training is running and banking epochs autonomously.** The 2B CPT resumed from the Meta
`vjepa2_1_vitG_384.pt` init and has progressed **e15 → e81+** (66+ epochs) across 4 self-healing
capacity jobs. Recipe: HSDP `shard_grad_op`, OFI transport, `CCL_WORKER_COUNT=1`,
`CCL_ALLREDUCE=ring`, `VJEPA_NUM_WORKERS=0`, `VJEPA_TRUE_ACCUM=1` (ga=1), fixedshape config,
**6h walltime in-script** (every resubmit successor inherits 6h — user hard constraint: no 12h jobs).
loss_pred healthy ~0.31-0.33; total loss rises only via the scheduled λ ramp (0→~0.5), every
component stable. LR decay is **step-based** (WarmupCosine, 999 warmup / 9990 T_max steps); the
ipe reslice (ipe30×ep333 = ipe333×ep30 = 9990 steps) is exactly LR-equivalent to the 1B recipe.
Current ~2040 steps ≈ **1B-epoch 6** (1B saturated ~ep19 ≈ 6327 steps ≈ 2B-ep211) — plenty of runway.

**Startup blocker SOLVED — Mode A shm crash = `VJEPA_NUM_WORKERS=0`** (DataLoader worker mp/shm
handoff fragile at 192 ranks; NOT /dev/shm exhaustion). 3× clean 16n startups gated it; holds.

**THREE DISTINCT RUNTIME FAILURE MODES (do not conflate — each has a different cause + fix):**

1. **Silent fabric hang — ROOT-CAUSED 2026-07-05 (job 8645821 forensics, 384 stacks captured).**
   The per-rank faulthandler watchdog fired at 600s and dumped stacks: **~396 thread-frames blocked in
   the BACKWARD gradient AllReduce** (`distributed_c10d.py:3239 all_reduce` ← `fsdp _reduce_grad` ←
   `_post_backward_hook`) while **24 frames are one collective BEHIND** in the next iter's forward
   all-gather (`_pre_forward_unshard` → `_unshard`). **MECHANISM = FSDP collective DESYNC, not a dead
   rank:** a few ranks lag entering the fwd all-gather; the majority reach the backward AllReduce and
   block forever waiting for the stragglers, which wait on a collective the majority already left →
   deadlock. **This UNIFIES spikes + hangs as ONE phenomenon at different severity** — fabric
   contention makes ranks lag; mild → recoverable cohort-wide spike (87% stuck, then proceed; flat ~2s
   backward floor rules out mem-accum), severe → straggler never catches up → permanent deadlock →
   watchdog kills. It is the HSDP ZERO2 grad-AllReduce over the 16-node replicate dim. 1B avoided it
   (~half grad volume → shorter collectives → smaller straggler window). Handling: watchdog (1800s) →
   forensics (SIGUSR1 stacks + nodefile) → pkill → auto-resubmit. **This is the concrete reproducer
   AGPT/torchtitan lacked** ("silent, no traceback") — we now HAVE the traceback → ALCF ticket.

2. **Stochastic bf16 NaN** (6 events: e16/r180, e23/r58, e31/r175, e36/r141, e51/r3, e57/r152 —
   every one a DISTINCT rank AND host). ~1/400 opt-steps, definitively stochastic (not a bad node,
   not checkpoint-deterministic, not a corrupt clip — full source scan = 0 non-finite clips; the 1B
   ran the same raw sources clean unprotected). **FIX = symmetric NaN-skip guard** (all_reduce MIN
   finite flag → all ranks skip the opt step together, EMA/schedule advance, no crash), logged with
   rank/host/itr. Zero crashes where the old `assert not np.isnan(loss)` would have been 6. NO grad
   clipping added (Meta recipe uses none; bounded L1-to-EMA loss; the two numerical traps —
   loss_exp=1.0 and 1/d_ij — already closed). Verdict: survivability-solved, not worth root-causing.

3. **Bad-node death / signal 9** (1×: job 8645296, rank 23 "died from signal 9" on x4213c5s5b0n0,
   cascade SIGTERM, PBS Exit_status=0, used 3:54/6:00 — NOT walltime, NOT our watchdog, NOT a hang).
   The SYSTEM killed a rank (OOM-killer / node health-check / hardware) = AGPT's documented
   "shepherd died from signal 9" bad-node signature. Auto-resubmit still recovered (8645821). Added
   `bad_nodes.txt` logging to track host recurrence for ALCF ticketing / future PBS node-exclusion.

**HANG FORENSICS (the real gap the user identified — we were blind-killing hangs).** AGPT and PRISM
both instrument the stuck rank; we captured nothing. Now shipped (env-gated, default-on in capacity):
- Per-rank in-process watchdog (`VJEPA_ITER_WATCHDOG_S=600`, train.py): faulthandler
  `dump_traceback_later` armed each iter → the stuck rank dumps its ALL-THREAD stack (exact blocked
  collective/line) BEFORE the shell kill. Validated at 1n (stall → correct stack).
- SIGUSR1 → faulthandler all-thread dump-and-continue (SIGABRT can't be registered on this build).
  Shell watchdog SIGUSR1s all 192 ranks + snapshots the nodefile to `hang_diag/` before pkill.
- Candidate fix staged behind `VJEPA_CCL_TMP_BUF=1` (default OFF): `CCL_SYCL_*_TMP_BUF=1` = persistent
  temp buffers instead of L0 IPC handles (torchtune signature-#3 workaround). Enable only if forensics
  confirm the L0-IPC path.
Ruled OUT with evidence: DAOS-17499 `libpil4dfs` FSDP-AllGather hang (PRISM's cause) — we are on
Lustre `/lus/flare`, no pil4dfs.

**CCL_ALLREDUCE A/B (DONE, scripts/vitG384_allreduce_ab_16n.sh, fixedshape ipe200 no-save):**
- RING (job 8645758, iters50+): backward p50=4820 p90=8314 p99=20207; spike-iters=**2/116 (2%)**; iter p50=9.6s
- double_tree (job 8645804, iters50+): backward p50=5031 p90=9259 p99=19835; spike-iters=**3/144 (2%)**; iter p50=10.0s
- **VERDICT: double_tree does NOT help — ring equal-or-better (~4% faster p50, identical 2% spike rate).**
  Ring chain-depth is NOT the spike driver (log-depth double_tree would have flattened them; it didn't).
  Both isolated runs spiked only ~2% vs the campaign's frequent spikes → the driver is the FABRIC
  ENVIRONMENT (inter-job dragonfly contention + bad nodes), NOT the AllReduce algorithm. **Keep ring.**
  The reviewer's ring-contention hypothesis is refuted by experiment; the spikes are accept-the-tax +
  ALCF escalation, now proven not assumed.

**Track record:** ~1 disruptive event / 3.5-4h, ALL self-healed (no lost progress beyond a partial
epoch). Net e15→e81 (66 epochs) across 4 jobs, fully autonomous. This SUPERSEDES the 2026-07-04
"FINAL VERDICT" below (which predated the NaN guard, forensics, and the three-mode taxonomy).

---

## FINAL VERDICT (2026-07-04, SUPERSEDED by the 2026-07-05 section above): fabric contention is the intrinsic 16n tax — accept it, make training survive it

Both fabric levers are now **exhausted by experiment**:
- **CCL_WORKER_COUNT=4** (8643434): failed (loader-fill hang; orthogonal, demoted).
- **TRUE_ACCUM=2** (8643462): **did NOT fix it** — wedged at 40 iters with a terminal 543 s stall
  → watchdog kill (vs env-diff's 74 iters at ga=1). Memory stayed perfect (l0-free 10.7 GiB flat),
  so true-accum is *correct and affordable* — it just doesn't cure the fabric stalls, because the
  §4g host-side collective stalls are intrinsic to CCL/OFI at this 16n scale, not a function of
  collective count or worker threads. This is exactly PRISM's documented, accepted congestion tax
  (`scaling_study.md:180`).

**The reframe that matters:** the spikes are a **throughput** problem, NOT **correctness** — loss
was sane in every run (0.33→0.31, no NaN), memory healthy, HSDP working. The model WILL converge;
it's just slower with occasional multi-minute recoverable stalls. So the goal is no longer "fix the
fabric" — it is **make the training run survive the stalls unattended**:
1. **Recipe = proven plain base** (`VJEPA_TRUE_ACCUM=1`, fixedshape, workers=1) — the validated
   74-iter config. Do NOT ship the unvalidated 2× batch/true-accum into a long run when it gave no
   benefit; keep the recipe we trust.
2. **Checkpoint cadence is already fine:** `CHECKPOINT_FREQ=1` = every epoch; an epoch ≈ 2.1 h clean
   (~4 h with stalls), so a 12 h run checkpoints 3-5× and auto-resumes from `latest.pth.tar`
   (train.py:375, verified). A crash loses ≤ ~1 epoch. Mid-epoch checkpointing = untested code,
   NOT worth the risk tonight.
3. **The only real gap = no auto-resubmit** if a stall hangs the job past the watchdog. Fix with a
   self-healing wrapper (resubmit-on-death) around the capacity job.

---

## OVERNIGHT PLAN (2026-07-04, live) — decision tree to a stable launch

Two levers now exist for the §4g fabric/host-stall residual, both PRISM-informed:
- **CCL_WORKER_COUNT=1→4** (zero code change; more progress-engine threads for the host-side
  collective drain that §4g pinned). Under test: **job 8643434** (16n, ga=1, ipe200).
- **VJEPA_TRUE_ACCUM=2** (real accumulation across 2 loader batches → 1 collective/step instead
  of 2; §4f). Implemented in `train.py`. 1n memory smoke: **job 8643448**. 16n stacked run
  (workers4 + true_accum2): `scripts/vitG384_hsdp_trueaccum_16n.sh` (ready, not launched).

**Gating decision tree (in priority order — cheapest healthy option wins):**
1. **8643434 (workers=4) flattens the spikes** (no host-stall, backward cohort-flat over ipe200)
   → cheapest win, NO recipe change. Launch training with workers=4 only.
2. **workers=4 helps but residual remains AND 8643448 smoke PASSES** (l0-free>5GiB, loss sane)
   → launch `vitG384_hsdp_trueaccum_16n.sh` (stacks both). If it flattens → launch training with
   workers=4 + true_accum=2.
3. **neither fully flattens** → the recoverable spikes are the 16n tax; the **self-resubmitting
   1h chain** (`vitG384_chain_debugscaling.sh`) is the robust vehicle — each slice is walltime-
   bounded so a host-side hang just ends the slice and the successor resumes from latest.pth.tar.

**LIVE STATUS (2026-07-04 ~05:00):**
- CCL_WORKER_COUNT=4 A/B (8643434): **FAILED** — WebDataset first-batch buffer-fill didn't
  complete before the 600s watchdog (all 192 ranks reached loader-init + data started flowing).
  Orthogonal loader-fill flakiness, but **demoted workers=4**; launch path reverted to the
  proven **workers=1** base (only 74-iter success used it). Raised first-iter deadline 600→900s.
- True-accum 1n smoke (8643448): **was a NO-OP** — copied a runtime cfg whose folder=`smoke_weak`
  had a STALE Jul-1 `latest.pth.tar` (epoch=1); trainer auto-resumed it → `range(1, epochs=1)`
  empty → 0 iters, exit 0, true-accum path never ran. NOT a code/loader bug. Fixed: smoke now
  patches folder to a unique CKPT_DIR + rm stale ckpt → fresh Meta init. Deleted the confounding
  `smoke_weak/latest.pth.tar`. **Re-running as 8643455** — this is the real gate.
  (This also means the earlier ga2 "PASS" was spurious — stale rows; neither accum path had
  actually executed. The true-accum implementation is still UNVALIDATED until 8643455 returns.)
- **True-accum 1n smoke (8643455): PASS** (fresh Meta init, epoch 0) — 40 real iters, true_accum=2
  engaged, loss 0.330→0.313 sane, steady ~9.8s/iter. **MEMORY GATE PASS: l0-free min 10.3 GiB,
  flat** — the `no_sync` full unsharded gradient hold fits with >10 GiB headroom, decisively
  answering reviewer point 7 and validating the deliberate divergence from PRISM's memory-bound
  7B (no_sync-OFF). True-accum implementation empirically validated. **16n verification: 8643462.**

**Launch vehicle:** `capacity` queue IS available tonight (22 running). Two options:
- `scripts/vitG384_capacity.sh` — single 12h job. FIXED tonight (fixedshape cfg, flags unset,
  workers=4). Auto-resumes from latest.pth.tar (verified train.py:375 — non-anneal path resumes
  unconditionally if the ckpt exists). BUT no watchdog/auto-resubmit → a multi-hour host hang
  idles the allocation to walltime.
- `scripts/vitG384_chain_debugscaling.sh` — self-resubmitting 1h slices, EXIT_AFTER_CKPT. Robust
  to hangs by construction. STILL HAS config drift (cleandata + workers=1) — must get the same
  fix before use as a real vehicle.

**Recipe decisions (unattended-safe, see memory `true-accum-lr-decision`):** LR unchanged at
7.5e-5 even though true_accum doubles effective batch — a less-noisy gradient at fixed LR is
*more* conservative per-sample, and LR-hotness is this model's demonstrated collapse mode. Never-
diverge >> squeeze-throughput for an unattended launch. `no_sync` ON diverges from PRISM's 7B
(they were memory-bound; our 2B fits the ~4GB bf16 full grad in ~15GB free L0 — fabric-bound).

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

### ⚠️ REVIEWER CORRECTION (2026-07-04) — the segment-growth conclusion above is NOT yet supported

A second reviewer flagged that the "allocator segment growth → CCL MR accumulation →
pluggable-allocator fix" conclusion (the three paragraphs above) was committed **on
inference, not measurement**, and has two unresolved tensions. Both check out against our
own data — the conclusion is **retracted pending measurement**:

1. **Spike-and-RECOVER contradicts monotonic depletion.** Our data: 245 s → 46 s →
   baseline ~7 s. Monotonic L0 depletion does not recover — it escalates to banned:1 (what
   DDP did). A transient that clears is the signature of a **one-rank straggler propagating
   through the collective barrier**, not a permanently-out-of-L0 pool.
2. **Per-rank evidence CONFIRMS one-rank straggler, not symmetric pool exhaustion**
   (job 8643320, checked rank0/50/100/150/191 at the spike iters):
   - iter 245 s: **rank50 `backward=241 s`** (the straggler) while all others show timer-
     overflow (`-102`, i.e. *waiting at the barrier*).
   - iter 326 s: now **rank0/100/150 wait** (`-21`) while rank50 is slow again and rank191
     is fast (`5`). **Different rank stalls each spike.** A symmetric allocator-pool problem
     would slow *all* ranks together; this does not. (Matches the earlier
     `vitG-2b-allreduce-spikes.md` note: "rank 150 +328 s late, different rank each spike,
     compute uniform" — which the segment-growth writeup overlooked.)
3. **Wrong counter measured.** The segment story rests on "the pool touches new segments,"
   which lives in `memory_reserved`, NOT the `max_memory_allocated` we logged (bytes in
   use). That was never measured. **Now added** (train.py logs `resv:` alongside `mem:`) —
   flat reserved ⇒ segment-growth hypothesis is **false**; climbing-then-plateau ⇒ warmup
   fragmentation (fix by pre-growing, not swapping allocators); unbounded climb ⇒ allocator
   implicated. Job **8643336** (ipe200, resv logging) is the decisive test.

**Also:** the pluggable-allocator fix, even if reserved climbs, is an **extrapolation not a
match** — torchtune validated `usm_caching_alloc.so` under FSDP2/GRPO, not FSDP1/HSDP, and
`UPSTREAM_FILING_DRAFT_l0_resource_pool.md` signature #4 is literally
"XPUPluggableAllocator + FSDP → banned:1 at step 1" (at 32B; likely not fatal at 2B, but it
is the exact fragile combination). So it is a last resort, not a first move.

**Corrected plan — cheap/safe/mechanism-aligned levers, in order (BEFORE any allocator swap):**
1. **Run fixed-shape to iter 200+** (job 8643336) — does spike frequency **plateau**
   (warmup artifact → launchable, possibly with checkpoint-restart) or **escalate to
   banned:1** (real depletion → allocator swap justified)? Everything downstream depends on
   this; we keep cutting at 68–177 iters and cannot tell which. **This run is isolated** (no
   grad-accum, no ZE change) so the `resv:` signal is clean.
2. **`FSDP_NO_SYNC_ACCUM` + `gradient_accumulation_steps=2`** — PRISM's single biggest win;
   halves inter-node AR frequency = halves the collective-stall/MR-registration rate, the
   exact accumulator. Already implemented (`VJEPA_GRAD_ACCUM`) but the spike scripts don't
   use it. Additive, reversible, throughput win regardless. NOTE: at bs=2, grad_accum=2 →
   micro-bs=1, which trips the `weight_distance_loss` bs≥2 landmine — mitigated by the
   `squeeze(1)` fix in `models/utils/modules.py`, but must be smoke-verified first.
3. **Unset `ZE_AFFINITY_MASK` (+ `set_device(local_rank)`)** — `main_dist_aurora.py:35`
   pins one tile/rank → CCL sees `node_dev_uuids` size 1 → intra-node collectives
   host-fallback (~1.75 GB/s). Slow collectives amplify every stall. PRISM unsets it in
   production. ~5 % + removes a contention amplifier. (Needs care: the 144-context concern
   the comment cites is real — verify context count doesn't explode.)
4. **Allocator warmup / pre-grow** — only if #1 shows reserved plateaus: front-load a
   max-shape forward/backward before the timed loop (à la torchtune
   `TORCHTUNE_COLOCATE_WARMUP_AT_MAX`). If that flattens the spikes it was warmup
   fragmentation and NO allocator swap is needed.

### ✅ MEASUREMENT VERDICT (2026-07-04, job 8643336, ipe200 + `resv:` logging)

The reviewer's diagnostic gate was run. Result: **the segment-growth hypothesis is
FALSIFIED, and the pluggable-allocator swap is NOT needed.**

- **`memory_reserved` is DEAD FLAT at 45.8 GiB from iter 0 → 67** (allocated flat 22 GB
  the whole time). The allocator is *not* touching new segments → CCL is *not* minting new
  MRs from pool growth. The segment-growth story is false; the pluggable allocator would
  have treated the wrong disease. (This is exactly the counter the reviewer said to measure,
  and it decided it cleanly.)
- **The residual is spike-and-RECOVER around a flat baseline, not escalation.** Full 73-iter
  trace (rank0, seconds): baseline repeatedly returns to **5–9 s**; spikes are bounded and
  transient — e.g. iters 60–67 = `15 16 50 18 104 8 7 9` (a 104 s straggler that recovers
  fully to 7 s the next iter). Post-warmup: **median 10.9 s, max 104 s, only 3 of 53 iters
  > 30 s.** No climb, no banned:1. The prior run (8643320) was at 245/326 s and dead by
  iter 69 — this run sails past that point healthy, so those mega-spikes were **run/node-
  specific degradation** (cf. 8643289 dying of a socket-comm auth fault), not the intrinsic
  behavior.
- **Per-rank data confirms the mechanism is a one-rank collective straggler** (from 8643320,
  still the clearest capture): at a spike, ONE rank sits in `backward` while all others show
  timer-overflow *waiting at the barrier*, and it's a **different rank each spike**. That is
  collective *contention/jitter*, not a symmetric memory problem.

**Revised conclusion (SUPERSEDED — see §4c below).** ~~Two causes, one-rank straggler
residual, no third memory cause, launch-capable.~~ This was wrong on the residual: it read
the spike from rank0's max only. The cross-rank distribution (§4c) shows the spike is
**cohort-wide**, and `reserved`-flat does **not** rule out CCL/OFI external growth.

---

## 4c. THIRD-REVIEW CORRECTION (2026-07-04): the residual is COHORT-WIDE, not one-rank

**What was wrong.** §4b concluded "one-rank collective straggler, different rank each
spike." That came from reading only `log_r0.csv`'s **max**. Computing the cross-rank
distribution from **all 192 per-rank CSVs** (`scripts/analyze_straggler.py`) shows the
opposite — at every spike, `min << p50 ≈ p90 ≈ max`:

| itr | min_bwd | p50_bwd | p90_bwd | max_bwd | class |
|-----|---------|---------|---------|---------|-------|
| 40  | 13.9 | 16.3 | 16.4 | 16.4 | cohort |
| 50  | 12.5 | 14.2 | 14.3 | 14.4 | cohort |
| 61  | 7.8  | 46.6 | 47.1 | 48.0 | cohort |
| 63  | 3.8  | 101.8| 102.2| 104.3 | cohort |
| 72  | 16.7 | 22.5 | 23.0 | 23.6 | cohort |

At iter 63 the **median** rank spent 102 s in backward — ~all 192 ranks inflated together,
not one. A one-rank straggler would show `min ≈ p50 ≈ p90` with only `max` jumping; the data
is the reverse. `analyze_straggler.py` classifies **0 of 73 iters as "1RANK", all spikes as
"COHORT".** This means **CPU-binding cannot root-fix it** (it targets per-rank NUMA jitter),
and neither can `ZE_AFFINITY_MASK` or `grad_accum` in any root sense. The socket-aware
binding I was about to implement is aimed at the wrong mechanism — dropped.

**Two tells point at external CCL/OFI accumulation, NOT "done":**
1. **The backward floor creeps up:** `min` backward rises from ~2.0–2.3 s (iters 1–8) to
   4–16 s (iters 40–72). A flat floor with isolated max-spikes = jitter; a **rising floor +
   synchronized cohort spikes escalating over time** = external-resource accumulation — the
   same class as `ccl_external_memory_growth_32b.md`, slower and not yet fatal in 73 iters.
2. **`reserved`-flat does NOT rule this out.** The §4b measurement (reserved dead-flat
   45.8 GiB) correctly killed the *PyTorch-pool segment-growth* hypothesis, but CCL/OFI
   external growth lives in the **L0 driver free pool**, which is invisible to `reserved` by
   definition (that is the entire point of the torchtune 32B report). So "reserved flat →
   no third cause" was an over-claim. The residual is plausibly a slower version of the
   known external-memory accumulation, now on the **inter-node replicate-dim AllReduce**
   (CXI/OFI path) rather than the intra-node XeLink path — and note our one env mitigation,
   `CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD`, governs **XeLink IPC**, not the inter-node
   CXI path the spike lives on.

**Compute is confirmed NOT the cause:** fwd-target/fwd-context are flat ~1.7–2.5 s across all
ranks every iter, so PRISM's seq-length-variance straggler root cause is genuinely eliminated
by the fixed-shape masks. The residual is a **synchronized, escalating, backward-collective
inflation with a creeping floor**.

### The decisive next measurement (instrumented, committed — run BEFORE any lever)

Reasoning from stdout maxes is what produced two wrong diagnoses. The next 16n run carries
two probes so the mechanism is read directly, not inferred:
1. **free-L0 per rank per iter** — `torch.xpu.mem_get_info()` free bytes, and the torchtune
   MEMPROBE metric `external = l0_used − torch_alloc` (memory held by CCL/L0 outside
   PyTorch's pool). Logged to CSV (`l0-free-mib`, `l0-ext-mib`) and stdout (`l0free`,
   `l0ext`). Committed in `app/vjepa_2_1/train.py`.
2. **cross-rank per-phase distribution** — `scripts/analyze_straggler.py` reads all 192 CSVs
   and prints the min/p50/p90/max backward spread + floor/free-L0 trend + a COHORT/1RANK/flat
   classifier per iter. Committed.

**The clean fork this single run resolves:**
- **free-L0 creeps DOWN as the backward floor creeps UP** → external CCL/OFI registration
  accumulation on the inter-node AllReduce → the real fix is a **static/persistent collective
  buffer** (torchtune `static_xccl_buffer_weight_sync.md`: a fixed VA for the collective so no
  new OFI MR is ever registered), NOT CPU-binding/grad_accum/allocator-swap.
- **free-L0 FLAT while timing spikes cohort-wide** → pure **fabric contention** (inter-job
  CXI congestion), which PRISM (`data.md:540–700`, `scaling_study.md:180`) investigated over
  ~8 jobs and accepted as a throughput tax ("AllReduce latency itself is only 24.7 ms/step —
  NOT the bottleneck"). In that case **`grad_accum` (fewer, larger ARs) is the acknowledged-
  correct mitigation, not a band-aid**, and occasional recoverable cohort spikes at 16n are
  documented expected behavior — "perfectly flat at 16n" is likely not a reachable target.

**Status: NOT launching anything until this measurement returns.** Two churn sources remain
fixed (wrap, per-step masks); they converted a fatal runaway into a run that survives 73+
iters. Whether the residual needs the static-buffer fix or is an accepted fabric tax is
decided by free-L0, not by another guess. Pluggable allocator and checkpoint-restart remain
off the table. Socket-aware CPU-binding is dropped (wrong mechanism for a cohort-wide spike).

**Also fixed in passing:** HSDP resume derefed `None` when a checkpoint contained opt
state (opt is built post-wrap, passed as `None` to `load_checkpoint`). Now guarded on
the local opt object. And the 1n smoke's EMA test hung because it inherited
`WORLD_SIZE=12` from the training stage while launched `-n 2`; fixed with `env -u
WORLD_SIZE`. EMA correctness re-confirmed PASS under **both** wrap modes (max_err 2.98e-08).

---

## 4d. FOURTH-REVIEW CORRECTION (2026-07-04): the two "insurance" env flags ARE the accumulator

**The reframe.** §4c framed the fork as "static-buffer fix vs accepted fabric tax" and
queued a free-L0 gate to decide. A fourth review caught a step upstream of that fork: the
gate script itself sets **three env flags that PRISM's production launcher
(`tools/launch_aurora_web.py`) is grep-clean of** — yet PRISM runs HSDP+ZERO2 at 20N stably:

```
CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536   # never evict IPC handles  (accumulate mode)
FI_MR_CACHE_MONITOR=disabled                     # never invalidate fabric MR cache
PYTORCH_ALLOC_CONF=garbage_collection_threshold:0.95   # GC-threshold, also added by us
```

All three were added by *me*, preemptively, "to avoid `banned:1` eviction crashes" —
textbook fix-a-bug-you-haven't-observed. `65536`=never-evict + MR-monitor-off is **by
construction** an accumulate-forever cache. **All three must be dropped on the env-diff run**:
if only the two cache flags were unset and the run flattened, we couldn't tell whether it was
the cache flags or the GC threshold without a second run — dropping all three makes a flat
result unambiguous in one job. The signature §4c attributed to a possible new external-memory
class — **cohort-wide spike + recover + a creeping floor** — is *exactly* what a never-evict
cache does when it periodically rebuilds/compacts under pressure: every rank does identical
collective volume per step under HSDP, so all 192 hit the same cache threshold at the same
iter (deterministic → cohort-wide, not one straggler), pay a synchronized rebuild, recover.
PRISM has run Qwen3-0.6B HSDP+ZERO2 at 20N for hundreds of steps **without** either flag.

This does not contradict §4c — the spike really is cohort-wide, and reserved-flat really
does *not* rule out external growth. It identifies the *source* of that external growth as a
knob we set, not a fabric mystery. It also explains why §4a's own reasoning ("with
`CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD=65536` handles pile up") pointed at the flag while
we kept the flag set.

**Two upstream verifications done while acting on this (both PASS — neither is the bug):**
- **Timing is not a sync-scoping artifact.** `PhaseTimer` (`src/utils/logging.py:72-88`)
  enqueues XPU events on the default stream in program order and syncs once in `to_dict()`;
  in-stream ordering means leftover forward async ops complete before the `backward_start`
  event fires, so `backward-ms` is the true device backward, not forward spillover.
- **Input tensor shape is constant.** `vitG384_fixedshape.yaml` has `dataset_fpcs` = 15×`16`,
  `batch_size:2`, `crop_size:384`, `tubelet:2`, `fps:4` → every clip is `(2,3,16,384,384)`
  and masks are pinned. No residual video-shape VA churn beyond the masks already fixed.

**Corrected order of operations (supersedes §4c's "run the free-L0 gate first").** Sequenced
as separate jobs so each result has full attribution — combining them = one job with ambiguous
results and a smoke-verification gap:
1. **Env-diff test — the single decisive run.** `unset` **all three** flags above; keep
   fixed-shape masks + top-level wrap; change *nothing* else; ipe200. No dependency on any
   unverified fix. This run *also* carries the free-L0 probe, so it resolves §4c's fork
   simultaneously — strictly more informative than the flags-on gate. If the drift/floor
   flatten → the flags were the accumulator (predicted) → **close out, launch-capable**. If
   `banned:1` returns → hunt the true shape/allocator interaction with default telemetry
   instead of masking it.
2. **squeeze-fix smoke (1n) — gate before ga=2.** ga=2 at bs=2 → micro-bs=1, which trips the
   `weight_distance_loss` bs≥2 landmine. The `squeeze(1)` fix in
   `src/models/utils/modules.py` is claimed to handle it but is **not yet smoke-verified**.
   Verify on 1 node before ga=2 touches a 16n run.
3. **If #1 still spikes:** apply PRISM's production config **exactly** — `FSDP_NO_SYNC_ACCUM=1`
   + `gradient_accumulation_steps=2` (our `VJEPA_GRAD_ACCUM=2`), not a variant. Halves
   inter-node AllReduce frequency; this is the OLMo-3 7B E2E production setting, **not** a
   band-aid. Depends on #2 passing.
4. **Only if #1+#3 still spike:** *then* read free-L0 across the spike → creeps down ⇒
   static/persistent CCL buffer; flat ⇒ accepted congestion tax (`scaling_study.md:180`).

**Do NOT** swap the caching allocator, add socket-aware CPU-binding, restart-checkpoint, or
disable SDPA — all either ruled out (§4c) or research-grade changes for what is most likely a
two-line env deletion. **Do NOT** launch the 12h capacity job until #1 (and #2 if needed) come
back flat over 200+ iters.

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

## 4g. DEEPER LOOK (2026-07-04): the killer spikes are HOST-SIDE, not backward-collective

Re-examining job 8643398's per-phase CSV (not just `backward-ms`) reframes the residual. The
terminal-region iters, rank0, all columns in ms:

| itr | iter-ms | gpu-ms | Σphases | **UNTRACKED = iter−gpu** | bwd-ms |
|-----|---------|--------|---------|--------------------------|--------|
| 68  | 23070   | 23007  | 23037   | 63                       | 20639  |
| 69  | 85630   | 85515  | 85515   | 115                      | 82829  |
| 70  | **365317** | **21221** | 21234 | **344096**             | 18966  |
| 71  | 56818   | 56672  | 56672   | 146                      | 53941  |
| 72  | 25754   | 25711  | 25712   | 43                       | 22904  |

Two distinct spike modes are now visible:
- **In-GPU spikes** (68,69,71,72): iter-ms ≈ gpu-ms ≈ bwd-ms — the backward collective genuinely
  inflates (this is the "cohort backward inflation" §4c/§4e measured; real, still fabric).
- **Host-side stall** (iter 70): iter-ms=365 s but gpu-ms=21 s → **344 s is UNTRACKED wall time
  outside every GPU phase**, synchronized across all 192 ranks (rank0/50/100/150/191 all ≈365.5 s
  wall, ≈19 s backward). This is a Python/CCL **host-side** stall — a collective's host-side
  progress hanging on a congested fabric then clearing — NOT captured by `backward-ms` at all.

**The run recovered from the 365 s spike** (iters 71→73 back to ~25 s) then went fully silent
after iter 73 (720 s no-new-rows → watchdog kill), with **zero crash signatures** (0 tracebacks,
0 OOM, 0 shm errors). So the failure mode is **intermittent multi-minute host-side collective
stalls that mostly recover but occasionally hang past the watchdog** — a *stability/tail-latency*
problem, not the steady memory-starvation the DDP wedge was.

**Why this matters for the fix:**
1. It **strengthens the CCL_WORKER_COUNT=4 probe** (job 8643434): host-side collective progress
   is exactly what more CCL progress-engine worker threads accelerate. `WORKER_COUNT=1` (our
   value) means a single thread drives all collective host-side progress; PRISM uses 4. A
   starved progress thread is a textbook cause of multi-second host-side collective stalls.
2. It **re-weights grad-accum**: under FSDP `shard_grad_op`, only accumulation that reduces the
   *number of inter-node collectives per optimizer step* reduces stall opportunities. True
   loader-batch accumulation with `no_sync` on non-final microbatches does this (fewer
   ReduceScatters/AllGathers per step); our current microbatch-slicing does NOT. See §4f.
3. `backward-ms`-based analysis undercounts the problem — future runs must track `iter-ms` vs
   `gpu-ms` (the UNTRACKED delta) as a first-class metric. `analyze_straggler.py` reads iter-time
   already; add the untracked-delta column.

---

## 4e. ENV-DIFF VERDICT (2026-07-04, job 8643398): flags NOT the accumulator → fabric contention

The env-diff run (all three insurance flags **unset**, fixed-shape masks + top-level wrap,
ipe200) settled every branch of the §4c/§4d fork in a single job. It ran 74 iters, then a
terminal spike stalled >720 s and the watchdog killed it. Analyzed across all 192 CSVs
(`analyze_straggler.py`); backward figures below are **seconds**:

| metric | value | reading |
|--------|-------|---------|
| span classification | 24/74 iters COHORT (min<<p50≈p90≈max) | cohort-wide, as §4c |
| p50 backward trend | 10.3 s (i0-9) → 30.1 s (i70-79), peaks 57/68/83 s | escalating |
| **l0-free** | 15240 → 15235 MiB (**−5 MiB / 74 it**) | **FLAT** |
| **l0-ext** | 43695 → 43699 MiB (**+4 MiB / 74 it**) | **FLAT (noise)** |
| banned:1 / shm-crash | 0 | clean (prior crash was a transient node) |

**Three conclusions, each from data not inference:**
1. **The flag hypothesis (§4d) is FALSIFIED.** With all three flags removed the run still
   spikes cohort-wide and still escalates. The never-evict IPC cache / MR-monitor were not
   the accumulator. Removing them changed nothing — which is exactly what the env-diff was
   designed to reveal, and why running it first (before any lever) was correct.
2. **The static-buffer branch (§4c "yes" arm) is RULED OUT.** `l0-free` and `l0-ext` are dead
   flat (±5 MiB over 74 iters) with ~15 GB of headroom — there is **no external CCL/OFI
   registration growth**. `static_xccl_buffer_weight_sync` would treat an absent disease.
   (HSDP itself is confirmed working: 15 GB free vs the 90%-full DDP tile.)
3. **Per our pre-registered decision tree** (span=COHORT + l0-free FLAT + l0-ext FLAT):
   → **fabric contention** — PRISM's accepted congestion tax (`scaling_study.md:180`) →
   **`grad_accum` is the legitimate mitigation, not a band-aid.**

**One honest caveat.** A *monotonically escalating* p50 (10→30 s) is a slightly stronger
signal than pure stationary contention would give — it hints the fabric got busier over the
02:42→02:5x window (inter-job dragonfly load, which varies run-to-run). But it is definitively
**not** the memory-accumulation class, and the fix is selected by the memory reading, not the
timing shape. grad_accum (halving inter-node AllReduce frequency) is the correct lever whether
the contention is stationary or rising.

**Next (pre-registered §4d order of operations, step 2 then 3):**
- **Step 2 — 1n squeeze-fix smoke gate.** The `squeeze(1)` fix is present
  (`app/vjepa_2_1/models/utils/masks_dist.py:81`, documented against the bs≥2 `d_ij` landmine)
  and the ga>1 guard (`train.py:761`) blocks only `loss_reg_std_mult`, which our config does
  NOT set — so ga=2 is not hard-blocked. Verify ga=2 @ bs=2 → micro-bs=1 runs end-to-end on
  1 node before it touches 16n.
- **Step 3 — 16n with `VJEPA_GRAD_ACCUM=2` + `FSDP_NO_SYNC_ACCUM=1`** (PRISM's exact OLMo-3 7B
  production config). PASS = the cohort spikes flatten / iter-time stops escalating over ipe200.

> **⚠️ Steps 2–3 above are SUPERSEDED by §4f. The grad_accum lever as implemented does NOT
> reduce inter-node AllReduce frequency — see below before acting on it.**

---

## 4f. FIFTH-REVIEW CORRECTION (2026-07-04): `VJEPA_GRAD_ACCUM=2` is microbatching, not accumulation

A fifth review challenged the §4e/§4d claim that `VJEPA_GRAD_ACCUM=2` "halves inter-node
AllReduce frequency." **Validated against the code — the claim is FALSE as implemented, and the
1n smoke did not prove otherwise.** Job 8643428 (the 16n grad_accum run) was **killed while
still queued** — no compute wasted — because it tested a lever that by construction cannot
address fabric contention.

**What the code actually does** (`app/vjepa_2_1/train.py:785-828`, `:1021-1040`):
- The training loop does exactly **one `next(loader)` per `itr`** (`:792`).
- The ga>1 path **slices that single loader batch** into `grad_accum` microbatches
  (`mb = batch_size // grad_accum`; `sl = slice(j*mb, (j+1)*mb)`), runs `no_sync()` on all but
  the last, and syncs once.
- Under HSDP the inter-node collective is one reduce-scatter/all-reduce over the **model
  gradient**, whose size and per-optimizer-step count are **independent of batch size**. Slicing
  bs=2 into 2×bs=1 leaves collective size, count, and frequency per optimizer step **identical
  to the ga=1 baseline**. Effective batch stays 2.

So this is **microbatching** (lower activation peak, collectives spread slightly in wall-clock),
**not PRISM-style accumulation across multiple loader batches**. To actually halve inter-node AR
frequency per sample you must accumulate over N separate `next(loader)` batches with one sync —
which changes effective global batch (a recipe change), not a drop-in env flip. The doc's
"PRISM exact production config" equivalence was wrong: PRISM accumulates loader batches; our env
slices one.

**Two more confirmed defects this surfaced:**
1. **`backward-ms` is polluted under ga>1** (`:1029-1044`): `fwd_context_done` is marked at
   microbatch 0 but `backward_done` after the whole loop, so `backward-ms` includes
   microbatch-1's *forward*. It is **not comparable** to the env-diff's clean ga=1 `backward-ms`
   — the very metric the run was meant to compare. Any ga>1 analysis must switch to
   `iter-time(ms)`/`gpu-time(ms)` cross-rank spread, or add per-microbatch phase timers.
2. **`no_sync()` memory was never gated.** FSDP `no_sync()` retains the unsharded/full gradient
   until the synced microbatch, which can erode the HSDP headroom that fixed the original DDP
   problem. The 1n smoke gated only on "no IndexError / no NaN" — it captured **no
   mem/resv/l0free/l0ext**. Unmeasured risk.

**The diagnosis (§4e) still stands** — the env-diff run 8643398 was ga=1, so its `backward-ms`
was clean and the flat-memory → fabric-contention verdict is unaffected. Only the *proposed
mitigation* was wrong.

**Corrected lever menu for fabric contention (pick after deciding intent):**
- **True grad accumulation** — implement accumulation across multiple `next(loader)` batches
  (accept larger effective global batch or adjust the LR/schedule). Only this actually reduces
  inter-node AR frequency per sample.
- **CCL AllReduce algorithm / chunking A/B** (reviewer point 5) — the current scripts force
  `CCL_ALLREDUCE=ring` + `CCL_CHUNK_SIZE=16MiB` (chosen for the *DDP* workload, ~4% over
  topo/rabenseifner). Under HSDP + fixed-shape + top-level wrap with a spike/floor failure mode,
  the AR algorithm and chunk size are **first-class variables**. Low-cost A/B: fixed-shape +
  top-level HSDP, vary `CCL_ALLREDUCE`/`CCL_CHUNK_SIZE` (ring/chunked vs default vs PRISM's exact
  env block), metric = p50/p90 backward (or iter-time) cross-rank trend, not mean.
- **Accept the tax** — if AR latency is genuinely small (PRISM `scaling_study.md:180` measured
  24.7 ms/step) and the escalation is inter-job dragonfly load, occasional recoverable cohort
  spikes may just be the 16n reality.

**Pre-capacity config-drift fixes (reviewer points 2–4, all confirmed; capacity not yet
launched):** `scripts/vitG384_capacity.sh` currently (a) uses `vitG384_cleandata.yaml` (variable
masks — reintroduces churn source #2; `num_keep_*` live only in `vitG384_fixedshape.yaml:197-213`),
(b) keeps all three falsified insurance flags (`:103-105`), (c) has no grad-accum. Must be brought
to the measured env-diff baseline (fixed-shape config, flags unset) **before** any long launch.

---

## 5. Current state of the tree (branch `aurora`)
- HSDP code (`hsdp.py`, trainer branch, EMA test) committed & 1n-verified.
- OFI transport applied to `scripts/vitG384_hsdp_spike_16n.sh` and
  `scripts/vitG384_capacity.sh` (production 12h launcher, converted DDP→HSDP+OFI, pt213 venv).
- Grad-accum (`VJEPA_GRAD_ACCUM`) + `d_weights` `squeeze(1)` bs≥2 fix committed (secondary lever).
- No long run launched — blocked on resolving the §4 drift.

## 6. Current open question (updated 2026-07-04)

Two structural churn sources are **fixed & verified** (top-level FSDP wrap; fixed-shape
masks — the dominant one). They converted a fatal monotonic runaway into a run that survives
73+ iters with a healthy ~7–11 s baseline. The **remaining** residual is a *cohort-wide,
escalating, backward-collective inflation with a creeping floor* (§4c) — NOT a one-rank
straggler and NOT the PyTorch pool (both ruled out by data).

**The single decisive run (§4d): the env-diff test.** Before the free-L0 fork can even be
read cleanly, remove **all three** "insurance" flags we added ourselves
(`CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD`, `FI_MR_CACHE_MONITOR`, `PYTORCH_ALLOC_CONF`) —
PRISM's production launcher is grep-clean of all three and runs HSDP at 20N stably. Dropping
all three (not just the two cache flags) makes a flat result unambiguous in one job. Keep
fixed-shape masks + top-level wrap, change nothing else, ipe200. The run carries the free-L0
probe, so it *also* answers:
- **floor/drift flatten** → the flags were the accumulator (predicted) → **close out, launch-capable**.
- **still spikes, free-L0 creeps down** → real external CCL/OFI accumulation → static/persistent
  collective buffer (torchtune `static_xccl_buffer_weight_sync.md`).
- **still spikes, free-L0 flat** → fabric contention, accepted tax (`scaling_study.md:180`) →
  apply PRISM's exact production config `FSDP_NO_SYNC_ACCUM=1` + `gradient_accumulation_steps=2`,
  **after** the 1n `squeeze(1)` smoke gate (ga=2 → micro-bs=1 trips the `weight_distance_loss`
  bs≥2 landmine). Sequenced as 3 jobs = full attribution + verified fix; combined = 1 ambiguous job.

Run to execute: `scripts/vitG384_hsdp_fixedshape_16n.sh` (all three flags now unset; emits
`l0-free-mib`/`l0-ext-mib` per rank per iter), analyze with `scripts/analyze_straggler.py <run_folder>`.

**RESOLVED (2026-07-04, job 8643398 — see §4e).** l0-free and l0-ext are both dead flat
(±5 MiB / 74 iters); the flags were NOT the accumulator and there is NO external-memory
accumulation. Verdict = **fabric contention**.

**CORRECTED (2026-07-04 — see §4f).** The proposed `VJEPA_GRAD_ACCUM=2` lever is
**microbatching, not accumulation across loader batches** — it does NOT reduce inter-node
AllReduce frequency per optimizer step (verified in `train.py:785-828,1021-1040`), so it cannot
address fabric contention. The 16n grad_accum job (8643428) was killed while queued. Real levers
now: (a) implement TRUE grad accumulation across loader batches, or (b) A/B the CCL AllReduce
algorithm/chunk size, or (c) accept the tax. Also: `backward-ms` is polluted under ga>1, and the
capacity launcher still has config drift (variable-mask config + falsified flags) to fix first.

Full blow-by-blow: memory `vitG-2b-allreduce-spikes.md`. Key repro scripts:
`scripts/vitG384_hsdp_fixedshape_16n.sh` (16n gate + free-L0 probe),
`scripts/analyze_straggler.py` (cross-rank distribution + free-L0 trend),
`scripts/vitG384_hsdp_2n_ofi.sh` (2n clean repro), `scripts/collective_probe.py`.
