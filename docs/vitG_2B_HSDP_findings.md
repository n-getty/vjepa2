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

## 4. OPEN PROBLEM: iter-time drift at 16n (needs the real workaround)

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
