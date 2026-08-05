<!-- Copied into vjepa2 from torchtune (validated there 2026-07-09). Kept in sync manually. -->

## ►► STATUS (2026-07-09): implemented, HW-gate pending

Steps 1–2 landed in `app/vjepa_2_1/models/utils/modules.py`, default OFF:
- Flag `VJEPA_USE_XPU_FLASH=1` (env). Off → `_sdpa()` is bit-for-bit the old path.
- `_to_bshd_memory` + `_xpu_flash_sdpa` force `SDPBackend.FLASH_ATTENTION` with BSHD-coerced q/k/v;
  one-shot `[vjepa2] xpu_flash=engaged` log line.
- Eligibility gates on `head_dim ∈ {64,96,128,192,256}` (NOT the guide's loose `≤256`): the
  predictor runs head_dim=32, so a loose gate would raise `No available kernel` there. All encoder
  variants (large/giant_xformers/gigantic_xformers) are head_dim=64 → eligible; predictor falls to
  math. Win is encoder-only (fine: gigantic depth=48 dominates compute).
- `tests/models/test_sdpa_layout.py` extended with 2 flash-coerce tests; 4/4 pass on CPU.

**HW-validated 2026-07-09 (job 8659829, node x4401c0s3b0n0, torch 2.10.0a0):**
1. Correctness gate PASSED (`scripts/gate_flash_sdpa_xpu.py`, ViT-gigantic shape [1,26,4608,64]):
   bf16 flash-vs-math cos=0.999988 max|Δ|=1.95e-3 (<0.03); fp16 cos=1.000000. Scrambling ruled
   out. `diag_sdpa_xpu.py` can't test flash with the flag on (its force_math leg nests inside the
   forced-FLASH context) — use `gate_flash_sdpa_xpu.py` instead.
2. Same-node A/B on SMOKE_vitG384 (12 tiles, bs=2, 384px, **use_activation_checkpointing: true**):
   flash engaged 12/12 ranks. **peak mem IDENTICAL 57.6 GiB off vs on; iter-time within ±2% at
   matched iters (meter is a cumulative AverageMeter — compare same iter#).** The `gpu:` phase
   timer reads garbage (negative) on the fused kernel — use total `iter:` only.

**Result: under activation checkpointing, flash gives NO memory win and NO step-time win here.**
Expected: act-ckpt already frees the S² score tensor across backward, so flash's headline (killing
S² materialization) is redundant. The real lever is **act-ckpt OFF + flash** — flash may let you
drop checkpointing (saving its recompute cost → throughput) where math would OOM at bs=2/384px on
a 64 GiB tile. NOT YET TESTED. Also untested: whether flash raises the max per-rank batch.

Gotchas found wiring the A/B (not flash bugs): (a) `set -u` aborts at `module load frameworks`
(lmod refs unbound ZSH_EVAL_CONTEXT); (b) `--train_mode` ignores `--folder` and reads `folder:`
from YAML, and SMOKE has `load_checkpoint: true` — two legs sharing the folder makes leg 2 resume
leg 1's checkpoint and run 0 iters. Give each leg a fresh folder + `load_checkpoint: false`.

**Remaining:** downstream metric-parity check before defaulting the flag on for a real run.

## ►► THROUGHPUT MATRIX (job 8659973, node x4500c1s0b0n0, HSDP _HYBRID_SHARD_ZERO2, SMOKE_vitG384 12 tiles, same-node) — 2026-07-09

| leg | bs | peak mem (torch) | iter ms | clips/s/tile | vs baseline |
|---|---|---|---|---|---|
| L1 ckpt-ON  flash-OFF (production) | 2 | 22.0 GB | 7022 | 0.285 | — |
| L2 ckpt-OFF flash-OFF | 2 | 46.2 GB | 6440 | 0.311 | **+9.0%** |
| L3 ckpt-OFF flash-ON  | 2 | 43.2 GB | 6410 | 0.312 | **+9.5%** |
| L4 ckpt-OFF flash-ON  | 3 | 52.1 GB → **OOM** | — | — | doesn't fit |

**Actionable win: `use_activation_checkpointing: false` on the HSDP configs = ~9% throughput,
bit-identical output, zero recipe risk.** Removes the recompute-forward tax on the backward phase
(49% of step). Only affordable because HSDP shards optimizer state to 22 GB baseline — under DDP
(57.6 GB) ckpt-off would be ~82 GB and OOM.

**Flash is a MEMORY lever here, not a speed one:** with ckpt-off the S² tensor is live again, so
flash frees 3 GB (46.2→43.2) but adds only −0.5% time. That margin is NOT enough for bs=3.

**bs=3 OOMs** (UR_RESULT_ERROR_OUT_OF_RESOURCES at FSDP `_mp_shard.copy_`). CRITICAL: torch `mem:`
undercounts true L0 peak by ~12 GB (FSDP bf16 all-gather transient buffers) — bs=2 43.2 GB torch ≈
54 GB real; bs=3 52 GB torch → >64 GB → OOM. bs=2 is the ceiling. Don't size headroom off `mem:`.

**fp32 vs bf16 does NOT affect throughput** (asked 2026-07-09): from live 16n phase timing,
opt-step = 55.7 ms of 10425 ms iter = 0.5%. Compute is already bf16 (FSDP MixedPrecision
param_dtype=bf16); fp32 is only the optimizer master, read once/step, sharded to ~1 GB/tile under
ZeRO-2. Pure-bf16 optimizer = recipe risk (JEPA collapse), ~0 speed gain. Keep fp32 master.

## ►► vjepa2-specific start here

The live Aurora path is `app/vjepa_2_1/` (NOT `src/` — that's the original Meta tree, which still
passes `attn_mask`). All attention flows through ONE helper: `_sdpa()` in
`app/vjepa_2_1/models/utils/modules.py:17-34`, called by RoPEAttention / Attention /
CrossAttention. It is **already mask-free**, bidirectional (`is_causal=False`), bf16,
head_dim=64, dropout=0 — every precondition is met.

Change: replace the body of `_sdpa()` with a call to `xpu_flash_sdpa(...)` (defined below), gated
behind a flag. Apply the BSHD coerce AFTER the RoPE `torch.cat` (`modules.py:302-306`), which
produces non-standard strides. Transpose the output back to `[B,H,S,D]` so the downstream
`x.transpose(1,2).reshape(B,N,C)` is unchanged.

**Keep `tests/models/test_sdpa_layout.py` green.** That test asserts a *naive* BSHD transpose
scrambles output (it attends the wrong axis). The coerce here is different: it changes only the
memory *stride*, not which axis is attended (transpose→contiguous→transpose-back is numerically
identical to the BHSD reference), so the test must still pass. Extend it to cover the flash path.
Reuse `scripts/diag_sdpa_xpu.py` for parity.

The generic guide follows.

---

# Porting guide: native XPU fused flash-attention (Intel Max GPU / Aurora PVC)

**Audience:** agents working in any repo that runs attention on Aurora XPU (PRISM, vjepa2,
other torch models). Self-contained — you do NOT need torchtune context. Validated 2026-07-09
(torchtune BioReason SFT: **~2× step time**, **~8× attention memory**, bs>1 unblocked where math
OOMs). Full background: `xpu_flash_attention_gate0_result_20260709.md`,
`xpu_flash_attention_applicability_map_20260709.md`.

## TL;DR — what this is

`frameworks/2025.3.1` ships a native SYCL-TLA fused flash-attention kernel (fwd **and**
backward) at
`.../torch/lib/libtorch-xpu-ops-sycltla-mha_{fwd,bwd}.so`. It is ~2–8× more memory-efficient
than the XPU "math" SDPA backend (which materializes the `[B,H,S,S]` fp32 score tensor). It is
**already present** — no build, no Triton, no compile. But two things stop `F.scaled_dot_product_attention`
from using it, and you must fix both at the call site.

## The two things you must do

1. **Force the FLASH backend** (required on torch ≤ 2.10; keep it on all versions — see below).
   Wrap the SDPA call in `torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION])`.
2. **Put q/k/v in BSHD memory layout.** The kernel wants shape `[B,H,S,D]` whose *storage* is a
   contiguous `[B,S,H,D]` transposed on dims 1↔2. Plain C-contiguous `[B,H,S,D]` (e.g. from a
   `permute` of a fused-qkv tensor) is **rejected with `RuntimeError: No available kernel`**.
   Coerce each of q/k/v (see helper below).

### torch-version dependence of the auto-dispatch (verified from tagged PyTorch source)

The XPU SDPA backend priority array in `aten/src/ATen/native/mkldnn/xpu/Attention.cpp` changed:

| torch | XPU SDPA priority order | auto picks flash? |
|---|---|---|
| **2.10** (Aurora `module load frameworks`, 2.10.0a0) | `overrideable, **math**, flash, ...` | **NO** — math is ahead of flash, so unforced SDPA materializes the O(S²) tensor (empirically ~8 GiB) |
| **2.11 / 2.12 / 2.13** | `overrideable, **flash**, math, ...` | YES — flash is tried before math on eligible inputs |

**The flip is at torch 2.11, not 2.13** (2.13 is real — released 2026-07-08 — but not where the
default changed). PRs: #156669 / #159464 / #167057 (SYCL-TLA kernels, Dec 2025) / #170414 (BHSD,
Jan 2026).

**Keep forcing on ALL versions:** on 2.10 it's required (math ahead); on 2.11+ `overrideable`
(oneDNN) still sits *ahead* of flash, so an unforced call could route to oneDNN instead of the
validated SYCL-TLA kernel. Forcing is a no-op when flash is already default and surfaces a
`No available kernel` error instead of a silent 8 GiB math fallback on an unsupported shape.

## Preconditions (ALL must hold, else it falls back to math — that's fine, just no speedup)

| requirement | why |
|---|---|
| device is **XPU** | kernel is XPU-only |
| dtype **bf16 or fp16** | no fp32 flash kernel |
| **head_dim ≤ 256**, q/k/v same last dim | compiled kernel coverage {64,96,128,192,256} |
| **`attn_mask=None`** | ANY boolean/additive mask → math. Drop the mask if it's just a plain causal or all-ones; keep math if you truly need arbitrary masking. |
| **`dropout_p == 0`** | fused kernels reject dropout>0 |
| **causal OR bidirectional** both OK | if `is_causal=True`, requires `seqlen_q == seqlen_k` (breaks incremental decode; fine for training/prefill) |

Note: **bidirectional (encoder/ViT) attention qualifies** — you do NOT need a causal model.

## Drop-in helper (copy this)

```python
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

def _to_bshd_memory(t: torch.Tensor) -> torch.Tensor:
    """Coerce [B, H, S, D] -> BSHD memory (shape unchanged; storage = contiguous
    [B,S,H,D] transposed 1<->2). No-op if already in that layout."""
    if t.transpose(1, 2).is_contiguous():
        return t
    return t.transpose(1, 2).contiguous().transpose(1, 2)

def xpu_flash_sdpa(q, k, v, *, is_causal, dropout_p=0.0):
    """q,k,v: [B, H, S, D] (GQA already expanded). Returns [B, H, S, D].
    Uses the native XPU fused flash kernel when eligible; else falls back to SDPA
    (which on XPU is the math backend)."""
    eligible = (
        q.device.type == "xpu"
        and q.dtype in (torch.bfloat16, torch.float16)
        and q.shape[-1] <= 256
        and dropout_p == 0.0
    )
    if not eligible:
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=is_causal
        )
    q, k, v = _to_bshd_memory(q), _to_bshd_memory(k), _to_bshd_memory(v)
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=dropout_p, is_causal=is_causal
        )
```

**Gate it behind a flag** (e.g. `USE_XPU_FLASH=1`) defaulting OFF, so you can A/B and revert
instantly. Log once whether it engaged (see "verify" below) — silent fallback to math is the
#1 way to think it's on when it isn't.

## Where to wire it, by repo

- **PRISM** (`BaseMM_PRISM`): the HF LM backbone already forces `is_causal=True, attn_mask=None`
  on XPU (`src/model.py:943-969`). Simplest: wrap `self.backbone(...)` at `src/model.py:963` in
  `with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):`. HF attention modules usually already emit
  BSHD-strided q/k/v (`view(B,S,H,D).transpose(1,2)`), so the coerce may be unnecessary — but
  **verify** (if you see "No available kernel", a backbone emits C-contiguous and needs the
  helper's coerce via a monkeypatch of its attn, or `attn_implementation` shim).
- **vjepa2** (`app/vjepa_2_1/models/utils/modules.py`): replace the body of `_sdpa()` (L17-34,
  the single helper all 3 attention classes call) with `xpu_flash_sdpa(...)`. It's bidirectional
  (`is_causal=False`), already mask-free, bf16, head_dim=64. Coerce after the RoPE `cat`
  (which makes non-standard strides). Keep `tests/models/test_sdpa_layout.py` green: that test
  proves a *naive* BSHD transpose scrambles output — the coerce here preserves the attended axis
  (transpose→contiguous→transpose-back is numerically identical to BHSD), so it must still pass.
- **Any model using `F.scaled_dot_product_attention` directly:** swap the call for
  `xpu_flash_sdpa(...)` and drop the mask arg if it was a plain causal/None.

## Verify it actually engaged (do NOT skip)

**1. Memory + numerics micro-test** (single tile, ~30s). Adjust B/H/S/D to your model:

```python
# ZE_AFFINITY_MASK=0 python this.py   (on a compute node; XPU not on login nodes)
import torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
import torch.nn.attention as tna; tna.WARN_FOR_UNFUSED_KERNELS = True  # prints reject reason
dev="xpu"; B,H,S,D=1,32,4096,128           # <- your shape
def mk(): 
    x=torch.randn(B,S,H,D,dtype=torch.bfloat16,device=dev,requires_grad=True)
    return x.transpose(1,2)                  # BSHD memory (what the helper produces)
q,k,v=mk(),mk(),mk()
torch.xpu.reset_peak_memory_stats()
with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
    o=F.scaled_dot_product_attention(q,k,v,attn_mask=None,is_causal=True)
o.float().pow(2).mean().backward()
print(f"peak {torch.xpu.max_memory_allocated()/1024**3:.2f} GiB (fused if << S^2)")
# math S^2 ref would be ~ B*H*S*S*4/1e9 GiB per call; fused should be a few hundred MiB-ish
```
If it raises `No available kernel` → a precondition failed (most likely layout: you passed
C-contiguous `[B,H,S,D]`; use `_to_bshd_memory`). The `WARN_FOR_UNFUSED_KERNELS` print names the
exact failed check (e.g. "requires ... BSHD layout", "does not support non-null attn_mask").

**2. In your training loop:** log once at the first attention call and grep for it. Compare peak
memory and step time flag-off vs flag-on on the SAME node (Aurora has ~1.8× node-to-node
variance — only same-node back-to-back A/Bs are valid). Confirm loss is bf16-close to the math
baseline (expect fwd `max|Δ|` under ~0.03).

## Landmines

- **Silent fallback to math.** If any precondition fails, `F.sdpa` just runs math — no error, no
  speedup. Always verify engagement with a log line + a memory delta, never assume.
- **`attn_mask` disqualifies it.** If your model passes a padding/causal *tensor* mask, flash
  won't run. Options: (a) drop it if it's a plain causal (use `is_causal=True` instead), (b) for
  right-padded batches causal attention is safe mask-free, (c) if you need arbitrary masking,
  this kernel can't help — keep math or use a masked flash variant.
- **Don't call `torch.xpu.empty_cache()` in an FSDP training loop** — it leaks Level-Zero UR
  handles on Aurora (unrelated to flash, but you'll hit it when chasing memory). 
- **Numerics are bf16-close, not bit-exact** vs math. Validate downstream metric parity (loss,
  eval score) before defaulting the flag on.
- **HW-verify per target; do not assume the torchtune 2× transfers.** Comm-bound workloads
  (e.g. large-model multi-node where collectives dominate the step) see a smaller step-time win
  even though the memory win is the same.

## One-line summary to paste into a repo's issue

> Aurora ships a native XPU fused flash-attention (fwd+bwd, `libtorch-xpu-ops-sycltla-mha_*.so`,
> bf16/fp16, head_dim≤256, causal+bidir). To use it: at each `F.scaled_dot_product_attention`
> call with `attn_mask=None, dropout=0`, wrap in `sdpa_kernel([SDPBackend.FLASH_ATTENTION])` and
> coerce q/k/v to BSHD memory. ~2–8× vs the math backend; auto-dispatch picks math on torch 2.10
> (flash-before-math starts at 2.11, but oneDNN still sits ahead) so force it on all versions.
> Gate behind a flag, verify engagement (WARN_FOR_UNFUSED_KERNELS) + memory delta.
