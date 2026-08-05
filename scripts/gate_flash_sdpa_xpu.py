"""Correctness gate for the Aurora native fused flash SDPA path.

Directly A/Bs app.vjepa_2_1.models.utils.modules._sdpa with the flash kernel
FORCED (VJEPA_USE_XPU_FLASH=1 -> _xpu_flash_sdpa) against the math/default path,
at the encoder's real attention shape. Unlike diag_sdpa_xpu.py this does NOT nest
sdpa_kernel contexts, so the math leg is a clean reference.

Run on a held Aurora tile:
    ZE_AFFINITY_MASK=0 VJEPA_USE_XPU_FLASH=1 python scripts/gate_flash_sdpa_xpu.py

Go/no-go:
  fp32 flash-vs-math cos > 0.999   (algorithmic parity; catches layout scramble)
  bf16 flash-vs-math fwd max|Δ| < 0.03  (bf16 numerics floor)
Exit code 0 = PASS, 1 = FAIL.
"""
import os
import sys

os.environ.setdefault("ZE_AFFINITY_MASK", "0")
# Force flag ON for this process regardless of caller (the whole point of the gate).
os.environ["VJEPA_USE_XPU_FLASH"] = "1"

REPO = "/lus/flare/projects/ModCon/ngetty/vjepa2"
sys.path.insert(0, REPO)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import app.vjepa_2_1.models.utils.modules as mod  # noqa: E402


def _math_sdpa(q, k, v):
    """Explicit reference attention over the sequence axis (dim -2) of BHND."""
    scale = q.shape[-1] ** -0.5
    attn = (q @ k.transpose(-2, -1) * scale).softmax(dim=-1)
    return attn @ v


def compare(tag, a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    cos = F.cosine_similarity(a, b, dim=0).item()
    max_abs = (a - b).abs().max().item()
    rel = ((a - b).norm() / (a.norm() + 1e-9)).item()
    print(f"  {tag:24s} cos={cos:.6f}  max|Δ|={max_abs:.4e}  relL2={rel:.4e}")
    return cos, max_abs


def run(dtype):
    # ViT-gigantic encoder attention shape at 384px/16f: head_dim=64, 26 heads.
    # N = T/2 tubelets * (384/16)^2 = 8 * 576 = 4608 tokens. B=1 tile.
    B, H, N, D = 1, 26, 4608, 64
    g = torch.Generator().manual_seed(0)
    q = torch.randn(B, H, N, D, generator=g)
    k = torch.randn(B, H, N, D, generator=g)
    v = torch.randn(B, H, N, D, generator=g)
    dev = "xpu"
    q, k, v = q.to(dev, dtype), k.to(dev, dtype), v.to(dev, dtype)

    assert mod._xpu_flash_eligible(q, 0.0, False), (
        f"inputs NOT flash-eligible (dtype={dtype}, head_dim={D}); gate can't test flash"
    )
    flash = mod._sdpa(q, k, v)          # flag ON + eligible -> forced flash
    math = _math_sdpa(q, k, v)          # clean reference
    return flash, math


def main():
    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "no XPU visible"
    print(f"torch={torch.__version__}  VJEPA_USE_XPU_FLASH={mod._USE_XPU_FLASH}  "
          f"imports_ok={mod._XPU_FLASH_IMPORTS_OK}  ZE_AFFINITY_MASK="
          f"{os.environ.get('ZE_AFFINITY_MASK')}")
    print(f"head_dim gate set: {mod._XPU_FLASH_HEAD_DIMS}\n")

    ok = True

    # No fp32 flash kernel exists (eligibility is bf16/fp16 only), so parity is
    # checked in the training dtype (bf16) plus fp16 as a second-precision
    # cross-check. The fp32 value-preservation of the BSHD coerce is covered by
    # tests/models/test_sdpa_layout.py on CPU.
    print("=== bf16 (training dtype) ===")
    bf_flash, bf_math = run(torch.bfloat16)
    cos_bf, bf_maxabs = compare("flash vs math", bf_flash, bf_math)
    if not (cos_bf > 0.999 and bf_maxabs < 0.03):
        ok = False

    print("\n=== fp16 (algorithmic cross-check) ===")
    h_flash, h_math = run(torch.float16)
    cos_h, max_h = compare("flash vs math", h_flash, h_math)
    if not (cos_h > 0.999):
        ok = False

    print("\n=== VERDICT ===")
    if ok:
        print("  PASS — flash SDPA is numerically parity with math "
              "(bf16 cos>0.999, max|Δ|<0.03).")
    else:
        print("  FAIL — flash SDPA diverges from math beyond bf16 floor. "
              "Do NOT enable VJEPA_USE_XPU_FLASH for real runs.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
