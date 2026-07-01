"""Diagnostic: is XPU FlashAttention (use_sdpa=True) numerically wrong for the
V-JEPA 2.1 ViT-L encoder, and is it a precision (bf16) or algorithmic bug?

Runs ONE fixed input through the SAME encoder weights under several attention
conditions and compares the final encoder output features. No training, no DDP,
single tile. Intended to be run on a held Aurora node:

    ZE_AFFINITY_MASK=0 python scripts/diag_sdpa_xpu.py \
        --checkpoint /flare/ModCon/ngetty/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt

Conditions (all on identical weights + identical input):
  math      : use_sdpa=False  -> explicit (q@k^T) softmax @ v   [reference]
  sdpa      : use_sdpa=True    -> _sdpa() -> FlashAttentionXPU   [suspect]
  sdpa_math : use_sdpa=True    -> _sdpa() forced to SDPBackend.MATH

We run the whole comparison in fp32 and in bf16 (the training dtype) so we can
tell precision from algorithm:
  - math(fp32) vs sdpa(fp32)      : if they diverge in fp32, it's algorithmic.
  - math(bf16) vs math(fp32)      : pure precision floor of the model.
  - sdpa(bf16) vs sdpa_math(bf16) : is FlashAttentionXPU worse than SDPA-math at
                                     the same precision?
"""
import argparse
import os
import sys

# Pin one tile before importing torch (match the probe's per-rank pinning).
os.environ.setdefault("ZE_AFFINITY_MASK", os.environ.get("ZE_AFFINITY_MASK", "0"))

REPO = "/lus/flare/projects/ModCon/ngetty/vjepa2"
sys.path.insert(0, REPO)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import app.vjepa_2_1.models.vision_transformer as vit  # noqa: E402
import app.vjepa_2_1.models.utils.modules as mod  # noqa: E402


def build_encoder(use_sdpa, device, checkpoint_path):
    enc_kwargs = dict(
        model_name="vit_large",
        patch_size=16,
        tubelet_size=2,
        img_temporal_dim_size=1,
        uniform_power=True,
        use_rope=True,
        use_sdpa=use_sdpa,
    )
    model = vit.__dict__["vit_large"](img_size=256, num_frames=16, **enc_kwargs)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    key = "target_encoder" if "target_encoder" in ckpt else "encoder"
    sd = ckpt[key]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
    for k, v in model.state_dict().items():
        if k in sd and sd[k].shape != v.shape:
            sd[k] = v  # keep model's tensor on shape mismatch (matches probe)
    msg = model.load_state_dict(sd, strict=False)
    missing = [k for k in msg.missing_keys if "pos_embed" not in k]
    if missing:
        print(f"  [warn] missing keys (non-posembed): {missing[:6]}"
              f"{' ...' if len(missing) > 6 else ''}")
    del ckpt
    # Keep fp32 weights; precision comes from autocast, exactly as the probe
    # (evals/.../eval.py wraps the encoder forward in torch.amp.autocast) and
    # the trainer do. Hard-casting weights to bf16 desyncs the RoPE path (q/k
    # rotated in fp32, v left bf16) and SDPA rejects mismatched dtypes.
    return model.to(device=device, dtype=torch.float32).eval()


def run_forward(model, x, dtype, force_math=False):
    autocast = torch.amp.autocast(device_type="xpu", dtype=dtype,
                                  enabled=(dtype == torch.bfloat16))
    with torch.no_grad(), autocast:
        if force_math:
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with sdpa_kernel([SDPBackend.MATH]):
                out = model(x)
        else:
            out = model(x)
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out.float()


def compare(name, a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    cos = F.cosine_similarity(a, b, dim=0).item()
    max_abs = (a - b).abs().max().item()
    rel = ((a - b).norm() / (a.norm() + 1e-9)).item()
    print(f"  {name:28s} cos={cos:.6f}  max|Δ|={max_abs:.4e}  relL2={rel:.4e}")
    return cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="/flare/ModCon/ngetty/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt")
    args = ap.parse_args()

    assert hasattr(torch, "xpu") and torch.xpu.is_available(), "no XPU visible"
    device = "xpu"
    print(f"torch={torch.__version__}  _IS_XPU={mod._IS_XPU}  "
          f"ZE_AFFINITY_MASK={os.environ.get('ZE_AFFINITY_MASK')}")
    print(f"checkpoint={args.checkpoint}\n")

    # One fixed fp32 input, reused for every condition (autocast handles bf16).
    g = torch.Generator().manual_seed(0)
    x = torch.randn(1, 3, 16, 256, 256, generator=g).to(device=device)

    results = {}
    # Build each encoder once; run it under both precisions.
    m_math = build_encoder(use_sdpa=False, device=device,
                           checkpoint_path=args.checkpoint)
    m_sdpa = build_encoder(use_sdpa=True, device=device,
                           checkpoint_path=args.checkpoint)
    for dtype in (torch.float32, torch.bfloat16):
        tag = "fp32" if dtype == torch.float32 else "bf16"
        results[f"math_{tag}"] = run_forward(m_math, x, dtype)
        results[f"sdpa_{tag}"] = run_forward(m_sdpa, x, dtype)
        results[f"sdpamath_{tag}"] = run_forward(m_sdpa, x, dtype, force_math=True)
    del m_math, m_sdpa; torch.xpu.empty_cache()

    print("\n=== ALGORITHMIC (compare at SAME precision) ===")
    print("fp32:")
    compare("math vs sdpa", results["math_fp32"], results["sdpa_fp32"])
    compare("math vs sdpa(forced-math)", results["math_fp32"], results["sdpamath_fp32"])
    compare("sdpa vs sdpa(forced-math)", results["sdpa_fp32"], results["sdpamath_fp32"])
    print("bf16:")
    compare("math vs sdpa", results["math_bf16"], results["sdpa_bf16"])
    compare("math vs sdpa(forced-math)", results["math_bf16"], results["sdpamath_bf16"])
    compare("sdpa vs sdpa(forced-math)", results["sdpa_bf16"], results["sdpamath_bf16"])

    print("\n=== PRECISION (compare bf16 vs fp32 within a path) ===")
    compare("math:   bf16 vs fp32", results["math_bf16"], results["math_fp32"])
    compare("sdpa:   bf16 vs fp32", results["sdpa_bf16"], results["sdpa_fp32"])

    print("\n=== INTERPRETATION ===")
    cos_alg_fp32 = F.cosine_similarity(
        results["math_fp32"].flatten(), results["sdpa_fp32"].flatten(), dim=0).item()
    if cos_alg_fp32 < 0.999:
        print("  -> FlashAttentionXPU diverges from math in fp32: ALGORITHMIC bug "
              "in the XPU SDPA kernel (not just bf16 rounding).")
    else:
        print("  -> SDPA matches math in fp32; any bf16 gap is precision, not algorithm.")


if __name__ == "__main__":
    main()
