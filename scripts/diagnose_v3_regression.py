#!/usr/bin/env python3
"""Diagnose the MONOTONIC downstream regression of v3 surgical CPT.

Downstream (asformer action-seg) regresses each epoch (full-data cached probe:
e9 +2.2 -> e14 +1.4 -> e19 +0.3 vs Meta) while the pretraining loss PLATEAUS at
e5. So the encoder keeps changing without the pretext improving, and those
changes erode action-relevant features. This measures WHAT degrades, across the
checkpoint trajectory [Meta, e4, e9, e14, e19], on surgical vs kinetics clips.

Metrics per checkpoint (computed on the SAME fixed clip sample):
  A. TOKEN-level (what the asformer head actually consumes -- per-frame tokens):
     - token effective rank (collapse: rank falls = fewer useful dims)
     - token anisotropy: mean pairwise cosine of tokens within a clip
       (HIGH = tokens becoming identical = temporal/spatial detail lost ->
        exactly what kills action segmentation)
     - per-frame token variance (how much tokens vary across time in a clip;
       LOW = temporal dynamics collapsed)
  B. CLIP-level (mean-pooled):
     - embedding effective rank (global collapse)
     - cosine drift from Meta (how far the encoder moved)
  C. Split by domain: surgical (surgvu24, sitl, ...) vs kinetics400, so we see
     if collapse is SPECIFIC to the static surgical domain (degenerate-pretext
     hypothesis) or global (EMA co-drift hypothesis).

Run on a held node (single process fine; ~few hundred clips fp32 inference).
  python scripts/diagnose_v3_regression.py --n 24 --out /flare/.../diag.json
"""
import argparse
import json
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/lus/flare/projects/ModCon/ngetty/vjepa2")
from scripts.analyze_features import build_encoder, sample_clips  # reuse

V3 = "/flare/ModCon/ngetty/checkpoints/surg_2_1_v3_lambdaoff_cleandata/lr75e6_wu2_256_n16g12_weak"
META = "/flare/ModCon/ngetty/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt"
# (tag, path, key). Meta uses ema_encoder; v3 ckpts use target_encoder (what the
# PROBE reads -- so we diagnose the exact weights the downstream sees).
CKPTS = [
    ("meta", META, "ema_encoder"),
    ("e4", f"{V3}/e4.pth.tar", "target_encoder"),
    ("e9", f"{V3}/e9.pth.tar", "target_encoder"),
    ("e14", f"{V3}/e14.pth.tar", "target_encoder"),
    ("e19", f"{V3}/e19.pth.tar", "target_encoder"),
]
# domain-split sources
SURG = [("surgvu24", "/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded/surgvu24"),
        ("sitl", "/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded/sitl")]
GEN = [("kinetics400", "/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded/kinetics400")]


def effective_rank(X, eps=1e-9):
    """Effective rank = exp(entropy of normalized singular values). X: [N, D]."""
    Xc = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    s = s[s > eps]
    if len(s) == 0:
        return 0.0
    p = s / s.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


@torch.no_grad()
def encode_tokens(encoder, clip):
    """clip (T,H,W,3) uint8 -> token matrix [N_tokens, D] (pre-pool) for this clip."""
    x = torch.from_numpy(clip).float() / 255.0
    x = x.permute(3, 0, 1, 2).unsqueeze(0)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
    x = ((x - mean) / std).to(next(encoder.parameters()).device)
    out = encoder([x])
    if isinstance(out, (list, tuple)):
        out = out[0]
    if out.dim() == 3:
        out = out[0]  # [N, D]
    return out.float().cpu().numpy()


def clip_token_metrics(tokens):
    """tokens [N, D] for one clip. Returns (anisotropy, token_var)."""
    # anisotropy: mean pairwise cosine of a random subsample of tokens
    N = tokens.shape[0]
    idx = np.random.RandomState(0).choice(N, size=min(N, 256), replace=False)
    T = tokens[idx]
    Tn = T / (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
    cos = Tn @ Tn.T
    iu = np.triu_indices(len(Tn), k=1)
    anis = float(cos[iu].mean())
    token_var = float(tokens.var(axis=0).mean())  # variance across tokens per dim
    return anis, token_var


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24, help="clips per source")
    ap.add_argument("--out", default="/flare/ModCon/ngetty/probe_bench/v3_diag.json")
    args = ap.parse_args()
    dev = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")
    print(f"device={dev}")

    # Sample clips ONCE (same inputs for every checkpoint -> fair comparison).
    print("=== sampling clips (fixed across checkpoints) ===")
    clip_sets = {}
    for label, srcs in (("surg", SURG), ("gen", GEN)):
        clips = []
        for name, path in srcs:
            try:
                clips += sample_clips(path, args.n, target_hw=256)
            except Exception as e:
                print(f"  skip {name}: {e}")
        clip_sets[label] = clips
        print(f"  {label}: {len(clips)} clips")

    results = {}
    meta_pooled = {}  # for cosine drift
    for tag, path, key in CKPTS:
        print(f"\n=== {tag} ({key}) ===")
        t0 = time.time()
        enc = build_encoder(path, key, dev)
        row = {}
        for label, clips in clip_sets.items():
            pooled, anis_list, tvar_list = [], [], []
            all_tokens = []
            for c in clips:
                tok = encode_tokens(enc, c)          # [N, D]
                pooled.append(tok.mean(0))
                a, tv = clip_token_metrics(tok)
                anis_list.append(a); tvar_list.append(tv)
                all_tokens.append(tok[np.random.RandomState(1).choice(tok.shape[0], 32, replace=False)])
            P = np.stack(pooled)                      # [Nclips, D]
            Tok = np.concatenate(all_tokens)          # [Nclips*32, D]
            row[label] = {
                "clip_eff_rank": effective_rank(P),
                "token_eff_rank": effective_rank(Tok),
                "token_anisotropy": float(np.mean(anis_list)),
                "token_var": float(np.mean(tvar_list)),
                "pooled_norm": float(np.linalg.norm(P, axis=1).mean()),
            }
            # cosine drift from meta (pooled centroid)
            cen = P.mean(0); cen /= (np.linalg.norm(cen) + 1e-9)
            if tag == "meta":
                meta_pooled[label] = cen
            else:
                row[label]["cos_to_meta"] = float(cen @ meta_pooled[label])
            r = row[label]
            print(f"  {label}: clip_rank={r['clip_eff_rank']:.1f} tok_rank={r['token_eff_rank']:.1f} "
                  f"anis={r['token_anisotropy']:.3f} tok_var={r['token_var']:.4f} "
                  f"cos2meta={r.get('cos_to_meta', 1.0):.4f}")
        results[tag] = row
        del enc
        if dev.type == "xpu":
            torch.xpu.empty_cache()
        print(f"  ({time.time()-t0:.0f}s)")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n=== wrote {args.out} ===")
    # Summary table: the degradation curve
    print("\n=== DEGRADATION CURVE (surgical domain) ===")
    print(f"{'ckpt':6}{'tok_rank':>10}{'anisotropy':>12}{'tok_var':>10}{'cos2meta':>10}")
    for tag, _, _ in CKPTS:
        s = results[tag]["surg"]
        print(f"{tag:6}{s['token_eff_rank']:10.1f}{s['token_anisotropy']:12.3f}"
              f"{s['token_var']:10.4f}{s.get('cos_to_meta', 1.0):10.4f}")


if __name__ == "__main__":
    main()
