"""Metric B: frozen common-space representational error (cross-scale comparable).

The design doc's original Metric B (score each run's PREDICTOR against a fixed T* target) is blocked:
the predictor emits `4 * encoder_embed_dim` (hierarchical levels) — 4096 for vit_large, 6656 for
vit_gigantic — so different-sized runs cannot be scored against one fixed T* through their own
predictors without adding trained projection params (which defeats "frozen common space"). Confirmed
in app/vjepa_2_1/models/predictor.py (predictor_proj out = len(hierarchical_layers)*out_embed_dim).

Reformulation that keeps the goal and sidesteps the blocker: measure how well each run's frozen
ENCODER representation *linearly predicts* the fixed reference T* representation, on a held-out set.
For run r with encoder features X_r (n_tokens, d_r) and T* features Y (n_tokens, d_T*), fit a ridge
linear map W: d_r -> d_T* minimizing ||X_r W - Y||^2 + λ||W||^2, and report the normalized residual

    metric_b_error = mean_j [ ||Y_j - X_r W_j||^2 / ||Y_j - mean(Y_j)||^2 ]   (= 1 - R^2, averaged)

Why this is well-posed as a scaling y-axis (design doc §2, Metric B):
  - ONE ruler for all N: every run is scored by how much of T*'s FIXED representation it can explain.
  - Dimension-agnostic: the linear map absorbs d_r vs d_T* differences by construction — no predictor,
    no dim-mismatch blocker.
  - Lower is better and floored: a perfect linear predictor of T* gives 0; an uninformative encoder
    gives ~1. Comparable across scales because Y (the target space) never changes.
  - Uses only the encoder — the thing we scale — not the predictor.
  - Train the map on a train split, report residual on a HELD-OUT split (no overfitting the map).

This file has two layers:
  - pure alignment math (`ridge_fit`, `normalized_residual`) — CPU, unit-tested on synthetic data.
  - a `run` driver that loads a run encoder + a fixed T* encoder, extracts features over a fixed clip
    set, fits/evaluates the map, and writes metric_B.json into the run folder (GPU/MPI; launch with
    the Aurora launcher). The driver is thin; the math is what we validate here.
"""

import argparse
import glob
import json
import os

import numpy as np


# ----------------------------------------------------------------------------
# Pure alignment math (CPU, testable).
# ----------------------------------------------------------------------------


def ridge_fit(X, Y, lam):
    """Closed-form ridge: W = (X^T X + λI)^-1 X^T Y.  X:(n,d_x)  Y:(n,d_y) -> W:(d_x,d_y)."""
    d = X.shape[1]
    XtX = X.T @ X
    XtX.flat[:: d + 1] += lam  # add λ to diagonal in place
    return np.linalg.solve(XtX, X.T @ Y)


def normalized_residual(X, Y, W):
    """Mean over output dims of (residual SS / total SS) = 1 - R^2 (averaged per-dim).

    0 => X linearly reconstructs Y perfectly; ~1 => no better than predicting Y's mean.
    """
    pred = X @ W
    resid_ss = ((Y - pred) ** 2).sum(axis=0)
    total_ss = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    total_ss = np.where(total_ss <= 0, np.nan, total_ss)
    r = resid_ss / total_ss
    return float(np.nanmean(r))


def layernorm_np(Y):
    """LayerNorm over last dim (matches forward_target's F.layer_norm on T* features)."""
    mu = Y.mean(axis=-1, keepdims=True)
    var = Y.var(axis=-1, keepdims=True)
    return (Y - mu) / np.sqrt(var + 1e-6)


def align_error(X_train, Y_train, X_test, Y_test, lam):
    """Fit ridge on train, report normalized residual on held-out test. Y is LayerNorm'd first."""
    Y_train = layernorm_np(Y_train)
    Y_test = layernorm_np(Y_test)
    W = ridge_fit(X_train, Y_train, lam)
    return normalized_residual(X_test, Y_test, W)


# ----------------------------------------------------------------------------
# Feature extraction (GPU driver). Kept minimal; imported lazily so the math
# above stays importable without torch.
# ----------------------------------------------------------------------------


def _build_encoder(model_name, checkpoint, checkpoint_key, resolution, frames_per_clip, device):
    import torch
    from src.models import vision_transformer as vit

    model = vit.__dict__[model_name](img_size=resolution, num_frames=frames_per_clip)
    ckpt = torch.load(checkpoint, map_location="cpu")
    sd = ckpt[checkpoint_key]
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad = False
    return model


def _extract_features(encoder, loader, device, max_clips, subsample_tokens):
    """Return (n, d) feature matrix: encoder output tokens, optionally subsampled per clip."""
    import torch

    feats = []
    seen = 0
    with torch.no_grad():
        for batch in loader:
            clip = batch[0] if isinstance(batch, (list, tuple)) else batch
            clip = clip.to(device)
            out = encoder(clip)
            if isinstance(out, (list, tuple)):
                out = out[0]
            out = out.float().cpu().numpy()  # (B, N, D)
            B, N, D = out.shape
            for b in range(B):
                tok = out[b]
                if subsample_tokens and subsample_tokens < N:
                    idx = np.linspace(0, N - 1, subsample_tokens).astype(int)
                    tok = tok[idx]
                feats.append(tok)
                seen += 1
                if max_clips and seen >= max_clips:
                    break
            if max_clips and seen >= max_clips:
                break
    return np.concatenate(feats, axis=0)


def run(run_dir, t_star_ckpt, t_star_model, resolution, frames_per_clip, data_glob,
        lam, max_clips, subsample_tokens, train_frac, seed):
    """Load run encoder + T*, extract features on a fixed clip set, fit/eval ridge, write metric_B.json."""
    import torch  # noqa: F401  (import here so pure-math import path needs no torch)

    sc = json.load(open(os.path.join(run_dir, "scaling.json")))
    run_model = sc["model_name"]
    # find run checkpoint
    ckpt = None
    for name in ("latest.pt", "latest.pth.tar"):
        if os.path.exists(os.path.join(run_dir, name)):
            ckpt = os.path.join(run_dir, name)
            break
    if ckpt is None:
        cands = sorted(glob.glob(os.path.join(run_dir, "*.pt")))
        ckpt = cands[-1] if cands else None
    if ckpt is None:
        raise SystemExit(f"no checkpoint in {run_dir}")

    device = "xpu" if _has_xpu() else "cpu"

    enc_run = _build_encoder(run_model, ckpt, "target_encoder", resolution, frames_per_clip, device)
    enc_tst = _build_encoder(t_star_model, t_star_ckpt, "target_encoder", resolution, frames_per_clip, device)

    # only read ~max_clips files (+margin for undecodable skips), not the whole 82K pool
    max_files = int(max_clips * 1.3) + 8 if max_clips else None
    loader = _fixed_clip_loader(data_glob, resolution, frames_per_clip, seed, max_files=max_files)
    X = _extract_features(enc_run, loader, device, max_clips, subsample_tokens)
    loader = _fixed_clip_loader(data_glob, resolution, frames_per_clip, seed, max_files=max_files)  # same order
    Y = _extract_features(enc_tst, loader, device, max_clips, subsample_tokens)

    n = min(len(X), len(Y))
    X, Y = X[:n], Y[:n]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    ntr = int(train_frac * n)
    tr, te = perm[:ntr], perm[ntr:]
    err = align_error(X[tr], Y[tr], X[te], Y[te], lam)

    out = {"metric_b_error": err, "d_run": int(X.shape[1]), "d_tstar": int(Y.shape[1]),
           "n_tokens": int(n), "t_star_model": t_star_model, "lam": lam, "source": "frozen_tstar_linear"}
    with open(os.path.join(run_dir, "metric_B.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"{os.path.basename(run_dir)}: metric_b_error={err:.4f} (d_run={X.shape[1]} -> d_T*={Y.shape[1]})")
    return out


def _has_xpu():
    try:
        import torch
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    except Exception:
        return False


def _fixed_clip_loader(data_glob, resolution, frames_per_clip, seed, batch_size=4, max_files=None):
    """Self-contained deterministic clip loader: decord-decode a fixed set of videos -> batched
    (B, C, T, H, W) float tensors normalized like the trainer (ImageNet mean/std). Decoupled from the
    training data_manager (whose init_data signature has no crop_size and needs a transform+collator);
    for an eval-only feature extractor a thin direct reader is more robust. Deterministic: sorted file
    list + fixed uniform frame sampling + fixed seed, so every run/T* sees the SAME clips (required for
    a fair fixed-ruler comparison). Yields batches; skips undecodable clips."""
    import torch
    import decord

    decord.bridge.set_bridge("native")
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    paths = sorted(glob.glob(data_glob))
    if max_files:
        paths = paths[:max_files]

    def _load_one(p):
        try:
            vr = decord.VideoReader(p, width=resolution, height=resolution)
            n = len(vr)
            if n < 1:
                return None
            idx = np.linspace(0, n - 1, frames_per_clip).astype(int)
            frames = vr.get_batch(idx).asnumpy()          # (T, H, W, C) uint8
            x = frames.astype(np.float32) / 255.0
            x = (x - mean) / std
            x = np.transpose(x, (3, 0, 1, 2))              # (C, T, H, W)
            return torch.from_numpy(x)
        except Exception:
            return None

    class _Loader:
        def __iter__(self):
            buf = []
            for p in paths:
                t = _load_one(p)
                if t is None:
                    continue
                buf.append(t)
                if len(buf) == batch_size:
                    yield (torch.stack(buf),)
                    buf = []
            if buf:
                yield (torch.stack(buf),)

    return _Loader()


# Default T* = Meta V-JEPA2 ViT-g/16 @384 (epoch 40, world_size 512 — the released general pretrain).
# General (no domain bias); loads via checkpoint_key='target_encoder'. Registry name = vit_giant.
T_STAR_CKPT = "/flare/ModCon/ngetty/checkpoints/vjepa2_1_vitg_384.pt"
T_STAR_MODEL = "vit_giant"
LADDER_ORDER = ["vit_tiny", "vit_small", "vit_base", "vit_large", "vit_giant", "vit_gigantic"]


def _model_from_dir(rd):
    b = os.path.basename(rd)
    for m in LADDER_ORDER:
        if b.startswith(m):
            return m
    return "?"


def run_all(runs_root, t_star_ckpt, t_star_model, data_glob, resolution, frames,
            lam, max_clips, subsample_tokens, train_frac, seed):
    """Score every completed run under runs_root, then apply the CEILING GATE (design doc §7b).

    Ceiling gate: Metric B is trustworthy only if 1-R2 still DECREASES as N grows at the top. Our
    vit_gigantic (1.9B) is BIGGER than T* (1B ViT-g), so predicting T* may saturate there. We score
    all runs, then per budget check metric_b decreases giant->gigantic. If not (ceiling hit), FLAG the
    gigantic cell -> report Metric B tiny..giant, use Metric A (SSv2) for gigantic instead."""
    import json as _json
    from collections import defaultdict
    runs = sorted(d for d in glob.glob(os.path.join(runs_root, "*")) if os.path.isdir(d))
    scored = []
    for rd in runs:
        sj = os.path.join(rd, "scaling.json")
        has_ckpt = os.path.exists(os.path.join(rd, "latest.pth.tar")) or glob.glob(os.path.join(rd, "*.pth.tar"))
        if not (os.path.exists(sj) and has_ckpt):
            continue
        try:
            run(rd, t_star_ckpt, t_star_model, resolution, frames, data_glob,
                lam, max_clips, subsample_tokens, train_frac, seed)
            mb = _json.load(open(os.path.join(rd, "metric_B.json")))
            scored.append((os.path.basename(rd), _model_from_dir(rd), mb.get("metric_b_error")))
        except Exception as e:
            print(f"  SKIP {os.path.basename(rd)}: {e}")
    print("\n=== CEILING GATE (metric_b must decrease giant->gigantic) ===")
    by_budget = defaultdict(dict)
    for name, model, err in scored:
        budget = name.split("_C")[-1] if "_C" in name else "?"
        by_budget[budget][model] = err
    flagged = []
    for budget, d in sorted(by_budget.items()):
        g, gg = d.get("vit_giant"), d.get("vit_gigantic")
        if g is not None and gg is not None:
            ok = gg < g
            print(f"  C{budget}: giant={g:.4f} gigantic={gg:.4f} -> "
                  f"{'OK' if ok else 'CEILING HIT — flag gigantic, use Metric A there'}")
            if not ok:
                flagged.append(f"vit_gigantic_C{budget}")
    print(f"  flagged: {flagged or 'none'}")
    return scored, flagged


def main():
    ap = argparse.ArgumentParser(description="Metric B: frozen-T* linear-predictivity error")
    ap.add_argument("--run-dir", help="a single sweep run folder (has scaling.json + ckpt)")
    ap.add_argument("--runs-root", help="batch: score ALL runs under this dir + ceiling gate")
    ap.add_argument("--t-star-ckpt", default=T_STAR_CKPT, help="fixed reference encoder checkpoint")
    ap.add_argument("--t-star-model", default=T_STAR_MODEL, help="T* model_name")
    ap.add_argument("--data-glob", required=True, help="held-out clips glob (fixed eval set)")
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--lam", type=float, default=1.0, help="ridge regularization")
    ap.add_argument("--max-clips", type=int, default=512)
    ap.add_argument("--subsample-tokens", type=int, default=64, help="tokens kept per clip (0=all)")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=239)
    args = ap.parse_args()
    if args.runs_root:
        run_all(args.runs_root, args.t_star_ckpt, args.t_star_model, args.data_glob,
                args.resolution, args.frames, args.lam, args.max_clips, args.subsample_tokens,
                args.train_frac, args.seed)
    elif args.run_dir:
        run(args.run_dir, args.t_star_ckpt, args.t_star_model, args.resolution, args.frames,
            args.data_glob, args.lam, args.max_clips, args.subsample_tokens, args.train_frac, args.seed)
    else:
        ap.error("need --run-dir or --runs-root")


if __name__ == "__main__":
    main()
