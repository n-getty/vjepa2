"""Feature-space structure analysis of the phase-1 V-JEPA 2.1 checkpoint.

Loads BOTH the init encoder and our trained encoder, computes mean-pooled
video embeddings on a stratified sample of held-out clips from every dataset
in the surgical corpus, then reports:

  1. Per-dataset embedding norm distribution + effective rank (anti-collapse)
  2. Inter-dataset cluster separability (silhouette score)
  3. k-NN source-dataset prediction accuracy (lower = LESS dataset-bias = BETTER)
  4. Cross-dataset cosine similarity matrix
  5. Compares (1-4) between INIT encoder and our TRAINED encoder

A healthy trained encoder should:
  - Have LOWER k-NN dataset-bias than the init (it learned surgical features
    that generalize across the source datasets, not just memorize source
    statistics)
  - Have HIGHER or comparable effective rank (no dim collapse)
  - Preserve roughly the inter-dataset structure of the init

Run via mpiexec on a held node (single-process is fine — this is CPU-bound
for the I/O part and small for the encoder forward).
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/lus/flare/projects/ModCon/ngetty/vjepa2")
from app.vjepa_2_1.utils import init_video_model


CKPT_TRAINED = "/flare/ModCon/ngetty/checkpoints/surg_2_1_v1/phase1_warmup_n16g12_weak/latest.pth.tar"
CKPT_INIT = "/flare/ModCon/ngetty/checkpoints/vjepa2_1_vitl_dist_vitG_384.pt"

DATASETS = [
    ("crcd", "/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded/crcd"),
    ("endovis15", "/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded/endovis15"),
    ("jigsaw", "/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded/jigsaw"),
    ("kinetics400", "/flare/ModCon/ngetty/data/surg_vid_webdataset/kinetics400"),
    ("miccai_2017", "/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded/miccai_2017"),
    ("miccai_endoseg", "/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded/miccai_endoseg"),
    ("sitl", "/flare/ModCon/ngetty/data/surg_vid_webdataset/sitl"),
    ("surgenet_robotic", "/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded/surgenet_robotic"),
    ("surgtoolloc2022", "/flare/ModCon/ngetty/data/surg_vid_webdataset/surgtoolloc2022"),
    ("surgvisdom", "/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded/surgvisdom"),
    ("surgvu24", "/flare/ModCon/ngetty/data/surg_vid_webdataset/surgvu24"),
]


def build_encoder(ckpt_path, ckpt_key, device):
    """Build a vit_large encoder and load weights from the given checkpoint key."""
    encoder, _ = init_video_model(
        device=device,
        patch_size=16,
        max_num_frames=16,
        tubelet_size=2,
        model_name="vit_large",
        crop_size=256,
        pred_depth=24,
        pred_num_heads=12,
        pred_embed_dim=384,
        uniform_power=True,
        use_mask_tokens=True,
        num_mask_tokens=2,
        zero_init_mask_tokens=True,
        use_sdpa=True,
        use_rope=True,
        use_activation_checkpointing=False,
        is_causal=False,
        pred_is_causal=False,
        img_temporal_dim_size=1,
        n_registers=0,
        n_registers_predictor=0,
        has_cls_first=False,
        interpolate_rope=True,
        modality_embedding=True,
    )
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck[ckpt_key]
    # Strip DDP prefixes
    sd_clean = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        sd_clean[nk] = v
    msg = encoder.load_state_dict(sd_clean, strict=False)
    print(f"  [{ckpt_path}:{ckpt_key}] loaded with {len(msg.missing_keys)} missing, {len(msg.unexpected_keys)} unexpected")
    if msg.missing_keys[:3]:
        print(f"    missing example: {msg.missing_keys[:3]}")
    if msg.unexpected_keys[:3]:
        print(f"    unexpected example: {msg.unexpected_keys[:3]}")
    encoder.eval()
    return encoder.to(device)


@torch.no_grad()
def embed_video(encoder, clip):
    """clip: (T, H, W, 3) uint8 -> embedding (D,) on CPU.

    Normalizes per ImageNet, runs through MultiSeqWrapper, mean-pools tokens.
    """
    # to (B=1, C=3, T, H, W) float, normalized
    x = torch.from_numpy(clip).float() / 255.0  # (T, H, W, 3)
    x = x.permute(3, 0, 1, 2).unsqueeze(0)       # (1, 3, T, H, W)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
    x = (x - mean) / std
    # Keep input fp32 — encoder is fp32 in this eval (no autocast either, since
    # XPU autocast behavior varies; fp32 forward on a 304M model is fine for
    # a few hundred clips of inference)
    x = x.to(next(encoder.parameters()).device)
    out = encoder([x])  # list-input → list-output
    if isinstance(out, (list, tuple)):
        out = out[0]
    if out.dim() == 3:    # (B, N, D)
        out = out.mean(dim=1)
    return out.float().cpu().numpy()[0]


def sample_clips(dataset_dir, n_clips, frames_per_clip=16, frame_step=4, target_hw=256):
    """Pull n_clips video clips from a WebDataset shard. Returns list of (T,H,W,3) uint8 arrays."""
    import tarfile
    import io
    from decord import VideoReader, cpu

    tars = sorted(Path(dataset_dir).glob("*.tar"))
    if not tars:
        return []

    clips = []
    # Walk shards until we have enough
    for tar_path in tars:
        if len(clips) >= n_clips:
            break
        with tarfile.open(tar_path, "r|") as tf:
            current_key = None
            video_bytes = None
            for m in tf:
                if not m.isfile():
                    continue
                name = m.name
                dot = name.find(".")
                key = name[:dot] if dot > 0 else name
                ext = name[dot+1:] if dot > 0 else ""
                if ext in ("mp4", "avi", "mov", "webm", "mkv"):
                    fobj = tf.extractfile(m)
                    if fobj is None:
                        continue
                    video_bytes = fobj.read()
                    try:
                        vr = VideoReader(io.BytesIO(video_bytes), num_threads=1, ctx=cpu(0))
                        if len(vr) < frames_per_clip * frame_step:
                            continue
                        # Take frames evenly from the middle of the video
                        n = len(vr)
                        start = max(0, (n - frames_per_clip * frame_step) // 2)
                        idx = [start + i * frame_step for i in range(frames_per_clip)]
                        idx = [min(i, n - 1) for i in idx]
                        buf = vr.get_batch(idx).asnumpy()  # (T, H, W, 3)
                        # Center crop to square, resize to target_hw
                        T, H, W, _ = buf.shape
                        s = min(H, W)
                        y0 = (H - s) // 2
                        x0 = (W - s) // 2
                        buf = buf[:, y0:y0+s, x0:x0+s, :]
                        if s != target_hw:
                            # Cheap nearest-neighbor resize via torch
                            t = torch.from_numpy(buf).permute(0, 3, 1, 2).float()
                            t = F.interpolate(t, size=target_hw, mode="bilinear", align_corners=False)
                            buf = t.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8).numpy()
                        clips.append(buf)
                        if len(clips) >= n_clips:
                            break
                    except Exception as e:
                        continue
    return clips


def compute_embeddings(encoder, samples_per_ds, label):
    print(f"\n=== Computing {label} embeddings ({samples_per_ds}/dataset) ===")
    embs = {}
    for name, path in DATASETS:
        if not os.path.isdir(path):
            print(f"  {name}: SKIP (dir missing)")
            continue
        t0 = time.time()
        clips = sample_clips(path, samples_per_ds)
        if not clips:
            print(f"  {name}: SKIP (no clips loaded)")
            continue
        ds_embs = []
        for clip in clips:
            try:
                e = embed_video(encoder, clip)
                ds_embs.append(e)
            except Exception as ex:
                print(f"    skip clip: {ex}")
        if not ds_embs:
            continue
        E = np.stack(ds_embs)
        embs[name] = E
        dt = time.time() - t0
        print(f"  {name:<22s}  n={len(ds_embs):3d}  D={E.shape[1]}  norm_mean={np.linalg.norm(E,axis=1).mean():.3f}  ({dt:.1f}s)")
    return embs


def effective_rank(E, eps=1e-6):
    """Effective rank via normalized singular value entropy."""
    s = np.linalg.svd(E - E.mean(0, keepdims=True), compute_uv=False)
    p = s / (s.sum() + eps)
    p = p[p > eps]
    return float(np.exp(-(p * np.log(p)).sum()))


def knn_dataset_pred(embs, k=5):
    """How well can we predict source dataset from embedding via k-NN?
    Lower accuracy = features are dataset-invariant = good for downstream transfer."""
    names = list(embs.keys())
    X = np.concatenate([embs[n] for n in names])
    y = np.concatenate([np.full(len(embs[n]), i) for i, n in enumerate(names)])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    # leave-one-out k-NN
    sims = X @ X.T
    np.fill_diagonal(sims, -1)
    nn_idx = np.argpartition(-sims, k, axis=1)[:, :k]
    nn_y = y[nn_idx]
    # majority vote
    preds = np.array([np.bincount(row, minlength=len(names)).argmax() for row in nn_y])
    return float((preds == y).mean()), 1.0 / len(names)


def silhouette(embs, max_per_ds=50):
    """Mean silhouette score across datasets (uses cosine)."""
    names = list(embs.keys())
    X = np.concatenate([embs[n][:max_per_ds] for n in names])
    y = np.concatenate([np.full(min(len(embs[n]), max_per_ds), i) for i, n in enumerate(names)])
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    sims = X @ X.T
    dists = 1.0 - sims
    sils = []
    for i in range(len(X)):
        own = y[i]
        a = dists[i, (y == own) & (np.arange(len(X)) != i)].mean() if (y == own).sum() > 1 else 0
        b_options = [dists[i, y == c].mean() for c in range(len(names)) if c != own and (y == c).sum() > 0]
        if not b_options:
            continue
        b = min(b_options)
        sils.append((b - a) / max(a, b, 1e-8))
    return float(np.mean(sils)) if sils else 0.0


def cross_ds_similarity(embs):
    """Mean cosine similarity matrix between dataset centroids."""
    names = list(embs.keys())
    centroids = []
    for n in names:
        c = embs[n].mean(0)
        c /= np.linalg.norm(c) + 1e-8
        centroids.append(c)
    C = np.stack(centroids)
    return C @ C.T, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-per-dataset", type=int, default=30)
    ap.add_argument("--out-dir", default="/flare/ModCon/ngetty/checkpoints/surg_2_1_v1/feature_analysis")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("xpu:0") if torch.xpu.is_available() else torch.device("cpu")
    print(f"device: {device}")

    print("\n[1/4] Building encoders...")
    trained = build_encoder(CKPT_TRAINED, "encoder", device)
    init = build_encoder(CKPT_INIT, "ema_encoder", device)

    print("\n[2/4] Sampling and embedding clips...")
    embs_trained = compute_embeddings(trained, args.samples_per_dataset, "TRAINED (phase 1 epoch 12)")
    embs_init = compute_embeddings(init, args.samples_per_dataset, "INIT (vjepa2_1 vitG-distilled ViT-L)")

    print("\n[3/4] Computing metrics...")
    results = {}
    for label, embs in [("init", embs_init), ("trained", embs_trained)]:
        if not embs:
            continue
        # Per-dataset effective rank
        eff_ranks = {n: effective_rank(E) for n, E in embs.items()}
        # k-NN dataset bias
        acc, chance = knn_dataset_pred(embs)
        # Silhouette
        sil = silhouette(embs)
        # Cross-dataset similarity matrix
        sim_mat, names = cross_ds_similarity(embs)
        results[label] = {
            "effective_rank": eff_ranks,
            "knn_dataset_acc": acc,
            "knn_dataset_chance": chance,
            "silhouette": sil,
            "centroid_similarity": sim_mat.tolist(),
            "dataset_names": names,
        }

    print("\n[4/4] Reporting...")
    print()
    print(f"{'metric':<35s} {'INIT':>12s} {'TRAINED':>12s}  Δ")
    print("=" * 70)

    if "init" in results and "trained" in results:
        # k-NN dataset bias (lower = better generalization)
        i = results["init"]["knn_dataset_acc"]
        t = results["trained"]["knn_dataset_acc"]
        c = results["init"]["knn_dataset_chance"]
        print(f"{'k-NN dataset-bias acc':<35s} {i:>12.4f} {t:>12.4f}  {t-i:+.4f}  (chance={c:.3f}; lower=better)")
        # Silhouette (per-dataset cluster cohesion)
        i = results["init"]["silhouette"]
        t = results["trained"]["silhouette"]
        print(f"{'silhouette (cohesion vs sep)':<35s} {i:>12.4f} {t:>12.4f}  {t-i:+.4f}")
        # Effective rank mean across datasets
        ier = np.mean(list(results["init"]["effective_rank"].values()))
        ter = np.mean(list(results["trained"]["effective_rank"].values()))
        print(f"{'effective rank (mean across ds)':<35s} {ier:>12.2f} {ter:>12.2f}  {ter-ier:+.2f}  (higher=less collapse)")

        # Per-dataset effective rank
        print()
        print("Per-dataset effective rank:")
        print(f"{'dataset':<22s} {'INIT':>10s} {'TRAINED':>10s}  Δ")
        for name in results["init"]["effective_rank"]:
            ir = results["init"]["effective_rank"][name]
            tr = results["trained"]["effective_rank"].get(name, 0.0)
            print(f"  {name:<20s} {ir:>10.2f} {tr:>10.2f}  {tr-ir:+.2f}")

    # Save full results
    out_path = Path(args.out_dir) / "feature_analysis.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
