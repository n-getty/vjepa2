"""Compute F1@k and edit-score (segmental metrics) for an ASFormer probe.

Run inference on the val set, stitch the per-frame predictions back into
per-video timelines using the sample path (`/.../video_XX/frame_NNNNNNNNN_ctxK.mp4`),
then compute the SAR-RARP50 challenge-style segmental F1@{10,25,50} and
edit (segmental Levenshtein) score on top of per-frame macro F1 / accuracy.

Usage (single GPU is enough — inference only):
    python eval_segmental_f1.py \\
        --yaml  /path/to/rendered/probe.yaml \\
        --checkpoint /path/to/best.pt \\
        --out /path/to/segmental_metrics.json

The yaml is the rendered probe yaml (the one staged into vjepa2_polaris/.runtime_configs).
The checkpoint is best.pt or latest.pt from the probe run.

Why this is not just per-frame F1 with a different number:
    - Per-frame macro F1 weighs every frame equally; over-segmentation barely
      shows up.
    - F1@k pairs predicted segments with GT segments by IoU >= k% (k in
      {10,25,50}). Over-segmentation creates many extra predicted segments
      -> all become FPs -> F1@10 drops sharply.
    - Edit score is segmental Levenshtein normalised by the longer label
      sequence; captures order-of-segments correctness.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

# Make the Aurora repo importable for `evals.*` and `src.*`.
sys.path.insert(0, "/lus/flare/projects/ModCon/ngetty/vjepa2")

# XPU support: importing intel_extension_for_pytorch registers torch.xpu kernels.
try:
    import intel_extension_for_pytorch  # noqa: F401
except Exception:
    pass


def _pick_device():
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

from evals.video_classification_frozen.models import init_module  # noqa: E402
from evals.video_classification_frozen.eval import (  # noqa: E402
    adapt_state_dict_for_model,
)
from evals.video_classification_frozen.utils import make_transforms  # noqa: E402
from src.datasets.video_dataset import VideoDataset  # noqa: E402
from src.models.asformer_head import ASFormerHead  # noqa: E402
from src.utils.checkpoint_loader import robust_checkpoint_loader  # noqa: E402


# ---------------------------------------------------------------------------
# Sample-path parsing
# ---------------------------------------------------------------------------
# Accept video_NN/ and video_NN_M/ (SAR-RARP50 has split-video subdirs).
_PATH_RE = re.compile(r"(video_\d+(?:_\d+)?)/frame_(\d+)_ctx(\d+)\.mp4$")


def parse_sample_path(p: str) -> tuple[str, int]:
    m = _PATH_RE.search(p)
    if not m:
        raise ValueError(f"Unrecognised sample path: {p}")
    return m.group(1), int(m.group(2))


# ---------------------------------------------------------------------------
# Segmental metrics (Lea et al. ED-TCN reference implementation)
# ---------------------------------------------------------------------------
def labels_to_segments(seq: np.ndarray, bg_class: int | None = None):
    segs: list[tuple[int, int, int]] = []
    if len(seq) == 0:
        return segs
    cur = int(seq[0]); start = 0
    for i in range(1, len(seq)):
        if int(seq[i]) != cur:
            if bg_class is None or cur != bg_class:
                segs.append((cur, start, i))
            cur = int(seq[i]); start = i
    if bg_class is None or cur != bg_class:
        segs.append((cur, start, len(seq)))
    return segs


def f1_at_k(pred_seq, gt_seq, k: float, bg_class=None):
    p_segs = labels_to_segments(pred_seq, bg_class=bg_class)
    g_segs = labels_to_segments(gt_seq, bg_class=bg_class)
    matched = [False] * len(g_segs)
    tp, fp = 0, 0
    for p_cls, p_s, p_e in p_segs:
        best_iou, best_idx = 0.0, -1
        for j, (g_cls, g_s, g_e) in enumerate(g_segs):
            if g_cls != p_cls or matched[j]:
                continue
            inter = max(0, min(p_e, g_e) - max(p_s, g_s))
            union = max(p_e, g_e) - min(p_s, g_s)
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou, best_idx = iou, j
        if best_iou >= k and best_idx >= 0:
            tp += 1; matched[best_idx] = True
        else:
            fp += 1
    fn = matched.count(False)
    return tp, fp, fn


def edit_score(pred_seq, gt_seq, bg_class=None):
    p = [c for (c, _, _) in labels_to_segments(pred_seq, bg_class=bg_class)]
    g = [c for (c, _, _) in labels_to_segments(gt_seq, bg_class=bg_class)]
    m, n = len(p), len(g)
    if m == 0 and n == 0:
        return 100.0
    if m == 0 or n == 0:
        return 0.0
    dp = np.zeros((m + 1, n + 1), dtype=np.int64)
    dp[:, 0] = np.arange(m + 1); dp[0, :] = np.arange(n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if p[i - 1] == g[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return (1 - dp[m, n] / max(m, n)) * 100.0


def per_class_metrics(preds, labels, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    valid = (labels >= 0) & (labels < num_classes) & (preds >= 0) & (preds < num_classes)
    np.add.at(cm, (labels[valid], preds[valid]), 1)
    tp = np.diag(cm).astype(np.float64)
    pred_sum = cm.sum(axis=0).astype(np.float64)
    true_sum = cm.sum(axis=1).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(pred_sum > 0, tp / pred_sum, 0.0)
        recall = np.where(true_sum > 0, tp / true_sum, 0.0)
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float((f1 * true_sum).sum() / max(true_sum.sum(), 1)),
        "per_class_f1": f1.tolist(),
    }


# ---------------------------------------------------------------------------
# Model construction (mirrors eval_perclass_f1.py)
# ---------------------------------------------------------------------------
def build_encoder_and_heads(cfg, device):
    args_pretrain = cfg["model_kwargs"]
    args_exp = cfg["experiment"]
    args_data = args_exp["data"]
    args_classifier = args_exp["classifier"]
    args_opt = args_exp["optimization"]

    encoder = init_module(
        module_name=args_pretrain["module_name"],
        frames_per_clip=args_data["frames_per_clip"],
        resolution=args_data["resolution"],
        checkpoint=args_pretrain["checkpoint"],
        model_kwargs=args_pretrain["pretrain_kwargs"],
        wrapper_kwargs=args_pretrain.get("wrapper_kwargs", {}),
        device=device,
    )
    if args_classifier.get("name") != "asformer":
        raise ValueError("This script only handles classifier.name=asformer")

    ak = args_classifier.get("asformer_kwargs", {})
    n_heads = len(args_opt["multihead_kwargs"])

    def _build():
        return ASFormerHead(
            embed_dim=encoder.embed_dim,
            num_classes=args_data["num_classes"],
            num_clips=ak.get("num_clips", args_data.get("num_segments", 1)),
            tokens_per_clip=ak.get("tokens_per_clip", args_data["frames_per_clip"] // 2),
            num_layers=ak.get("num_layers", 10),
            num_heads=ak.get("num_heads", 8),
            mlp_ratio=ak.get("mlp_ratio", 4.0),
            dropout=ak.get("dropout", 0.0),
            return_sequence=ak.get("return_sequence", True),
            temporal_tokens=ak.get("temporal_tokens", None),
        ).to(device)

    classifiers = [_build() for _ in range(n_heads)]
    return encoder, classifiers


def load_classifier_weights(classifiers, checkpoint_path, device):
    ckpt = robust_checkpoint_loader(checkpoint_path, map_location=torch.device("cpu"))
    state_dicts = ckpt["classifiers"]
    if len(state_dicts) != len(classifiers):
        raise ValueError(
            f"Checkpoint has {len(state_dicts)} classifier(s) but "
            f"{len(classifiers)} were built from the YAML."
        )
    for c, sd in zip(classifiers, state_dicts):
        adapted = adapt_state_dict_for_model(c, sd)
        c.load_state_dict(adapted)
        c.eval()
    return ckpt.get("epoch", -1)


# ---------------------------------------------------------------------------
# Dataset (direct VideoDataset with return_sample_path=True)
# ---------------------------------------------------------------------------
def build_val_dataset(cfg):
    args_data = cfg["experiment"]["data"]
    transform = make_transforms(
        training=False,
        num_views_per_clip=args_data.get("num_views_per_segment", 1),
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4 / 3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=args_data["resolution"],
        normalize=args_data.get("normalization")
            or [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)],
    )
    return VideoDataset(
        data_paths=[args_data["dataset_val"]],
        frames_per_clip=args_data["frames_per_clip"],
        frame_step=args_data.get("frame_step", 1),
        num_clips=args_data.get("num_segments", 1),
        random_clip_sampling=False,
        allow_clip_overlap=True,
        filter_short_videos=False,
        filter_long_videos=int(10**9),
        duration=args_data.get("clip_duration", None),
        sequence_labels=args_data.get("sequence_labels", False),
        return_sample_path=True,
        transform=transform,
        shared_transform=None,
    )


@torch.no_grad()
def run_inference(encoder, classifier, dataset, device, use_bfloat16):
    encoder.eval()
    paths_all: list[str] = []
    preds_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    n = len(dataset)
    for i in range(n):
        item = dataset[i]
        # In eval mode, VideoTransform.__call__ wraps each clip's output as
        # a single-element list to make room for a "views" axis. So buffer is
        # list[list[Tensor[C,T,H,W]]] — outer=clips, inner=views.
        buffer, label, clip_indices, sample_path = item
        # Encoder also expects outer=segments, inner=views, each [B,C,T,H,W].
        clips = [
            [v.unsqueeze(0).to(device, non_blocking=True) for v in views]
            for views in buffer
        ]
        # clip_indices comes from VideoDataset as a list of np.ndarrays (one per
        # clip). Convert to Tensor and add batch dim before sending to device.
        clip_idx_t = [
            torch.as_tensor(ci).unsqueeze(0).to(device, non_blocking=True)
            for ci in clip_indices
        ]
        dev_type = device.type  # "xpu", "cuda", or "cpu"
        if dev_type in ("xpu", "cuda"):
            amp_ctx = torch.amp.autocast(dev_type, dtype=torch.bfloat16, enabled=use_bfloat16)
        else:
            amp_ctx = torch.amp.autocast("cpu", enabled=False)
        with amp_ctx:
            outputs_per_view = encoder(clips, clip_idx_t)
            head_outputs = [classifier(o) for o in outputs_per_view]
            probs = sum(F.softmax(o.float(), dim=-1) for o in head_outputs)
            preds = probs.argmax(dim=-1).reshape(-1).cpu().numpy()
        labels_np = (label.reshape(-1).cpu().numpy()
                     if isinstance(label, torch.Tensor) else
                     np.asarray(label).reshape(-1))
        paths_all.append(sample_path)
        preds_all.append(preds)
        labels_all.append(labels_np)
        if i % 50 == 0:
            print(f"  inferred {i}/{n}", flush=True)
    return paths_all, preds_all, labels_all


def stitch_per_video(paths, preds, labels):
    """Group clips by video, sort by start_frame, concat into per-video timelines.
    Where overlapping clips disagree, the later prediction wins (final-write).
    Returns: dict video_id -> (pred_seq, label_seq).
    """
    by_video: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = defaultdict(list)
    for p, pred, lab in zip(paths, preds, labels):
        vid, start = parse_sample_path(p)
        by_video[vid].append((start, pred, lab))

    timelines: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for vid, segs in by_video.items():
        segs.sort(key=lambda x: x[0])
        last_start, last_pred, _ = segs[-1]
        total_len = last_start + len(last_pred)
        pred_full = np.full(total_len, -1, dtype=np.int64)
        lab_full = np.full(total_len, -1, dtype=np.int64)
        for start, pred, lab in segs:
            end = start + len(pred)
            if end > total_len:
                _pf = np.full(end, -1, dtype=np.int64)
                _lf = np.full(end, -1, dtype=np.int64)
                _pf[:total_len] = pred_full; _lf[:total_len] = lab_full
                pred_full, lab_full, total_len = _pf, _lf, end
            pred_full[start:end] = pred
            lab_full[start:end] = lab
        valid = lab_full >= 0
        if not valid.any():
            continue
        first, last = np.where(valid)[0][[0, -1]]
        timelines[vid] = (pred_full[first:last + 1].copy(),
                          lab_full[first:last + 1].copy())
    return timelines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, help="rendered probe yaml")
    ap.add_argument("--checkpoint", required=True, help="best.pt or latest.pt")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    ap.add_argument("--head-idx", type=int, default=0,
                    help="which classifier head to evaluate (default 0)")
    ap.add_argument("--bg-class", type=int, default=None,
                    help="optional background class to ignore in segmental metrics")
    args = ap.parse_args()

    with open(args.yaml) as f:
        cfg = yaml.safe_load(f)

    device = _pick_device()
    print(f"device: {device}", flush=True)
    if device.type == "cpu" and os.environ.get("SEGF1_ALLOW_CPU") != "1":
        print("FATAL: no XPU/CUDA device found — refusing to run full val inference "
              "on CPU (set SEGF1_ALLOW_CPU=1 to override). Check ZE_FLAT_DEVICE_HIERARCHY "
              "/ frameworks module / ZE_AFFINITY_MASK.", flush=True)
        sys.exit(2)
    encoder, classifiers = build_encoder_and_heads(cfg, device)
    epoch = load_classifier_weights(classifiers, args.checkpoint, device)
    print(f"loaded classifiers from epoch {epoch}", flush=True)

    dataset = build_val_dataset(cfg)
    print(f"val dataset: {len(dataset)} clips", flush=True)
    use_bf16 = cfg["experiment"]["optimization"].get("use_bfloat16", True)

    paths, preds, labels = run_inference(
        encoder, classifiers[args.head_idx], dataset, device, use_bf16,
    )

    num_classes = cfg["experiment"]["data"]["num_classes"]
    flat_pred = np.concatenate(preds)
    flat_lab = np.concatenate(labels)
    frame_metrics = per_class_metrics(flat_pred, flat_lab, num_classes)
    print(f"per-frame: acc={frame_metrics['accuracy']*100:.2f} "
          f"macro_f1={frame_metrics['macro_f1']*100:.2f}", flush=True)

    timelines = stitch_per_video(paths, preds, labels)
    print(f"stitched {len(timelines)} videos", flush=True)

    ks = [0.10, 0.25, 0.50]
    tot = {k: [0, 0, 0] for k in ks}
    edits = []
    for vid, (p, g) in timelines.items():
        for k in ks:
            tp, fp, fn = f1_at_k(p, g, k, bg_class=args.bg_class)
            tot[k][0] += tp; tot[k][1] += fp; tot[k][2] += fn
        edits.append(edit_score(p, g, bg_class=args.bg_class))

    seg_metrics: dict = {}
    for k in ks:
        tp, fp, fn = tot[k]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        label = f"F1@{int(k*100)}"
        seg_metrics[label] = {"precision": prec, "recall": rec, "f1": f1,
                              "tp": tp, "fp": fp, "fn": fn}
        print(f"  {label}: P={prec*100:.2f} R={rec*100:.2f} F1={f1*100:.2f}", flush=True)
    seg_metrics["edit_score"] = float(np.mean(edits)) if edits else 0.0
    print(f"  edit_score: {seg_metrics['edit_score']:.2f}", flush=True)

    out = {
        "checkpoint": args.checkpoint,
        "yaml": args.yaml,
        "epoch": epoch,
        "num_videos": len(timelines),
        "num_frames": int(sum(len(p) for p, _ in timelines.values())),
        "per_frame": frame_metrics,
        "segmental": seg_metrics,
        "bg_class": args.bg_class,
    }
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
