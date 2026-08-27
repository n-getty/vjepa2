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

Also runnable under mpiexec for multi-rank/multi-node scoring (see
scripts/run_sitl_segf1_aurora.sh): each rank scores a disjoint round-robin
shard of the val clips and rank 0 gathers + stitches/scores the union. This
uses plain mpi4py (rank/size + one gather), NOT torch.distributed -- there is
no gradient/collective need here, so the whole XPU multi-node
backend-selection/device_id story in src/utils/distributed.py doesn't apply.

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
import time
from collections import defaultdict
from pathlib import Path

# Defensive: pin per-rank XPU tile BEFORE importing torch, mirroring
# app/main_dist_aurora.py's tile-pinning -- without this, every rank on a node
# opens all 12 tile contexts and Level-Zero serializes them onto tile 0,
# defeating the whole point of multi-rank parallel inference.
for _var in ("PALS_LOCAL_RANKID", "PMI_LOCAL_RANK", "MPI_LOCALRANKID",
             "OMPI_COMM_WORLD_LOCAL_RANK", "LOCAL_RANK"):
    if _var in os.environ:
        os.environ["ZE_AFFINITY_MASK"] = os.environ[_var]
        break

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score

# Make the repo root importable for `evals.*` and `src.*`, regardless of
# which checkout (Aurora/flare, Polaris/eagle, ...) this script lives in.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
# Accept video_NN/ and video_NN_M/ (SAR-RARP50 has split-video subdirs) as well
# as videoNN/ with no underscore (SITL-phase's naming, e.g. video04/).
_PATH_RE = re.compile(r"(video_?\d+(?:_\d+)?)/frame_(\d+)_ctx(\d+)\.mp4$")


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


def merge_short_segments(seq: np.ndarray, min_len: int) -> np.ndarray:
    """Collapse any run shorter than `min_len` frames into its preceding
    segment's label (first segment merges into the following one instead).

    Root cause this exists for: raw per-frame argmax on this probe flickers
    near phase boundaries (median predicted-segment length was 3-5 frames,
    vs. ~236 true segments in the val set) -- each flicker mints a spurious
    segment that `labels_to_segments` counts as a full false positive, so
    F1@k swings ~12pts seed-to-seed on flicker count alone while per-frame
    macro-F1/mAP (which don't care about segment boundaries) stay within
    ~1-2pts. This is majority-vote-style boundary cleanup, not a change to
    the underlying classifier -- applied identically to every seed/checkpoint
    before segmental scoring, never selected post-hoc per result.
    """
    if min_len <= 1:
        return seq
    seq = seq.copy()
    while True:
        segs = labels_to_segments(seq)
        if len(segs) <= 1:
            return seq
        short = [(c, s, e) for c, s, e in segs if e - s < min_len]
        if not short:
            return seq
        c, s, e = short[0]
        idx = segs.index((c, s, e))
        if idx > 0:
            seq[s:e] = segs[idx - 1][0]
        else:
            seq[s:e] = segs[idx + 1][0]


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


def compute_map(scores, labels, num_classes):
    """Per-class one-vs-rest AP (sklearn) + macro mAP over classes present in
    `labels`. Mirrors scripts/eval_grasp_map.py::compute_map -- same per-token
    softmax-score convention, ported here so SITL-phase (and any other
    ASFormer probe scored by this script) gets mAP alongside segmental F1/edit
    without a separate one-off script. Classes absent from `labels` are
    skipped from the macro mean (sklearn convention)."""
    valid = (labels >= 0) & (labels < num_classes)
    scores, labels = scores[valid], labels[valid]
    aps = []
    for c in range(num_classes):
        y_true = (labels == c).astype(np.int64)
        if y_true.sum() == 0:
            aps.append(float("nan"))
            continue
        aps.append(float(average_precision_score(y_true, scores[:, c])))
    aps = np.array(aps, dtype=np.float64)
    mAP = float(np.nanmean(aps))
    return mAP, aps.tolist()


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
    return ckpt.get("epoch", -1), ckpt


def load_finetuned_encoder(encoder, ckpt, device):
    """Load fine-tuned encoder state from a FT checkpoint. No-op for frozen probes."""
    if "encoder" not in ckpt:
        return False
    encoder.load_state_dict(ckpt["encoder"], strict=False)
    encoder.eval()
    print("Loaded fine-tuned encoder state from checkpoint.", flush=True)
    return True


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


def _save_resume_checkpoint(path, meta, paths_all, probs_all, labels_all, next_idx):
    """Atomic write (tmp file + rename) so a kill mid-save can't corrupt the
    checkpoint that a resumed run would otherwise trust."""
    tmp = path + ".tmp"
    torch.save({
        "meta": meta,
        "next_idx": next_idx,
        "paths_all": paths_all,
        "probs_all": probs_all,
        "labels_all": labels_all,
    }, tmp)
    os.replace(tmp, path)


def _load_resume_checkpoint(path, meta, n):
    """Returns (start_idx, paths_all, probs_all, labels_all) -- (0, [], [[]...], [])
    if no usable checkpoint exists (missing, corrupt, or meta/size mismatch
    against the current run -- e.g. a different checkpoint/yaml/head count, or
    a val set that changed size, would silently produce wrong results if
    resumed against)."""
    n_heads = meta["n_heads"]
    if not path or not os.path.exists(path):
        return 0, [], [[] for _ in range(n_heads)], []
    try:
        d = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"WARNING: resume cache at {path} unreadable ({e}); starting from scratch", flush=True)
        return 0, [], [[] for _ in range(n_heads)], []
    if d.get("meta") != meta:
        print(f"WARNING: resume cache at {path} meta mismatch {d.get('meta')} != {meta}; "
              f"starting from scratch", flush=True)
        return 0, [], [[] for _ in range(n_heads)], []
    next_idx = d["next_idx"]
    if next_idx > n:
        print(f"WARNING: resume cache next_idx={next_idx} > dataset len={n}; starting from scratch", flush=True)
        return 0, [], [[] for _ in range(n_heads)], []
    print(f"RESUMING from cached checkpoint {path}: {next_idx}/{n} clips already done", flush=True)
    return next_idx, d["paths_all"], d["probs_all"], d["labels_all"]


@torch.no_grad()
def run_inference(encoder, classifiers, dataset, device, use_bfloat16,
                   resume_cache=None, save_every=200, run_meta=None,
                   shard_indices=None, log_prefix=""):
    """Run the encoder once per clip and every classifier head on top of it.

    Returns per-clip *probabilities* per head (not just argmax) so callers can
    either evaluate a single head (backward-compatible) or average probabilities
    across heads BEFORE argmax/stitching to build a head-ensemble (must happen
    pre-argmax: segmental F1 is boundary-sensitive, so averaging post-hoc
    per-video label sequences, the way triplet's NPZ ensemble does, would not
    be equivalent here).

    If `resume_cache` is given, progress is periodically persisted (every
    `save_every` clips, atomically) to that path, and a matching checkpoint
    found there on startup is resumed from instead of restarting the full val
    pass -- this depends on `dataset[i]` ordering being stable/deterministic
    across the killed and resumed process (true here: VideoDataset iterates a
    fixed CSV with no shuffling), and on `run_meta` (checkpoint/yaml/head
    count/dataset size) matching exactly, checked by `_load_resume_checkpoint`.

    `shard_indices`, if given, restricts this call to that subset of dataset
    indices (round-robin across ranks for multi-node scoring -- see
    `main()`'s MPI split). Per-clip inference has no cross-clip state, so any
    partition is valid; results from all ranks are gathered and concatenated
    before stitching/scoring, so partition order doesn't matter.
    """
    encoder.eval()
    for c in classifiers:
        c.eval()
    n_heads = len(classifiers)
    n = len(dataset)
    idx_list = list(shard_indices) if shard_indices is not None else list(range(n))
    n_local = len(idx_list)
    meta = dict(run_meta or {})
    meta["n_heads"] = n_heads
    meta["n_clips"] = n_local
    start_i, paths_all, probs_all, labels_all = (
        _load_resume_checkpoint(resume_cache, meta, n_local) if resume_cache
        else (0, [], [[] for _ in range(n_heads)], [])
    )
    for local_i in range(start_i, n_local):
        i = idx_list[local_i]
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
        # Optional test-time augmentation (SEGF1_TTA=1): average softmax over the
        # original clip and its horizontal flip (label-preserving for action
        # segmentation). Each clip view is [B,C,T,H,W]; flip the W (last) dim.
        tta = os.environ.get("SEGF1_TTA", "0") == "1"
        clip_variants = [clips]
        if tta:
            clips_flip = [[v.flip(-1) for v in views] for views in clips]
            clip_variants.append(clips_flip)
        with amp_ctx:
            head_probs = [None] * n_heads
            for cv in clip_variants:
                outputs_per_view = encoder(cv, clip_idx_t)
                for hi, classifier in enumerate(classifiers):
                    head_outputs = [classifier(o) for o in outputs_per_view]
                    p = sum(F.softmax(o.float(), dim=-1) for o in head_outputs)
                    head_probs[hi] = p if head_probs[hi] is None else head_probs[hi] + p
        for hi in range(n_heads):
            probs_all[hi].append(
                head_probs[hi].reshape(-1, head_probs[hi].shape[-1]).cpu().numpy()
            )
        labels_np = (label.reshape(-1).cpu().numpy()
                     if isinstance(label, torch.Tensor) else
                     np.asarray(label).reshape(-1))
        paths_all.append(sample_path)
        labels_all.append(labels_np)
        if local_i % 50 == 0:
            print(f"{log_prefix}  inferred {local_i}/{n_local}", flush=True)
        if resume_cache and (local_i + 1) % save_every == 0:
            _save_resume_checkpoint(resume_cache, meta, paths_all, probs_all, labels_all, local_i + 1)
            print(f"{log_prefix}  checkpointed {local_i + 1}/{n_local} to {resume_cache}", flush=True)
    if resume_cache:
        _save_resume_checkpoint(resume_cache, meta, paths_all, probs_all, labels_all, n_local)
    return paths_all, probs_all, labels_all


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


def score_from_probs(probs_all, paths, labels, num_classes, bg_class, tag="",
                      min_seg_frames: int = 1):
    """argmax -> per-video stitch -> per-frame + segmental (F1@k, edit) metrics.

    `probs_all` is a list of per-clip probability arrays (one per clip, same
    order as `paths`/`labels`); argmax happens HERE so an ensemble's averaged
    probabilities get argmax'd (and then stitched) exactly like a single head's
    would, rather than averaging post-hoc label sequences.

    `min_seg_frames` (default 1 = off) runs `merge_short_segments` on each
    video's stitched prediction timeline before segmental (F1@k, edit) scoring
    only -- per-frame accuracy/macro-F1/mAP are computed on the raw,
    unsmoothed predictions, since those metrics were never the unstable ones.
    """
    preds = [p.argmax(axis=-1) for p in probs_all]
    flat_pred = np.concatenate(preds)
    flat_lab = np.concatenate(labels)
    flat_probs = np.concatenate(probs_all).astype(np.float32)
    frame_metrics = per_class_metrics(flat_pred, flat_lab, num_classes)
    print(f"[{tag}] per-frame: acc={frame_metrics['accuracy']*100:.2f} "
          f"macro_f1={frame_metrics['macro_f1']*100:.2f}", flush=True)

    mAP, per_class_ap = compute_map(flat_probs, flat_lab, num_classes)
    print(f"[{tag}] mAP={mAP*100:.2f}", flush=True)

    timelines = stitch_per_video(paths, preds, labels)
    print(f"[{tag}] stitched {len(timelines)} videos", flush=True)

    ks = [0.10, 0.25, 0.50]
    tot = {k: [0, 0, 0] for k in ks}
    edits = []
    for vid, (p, g) in timelines.items():
        if min_seg_frames > 1:
            p = merge_short_segments(p, min_seg_frames)
        for k in ks:
            tp, fp, fn = f1_at_k(p, g, k, bg_class=bg_class)
            tot[k][0] += tp; tot[k][1] += fp; tot[k][2] += fn
        edits.append(edit_score(p, g, bg_class=bg_class))

    seg_metrics: dict = {}
    for k in ks:
        tp, fp, fn = tot[k]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        label = f"F1@{int(k*100)}"
        seg_metrics[label] = {"precision": prec, "recall": rec, "f1": f1,
                              "tp": tp, "fp": fp, "fn": fn}
        print(f"  [{tag}] {label}: P={prec*100:.2f} R={rec*100:.2f} F1={f1*100:.2f}", flush=True)
    seg_metrics["edit_score"] = float(np.mean(edits)) if edits else 0.0
    seg_metrics["min_seg_frames"] = min_seg_frames
    print(f"  [{tag}] edit_score: {seg_metrics['edit_score']:.2f}", flush=True)

    return {
        "num_videos": len(timelines),
        "num_frames": int(sum(len(p) for p, _ in timelines.values())),
        "per_frame": frame_metrics,
        "segmental": seg_metrics,
        "map": {"mAP": mAP, "per_class_ap": per_class_ap},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True, help="rendered probe yaml")
    ap.add_argument("--checkpoint", required=True, help="best.pt or latest.pt")
    ap.add_argument("--out", default=None, help="optional JSON output path")
    ap.add_argument("--head-idx", type=int, default=0,
                    help="which classifier head to evaluate (default 0)")
    ap.add_argument("--bg-class", type=int, default=None,
                    help="optional background class to ignore in segmental metrics")
    ap.add_argument("--min-seg-frames", type=int, default=1,
                    help="merge predicted segments shorter than this many "
                         "frames into a neighbor before F1@k/edit scoring "
                         "(default 1 = off). Per-frame accuracy/macro-F1/mAP "
                         "are unaffected -- this only smooths the argmax "
                         "sequence used for segmental metrics, which are "
                         "otherwise hypersensitive to frame-level flicker "
                         "near phase boundaries (see the SITL-phase seed-"
                         "spread root-cause writeup). 16-24 (~1 clip window) "
                         "is the validated starting point.")
    ap.add_argument("--ensemble", action="store_true",
                    help="also score the mean-of-heads probability ensemble "
                         "(averaged BEFORE argmax/stitching) alongside every "
                         "individual head. NOTE: the printed 'BEST HEAD' value "
                         "is the max of --ensemble's per-head F1@10 draws using "
                         "the same TEST metric being reported -- useful as a "
                         "per-checkpoint diagnostic, but do NOT aggregate it "
                         "across seeds/checkpoints and report significance on "
                         "it (that is test-set peeking, not a valid statistic "
                         "-- see the sar-head-ensemble-result memory).")
    ap.add_argument("--dump-probs", default=None,
                    help="torch.save({paths,probs,probs_all_heads,labels}) "
                         "PER-CLIP probabilities (pre-argmax, pre-stitch) to "
                         "this path, for cross-seed ensembling by "
                         "scripts/aggregate_sar_seeds.py / "
                         "scripts/aggregate_sitl_seeds.py (mirrors triplet's "
                         "dump_probs_path mechanism). `probs` is --head-idx's "
                         "probabilities (back-compat key); `probs_all_heads` "
                         "is every head's probabilities, needed to reproduce "
                         "the within-checkpoint multi-head test-time ensemble "
                         "when re-scoring with --min-seg-frames. Independent "
                         "of --ensemble.")
    ap.add_argument("--resume-cache", default=None,
                    help="path to periodically checkpoint raw inference "
                         "progress (paths/probs/labels + next index), so a "
                         "walltime kill mid-val-pass can resume instead of "
                         "restarting from clip 0. Re-running with the same "
                         "path + same --yaml/--checkpoint/--head-idx resumes "
                         "automatically; a meta mismatch (different "
                         "checkpoint, dataset size, or head count) is "
                         "detected and ignored rather than trusted.")
    ap.add_argument("--resume-save-every", type=int, default=200,
                    help="checkpoint --resume-cache every N clips (default 200)")
    args = ap.parse_args()

    # Multi-rank clip sharding (opt-in via mpiexec -- plain single-process
    # invocation is unaffected). Inference has no cross-clip state, so each
    # rank scores a disjoint round-robin slice of the val set and rank 0
    # gathers + stitches/scores the union -- no torch.distributed process
    # group, no CCL/xccl backend selection, none of the XPU multi-node
    # device_id/DataLoader-worker hangs documented in src/utils/distributed.py
    # apply here, because there is no collective besides this one gather.
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, world_size = comm.Get_rank(), comm.Get_size()
    except Exception:
        comm, rank, world_size = None, 0, 1
    log_prefix = f"[rank{rank}/{world_size}]" if world_size > 1 else ""

    with open(args.yaml) as f:
        cfg = yaml.safe_load(f)

    device = _pick_device()
    print(f"{log_prefix} device: {device}", flush=True)
    cpu_bad = device.type == "cpu" and os.environ.get("SEGF1_ALLOW_CPU") != "1"
    if comm is not None and world_size > 1:
        cpu_bad = comm.allreduce(cpu_bad, op=MPI.LOR)
    if cpu_bad:
        if rank == 0:
            print("FATAL: no XPU/CUDA device found — refusing to run full val inference "
                  "on CPU (set SEGF1_ALLOW_CPU=1 to override). Check ZE_FLAT_DEVICE_HIERARCHY "
                  "/ frameworks module / ZE_AFFINITY_MASK.", flush=True)
        sys.exit(2)
    encoder, classifiers = build_encoder_and_heads(cfg, device)
    epoch, ckpt = load_classifier_weights(classifiers, args.checkpoint, device)
    load_finetuned_encoder(encoder, ckpt, device)
    if rank == 0:
        print(f"loaded classifiers from epoch {epoch}", flush=True)

    dataset = build_val_dataset(cfg)
    n = len(dataset)
    if rank == 0:
        print(f"val dataset: {n} clips across {world_size} rank(s)", flush=True)
    use_bf16 = cfg["experiment"]["optimization"].get("use_bfloat16", True)
    num_classes = cfg["experiment"]["data"]["num_classes"]

    shard_indices = list(range(rank, n, world_size)) if world_size > 1 else None
    # Per-rank resume cache: sharing one path across ranks would have every
    # rank clobber the same file with its own (disjoint) subset.
    resume_cache = args.resume_cache
    if resume_cache and world_size > 1:
        resume_cache = f"{resume_cache}.rank{rank}"

    # Always run every head's forward pass (cheap relative to the encoder, which
    # dominates cost and only runs once per clip regardless of --ensemble).
    run_meta = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "yaml": os.path.abspath(args.yaml),
        "epoch": epoch,
    }
    t0 = time.time()
    paths, probs_all_heads, labels = run_inference(
        encoder, classifiers, dataset, device, use_bf16,
        resume_cache=resume_cache, save_every=args.resume_save_every,
        run_meta=run_meta, shard_indices=shard_indices, log_prefix=log_prefix,
    )
    print(f"{log_prefix} local shard done: {len(paths)} clips in {time.time() - t0:.1f}s",
          flush=True)

    if world_size > 1:
        gathered = comm.gather((paths, probs_all_heads, labels), root=0)
        if rank != 0:
            return
        n_heads = len(classifiers)
        paths, labels = [], []
        probs_all_heads = [[] for _ in range(n_heads)]
        for g_paths, g_probs, g_labels in gathered:
            paths.extend(g_paths)
            labels.extend(g_labels)
            for hi in range(n_heads):
                probs_all_heads[hi].extend(g_probs[hi])
        print(f"gathered {len(paths)}/{n} clips from {world_size} ranks", flush=True)

    if args.dump_probs:
        os.makedirs(os.path.dirname(args.dump_probs) or ".", exist_ok=True)
        torch.save({
            "paths": paths,
            "probs": probs_all_heads[args.head_idx],  # list of [T, C] per-clip arrays (back-compat)
            "probs_all_heads": probs_all_heads,  # list (per head) of lists of [T, C] arrays
            "labels": labels,
            "head_idx": args.head_idx,
            "checkpoint": args.checkpoint,
        }, args.dump_probs)
        print(f"dumped {len(probs_all_heads)} head(s) probs to {args.dump_probs}", flush=True)

    if not args.ensemble:
        result = score_from_probs(
            probs_all_heads[args.head_idx], paths, labels, num_classes,
            args.bg_class, tag=f"head{args.head_idx}",
            min_seg_frames=args.min_seg_frames,
        )
        out = {
            "checkpoint": args.checkpoint, "yaml": args.yaml, "epoch": epoch,
            "bg_class": args.bg_class, **result,
        }
    else:
        n_heads = len(classifiers)
        per_head = []
        for hi in range(n_heads):
            r = score_from_probs(probs_all_heads[hi], paths, labels, num_classes,
                                  args.bg_class, tag=f"head{hi}",
                                  min_seg_frames=args.min_seg_frames)
            per_head.append(r)
        best_hi = max(range(n_heads), key=lambda hi: per_head[hi]["segmental"]["F1@10"]["f1"])

        ens_probs = [
            sum(probs_all_heads[hi][ci] for hi in range(n_heads)) / n_heads
            for ci in range(len(paths))
        ]
        ens_result = score_from_probs(ens_probs, paths, labels, num_classes,
                                       args.bg_class, tag="ensemble",
                                       min_seg_frames=args.min_seg_frames)

        best_f1 = per_head[best_hi]["segmental"]["F1@10"]["f1"]
        ens_f1 = ens_result["segmental"]["F1@10"]["f1"]
        print("\n" + "=" * 60, flush=True)
        print(f"BEST HEAD {best_hi}  ->  F1@10 = {best_f1*100:.2f}", flush=True)
        tag = "ENSEMBLE > best-head" if ens_f1 > best_f1 else "best-head >= ENSEMBLE"
        print(f"ENSEMBLE      ->  F1@10 = {ens_f1*100:.2f}   ({tag}, Δ={(ens_f1-best_f1)*100:+.2f})",
              flush=True)
        print("=" * 60, flush=True)

        out = {
            "checkpoint": args.checkpoint, "yaml": args.yaml, "epoch": epoch,
            "bg_class": args.bg_class,
            "per_head": per_head, "best_head": best_hi,
            "ensemble": ens_result,
        }

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
