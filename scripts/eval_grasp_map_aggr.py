"""GraSP mAP re-score under DIFFERENT prediction aggregations — no retrain, no encoder.

Reads the SAME exported val feature cache + trained heads as eval_grasp_map_cached.py,
but reports mAP under three aggregations so we can quantify the TAPIS-vs-ours
protocol disparity (overlapping windows / all-token flattening):

  A. flatten-all   — the CURRENT scorer: every 24 tokens x every overlapping
                     window flattened into one pool (baseline; should reproduce
                     eval_grasp_map_cached.py).
  B. center-clip   — keep ONLY the center clip's 8 tokens (tokens [8:16] of the
                     3-clip / 24-token ctx3 layout). Context clips (0-7, 16-23)
                     have truncated temporal context; TAPIS scores a centered
                     keyframe. Isolates "do edge/context tokens drag mAP down?".
  C. time-dedup    — map each token to its absolute clip-frame position
                     (start_frame + 2*global_token_idx) within its CASE, average
                     the softmax probs of ALL windows/tokens landing on the same
                     (case, frame), then AP over unique keyframes. Closest
                     apples-to-apples with TAPIS's one-prediction-per-keyframe.

Token layout (ctx3): 24 tokens = [clip0:0-7 | clip1:8-15 | clip2:16-23], clip-major
and temporally ordered; tubelet_size=2 -> 2 frames/token over the 48-frame span;
consecutive windows stride 12 frames (= 6 tokens), so overlapping windows' token
grids align on shared absolute frames. The dedup prints the collapse ratio as a
self-check: a correct mapping collapses the pool ~4x (48-frame span / 12-frame stride).
"""
import argparse
import os
import re
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets.backbone_feature_cache import make_backbone_feature_cache, read_cache_pooled
from scripts.eval_grasp_map_cached import build_heads

FRAMES_PER_TOKEN = 2  # tubelet_size=2
# Old ctx3 clips are named `frame_<start_sec>_ctx3.mp4`; the official-frames
# rebuild (scripts/build_grasp_ctx_official.py) names them `kf_<center_rank>_ctx.mp4`.
# Match either -- an unanchored miss used to fall back to start=0 for EVERY window,
# which silently collapses the dedup key space instead of erroring.
_START_RE = re.compile(r"(?:frame|kf)_0*(\d+)")
_CASE_RE = re.compile(r"(CASE\d+)")


def _map(scores, labels, num_classes):
    valid = (labels >= 0) & (labels < num_classes)
    scores, labels = scores[valid], labels[valid]
    aps = []
    for c in range(num_classes):
        yt = (labels == c).astype(np.int64)
        aps.append(float("nan") if yt.sum() == 0 else float(average_precision_score(yt, scores[:, c])))
    return float(np.nanmean(aps)), np.array(aps)


def _parse_key(path):
    """(case_id, start_frame) from .../CASE041/frame_000013_ctx3.mp4."""
    p = path if isinstance(path, str) else str(path)
    case = _CASE_RE.search(p)
    start = _START_RE.search(os.path.basename(p))
    if start is None:
        # Fail loudly: a silent start=0 keys every window to the same frame and
        # makes the dedup aggregations quietly meaningless.
        raise ValueError(f"could not parse a start/center index from clip name: {p}")
    return (case.group(1) if case else p), int(start.group(1))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data = cfg["experiment"]["data"]
    num_classes = data["num_classes"]
    use_bf16 = bool(cfg["experiment"]["optimization"].get("use_bfloat16", True))
    val_cache_root = data["val_cache_root"]
    device = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available()
                          else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = args.ckpt or os.path.join(cfg["folder"], "video_classification_frozen", cfg["tag"], "best.pt")
    print(f"[aggr] device={device} ckpt={ckpt}\n[aggr] val_cache_root={val_cache_root}", flush=True)

    pooled = read_cache_pooled(val_cache_root)
    spatial_prepooled = pooled == "mean"
    _, loader, _ = make_backbone_feature_cache(
        cache_root=val_cache_root, batch_size=args.batch_size, training=False,
        rank=0, world_size=1, num_workers=data.get("cache_num_workers", 8),
        pin_mem=True, persistent_workers=False, require_complete_export=True,
    )
    first = next(iter(loader))
    embed_dim = first[0].shape[-1]
    heads = build_heads(cfg, embed_dim, spatial_prepooled, device)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    for h, sd in zip(heads, ck["classifiers"]):
        h.load_state_dict(sd)
    n_heads = len(heads)
    print(f"[aggr] loaded {n_heads} heads from ckpt epoch={ck.get('epoch', -1)}", flush=True)

    # Per-head, accumulate probs + labels + (case,frame) key for every token.
    probs_h = [[] for _ in range(n_heads)]
    labels_all, keys_all = [], []
    T_total = None
    for itr, dbatch in enumerate(loader):
        feats = dbatch[0].to(device, non_blocking=True)   # [B, V, NC, TS, D]
        labels = dbatch[1]                                # [B, T]
        paths = dbatch[3]                                 # list[str] length B
        B, T = labels.shape
        T_total = T
        with torch.amp.autocast(device.type, dtype=torch.bfloat16 if use_bf16 else torch.float16, enabled=use_bf16):
            views = [feats[:, v] for v in range(feats.shape[1])]
            for hi, head in enumerate(heads):
                p = sum(F.softmax(head(v).float(), dim=-1) for v in views) / len(views)  # [B, T, C]
                probs_h[hi].append(p.cpu().numpy())
        labels_all.append(labels.numpy())
        # (case, absolute_frame) per token
        batch_keys = np.empty((B, T), dtype=object)
        for b in range(B):
            case, start = _parse_key(paths[b])
            for t in range(T):
                batch_keys[b, t] = (case, start + FRAMES_PER_TOKEN * t)
        keys_all.append(batch_keys)
        if itr % 50 == 0:
            print(f"[aggr] itr {itr}/{len(loader)}", flush=True)

    labels = np.concatenate(labels_all, axis=0)                    # [N, T]
    keys = np.concatenate(keys_all, axis=0)                        # [N, T] object
    probs = [np.concatenate(ph, axis=0) for ph in probs_h]         # each [N, T, C]
    N = labels.shape[0]
    print(f"[aggr] {N} windows x {T_total} tokens", flush=True)

    def report(tag, sel_slice=None, dedup=False):
        print(f"\n{'='*60}\n[aggr] AGGREGATION: {tag}", flush=True)
        best = (-1.0, -1)
        per_head_ap = []
        for hi in range(n_heads):
            p = probs[hi]
            lab = labels
            if sel_slice is not None:
                p = p[:, sel_slice, :]
                lab = labels[:, sel_slice]
            if dedup:
                k = keys[:, sel_slice] if sel_slice is not None else keys
                flat_k = k.reshape(-1)
                flat_p = p.reshape(-1, p.shape[-1])
                flat_l = lab.reshape(-1)
                # group by key: average probs, take (consistent) label
                order = {}
                for i, key in enumerate(flat_k):
                    order.setdefault(key, []).append(i)
                uniq = list(order.keys())
                dp = np.zeros((len(uniq), flat_p.shape[-1]), dtype=np.float32)
                dl = np.zeros(len(uniq), dtype=np.int64)
                for j, key in enumerate(uniq):
                    idx = order[key]
                    dp[j] = flat_p[idx].mean(0)
                    dl[j] = flat_l[idx[0]]  # label constant per keyframe
                sc, y = dp, dl
                if hi == 0:
                    print(f"[aggr]   dedup: {flat_p.shape[0]} -> {len(uniq)} unique "
                          f"keyframes (collapse {flat_p.shape[0]/max(1,len(uniq)):.2f}x)", flush=True)
            else:
                sc = p.reshape(-1, p.shape[-1]).astype(np.float32)
                y = lab.reshape(-1).astype(np.int64)
            mAP, aps = _map(sc, y, num_classes)
            per_head_ap.append((mAP, aps))
            print(f"[aggr]   head {hi}: mAP={mAP*100:.2f}", flush=True)
            if mAP > best[0]:
                best = (mAP, hi)
        # ensemble
        ens_sc = None
        return best, per_head_ap

    # "Center clip" = the middle clip's tokens. The old hardcoded T//3 assumed the
    # 3-clip ctx3 layout; at 4 clips it selects tokens 10-20, straddling two clips.
    # Derive from the actual clip count instead. With an even clip count there is
    # no single middle clip, so take the two central ones.
    n_clips = cfg["experiment"]["classifier"]["asformer_kwargs"].get(
        "num_clips", cfg["experiment"]["data"]["num_segments"]
    )
    tpc = T_total // n_clips
    if n_clips % 2:
        center = slice((n_clips // 2) * tpc, (n_clips // 2 + 1) * tpc)
    else:
        center = slice((n_clips // 2 - 1) * tpc, (n_clips // 2 + 1) * tpc)
    print(f"[aggr] n_clips={n_clips} tokens/clip={tpc} -> center tokens "
          f"[{center.start}:{center.stop}]", flush=True)
    bA, _ = report("A. flatten-all (current scorer)")
    bB, _ = report("B. center-clip only", sel_slice=center)
    bC, _ = report("C. time-dedup (all tokens -> unique keyframes)", dedup=True)
    bD, _ = report("D. center-clip + time-dedup", sel_slice=center, dedup=True)

    print(f"\n{'='*60}\n[aggr] SUMMARY (best-head mAP):")
    print(f"[aggr]   A flatten-all      = {bA[0]*100:.2f}  (head {bA[1]})")
    print(f"[aggr]   B center-clip      = {bB[0]*100:.2f}  (head {bB[1]})")
    print(f"[aggr]   C time-dedup       = {bC[0]*100:.2f}  (head {bC[1]})")
    print(f"[aggr]   D center+dedup     = {bD[0]*100:.2f}  (head {bD[1]})")
    print(f"[aggr]   TAPIS SOTA         = 76.72 (v3; 76.07 v2)")
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
