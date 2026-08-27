#!/usr/bin/env python3
"""Aggregate the 3 SITL-phase seed dumps (job 8784451's patience=6/2-head
production probe) into mean±std per metric, and score the cross-seed
probability-ensemble.

Adapted from scripts/aggregate_sar_seeds.py -- same discipline, generalized
to report every metric this campaign was asked to record (F1@10/25/50,
edit_score, mAP), not just F1@10:
  - Each seed is scored ALONE, using its own within-checkpoint multi-head
    test-time ensemble (mean-of-heads probs, matching the "ensemble" field
    already reported per-seed in segf1_seed{N}.json) -- mean±std across
    seeds, no cherry-picking.
  - The cross-seed ensemble is a FIXED probability-averaging rule (mean of
    each seed's already-head-ensembled probs) that never looks at any target
    metric to decide anything.
This script deliberately does NOT report a "best seed" number for any metric
as though it were an aggregate statistic -- selecting the max of N seeds'
draws by the exact metric being reported is test-set peeking, not a valid
statistic (see the sar-head-ensemble-result / aggregate_sar_seeds.py
precedent, caught and retracted once already on this exact axis).

Requires all seeds' dumped `paths` to align exactly (same val clip set)
before averaging -- asserted, not assumed (align_by_path returns None on any
mismatch and the ensemble step is skipped, not silently computed on a
misaligned subset).

Usage:
  python aggregate_sitl_seeds.py --dumps probs_seed0.pt probs_seed1.pt probs_seed2.pt \\
      [--num-classes 12] [--json-out out.json]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from eval_segmental_f1 import score_from_probs  # noqa: E402

METRICS = ["accuracy", "macro_f1", "F1@10", "F1@25", "F1@50", "edit_score", "map"]


def load_dump(path):
    """Returns (paths, probs, labels) where `probs` is the within-checkpoint
    multi-head test-time ensemble (mean over heads) when the dump has
    `probs_all_heads` (current format); falls back to the single dumped head
    (`probs`, old-format dumps) with a loud warning, since that does NOT
    match the headline per-seed numbers already reported (those are the
    head-ensemble, not head-idx 0 alone)."""
    d = torch.load(path, map_location="cpu", weights_only=False)
    if "probs_all_heads" in d and len(d["probs_all_heads"]) > 1:
        heads = d["probs_all_heads"]
        n_heads = len(heads)
        n_clips = len(d["paths"])
        probs = [
            sum(heads[hi][ci] for hi in range(n_heads)) / n_heads
            for ci in range(n_clips)
        ]
    else:
        print(f"WARNING: {path} has no multi-head dump (old format) -- "
              f"using head{d.get('head_idx', 0)} alone, will NOT match the "
              f"reported head-ensemble numbers for this seed", flush=True)
        probs = d["probs"]
    return d["paths"], probs, d["labels"]


def align_by_path(dumps):
    """Reorders every dump's probs/labels to the first dump's path order.
    Returns None if any dump's path SET differs from the reference -- a
    mismatch means the seeds scored different val orderings/subsets and
    averaging would silently mix mislabeled clips."""
    ref_paths = dumps[0][0]
    ref_set = set(ref_paths)
    aligned = []
    for paths, probs, labels in dumps:
        if set(paths) != ref_set:
            return None
        idx = {p: i for i, p in enumerate(paths)}
        order = [idx[p] for p in ref_paths]
        aligned.append((list(ref_paths),
                        [probs[i] for i in order],
                        [labels[i] for i in order]))
    return aligned


def extract_metric(result, name):
    if name == "accuracy":
        return result["per_frame"]["accuracy"] * 100.0
    if name == "macro_f1":
        return result["per_frame"]["macro_f1"] * 100.0
    if name == "edit_score":
        return result["segmental"]["edit_score"]
    if name == "map":
        return result["map"]["mAP"] * 100.0
    return result["segmental"][name]["f1"] * 100.0


def ensemble_score(aligned, num_classes, bg_class, min_seg_frames=1):
    ref_paths, ref_labels = aligned[0][0], aligned[0][2]
    n_clips = len(ref_paths)
    ens_probs = [
        sum(aligned[si][1][ci] for si in range(len(aligned))) / len(aligned)
        for ci in range(n_clips)
    ]
    return score_from_probs(ens_probs, ref_paths, ref_labels, num_classes, bg_class,
                            tag="sitl_seed_ensemble", min_seg_frames=min_seg_frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", required=True,
                    help="paths to probs_seed{N}.pt dumps (from "
                         "eval_segmental_f1.py --dump-probs), one per seed")
    ap.add_argument("--num-classes", type=int, default=12)
    ap.add_argument("--bg-class", type=int, default=None)
    ap.add_argument("--min-seg-frames", type=int, default=1,
                    help="merge predicted segments shorter than this many "
                         "frames into a neighbor before F1@k/edit scoring "
                         "(default 1 = off); per-frame accuracy/macro-F1/mAP "
                         "are unaffected. See eval_segmental_f1.py's "
                         "--min-seg-frames for the root-cause writeup; "
                         "16-24 (~1 clip window) is the validated setting.")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    dumps = [load_dump(p) for p in args.dumps]
    per_seed = []
    for p, d in zip(args.dumps, dumps):
        r = score_from_probs(d[1], d[0], d[2], args.num_classes, args.bg_class,
                              tag=os.path.basename(p),
                              min_seg_frames=args.min_seg_frames)
        per_seed.append(r)

    out = {"seeds": args.dumps, "per_seed": {}, "mean_std": {}}
    print(f"\n{'metric':<12} {'per-seed':>28}   {'mean±std (no peeking)':>22}")
    print("-" * 70)
    for m in METRICS:
        vals = [extract_metric(r, m) for r in per_seed]
        mean, std = float(np.mean(vals)), float(np.std(vals))
        out["per_seed"][m] = vals
        out["mean_std"][m] = {"mean": mean, "std": std}
        vals_str = "/".join(f"{v:.2f}" for v in vals)
        print(f"{m:<12} {vals_str:>28}   {f'{mean:6.2f}±{std:5.2f}':>22}")

    aligned = align_by_path(dumps)
    if aligned is None:
        print("\nensemble: SKIPPED (path mismatch across seed dumps)")
        out["ensemble"] = None
    else:
        ens = ensemble_score(aligned, args.num_classes, args.bg_class,
                             min_seg_frames=args.min_seg_frames)
        print(f"\n{'metric':<12} {'seed-ensemble':>14}   {'vs mean-of-seeds':>20}")
        print("-" * 55)
        ens_out = {}
        for m in METRICS:
            ens_val = extract_metric(ens, m)
            delta = ens_val - out["mean_std"][m]["mean"]
            ens_out[m] = ens_val
            print(f"{m:<12} {ens_val:14.2f}   Δ={delta:+.2f}")
        out["ensemble"] = ens_out

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
