#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Measure content overlap between two video sets via DCT pHash.

Purpose: decide whether surgenet_laparoscopic (3532 uncleaned YouTube clips on
eagle) is worth transferring+cleaning, or whether LEMON (a newer, larger,
already-cleaned YouTube scrape) already subsumes it. Both are YouTube-sourced so
overlap is expected; this quantifies the DISTINCT fraction.

Two phases:
  build-ref : hash a REFERENCE set into a pooled hash file. Sources:
              --from-staging <dir>  (WebDataset tars, e.g. lemon_staging/*.tar)
              --from-tree    <dir>  (loose *.mp4 tree, e.g. surgenet_laparoscopic/)
  gate      : hash a QUERY set (--from-tree/--from-staging) and, per source
              video, report dup fraction vs the reference. A query video is
              "covered" if >threshold of its sampled frames match the ref.

Parallel via --start/--end over the (sorted) source list + --finalize merge.
Reuses scripts/phash_util.py. Frame pHash catches re-encode/rescale, NOT crop
(same caveat as the LEMON gate).

This is a MEASUREMENT tool (reports overlap); it does not move/delete data.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import tarfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phash_util as ph  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["build-ref", "gate", "finalize-ref", "finalize-gate"],
                   required=True)
    p.add_argument("--from-staging", default=None, help="WebDataset tar dir (source tars).")
    p.add_argument("--from-tree", default=None, help="Loose *.mp4 tree (recursive).")
    p.add_argument("--ref", default=None, help="Reference hash json (build-ref out / gate in).")
    p.add_argument("--partial-dir", default=None, help="Dir for parallel partials.")
    p.add_argument("--out", default=None, help="gate: summary json out.")
    p.add_argument("--frames-per-video", type=int, default=16,
                   help="Frames sampled per source video (both ref and query).")
    p.add_argument("--threshold", type=float, default=0.10,
                   help="gate: >this frac of a query video's frames matching ref = covered.")
    p.add_argument("--start", type=int, default=None, help="Source-list slice start.")
    p.add_argument("--end", type=int, default=None, help="Source-list slice end (excl).")
    return p.parse_args()


def _list_sources(args):
    """Return sorted list of (source_name, kind, locator).

    staging: each tar is one source video -> (stem, 'tar', tar_path).
    tree:    each *.mp4 is one source video -> (relpath, 'file', file_path).
    """
    if args.from_staging:
        tars = sorted(glob.glob(os.path.join(args.from_staging, "*.tar")))
        return [(os.path.basename(t)[:-4], "tar", t) for t in tars]
    if args.from_tree:
        vids = []
        for root, _, files in os.walk(args.from_tree):
            for fn in files:
                if fn.lower().endswith((".mp4", ".avi", ".mkv", ".mov", ".m4v")):
                    fp = os.path.join(root, fn)
                    rel = os.path.relpath(fp, args.from_tree)
                    vids.append((rel, "file", fp))
        vids.sort()
        return vids
    raise SystemExit("need --from-staging or --from-tree")


def _hashes_for_source(kind, locator, fpv):
    """Sample+hash frames for one source video (tar of clips, or a single file)."""
    hs = []
    if kind == "tar":
        with tarfile.open(locator, "r|") as tf:
            for m in tf:
                if not (m.isfile() and m.name.endswith(".mp4")):
                    continue
                try:
                    buf = tf.extractfile(m).read()
                    for fr in ph.sample_gray_frames(io.BytesIO(buf), n=fpv, seek=True):
                        hs.append(ph.phash_gray(fr))
                except Exception:
                    continue
    else:
        try:
            for fr in ph.sample_gray_frames(locator, n=fpv, seek=True):
                hs.append(ph.phash_gray(fr))
        except Exception:
            pass
    return hs


def _pdir(args):
    return args.partial_dir or ((args.ref or args.out) + ".partials")


def build_ref(args):
    srcs = _list_sources(args)
    lo = args.start or 0
    hi = args.end if args.end is not None else len(srcs)
    sel = srcs[lo:hi]
    print(f"[build-ref] {len(srcs)} sources; this worker [{lo}:{hi}) -> {len(sel)}", flush=True)
    hs = []
    for i, (name, kind, loc) in enumerate(sel):
        hs.extend(_hashes_for_source(kind, loc, args.frames_per_video))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(sel)} -> {len(hs)} hashes", flush=True)
    pdir = _pdir(args)
    os.makedirs(pdir, exist_ok=True)
    pp = os.path.join(pdir, f"ref_{lo}_{hi}.json")
    with open(pp, "w") as f:
        json.dump({"hashes": [f"{h:016x}" for h in hs]}, f)
    print(f"[build-ref] {len(hs)} hashes -> {pp}", flush=True)


def finalize_ref(args):
    pdir = _pdir(args)
    hs = []
    for pp in sorted(glob.glob(os.path.join(pdir, "ref_*.json"))):
        hs.extend(int(h, 16) for h in json.load(open(pp))["hashes"])
    ph.save_ref(args.ref, hs, [], {"n_frames": len(hs), "role": "overlap-ref"})
    d = json.load(open(args.ref))
    print(f"[finalize-ref] {len(hs)} frames -> {len(d['eval_hashes'])} unique -> {args.ref}",
          flush=True)


def gate(args):
    ref, _, meta = ph.load_ref(args.ref)  # ref stored in eval_hashes slot
    srcs = _list_sources(args)
    lo = args.start or 0
    hi = args.end if args.end is not None else len(srcs)
    sel = srcs[lo:hi]
    print(f"[gate] ref {ref.size} hashes; {len(sel)} query sources [{lo}:{hi}); thr={args.threshold}",
          flush=True)
    results = []
    covered = 0
    for name, kind, loc in sel:
        hs = _hashes_for_source(kind, loc, args.frames_per_video)
        if not hs:
            results.append({"source": name, "dup": None, "n": 0})
            continue
        cands = list(hs)
        d = ph.dup_fraction(cands, ref)
        cov = d > args.threshold
        if cov:
            covered += 1
        results.append({"source": name, "dup": round(d, 4), "n": len(hs), "covered": cov})
    pdir = _pdir(args)
    os.makedirs(pdir, exist_ok=True)
    pp = os.path.join(pdir, f"gate_{lo}_{hi}.json")
    with open(pp, "w") as f:
        json.dump({"range": [lo, hi], "n": len(sel), "covered": covered, "results": results}, f)
    print(f"[gate] [{lo}:{hi}): {covered}/{len(sel)} covered -> {pp}", flush=True)


def finalize_gate(args):
    pdir = _pdir(args)
    allr = []
    for pp in sorted(glob.glob(os.path.join(pdir, "gate_*.json"))):
        allr.extend(json.load(open(pp))["results"])
    scored = [r for r in allr if r.get("dup") is not None]
    covered = [r for r in scored if r.get("covered")]
    distinct = [r for r in scored if not r.get("covered")]
    import numpy as np
    dups = np.array([r["dup"] for r in scored]) if scored else np.array([0.0])
    summ = {
        "total_query_videos": len(allr),
        "scored": len(scored),
        "unreadable": len(allr) - len(scored),
        "threshold": args.threshold,
        "covered_by_ref": len(covered),
        "distinct_from_ref": len(distinct),
        "pct_distinct": round(100 * len(distinct) / max(1, len(scored)), 1),
        "dup_stats": {"mean": round(float(dups.mean()), 3),
                      "median": round(float(np.median(dups)), 3),
                      "p90": round(float(np.percentile(dups, 90)), 3)},
        "top20_most_distinct": sorted(distinct, key=lambda r: r["dup"])[:20],
    }
    with open(args.out, "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps({k: summ[k] for k in
                      ["total_query_videos", "covered_by_ref", "distinct_from_ref",
                       "pct_distinct", "dup_stats"]}, indent=2), flush=True)
    print(f"-> {args.out}", flush=True)


def main():
    args = parse_args()
    t0 = time.time()
    {"build-ref": build_ref, "gate": gate,
     "finalize-ref": finalize_ref, "finalize-gate": finalize_gate}[args.mode](args)
    print(f"elapsed {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
