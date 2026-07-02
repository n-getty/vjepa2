#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Scrub eval-leaked source videos from a training set (crop-ROBUST).

Our triplet eval (yt_robotic_chole) was carved from the same YouTube robotic-
chole scrape our training data draws from. To keep future results doubt-free,
drop any training source video that overlaps the eval — using a CROP-AUGMENTED
eval reference so LEMON's cropped/masked copies are caught too (raw frame pHash
is crop-blind; see phash_util.phash_variants).

Modes:
  build-eval-ref : hash the eval clips WITH crop/mask variants -> eval_ref json.
                   Source = yt_chole_tool_windows/yt_robotic_chole_Batch*/*.mp4.
                   Parallel by --batch-start/-end; --finalize merges partials.
  gate-staging   : gate a WebDataset staging dir (one tar per source video) vs
                   the eval ref; move leaked source tars to <staging>/_evalleak/.
  gate-tree      : gate a loose *.mp4 tree; write a drop-list json of leaked files
                   (caller removes / excludes them). Per-subdir tallies.
  gate-shards    : gate a resharded dir (many source videos per shard) by hashing
                   each clip and reporting per-SOURCE-VIDEO leak (needs re-reshard
                   to actually drop — reports the leaked source-video ids).

A source is "leaked" if >--threshold of its sampled frames match the eval ref
within Hamming 6/64. Query side hashes the RAW frame (the reference carries the
crop/mask variants). Reports are auditable JSON; moves are reversible.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import sys
import tarfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phash_util as ph  # noqa: E402

SRC_RE = re.compile(r"^(?P<prefix>.+?)__(?P<source>.+)_clip_(?P<clip>\d+)$")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", required=True,
                   choices=["build-eval-ref", "finalize-eval-ref",
                            "gate-staging", "gate-tree", "gate-shards"])
    p.add_argument("--eval-dir", default=None, help="yt_chole_tool_windows dir.")
    p.add_argument("--eval-ref", default=None, help="Crop-augmented eval ref json.")
    p.add_argument("--partial-dir", default=None)
    p.add_argument("--staging", default=None, help="gate-staging: dir of <prefix>__<src>.tar.")
    p.add_argument("--tree", default=None, help="gate-tree: loose *.mp4 root.")
    p.add_argument("--shards", default=None, help="gate-shards: resharded dir of *.tar.")
    p.add_argument("--out", default=None, help="gate-*: report json.")
    p.add_argument("--threshold", type=float, default=0.10)
    p.add_argument("--eval-frames-per-clip", type=int, default=4)
    p.add_argument("--frames-per-video", type=int, default=16)
    p.add_argument("--batch-start", type=int, default=None)
    p.add_argument("--batch-end", type=int, default=None)
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--apply", action="store_true",
                   help="gate-staging: actually move leaked tars (else dry-run).")
    return p.parse_args()


def _pdir(args):
    return args.partial_dir or ((args.eval_ref or args.out) + ".partials")


# ---------- build crop-augmented eval reference ----------
def build_eval_ref(args):
    batches = sorted(glob.glob(os.path.join(args.eval_dir, "yt_robotic_chole_Batch*")))
    lo = args.batch_start or 0
    hi = args.batch_end if args.batch_end is not None else len(batches)
    sel = batches[lo:hi]
    print(f"[eval-ref] {len(batches)} batches; [{lo}:{hi}) -> {len(sel)}", flush=True)
    hs = []
    for bd in sel:
        clips = sorted(glob.glob(os.path.join(bd, "*.mp4")))
        n0 = len(hs)
        for cp in clips:
            try:
                for fr in ph.sample_gray_frames(cp, n=args.eval_frames_per_clip, seek=False):
                    hs.extend(ph.phash_variants(fr))   # <-- crop/mask augmented
            except Exception:
                continue
        print(f"  {os.path.basename(bd)}: {len(clips)} clips -> {len(hs)-n0} hashes", flush=True)
    pdir = _pdir(args)
    os.makedirs(pdir, exist_ok=True)
    pp = os.path.join(pdir, f"evalref_{lo}_{hi}.json")
    with open(pp, "w") as f:
        json.dump({"hashes": [f"{h:016x}" for h in hs]}, f)
    print(f"[eval-ref] {len(hs)} hashes -> {pp}", flush=True)


def finalize_eval_ref(args):
    pdir = _pdir(args)
    hs = []
    for pp in sorted(glob.glob(os.path.join(pdir, "evalref_*.json"))):
        hs.extend(int(h, 16) for h in json.load(open(pp))["hashes"])
    ph.save_ref(args.eval_ref, hs, [], {"n_frames": len(hs), "role": "eval-aug-ref",
                                        "augmented": True})
    d = json.load(open(args.eval_ref))
    print(f"[finalize] {len(hs)} -> {len(d['eval_hashes'])} unique -> {args.eval_ref}", flush=True)


# ---------- gating ----------
def _clip_hashes(buf_or_path, n):
    return [ph.phash_gray(f) for f in ph.sample_gray_frames(buf_or_path, n=n, seek=True)]


def gate_staging(args):
    ev, _, _ = ph.load_ref(args.eval_ref)
    tars = sorted(glob.glob(os.path.join(args.staging, "*.tar")))
    lo = args.start or 0
    hi = args.end if args.end is not None else len(tars)
    sel = tars[lo:hi]
    leakdir = os.path.join(args.staging, "_evalleak")
    if args.apply:
        os.makedirs(leakdir, exist_ok=True)
    res = []
    nleak = 0
    for tp in sel:
        hs = []
        try:
            with tarfile.open(tp, "r|") as tf:
                for m in tf:
                    if m.isfile() and m.name.endswith(".mp4"):
                        hs.extend(_clip_hashes(io.BytesIO(tf.extractfile(m).read()),
                                               args.frames_per_video))
                        if len(hs) >= args.frames_per_video * 8:
                            break
        except Exception as e:
            res.append({"tar": os.path.basename(tp), "err": repr(e)[:50]})
            continue
        if not hs:
            continue
        d = ph.dup_fraction(hs, ev)
        leak = d > args.threshold
        if leak:
            nleak += 1
            if args.apply:
                os.replace(tp, os.path.join(leakdir, os.path.basename(tp)))
        res.append({"tar": os.path.basename(tp), "dup": round(d, 4), "leak": leak})
    pdir = _pdir(args)
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, f"gs_{lo}_{hi}.json"), "w") as f:
        json.dump({"range": [lo, hi], "n": len(sel), "leak": nleak, "results": res}, f)
    print(f"[gate-staging] [{lo}:{hi}): {nleak}/{len(sel)} leaked "
          f"({'MOVED' if args.apply else 'dry-run'})", flush=True)


def gate_tree(args):
    ev, _, _ = ph.load_ref(args.eval_ref)
    vids = []
    for root, _, files in os.walk(args.tree):
        for fn in files:
            if fn.lower().endswith((".mp4", ".avi", ".mkv", ".mov", ".m4v")):
                vids.append(os.path.join(root, fn))
    vids.sort()
    lo = args.start or 0
    hi = args.end if args.end is not None else len(vids)
    sel = vids[lo:hi]
    res = []
    nleak = 0
    for fp in sel:
        try:
            hs = _clip_hashes(fp, args.frames_per_video)
        except Exception:
            continue
        if not hs:
            continue
        d = ph.dup_fraction(hs, ev)
        leak = d > args.threshold
        if leak:
            nleak += 1
        res.append({"file": os.path.relpath(fp, args.tree), "dup": round(d, 4), "leak": leak})
    pdir = _pdir(args)
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, f"gt_{lo}_{hi}.json"), "w") as f:
        json.dump({"range": [lo, hi], "n": len(sel), "leak": nleak, "results": res}, f)
    print(f"[gate-tree] [{lo}:{hi}): {nleak}/{len(sel)} leaked", flush=True)


def gate_shards(args):
    """Report leaked SOURCE VIDEOS in a resharded dir (aggregates clips by source)."""
    ev, _, _ = ph.load_ref(args.eval_ref)
    shards = sorted(glob.glob(os.path.join(args.shards, "*.tar")))
    lo = args.start or 0
    hi = args.end if args.end is not None else len(shards)
    sel = shards[lo:hi]
    # accumulate dup per source video across its clips (max over clips)
    src_dup = {}
    for sp in sel:
        with tarfile.open(sp, "r|") as tf:
            for m in tf:
                if not (m.isfile() and m.name.endswith(".mp4")):
                    continue
                key = m.name[:-4]
                mm = SRC_RE.match(key)
                src = mm.group("source") if mm else key
                try:
                    hs = _clip_hashes(io.BytesIO(tf.extractfile(m).read()),
                                      args.frames_per_video)
                except Exception:
                    continue
                if not hs:
                    continue
                d = ph.dup_fraction(hs, ev)
                if d > src_dup.get(src, -1):
                    src_dup[src] = d
    leaked = {s: round(d, 4) for s, d in src_dup.items() if d > args.threshold}
    pdir = _pdir(args)
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, f"gsh_{lo}_{hi}.json"), "w") as f:
        json.dump({"range": [lo, hi], "sources": len(src_dup),
                   "leaked_sources": leaked}, f)
    print(f"[gate-shards] [{lo}:{hi}): {len(leaked)}/{len(src_dup)} source videos leaked",
          flush=True)


def main():
    args = parse_args()
    {"build-eval-ref": build_eval_ref, "finalize-eval-ref": finalize_eval_ref,
     "gate-staging": gate_staging, "gate-tree": gate_tree,
     "gate-shards": gate_shards}[args.mode](args)


if __name__ == "__main__":
    main()
