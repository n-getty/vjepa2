#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Build the perceptual-hash reference used to dedup LEMON before segmenting.

LEMON (Surg-3M) and our surgical corpus are all YouTube-scraped, but our data
stripped the YouTube IDs, so overlap can't be found by ID-join. This builds a
DCT-pHash reference from two pools so the LEMON gate can drop matching videos:

  eval_hashes  : the yt_robotic_chole triplet EVAL (9 Batch source videos).
                 A LEMON video matching these = eval LEAKAGE (correctness-critical).
  train_hashes : the surgenet_robotic TRAINING set (~3624 clips).
                 A LEMON video matching these = train DUPLICATION (memorization).

DENSE by default: every eval window clip + every surgenet clip is sampled (the
first sparse build had only 72% eval self-recall — it missed ~28% of clips that
were literally IN the eval set, because pHash needs near-exact frame matches and
sparse sampling left gaps). Eval clips are 4s windows -> sequential-decode +
subsample (seek is wasteful on tiny clips). surgenet clips are 60s -> seek.

Parallel: --pool {eval,surgenet} + range slicing (--batch-start/-end for eval by
Batch dir; --shard-start/-end for surgenet by shard) write per-range partials;
--finalize merges them into phash_ref.json. See scripts/phash_util.py.

Usage: driven by scripts/build_phash_ref_pbs.sh (fan-out + finalize).
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
    p.add_argument("--pool", choices=["eval", "surgenet"], default=None,
                   help="Which pool this worker builds (omit with --finalize).")
    p.add_argument("--eval-dir", default=None,
                   help="yt_chole_tool_windows dir (contains yt_robotic_chole_Batch*/).")
    p.add_argument("--surgenet-dir", default=None,
                   help="surgenet_robotic resharded dir (contains *.tar shards).")
    p.add_argument("--out", required=True, help="Final phash_ref.json path.")
    p.add_argument("--partial-dir", default=None,
                   help="Dir for per-range partial jsons (default: <out>.partials).")
    p.add_argument("--eval-frames-per-clip", type=int, default=4,
                   help="Frames sampled per eval window clip (dense: every clip).")
    p.add_argument("--train-frames-per-clip", type=int, default=4,
                   help="Frames sampled per surgenet clip.")
    p.add_argument("--batch-start", type=int, default=None, help="eval: first Batch dir index.")
    p.add_argument("--batch-end", type=int, default=None, help="eval: last Batch dir index (excl).")
    p.add_argument("--shard-start", type=int, default=None, help="surgenet: first shard index.")
    p.add_argument("--shard-end", type=int, default=None, help="surgenet: last shard index (excl).")
    p.add_argument("--finalize", action="store_true", help="Merge partials -> out; no pool build.")
    return p.parse_args()


def _partial_dir(args):
    return args.partial_dir or (args.out + ".partials")


def build_eval(eval_dir, fpc, b_lo, b_hi, pdir):
    batches = sorted(glob.glob(os.path.join(eval_dir, "yt_robotic_chole_Batch*")))
    lo = b_lo if b_lo is not None else 0
    hi = b_hi if b_hi is not None else len(batches)
    sel = batches[lo:hi]
    print(f"[eval] {len(batches)} batches; this worker [{lo}:{hi}) -> {len(sel)}", flush=True)
    hashes = []
    for bd in sel:
        clips = sorted(glob.glob(os.path.join(bd, "*.mp4")))
        n0 = len(hashes)
        for cp in clips:
            try:
                # short 4s clips: sequential decode + subsample (seek=False)
                for fr in ph.sample_gray_frames(cp, n=fpc, seek=False):
                    hashes.append(ph.phash_gray(fr))
            except Exception:
                continue
        print(f"  {os.path.basename(bd)}: {len(clips)} clips -> {len(hashes)-n0} frames", flush=True)
    os.makedirs(pdir, exist_ok=True)
    pp = os.path.join(pdir, f"eval_{lo}_{hi}.json")
    with open(pp, "w") as f:
        json.dump({"pool": "eval", "hashes": [f"{h:016x}" for h in hashes]}, f)
    print(f"[eval] wrote {len(hashes)} hashes -> {pp}", flush=True)


def build_surgenet(surg_dir, fpc, s_lo, s_hi, pdir):
    shards = sorted(glob.glob(os.path.join(surg_dir, "*.tar")))
    lo = s_lo if s_lo is not None else 0
    hi = s_hi if s_hi is not None else len(shards)
    sel = shards[lo:hi]
    print(f"[surgenet] {len(shards)} shards; this worker [{lo}:{hi}) -> {len(sel)}", flush=True)
    hashes = []
    for si, sp in enumerate(sel):
        with tarfile.open(sp, "r|") as tf:
            for m in tf:
                if not (m.isfile() and m.name.endswith(".mp4")):
                    continue
                try:
                    buf = tf.extractfile(m).read()
                    # 60s clips: seek-sample
                    for fr in ph.sample_gray_frames(io.BytesIO(buf), n=fpc, seek=True):
                        hashes.append(ph.phash_gray(fr))
                except Exception:
                    continue
    os.makedirs(pdir, exist_ok=True)
    pp = os.path.join(pdir, f"surgenet_{lo}_{hi}.json")
    with open(pp, "w") as f:
        json.dump({"pool": "surgenet", "hashes": [f"{h:016x}" for h in hashes]}, f)
    print(f"[surgenet] wrote {len(hashes)} hashes -> {pp}", flush=True)


def finalize(args):
    pdir = _partial_dir(args)
    ev, tr = [], []
    parts = sorted(glob.glob(os.path.join(pdir, "*.json")))
    for pp in parts:
        d = json.load(open(pp))
        hs = [int(h, 16) for h in d["hashes"]]
        (ev if d["pool"] == "eval" else tr).extend(hs)
    meta = {
        "eval_frames": len(ev), "train_frames": len(tr),
        "match_bits": ph.MATCH_BITS, "hash_size": ph.HASH_SIZE,
        "partials": len(parts), "dense": True,
    }
    ph.save_ref(args.out, ev, tr, meta)
    d = json.load(open(args.out))
    print(f"FINALIZE {len(parts)} partials -> {args.out}", flush=True)
    print(f"  eval:  {len(ev)} frames -> {len(d['eval_hashes'])} unique", flush=True)
    print(f"  train: {len(tr)} frames -> {len(d['train_hashes'])} unique", flush=True)


def main():
    args = parse_args()
    t0 = time.time()
    if args.finalize:
        finalize(args)
    elif args.pool == "eval":
        build_eval(args.eval_dir, args.eval_frames_per_clip,
                   args.batch_start, args.batch_end, _partial_dir(args))
    elif args.pool == "surgenet":
        build_surgenet(args.surgenet_dir, args.train_frames_per_clip,
                       args.shard_start, args.shard_end, _partial_dir(args))
    else:
        raise SystemExit("specify --pool {eval,surgenet} or --finalize")
    print(f"elapsed {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
