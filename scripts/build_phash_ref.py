#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Build the perceptual-hash reference used to dedup LEMON before segmenting.

LEMON (Surg-3M) and our surgical corpus are all YouTube-scraped, but our data
stripped the YouTube IDs, so overlap can't be found by ID-join. This builds a
DCT-pHash reference from two pools so the segmenter (scripts/segment_videos_to_wds.py)
can drop LEMON videos whose content matches:

  eval_hashes  : the yt_robotic_chole triplet EVAL (9 Batch source videos).
                 A LEMON video matching these = eval LEAKAGE (correctness-critical).
  train_hashes : the surgenet_robotic TRAINING set (~3624 clips).
                 A LEMON video matching these = train DUPLICATION (memorization).

The 9 eval Batches are continuous source videos tiled into ~2300 4s window clips
each (yt_chole_tool_windows/yt_robotic_chole_Batch{0..8}/*.mp4); we sample frames
across those clips. surgenet_robotic lives as WebDataset tar shards; we sample
~1 frame per clip.

Output: phash_ref.json (hashes as hex strings). See scripts/phash_util.py.

Usage (frameworks python):
  python3 scripts/build_phash_ref.py \
      --eval-dir  /flare/ModCon/ngetty/data/yt_chole_tool_windows \
      --surgenet-dir /flare/ModCon/ngetty/data/surg_vid_webdataset_resharded/surgenet_robotic \
      --out /flare/ModCon/ngetty/data/LEMON/phash_ref.json
"""
from __future__ import annotations

import argparse
import glob
import io
import os
import sys
import tarfile
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phash_util as ph  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--eval-dir", required=True,
                   help="yt_chole_tool_windows dir (contains yt_robotic_chole_Batch*/).")
    p.add_argument("--surgenet-dir", required=True,
                   help="surgenet_robotic resharded dir (contains *.tar shards).")
    p.add_argument("--out", required=True, help="Output phash_ref.json path.")
    p.add_argument("--eval-frames-per-clip", type=int, default=2,
                   help="Frames sampled per eval window clip.")
    p.add_argument("--eval-clip-stride", type=int, default=3,
                   help="Take every Nth eval clip (they overlap 4s@1s stride).")
    p.add_argument("--train-frames-per-clip", type=int, default=2,
                   help="Frames sampled per surgenet_robotic clip.")
    return p.parse_args()


def build_eval(eval_dir, fpc, clip_stride):
    """Sample frames across the 9 eval Batch source videos -> list of hashes."""
    hashes = []
    batches = sorted(glob.glob(os.path.join(eval_dir, "yt_robotic_chole_Batch*")))
    print(f"[eval] {len(batches)} batches under {eval_dir}", flush=True)
    for bd in batches:
        clips = sorted(glob.glob(os.path.join(bd, "*.mp4")))[::clip_stride]
        n0 = len(hashes)
        for cp in clips:
            try:
                for fr in ph.sample_gray_frames(cp, n=fpc):
                    hashes.append(ph.phash_gray(fr))
            except Exception as e:
                print(f"  [eval-skip] {os.path.basename(cp)}: {repr(e)[:60]}", flush=True)
        print(f"  {os.path.basename(bd)}: {len(clips)} clips (stride) -> "
              f"{len(hashes)-n0} frames", flush=True)
    return hashes


def build_surgenet(surg_dir, fpc):
    """Sample frames from surgenet_robotic tar shards -> list of hashes."""
    hashes = []
    shards = sorted(glob.glob(os.path.join(surg_dir, "*.tar")))
    print(f"[surgenet] {len(shards)} shards under {surg_dir}", flush=True)
    tmpd = tempfile.mkdtemp(prefix="phref_")
    try:
        for si, sp in enumerate(shards):
            with tarfile.open(sp, "r|") as tf:
                for m in tf:
                    if not (m.isfile() and m.name.endswith(".mp4")):
                        continue
                    try:
                        buf = tf.extractfile(m).read()
                        for fr in ph.sample_gray_frames(io.BytesIO(buf), n=fpc):
                            hashes.append(ph.phash_gray(fr))
                    except Exception:
                        continue
            if (si + 1) % 20 == 0:
                print(f"  {si+1}/{len(shards)} shards -> {len(hashes)} frames", flush=True)
    finally:
        import shutil
        shutil.rmtree(tmpd, ignore_errors=True)
    return hashes


def main():
    args = parse_args()
    t0 = time.time()
    ev = build_eval(args.eval_dir, args.eval_frames_per_clip, args.eval_clip_stride)
    tr = build_surgenet(args.surgenet_dir, args.train_frames_per_clip)
    meta = {
        "eval_dir": os.path.abspath(args.eval_dir),
        "surgenet_dir": os.path.abspath(args.surgenet_dir),
        "eval_frames": len(ev),
        "train_frames": len(tr),
        "match_bits": ph.MATCH_BITS,
        "hash_size": ph.HASH_SIZE,
        "built_s": round(time.time() - t0, 1),
    }
    ph.save_ref(args.out, ev, tr, meta)
    # report unique counts (save_ref dedups)
    import json
    d = json.load(open(args.out))
    print(f"DONE in {meta['built_s']}s -> {args.out}", flush=True)
    print(f"  eval:  {len(ev)} frames -> {len(d['eval_hashes'])} unique hashes", flush=True)
    print(f"  train: {len(tr)} frames -> {len(d['train_hashes'])} unique hashes", flush=True)


if __name__ == "__main__":
    main()
