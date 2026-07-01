#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Source-side degenerate (black/frozen) clip removal for a WebDataset source.

Motivation
----------
``VideoDecoder`` drops black clips *at runtime* via ``min_clip_std`` (the v3 data
fix). That works but re-decodes and re-rejects the same ~19% of surgvu24 on every
epoch, and any data staging/copy still moves the dead weight. This script bakes
the filter into the shards **once**: it re-writes a source into a ``*_clean``
copy that contains only samples whose raw per-pixel std >= ``--min-clip-std``.

It reuses the EXACT runtime signal — ``VideoDecoder.loadvideo_decord`` followed by
``np.asarray(buffer).std()`` on the raw 0-255 buffer — so it drops precisely what
the runtime filter drops (validated at ~22.8% on surgvu24 by
``verify_black_clip_filter.py``). ``random_clip_sampling`` is forced OFF so the
verdict is deterministic and reproducible across a re-run/resume.

Shard names are preserved 1:1 (``surgvu24-000123.tar`` -> ``surgvu24-000123.tar``)
so the output ``metadata.json`` shard list lines up and configs only need the dir
path swapped. Kept samples carry ALL their extensions (mp4/json/cls) intact.

Parallelism
-----------
Pass ``--shard-start/--shard-end`` to process a contiguous shard range; run many
in a PBS array/mpiexec fan-out over disjoint ranges. Each worker writes a
per-range partial summary ``_partial_<start>_<end>.json``. When all ranges are
done, run ``--finalize`` (no range) to merge partials into ``metadata.json`` +
``reshard_summary.json``.

Idempotent: an output shard that already exists is skipped unless ``--force``.

Usage
-----
  # local smoke on 2 shards (foreground, frameworks python):
  python3 scripts/filter_black_clips_reshard.py \
      --input  /flare/.../surg_vid_webdataset_resharded/surgvu24 \
      --output /flare/.../surg_vid_webdataset_resharded/surgvu24_clean \
      --shard-start 0 --shard-end 2

  # finalize after all ranges complete:
  python3 scripts/filter_black_clips_reshard.py \
      --input  .../surgvu24 --output .../surgvu24_clean --finalize
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
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets.webdataset import VideoDecoder  # noqa: E402

VIDEO_EXTS = ("mp4", "avi", "mov", "webm", "mkv", "flv")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, help="Input source dir with *.tar shards.")
    p.add_argument("--output", required=True, help="Output *_clean dir (created).")
    p.add_argument("--min-clip-std", type=float, default=1.0,
                   help="Raw 0-255 per-pixel std floor; below this = degenerate (default 1.0).")
    p.add_argument("--fps", type=int, default=4, help="Sampling fps for the std probe (match training).")
    p.add_argument("--frames-per-clip", type=int, default=16, help="Frames sampled for the std probe.")
    p.add_argument("--shard-start", type=int, default=None, help="First shard index (inclusive).")
    p.add_argument("--shard-end", type=int, default=None, help="Last shard index (exclusive).")
    p.add_argument("--force", action="store_true", help="Overwrite existing output shards.")
    p.add_argument("--finalize", action="store_true",
                   help="Merge per-range partials into metadata.json + reshard_summary.json and exit.")
    return p.parse_args()


def _decoder(fps, fpc):
    # min_clip_std=0 so loadvideo_decord runs without the decoder itself dropping;
    # we compute std ourselves. random_clip_sampling off for deterministic verdicts.
    return VideoDecoder(
        frames_per_clip=fpc,
        frame_step=None,   # decoder requires EXACTLY one of frame_step/fps/duration
        num_clips=1,
        fps=fps,
        random_clip_sampling=False,
        transform=None,
        shared_transform=None,
        min_clip_std=0.0,
    )


def _group_members(tar_path):
    """Yield (key, [(TarInfo, bytes), ...]) grouped by sample key, preserving order."""
    groups = defaultdict(list)
    order = []
    with tarfile.open(tar_path, "r|") as tf:
        for m in tf:
            if not m.isfile():
                continue
            dot = m.name.find(".")
            key = m.name[:dot] if dot > 0 else m.name
            if key not in groups:
                order.append(key)
            groups[key].append((m, tf.extractfile(m).read()))
    for key in order:
        yield key, groups[key]


def _clip_std(dec, members, fpc):
    """Raw per-pixel std of the sampled buffer for a sample's video member.

    Returns (std, ok). ok=False on decode failure -> treated as KEEP (we do not
    want decode flakiness to silently delete real data)."""
    vid_bytes = None
    for tinfo, data in members:
        ext = tinfo.name.rsplit(".", 1)[-1].lower()
        if ext in VIDEO_EXTS:
            vid_bytes = data
            break
    if vid_bytes is None:
        return None, False  # no video member -> keep (unexpected; don't drop)
    buf, _ = dec.loadvideo_decord(vid_bytes, fpc)
    if buf is None or len(buf) == 0:
        return None, False  # decode failure -> keep
    return float(np.asarray(buf, dtype=np.float32).std()), True


def process_range(args, shards, lo, hi):
    dec = _decoder(args.fps, args.frames_per_clip)
    os.makedirs(args.output, exist_ok=True)
    per_shard = {}
    t0 = time.time()
    for si in range(lo, hi):
        in_path = shards[si]
        base = os.path.basename(in_path)
        out_path = os.path.join(args.output, base)
        if os.path.exists(out_path) and not args.force:
            print(f"  [skip] {base} exists (use --force)", flush=True)
            continue
        kept = dropped = decode_fail = 0
        tmp_path = out_path + ".tmp"
        with tarfile.open(tmp_path, "w") as out:
            for key, members in _group_members(in_path):
                std, ok = _clip_std(dec, members, args.frames_per_clip)
                if ok and std < args.min_clip_std:
                    dropped += 1
                    continue
                if not ok:
                    decode_fail += 1  # kept, but counted
                for tinfo, data in members:
                    out.addfile(tinfo, io.BytesIO(data))
                kept += 1
        os.replace(tmp_path, out_path)
        per_shard[base] = {"kept": kept, "dropped": dropped, "decode_fail": decode_fail}
        tot = kept + dropped
        print(f"  [{si:04d}] {base}: kept={kept} dropped={dropped} "
              f"({100*dropped/tot if tot else 0:.1f}%) decode_fail={decode_fail}", flush=True)
    partial = os.path.join(args.output, f"_partial_{lo}_{hi}.json")
    with open(partial, "w") as f:
        json.dump({"range": [lo, hi], "elapsed_s": time.time() - t0,
                   "min_clip_std": args.min_clip_std, "per_shard": per_shard}, f, indent=2)
    print(f"wrote {partial}", flush=True)


def finalize(args, shards):
    name = os.path.basename(os.path.normpath(args.output)).replace("_clean", "")
    partials = sorted(glob.glob(os.path.join(args.output, "_partial_*.json")))
    merged = {}
    for p in partials:
        merged.update(json.load(open(p)).get("per_shard", {}))
    out_shards = sorted(f for f in os.listdir(args.output) if f.endswith(".tar"))
    total_kept = sum(v["kept"] for v in merged.values())
    total_dropped = sum(v["dropped"] for v in merged.values())
    total_fail = sum(v.get("decode_fail", 0) for v in merged.values())
    meta = {"name": name, "shard_count": len(out_shards),
            "sample_count": int(total_kept), "shard_urls": out_shards}
    with open(os.path.join(args.output, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    summary = {"dataset": name, "input_dir": args.input, "output_dir": args.output,
               "output_shards": len(out_shards), "kept": total_kept,
               "dropped": total_dropped, "decode_fail_kept": total_fail,
               "drop_pct": 100 * total_dropped / (total_kept + total_dropped)
               if (total_kept + total_dropped) else 0.0,
               "min_clip_std": args.min_clip_std,
               "shards_processed": len(merged), "shards_expected": len(shards)}
    with open(os.path.join(args.output, "reshard_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if len(merged) != len(shards):
        print(f"WARNING: processed {len(merged)} of {len(shards)} input shards — "
              "some ranges may be missing.", file=sys.stderr)


def main():
    args = parse_args()
    shards = sorted(glob.glob(os.path.join(args.input, "*.tar")))
    if not shards:
        raise SystemExit(f"No .tar shards in {args.input}")
    if args.finalize:
        finalize(args, shards)
        return
    lo = args.shard_start if args.shard_start is not None else 0
    hi = args.shard_end if args.shard_end is not None else len(shards)
    hi = min(hi, len(shards))
    print(f"[{os.path.basename(args.input)}] filtering shards [{lo}:{hi}) of {len(shards)}; "
          f"min_clip_std={args.min_clip_std}", flush=True)
    process_range(args, shards, lo, hi)


if __name__ == "__main__":
    main()
