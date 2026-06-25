#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""On-data verification of the degenerate (black) clip reject in VideoDecoder.

Runs the REAL VideoDecoder.decode over real webdataset shards and confirms:
  1. surgvu24: a meaningful fraction (~expected ~19%) of clips decode to
     near-zero std and are DROPPED by the filter (return None).
  2. a clean source (kinetics400): ~0% dropped, all real clips pass.
  3. with the filter disabled (min_clip_std=0) NOTHING is dropped, so the
     drops in (1) are entirely attributable to the filter, not decode errors.

This is the one thing the offline pytest cannot cover: that the std floor
fires on the actual corrupt bytes and leaves genuine surgical data untouched.

Usage (on a compute node, frameworks python):
  python3 scripts/verify_black_clip_filter.py \
      --surg /flare/ModCon/ngetty/data/surg_vid_webdataset_resharded/surgvu24 \
      --clean /flare/ModCon/ngetty/data/surg_vid_webdataset_resharded/kinetics400 \
      --n 400
"""
import argparse
import io
import os
import sys
import tarfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.datasets.webdataset import VideoDecoder  # noqa: E402


def _iter_samples_from_tar(tar_path, limit):
    """Yield (key, {ext: bytes}) sample dicts from one tar, up to `limit`."""
    groups = {}
    order = []
    with tarfile.open(tar_path, "r|") as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = m.name
            dot = name.find(".")
            key, ext = (name[:dot], name[dot + 1:]) if dot > 0 else (name, "")
            if key not in groups:
                groups[key] = {"__key__": key}
                order.append(key)
            groups[key][ext] = tf.extractfile(m).read()
            # Emit as soon as a group has both a media + label, to stream.
            if len(order) > limit + 8:
                break
    out = []
    for key in order:
        out.append(groups[key])
        if len(out) >= limit:
            break
    return out


def _decoder(min_clip_std):
    return VideoDecoder(
        frames_per_clip=16,
        frame_step=4,
        num_clips=1,
        random_clip_sampling=True,
        transform=None,        # keep raw buffer path; filter runs pre-transform
        shared_transform=None,
        min_clip_std=min_clip_std,
    )


def _raw_std(dec, sample):
    """Independently decode a sample's raw buffer and return its per-pixel std
    (or None on decode failure) — for the evidence histogram, mirroring the
    media discovery in VideoDecoder.decode."""
    for key in ("video.mp4", "video.avi", "video.mov", "video.webm",
                "video.mkv", "mp4", "avi", "mov", "webm", "mkv", "flv"):
        if key in sample:
            buf, _ = dec.loadvideo_decord(sample[key], 16)
            if len(buf) == 0:
                return None
            return float(np.asarray(buf, dtype=np.float32).std())
    return None


def run_source(name, path, n, min_clip_std):
    shards = sorted(f for f in os.listdir(path) if f.endswith(".tar"))
    dec = _decoder(min_clip_std)
    seen = dropped = 0
    near_zero = 0          # raw std < 1.0 (degenerate, regardless of filter)
    si = 0
    while seen < n and si < len(shards):
        samples = _iter_samples_from_tar(os.path.join(path, shards[si]), n - seen)
        si += 1
        for s in samples:
            std = _raw_std(dec, s)              # evidence readback
            out = dec.decode(s, 16, source_name=name)  # real keep/drop verdict
            seen += 1
            if std is not None and std < 1.0:
                near_zero += 1
            if out is None:
                dropped += 1
            if seen >= n:
                break
    return {"name": name, "seen": seen, "dropped": dropped,
            "near_zero": near_zero,
            "drop_pct": 100.0 * dropped / seen if seen else 0.0,
            "near_zero_pct": 100.0 * near_zero / seen if seen else 0.0,
            "min_clip_std": min_clip_std}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surg", required=True, help="corrupt source dir (surgvu24)")
    ap.add_argument("--clean", required=True, help="clean source dir (kinetics400)")
    ap.add_argument("--n", type=int, default=400, help="clips per source")
    args = ap.parse_args()

    print(f"=== Verifying black-clip filter on {args.n} clips/source ===\n")

    print("[1] surgvu24, filter ON (min_clip_std=1.0):")
    r1 = run_source("surgvu24", args.surg, args.n, 1.0)
    print(f"    seen={r1['seen']} dropped={r1['dropped']} "
          f"({r1['drop_pct']:.1f}% degenerate)\n")

    print("[2] surgvu24, filter OFF (min_clip_std=0.0):")
    r2 = run_source("surgvu24", args.surg, args.n, 0.0)
    print(f"    seen={r2['seen']} dropped={r2['dropped']} "
          f"({r2['drop_pct']:.1f}% — should be ~0, i.e. drops in [1] are the "
          f"filter not decode errors)\n")

    print("[3] kinetics400, filter ON (min_clip_std=1.0):")
    r3 = run_source("kinetics400", args.clean, args.n, 1.0)
    print(f"    seen={r3['seen']} dropped={r3['dropped']} "
          f"({r3['drop_pct']:.1f}% — should be ~0, clean source)\n")

    # Verdicts
    ok = True
    if not (r1["drop_pct"] > 5.0):
        print("FAIL: surgvu24 filter-ON drop% unexpectedly low "
              "(expected meaningful fraction, ~19%).")
        ok = False
    if not (r2["drop_pct"] < 1.0):
        print("FAIL: surgvu24 filter-OFF dropped clips — decode errors, not "
              "the filter. Investigate.")
        ok = False
    if not (r3["drop_pct"] < 2.0):
        print("FAIL: clean source drop% too high — filter is eating real data.")
        ok = False
    print("\n=== RESULT:", "PASS" if ok else "FAIL", "===")
    print(f"surgvu24 ON={r1['drop_pct']:.1f}%  OFF={r2['drop_pct']:.1f}%  "
          f"kinetics ON={r3['drop_pct']:.1f}%")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
