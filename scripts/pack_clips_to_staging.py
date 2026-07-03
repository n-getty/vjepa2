#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Pack a directory of ALREADY-CLIPPED videos into WebDataset staging tars.

For inputs that are already uniform short clips (e.g. surgenet_lap/clips_1min/,
1859 60s clips) — no re-segmentation needed. Wraps each clip as a
{mp4,json,cls} triple with a reshard-parseable key and bundles them into a few
staging tars, so reshard_webdataset.py can shuffle them alongside freshly
segmented sources.

Key: <dataset>__<source-prefix>_<stem>_clip_0000 (each input clip is its own
"source video" with a single clip; reshard shuffles across them). The trailing
_clip_0000 makes SOURCE_VIDEO_RE parse it.

Stdlib + no decode (pure repack), so this is fast.
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import tarfile

_SANITIZE = re.compile(r"[^A-Za-z0-9_]+")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True, help="Dir of already-clipped *.mp4.")
    p.add_argument("--output-staging", required=True)
    p.add_argument("--dataset", required=True, help="source_dataset label + key prefix.")
    p.add_argument("--source-prefix", default="clip", help="Key infix to namespace these.")
    p.add_argument("--clips-per-tar", type=int, default=100)
    p.add_argument("--exts", default=".mp4,.avi,.mkv,.mov,.m4v")
    return p.parse_args()


def _add(tar, name, data):
    ti = tarfile.TarInfo(name=name)
    ti.size = len(data)
    ti.mtime = 0
    tar.addfile(ti, io.BytesIO(data))


def main():
    args = parse_args()
    exts = tuple(args.exts.split(","))
    vids = sorted(f for f in os.listdir(args.input_dir) if f.lower().endswith(exts))
    os.makedirs(args.output_staging, exist_ok=True)
    print(f"[pack] {len(vids)} clips from {args.input_dir}", flush=True)
    n = 0
    tar = None
    ti = 0
    for i, fn in enumerate(vids):
        if i % args.clips_per_tar == 0:
            if tar is not None:
                tar.close()
            tpath = os.path.join(args.output_staging,
                                 f"{args.dataset}__{args.source_prefix}pack{ti:04d}.tar")
            tmp = tpath + ".tmp"
            tar = tarfile.open(tmp, "w")
            tar._tmp, tar._final = tmp, tpath  # stash for rename
            ti += 1
        stem = _SANITIZE.sub("_", os.path.splitext(fn)[0]).strip("_")
        key = f"{args.dataset}__{args.source_prefix}_{stem}_clip_0000"
        with open(os.path.join(args.input_dir, fn), "rb") as f:
            data = f.read()
        _add(tar, f"{key}.mp4", data)
        _add(tar, f"{key}.json",
             json.dumps({"source_dataset": args.dataset,
                         "source_path": os.path.abspath(os.path.join(args.input_dir, fn)),
                         "label": 0, "subset": args.source_prefix}).encode())
        _add(tar, f"{key}.cls", b"0")
        n += 1
    if tar is not None:
        tar.close()
    # rename .tmp -> final
    for t in glob.glob(os.path.join(args.output_staging,
                                    f"{args.dataset}__{args.source_prefix}pack*.tar.tmp")):
        os.replace(t, t[:-4])
    print(f"[pack] wrote {n} clips into {ti} tars", flush=True)


if __name__ == "__main__":
    main()
