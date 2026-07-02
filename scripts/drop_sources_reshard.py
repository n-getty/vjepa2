#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Reshard a WebDataset dir while DROPPING clips from a set of source videos.

Used to remove eval-leaked source videos from an already-resharded set
(surgenet_robotic) without a staging dir. Reads --input/*.tar, skips any sample
whose source-video (parsed from the key via SOURCE_VIDEO_RE) is in --drop-json
(a {source_id: dup} map), and writes clean shards + metadata to --output.

Single-pass streaming (index -> shuffle -> write), reusing reshard_webdataset's
helpers. surgenet_robotic is small (~500 shards / 3624 clips) so serial is fine.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reshard_webdataset as rw  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--drop-json", required=True, help="{source_id: score} to drop.")
    p.add_argument("--prefix", default=None)
    p.add_argument("--target-shards", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    drop = set(json.load(open(args.drop_json)).keys())
    prefix = args.prefix or args.input.name
    if args.output.exists() and any(args.output.glob("*.tar")):
        if not args.force:
            raise SystemExit(f"{args.output} has tars; --force to overwrite")
        for p in args.output.glob("*.tar"):
            p.unlink()

    tars = rw.list_input_tars(args.input)
    print(f"[{prefix}] indexing {len(tars)} shards; dropping {len(drop)} source videos", flush=True)
    index = rw.index_samples(tars)
    keys = list(index.keys())
    kept, dropped = [], 0
    for k in keys:
        m = rw.SOURCE_VIDEO_RE.match(k)
        src = m.group("source") if m else None
        if src in drop:
            dropped += 1
        else:
            kept.append(k)
    print(f"[{prefix}] {len(keys)} samples -> keep {len(kept)}, drop {dropped}", flush=True)

    # rebuild index with only kept keys
    kept_index = {k: index[k] for k in kept}
    n = len(kept)
    tgt = args.target_shards or max(1, n // 8)
    tgt = max(1, min(tgt, max(1, n // 4)))
    groups = rw.group_by_source_video(kept)
    shards = rw.assign_keys_to_shards(groups, tgt, args.seed)
    nmem, nbytes = rw.write_shards(kept_index, shards, args.output, prefix)
    urls = [f"{prefix}-{i:06d}.tar" for i in range(tgt)]
    json.dump({"name": prefix, "shard_count": tgt, "sample_count": n, "shard_urls": urls},
              open(args.output / "metadata.json", "w"), indent=2)
    json.dump({"dataset": prefix, "kept": n, "dropped_sources": len(drop),
               "dropped_clips": dropped, "output_shards": tgt, "members": nmem},
              open(args.output / "reshard_summary.json", "w"), indent=2)
    print(f"[{prefix}] wrote {tgt} shards, {n} samples, {nmem} members", flush=True)


if __name__ == "__main__":
    main()
