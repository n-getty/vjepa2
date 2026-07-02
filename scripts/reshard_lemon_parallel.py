#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Parallel reshard for LEMON's large staging (4162 source tars, ~1.9 TB).

The serial scripts/reshard_webdataset.py walltime-killed on LEMON: its single-
threaded index-then-write over 4162 tars can't fit a 1h window (sitl_2026 at
179 GB was fine). This fans N workers over DISJOINT subsets of source tars.
Each worker indexes only its subset and writes its own uniquely-named shards
(`lemon-wII-NNNNNN.tar`) via reshard_webdataset.write_shards — so there is no
shared output-shard write contention and no global numbering to coordinate.
Cross-source shuffle is preserved WITHIN each worker's shards (each worker still
sees ~170 source videos). --finalize merges all worker shards into one
metadata.json (the loader only consumes the shard_urls list; names don't matter).

Balance: tars are size-sorted and assigned round-robin (worker i gets
sorted[i::N]) so each worker gets near-equal total bytes.

Usage (driven by reshard_lemon_pbs.sh):
  # worker:
  python3 scripts/reshard_lemon_parallel.py --staging <dir> --output <dir> \
      --n-workers 24 --worker-id 3 --shards-per-worker 22 --seed 0
  # finalize:
  python3 scripts/reshard_lemon_parallel.py --staging <dir> --output <dir> \
      --finalize
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reshard_webdataset as rw  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--staging", required=True, help="lemon_staging dir (lemon__*.tar).")
    p.add_argument("--output", required=True, help="Output dir for lemon-wII-*.tar shards.")
    p.add_argument("--n-workers", type=int, default=24)
    p.add_argument("--worker-id", type=int, default=None, help="This worker's index [0,n).")
    p.add_argument("--shards-per-worker", type=int, default=22)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--finalize", action="store_true")
    return p.parse_args()


def _sorted_tars(staging):
    tars = glob.glob(os.path.join(staging, "lemon__*.tar"))
    # size-desc so round-robin i::N balances total bytes per worker
    tars.sort(key=lambda t: os.path.getsize(t), reverse=True)
    return tars


def run_worker(args):
    tars = _sorted_tars(args.staging)
    subset = [Path(t) for t in tars[args.worker_id::args.n_workers]]
    print(f"[w{args.worker_id:02d}] {len(subset)}/{len(tars)} source tars", flush=True)
    index = rw.index_samples(subset)
    keys = list(index.keys())
    if not keys:
        print(f"[w{args.worker_id:02d}] no samples; skipping", flush=True)
        return
    groups = rw.group_by_source_video(keys)
    nsh = max(1, min(args.shards_per_worker, max(1, len(keys) // 4)))
    shards = rw.assign_keys_to_shards(groups, nsh, args.seed + args.worker_id)
    prefix = f"lemon-w{args.worker_id:02d}"
    os.makedirs(args.output, exist_ok=True)
    n_members, n_bytes = rw.write_shards(index, shards, Path(args.output), prefix)
    names = [f"{prefix}-{i:06d}.tar" for i in range(nsh)]
    pj = os.path.join(args.output, f"_wpartial_{args.worker_id:02d}.json")
    with open(pj, "w") as f:
        json.dump({"worker": args.worker_id, "samples": len(keys),
                   "shards": names, "members": n_members, "bytes": n_bytes}, f)
    print(f"[w{args.worker_id:02d}] {len(keys)} samples -> {nsh} shards, "
          f"{n_members} members, {n_bytes/1e9:.1f} GB", flush=True)


def finalize(args):
    parts = sorted(glob.glob(os.path.join(args.output, "_wpartial_*.json")))
    shard_urls, n_samples = [], 0
    for pj in parts:
        d = json.load(open(pj))
        shard_urls.extend(d["shards"]); n_samples += d["samples"]
    shard_urls = sorted(shard_urls)
    # sanity: every listed shard exists
    missing = [u for u in shard_urls if not os.path.exists(os.path.join(args.output, u))]
    if missing:
        raise SystemExit(f"{len(missing)} listed shards missing, e.g. {missing[:3]}")
    meta = {"name": "lemon", "shard_count": len(shard_urls),
            "sample_count": n_samples, "shard_urls": shard_urls}
    with open(os.path.join(args.output, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(args.output, "reshard_summary.json"), "w") as f:
        json.dump({"dataset": "lemon", "workers": len(parts),
                   "output_shards": len(shard_urls), "samples": n_samples,
                   "parallel": True}, f, indent=2)
    print(f"FINALIZE: {len(parts)} workers -> {len(shard_urls)} shards, "
          f"{n_samples} samples", flush=True)


def main():
    args = parse_args()
    t0 = time.time()
    if args.finalize:
        finalize(args)
    elif args.worker_id is not None:
        run_worker(args)
    else:
        raise SystemExit("specify --worker-id or --finalize")
    print(f"elapsed {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
