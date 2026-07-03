#!/usr/bin/env python3
"""Reshard surgical WebDataset tars, shuffling clips across source videos.

Per the V-JEPA 2.1 Aurora plan, several datasets in
``/flare/ModCon/ngetty/data/surg_vid_webdataset`` are under-sharded for a
16-node × 12-rank topology with ``split_by_node``: 6 datasets are singletons
and a couple more have fewer shards than nodes. This script rewrites a given
dataset into ``--target-shards`` output tars while ensuring each output shard
contains clips drawn from many source videos (not a single source video).

Each sample in the input shards is a triple ``<key>.{mp4,json,cls}`` where the
key looks like ``clips_1min__<source_video>_clip_<N>``. We parse the source
video from the key, group samples by source, then round-robin source videos
into output shards so within-shard correlation drops sharply.

Usage::

    python scripts/reshard_webdataset.py \\
        --input  /flare/ModCon/ngetty/data/surg_vid_webdataset/jigsaw \\
        --output /flare/ModCon/ngetty/data/surg_vid_webdataset_resharded/jigsaw \\
        --target-shards 32 --seed 0

Only stdlib is used (``tarfile``) so the script runs anywhere — no
``webdataset`` install required. Idempotent: existing outputs are detected and
the run aborts unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tarfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


# Split a member name at the FIRST dot, matching WebDataset's group-by-key
# semantics (base_plus_ext). This is required for image samples whose media
# member is ``<key>.image.jpg`` (two dots): a last-dot split would key the JPEG
# as ``<key>.image`` while its ``.json``/``.cls`` key as ``<key>``, scattering
# the triple across shards. Single-dot video members (``<key>.mp4``) are
# unaffected — first- and last-dot splits agree for them.
SAMPLE_KEY_RE = re.compile(r"^(?P<key>[^.]+)\.(?P<ext>.+)$")
SOURCE_VIDEO_RE = re.compile(r"^(?P<prefix>.+?)__(?P<source>.+)_clip_(?P<clip>\d+)$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path, help="Input dataset dir containing *.tar shards.")
    p.add_argument("--output", required=True, type=Path, help="Output dir; will be created.")
    p.add_argument("--target-shards", type=int, default=32, help="Desired output shard count (default: 32).")
    p.add_argument("--min-samples-per-shard", type=int, default=4,
                   help="Lower bound; clamps target if dataset is small.")
    p.add_argument("--seed", type=int, default=0, help="Shuffle seed.")
    p.add_argument("--force", action="store_true", help="Overwrite existing output shards.")
    p.add_argument("--prefix", type=str, default=None,
                   help="Shard filename prefix (default: input dir name).")
    return p.parse_args()


def list_input_tars(input_dir: Path) -> List[Path]:
    tars = sorted(input_dir.glob("*.tar"))
    if not tars:
        raise SystemExit(f"No .tar shards found in {input_dir}")
    return tars


def index_samples(tars: List[Path]) -> Dict[str, Dict[str, Tuple[Path, str]]]:
    """Scan all tars; return {sample_key: {ext: (tar_path, member_name)}}.

    Reads only the tar index (no payloads), so this pass is I/O-cheap.
    """
    index: Dict[str, Dict[str, Tuple[Path, str]]] = defaultdict(dict)
    for tar_path in tars:
        with tarfile.open(tar_path, "r|") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                m = SAMPLE_KEY_RE.match(member.name)
                if not m:
                    continue
                key = m.group("key")
                ext = m.group("ext")
                index[key][ext] = (tar_path, member.name)
    return index


def group_by_source_video(sample_keys: List[str]) -> Dict[str, List[str]]:
    """Group sample keys by their source video (parsed from the key pattern)."""
    groups: Dict[str, List[str]] = defaultdict(list)
    unknown = 0
    for key in sample_keys:
        m = SOURCE_VIDEO_RE.match(key)
        if m:
            groups[m.group("source")].append(key)
        else:
            groups["__unknown__"].append(key)
            unknown += 1
    if unknown:
        print(f"  warning: {unknown} keys did not match the source-video pattern; "
              "they are grouped under '__unknown__' and will be distributed normally.",
              file=sys.stderr)
    return groups


def assign_keys_to_shards(
    groups: Dict[str, List[str]],
    n_shards: int,
    seed: int,
) -> List[List[str]]:
    """Interleave sample keys from different source videos into ``n_shards`` buckets.

    Algorithm: shuffle clips within each source video, then build a single
    interleaved stream by round-robin draining of source-video queues taken in
    a shuffled order. Walk that stream and dispatch to shards round-robin.
    Result: any contiguous span of the stream — and therefore any shard —
    contains clips from many different source videos.
    """
    rng = random.Random(seed)

    queues: List[Tuple[str, List[str]]] = []
    for src, keys in groups.items():
        keys = list(keys)
        rng.shuffle(keys)
        queues.append((src, keys))
    rng.shuffle(queues)

    stream: List[str] = []
    while queues:
        next_round: List[Tuple[str, List[str]]] = []
        for src, q in queues:
            stream.append(q.pop())
            if q:
                next_round.append((src, q))
        rng.shuffle(next_round)
        queues = next_round

    shards: List[List[str]] = [[] for _ in range(n_shards)]
    for i, key in enumerate(stream):
        shards[i % n_shards].append(key)
    return shards


def write_shards(
    index: Dict[str, Dict[str, Tuple[Path, str]]],
    shard_assignments: List[List[str]],
    output_dir: Path,
    prefix: str,
) -> Tuple[int, int]:
    """Stream members from source tars into assigned output shards.

    To avoid opening each source tar once per output shard, we invert the plan:
    iterate source tars in order, look up each member's destination shard, and
    append to that shard's writer. Output writers are kept open for the
    duration; this is fine for a few dozen shards.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    key_to_shard: Dict[str, int] = {}
    for shard_idx, keys in enumerate(shard_assignments):
        for k in keys:
            key_to_shard[k] = shard_idx

    src_tars = sorted({tp for ext_map in index.values() for (tp, _) in ext_map.values()})
    writers: List[tarfile.TarFile] = []
    out_paths: List[Path] = []
    for i in range(len(shard_assignments)):
        out = output_dir / f"{prefix}-{i:06d}.tar"
        out_paths.append(out)
        writers.append(tarfile.open(out, "w"))

    n_members = 0
    try:
        for tar_path in src_tars:
            with tarfile.open(tar_path, "r|") as tf:
                for member in tf:
                    if not member.isfile():
                        continue
                    m = SAMPLE_KEY_RE.match(member.name)
                    if not m:
                        continue
                    key = m.group("key")
                    if key not in key_to_shard:
                        continue
                    shard_idx = key_to_shard[key]
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        continue
                    writers[shard_idx].addfile(member, fobj)
                    n_members += 1
    finally:
        for w in writers:
            w.close()

    n_bytes = sum(p.stat().st_size for p in out_paths)
    return n_members, n_bytes


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input.resolve()
    output_dir: Path = args.output.resolve()
    prefix = args.prefix or input_dir.name

    if output_dir.exists() and any(output_dir.glob("*.tar")):
        if not args.force:
            raise SystemExit(f"Output {output_dir} already contains .tar files; pass --force to overwrite.")
        for p in output_dir.glob("*.tar"):
            p.unlink()

    t0 = time.time()
    tars = list_input_tars(input_dir)
    print(f"[{prefix}] indexing {len(tars)} input shards from {input_dir}")
    index = index_samples(tars)
    sample_keys = list(index.keys())
    n_samples = len(sample_keys)
    if n_samples == 0:
        raise SystemExit(f"No samples found in {input_dir}")

    # Clamp target shards so each shard gets at least min_samples_per_shard.
    max_by_size = max(1, n_samples // args.min_samples_per_shard)
    n_shards = max(1, min(args.target_shards, max_by_size))
    if n_shards != args.target_shards:
        print(f"[{prefix}] clamping target shards {args.target_shards} -> {n_shards} "
              f"(only {n_samples} samples; --min-samples-per-shard={args.min_samples_per_shard})")

    groups = group_by_source_video(sample_keys)
    print(f"[{prefix}] {n_samples} samples across {len(groups)} source videos; "
          f"writing {n_shards} shards to {output_dir}")

    shards = assign_keys_to_shards(groups, n_shards, args.seed)
    n_members, n_bytes = write_shards(index, shards, output_dir, prefix)

    shard_urls = [f"{prefix}-{i:06d}.tar" for i in range(n_shards)]
    metadata = {
        "name": prefix,
        "shard_count": n_shards,
        "sample_count": n_samples,
        "shard_urls": shard_urls,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    elapsed = time.time() - t0
    summary = {
        "dataset": prefix,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "input_shards": len(tars),
        "output_shards": n_shards,
        "samples": n_samples,
        "source_videos": len(groups),
        "members_written": n_members,
        "bytes": n_bytes,
        "elapsed_s": elapsed,
        "seed": args.seed,
    }
    summary_path = output_dir / "reshard_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[{prefix}] done in {elapsed:.1f}s: {n_members} members, "
          f"{n_bytes / 1e9:.2f} GB across {n_shards} shards. "
          f"Summary at {summary_path}")


if __name__ == "__main__":
    main()
