#!/usr/bin/env python3
"""Emit a ``metadata.json`` for a WebDataset shard directory.

Scans ``<input>/*.tar`` and writes ``<input>/metadata.json`` with
``name``, ``shard_count``, ``sample_count``, and ``shard_urls`` keys —
matching the schema consumed by ``src/datasets/webdataset.py`` and produced
by ``scripts/reshard_webdataset.py``.

By default the sample count is counted exactly across all shards. Pass
``--estimate-from N`` to count one (or N) probe shards and extrapolate when
the dataset is large.

Stdlib only.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path,
                   help="Dataset directory containing *.tar shards.")
    p.add_argument("--name", default=None,
                   help="Override dataset name (default: input dir basename).")
    p.add_argument("--estimate-from", type=int, default=0,
                   help="If >0, sample count is probe_samples * shard_count using "
                        "the first N shards as probes (faster for large corpora).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing metadata.json.")
    return p.parse_args()


def count_keys_in_tar(tar_path: Path) -> int:
    seen = set()
    with tarfile.open(tar_path, "r|") as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = member.name
            dot = name.find(".")
            key = name[:dot] if dot > 0 else name
            seen.add(key)
    return len(seen)


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Not a directory: {input_dir}")

    shard_files = sorted(p.name for p in input_dir.glob("*.tar"))
    if not shard_files:
        raise SystemExit(f"No .tar shards in {input_dir}")

    meta_path = input_dir / "metadata.json"
    if meta_path.exists() and not args.force:
        raise SystemExit(f"{meta_path} already exists; pass --force to overwrite.")

    name = args.name or input_dir.name

    if args.estimate_from > 0:
        probes = shard_files[:args.estimate_from]
        per_shard = []
        for s in probes:
            n = count_keys_in_tar(input_dir / s)
            per_shard.append(n)
            print(f"  probe {s}: {n} samples", file=sys.stderr)
        avg = sum(per_shard) / len(per_shard)
        sample_count = int(round(avg * len(shard_files)))
        estimated = True
    else:
        sample_count = 0
        for i, s in enumerate(shard_files):
            n = count_keys_in_tar(input_dir / s)
            sample_count += n
            if (i + 1) % 20 == 0 or i == len(shard_files) - 1:
                print(f"  scanned {i+1}/{len(shard_files)} shards "
                      f"(running sample_count={sample_count})", file=sys.stderr)
        estimated = False

    meta = {
        "name": name,
        "shard_count": len(shard_files),
        "sample_count": int(sample_count),
        "shard_urls": shard_files,
    }
    if estimated:
        meta["estimated"] = True

    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"wrote {meta_path}: {len(shard_files)} shards, "
          f"{sample_count} samples ({'estimated' if estimated else 'exact'})")


if __name__ == "__main__":
    main()
