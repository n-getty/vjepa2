#!/usr/bin/env python3
"""Ingest the raw full Kinetics-400 download into the project's WebDataset clip format.

The raw download at ``/flare/ModCon/ngetty/data/kinetics400_full/{train,val,test}/`` is a set of
``part_N.tar.gz`` archives, each a gzipped tar of ALREADY-CLIPPED, independent mp4s:
    ./<ytid>_<start>_<end>.mp4     # 480x360, 30fps, ~10s, h264, ~0.9MB each
(~1000 clips/part, 243 train parts, ~243K train clips total, 345 GB).

The trainer's WebDataset loader (src/datasets/webdataset.py) downsamples fps at decode and
RandomResizedCrops resolution at transform time, so clips are consumed at native res — there is NO
reason to re-encode. This is a pure REPACK: wrap each raw mp4 with the {json,cls} sidecars the loader
contract requires and shuffle clips across parts into K output shards. JEPA pretraining is label-free
(the existing kinetics400 subset carries label:0 throughout), so no K400 label map is needed.

Output contract (matches surg_vid_webdataset_resharded/kinetics400 exactly):
    <NNNNNNNN>.mp4    the clip (stream-copied, byte-identical to the source member)
    <NNNNNNNN>.json   {"source_dataset":"kinetics400","source_path":<orig member name>,"label":0}
    <NNNNNNNN>.cls    "0"
plus a metadata.json {name, shard_count, sample_count, shard_urls}.

Parallelism: the 243 parts are independent gzip streams, so we fan WORKERS over disjoint slices of
the sorted part list (--worker-id / --num-workers). Each worker owns a contiguous output-shard range
(no writer contention, no cross-worker coordination) and writes a per-worker manifest; a final
``--merge`` pass concatenates the per-worker manifests into one metadata.json. Clips within a worker
round-robin across that worker's shards so any shard mixes many source parts (decorrelation).

Usage:
  # smoke: one worker, first part only
  python3 scripts/ingest_k400full_to_wds.py \
      --input /flare/ModCon/ngetty/data/kinetics400_full/train \
      --output /flare/ModCon/ngetty/data/kinetics400_full_wds/kinetics400 \
      --shards-per-worker 4 --num-workers 1 --worker-id 0 --max-parts 1

  # production: 48 workers (fan over parts), each writes shards [worker_id*SPW, +SPW)
  #   launched from a PBS mpiexec; each rank sets --worker-id from PMI_RANK
  python3 scripts/ingest_k400full_to_wds.py --input .../train --output .../kinetics400 \
      --shards-per-worker 12 --num-workers 48 --worker-id $RANK

  # after all workers finish, merge the per-worker manifests:
  python3 scripts/ingest_k400full_to_wds.py --output .../kinetics400 --merge --num-workers 48
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import tarfile
import time
from pathlib import Path


def list_parts(input_dir: Path):
    parts = sorted(input_dir.glob("part_*.tar.gz"))
    if not parts:
        raise SystemExit(f"no part_*.tar.gz in {input_dir}")
    return parts


def worker_slice(parts, num_workers, worker_id):
    """Contiguous slice of the sorted part list for this worker."""
    n = len(parts)
    per = (n + num_workers - 1) // num_workers
    lo = worker_id * per
    hi = min(lo + per, n)
    return parts[lo:hi]


def ingest_worker(input_dir, output_dir, shards_per_worker, num_workers, worker_id,
                  dataset, seed, max_parts):
    parts = list_parts(input_dir)
    my_parts = worker_slice(parts, num_workers, worker_id)
    if max_parts:
        my_parts = my_parts[:max_parts]
    if not my_parts:
        print(f"[w{worker_id}] no parts assigned; exiting")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_base = worker_id * shards_per_worker
    out_paths = [output_dir / f"{dataset}-{shard_base + i:06d}.tar" for i in range(shards_per_worker)]
    writers = [tarfile.open(p, "w") for p in out_paths]

    rng = random.Random(seed + worker_id)
    # global-ish key offset so keys don't collide across workers (8-digit, part-major)
    key_ctr = worker_id * 10_000_000
    n_clips = 0
    t0 = time.time()
    try:
        for pi, part in enumerate(my_parts):
            with tarfile.open(part, "r:gz") as tf:
                for member in tf:
                    if not member.isfile() or not member.name.endswith(".mp4"):
                        continue
                    fobj = tf.extractfile(member)
                    if fobj is None:
                        continue
                    data = fobj.read()
                    key = f"{key_ctr:08d}"
                    key_ctr += 1
                    # round-robin destination shard; rng offset so parts interleave
                    shard_idx = (n_clips + rng.randint(0, shards_per_worker - 1)) % shards_per_worker
                    w = writers[shard_idx]
                    orig = os.path.basename(member.name)
                    sidecar = {
                        "source_dataset": dataset,
                        "source_path": orig,
                        "label": 0,
                    }
                    _add_bytes(w, f"{key}.mp4", data)
                    _add_bytes(w, f"{key}.json", json.dumps(sidecar).encode())
                    _add_bytes(w, f"{key}.cls", b"0")
                    n_clips += 1
            print(f"[w{worker_id}] part {pi+1}/{len(my_parts)} {part.name} "
                  f"cum_clips={n_clips} ({time.time()-t0:.0f}s)", flush=True)
    finally:
        for w in writers:
            w.close()

    manifest = {
        "worker_id": worker_id,
        "shard_urls": [p.name for p in out_paths],
        "sample_count": n_clips,
        "parts": [p.name for p in my_parts],
    }
    (output_dir / f"_manifest_w{worker_id:04d}.json").write_text(json.dumps(manifest, indent=2))
    print(f"[w{worker_id}] DONE {n_clips} clips -> {shards_per_worker} shards "
          f"[{shard_base}..{shard_base+shards_per_worker-1}] in {time.time()-t0:.0f}s", flush=True)


def _add_bytes(tar, name, data):
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def merge_manifests(output_dir, dataset, num_workers):
    """Concatenate per-worker manifests into one metadata.json (idempotent)."""
    shard_urls, total = [], 0
    found = 0
    for wid in range(num_workers):
        mf = output_dir / f"_manifest_w{wid:04d}.json"
        if not mf.exists():
            print(f"  WARNING: missing {mf.name} (worker {wid} did not finish?)")
            continue
        m = json.loads(mf.read_text())
        # keep only shards that actually have data
        shard_urls.extend(m["shard_urls"])
        total += m["sample_count"]
        found += 1
    # drop empty shards from the url list (a worker with 0 clips still opened empty tars)
    shard_urls = [u for u in shard_urls if (output_dir / u).exists() and (output_dir / u).stat().st_size > 1024]
    shard_urls.sort()
    metadata = {
        "name": dataset,
        "shard_count": len(shard_urls),
        "sample_count": total,
        "shard_urls": shard_urls,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"merged {found}/{num_workers} worker manifests: "
          f"{total} clips across {len(shard_urls)} non-empty shards -> {output_dir}/metadata.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, help="dir of part_*.tar.gz (train/val/test)")
    ap.add_argument("--output", type=Path, required=True, help="output WDS dir")
    ap.add_argument("--dataset", default="kinetics400", help="source_dataset + shard prefix")
    ap.add_argument("--shards-per-worker", type=int, default=12)
    ap.add_argument("--num-workers", type=int, default=1)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-parts", type=int, default=0, help="cap parts per worker (smoke); 0=all")
    ap.add_argument("--merge", action="store_true", help="merge per-worker manifests into metadata.json")
    args = ap.parse_args()

    if args.merge:
        merge_manifests(args.output, args.dataset, args.num_workers)
        return
    if not args.input:
        raise SystemExit("--input required unless --merge")
    ingest_worker(args.input, args.output, args.shards_per_worker, args.num_workers,
                  args.worker_id, args.dataset, args.seed, args.max_parts)


if __name__ == "__main__":
    main()
