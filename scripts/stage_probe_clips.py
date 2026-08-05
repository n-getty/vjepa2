#!/usr/bin/env python3
"""Stage a probe's video-clip corpus onto this compute node's /tmp.

The SAR-RARP50 (and similar) ASFormer/attentive probes read clips through a
CSV-of-paths VideoDataset (src/datasets/video_dataset.py): each CSV row is an
absolute clip path + per-frame labels. On Aurora the corpus is thousands of
small .mp4 files on Lustre; the live/uncached probe is bottlenecked on random
small-file reads. The corpus is small (~13 GB for SAR-RARP50), so the whole set
fits in the RAM-backed /tmp tmpfs, and every node stages the FULL set (all ranks
sample all clips, so there is no per-node slice like the WebDataset stager has).

This mirrors scripts/stage_node_shards.py: run one MPI rank per node,
    mpiexec -n NUM_NODES -ppn 1 --cpu-bind none \
        python scripts/stage_probe_clips.py \
            --config configs/heads/.../<probe>.yaml \
            --local-root /tmp/vjepa_data/$PBS_JOBID

For each node it:
  1. reads dataset_train / dataset_val CSVs from the probe config,
  2. copies every referenced clip into <local-root>/clips/<rel-path>, where
     rel-path is the clip path with its common Lustre prefix stripped,
  3. writes prefix-swapped CSVs into <local-root>/csv/<name>.csv pointing at the
     staged clips.
app/main_dist_aurora.py then repoints experiment.data.dataset_{train,val} at the
local CSVs (only if they exist, so unstaged runs are untouched).

Idempotent: clips already present with the right size are skipped (atomic
.part + rename), so a resubmit into the same $PBS_JOBID root is cheap.
"""

import argparse
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml


def _node_rank():
    """Identify this node's index (0..num_nodes-1) when launched -ppn 1.

    Same logic as stage_node_shards.py: with `mpiexec -n N -ppn 1` every MPI
    process IS the node it runs on, so the per-node MPI rank is the node index.
    """
    for var in ("PALS_RANKID", "PMI_RANK", "PMIX_RANK", "OMPI_COMM_WORLD_RANK"):
        v = os.environ.get(var)
        if v is not None:
            return int(v)
    # Single-node interactive fallback.
    return 0


def _copy_one(src, dst):
    """Copy src->dst atomically, skipping if an equal-size dst already exists."""
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        return False
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".part"
    shutil.copy2(src, tmp)
    os.rename(tmp, dst)
    return True


def _read_clip_paths(csv_path):
    """First whitespace-delimited token of each non-empty line is the clip path.

    Matches VideoDataset's ` `-delimited parse (video_dataset.py:214). We only
    need column 0 (the path); labels stay untouched in the rewritten CSV.
    """
    paths = []
    with open(csv_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            paths.append(line.split(" ", 1)[0])
    return paths


def _common_prefix(paths):
    """Longest common DIRECTORY prefix of the clip paths (dir granularity, so
    we never split mid-filename)."""
    cp = os.path.commonpath(paths)
    return cp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="probe eval YAML")
    p.add_argument("--local-root", required=True,
                   help="Destination root on this node, e.g. /tmp/vjepa_data/$PBS_JOBID")
    p.add_argument("--workers", type=int, default=16,
                   help="Parallel copy threads")
    args = p.parse_args()

    node_rank = _node_rank()
    host = os.environ.get("HOSTNAME", "?")
    t0 = time.time()
    print(f"[node {node_rank} {host}] probe-clip staging start {time.strftime('%F %T')}",
          flush=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    edata = cfg.get("experiment", {}).get("data", {})
    csvs = [edata.get("dataset_train"), edata.get("dataset_val")]
    csvs = [c for c in csvs if c]
    if not csvs:
        print(f"[node {node_rank}] no dataset_train/val in config; nothing to do",
              flush=True)
        return

    # Gather the union of clip paths across the CSVs, and their common prefix so
    # relative structure (train1/video_01/...) is preserved under local clips/.
    all_paths = []
    per_csv_paths = {}
    for c in csvs:
        pths = _read_clip_paths(c)
        per_csv_paths[c] = pths
        all_paths.extend(pths)
    uniq = sorted(set(all_paths))
    prefix = _common_prefix(uniq)
    print(f"[node {node_rank}] {len(uniq)} unique clips, common prefix {prefix}",
          flush=True)

    clips_root = os.path.join(args.local_root, "clips")
    csv_root = os.path.join(args.local_root, "csv")
    os.makedirs(clips_root, exist_ok=True)
    os.makedirs(csv_root, exist_ok=True)

    def _local_clip(src):
        rel = os.path.relpath(src, prefix)
        return os.path.join(clips_root, rel)

    # Preflight: sum bytes vs /tmp free, fail fast (matches stage_node_shards).
    need = 0
    for src in uniq:
        try:
            need += os.path.getsize(src)
        except OSError:
            pass
    try:
        st = os.statvfs(args.local_root)
        free = st.f_bavail * st.f_frsize
    except Exception:
        free = -1
    print(f"[node {node_rank}] preflight: need ~{need/2**30:.1f} GiB, "
          f"/tmp free ~{free/2**30:.1f} GiB", flush=True)
    if free >= 0 and need > free * 0.95:
        raise SystemExit(
            f"[node {node_rank}] ABORT: staging needs ~{need/2**30:.1f} GiB but "
            f"/tmp has ~{free/2**30:.1f} GiB free."
        )

    # Copy clips in parallel.
    copied = 0
    missing = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {}
        for src in uniq:
            if not os.path.exists(src):
                missing += 1
                continue
            futs[ex.submit(_copy_one, src, _local_clip(src))] = src
        for fut in as_completed(futs):
            if fut.result():
                copied += 1
    if missing:
        print(f"[node {node_rank}] WARNING: {missing} clip paths missing at source",
              flush=True)

    # Write prefix-swapped CSVs pointing at the local clips.
    for c, pths in per_csv_paths.items():
        out = os.path.join(csv_root, os.path.basename(c))
        with open(c) as fin, open(out + ".part", "w") as fout:
            for line in fin:
                stripped = line.rstrip("\n")
                if not stripped.strip():
                    fout.write(line)
                    continue
                path, sep, rest = stripped.partition(" ")
                fout.write(_local_clip(path) + sep + rest + "\n")
        os.rename(out + ".part", out)
        print(f"[node {node_rank}] wrote local CSV {out}", flush=True)

    dt = time.time() - t0
    try:
        free = os.statvfs(args.local_root)
        free_gb = free.f_bavail * free.f_frsize / 2**30
    except Exception:
        free_gb = -1
    print(f"[node {node_rank}] DONE {len(uniq)} clips ({copied} newly copied) "
          f"in {dt:.1f}s; /tmp free: {free_gb:.1f} GiB", flush=True)


if __name__ == "__main__":
    main()
