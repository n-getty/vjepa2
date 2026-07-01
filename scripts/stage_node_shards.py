#!/usr/bin/env python3
"""Stage WebDataset shards onto this compute node's /tmp.

Each node only stages the union of shards its local ranks will read, given
the WebDataset loader's per-rank URL slicing (urls[rank::world_size]).

For node N (0..num_nodes-1) with local_world_size ranks per node:
    rank r in {N*local_world_size .. N*local_world_size + local_world_size - 1}
    rank r reads shards i where i % world_size == r
    node N's union: {i : (i % world_size) // local_world_size == N}

Datasets with fewer shards than world_size: every node stages all of them
(matches the loader's fallback in src/datasets/webdataset.py:_make_stream).

Designed to be run via:
    mpiexec -n NUM_NODES -ppn 1 --cpu-bind none \
        python scripts/stage_node_shards.py \
            --params /path/to/params-pretrain.yaml \
            --local-root /tmp/vjepa_data/$PBS_JOBID \
            --num-nodes 16 --local-world-size 12

Each MPI rank == one node; PALS_RANKID identifies which node-id we are.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml


def _node_rank():
    """Identify this node's index (0..num_nodes-1) when launched -ppn 1."""
    # When mpiexec -n N -ppn 1, every MPI process IS the node it runs on.
    # PALS_RANKID is the per-node MPI rank for Aurora's PALS launcher; with
    # -ppn 1 this equals the node index. PMI_RANK is the fallback on systems
    # that don't expose PALS vars.
    for var in ("PALS_RANKID", "PMI_RANK", "PMIX_RANK", "OMPI_COMM_WORLD_RANK"):
        v = os.environ.get(var)
        if v is not None:
            return int(v)
    raise RuntimeError(
        "could not find node rank: no PALS_RANKID/PMI_RANK/PMIX_RANK in env"
    )


def _shards_for_node(num_shards, node_rank, num_nodes, local_world_size):
    """Indices of shards this node should hold given the loader's slicing."""
    world_size = num_nodes * local_world_size
    if num_shards < world_size:
        return list(range(num_shards))
    return [
        i for i in range(num_shards)
        if (i % world_size) // local_world_size == node_rank
    ]


def _copy_one(src, dst):
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        return dst, False  # already there
    tmp = dst + ".part"
    shutil.copy2(src, tmp)
    os.rename(tmp, dst)
    return dst, True


def stage_dataset(src_dir, dst_dir, node_rank, num_nodes, local_world_size,
                  num_workers=8):
    """Stage this node's slice of src_dir into dst_dir. Returns counts."""
    os.makedirs(dst_dir, exist_ok=True)
    shards = sorted(f for f in os.listdir(src_dir) if f.endswith(".tar"))
    chosen_idx = _shards_for_node(len(shards), node_rank, num_nodes, local_world_size)
    chosen = [shards[i] for i in chosen_idx]
    if not chosen:
        return 0, 0
    copied = 0
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futs = [
            ex.submit(_copy_one, os.path.join(src_dir, n), os.path.join(dst_dir, n))
            for n in chosen
        ]
        for f in as_completed(futs):
            _, did_copy = f.result()
            if did_copy:
                copied += 1
    # Always copy metadata.json from source (if present). Without it, the
    # loader re-scans the local dir and writes a metadata reflecting only
    # this node's slice — silently shrinking sample_count for ipe math.
    src_meta = os.path.join(src_dir, "metadata.json")
    dst_meta = os.path.join(dst_dir, "metadata.json")
    if os.path.exists(src_meta):
        shutil.copy2(src_meta, dst_meta)
    return len(chosen), copied


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True, help="params-pretrain.yaml")
    p.add_argument("--local-root", required=True,
                   help="Destination root on this node, e.g. /tmp/vjepa_data/$PBS_JOBID")
    p.add_argument("--num-nodes", type=int, required=True)
    p.add_argument("--local-world-size", type=int, required=True,
                   help="Ranks per node (ppn passed to trainer)")
    p.add_argument("--workers", type=int, default=8,
                   help="Parallel copies per dataset")
    args = p.parse_args()

    node_rank = _node_rank()
    host = os.environ.get("HOSTNAME", "?")
    t0 = time.time()
    print(f"[node {node_rank}/{args.num_nodes} {host}] staging start {time.strftime('%F %T')}",
          flush=True)

    with open(args.params) as f:
        params = yaml.safe_load(f)
    sources = params.get("data", {}).get("datasets", [])
    if not sources:
        print(f"[node {node_rank}] no datasets in params; nothing to do", flush=True)
        return

    os.makedirs(args.local_root, exist_ok=True)

    # Preflight: sum the bytes THIS node will stage and compare to /tmp free
    # space, failing fast with an actionable message rather than dying mid-copy
    # with a cryptic OSError(28). The classic trip is pointing a big dataset at
    # a non-resharded dir with < world_size shards: the loader keeps the full
    # shard list per node, so a 300 GB dataset lands on every node's tmpfs.
    need_bytes = 0
    for src in sources:
        try:
            shards = sorted(f for f in os.listdir(src) if f.endswith(".tar"))
        except FileNotFoundError:
            continue
        idx = _shards_for_node(len(shards), node_rank, args.num_nodes,
                               args.local_world_size)
        for i in idx:
            try:
                need_bytes += os.path.getsize(os.path.join(src, shards[i]))
            except OSError:
                pass
    try:
        st = os.statvfs(args.local_root)
        free_bytes = st.f_bavail * st.f_frsize
    except Exception:
        free_bytes = -1
    need_gb = need_bytes / 2**30
    free_gb = free_bytes / 2**30 if free_bytes >= 0 else -1
    print(f"[node {node_rank}] preflight: need ~{need_gb:.1f} GiB, "
          f"/tmp free ~{free_gb:.1f} GiB", flush=True)
    if free_bytes >= 0 and need_bytes > free_bytes * 0.95:
        raise SystemExit(
            f"[node {node_rank}] ABORT: staging needs ~{need_gb:.1f} GiB but "
            f"/tmp has ~{free_gb:.1f} GiB free. Likely a non-resharded dataset "
            f"(<{args.num_nodes * args.local_world_size} shards) being copied "
            f"in full per node. Use the _resharded dataset paths, or reduce the "
            f"dataset set."
        )

    total_chosen = 0
    total_copied = 0
    for src in sources:
        name = os.path.basename(src.rstrip("/"))
        dst = os.path.join(args.local_root, name)
        ts = time.time()
        try:
            n_chosen, n_copied = stage_dataset(
                src, dst, node_rank, args.num_nodes, args.local_world_size,
                num_workers=args.workers,
            )
        except FileNotFoundError as e:
            print(f"[node {node_rank}] SKIP {name}: {e}", flush=True)
            continue
        dt = time.time() - ts
        print(
            f"[node {node_rank}] {name}: {n_chosen} shards on this node "
            f"({n_copied} newly copied) in {dt:.1f}s",
            flush=True,
        )
        total_chosen += n_chosen
        total_copied += n_copied

    # Free-space sanity report.
    try:
        st = os.statvfs(args.local_root)
        free_gb = st.f_bavail * st.f_frsize / 2**30
    except Exception:
        free_gb = -1
    dt = time.time() - t0
    print(
        f"[node {node_rank}] DONE {total_chosen} shards "
        f"({total_copied} new) in {dt:.1f}s; "
        f"/tmp free: {free_gb:.1f} GiB",
        flush=True,
    )


if __name__ == "__main__":
    main()
