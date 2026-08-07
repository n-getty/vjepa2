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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from src.datasets.shard_window import shards_for_node  # noqa: E402


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


# Minimum shards a node keeps per source under --partition-mode nodes. The
# loader slices the node's staged dir by LOCAL rank (12) and then again by
# DataLoader worker (num_workers=2), so a node needs >= 12*2 shards for every
# worker to get a non-degenerate pool. Below that, workers re-read one tar.
SHARDS_PER_NODE_MIN = 24


def _shards_for_node(num_shards, node_rank, num_nodes, local_world_size):
    """Indices of shards this node should hold given the loader's slicing.

    Legacy (default) partition: disjoint slice keyed on the GLOBAL world size.
    A source with fewer shards than world_size is replicated in full on every
    node -- which is what makes this mode unusable past ~32 nodes (at 256n,
    world_size=3072 exceeds every source's shard count, so the whole corpus
    lands on every node's /tmp). Kept as the default so the in-flight 16n
    chains keep staging byte-identical shard sets across resumes.
    """
    world_size = num_nodes * local_world_size
    if num_shards < world_size:
        return list(range(num_shards))
    return [
        i for i in range(num_shards)
        if (i % world_size) // local_world_size == node_rank
    ]


def _shards_for_node_bynode(num_shards, node_rank, num_nodes,
                            min_shards=SHARDS_PER_NODE_MIN, max_shards=0):
    """Indices of shards this node holds, partitioned by NODE COUNT.

    Thin wrapper over ``src.datasets.shard_window.shards_for_node`` -- see that
    module for the window definition and for why the same code has to serve
    both the stager and the DAOS-side ``VJEPA_SHARD_CAP`` reader (a capped
    staged arm and a capped DAOS arm are only comparable if they draw the
    identical shard set).

    Why a node-count partition is correct here, and why the global-world_size
    math was never needed: every production launcher sets
    ``WDS_LOCAL_SLICING=1``, so ``src/datasets/webdataset.py:_make_stream``
    slices the URL list by LOCAL rank/world (12) -- never by the 3072-rank
    global world. And ``_load_or_build_metadata`` re-lists the node's local dir
    and overwrites ``shard_urls`` with whatever is actually present, keeping
    ``sample_count`` from the copied metadata.json (so the ipe math is
    unaffected). The loader therefore consumes whatever subset a node holds:
    the staging partition is a free parameter, and the "source needs >=
    nodes*12 shards" threshold was an artifact of this function rather than a
    property of the data path.
    """
    return shards_for_node(num_shards, node_rank, num_nodes,
                           min_shards=min_shards, max_shards=max_shards)


def _copy_one(src, dst):
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        return dst, False  # already there
    tmp = dst + ".part"
    shutil.copy2(src, tmp)
    os.rename(tmp, dst)
    return dst, True


def _choose_shards(num_shards, node_rank, num_nodes, local_world_size,
                   partition_mode="world", min_shards=SHARDS_PER_NODE_MIN,
                   max_shards=0):
    """Dispatch to the legacy or by-node partition.

    ``max_shards`` applies only to the by-node partition. The legacy mode is
    left untouched by design: in-flight chains depend on it staging
    byte-identical shard sets across resumes.
    """
    if partition_mode == "nodes":
        return _shards_for_node_bynode(num_shards, node_rank, num_nodes,
                                       min_shards=min_shards,
                                       max_shards=max_shards)
    return _shards_for_node(num_shards, node_rank, num_nodes, local_world_size)


def stage_dataset(src_dir, dst_dir, node_rank, num_nodes, local_world_size,
                  num_workers=8, partition_mode="world",
                  min_shards=SHARDS_PER_NODE_MIN, max_shards=0):
    """Stage this node's slice of src_dir into dst_dir. Returns counts."""
    os.makedirs(dst_dir, exist_ok=True)
    shards = sorted(f for f in os.listdir(src_dir) if f.endswith(".tar"))
    chosen_idx = _choose_shards(len(shards), node_rank, num_nodes,
                                local_world_size, partition_mode, min_shards,
                                max_shards)
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
    p.add_argument("--partition-mode", choices=("world", "nodes"),
                   default="world",
                   help="'world' (default, legacy): disjoint slice keyed on "
                        "num_nodes*local_world_size; replicates any source with "
                        "fewer shards than that -- unusable past ~32 nodes. "
                        "'nodes': wraparound window keyed on node count with a "
                        "--min-shards-per-node floor; required for >=64 nodes.")
    p.add_argument("--min-shards-per-node", type=int,
                   default=SHARDS_PER_NODE_MIN,
                   help="Floor on shards/node under --partition-mode nodes. "
                        "Should be >= local_world_size * dataloader workers so "
                        "every worker gets a distinct shard.")
    p.add_argument("--src-root", default=None,
                   help="Read sources from <src-root>/<basename> instead of "
                        "the absolute paths in --params. Mirrors what "
                        "app/main_dist_aurora.py --local_data_root does on the "
                        "trainer side, so a DAOS-resident corpus can be staged "
                        "without editing the config.")
    p.add_argument("--max-shards-per-node", type=int, default=0,
                   help="Cap on shards/node/source under --partition-mode "
                        "nodes (0 = unlimited). THROUGHPUT ARMS ONLY -- this "
                        "deliberately breaks full-corpus coverage so a short "
                        "arm at small node count can be staged in minutes "
                        "instead of the ~90 min the S/N stride would need. "
                        "Never use it for a training run.")
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
    if args.src_root:
        sources = [os.path.join(args.src_root, os.path.basename(s.rstrip("/")))
                   for s in sources]

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
        idx = _choose_shards(len(shards), node_rank, args.num_nodes,
                             args.local_world_size, args.partition_mode,
                             args.min_shards_per_node,
                             args.max_shards_per_node)
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
        if args.partition_mode == "world":
            hint = (
                f"Likely a non-resharded dataset "
                f"(<{args.num_nodes * args.local_world_size} shards) being "
                f"copied in full per node. Use the _resharded dataset paths, "
                f"reduce the dataset set, or switch to "
                f"--partition-mode nodes (required past ~32 nodes)."
            )
        else:
            hint = (
                f"Under --partition-mode nodes each node takes "
                f"max(ceil(S/{args.num_nodes}), {args.min_shards_per_node}) "
                f"shards per source, so at small node counts the ceil(S/N) "
                f"STRIDE dominates (at N=2 that is half the corpus per node) "
                f"and at large N the floor does. Lower --min-shards-per-node "
                f"(>= local_world_size * dataloader workers), drop the largest "
                f"sources, or -- for a short throughput arm only, never a "
                f"training run -- cap coverage with --max-shards-per-node."
            )
        raise SystemExit(
            f"[node {node_rank}] ABORT: staging needs ~{need_gb:.1f} GiB but "
            f"/tmp has ~{free_gb:.1f} GiB free. {hint}"
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
                partition_mode=args.partition_mode,
                min_shards=args.min_shards_per_node,
                max_shards=args.max_shards_per_node,
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
