"""Per-size launch topology for the JEPA scaling sweep.

The global batch is HELD FIXED (=96) across the whole sweep — that is the D-vs-N invariant. But
`global_batch = tiles * per_rank_bs`, and how we FACTOR it is free and per-size:

  - small models (tiny/small/base) are OVERHEAD-bound at 256px (pilot: fwd-target flat ~520ms across a
    15x param range). Fewest tiles + largest per_rank_bs that fits one tile => least fixed DDP comm,
    best amortization of the per-iter floor. Multiple small cells then PACK onto one node's 12 tiles.
  - large is compute-bound and fits comfortably => DDP across a node.
  - giant/gigantic need HSDP (VJEPA_DIST_STRATEGY=hsdp): intra-node param/grad/opt-state sharding buys
    L0 headroom and removes the 2B backward wedge (docs/vitG_2B_HSDP_findings.md). ckpt-off is only
    safe under HSDP; DDP would OOM the tile.

This module maps model_name -> a LaunchSpec (tiles, per_rank_bs, dist_strategy, env, nodes). The
per_rank_bs ceilings come from the calibration sweep (scaling/read_calib.py); until those land we use
conservative defaults and expose `set_max_bs()` so the calibrated table is a one-line update.

Pure/side-effect-free so it unit-tests without hardware.
"""

from dataclasses import dataclass, field

GLOBAL_BATCH = 96
TILES_PER_NODE = 12

# Max per-rank batch per model at 256px on one XPU tile. VALIDATED by calibration (job 8660398,
# 2026-07-10): all listed batches ran clean (rc=0, no OOM) — these are SAFE, not necessarily maximal
# (none hit the memory ceiling; large bs1 had 53 GB free). Strong batch-amortization measured:
# clips/s/tile rose 7x (small bs2->32), 5.2x (base bs2->16), 3.9x (large bs1->4) as the ~520ms fixed
# per-iter overhead got amortized. giant/gigantic NOT directly calibrated (need HSDP+multi-tile);
# values are the design defaults pending an HSDP calib pass.
_MAX_BS = {
    "vit_tiny": 32,     # 10.92 clips/s/tile @ bs32 (131/node)
    "vit_small": 32,    # 10.19 clips/s/tile @ bs32 (122/node)
    "vit_base": 16,     # 6.53 clips/s/tile @ bs16 (78/node)
    "vit_large": 4,     # 1.48 clips/s/tile @ bs4 (18/node); DDP fine, 53GB headroom -> no HSDP
    "vit_giant": 2,     # HSDP bs2 proven at 256px (2n OFI smoke ran the HEAVIER gigantic@384/bs2)
    "vit_gigantic": 2,  # was 1 (uncalibrated); the 2n OFI smoke ran gigantic-class @384/bs2 on a
                        # tile -> 256px/bs2 fits with ~2.25x token margin. bs2 also keeps the
                        # d_weights loss off the per-rank bs==1 landmine (d_ij.unsqueeze(2)).
}

# Which sizes get HSDP (the rest DDP). Calibration/OOM may push 'large' here too.
_HSDP = {"vit_giant", "vit_gigantic"}

# OFI transport env for HSDP cells (baked into each cell's _launch.json "env" and re-exported by
# scaling.overnight_chain.emit_launch_block before mpiexec). This OVERRIDES the chain's global
# AURORA_ENV, which sets the DDP transport (CCL_PROCESS_LAUNCHER=pmix + CCL_ATL_TRANSPORT=mpi +
# CCL_KVS_MODE=mpi). HSDP needs launcher=none + ofi + a fabric KVS on hsn0 (PRISM-validated; matches
# the 2n giant/gigantic OFI smoke). The empty CCL_KVS_* strings NEUTRALIZE the global mpi-KVS vars.
# NOTE: this env change is necessary but NOT sufficient — emit_launch_block ALSO drops `--pmi=pmix`
# from mpiexec for dist_strategy=="hsdp" (launcher=none conflicts with a pmix PMI). Keep both in sync.
_HSDP_OFI_ENV = {
    "CCL_PROCESS_LAUNCHER": "none",
    "CCL_ATL_TRANSPORT": "ofi",
    "CCL_KVS_IFACE": "hsn0",
    "CCL_KVS_MODE": "",              # unset global =mpi
    "CCL_KVS_USE_MPI_RANKS": "",     # unset global =1
    "FI_CXI_RX_MATCH_MODE": "hybrid",
    "FI_CXI_OFLOW_BUF_SIZE": "8388608",
    "FI_CXI_DEFAULT_CQ_SIZE": "131072",
    "FI_MR_CACHE_MONITOR": "disabled",
    "PYTORCH_ALLOC_CONF": "garbage_collection_threshold:0.95",
    "CCL_ZE_CACHE_OPEN_IPC_HANDLES_THRESHOLD": "65536",
    "MPICH_GPU_SUPPORT_ENABLED": "1",
    "LOCAL_WORLD_SIZE": "12",        # also derivable from PALS_LOCAL_SIZE; set explicit for the mesh
    "FSDP_SHARDING": "shard_grad_op",
}


@dataclass
class LaunchSpec:
    model_name: str
    tiles: int              # ranks in this cell's world
    per_rank_bs: int        # clips per tile per iter (tiles*per_rank_bs == GLOBAL_BATCH)
    dist_strategy: str      # "ddp" | "hsdp"
    nodes: int              # whole nodes this cell needs (tiles/TILES_PER_NODE, >=1 for hsdp)
    env: dict = field(default_factory=dict)   # extra per-cell env (VJEPA_DIST_STRATEGY etc.)

    def global_batch(self):
        return self.tiles * self.per_rank_bs


def set_max_bs(table):
    """Overwrite the per-tile max-batch table from calibration results."""
    _MAX_BS.update(table)


@dataclass
class LaunchSpec2(LaunchSpec):
    accum: int = 1          # gradient-accumulation micro-steps (tiles*per_rank_bs*accum == GLOBAL_BATCH)

    def global_batch(self):
        return self.tiles * self.per_rank_bs * self.accum


def topology_for(model_name, global_batch=GLOBAL_BATCH, max_nodes=1):
    """Return the LaunchSpec for a model at the fixed global batch, bounded to <= max_nodes.

    Policy (the whole point is: hold global batch fixed, minimize scheduling pain):
      * SMALL/overhead-bound models: fewest tiles (=least DDP comm), per_rank_bs up to the tile max,
        multiple cells PACK onto a node. Never need >1 node.
      * BIG/memory-tight models: cap tiles at `max_nodes` nodes, then make up the rest of the global
        batch with gradient accumulation (VJEPA_TRUE_ACCUM). This keeps a cell on a bounded node count
        (easy to schedule, HSDP stays intra-node) at the cost of accum x wall-clock. Node-HOURS are
        ~identical to spreading across many nodes (same total FLOPs), so accum trades wall-clock for
        node-count, not total compute. Set max_nodes higher to prefer wall-clock over node-frugality.
    """
    max_bs = _MAX_BS.get(model_name)
    if max_bs is None:
        raise ValueError(f"no calibrated max_bs for {model_name}; run calibration or set_max_bs()")
    hsdp = model_name in _HSDP
    max_tiles = max_nodes * TILES_PER_NODE

    # tile counts that divide the global batch, ascending (fewest tiles first), within the node cap
    candidates = [t for t in range(1, global_batch + 1)
                  if global_batch % t == 0 and t <= max_tiles]
    if hsdp:
        candidates = [t for t in candidates if t % TILES_PER_NODE == 0]  # whole nodes for intra-node shard

    def _spec(tiles, per_rank, accum):
        nodes = max(1, -(-tiles // TILES_PER_NODE))
        env = {}
        if accum > 1:
            env["VJEPA_TRUE_ACCUM"] = str(accum)
        if hsdp:
            env["VJEPA_DIST_STRATEGY"] = "hsdp"
            env["VJEPA_NUM_WORKERS"] = "0"   # no persistent workers after mesh init (deadlock guard)
            env.update(_HSDP_OFI_ENV)        # launcher=none + ofi transport (see _HSDP_OFI_ENV note)
        return LaunchSpec2(model_name, tiles, per_rank, "hsdp" if hsdp else "ddp", nodes, env, accum)

    # 1) try to hit global batch with NO accum, fewest tiles
    for tiles in candidates:
        per_rank = global_batch // tiles
        if per_rank <= max_bs:
            return _spec(tiles, per_rank, 1)

    # 2) can't fit within max_nodes without accum. Use the MOST tiles allowed, per_rank=max_bs, and
    #    accum to close the gap. tiles*per_rank must divide global_batch for a whole accum count.
    for tiles in reversed(candidates):
        for per_rank in range(max_bs, 0, -1):
            micro = tiles * per_rank
            if micro and global_batch % micro == 0:
                accum = global_batch // micro
                return _spec(tiles, per_rank, accum)
    raise ValueError(f"{model_name}: cannot factor global_batch={global_batch} within "
                     f"max_nodes={max_nodes} at max_bs={max_bs}")


def packing_plan(specs):
    """Given a list of LaunchSpecs, greedily PACK sub-node cells (tiles<12) onto shared nodes.

    Returns a list of 'node groups': each is a list of (spec, tile_offset) that co-reside on one node
    (their tiles sum to <= TILES_PER_NODE). Whole-node/multi-node cells (hsdp, or tiles multiple of 12)
    each get their own dedicated node(s) and are returned as singleton groups. This is what lets the
    entire small end of a budget land on 1-2 nodes instead of one-node-per-cell.
    """
    small = [s for s in specs if s.tiles < TILES_PER_NODE]
    big = [s for s in specs if s.tiles >= TILES_PER_NODE]
    # first-fit-decreasing on tile footprint
    small.sort(key=lambda s: s.tiles, reverse=True)
    nodes = []  # each: {"used": int, "cells": [(spec, offset)]}
    for s in small:
        placed = False
        for nd in nodes:
            if nd["used"] + s.tiles <= TILES_PER_NODE:
                nd["cells"].append((s, nd["used"]))
                nd["used"] += s.tiles
                placed = True
                break
        if not placed:
            nodes.append({"used": s.tiles, "cells": [(s, 0)]})
    groups = [nd["cells"] for nd in nodes]
    for b in big:
        groups.append([(b, 0)])   # dedicated
    return groups


if __name__ == "__main__":
    # quick self-check / preview of the current policy
    ladder = ["vit_tiny", "vit_small", "vit_base", "vit_large", "vit_giant", "vit_gigantic"]
    specs = []
    print(f"global_batch={GLOBAL_BATCH}, tiles/node={TILES_PER_NODE}")
    print(f"{'model':14s} {'tiles':>5} {'bs':>4} {'strategy':>8} {'nodes':>5}  env")
    for m in ladder:
        s = topology_for(m)
        specs.append(s)
        assert s.global_batch() == GLOBAL_BATCH, (m, s.global_batch())
        print(f"{m:14s} {s.tiles:5d} {s.per_rank_bs:4d} {s.dist_strategy:>8} {s.nodes:5d}  {s.env}")
    print("\npacking plan (small cells share nodes):")
    for i, grp in enumerate(packing_plan(specs)):
        cells = ", ".join(f"{sp.model_name}(t{sp.tiles}@{off})" for sp, off in grp)
        print(f"  node group {i}: {cells}")
