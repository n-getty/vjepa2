"""Partition math for per-node WebDataset staging (scripts/stage_node_shards.py).

The by-node partition is what unblocks >=64-node training: the legacy partition
keys on the GLOBAL world size (nodes*12) and replicates any source with fewer
shards than that onto every node, so at 256 nodes (world_size=3072) the entire
corpus lands on every node's /tmp and the preflight aborts.

These tests pin the two properties the loader actually depends on:
  * COVERAGE -- the union over nodes is every shard (no data silently dropped).
  * FLOOR    -- each node holds >= min(S, min_shards) shards, so that after the
                loader's local-rank slice (12) and DataLoader worker slice (2)
                no worker is left with an empty or single-shard pool.
"""

import importlib.util
import os

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "stage_node_shards",
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts",
                 "stage_node_shards.py"),
)
sns = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sns)


# Live per-source shard counts for the v2 CPT corpus (2026-08-03).
CORPUS = {
    "small_surg": 105, "sitl": 600, "surgenet_robotic_clean": 413,
    "surgtoolloc2022": 1500, "surgvu24_clean": 2000, "grasp_noleak": 256,
    "cholec80": 256, "sitl_2026": 1454, "lemon": 528, "heichole_512": 256,
    "multibypass140": 818, "gynsurg": 256, "lapgyn6_events": 256,
    "surgenet_lap": 256, "openh": 2293, "pe_video": 1968,
}


@pytest.mark.parametrize("num_shards", [1, 5, 24, 105, 256, 528, 2293])
@pytest.mark.parametrize("num_nodes", [1, 16, 64, 256])
def test_bynode_covers_every_shard(num_shards, num_nodes):
    """Union over all nodes must be the full shard set -- no dropped data."""
    seen = set()
    for n in range(num_nodes):
        seen.update(sns._shards_for_node_bynode(num_shards, n, num_nodes))
    assert seen == set(range(num_shards))


@pytest.mark.parametrize("num_shards", [1, 5, 24, 105, 256, 2293])
@pytest.mark.parametrize("num_nodes", [1, 16, 64, 256])
def test_bynode_respects_floor_and_bounds(num_shards, num_nodes):
    """Every node gets >= min(S, floor) shards, in range, without duplicates."""
    floor = sns.SHARDS_PER_NODE_MIN
    for n in range(num_nodes):
        idx = sns._shards_for_node_bynode(num_shards, n, num_nodes)
        assert len(idx) == len(set(idx)), "duplicate shard index on one node"
        assert all(0 <= i < num_shards for i in idx), "index out of range"
        assert len(idx) >= min(num_shards, floor)


def test_bynode_disjoint_when_floor_does_not_bind():
    """With plenty of shards the windows are a clean disjoint partition."""
    num_shards, num_nodes = 2048, 16  # 128 per node >> the 24 floor
    windows = [set(sns._shards_for_node_bynode(num_shards, n, num_nodes))
               for n in range(num_nodes)]
    for a in range(num_nodes):
        for b in range(a + 1, num_nodes):
            assert not (windows[a] & windows[b])


def test_bynode_overlaps_only_where_floor_binds():
    """Small sources overlap across nodes -- intended, and the reason 256n works."""
    # 105 shards over 256 nodes: stride 1, floor 24 -> every node holds 24.
    idx = sns._shards_for_node_bynode(105, 7, 256)
    assert len(idx) == 24


def test_legacy_partition_unchanged():
    """The default mode must stay byte-identical for the in-flight 16n chains."""
    num_nodes, lws = 16, 12
    for num_shards in (105, 256, 600, 2293):
        for n in (0, 5, 15):
            got = sns._choose_shards(num_shards, n, num_nodes, lws,
                                     partition_mode="world")
            expect = sns._shards_for_node(num_shards, n, num_nodes, lws)
            assert got == expect
    # And the replicate-everything fallback still triggers below world_size.
    assert sns._shards_for_node(191, 3, 16, 12) == list(range(191))


def test_256n_fits_in_tmp():
    """The whole point: the v2 corpus must fit a node's /tmp at 256 nodes.

    Sizes are per-source averages (total bytes / shard count) from disk, so the
    estimate is the shard COUNT times the mean shard size -- good to ~10%.
    """
    avg_gb = {  # measured 2026-08-03, GB per shard
        "small_surg": 0.01, "sitl": 0.148, "surgenet_robotic_clean": 0.048,
        "surgtoolloc2022": 0.135, "surgvu24_clean": 0.160, "grasp_noleak": 0.402,
        "cholec80": 0.270, "sitl_2026": 0.378, "lemon": 1.744,
        "heichole_512": 0.051, "multibypass140": 0.449, "gynsurg": 0.176,
        "lapgyn6_events": 0.203, "surgenet_lap": 0.148, "openh": 0.154,
    }
    per_node = sum(
        len(sns._shards_for_node_bynode(CORPUS[s], 0, 256)) * gb
        for s, gb in avg_gb.items()
    )
    # ~503 GiB usable /tmp; the stager aborts above 95% of free space.
    assert per_node < 400, f"per-node staging {per_node:.0f} GB too close to /tmp"


def test_legacy_partition_would_overflow_at_256n():
    """Guards the premise: legacy mode really does blow up at 256 nodes."""
    total = sum(CORPUS.values())
    staged = sum(len(sns._shards_for_node(s, 0, 256, 12)) for s in CORPUS.values())
    assert staged == total, "every source replicates in full under legacy at 256n"
