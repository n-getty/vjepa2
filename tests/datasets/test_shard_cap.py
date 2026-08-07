"""VJEPA_SHARD_CAP: the DAOS-side reader must draw the SAME window the stager copies.

This is the control arm of the staged-vs-DAOS test for the dataload tail. The
staged arm has to be capped (the uncapped S/N window at N=2 is half the corpus,
~90 min of copying), so comparing it to a full-corpus DAOS arm would move the
storage path and the working-set size together. The separating control is a
DAOS arm reading the identical capped window -- which only works if both sides
compute the window with the same code and the same arguments.

These tests pin that agreement, and pin that the cap is OFF by default (a
capped training run would silently see a fixed prefix of every source).
"""

import importlib.util
import os

import pytest

from src.datasets.shard_window import shards_for_node

_SPEC = importlib.util.spec_from_file_location(
    "stage_node_shards",
    os.path.join(os.path.dirname(__file__), "..", "..", "scripts",
                 "stage_node_shards.py"),
)
sns = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sns)


@pytest.mark.parametrize("num_shards", [1, 5, 24, 105, 256, 2293])
@pytest.mark.parametrize("num_nodes", [1, 2, 16, 64])
@pytest.mark.parametrize("cap", [0, 24, 50, 200])
def test_stager_and_reader_agree(num_shards, num_nodes, cap):
    """The one property the three-arm design rests on."""
    for n in range(num_nodes):
        staged = sns._shards_for_node_bynode(num_shards, n, num_nodes,
                                             max_shards=cap)
        read = shards_for_node(num_shards, n, num_nodes, max_shards=cap)
        assert staged == read


@pytest.mark.parametrize("num_shards", [105, 256, 2293])
@pytest.mark.parametrize("num_nodes", [2, 16])
def test_cap_binds_from_above_and_keeps_the_floor(num_shards, num_nodes):
    """A cap shrinks the window but never below min(S, floor)."""
    floor = sns.SHARDS_PER_NODE_MIN
    for n in range(num_nodes):
        idx = shards_for_node(num_shards, n, num_nodes, max_shards=50)
        assert len(idx) <= 50
        assert len(idx) == len(set(idx))
        assert all(0 <= i < num_shards for i in idx)
        uncapped = len(shards_for_node(num_shards, n, num_nodes))
        assert len(idx) == min(50, uncapped)
        # The cap is allowed to cut below the floor -- that is what makes a
        # small arm affordable -- but only when the caller asked for less than
        # the floor. At cap=50 > floor=24 the floor still holds.
        assert len(idx) >= min(num_shards, floor)


def test_cap_preserves_distinct_windows_across_nodes():
    """Different nodes must still read DIFFERENT shards under a cap.

    Truncating every node to a common prefix instead would put all nodes on the
    same shards -- which for a page-cache arm is exactly the confound being
    controlled for, silently introduced by the control itself.
    """
    windows = [set(shards_for_node(2293, n, 16, max_shards=50))
               for n in range(16)]
    starts = {min(w) for w in windows}
    assert len(starts) == 16, "nodes collapsed onto the same window start"


def test_uncapped_still_covers_the_corpus():
    """cap=0 must leave the existing partition untouched (coverage intact)."""
    seen = set()
    for n in range(16):
        seen.update(shards_for_node(2293, n, 16, max_shards=0))
    assert seen == set(range(2293))


def test_cap_is_off_by_default():
    """A capped training run would see a fixed subset of every source."""
    import src.datasets.webdataset as wdsmod
    assert wdsmod._SHARD_CAP == 0, (
        "VJEPA_SHARD_CAP leaked into the environment or the default changed; "
        "capped reads are for throughput arms only"
    )


@pytest.mark.parametrize("rank,world,lws,expect", [
    (0, 12, 12, (0, 1)),
    (11, 12, 12, (0, 1)),
    (12, 24, 12, (1, 2)),
    (23, 24, 12, (1, 2)),
    (767, 768, 12, (63, 64)),
])
def test_node_identity_from_torch_world(rank, world, lws, expect, monkeypatch):
    """Node index/count come from the TORCH world, matching hsdp.py's mesh math."""
    import src.datasets.webdataset as wdsmod
    monkeypatch.setenv("LOCAL_WORLD_SIZE", str(lws))
    assert wdsmod._node_identity(rank, world) == expect


@pytest.mark.parametrize("rank,world,lws", [
    (None, None, 12),   # single-process loader construction
    (0, 12, 0),         # local world size unknown
    (0, 4, 12),         # nonsensical: world smaller than a node
])
def test_node_identity_degrades_to_one_node(rank, world, lws, monkeypatch):
    """A misdetected topology must give a smaller read set, never an exception."""
    import src.datasets.webdataset as wdsmod
    monkeypatch.setenv("LOCAL_WORLD_SIZE", str(lws))
    assert wdsmod._node_identity(rank, world) == (0, 1)
