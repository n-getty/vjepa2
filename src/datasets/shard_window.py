"""Which shards of a source a given node holds.

One function, two callers, and they MUST agree:

  * ``scripts/stage_node_shards.py`` uses it to decide what to copy to /tmp.
  * ``src/datasets/webdataset.py`` uses it (under ``VJEPA_SHARD_CAP``) to
    restrict the URL list a node draws from when reading DAOS directly.

The second caller exists only to make a controlled storage A/B possible. A
staged arm reads a *capped* window off local tmpfs; comparing it to a
full-corpus DAOS arm changes the storage path AND the working-set size at
once, so a staged win would not separate "the DAOS path is slow" from "the
working set now fits in page cache". The control is a third arm that reads the
IDENTICAL window from DAOS. That only works if both sides compute the window
the same way -- hence one shared function rather than two copies that drift.
"""


def shards_for_node(num_shards, node_rank, num_nodes, min_shards=24,
                    max_shards=0):
    """Indices of shards node ``node_rank`` holds, partitioned by NODE COUNT.

    Each node takes a contiguous wraparound window of
    ``max(ceil(S/N), min_shards)`` shards starting at ``floor(n*S/N)``.

    Windows are sized >= the S/N stride, so their union covers every shard.
    Windows OVERLAP between nodes once ``min_shards`` binds (small sources) --
    intended and harmless: distinct nodes holding the same shard still draw
    different clips from it (``resampled=True`` + per-rank shuffle).

    ``max_shards`` (0 = unlimited) caps the window from ABOVE, and unlike the
    floor it BREAKS full-corpus coverage on purpose. It exists for short
    throughput arms, where the corpus is not the point: a 100-iteration arm at
    24 ranks x bs=2 consumes 4800 clips total, but the S/N stride at small N is
    enormous (at N=2 each node's window is HALF the corpus, ~2317 GiB, ~90 min
    to stage at the measured 0.43 GB/s/node -- longer than the queue slot).
    Never use it for a training run: the run would see only a fixed prefix of
    each source.

    The cap keeps each node's window START, so different nodes still hold
    different shards. Truncating to a common prefix instead would put every
    node on the same shards and quietly change what a page-cache arm measures.
    """
    if num_shards <= 0:
        return []
    stride = -(-num_shards // num_nodes)  # ceil(S/N)
    take = min(max(stride, min_shards), num_shards)
    if max_shards > 0:
        take = min(take, max_shards)
    start = (node_rank * num_shards) // num_nodes
    return sorted({(start + k) % num_shards for k in range(take)})
