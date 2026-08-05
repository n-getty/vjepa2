# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Determinism/correctness tests for the low-shot ``train_frac`` subsetting hook.

The hook selects a random ``train_frac`` fraction of TRAIN samples that must be:
  - identical across repeated calls and across distributed ranks (depends only on
    ``(n, seed)``), so every rank subsets to the same set before the sampler shards it;
  - a strict no-op when ``train_frac == 1.0`` (full-data path unchanged);
  - sorted (does not reorder the underlying dataset).
"""

import unittest

from src.datasets.video_dataset import seeded_subset_indices


class TestSeededSubsetIndices(unittest.TestCase):
    def test_deterministic_and_rank_invariant(self):
        # Same (n, seed) -> identical result, no matter how many times / "where" called.
        a = seeded_subset_indices(6031, 0.1, seed=0)
        b = seeded_subset_indices(6031, 0.1, seed=0)
        self.assertEqual(a, b)
        # There is no rank argument, so the selection is rank-invariant by construction.
        # Simulate "different rank" by interleaving other RNG usage; result must not change.
        import random

        random.random()
        c = seeded_subset_indices(6031, 0.1, seed=0)
        self.assertEqual(a, c)

    def test_sorted(self):
        a = seeded_subset_indices(6031, 0.1, seed=0)
        self.assertEqual(a, sorted(a))
        self.assertEqual(len(a), len(set(a)))  # no duplicates

    def test_sizes(self):
        self.assertEqual(len(seeded_subset_indices(6031, 0.1, seed=0)), round(6031 * 0.1))
        self.assertEqual(len(seeded_subset_indices(6031, 0.01, seed=0)), round(6031 * 0.01))

    def test_full_frac_is_noop(self):
        self.assertIsNone(seeded_subset_indices(6031, 1.0, seed=0))

    def test_seed_varies_selection(self):
        self.assertNotEqual(
            seeded_subset_indices(6031, 0.1, seed=0),
            seeded_subset_indices(6031, 0.1, seed=1),
        )

    def test_tiny_fraction_floor(self):
        # Never returns an empty subset.
        self.assertEqual(len(seeded_subset_indices(5, 0.01, seed=0)), 1)

    def test_invalid_frac(self):
        for bad in (0.0, -0.1, 1.5):
            with self.assertRaises(ValueError):
                seeded_subset_indices(10, bad, seed=0)


class TestShardOrderSamplerSubset(unittest.TestCase):
    """The cached probe sampler must honor ``subset_indices`` while preserving
    shard-contiguous order and rank sharding."""

    def _make_sampler(self, subset_indices, num_replicas, rank, shuffle=False):
        from src.datasets.backbone_feature_cache import ShardOrderDistributedSampler

        # Minimal fake dataset exposing the two attributes the sampler reads.
        class _FakeCache:
            def __init__(self, shard_ranges):
                self.shard_ranges = shard_ranges
                self._n = shard_ranges[-1][1]

            def __len__(self):
                return self._n

        # 3 shards of 4 samples each -> global indices 0..11.
        ds = _FakeCache([(0, 4), (4, 8), (8, 12)])
        return ShardOrderDistributedSampler(
            ds,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            subset_indices=subset_indices,
        )

    def test_subset_restricts_indices(self):
        keep = [1, 3, 5, 9]
        s = self._make_sampler(keep, num_replicas=1, rank=0, shuffle=False)
        got = list(iter(s))
        # Only kept indices appear, in shard-contiguous order.
        self.assertEqual(sorted(got), keep)
        self.assertEqual(got, [1, 3, 5, 9])

    def test_subset_len_and_sharding(self):
        keep = [0, 2, 4, 6, 8, 10]  # 6 samples, 2 ranks -> 3 each
        r0 = list(iter(self._make_sampler(keep, num_replicas=2, rank=0, shuffle=False)))
        r1 = list(iter(self._make_sampler(keep, num_replicas=2, rank=1, shuffle=False)))
        self.assertEqual(len(r0), 3)
        self.assertEqual(len(r1), 3)
        # Together the two ranks cover the subset with no overlap.
        self.assertEqual(sorted(r0 + r1), keep)

    def test_none_subset_is_full(self):
        s = self._make_sampler(None, num_replicas=1, rank=0, shuffle=False)
        self.assertEqual(list(iter(s)), list(range(12)))


if __name__ == "__main__":
    unittest.main()
