"""Unit tests for WebDataset source-mixing probabilities.

Guards root cause #2: uniform-per-source RandomMix probs oversampled tiny
datasets by ~10^4 (an 8-clip set got the same training share as a 136k-clip
set), collapsing the effective corpus onto a few memorized clips. These tests
pin the temperature-sampling semantics that fix it.
"""
import math

import pytest

from src.datasets.webdataset import compute_mixing_probs


# Realistic phase-1 surgical corpus (sample_count per dataset).
CORPUS = [207, 8, 206, 136246, 18, 15, 13969, 3624, 24686, 472, 51870]


def test_probs_sum_to_one():
    for T in (0.0, 0.3, 0.5, 1.0):
        probs = compute_mixing_probs(CORPUS, temperature=T)
        assert math.isclose(sum(probs), 1.0, rel_tol=1e-9)
        assert len(probs) == len(CORPUS)
        assert all(p >= 0 for p in probs)


def test_temperature_1_is_size_proportional():
    """T=1 => realized fraction == true corpus fraction (no distortion)."""
    probs = compute_mixing_probs(CORPUS, temperature=1.0)
    total = sum(CORPUS)
    for p, c in zip(probs, CORPUS):
        assert math.isclose(p, c / total, rel_tol=1e-9)


def test_temperature_0_is_uniform_per_source():
    """T=0 reproduces the OLD broken behavior (each source 1/N)."""
    probs = compute_mixing_probs(CORPUS, temperature=0.0)
    n = len(CORPUS)
    for p in probs:
        assert math.isclose(p, 1.0 / n, rel_tol=1e-9)


def test_tiny_set_oversampling_shrinks_with_temperature():
    """The 8-clip set's share must fall monotonically as T rises 0->1."""
    idx = CORPUS.index(8)
    shares = [compute_mixing_probs(CORPUS, temperature=T)[idx]
              for T in (0.0, 0.3, 0.5, 1.0)]
    assert shares == sorted(shares, reverse=True)
    # Old behavior gave it ~9%; T=0.5 must cut it by >10x.
    assert shares[0] > 0.08          # T=0 ~ 1/11
    assert shares[2] < 0.005         # T=0.5


def test_default_temperature_is_safe():
    """The function default must NOT be the degenerate uniform-per-source."""
    probs_default = compute_mixing_probs(CORPUS)
    probs_half = compute_mixing_probs(CORPUS, temperature=0.5)
    assert probs_default == probs_half


def test_datasets_weights_multiply_on_top():
    """Explicit per-source weights scale the temperature-based base."""
    counts = [100, 100]            # equal size, so temperature is neutral
    probs = compute_mixing_probs(counts, datasets_weights=[3.0, 1.0],
                                 temperature=1.0)
    assert math.isclose(probs[0] / probs[1], 3.0, rel_tol=1e-9)


def test_zero_count_does_not_crash():
    probs = compute_mixing_probs([0, 100], temperature=1.0)
    assert math.isclose(sum(probs), 1.0, rel_tol=1e-9)
    # count clamped to >=1, so the empty source gets a tiny but finite share.
    assert probs[0] > 0


def test_weights_length_mismatch_raises():
    with pytest.raises(ValueError):
        compute_mixing_probs([1, 2, 3], datasets_weights=[1.0, 1.0])


def test_empty_input():
    assert compute_mixing_probs([]) == []
