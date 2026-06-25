# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Unit tests for the degenerate (black/frozen) clip reject in VideoDecoder.

These isolate the filter logic by stubbing the actual video decode so no mp4
bytes / decord are required for the assertions about drop-vs-keep behavior.
"""
import numpy as np

from src.datasets import webdataset as wds_mod
from src.datasets.webdataset import VideoDecoder


def _make_decoder(min_clip_std):
    # frame_step path (exactly one of fps/duration/frame_step must be set).
    return VideoDecoder(
        frames_per_clip=4,
        frame_step=1,
        num_clips=1,
        transform=None,
        shared_transform=None,
        min_clip_std=min_clip_std,
    )


def _sample_with_buffer(decoder, buffer):
    """Build a fake wds sample and stub the decode to return `buffer`."""
    decoder.loadvideo_decord = lambda media_bytes, fpc: (buffer, [np.arange(fpc)])
    return {
        "__key__": "fake/key",
        "label.txt": b"3",
        "video.mp4": b"\x00\x00\x00",  # bytes content irrelevant (decode stubbed)
    }


def test_black_clip_is_dropped():
    decoder = _make_decoder(min_clip_std=1.0)
    black = np.zeros((4, 16, 16, 3), dtype=np.uint8)  # std == 0
    sample = _sample_with_buffer(decoder, black)
    assert decoder.decode(sample, 4, source_name="surgvu24") is None


def test_real_clip_is_kept():
    decoder = _make_decoder(min_clip_std=1.0)
    rng = np.random.RandomState(0)
    real = rng.randint(0, 256, size=(4, 16, 16, 3)).astype(np.uint8)  # std ~74
    sample = _sample_with_buffer(decoder, real)
    out = decoder.decode(sample, 4, source_name="kinetics400")
    assert out is not None
    buffer, label, clip_indices = out
    assert label == 3
    assert len(buffer) == 1  # num_clips


def test_near_black_below_floor_dropped():
    # A clip with tiny but nonzero variance (e.g. sensor noise floor) under the
    # threshold is still dropped.
    decoder = _make_decoder(min_clip_std=1.0)
    near = np.zeros((4, 16, 16, 3), dtype=np.uint8)
    near[0, 0, 0, 0] = 1  # std ~ 0.01, well below 1.0
    sample = _sample_with_buffer(decoder, near)
    assert decoder.decode(sample, 4, source_name="surgvu24") is None


def test_filter_disabled_keeps_black():
    # min_clip_std == 0 disables the reject (escape hatch).
    decoder = _make_decoder(min_clip_std=0.0)
    black = np.zeros((4, 16, 16, 3), dtype=np.uint8)
    sample = _sample_with_buffer(decoder, black)
    out = decoder.decode(sample, 4, source_name="surgvu24")
    assert out is not None


def test_drop_keep_tallies_recorded():
    # The instrumentation counters increment per source.
    wds_mod._clip_diag["dropped"].clear()
    wds_mod._clip_diag["kept"].clear()
    wds_mod._clip_diag["seen"] = 0
    decoder = _make_decoder(min_clip_std=1.0)

    black = np.zeros((4, 16, 16, 3), dtype=np.uint8)
    decoder.decode(_sample_with_buffer(decoder, black), 4, source_name="surgvu24")

    rng = np.random.RandomState(1)
    real = rng.randint(0, 256, size=(4, 16, 16, 3)).astype(np.uint8)
    decoder.decode(_sample_with_buffer(decoder, real), 4, source_name="surgvu24")

    assert wds_mod._clip_diag["dropped"].get("surgvu24") == 1
    assert wds_mod._clip_diag["kept"].get("surgvu24") == 1
