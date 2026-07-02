#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Shared perceptual-hash (DCT pHash) helpers for LEMON dedup.

Used by scripts/build_phash_ref.py (builds the reference hash pools from the
yt_robotic_chole eval + surgenet_robotic training set) and by
scripts/segment_videos_to_wds.py (gates each LEMON source video against those
pools before segmenting, to avoid eval leakage / train duplication).

No `imagehash` dependency (not installed in the Aurora frameworks env) — DCT
pHash is implemented directly on top of opencv + numpy. Hashes are 64-bit and
stored as hex strings in JSON; matching is a vectorized Hamming distance over a
packed uint64 array so a candidate frame can be compared to ~thousands of
reference hashes in one numpy op.
"""
from __future__ import annotations

import json
from typing import List, Optional

import cv2
import numpy as np

HASH_SIZE = 8          # low-freq DCT block edge -> 8*8 = 64-bit hash
DCT_SIZE = 32          # resize edge before DCT
MATCH_BITS = 6         # Hamming <= this -> "same frame"
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def phash_gray(gray: np.ndarray) -> int:
    """DCT pHash of a single-channel uint8 frame -> 64-bit int.

    grayscale -> 32x32 -> 2D DCT -> low-freq 8x8 -> threshold at median
    (excluding the DC term) -> pack 64 bits MSB-first.
    """
    if gray.ndim == 3:
        gray = gray[..., 0]
    img = cv2.resize(gray, (DCT_SIZE, DCT_SIZE), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(img)
    low = d[:HASH_SIZE, :HASH_SIZE]
    med = np.median(low.flatten()[1:])  # drop DC [0,0] from the median
    bits = (low > med).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def pack_refs(hashes: List[int]) -> np.ndarray:
    """List of 64-bit ints -> np.uint64 array for vectorized Hamming."""
    if not hashes:
        return np.empty(0, dtype=np.uint64)
    return np.asarray(hashes, dtype=np.uint64)


def min_hamming(cand: int, ref: np.ndarray) -> int:
    """Minimum Hamming distance from cand (int) to any hash in ref (uint64[]).

    Returns 64 if ref is empty (i.e. no match possible).
    """
    if ref.size == 0:
        return 64
    x = np.bitwise_xor(ref, np.uint64(cand))
    xb = x.view(np.uint8).reshape(-1, 8)
    dists = _POP[xb].sum(axis=1)
    return int(dists.min())


def dup_fraction(cands: List[int], ref: np.ndarray, match_bits: int = MATCH_BITS) -> float:
    """Fraction of candidate hashes within match_bits of ANY reference hash."""
    if not cands or ref.size == 0:
        return 0.0
    n_match = sum(1 for c in cands if min_hamming(c, ref) <= match_bits)
    return n_match / len(cands)


def sample_gray_frames(path_or_file, n: int = 48, seek: bool = True,
                       max_decode: int = 2048) -> List[np.ndarray]:
    """Decode ~n grayscale frames spread across a video.

    seek=True  (default, for LONG clips e.g. 60s LEMON/staged): seek to n
      evenly-spaced timestamps and decode one frame at each (cheap, avoids
      full decode).
    seek=False (for SHORT clips e.g. 4s eval windows): decode sequentially up
      to max_decode frames, then subsample n evenly-spaced. Faster than n seeks
      on a tiny clip and gives dense, deterministic coverage.

    Accepts a filesystem path or a file-like object (tarfile.extractfile).
    Raises on a genuinely unreadable/corrupt stream — the caller treats an
    exception (or an empty return) as "corrupt, skip". av imported lazily.
    """
    import av

    frames: List[np.ndarray] = []
    c = av.open(path_or_file)
    try:
        try:
            vs = c.streams.video[0]
        except (IndexError, KeyError):
            return frames
        total_s: Optional[float] = None
        if vs.duration is not None and vs.time_base is not None:
            total_s = float(vs.duration * vs.time_base)
        if (total_s is None or total_s <= 0) and c.duration:
            total_s = float(c.duration) / 1_000_000.0  # AV_TIME_BASE microseconds

        if seek and total_s and total_s > 1.0 and vs.time_base:
            for i in range(n):
                t = total_s * (i + 0.5) / n
                ts = int(t / float(vs.time_base))
                try:
                    c.seek(ts, stream=vs, any_frame=False, backward=True)
                    for fr in c.decode(vs):
                        frames.append(fr.to_ndarray(format="gray"))
                        break
                except Exception:
                    continue
        else:
            # Sequential: decode (capped) then subsample n evenly-spaced.
            buf = []
            for fr in c.decode(vs):
                buf.append(fr.to_ndarray(format="gray"))
                if len(buf) >= max_decode:
                    break
            if len(buf) <= n:
                frames = buf
            else:
                idx = [int(len(buf) * (i + 0.5) / n) for i in range(n)]
                frames = [buf[j] for j in idx]
    finally:
        c.close()
    return frames


def load_ref(path: str):
    """Load a phash_ref.json -> (eval_uint64[], train_uint64[], meta dict)."""
    with open(path) as f:
        d = json.load(f)
    ev = pack_refs([int(h, 16) for h in d.get("eval_hashes", [])])
    tr = pack_refs([int(h, 16) for h in d.get("train_hashes", [])])
    return ev, tr, d.get("meta", {})


def save_ref(path: str, eval_hashes: List[int], train_hashes: List[int], meta: dict):
    """Write phash_ref.json with hashes as hex strings (dedup within each pool)."""
    def _hex_unique(hs):
        return sorted({f"{h:016x}" for h in hs})
    out = {
        "eval_hashes": _hex_unique(eval_hashes),
        "train_hashes": _hex_unique(train_hashes),
        "meta": meta,
    }
    with open(path, "w") as f:
        json.dump(out, f)
