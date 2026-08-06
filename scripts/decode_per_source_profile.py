#!/usr/bin/env python3
"""Per-source decode profiler — isolates WHY the quartet is slow.

For each source independently: stream N raw samples straight from the .tar shards,
decode each with the SAME decord path the trainer uses, and record:
  - raw decode ms per video (VideoReader + frame gather)
  - clip std (the min_clip_std=1.0 filter drops <1.0 -> forces a re-draw)
  - drop rate (fraction that would be rejected -> each drop = one wasted decode
    the resampled stream has to retry, so effective cost = decode / (1 - drop_rate))
  - decoded frame count / resolution (raw decode cost driver)

No workers, no mixing, no model. Pin-points which of {intrinsic decode cost,
drop-retry amplification} makes a source expensive.
"""
import argparse
import io
import time

import numpy as np
import webdataset as wds
from decord import VideoReader, cpu

DATA_ROOT = "/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded"
VIDEO_KEYS = ("video.mp4", "video.avi", "video.mov", "video.webm", "video.mkv", "mp4", "avi")
FPC = 16
FSTP = 4  # frame_step (fps=4 path)
MIN_CLIP_STD = 1.0


def decode_one(video_bytes, fpc=FPC, fstp=FSTP):
    """Mirror WebDataset decode: open, pick indices, gather frames. Returns
    (decode_ms, nframes_total, clip_std, hw)."""
    t = time.time()
    vr = VideoReader(io.BytesIO(video_bytes), num_threads=1, ctx=cpu(0))
    n = len(vr)
    clip_len = fpc * fstp
    vr.seek(0)
    if n > clip_len:
        end = np.random.randint(clip_len, n) if n > clip_len else clip_len
        start = end - clip_len
        idx = np.linspace(start, end, num=fpc)
        idx = np.clip(idx, start, end - 1).astype(np.int64)
    else:
        idx = np.clip(np.linspace(0, max(n - 1, 0), num=fpc), 0, max(n - 1, 0)).astype(np.int64)
    arr = vr.get_batch(idx).asnumpy()
    dt = (time.time() - t) * 1000.0
    hw = (arr.shape[1], arr.shape[2]) if arr.ndim == 4 else None
    return dt, n, float(arr.std()), hw


def get_video_bytes(sample):
    for k in VIDEO_KEYS:
        if k in sample:
            return sample[k]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--n", type=int, default=60)
    # A source and its _g16 twin hold the SAME samples in the SAME shard order,
    # so seeding makes the pair a paired comparison instead of two independent
    # draws. Unseeded, the first g16 run came back with frac<1.0 = 0.015 -> 0.000
    # and min clip-std 0.00 -> 28.72, which reads exactly like "the re-encode
    # changed the corpus content" and is in fact 3 different clips out of 200.
    # Seeding removes that false alarm and tightens the p50 ratio, which is a
    # difference of medians over draws that are otherwise not the same videos.
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    np.random.seed(args.seed)

    import glob
    shards = sorted(glob.glob(f"{DATA_ROOT}/{args.source}/*.tar"))
    if not shards:
        print(f"NO SHARDS for {args.source}")
        return
    # resampled=False so the pair is a PAIRED comparison. WebDataset's `seed=`
    # cannot buy this: create_url_iterator() constructs ResampledShardList(urls)
    # without forwarding a seed, and ResampledShards.__iter__ then mixes
    # time_ns(), getpid() and os.urandom(4) into its own -- so a source and its
    # _g16 twin streamed DIFFERENT videos even at a fixed seed. That is how the
    # first g16 run produced frac<1.0 = 0.015 -> 0.000 and clip-std min
    # 0.00 -> 28.72, which reads like "the re-encode changed corpus content" and
    # was in fact 3 different clips out of 200.
    #
    # In-order iteration means N samples come from the first ceil(N/per_shard)
    # shards rather than the whole source, so this is no longer a random sample
    # of the corpus. Acceptable here and only here: the quantity of interest is
    # a WITHIN-PAIR ratio on identical clips, not a corpus-level absolute.
    ds = wds.WebDataset(shards, shardshuffle=False, resampled=False)

    decode_ms, stds, nframes, hws, drops = [], [], [], [], 0
    seen = 0
    for sample in ds:
        vb = get_video_bytes(sample)
        if vb is None:
            continue
        try:
            dt, n, std, hw = decode_one(vb)
        except Exception as e:
            print(f"  decode fail: {str(e)[:80]}")
            continue
        decode_ms.append(dt)
        stds.append(std)
        nframes.append(n)
        if hw:
            hws.append(hw)
        if std < MIN_CLIP_STD:
            drops += 1
        seen += 1
        if seen >= args.n:
            break

    def p(v, q):
        v = sorted(v)
        return v[min(len(v) - 1, int(q * len(v)))] if v else -1

    drop_rate = drops / max(seen, 1)
    amp = 1.0 / max(1e-9, (1.0 - drop_rate))  # effective decodes per yielded sample
    from collections import Counter
    hw_common = Counter(hws).most_common(3)
    print(f"\n=== {args.source}  (n={seen}) ===")
    print(f"raw decode ms   p50={p(decode_ms,.5):.0f}  p90={p(decode_ms,.9):.0f}  "
          f"max={max(decode_ms):.0f}  mean={np.mean(decode_ms):.0f}")
    print(f"frames/video    p50={p(nframes,.5):.0f}  p90={p(nframes,.9):.0f}  max={max(nframes)}")
    print(f"resolution(HxW) {hw_common}")
    print(f"clip std        median={np.median(stds):.2f}  "
          f"min={min(stds):.2f}  frac<1.0(DROP)={drop_rate:.3f}")
    print(f">> drop-retry amplification = {amp:.2f}x  "
          f"=> effective decode/sample p50 ~= {p(decode_ms,.5)*amp:.0f} ms")


if __name__ == "__main__":
    main()
