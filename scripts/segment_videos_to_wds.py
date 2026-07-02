#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Segment raw long surgical videos into the project's WebDataset clip format.

We downloaded two video-native datasets as raw full-procedure videos:
  - GraSP    (robotic prostatectomy)  : videos.tar.gz, videos/CASE###/V###_###.MP4
  - Cholec80 (laparoscopic chole, 25fps): cholec80.zip, videos/videoNN.mp4

Every existing source under ``surg_vid_webdataset_resharded/`` is ~1-minute clips
kept at NATIVE fps/resolution (the dataloader downsamples fps at decode and
RandomResizedCrops resolution — see src/datasets/webdataset.py:224 and
app/vjepa_2_1/transforms.py). So this script cuts each raw video into ~60s clips
by **stream copy (no re-encode)** via PyAV: fast, lossless, native fps/res kept.

Output is NOT sharded here. Per source video we write one raw tar
``<staging>/<dataset>__<source>.tar`` containing, per clip, three members that
match the existing contract exactly:
    <key>.mp4   the clip
    <key>.json  {"source_dataset","source_path","label":0}
    <key>.cls   "0"
Keys are ``<dataset>__<source>_clip_<NNN>`` so scripts/reshard_webdataset.py's
SOURCE_VIDEO_RE parses the source video and shuffles clips across sources into
shards (+ writes metadata.json). Run reshard on <staging> after this.

Parallelism: cholec80's zip is seekable -> fan workers over --video-start/--end
disjoint slices of the 80-video list. GraSP's 127GB tar.gz is a serial gzip
stream -> ONE serial worker (cannot fan within the archive).

Usage (frameworks python; PyAV required):
  # cholec80 smoke, first video:
  python3 scripts/segment_videos_to_wds.py --dataset cholec80 \
      --archive /flare/.../incoming_robotic/cholec80/cholec80.zip \
      --output-staging /flare/.../surg_vid_webdataset_resharded/cholec80_staging \
      --video-start 0 --video-end 1
  # grasp smoke, first 2 source videos:
  python3 scripts/segment_videos_to_wds.py --dataset grasp \
      --archive /flare/.../incoming_robotic/grasp/GraSP/GraSP_videos/videos.tar.gz \
      --output-staging /flare/.../surg_vid_webdataset_resharded/grasp_staging \
      --max-videos 2
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import tarfile
import time
import zipfile
from fractions import Fraction

import av

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
_SANITIZE = re.compile(r"[^A-Za-z0-9_]+")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", required=True, choices=["cholec80", "grasp"])
    p.add_argument("--archive", required=True, help="Source zip (cholec80) or tar.gz (grasp).")
    p.add_argument("--output-staging", required=True, help="Dir for per-source *.tar (created).")
    p.add_argument("--tmpdir", default="/tmp/segwork", help="Scratch for extracted source videos.")
    p.add_argument("--segment-seconds", type=float, default=60.0)
    p.add_argument("--min-clip-seconds", type=float, default=8.0,
                   help="Drop a trailing segment shorter than this.")
    p.add_argument("--video-start", type=int, default=None, help="cholec80: first video index (incl).")
    p.add_argument("--video-end", type=int, default=None, help="cholec80: last video index (excl).")
    p.add_argument("--max-videos", type=int, default=None, help="Process at most this many source videos.")
    p.add_argument("--force", action="store_true", help="Overwrite existing per-source output tars.")
    return p.parse_args()


def _sanitize(stem: str) -> str:
    return _SANITIZE.sub("_", stem).strip("_")


def _sidecars(key: str, dataset: str, source_path: str):
    """Return (json_bytes, cls_bytes) matching the existing sample contract."""
    meta = {"source_dataset": dataset, "source_path": source_path, "label": 0}
    return json.dumps(meta).encode("utf-8"), b"0"


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes):
    ti = tarfile.TarInfo(name=name)
    ti.size = len(data)
    ti.mtime = 0
    tar.addfile(ti, io.BytesIO(data))


def segment_one(src_path, source_name, dataset, out_tar, seg_s, min_s, source_ref=None):
    """Stream-copy segment one video file into <=seg_s clips, writing triples to out_tar.

    Returns (n_clips, n_dropped_short). Cuts on keyframes at/after each segment
    boundary; per-segment first packet is rebased to PTS/DTS 0 so decord's
    get_avg_fps()/seek behave (equivalent of ffmpeg -reset_timestamps 1).
    """
    inp = av.open(src_path)
    try:
        vs = inp.streams.video[0]
    except (IndexError, KeyError):
        inp.close()
        return 0, 0
    tb = vs.time_base or Fraction(1, 1000)
    seg_ticks = int(seg_s / float(tb))
    min_ticks = int(min_s / float(tb))

    n_clips = 0
    n_dropped = 0

    cur_buf = None
    cur_out = None
    cur_ostream = None
    seg_start_dts = None      # dts of first packet in current segment (for rebasing)
    seg_boundary = None       # absolute dts at which the current segment should end

    def _open_segment():
        nonlocal cur_buf, cur_out, cur_ostream
        cur_buf = io.BytesIO()
        cur_out = av.open(cur_buf, mode="w", format="mp4")
        cur_ostream = cur_out.add_stream_from_template(vs)  # stream-copy, no encoder

    def _close_segment():
        """Finish current segment; return its bytes (or None)."""
        nonlocal cur_buf, cur_out, cur_ostream
        if cur_out is None:
            return None
        cur_out.close()
        data = cur_buf.getvalue()
        cur_buf = cur_out = cur_ostream = None
        return data

    def _emit(data, span_ticks):
        nonlocal n_clips, n_dropped
        if data is None:
            return
        if span_ticks < min_ticks:
            n_dropped += 1
            return
        key = f"{dataset}__{source_name}_clip_{n_clips:04d}"
        _add_bytes(out_tar, f"{key}.mp4", data)
        jb, cb = _sidecars(key, dataset, source_ref or src_path)
        _add_bytes(out_tar, f"{key}.json", jb)
        _add_bytes(out_tar, f"{key}.cls", cb)
        n_clips += 1

    # NOTE: demux yields packets in DTS (decode) order, which is monotonic even
    # when PTS is reordered by B-frames. Drive segmentation off DTS; rebase both
    # pts and dts by the segment's starting dts so each clip starts near 0.
    last_dts = None
    for pkt in inp.demux(vs):
        if pkt.dts is None or pkt.pts is None:
            continue
        # New segment on a keyframe at/after the boundary (or the very first packet).
        if cur_out is None or (pkt.is_keyframe and seg_boundary is not None and pkt.dts >= seg_boundary):
            if cur_out is not None:
                span = (last_dts - seg_start_dts) if (last_dts is not None and seg_start_dts is not None) else 0
                _emit(_close_segment(), span)
            _open_segment()
            seg_start_dts = pkt.dts
            seg_boundary = pkt.dts + seg_ticks
        # Rebase timestamps so each segment starts near 0.
        pkt.stream = cur_ostream
        pkt.pts = pkt.pts - seg_start_dts
        pkt.dts = pkt.dts - seg_start_dts
        try:
            cur_out.mux(pkt)
        except Exception:
            # An edge/non-monotonic packet: skip it rather than abort the video.
            continue
        last_dts = pkt.dts + seg_start_dts

    # flush final segment
    if cur_out is not None:
        span = (last_dts - seg_start_dts) if (last_dts is not None and seg_start_dts is not None) else 0
        _emit(_close_segment(), span)
    inp.close()
    return n_clips, n_dropped


def _process_source(video_bytes, source_name, args, out_dir, source_ref):
    """Write one source video's bytes to tmp, segment into its own per-source tar.

    source_ref is the STABLE provenance string recorded in each clip's .json
    (e.g. ``<archive>::videos/video01.mp4``), not the ephemeral tmp path.
    """
    dataset = args.dataset
    out_tar_path = os.path.join(out_dir, f"{dataset}__{source_name}.tar")
    if os.path.exists(out_tar_path) and not args.force:
        print(f"  [skip] {os.path.basename(out_tar_path)} exists (--force to redo)", flush=True)
        return None
    os.makedirs(args.tmpdir, exist_ok=True)
    tmp_vid = os.path.join(args.tmpdir, f"{source_name}.mp4")
    with open(tmp_vid, "wb") as f:
        f.write(video_bytes)
    tmp_tar = out_tar_path + ".tmp"
    t0 = time.time()
    with tarfile.open(tmp_tar, "w") as out:
        n_clips, n_drop = segment_one(tmp_vid, source_name, dataset, out,
                                      args.segment_seconds, args.min_clip_seconds,
                                      source_ref=source_ref)
    if n_clips == 0:
        if os.path.exists(tmp_tar):
            os.remove(tmp_tar)
        print(f"  [warn] {source_name}: 0 clips produced (skipped)", flush=True)
        if os.path.exists(tmp_vid):
            os.remove(tmp_vid)
        return {"source": source_name, "clips": 0, "dropped": n_drop}
    os.replace(tmp_tar, out_tar_path)
    if os.path.exists(tmp_vid):
        os.remove(tmp_vid)
    dt = time.time() - t0
    print(f"  [{source_name}] clips={n_clips} dropped_short={n_drop} ({dt:.1f}s)", flush=True)
    return {"source": source_name, "clips": n_clips, "dropped": n_drop}


def run_cholec80(args, out_dir):
    zf = zipfile.ZipFile(args.archive)
    vids = sorted(n for n in zf.namelist()
                  if n.lower().endswith(VIDEO_EXTS) and "/" in n and not n.endswith("/"))
    lo = args.video_start if args.video_start is not None else 0
    hi = args.video_end if args.video_end is not None else len(vids)
    hi = min(hi, len(vids))
    sel = vids[lo:hi]
    if args.max_videos is not None:
        sel = sel[:args.max_videos]
    print(f"[cholec80] {len(vids)} videos total; processing [{lo}:{hi}) -> {len(sel)} videos", flush=True)
    results = []
    for zpath in sel:
        source_name = _sanitize(os.path.splitext(os.path.basename(zpath))[0])  # video01
        data = zf.read(zpath)
        source_ref = f"{os.path.abspath(args.archive)}::{zpath}"
        r = _process_source(data, source_name, args, out_dir, source_ref)
        if r:
            results.append(r)
    zf.close()
    return results, (lo, hi)


def run_grasp(args, out_dir):
    # 127GB gzip: serial stream, cannot seek. One pass; extract each *.MP4 to tmp then segment.
    print(f"[grasp] streaming {args.archive} (serial gzip)", flush=True)
    results = []
    count = 0
    with tarfile.open(args.archive, "r|gz") as tf:  # streaming mode
        for m in tf:
            if not m.isfile() or not m.name.lower().endswith(VIDEO_EXTS):
                continue
            # videos/CASE001/V001_001.MP4 -> CASE001_V001_001
            parts = m.name.replace("\\", "/").split("/")
            stem = os.path.splitext(parts[-1])[0]
            case = parts[-2] if len(parts) >= 2 else ""
            source_name = _sanitize(f"{case}_{stem}" if case else stem)
            if args.max_videos is not None and count >= args.max_videos:
                break
            data = tf.extractfile(m).read()
            source_ref = f"{os.path.abspath(args.archive)}::{m.name}"
            r = _process_source(data, source_name, args, out_dir, source_ref)
            if r:
                results.append(r)
            count += 1
    return results, (0, count)


def main():
    args = parse_args()
    out_dir = args.output_staging
    os.makedirs(out_dir, exist_ok=True)
    print(f"PyAV {av.__version__}; dataset={args.dataset}; seg={args.segment_seconds}s "
          f"min={args.min_clip_seconds}s; out={out_dir}", flush=True)

    if args.dataset == "cholec80":
        results, rng = run_cholec80(args, out_dir)
    else:
        results, rng = run_grasp(args, out_dir)

    total_clips = sum(r["clips"] for r in results)
    total_drop = sum(r["dropped"] for r in results)
    partial = os.path.join(out_dir, f"_partial_{rng[0]}_{rng[1]}.json")
    with open(partial, "w") as f:
        json.dump({"dataset": args.dataset, "range": list(rng),
                   "sources": len(results), "clips": total_clips,
                   "dropped_short": total_drop, "per_source": results}, f, indent=2)
    print(f"DONE range {rng}: {len(results)} sources, {total_clips} clips, "
          f"{total_drop} short-dropped. wrote {partial}", flush=True)


if __name__ == "__main__":
    main()
