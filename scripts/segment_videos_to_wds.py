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

A third input mode handles LEMON (Surg-3M): a flat DIRECTORY of loose YouTube
.mp4 files (no archive). Because LEMON is YouTube-scraped like our corpus but
lost its IDs, each source video is gated through a perceptual-hash reference
(scripts/build_phash_ref.py -> phash_ref.json) BEFORE segmenting: a video whose
sampled frames match the eval or surgenet_robotic pools above a threshold is
skipped (eval leakage / train duplication). LEMON sidecars also carry the
per-video ``robotic`` + ``procedure`` labels from labels.json.

Parallelism: cholec80's zip is seekable -> fan workers over --video-start/--end
disjoint slices of the 80-video list. GraSP's 127GB tar.gz is a serial gzip
stream -> ONE serial worker (cannot fan within the archive). LEMON's directory
is seekable -> fan workers over --video-start/--end slices of the sorted file list.

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phash_util as ph  # noqa: E402

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
_SANITIZE = re.compile(r"[^A-Za-z0-9_]+")


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", required=True,
                   choices=["cholec80", "grasp", "lemon", "heichole", "multibypass140",
                            "gynsurg", "lapgyn6_events"])
    p.add_argument("--archive", default=None, help="Source zip (cholec80) or tar.gz (grasp).")
    p.add_argument("--archives", default=None, nargs="+",
                   help="multibypass140/gynsurg/lapgyn6_events: one or more source zips.")
    p.add_argument("--input-dir", default=None,
                   help="lemon/heichole: directory of loose *.mp4 files (used instead of --archive).")
    p.add_argument("--output-staging", required=True, help="Dir for per-source *.tar (created).")
    p.add_argument("--tmpdir", default="/tmp/segwork", help="Scratch for extracted source videos.")
    p.add_argument("--segment-seconds", type=float, default=60.0)
    p.add_argument("--min-clip-seconds", type=float, default=8.0,
                   help="Drop a trailing segment shorter than this.")
    p.add_argument("--video-start", type=int, default=None,
                   help="cholec80/lemon: first video index (incl) for range fan-out.")
    p.add_argument("--video-end", type=int, default=None,
                   help="cholec80/lemon: last video index (excl) for range fan-out.")
    p.add_argument("--max-videos", type=int, default=None, help="Process at most this many source videos.")
    p.add_argument("--force", action="store_true", help="Overwrite existing per-source output tars.")
    # --- lemon-only: perceptual-hash dedup gate + label enrichment ---
    p.add_argument("--phash-ref", default=None,
                   help="lemon: phash_ref.json (eval+surgenet pools). If set, gate each source video.")
    p.add_argument("--labels-json", default=None,
                   help="lemon: labels.json for per-video robotic/procedure sidecar enrichment.")
    p.add_argument("--phash-threshold", type=float, default=0.10,
                   help="lemon: drop a video if >this fraction of sampled frames match the ref.")
    p.add_argument("--phash-sample-frames", type=int, default=48,
                   help="lemon: frames sampled per source video for the pHash gate.")
    return p.parse_args()


def _sanitize(stem: str, strip_edges: bool = True) -> str:
    s = _SANITIZE.sub("_", stem)
    return s.strip("_") if strip_edges else s


def _sidecars(key: str, dataset: str, source_path: str, extra: dict = None):
    """Return (json_bytes, cls_bytes) matching the existing sample contract.

    ``extra`` (LEMON: {"robotic":..., "procedure":[...]}) is merged in to
    preserve provenance the loader ignores but downstream analysis can use.
    """
    meta = {"source_dataset": dataset, "source_path": source_path, "label": 0}
    if extra:
        meta.update(extra)
    return json.dumps(meta).encode("utf-8"), b"0"


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes):
    ti = tarfile.TarInfo(name=name)
    ti.size = len(data)
    ti.mtime = 0
    tar.addfile(ti, io.BytesIO(data))


def segment_one(src_path, source_name, dataset, out_tar, seg_s, min_s, source_ref=None,
                extra=None):
    """Stream-copy segment one video file into <=seg_s clips, writing triples to out_tar.

    Returns (n_clips, n_dropped_short). Cuts on keyframes at/after each segment
    boundary; per-segment first packet is rebased to PTS/DTS 0 so decord's
    get_avg_fps()/seek behave (equivalent of ffmpeg -reset_timestamps 1).
    ``extra`` is merged into every clip's .json sidecar (LEMON labels).
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
        jb, cb = _sidecars(key, dataset, source_ref or src_path, extra=extra)
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


def _clip_duration_s(src_path):
    """Return the video duration in seconds (best-effort), or None."""
    try:
        c = av.open(src_path)
    except Exception:
        return None
    try:
        v = c.streams.video[0]
        if v.duration is not None and v.time_base is not None:
            return float(v.duration * v.time_base)
        if c.duration is not None:
            return float(c.duration) / 1e6  # AV_TIME_BASE
    except (IndexError, KeyError):
        return None
    finally:
        c.close()
    return None


def passthrough_one(src_path, source_name, dataset, out_tar, min_s, source_ref=None,
                    extra=None):
    """Emit a PRE-CUT clip as a single sample (no re-segmentation).

    GynSurg / LapGyn6-Events ship as already-short action/event clips. We keep
    each as one clip, dropping those shorter than min_s (V-JEPA needs >=4s for a
    16f@4fps window). Stream-copy remux so PTS/DTS are rebased to ~0 (decord's
    get_avg_fps/seek expect this), matching what segment_one produces. Returns
    (n_clips in {0,1}, n_dropped_short in {0,1}).
    """
    dur = _clip_duration_s(src_path)
    if dur is not None and dur < min_s:
        return 0, 1
    inp = av.open(src_path)
    try:
        vs = inp.streams.video[0]
    except (IndexError, KeyError):
        inp.close()
        return 0, 0
    buf = io.BytesIO()
    out = av.open(buf, mode="w", format="mp4")
    ostream = out.add_stream_from_template(vs)
    start_dts = None
    npkts = 0
    for pkt in inp.demux(vs):
        if pkt.dts is None or pkt.pts is None:
            continue
        if start_dts is None:
            start_dts = pkt.dts
        pkt.stream = ostream
        pkt.pts = pkt.pts - start_dts
        pkt.dts = pkt.dts - start_dts
        try:
            out.mux(pkt)
        except Exception:
            continue
        npkts += 1
    out.close()
    inp.close()
    if npkts == 0:
        return 0, 0
    data = buf.getvalue()
    key = f"{dataset}__{source_name}_clip_0000"
    _add_bytes(out_tar, f"{key}.mp4", data)
    jb, cb = _sidecars(key, dataset, source_ref or src_path, extra=extra)
    _add_bytes(out_tar, f"{key}.json", jb)
    _add_bytes(out_tar, f"{key}.cls", cb)
    return 1, 0


def _process_source(video_bytes, source_name, args, out_dir, source_ref,
                    extra=None, gate=None, mode="segment"):
    """Write one source video's bytes to tmp, segment into its own per-source tar.

    source_ref is the STABLE provenance string recorded in each clip's .json
    (e.g. ``<archive>::videos/video01.mp4``), not the ephemeral tmp path.
    ``extra`` is merged into every sidecar (LEMON labels). ``gate`` (LEMON) is a
    callable(tmp_vid_path) -> (drop:bool, reason:str, dup_frac_eval, dup_frac_train)
    run BEFORE segmentation; if it returns drop=True the video is skipped and a
    status dict with status="dropped" is returned. A decode/integrity failure is
    caught and returned as status="corrupt" rather than aborting the worker.
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
    t0 = time.time()

    # pHash dedup gate (LEMON): also serves as the integrity probe — a corrupt
    # video raises here and is logged as such, not fatal.
    if gate is not None:
        try:
            drop, reason, dfe, dft = gate(tmp_vid)
        except Exception as e:
            if os.path.exists(tmp_vid):
                os.remove(tmp_vid)
            print(f"  [corrupt] {source_name}: {repr(e)[:70]}", flush=True)
            return {"source": source_name, "clips": 0, "dropped": 0, "status": "corrupt"}
        if drop:
            if os.path.exists(tmp_vid):
                os.remove(tmp_vid)
            print(f"  [dup-drop] {source_name}: {reason} "
                  f"(eval={dfe:.2f} train={dft:.2f})", flush=True)
            return {"source": source_name, "clips": 0, "dropped": 0,
                    "status": "dropped", "reason": reason,
                    "dup_eval": dfe, "dup_train": dft}

    tmp_tar = out_tar_path + ".tmp"
    try:
        with tarfile.open(tmp_tar, "w") as out:
            if mode == "passthrough":
                n_clips, n_drop = passthrough_one(tmp_vid, source_name, dataset, out,
                                                  args.min_clip_seconds,
                                                  source_ref=source_ref, extra=extra)
            else:
                n_clips, n_drop = segment_one(tmp_vid, source_name, dataset, out,
                                              args.segment_seconds, args.min_clip_seconds,
                                              source_ref=source_ref, extra=extra)
    except Exception as e:
        if os.path.exists(tmp_tar):
            os.remove(tmp_tar)
        if os.path.exists(tmp_vid):
            os.remove(tmp_vid)
        print(f"  [corrupt] {source_name}: segment failed {repr(e)[:60]}", flush=True)
        return {"source": source_name, "clips": 0, "dropped": 0, "status": "corrupt"}
    if n_clips == 0:
        if os.path.exists(tmp_tar):
            os.remove(tmp_tar)
        print(f"  [warn] {source_name}: 0 clips produced (skipped)", flush=True)
        if os.path.exists(tmp_vid):
            os.remove(tmp_vid)
        return {"source": source_name, "clips": 0, "dropped": n_drop, "status": "empty"}
    os.replace(tmp_tar, out_tar_path)
    if os.path.exists(tmp_vid):
        os.remove(tmp_vid)
    dt = time.time() - t0
    print(f"  [{source_name}] clips={n_clips} dropped_short={n_drop} ({dt:.1f}s)", flush=True)
    return {"source": source_name, "clips": n_clips, "dropped": n_drop, "status": "ok"}


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


def _make_phash_gate(args):
    """Build the LEMON pHash gate closure, or None if --phash-ref not given.

    Returns callable(tmp_vid) -> (drop, reason, dup_eval, dup_train). Also acts
    as the integrity probe: sampling frames from a corrupt file raises, which
    _process_source catches and logs as corrupt.
    """
    if not args.phash_ref:
        return None
    ev_ref, tr_ref, meta = ph.load_ref(args.phash_ref)
    thr = args.phash_threshold
    n = args.phash_sample_frames
    print(f"[phash] ref: {ev_ref.size} eval + {tr_ref.size} train hashes; "
          f"threshold={thr}, sample={n} frames/video", flush=True)

    def gate(tmp_vid):
        frames = ph.sample_gray_frames(tmp_vid, n=n)
        if not frames:
            raise RuntimeError("no decodable frames")
        cands = [ph.phash_gray(f) for f in frames]
        dfe = ph.dup_fraction(cands, ev_ref)
        dft = ph.dup_fraction(cands, tr_ref)
        if dfe > thr:
            return True, "eval-leak", dfe, dft
        if dft > thr:
            return True, "train-dup", dfe, dft
        return False, "", dfe, dft

    return gate


def run_lemon(args, out_dir):
    """Segment a flat directory of loose LEMON *.mp4 files (range fan-out)."""
    input_dir = args.input_dir or args.archive
    if not input_dir or not os.path.isdir(input_dir):
        raise SystemExit(f"lemon: --input-dir must be a directory (got {input_dir!r})")
    vids = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(VIDEO_EXTS))
    lo = args.video_start if args.video_start is not None else 0
    hi = args.video_end if args.video_end is not None else len(vids)
    hi = min(hi, len(vids))
    sel = vids[lo:hi]
    if args.max_videos is not None:
        sel = sel[:args.max_videos]
    print(f"[lemon] {len(vids)} videos total; processing [{lo}:{hi}) -> {len(sel)}", flush=True)

    labels = {}
    if args.labels_json and os.path.exists(args.labels_json):
        with open(args.labels_json) as f:
            for e in json.load(f):
                labels[e["youtubeId"]] = e
        print(f"[lemon] loaded {len(labels)} label entries", flush=True)

    gate = _make_phash_gate(args)
    results = []
    for fn in sel:
        yid = os.path.splitext(fn)[0]          # youtubeId (11-char, reshard-safe)
        # Keep leading/trailing so a leading '-'/'_' (60/73 of 4194 IDs) isn't
        # stripped -> source_name stays faithful & unique (verified 0 collisions).
        source_name = _sanitize(yid, strip_edges=False)
        src_path = os.path.join(input_dir, fn)
        source_ref = os.path.abspath(src_path)
        extra = None
        lab = labels.get(yid)
        if lab is not None:
            extra = {"robotic": bool(lab.get("robotic")),
                     "procedure": lab.get("procedureName", [])}
        with open(src_path, "rb") as f:
            data = f.read()
        r = _process_source(data, source_name, args, out_dir, source_ref,
                            extra=extra, gate=gate)
        if r:
            results.append(r)
    return results, (lo, hi)


def run_heichole(args, out_dir):
    """HeiChole: a directory of loose full-procedure HD mp4s -> 60s segments.

    Same shape as run_lemon but no pHash gate (distinct source, not YouTube) and
    segment (not passthrough) mode since these are full ~30-40min procedures.
    """
    input_dir = args.input_dir or args.archive
    if not input_dir or not os.path.isdir(input_dir):
        raise SystemExit(f"heichole: --input-dir must be a directory (got {input_dir!r})")
    vids = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(VIDEO_EXTS))
    lo = args.video_start if args.video_start is not None else 0
    hi = args.video_end if args.video_end is not None else len(vids)
    hi = min(hi, len(vids))
    sel = vids[lo:hi]
    if args.max_videos is not None:
        sel = sel[:args.max_videos]
    print(f"[heichole] {len(vids)} videos total; processing [{lo}:{hi}) -> {len(sel)}", flush=True)
    results = []
    for fn in sel:
        source_name = _sanitize(os.path.splitext(fn)[0])  # HeiChole1
        src_path = os.path.join(input_dir, fn)
        with open(src_path, "rb") as f:
            data = f.read()
        r = _process_source(data, source_name, args, out_dir, os.path.abspath(src_path))
        if r:
            results.append(r)
    return results, (lo, hi)


def run_multibypass140(args, out_dir):
    """MultiBypass140: nested zips, <center>/videos/<ID>.mp4 -> 60s segments.

    Fan-out is per-ARCHIVE (pass one --archives entry per worker) since a single
    zip is seekable but the set is large. Center prefix kept in source_name so
    Bern/Stras IDs never collide.
    """
    archives = args.archives or ([args.archive] if args.archive else [])
    if not archives:
        raise SystemExit("multibypass140: pass --archives <zip> [<zip> ...]")
    results = []
    n_done = 0
    for arc in archives:
        zf = zipfile.ZipFile(arc)
        vids = sorted(n for n in zf.namelist()
                      if n.lower().endswith(VIDEO_EXTS) and "/videos/" in n.lower()
                      and not n.endswith("/"))
        print(f"[multibypass140] {os.path.basename(arc)}: {len(vids)} videos", flush=True)
        for zpath in vids:
            if args.max_videos is not None and n_done >= args.max_videos:
                break
            parts = zpath.replace("\\", "/").split("/")
            stem = os.path.splitext(parts[-1])[0]           # BBP01
            center = parts[-3] if len(parts) >= 3 else ""    # BernBypass70
            source_name = _sanitize(f"{center}_{stem}" if center else stem)
            data = zf.read(zpath)
            source_ref = f"{os.path.abspath(arc)}::{zpath}"
            r = _process_source(data, source_name, args, out_dir, source_ref)
            if r:
                results.append(r)
            n_done += 1
        zf.close()
        if args.max_videos is not None and n_done >= args.max_videos:
            break
    return results, (0, len(results))


def run_preclip_zip(args, out_dir):
    """GynSurg / LapGyn6-Events: zip(s) of PRE-CUT action/event clips.

    Each clip becomes ONE sample (passthrough mode) — no re-segmentation — with a
    min-length filter (drop <min_clip_seconds; default 8s but pass 4.0 for these).
    Every clip is its own 'source' (unique name from its zip path + running index)
    so reshard shuffles them across shards. Range fan-out is over the flat sorted
    clip list so multiple workers can split one big zip.
    """
    archives = args.archives or ([args.archive] if args.archive else [])
    if not archives:
        raise SystemExit(f"{args.dataset}: pass --archives <zip> [<zip> ...]")
    # Build the flat (archive, member) list across all archives, sorted for stable ranges.
    entries = []
    for arc in archives:
        zf = zipfile.ZipFile(arc)
        for n in sorted(zf.namelist()):
            if n.lower().endswith(VIDEO_EXTS) and not n.endswith("/"):
                entries.append((arc, n))
        zf.close()
    lo = args.video_start if args.video_start is not None else 0
    hi = args.video_end if args.video_end is not None else len(entries)
    hi = min(hi, len(entries))
    sel = entries[lo:hi]
    if args.max_videos is not None:
        sel = sel[:args.max_videos]
    print(f"[{args.dataset}] {len(entries)} clips total; processing [{lo}:{hi}) -> {len(sel)}",
          flush=True)
    # Open each archive once (cache handles).
    zcache = {}
    results = []
    for idx, (arc, member) in enumerate(sel, start=lo):
        zf = zcache.get(arc)
        if zf is None:
            zf = zcache[arc] = zipfile.ZipFile(arc)
        stem = os.path.splitext(os.path.basename(member))[0]
        # URL-encoded timestamps (%3A) + running index -> unique, reshard-safe.
        source_name = f"{_sanitize(stem)}_{idx:06d}"
        data = zf.read(member)
        source_ref = f"{os.path.abspath(arc)}::{member}"
        r = _process_source(data, source_name, args, out_dir, source_ref,
                            mode="passthrough")
        if r:
            results.append(r)
    for zf in zcache.values():
        zf.close()
    return results, (lo, hi)


def main():
    args = parse_args()
    out_dir = args.output_staging
    os.makedirs(out_dir, exist_ok=True)
    print(f"PyAV {av.__version__}; dataset={args.dataset}; seg={args.segment_seconds}s "
          f"min={args.min_clip_seconds}s; out={out_dir}", flush=True)

    if args.dataset == "cholec80":
        results, rng = run_cholec80(args, out_dir)
    elif args.dataset == "grasp":
        results, rng = run_grasp(args, out_dir)
    elif args.dataset == "heichole":
        results, rng = run_heichole(args, out_dir)
    elif args.dataset == "multibypass140":
        results, rng = run_multibypass140(args, out_dir)
    elif args.dataset in ("gynsurg", "lapgyn6_events"):
        results, rng = run_preclip_zip(args, out_dir)
    else:
        results, rng = run_lemon(args, out_dir)

    total_clips = sum(r["clips"] for r in results)
    total_drop = sum(r["dropped"] for r in results)
    n_corrupt = sum(1 for r in results if r.get("status") == "corrupt")
    n_leak = sum(1 for r in results if r.get("reason") == "eval-leak")
    n_traindup = sum(1 for r in results if r.get("reason") == "train-dup")
    partial = os.path.join(out_dir, f"_partial_{rng[0]}_{rng[1]}.json")
    with open(partial, "w") as f:
        json.dump({"dataset": args.dataset, "range": list(rng),
                   "sources": len(results), "clips": total_clips,
                   "dropped_short": total_drop,
                   "corrupt": n_corrupt, "eval_leak": n_leak, "train_dup": n_traindup,
                   "per_source": results}, f, indent=2)
    print(f"DONE range {rng}: {len(results)} sources, {total_clips} clips, "
          f"{total_drop} short-dropped, {n_corrupt} corrupt, "
          f"{n_leak} eval-leak, {n_traindup} train-dup. wrote {partial}", flush=True)


if __name__ == "__main__":
    main()
