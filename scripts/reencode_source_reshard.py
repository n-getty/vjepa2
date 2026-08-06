#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Source-side video RE-ENCODE for a WebDataset source (bitrate/resolution reduction).

Motivation
----------
Some sources (heichole: measured 1765 ms/video decode vs ~450-510 ms for the other
small-clean sources) are encoded at very high bitrate (~83 KB/frame at 1080p) with
sparse keyframes, so decord's seek-heavy random-clip sampling is ~4x slower per clip.
That inflated decode latency variance is the driver of Leonardo's 15-src stall
(docs/2026-07-05_15src_ablation_postmortem.md) and feeds our mode-1 collective desync.

The model only ever sees a 384px random crop, so decoding 1080p high-bitrate video is
wasted work. This script re-encodes each video member ONCE into a smaller-side,
CRF-controlled, dense-keyframe copy. Measured on a real heichole clip:
  1080p high-bitrate -> 512 short-side / CRF23 / g16:  156MB -> 10.8MB (7%),
  scatter-decode 14181 ms -> 377 ms (~38x faster, in line with the other sources).

It mirrors scripts/filter_black_clips_reshard.py exactly: 1:1 shard naming
(heichole-000123.tar -> heichole-000123.tar), ALL non-video members (json/cls)
copied verbatim, per-range partials + a --finalize merge into metadata.json.
Configs only need the dir path swapped (heichole -> heichole_512).

Encoder: the static ffmpeg shipped by the imageio-ffmpeg wheel (no system ffmpeg /
module needed on Aurora). Override with --ffmpeg or $IMAGEIO_FFMPEG_EXE.

Idempotent: an output shard that already exists is skipped unless --force. Sample
COUNT is preserved (re-encode never drops); if a re-encode fails the ORIGINAL bytes
are copied through (never silently lose data) and counted as encode_fail.

Usage
-----
  # local smoke on 2 shards:
  python3 scripts/reencode_source_reshard.py \
      --input  /flare/ModCon/ngetty/data/surg_vid_webdataset_resharded/heichole \
      --output /flare/ModCon/ngetty/data/surg_vid_webdataset_resharded/heichole_512 \
      --short-side 512 --crf 23 --gop 16 --shard-start 0 --shard-end 2

  # finalize after all ranges complete:
  python3 scripts/reencode_source_reshard.py \
      --input .../heichole --output .../heichole_512 --finalize
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import time
from collections import defaultdict

VIDEO_EXTS = ("mp4", "avi", "mov", "webm", "mkv", "flv")


def _default_ffmpeg():
    exe = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if exe and os.path.exists(exe):
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"  # fall back to PATH


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, help="Input source dir with *.tar shards.")
    p.add_argument("--output", required=True, help="Output dir (created).")
    p.add_argument("--short-side", type=int, default=512,
                   help="Target short-side resolution (default 512; model crop is 384). "
                        "Pass 0 to keep the source resolution untouched (GOP-only arm).")
    p.add_argument("--center-crop", type=float, default=None,
                   help="If set (0<FRAC<=1), center-crop each frame to FRAC of w and h BEFORE "
                        "the short-side scale. e.g. 0.70 keeps the central 70%% (removes burned-in "
                        "edge/corner overlays such as openh's console UI). Default None = no crop.")
    p.add_argument("--crf", type=int, default=23, help="x264 CRF (lower=higher quality; default 23).")
    p.add_argument("--gop", type=int, default=16,
                   help="Keyframe interval in frames (default 16); dense GOP kills seek cost.")
    p.add_argument("--ffmpeg", default=None, help="ffmpeg binary (default: imageio-ffmpeg static).")
    p.add_argument("--ffmpeg-threads", type=int, default=4,
                   help="x264 threads per encode (default 4); keep N_workers x this <= node threads.")
    p.add_argument("--shard-start", type=int, default=None, help="First shard index (inclusive).")
    p.add_argument("--shard-end", type=int, default=None, help="Last shard index (exclusive).")
    p.add_argument("--force", action="store_true", help="Overwrite existing output shards.")
    p.add_argument("--finalize", action="store_true",
                   help="Merge per-range partials into metadata.json + reshard_summary.json and exit.")
    return p.parse_args()


def _group_members(tar_path):
    """Yield (key, [(TarInfo, bytes), ...]) grouped by sample key, preserving order."""
    groups = defaultdict(list)
    order = []
    with tarfile.open(tar_path, "r|") as tf:
        for m in tf:
            if not m.isfile():
                continue
            dot = m.name.find(".")
            key = m.name[:dot] if dot > 0 else m.name
            if key not in groups:
                order.append(key)
            groups[key].append((m, tf.extractfile(m).read()))
    for key in order:
        yield key, groups[key]


def _reencode(ffmpeg, video_bytes, ext, short_side, crf, gop, threads=0, center_crop=None):
    """Re-encode one video's bytes -> new mp4 bytes. Returns (new_bytes, ok).

    ffmpeg needs a seekable input for most muxers, so we round-trip through temp
    files. On any failure we return (None, False) and the caller copies the
    original through (never lose data).

    ``threads`` caps x264's internal thread pool (0 = ffmpeg default = all cores).
    When running many worker processes in parallel, set a small value so N_workers
    x threads does not oversubscribe the node (libx264 is multithreaded)."""
    # Optional center-crop (removes burned-in edge/corner overlays) BEFORE the scale,
    # then scale so the SHORT side == short_side, keep aspect, force even dims (yuv420p).
    #
    # short_side=0 means DO NOT SCALE. The two levers here are separable and the
    # measurement says so: GOP alone buys 4.4-8.8x at unchanged pixels
    # (surgvu24_clean 3.38->0.41 s, cholec80 3.04->0.35 s), while sitl_2026 is
    # already GOP-30 and only moves on the resolution arm (3.55->2.35 g16 vs
    # ->0.61 at 512p). Scaling unconditionally would also UPSCALE the sources that
    # are already below the target -- cholec80 is 854x480, so a blanket
    # --short-side 512 would enlarge it, spending encode time and disk to make
    # decode slower. An unscaled encode is the correct GOP-only arm.
    filters = []
    if center_crop and 0 < center_crop < 1:
        # crop=W:H:X:Y centered; use in_w/in_h so it adapts per-clip (openh widths vary).
        filters.append(f"crop=in_w*{center_crop}:in_h*{center_crop}")
    if short_side and short_side > 0:
        filters.append(
            f"scale='if(gt(iw,ih),-2,{short_side})':'if(gt(iw,ih),{short_side},-2)'"
        )
    # yuv420p needs even dimensions; when we do not scale, the source may be odd.
    filters.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")
    vf = ",".join(filters)
    ti = tempfile.NamedTemporaryFile(suffix="." + ext, delete=False)
    to = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    ti_name, to_name = ti.name, to.name
    ti.write(video_bytes); ti.close(); to.close()
    try:
        cmd = [ffmpeg, "-y", "-nostdin", "-loglevel", "error", "-i", ti_name,
               "-vf", vf, "-c:v", "libx264", "-crf", str(crf),
               "-g", str(gop), "-keyint_min", str(gop),
               "-threads", str(threads),
               "-pix_fmt", "yuv420p", "-an",
               "-movflags", "+faststart", to_name]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not os.path.getsize(to_name):
            return None, False
        with open(to_name, "rb") as f:
            return f.read(), True
    except Exception:
        return None, False
    finally:
        for n in (ti_name, to_name):
            try:
                os.remove(n)
            except OSError:
                pass


def process_range(args, shards, lo, hi):
    ffmpeg = args.ffmpeg or _default_ffmpeg()
    os.makedirs(args.output, exist_ok=True)
    per_shard = {}
    t0 = time.time()
    for si in range(lo, hi):
        in_path = shards[si]
        base = os.path.basename(in_path)
        out_path = os.path.join(args.output, base)
        if os.path.exists(out_path) and not args.force:
            print(f"  [skip] {base} exists (use --force)", flush=True)
            continue
        n_samp = enc_ok = enc_fail = 0
        bytes_in = bytes_out = 0
        tmp_path = out_path + ".tmp"
        with tarfile.open(tmp_path, "w") as out:
            for key, members in _group_members(in_path):
                n_samp += 1
                for tinfo, data in members:
                    ext = tinfo.name.rsplit(".", 1)[-1].lower()
                    if ext in VIDEO_EXTS:
                        bytes_in += len(data)
                        new_bytes, ok = _reencode(
                            ffmpeg, data, ext, args.short_side, args.crf, args.gop,
                            threads=args.ffmpeg_threads, center_crop=args.center_crop)
                        if ok:
                            enc_ok += 1
                            bytes_out += len(new_bytes)
                            # keep the SAME member name/ext (still a valid mp4 stream);
                            # rewrite size so the tar header matches the new payload.
                            ni = tarfile.TarInfo(name=tinfo.name)
                            ni.size = len(new_bytes)
                            ni.mode = tinfo.mode
                            ni.mtime = tinfo.mtime
                            out.addfile(ni, io.BytesIO(new_bytes))
                        else:
                            enc_fail += 1
                            bytes_out += len(data)
                            out.addfile(tinfo, io.BytesIO(data))  # copy original through
                    else:
                        # json/cls/etc — copy verbatim
                        out.addfile(tinfo, io.BytesIO(data))
        os.replace(tmp_path, out_path)
        ratio = (100 * bytes_out / bytes_in) if bytes_in else 0.0
        per_shard[base] = {"samples": n_samp, "enc_ok": enc_ok, "enc_fail": enc_fail,
                           "bytes_in": bytes_in, "bytes_out": bytes_out}
        print(f"  [{si:04d}] {base}: samples={n_samp} enc_ok={enc_ok} "
              f"enc_fail={enc_fail} size={bytes_in/1e6:.0f}->{bytes_out/1e6:.0f}MB "
              f"({ratio:.0f}%)", flush=True)
    partial = os.path.join(args.output, f"_partial_{lo}_{hi}.json")
    with open(partial, "w") as f:
        json.dump({"range": [lo, hi], "elapsed_s": time.time() - t0,
                   "short_side": args.short_side, "crf": args.crf, "gop": args.gop,
                   "per_shard": per_shard}, f, indent=2)
    print(f"wrote {partial}", flush=True)


def finalize(args, shards):
    name = os.path.basename(os.path.normpath(args.output))
    partials = sorted(glob.glob(os.path.join(args.output, "_partial_*.json")))
    merged = {}
    for p in partials:
        merged.update(json.load(open(p)).get("per_shard", {}))
    out_shards = sorted(f for f in os.listdir(args.output) if f.endswith(".tar"))
    total_samp = sum(v["samples"] for v in merged.values())
    total_ok = sum(v["enc_ok"] for v in merged.values())
    total_fail = sum(v["enc_fail"] for v in merged.values())
    bytes_in = sum(v["bytes_in"] for v in merged.values())
    bytes_out = sum(v["bytes_out"] for v in merged.values())
    meta = {"name": name, "shard_count": len(out_shards),
            "sample_count": int(total_samp), "shard_urls": out_shards}
    with open(os.path.join(args.output, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    summary = {"dataset": name, "input_dir": args.input, "output_dir": args.output,
               "output_shards": len(out_shards), "sample_count": total_samp,
               "enc_ok": total_ok, "enc_fail": total_fail,
               "bytes_in_gb": round(bytes_in / 1e9, 2), "bytes_out_gb": round(bytes_out / 1e9, 2),
               "size_pct": round(100 * bytes_out / bytes_in, 1) if bytes_in else 0.0,
               "short_side": args.short_side, "crf": args.crf, "gop": args.gop,
               "shards_processed": len(merged), "shards_expected": len(shards)}
    with open(os.path.join(args.output, "reshard_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if len(merged) != len(shards):
        print(f"WARNING: processed {len(merged)} of {len(shards)} input shards — "
              "some ranges may be missing.", file=sys.stderr)


def main():
    args = parse_args()
    shards = sorted(glob.glob(os.path.join(args.input, "*.tar")))
    if not shards:
        raise SystemExit(f"No .tar shards in {args.input}")
    if args.finalize:
        finalize(args, shards)
        return
    lo = args.shard_start if args.shard_start is not None else 0
    hi = args.shard_end if args.shard_end is not None else len(shards)
    hi = min(hi, len(shards))
    print(f"[{os.path.basename(args.input)}] re-encoding shards [{lo}:{hi}) of {len(shards)}; "
          f"short_side={args.short_side} crf={args.crf} gop={args.gop} "
          f"ffmpeg={args.ffmpeg or _default_ffmpeg()}", flush=True)
    process_range(args, shards, lo, hi)


if __name__ == "__main__":
    main()
