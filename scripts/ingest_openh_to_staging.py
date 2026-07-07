#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Stream-ingest nvidia/PhysicalAI-Robotics-Open-H-Embodiment into WebDataset staging.

Open-H-Embodiment is a 50-institution medical robot-learning dataset (LeRobot v2.1
format) on the HF Hub. Its top level is only ``Surgical/`` + ``Endoscopy/`` — 48,545
MP4s, ~37K of which are RGB endoscope/surgical camera views (the rest are wrist cams,
depth, fluoroscopy X-ray, stereo-right duplicates, and static goal images — dropped
by KEEP_VIEWS below).

Why STREAM (download -> re-encode -> pack, per clip) instead of bulk download then
re-encode: the bulk is ~3.2 TB, dominated by cmr_surgical at 1080p/**60fps**. The
model only sees a 384px random crop at 4fps, so 1080p/60fps is ~93% wasted decode
work AND re-introduces the heichole decode-contention stall
(docs: reencode_source_reshard.py). We re-encode each clip to 512 short-side / 8fps /
CRF23 / g16 on the way in (measured 52 MB -> 8 MB, 15%, aspect + full duration kept)
and NEVER persist the raw bytes. Result: ~250-500 GB final instead of 3.2 TB.

Output: {key.mp4, key.json, key.cls} triples in staging tars, keyed
``openh__<embodiment>_<episode-stem>_clip_0000`` so reshard_webdataset.py parses the
source (SOURCE_VIDEO_RE) and shuffles across episodes. Then reshard the union.

The re-encode helper is lifted from scripts/reencode_source_reshard.py (adding ``-r
FPS``); the tar-add helper mirrors scripts/pack_clips_to_staging.py. ffmpeg is the
imageio-ffmpeg static (no system ffmpeg needed) via _default_ffmpeg().

Idempotent: writes one staging tar per worker range plus a ``_partial_<lo>_<hi>.json``
tally; a range whose tar already exists is skipped unless --force.

Usage (one worker range; the PBS driver fans many in parallel):
  python3 scripts/ingest_openh_to_staging.py \
      --output-staging /flare/.../surg_vid_webdataset_resharded/openh_staging \
      --tmpdir /tmp/openh_0 --file-start 0 --file-end 500
  # smoke: cap clips, sample across embodiments
  python3 scripts/ingest_openh_to_staging.py --output-staging /tmp/openh_smoke \
      --tmpdir /tmp/openh_s --smoke 8
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

REPO = "nvidia/PhysicalAI-Robotics-Open-H-Embodiment"
_SANITIZE = re.compile(r"[^A-Za-z0-9_]+")

# RGB surgical/endoscope PRIMARY camera views to KEEP. Note the second entry contains
# a non-ASCII character (the dataset ships it that way) — match it literally.
KEEP_VIEWS = {
    "observation.images.endoscope",
    "observation.images.endo三",       # 'observation.images.endo三' (cuhk)
    "observation.images.color",
    "observation.images.endoscope.left",   # keep LEFT of stereo pairs only
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--output-staging", required=True, help="Dir for staging *.tar (created).")
    p.add_argument("--tmpdir", default=None, help="Scratch for per-clip raw+encoded (default: system tmp).")
    p.add_argument("--file-start", type=int, default=None, help="First kept-file index (inclusive).")
    p.add_argument("--file-end", type=int, default=None, help="Last kept-file index (exclusive).")
    p.add_argument("--smoke", type=int, default=None,
                   help="Ignore ranges; take the first N kept files spread across embodiments.")
    p.add_argument("--short-side", type=int, default=512)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--crf", type=int, default=23)
    p.add_argument("--gop", type=int, default=16)
    p.add_argument("--ffmpeg", default=None, help="ffmpeg binary (default: imageio-ffmpeg static).")
    p.add_argument("--ffmpeg-threads", type=int, default=4)
    p.add_argument("--clips-per-tar", type=int, default=200,
                   help="Not used for range mode (one tar per range); reserved.")
    p.add_argument("--force", action="store_true", help="Overwrite an existing range tar.")
    p.add_argument("--list-only", action="store_true", help="Print kept-file count and exit.")
    return p.parse_args()


def _default_ffmpeg():
    exe = os.environ.get("IMAGEIO_FFMPEG_EXE")
    if exe and os.path.exists(exe):
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return "ffmpeg"


def list_kept_files():
    """Return the sorted list of kept MP4 repo paths (deterministic across workers)."""
    from huggingface_hub import HfApi
    api = HfApi()
    sibs = [s.rfilename for s in api.dataset_info(REPO).siblings if s.rfilename.endswith(".mp4")]
    kept = [s for s in sibs if s.split("/")[-2] in KEEP_VIEWS]
    return sorted(kept)


def _embodiment(path: str) -> str:
    """Top-2 path levels collapsed to a compact embodiment tag, e.g. Surgical/cmr_surgical -> cmr_surgical."""
    parts = path.split("/")
    return _SANITIZE.sub("_", parts[1]).strip("_") if len(parts) > 1 else "unk"


def _key_for(path: str) -> str:
    emb = _embodiment(path)
    # episode stem + a couple of disambiguating path parts (chunk, view) so keys are unique
    stem = _SANITIZE.sub("_", os.path.splitext(os.path.basename(path))[0]).strip("_")
    chunk = _SANITIZE.sub("_", path.split("/")[-3]).strip("_") if len(path.split("/")) >= 3 else "c"
    view = _SANITIZE.sub("_", path.split("/")[-2]).strip("_")
    src = f"{emb}_{chunk}_{view}_{stem}"
    return f"openh__{src}_clip_0000"


def _hf_url(path: str) -> str:
    from huggingface_hub import hf_hub_url
    return hf_hub_url(REPO, path, repo_type="dataset")


def _download(path: str, dst: str, opener, retries=3) -> bool:
    url = _hf_url(path)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url)
            with opener.open(req, timeout=180) as r, open(dst, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            if os.path.getsize(dst) > 0:
                return True
        except Exception as e:
            if attempt == retries - 1:
                print(f"    download FAIL {path}: {str(e)[:100]}", flush=True)
        try:
            os.remove(dst)
        except OSError:
            pass
    return False


def _reencode(ffmpeg, src, dst, short_side, fps, crf, gop, threads):
    """Re-encode src mp4 -> dst mp4 at short_side / fps / crf / gop. Returns ok bool.

    Lifted from reencode_source_reshard.py._reencode with an added -r FPS. Scale so the
    SHORT side == short_side, keep aspect, force even dims (yuv420p)."""
    vf = f"scale='if(gt(iw,ih),-2,{short_side})':'if(gt(iw,ih),{short_side},-2)'"
    cmd = [ffmpeg, "-y", "-nostdin", "-loglevel", "error", "-i", src,
           "-vf", vf, "-r", str(fps), "-c:v", "libx264", "-crf", str(crf),
           "-g", str(gop), "-keyint_min", str(gop), "-threads", str(threads),
           "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart", dst]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.returncode == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0
    except Exception:
        return False


def _add(tar, name, data):
    ti = tarfile.TarInfo(name=name)
    ti.size = len(data)
    ti.mtime = 0
    tar.addfile(ti, io.BytesIO(data))


def main():
    args = parse_args()
    kept = list_kept_files()
    if args.list_only:
        print(len(kept))
        return

    if args.smoke is not None:
        # spread the smoke sample across embodiments (first per emb, round-robin)
        by_emb = {}
        for p in kept:
            by_emb.setdefault(_embodiment(p), []).append(p)
        sel, i = [], 0
        while len(sel) < args.smoke:
            added = False
            for e in sorted(by_emb):
                if i < len(by_emb[e]):
                    sel.append(by_emb[e][i]); added = True
                    if len(sel) >= args.smoke:
                        break
            if not added:
                break
            i += 1
        files = sel
        tar_name = "openh__smoke.tar"
    else:
        lo = args.file_start if args.file_start is not None else 0
        hi = args.file_end if args.file_end is not None else len(kept)
        hi = min(hi, len(kept))
        files = kept[lo:hi]
        tar_name = f"openh__range_{lo:06d}_{hi:06d}.tar"

    os.makedirs(args.output_staging, exist_ok=True)
    tmpdir = args.tmpdir or tempfile.mkdtemp(prefix="openh_")
    os.makedirs(tmpdir, exist_ok=True)
    out_path = os.path.join(args.output_staging, tar_name)
    if os.path.exists(out_path) and not args.force:
        print(f"[skip] {tar_name} exists (use --force)", flush=True)
        return

    ffmpeg = args.ffmpeg or _default_ffmpeg()
    proxy = os.environ.get("https_proxy") or os.environ.get("http_proxy")
    handlers = [urllib.request.ProxyHandler({"http": proxy, "https": proxy})] if proxy else []
    opener = urllib.request.build_opener(*handlers)

    n_ok = n_dl_fail = n_enc_fail = 0
    bytes_in = bytes_out = 0
    t0 = time.time()
    tmp_path = out_path + ".tmp"
    raw = os.path.join(tmpdir, "raw.mp4")
    enc = os.path.join(tmpdir, "enc.mp4")
    print(f"[openh] {len(files)} clips -> {tar_name}  ss={args.short_side} fps={args.fps} "
          f"crf={args.crf} g={args.gop} ffmpeg={ffmpeg}", flush=True)
    with tarfile.open(tmp_path, "w") as tar:
        for idx, path in enumerate(files):
            for p in (raw, enc):
                try:
                    os.remove(p)
                except OSError:
                    pass
            if not _download(path, raw, opener):
                n_dl_fail += 1
                continue
            bytes_in += os.path.getsize(raw)
            if not _reencode(ffmpeg, raw, enc, args.short_side, args.fps, args.crf, args.gop,
                             args.ffmpeg_threads):
                n_enc_fail += 1
                continue
            with open(enc, "rb") as f:
                enc_bytes = f.read()
            bytes_out += len(enc_bytes)
            key = _key_for(path)
            _add(tar, f"{key}.mp4", enc_bytes)
            _add(tar, f"{key}.json", json.dumps({
                "source_dataset": "openh", "source_path": path,
                "embodiment": _embodiment(path), "label": 0}).encode("utf-8"))
            _add(tar, f"{key}.cls", b"0")
            n_ok += 1
            if n_ok % 200 == 0:
                el = time.time() - t0
                print(f"  ...{n_ok} ok ({el:.0f}s, {n_ok/el:.1f}/s, "
                      f"{bytes_out/1e9:.1f}GB out)", flush=True)
    os.replace(tmp_path, out_path)
    for p in (raw, enc):
        try:
            os.remove(p)
        except OSError:
            pass

    partial = os.path.join(args.output_staging,
                           f"_partial_{tar_name.replace('.tar','')}.json")
    with open(partial, "w") as f:
        json.dump({"tar": tar_name, "clips_ok": n_ok, "dl_fail": n_dl_fail,
                   "enc_fail": n_enc_fail, "bytes_in": bytes_in, "bytes_out": bytes_out,
                   "elapsed_s": time.time() - t0,
                   "short_side": args.short_side, "fps": args.fps,
                   "crf": args.crf, "gop": args.gop}, f, indent=2)
    ratio = (100 * bytes_out / bytes_in) if bytes_in else 0.0
    print(f"DONE {tar_name}: ok={n_ok} dl_fail={n_dl_fail} enc_fail={n_enc_fail} "
          f"size={bytes_in/1e9:.1f}->{bytes_out/1e9:.1f}GB ({ratio:.0f}%) "
          f"{time.time()-t0:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
