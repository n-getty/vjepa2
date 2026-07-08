#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Re-key the openh staging tars with a COLLISION-FREE key, then reshard.

Bug this fixes: ingest_openh_to_staging.py's original _key_for() built the sample
key from <embodiment>_<chunk>_<view>_<episode-stem> but DROPPED the task/dataset
middle path levels. Two clips that differ only in the task subdir (e.g.
hamlyn/knot_tying/.../episode_000000 vs hamlyn/suturing/.../episode_000000)
collided on the same key. WebDataset/reshard index by key, so 22,452 of 36,693
clips overwrote each other -> the first reshard reported only 14,241 samples.

No re-download needed: every sample's .json carries the unique source_path, so we
re-derive a unique key from it (identical to the corrected _key_for(): full path
minus the leading domain + the literal 'videos' dir, sanitized). This streams the
staging tars, rewrites member names to the new key (payloads untouched), writes
new staging tars, then reshards to openh/.

Verified offline: 36,693 samples -> 36,693 distinct keys, 0 collisions, all parse
under reshard's SOURCE_VIDEO_RE.

Usage:
  python3 scripts/rekey_openh_reshard.py \
      --staging /flare/.../surg_vid_webdataset_resharded/openh_staging \
      --rekeyed /flare/.../surg_vid_webdataset_resharded/openh_rekeyed \
      --final   /flare/.../surg_vid_webdataset_resharded/openh
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import subprocess
import sys
import tarfile

_S = re.compile(r"[^A-Za-z0-9_]+")


def new_key(source_path: str) -> str:
    """Collision-free key from the LeRobot source_path (domain-stripped full path)."""
    parts = source_path.split("/")
    rest = [p for p in parts[1:] if p != "videos"]     # drop leading Surgical/Endoscopy + 'videos'
    stem = rest[-1].rsplit(".", 1)[0]
    mid = "_".join(rest[:-1])
    src = _S.sub("_", f"{mid}_{stem}").strip("_")
    return f"openh__{src}_clip_0000"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--staging", required=True, help="Dir of openh__range_*.tar (collided keys).")
    p.add_argument("--rekeyed", required=True, help="Output dir for re-keyed staging tars.")
    p.add_argument("--final", required=True, help="Final resharded openh/ dir.")
    p.add_argument("--reshard-script", default=None,
                   help="Path to reshard_webdataset.py (default: sibling of this script).")
    p.add_argument("--no-reshard", action="store_true", help="Only re-key; skip reshard.")
    return p.parse_args()


def _iter_samples(tar_path):
    """Yield [(TarInfo, bytes), ...] per physical sample, split on each .mp4 boundary.

    Members are written contiguously as <key>.mp4, <key>.json, <key>.cls. We must
    NOT group by old key — colliding old keys within one tar would merge distinct
    physical samples and silently drop clips (the exact bug this script fixes).
    Instead start a fresh group at every .mp4 member, so each physical sample is
    emitted independently regardless of key collisions."""
    with tarfile.open(tar_path, "r|") as tf:
        cur = []
        for m in tf:
            if not m.isfile():
                continue
            ext = m.name.split(".", 1)[1] if "." in m.name else ""
            data = tf.extractfile(m).read()
            if ext == "mp4" and cur:
                yield cur
                cur = []
            cur.append((m, ext, data))
        if cur:
            yield cur


def main():
    args = parse_args()
    os.makedirs(args.rekeyed, exist_ok=True)
    tars = sorted(glob.glob(os.path.join(args.staging, "openh__range_*.tar")))
    if not tars:
        raise SystemExit(f"no openh__range_*.tar in {args.staging}")

    seen = set()
    n_in = n_out = n_dup = n_incomplete = 0
    for t in tars:
        base = os.path.basename(t)
        out = os.path.join(args.rekeyed, base)
        tmp = out + ".tmp"
        n_this = 0
        with tarfile.open(tmp, "w") as w:
            for members in _iter_samples(t):
                n_in += 1
                exts = {ext: data for (_mi, ext, data) in members}
                if not {"mp4", "json", "cls"} <= set(exts):
                    n_incomplete += 1
                    continue
                sp = json.loads(exts["json"])["source_path"]
                nk = new_key(sp)
                if nk in seen:
                    n_dup += 1
                    continue  # validated offline to be 0
                seen.add(nk)
                for ext in ("mp4", "json", "cls"):
                    data = exts[ext]
                    ni = tarfile.TarInfo(name=f"{nk}.{ext}")
                    ni.size = len(data)
                    ni.mtime = 0
                    w.addfile(ni, io.BytesIO(data))
                n_out += 1
                n_this += 1
        os.replace(tmp, out)
        print(f"  rekeyed {base}: {n_this} samples", flush=True)

    print(f"REKEY DONE: in={n_in} out={n_out} dropped_dupes={n_dup} "
          f"incomplete={n_incomplete} distinct={len(seen)}", flush=True)
    if n_dup:
        print(f"WARNING: {n_dup} residual dup keys dropped — investigate new_key().", file=sys.stderr)

    if args.no_reshard:
        return
    reshard = args.reshard_script or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "reshard_webdataset.py")
    tgt = max(64, n_out // 16)
    cmd = [sys.executable, reshard, "--input", args.rekeyed, "--output", args.final,
           "--prefix", "openh", "--target-shards", str(tgt), "--seed", "0", "--force"]
    print("reshard:", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
