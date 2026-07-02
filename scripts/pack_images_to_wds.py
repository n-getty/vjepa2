#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Pack surgical IMAGE datasets into the project's WebDataset sample contract.

These feed the vjepa_2_1 IMAGE BRANCH (img_data/img_mask), which our loader
reads as single-frame clips: src/datasets/webdataset.py:321 detects an
``image.jpg/png/...`` member, decodes it, and repeats the frame to fpc (with
min_clip_std auto-skipped for images). So each image becomes one sample:
    <key>.image.jpg   the still (re-encoded to JPEG for a uniform member name)
    <key>.json        {"source_dataset","source_path","label":0}
    <key>.cls         "0"
Note the media member is ``.image.jpg`` (NOT ``.mp4``) so the loader's image
branch picks it up. Keys are ``<dataset>__<source>_clip_<NNN>`` so
scripts/reshard_webdataset.py groups + shuffles them exactly like the video
sources; run reshard on the staging dir afterwards to make ``<name>_img/``.

Per-dataset extractors handle each archive's quirks and — critically — pack ONLY
real surgical frames, never masks/labels/features:
  - hyperkvasir : train/valid/test.zip, class-folder JPEGs (all real frames).
  - dsad        : DSAD.zip, keep ONLY ``**/image*.png`` (the 116k mask*.png and
                  anno_*/ masks are segmentation labels — NOT frames).
  - esad        : train/val zips, keep ``*.jpg`` only (skip the YOLO .txt labels).
  - psi_ava     : PSI-AVA.tar.gz, keep ONLY ``PSI-AVA/keyframes/CASE*/*.jpg``
                  (skip the multi-GB def_DETR features / annotations / weights).

CholecSeg8k is intentionally NOT here: its frames are Cholec80 (already held as
video) and its only new signal is masks the branch never consumes.

Usage (frameworks python; Pillow required):
  python3 scripts/pack_images_to_wds.py --dataset dsad \
      --archive /flare/.../incoming_robotic/dsad/DSAD.zip \
      --output-staging /flare/.../surg_vid_webdataset_resharded/dsad_img_staging
  # smoke: cap the count
  python3 scripts/pack_images_to_wds.py --dataset hyperkvasir \
      --archives .../hyperkvasir/train.zip .../hyperkvasir/valid.zip \
      --output-staging .../hyperkvasir_img_staging --max-images 50
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

from PIL import Image

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
_SANITIZE = re.compile(r"[^A-Za-z0-9_]+")
# Cap the number of images per staging tar so reshard has many groups to shuffle.
IMAGES_PER_TAR = 512


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", required=True,
                   choices=["hyperkvasir", "dsad", "esad", "psi_ava"])
    p.add_argument("--archive", default=None, help="Single source archive (zip or tar.gz).")
    p.add_argument("--archives", default=None, nargs="+",
                   help="Multiple source archives (e.g. hyperkvasir train/valid/test).")
    p.add_argument("--output-staging", required=True, help="Dir for staging *.tar (created).")
    p.add_argument("--max-images", type=int, default=None,
                   help="Process at most this many images (smoke test).")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--force", action="store_true", help="Overwrite existing staging tars.")
    return p.parse_args()


def _sanitize(stem: str) -> str:
    return _SANITIZE.sub("_", stem).strip("_")


def _to_jpeg_bytes(raw: bytes, quality: int) -> bytes | None:
    """Decode any image, drop alpha, re-encode to JPEG. None if undecodable."""
    try:
        im = Image.open(io.BytesIO(raw))
        im = im.convert("RGB")
    except Exception:
        return None
    out = io.BytesIO()
    im.save(out, format="JPEG", quality=quality)
    return out.getvalue()


def _add_bytes(tar: tarfile.TarFile, name: str, data: bytes):
    ti = tarfile.TarInfo(name=name)
    ti.size = len(data)
    ti.mtime = 0
    tar.addfile(ti, io.BytesIO(data))


class TarPacker:
    """Round-robins samples into capped staging tars: <dataset>__part_<NNN>.tar."""

    def __init__(self, out_dir, dataset, quality, force):
        self.out_dir = out_dir
        self.dataset = dataset
        self.quality = quality
        self.force = force
        self.part = 0
        self.in_part = 0
        self.n_ok = 0
        self.n_bad = 0
        self.tar = None
        os.makedirs(out_dir, exist_ok=True)

    def _rotate(self):
        if self.tar is not None:
            self.tar.close()
        path = os.path.join(self.out_dir, f"{self.dataset}__part_{self.part:04d}.tar")
        self.tar = tarfile.open(path, "w")
        self.in_part = 0

    def add(self, raw_bytes, source_ref):
        if self.tar is None or self.in_part >= IMAGES_PER_TAR:
            self._rotate()
            self.part += 1
        jpg = _to_jpeg_bytes(raw_bytes, self.quality)
        if jpg is None:
            self.n_bad += 1
            return
        # source = the part index; clip index = position within part. Together the
        # key is unique and reshard-parseable (<dataset>__part_<NNN>_clip_<MMM>).
        key = f"{self.dataset}__part_{self.part:04d}_clip_{self.in_part:04d}"
        _add_bytes(self.tar, f"{key}.image.jpg", jpg)
        meta = json.dumps({"source_dataset": self.dataset,
                           "source_path": source_ref, "label": 0}).encode("utf-8")
        _add_bytes(self.tar, f"{key}.json", meta)
        _add_bytes(self.tar, f"{key}.cls", b"0")
        self.in_part += 1
        self.n_ok += 1

    def close(self):
        if self.tar is not None:
            self.tar.close()


# ---- per-dataset iterators: yield (raw_bytes, source_ref) for REAL frames only ----

def iter_zip_images(archive, keep):
    """Yield images from a zip whose member name passes keep(name)."""
    zf = zipfile.ZipFile(archive)
    for n in zf.namelist():
        if n.endswith("/"):
            continue
        if keep(n):
            yield zf.read(n), f"{os.path.abspath(archive)}::{n}"
    zf.close()


def _is_image(n):
    return n.lower().endswith(IMAGE_EXTS)


def iter_hyperkvasir(archives):
    # class-folder JPEGs, all real frames.
    for arc in archives:
        yield from iter_zip_images(arc, _is_image)


def iter_dsad(archives):
    # KEEP only basename image*.png; DROP mask*.png and anno_*/ masks.
    def keep(n):
        base = n.rsplit("/", 1)[-1].lower()
        return base.startswith("image") and base.endswith(".png")
    for arc in archives:
        yield from iter_zip_images(arc, keep)


def iter_esad(archives):
    # KEEP .jpg frames; DROP .txt YOLO labels and obj.names.
    for arc in archives:
        yield from iter_zip_images(arc, _is_image)


def iter_psi_ava(archives):
    # tar.gz: KEEP only PSI-AVA/keyframes/CASE*/*.jpg (skip features/annots/weights).
    for arc in archives:
        with tarfile.open(arc, "r|gz") as tf:  # streaming
            for m in tf:
                if not m.isfile():
                    continue
                nl = m.name.replace("\\", "/")
                if "/keyframes/" in nl and _is_image(nl):
                    f = tf.extractfile(m)
                    if f is not None:
                        yield f.read(), f"{os.path.abspath(arc)}::{m.name}"


ITERATORS = {
    "hyperkvasir": iter_hyperkvasir,
    "dsad": iter_dsad,
    "esad": iter_esad,
    "psi_ava": iter_psi_ava,
}


def main():
    args = parse_args()
    archives = args.archives or ([args.archive] if args.archive else [])
    if not archives:
        raise SystemExit(f"{args.dataset}: pass --archive or --archives")
    out_dir = args.output_staging

    # Guard against a re-run leaving mixed old/new parts unless --force.
    if os.path.isdir(out_dir) and any(f.endswith(".tar") for f in os.listdir(out_dir)):
        if not args.force:
            raise SystemExit(f"{out_dir} already has *.tar; pass --force to overwrite")
        for f in os.listdir(out_dir):
            if f.endswith(".tar"):
                os.remove(os.path.join(out_dir, f))

    print(f"Pillow pack; dataset={args.dataset}; archives={len(archives)}; out={out_dir}",
          flush=True)
    packer = TarPacker(out_dir, args.dataset, args.jpeg_quality, args.force)
    t0 = time.time()
    for i, (raw, ref) in enumerate(ITERATORS[args.dataset](archives)):
        if args.max_images is not None and i >= args.max_images:
            break
        packer.add(raw, ref)
        if packer.n_ok and packer.n_ok % 2000 == 0:
            print(f"  ...{packer.n_ok} packed ({time.time()-t0:.0f}s)", flush=True)
    packer.close()

    summary = {"dataset": args.dataset, "packed": packer.n_ok,
               "undecodable": packer.n_bad, "parts": packer.part}
    with open(os.path.join(out_dir, "_pack_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"DONE {args.dataset}: {packer.n_ok} images, {packer.n_bad} undecodable, "
          f"{packer.part} parts, {time.time()-t0:.1f}s -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
