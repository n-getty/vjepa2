#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Re-gate already-segmented LEMON staging tars against a (denser) pHash ref.

The first segmentation pass gated each source video against a SPARSE reference
(72% eval self-recall) and dropped only 1 eval-leak. Rather than re-segment
925GB, this hashes the CLIPS ALREADY IN each staging tar against the rebuilt
DENSE reference and removes whole source tars whose dup fraction exceeds the
threshold. Segmentation is deterministic and lossless (stream copy), so the
staged clips are faithful samples of each source video — hashing them is
equivalent to re-gating the source, at a fraction of the cost (no re-decode of
the raw 925GB; the staged clips are already the kept content).

Per source tar: sample up to --max-frames grayscale frames spread across its
clips, pHash them, compare to eval + train pools. If dup_eval > threshold ->
eval-leak; elif dup_train > threshold -> train-dup. Matching tars are moved to
<staging>/_dropped/ (not deleted) so the decision is auditable/reversible.

Writes _regate_<start>_<end>.json partials (parallel by tar range); --finalize
merges them into _regate_summary.json.

Usage (frameworks python; driven by regate_lemon_pbs.sh):
  python3 scripts/regate_lemon_staging.py --staging <dir> --phash-ref <json> \
      --threshold 0.10 --max-frames 64 --tar-start N --tar-end M
  python3 scripts/regate_lemon_staging.py --staging <dir> --finalize
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import tarfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phash_util as ph  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--staging", required=True, help="lemon_staging dir (lemon__*.tar).")
    p.add_argument("--phash-ref", default=None, help="Dense phash_ref.json.")
    p.add_argument("--threshold", type=float, default=0.10)
    p.add_argument("--max-frames", type=int, default=64,
                   help="Max frames sampled across a tar's clips for the gate.")
    p.add_argument("--frames-per-clip", type=int, default=4,
                   help="Frames per clip before the max-frames cap.")
    p.add_argument("--tar-start", type=int, default=None)
    p.add_argument("--tar-end", type=int, default=None)
    p.add_argument("--dry-run", action="store_true", help="Report, do not move tars.")
    p.add_argument("--finalize", action="store_true")
    return p.parse_args()


def _hash_tar(tar_path, fpc, cap):
    """Sample frames across a staging tar's clips -> list of pHashes."""
    hs = []
    with tarfile.open(tar_path, "r|") as tf:
        for m in tf:
            if not (m.isfile() and m.name.endswith(".mp4")):
                continue
            try:
                buf = tf.extractfile(m).read()
                for fr in ph.sample_gray_frames(io.BytesIO(buf), n=fpc, seek=True):
                    hs.append(ph.phash_gray(fr))
            except Exception:
                continue
            if len(hs) >= cap:
                break
    return hs


def run_range(args):
    ev, tr, meta = ph.load_ref(args.phash_ref)
    print(f"[regate] ref: {ev.size} eval + {tr.size} train; thr={args.threshold}", flush=True)
    tars = sorted(glob.glob(os.path.join(args.staging, "lemon__*.tar")))
    lo = args.tar_start or 0
    hi = args.tar_end if args.tar_end is not None else len(tars)
    sel = tars[lo:hi]
    dropdir = os.path.join(args.staging, "_dropped")
    if not args.dry_run:
        os.makedirs(dropdir, exist_ok=True)
    results = []
    n_leak = n_dup = 0
    for tp in sel:
        src = os.path.basename(tp)[len("lemon__"):-len(".tar")]
        try:
            hs = _hash_tar(tp, args.frames_per_clip, args.max_frames)
        except Exception as e:
            print(f"  [err] {src}: {repr(e)[:60]}", flush=True)
            continue
        if not hs:
            continue
        dfe = ph.dup_fraction(hs, ev)
        dft = ph.dup_fraction(hs, tr)
        reason = None
        if dfe > args.threshold:
            reason = "eval-leak"; n_leak += 1
        elif dft > args.threshold:
            reason = "train-dup"; n_dup += 1
        if reason:
            print(f"  [{reason}] {src}: eval={dfe:.3f} train={dft:.3f}", flush=True)
            if not args.dry_run:
                os.replace(tp, os.path.join(dropdir, os.path.basename(tp)))
        results.append({"source": src, "dup_eval": round(dfe, 4),
                        "dup_train": round(dft, 4), "reason": reason,
                        "n_frames": len(hs)})
    pp = os.path.join(args.staging, f"_regate_{lo}_{hi}.json")
    with open(pp, "w") as f:
        json.dump({"range": [lo, hi], "n_tars": len(sel),
                   "eval_leak": n_leak, "train_dup": n_dup,
                   "results": results}, f)
    print(f"DONE [{lo}:{hi}): {len(sel)} tars, {n_leak} eval-leak, {n_dup} train-dup -> {pp}",
          flush=True)


def finalize(args):
    parts = sorted(glob.glob(os.path.join(args.staging, "_regate_*.json")))
    allr = []
    n_leak = n_dup = 0
    for pp in parts:
        d = json.load(open(pp))
        n_leak += d["eval_leak"]; n_dup += d["train_dup"]
        allr.extend(d["results"])
    # near-miss visibility: how many kept videos sit just under threshold
    kept = [r for r in allr if r["reason"] is None]
    near = sorted(kept, key=lambda r: max(r["dup_eval"], r["dup_train"]), reverse=True)[:20]
    summ = {
        "n_tars": len(allr), "eval_leak": n_leak, "train_dup": n_dup,
        "dropped_total": n_leak + n_dup,
        "top20_near_miss_kept": near,
    }
    out = os.path.join(args.staging, "_regate_summary.json")
    with open(out, "w") as f:
        json.dump(summ, f, indent=2)
    print(f"FINALIZE {len(parts)} partials: {len(allr)} tars, "
          f"{n_leak} eval-leak + {n_dup} train-dup dropped -> {out}", flush=True)
    if near:
        print("top near-miss KEPT (max dup just under threshold):", flush=True)
        for r in near[:8]:
            print(f"  {r['source']}: eval={r['dup_eval']} train={r['dup_train']}", flush=True)


def main():
    args = parse_args()
    t0 = time.time()
    if args.finalize:
        finalize(args)
    else:
        if not args.phash_ref:
            raise SystemExit("--phash-ref required unless --finalize")
        run_range(args)
    print(f"elapsed {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
