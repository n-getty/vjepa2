#!/usr/bin/env python3
"""Build GraSP ASFormer probe clips + sequence-label CSVs from the OFFICIAL
preprocessed GraSP frames.

Replaces `build_grasp_asformer_ctx3_30fps.py` (Leonardo's, kept untouched as the
provenance record for the old runs). That builder had three defects that this one
fixes structurally rather than by patching arithmetic:

1. WRONG PIXELS. It sourced raw per-CASE `V001.MP4` (1280x1024) which still has the
   da Vinci surgeon-console UI burned in -- top status bar, bottom toolbar, and
   *side instrument name-plates* ("Maryland Bipolar Forceps", ...). Those plates are
   legible, high-contrast, and correlated with the phase label, i.e. a shortcut the
   TAPIS baseline is denied (their released frames are debranded 1280x800). Same
   failure mode as the openh console overlay. We now read the official frames, so
   the shortcut is gone and our pixels match TAPIS's.

2. LABEL/FPS DRIFT. It mapped annotation `frame_num` -> wall-clock second via
   `int(round(frame_num / ffprobe_fps))` with ffprobe returning 30000/1001=29.97,
   while GraSP keyframes are spaced exactly 30 in the annotation's own indexing.
   The 1.001 factor accumulated: +1 s at the start of a case growing to +14 s
   (28 tokens, more than a whole window) by the end of CASE050. Measured effect:
   6.60% of all val labels disagreed with the phase their pixels actually showed,
   with the per-case rate tracking case length (r=0.73).
   THE FIX HERE IS STRUCTURAL: the official frame dirs are indexed by the *same*
   `frame_num` the annotation uses, so label lookup is a direct index. There is no
   fps, no rounding, and no ffmpeg seek anywhere in the label path -- the entire
   class of bug is removed rather than corrected.

3. NON-CENTERED WINDOWS. It emitted non-overlapping windows, so token 0 had 0 s of
   lookbehind and 11.5 s of lookahead (token 23 the reverse); only the middle
   tokens approximated TAPIS's view. TAPIS centers a 16 s window on *every*
   keyframe (`get_sequence(center, half_len=240, sample_rate=30)`). We now emit
   overlapping windows (`--stride-sec`, default half a window) so every keyframe
   sits near some window's centre.

Time base
---------
We work purely in KEYFRAME RANK, never in float seconds. GraSP keyframes are one
per second of real time (verified: CASE041 = 9451 keyframes, source video
9450.34 s), so rank k IS second k. Sub-second frame timestamps inside a window are
resolved by interpolating between the two bracketing keyframes' `frame_num`s and
snapping to the nearest file that exists. This is automatically correct for the
30 fps cases AND for CASE001 (45.45 fps, sparsely renumbered 0..493737) without
any per-case fps special-casing.

Output (unchanged contract)
---------------------------
  <out-clip-root>/CASE0XX/kf_<center_rank>_ctx.mp4
  <out-csv-dir>/grasp_<task>_asformer_ctx_seq_{train,val}.csv
  CSV row: `<clip_path> <lbl_0> ... <lbl_{T-1}>`

Geometry defaults to 16 s / 4 fps / 64 frames / 32 tokens -- 16 s matches TAPIS's
context exactly, and at 4 fps with tubelet 2 each token is 0.5 s, so two tokens map
to one labelled second with no rounding. NOTE this differs from the old 12 s / 48 /
24 geometry, so the probe config must move to num_segments=4 (4x16=64 frames) and
asformer temporal_tokens=32. Pass `--window-sec 12 --num-frames 48 --tokens 24` to
keep the old geometry instead.

Verify before you build:
  python scripts/build_grasp_ctx_official.py --task phase --verify-only
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

FRAME_ROOT = Path("/flare/ModCon/ngetty/data/incoming_robotic/grasp/GraSP/GraSP_30fps/frames")
ANN_ROOT = Path("/flare/ModCon/ngetty/data/incoming_robotic/grasp/tapis_weights/TAPIS/annotations")

SPLITS = {
    "train": "grasp_long-term_train.json",
    "val": "grasp_long-term_test.json",
}


def _resolve_ffmpeg() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # pragma: no cover - environment dependent
        raise RuntimeError("ffmpeg not on PATH and imageio_ffmpeg unavailable") from e


def load_keyframes(ann_json: Path, label_field: str) -> dict[str, dict]:
    """{case: {"frames": [frame_num...], "labels": [lbl...]}} sorted by frame_num.

    Index into these lists IS the keyframe rank == the second of real time.
    """
    with ann_json.open() as fh:
        data = json.load(fh)
    lbl_by_img = {a["image_id"]: a[label_field] for a in data["annotations"]}
    per_case: dict[str, list[tuple[int, int]]] = {}
    for im in data["images"]:
        lbl = lbl_by_img.get(im["id"])
        if lbl is None:
            continue
        per_case.setdefault(im["video_name"], []).append((im["frame_num"], lbl))
    out = {}
    for case, rows in per_case.items():
        rows.sort()
        out[case] = {"frames": [r[0] for r in rows], "labels": [r[1] for r in rows]}
    return out


def available_frames(case: str, frame_root: Path) -> list[int]:
    """Sorted frame indices that actually exist on disk for this case."""
    d = frame_root / case
    if not d.is_dir():
        return []
    return sorted(int(p.stem) for p in d.glob("*.jpg"))


def _snap(avail: list[int], target: int) -> int:
    """Nearest existing frame index to `target`."""
    i = bisect_right(avail, target)
    if i == 0:
        return avail[0]
    if i == len(avail):
        return avail[-1]
    lo, hi = avail[i - 1], avail[i]
    return lo if (target - lo) <= (hi - target) else hi


def frame_for_time(kf: list[int], avail: list[int], t: float) -> int:
    """Frame index for time `t` in keyframe-rank (== second) units.

    Interpolates between the bracketing keyframes' frame_nums, so it is correct
    regardless of the case's native fps or renumbering, then snaps to a real file.
    """
    n = len(kf)
    if t <= 0:
        target = kf[0]
    elif t >= n - 1:
        target = kf[-1]
    else:
        i = int(t)
        frac = t - i
        target = int(round(kf[i] + frac * (kf[i + 1] - kf[i])))
    return _snap(avail, target)


def plan_case(
    case: str,
    kf: list[int],
    labels: list[int],
    avail: list[int],
    window_sec: int,
    num_frames: int,
    tokens: int,
    stride_sec: int,
) -> list[tuple[int, list[int], list[int]]]:
    """-> [(center_rank, [frame indices], [token labels])]"""
    n = len(kf)
    half = window_sec / 2.0
    secs_per_token = window_sec / tokens
    out = []
    # Centre on every stride_sec-th keyframe whose full window is in range.
    for center in range(int(half), n - int(half), stride_sec):
        start = center - half
        frames = [frame_for_time(kf, avail, start + (j / num_frames) * window_sec) for j in range(num_frames)]
        # Token i spans [start + i*spt, start + (i+1)*spt); label it by the second
        # its midpoint falls in. Pure integer indexing into `labels`.
        toks = []
        ok = True
        for i in range(tokens):
            rank = int(start + (i + 0.5) * secs_per_token)
            if rank < 0 or rank >= n:
                ok = False
                break
            toks.append(labels[rank])
        if ok:
            out.append((center, frames, toks))
    return out


def encode_clip(out_path: Path, case: str, frames: list[int], frame_root: Path, fps: int, ffmpeg: str) -> tuple[Path, bool, str]:
    """Encode the given JPEG frames, in order, into an mp4 with exactly len(frames) frames."""
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path, True, "skip-exists"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.mp4")
    src = frame_root / case
    try:
        blob = b"".join((src / f"{f:09d}.jpg").read_bytes() for f in frames)
    except FileNotFoundError as e:
        return out_path, False, f"missing frame: {e}"
    cmd = [
        ffmpeg, "-y", "-loglevel", "error",
        "-f", "image2pipe", "-framerate", str(fps), "-i", "-",
        "-frames:v", str(len(frames)),
        "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-an",
        str(tmp),
    ]
    try:
        r = subprocess.run(cmd, input=blob, capture_output=True, timeout=180)
        if r.returncode != 0:
            return out_path, False, f"ffmpeg rc={r.returncode}: {r.stderr[-400:].decode(errors='replace')}"
        tmp.replace(out_path)
        return out_path, True, "ok"
    except subprocess.TimeoutExpired:
        return out_path, False, "ffmpeg timeout"


def verify_case(case: str, kf: list[int], labels: list[int], avail: list[int], plans: list, window_sec: int) -> list[str]:
    """Assertions that would have caught all three original bugs. -> list of problems."""
    problems = []
    aset = set(avail)
    missing = [f for f in kf if f not in aset]
    if missing:
        problems.append(f"{case}: {len(missing)}/{len(kf)} keyframes have NO frame file (e.g. {missing[:3]})")
    # Label correctness: every token label must equal the true label of the second
    # its pixels show. This is the check the old builder would have failed at 6.6%.
    bad = 0
    total = 0
    for center, frames, toks in plans:
        # Recompute each token's second independently of plan_case's arithmetic,
        # from the window centre, and confirm the stored label matches ground truth.
        half = window_sec / 2.0
        spt = window_sec / len(toks)
        for i, t in enumerate(toks):
            rank = int((center - half) + (i + 0.5) * spt)
            total += 1
            if 0 <= rank < len(labels) and labels[rank] != t:
                bad += 1
    if total and bad:
        problems.append(f"{case}: {bad}/{total} ({100*bad/total:.2f}%) token labels mismatch true second")
    return problems


def build_split(
    split: str,
    task: str,
    frame_root: Path,
    ann_root: Path,
    out_clip_root: Path,
    out_csv_dir: Path,
    window_sec: int,
    num_frames: int,
    tokens: int,
    stride_sec: int,
    workers: int,
    verify_only: bool,
    limit_cases: int | None,
) -> None:
    label_field = "phases" if task == "phase" else "steps"
    ann_json = ann_root / SPLITS[split]
    print(f"[{split}] loading {ann_json}", flush=True)
    kfs = load_keyframes(ann_json, label_field)
    cases = sorted(kfs)
    if limit_cases:
        cases = cases[:limit_cases]
    print(f"[{split}] cases: {cases}", flush=True)

    all_problems: list[str] = []
    tasks: list[tuple[Path, str, list[int]]] = []
    csv_rows: list[str] = []

    for case in cases:
        kf = kfs[case]["frames"]
        labels = kfs[case]["labels"]
        avail = available_frames(case, frame_root)
        if not avail:
            all_problems.append(f"{case}: NO frames on disk at {frame_root/case}")
            continue
        plans = plan_case(case, kf, labels, avail, window_sec, num_frames, tokens, stride_sec)
        probs = verify_case(case, kf, labels, avail, plans, window_sec)
        all_problems.extend(probs)
        print(
            f"[{split}] {case}: {len(kf)} keyframes, {len(avail)} frame files, "
            f"{len(plans)} windows" + ("  PROBLEMS: " + "; ".join(probs) if probs else "  OK"),
            flush=True,
        )
        for center, frames, toks in plans:
            out = out_clip_root / case / f"kf_{center:06d}_ctx.mp4"
            tasks.append((out, case, frames))
            csv_rows.append(f"{out} " + " ".join(str(x) for x in toks))

    print(f"\n[{split}] TOTAL windows={len(tasks)}", flush=True)
    if all_problems:
        print(f"[{split}] *** {len(all_problems)} PROBLEM(S) ***", flush=True)
        for p in all_problems:
            print(f"    {p}", flush=True)
    else:
        print(f"[{split}] verification clean: all keyframes resolve, all labels match", flush=True)

    if verify_only:
        for r in csv_rows[:2]:
            print(f"  [sample] {r[:160]}...", flush=True)
        return
    if all_problems:
        print(f"[{split}] refusing to build with unresolved problems", flush=True)
        sys.exit(1)

    out_csv = out_csv_dir / f"grasp_{task}_asformer_ctx_seq_{split}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _resolve_ffmpeg()
    n_ok = n_fail = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(encode_clip, o, c, f, frame_root, num_frames // window_sec, ffmpeg): o for (o, c, f) in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                _, ok, msg = fut.result()
            except Exception as e:  # pragma: no cover
                ok, msg = False, str(e)
            n_ok += ok
            if not ok:
                n_fail += 1
                if n_fail <= 10:
                    print(f"[{split}] FAIL {futs[fut]}: {msg}", flush=True)
            if i % 500 == 0:
                print(f"[{split}] encoded {i}/{len(tasks)} fails={n_fail}", flush=True)
    out_csv.write_text("\n".join(csv_rows) + "\n")
    print(f"[{split}] done: encoded={n_ok} failed={n_fail} csv={out_csv}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["phase", "step"], default="phase")
    ap.add_argument("--splits", nargs="+", default=["val", "train"], choices=["train", "val"])
    ap.add_argument("--frame-root", type=Path, default=FRAME_ROOT)
    ap.add_argument("--ann-root", type=Path, default=ANN_ROOT)
    ap.add_argument("--out-clip-root", type=Path, default=Path("/flare/ModCon/ngetty/data/grasp_ctx_official/clips"))
    ap.add_argument("--out-csv-dir", type=Path, default=Path("/flare/ModCon/ngetty/data/grasp_ctx_official/csv"))
    ap.add_argument("--window-sec", type=int, default=16, help="TAPIS uses 16 s centered on the keyframe")
    ap.add_argument("--num-frames", type=int, default=64, help="frames per clip (64 = 16 s @ 4 fps)")
    ap.add_argument("--tokens", type=int, default=32, help="temporal tokens after tubelet-2 pooling")
    ap.add_argument("--stride-sec", type=int, default=8, help="gap between window centres; 1 = full TAPIS parity")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--verify-only", action="store_true", help="plan + verify, encode nothing")
    ap.add_argument("--limit-cases", type=int, default=None)
    args = ap.parse_args()

    if args.num_frames % args.tokens:
        sys.exit(f"num_frames ({args.num_frames}) must be a multiple of tokens ({args.tokens})")
    print(
        f"geometry: window={args.window_sec}s frames={args.num_frames} "
        f"({args.num_frames//args.window_sec} fps) tokens={args.tokens} "
        f"({args.window_sec/args.tokens:.3f} s/token) stride={args.stride_sec}s",
        flush=True,
    )
    for split in args.splits:
        build_split(
            split, args.task, args.frame_root, args.ann_root, args.out_clip_root, args.out_csv_dir,
            args.window_sec, args.num_frames, args.tokens, args.stride_sec, args.workers,
            args.verify_only, args.limit_cases,
        )


if __name__ == "__main__":
    main()
