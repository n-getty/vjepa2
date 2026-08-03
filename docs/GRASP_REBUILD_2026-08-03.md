# GraSP probe data — rebuilt 2026-08-03 (read this before running a GraSP probe)

**TL;DR for collaborators:** the old GraSP clips had wrong labels 6.6% of the time and
leaked a visual shortcut. Point your config at the new CSVs and change 3 numbers.
Old results are not comparable to new ones.

---

## 1. Where the build is

```
/flare/ModCon/ngetty/data/grasp_ctx_official/
├── clips/CASE0XX/kf_<center_second>_ctx.mp4     # 14,545 clips, 65 GB
└── csv/
    ├── grasp_phase_asformer_ctx_seq_train.csv   # 9,191 rows (8 train cases)
    └── grasp_phase_asformer_ctx_seq_val.csv     # 5,354 rows (5 official TEST cases)
```

World-readable, group `ModCon` — read it in place, do not copy.

CSV format: `<clip_path> <lbl_0> ... <lbl_31>` = 33 space-delimited columns.
Each clip is **1280x800, 4 fps, 16.0 s, 64 frames**, centered on its keyframe.

Ready-made configs (both encoder arms, export + probe):
`configs/heads/grasp/official_ctx16/grasp_off16_{v2_e324,meta2b}_{export,probe}.yaml`

---

## 2. Changes needed to point an existing GraSP config at the new build

Exactly five edits. Copy from `configs/heads/grasp/official_ctx16/` if you'd rather not
hand-edit.

| key | old | new |
|---|---|---|
| `data.dataset_train` | `.../probes/csv/grasp_phase_asformer_ctx3_seq_train.csv` | `/flare/ModCon/ngetty/data/grasp_ctx_official/csv/grasp_phase_asformer_ctx_seq_train.csv` |
| `data.dataset_val` | `.../grasp_phase_asformer_ctx3_seq_val.csv` | `/flare/ModCon/ngetty/data/grasp_ctx_official/csv/grasp_phase_asformer_ctx_seq_val.csv` |
| `data.num_segments` | `3` | **`4`** |
| `classifier.asformer_kwargs.num_clips` | `3` | **`4`** |
| `classifier.asformer_kwargs.temporal_tokens` | `24` | **`32`** |

`frames_per_clip: 16`, `tokens_per_clip: 8`, and `resolution: 384` are unchanged
(4 x 16 = 64 frames; 4 x 8 = 32 tokens).

**Three things that are easy to get wrong:**

1. **Use a FRESH `train_cache_root` / `val_cache_root` / `folder` / `tag`.** The geometry
   changed, so any existing feature cache is invalid. Reusing a path will silently
   resume onto stale features or collide with an old run.
2. **Export both arms of a comparison at the SAME node count.** `DistributedSampler`
   round-up padding varies with world size — a 24-rank vs 48-rank export gave the two
   arms 3,576 vs 3,600 val samples, i.e. they were not scored on identical data.
3. **`class_weights` are marginally different** (recomputed on the new label
   distribution; within 0.4% of the old). Copy them from the shipped configs, or drop
   `class_weights` entirely — unweighted CE measured slightly *better* for macro-mAP.

No launcher change. `scripts/run_asformer_probe_aurora.sh` derives topology from
`PBS_NODEFILE` and is geometry-agnostic.

Scoring is unchanged: `scripts/eval_grasp_map_cached.py --config <probe.yaml>`.
`scripts/eval_grasp_map_aggr.py` also works (fixed for the 4-clip layout in d9b004a).

---

## 3. Why the rebuild happened

Three defects in the old builder (`build_grasp_asformer_ctx3_30fps.py`, Leonardo's —
left untouched as the provenance record). All three sat upstream of **every GraSP number
we have ever reported**.

**(a) Labels were wrong 6.60% of the time.** It mapped annotation `frame_num` to a second
via `int(round(frame_num / ffprobe_fps))`, with ffprobe returning `30000/1001` = 29.97 —
but GraSP keyframes are spaced **exactly 30**. The 1.001 factor accumulates: **+1 s at the
start of a case, growing to +14 s — 28 tokens, more than one whole 24-token window — by
the end of CASE050.** Measured: 5,631/85,304 val tokens carry a label that disagrees with
the phase their pixels show, and the per-case rate tracks case length (r=0.73, CASE051
2.94% -> CASE050 7.92%).

*Fixed structurally, not arithmetically:* the official frame directories are indexed by
the same `frame_num` the annotation uses, so label lookup is now a direct index. There is
no fps, no rounding, and no ffmpeg seek anywhere in the label path.

**(b) Wrong pixels — a visual shortcut.** Clips were cut from the RAW per-CASE da Vinci
video (1280x1024) with the surgeon-console UI burned in, including **legible instrument
name-plates that correlate with the phase label** ("Maryland Bipolar Forceps", ...). The
official GraSP release is debranded 1280x800. The old 384 center crop excluded the top and
bottom bars but **cut through the side name-plates**. Same failure mode as the openh
console-overlay poison. Direction matters: this **inflated** our scores, so the true gap to
SOTA was probably wider than reported, not narrower.

**(c) Windows were never keyframe-centered.** Non-overlapping 12 s blocks meant token 0 had
**zero lookbehind and 11.5 s lookahead**. TAPIS centers 16 s on every keyframe. Now matched.

---

## 4. Expect the numbers to go DOWN, and don't mix them with old ones

Removing (b) removes a shortcut, so the rebuilt absolute should land **below** the old
69.31. That is the fix working — a number comparable to published SOTA is worth more than
a higher number that isn't.

**Do not compare new numbers to any pre-2026-08-03 GraSP result.** Different pixels,
different labels, different window geometry. See the confound banner in
`docs/PROBE_RESULTS_LEDGER.md` §3.

Two further cautions for anyone drawing conclusions:

- **Single-seed noise on this probe is ~3 mAP, not the ±1.4 previously assumed.** Two
  encoders whose features are cos=0.9998 scored 2.91 apart, and best.pt-vs-latest.pt on
  one encoder swung 3.17. **Run 3 seeds before claiming any delta under ~3 mAP.**
- **Model selection is on the TEST split.** `dataset_val` is the official GraSP TEST
  cases (CASE041/047/050/051/053) and best-head/best-epoch are chosen on it. Fine for a
  symmetric A-vs-B comparison; inflates absolutes. For a blind number, carve a val split
  from the 8 TRAIN cases and select on that. Not yet done.

---

## 5. Rebuilding / other tasks

```bash
module load frameworks
# verify only, encodes nothing (fast, run this first):
python scripts/build_grasp_ctx_official.py --task phase --verify-only
# build (login node, ~30 min at 32 workers, no queue needed):
python scripts/build_grasp_ctx_official.py --task phase --splits val train \
    --stride-sec 8 --workers 32
```

`--task step` builds the 21-class step variant (same machinery, not yet built).
`--stride-sec` controls window overlap: 8 (default) gives 50% overlap and 14.5K clips;
`--stride-sec 1` is full TAPIS parity (~116K clips, ~2 h build, ~6.7 TB of features) —
**not recommended**, since at 93.8% overlap a frozen encoder just re-encodes near-duplicate
windows, and the heads are already overfitting rather than data-starved.

The builder hard-refuses to encode if verification fails, so a silent regression can't
reach a probe run.

Provenance: commits `6649109` (builder), `ac64a03` (configs), `d9b004a` (scorer fixes),
`31b84d3` (ledger banner).
