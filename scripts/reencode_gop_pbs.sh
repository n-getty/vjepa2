#!/usr/bin/env bash
# PBS launcher: GOP re-encode the sparse-keyframe sources (dense keyframes, so a
# 16-frame scatter-seek stops walking hundreds of frames per sample).
#
# WHY, and WHAT THIS DOES NOT BUY
# --------------------------------
# GOP is the measured predictor of decode time, not payload size. Pearson r vs
# decode: MB/clip +0.20, frame count +0.50, GOP +0.62, and min(16*GOP, frames)
# -- the frames a 16-seek scatter actually forces the decoder through -- +0.70.
# The largest source on disk (grasp_noleak, 96.6 MB/clip) is among the FASTEST.
# Full table: docs/THROUGHPUT_RECIPE_AURORA.md:779-840.
#
# This moves the BODY of the per-rank dataload distribution -- p50, mean, p90 --
# permanently, offline, at no scaling-slot cost. It does NOT fix the scaling
# asymptote: 8.7% of observed samples exceed the 10.74 s bs=2 pure-decode ceiling
# (2 x lapgyn6_events at 5.37 s, the slowest clip ever measured offline), median
# 15.1 s and max 67.8 s. A re-encode cannot make a clip decode faster than a clip
# decodes, so whatever produces those draws is untouched by this job. Do not
# report this as the scaling fix.
#
# TWO SEPARABLE LEVERS, and the source list is split on which one it needs:
#   --short-side 0  = GOP only, pixels untouched. Correct for the sparse-GOP
#                     sources. Verified on a real cholec80 clip: 3.93 -> 0.43 s
#                     (9.2x), 1742 frames preserved, 20.9 -> 13.8 MB.
#   --short-side 512 = also downscale. Needed ONLY by sitl_2026, which is already
#                     GOP-30, so its cost is pixels: g16 alone is 3.55 -> 2.35 s
#                     (1.5x) but 512p reaches 0.61 s (5.8x).
# Blanket downscaling would be wrong: cholec80 is 854x480, below the 512 target,
# so --short-side 512 would UPSCALE it -- more disk and slower decode.
#
# Submit ONE source per job (each is sized differently; see the table below):
#   qsub -A AuroraGPT -q debug -l select=1 -l walltime=01:00:00 \
#        -l filesystems=home:flare -v SRC=cholec80 scripts/reencode_gop_pbs.sh
#
#   source          shards   size   GOP   arm            measured decode
#   cholec80           256    70G   249   native g16     3.04 -> 0.35 s (8.8x)
#   surgvu24_clean    2000   320G   250   native g16     3.38 -> 0.41 s (8.2x)
#   lemon              528   922G    96   native g16     2.77 -> 0.63 s (4.4x)
#   sitl_2026         1454   551G    30   512p g16       3.55 -> 0.61 s (5.8x)
#
# Idempotent (existing output shards are skipped), sample COUNT is preserved, and
# a failed encode copies the ORIGINAL bytes through rather than dropping data. So
# a job that runs out of walltime is resumed by resubmitting it unchanged.
#
# NOTE: no `set -u` (memory set-u-module-load-trap: Lmod init trips it).
#
#PBS -N reenc_gop
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/
set -o pipefail

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
PY=/flare/ModCon/ngetty/venvs/torchtune-pt213-xpu/bin/python
DATA=/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded

SRC=${SRC:?set -v SRC=<source>, one of: cholec80 surgvu24_clean lemon sitl_2026}
IN=$DATA/$SRC

# Per-source arm. sitl_2026 is the ONLY one that needs the resolution lever --
# see the header. Everything else keeps native pixels.
case "$SRC" in
  sitl_2026) SHORT=512; SUFFIX=_g16_512 ;;
  cholec80|surgvu24_clean|lemon) SHORT=0; SUFFIX=_g16 ;;
  *) echo "FATAL: $SRC is not in the measured sparse-GOP set. Add it to the table"
     echo "       in the header with its measured decode delta first -- re-encoding"
     echo "       a source we have not benchmarked spends node hours on a guess."
     exit 1 ;;
esac
OUT=$DATA/${SRC}${SUFFIX}

[ -d "$IN" ] || { echo "FATAL: input missing: $IN"; exit 1; }
TOT=$(ls "$IN"/*.tar 2>/dev/null | wc -l)
[ "$TOT" -gt 0 ] || { echo "FATAL: no shards in $IN"; exit 1; }

LOGDIR=/flare/ModCon/ngetty/logs/reenc_${SRC}_workers
mkdir -p "$LOGDIR" "$OUT" /flare/ModCon/ngetty/logs
cd "$ROOT"
export PYTHONPATH="$ROOT:$PYTHONPATH"

# Aurora compute node = 104 cores / 208 threads. 32 workers x 4 x264 threads =
# 128 threads, ~60% of the node, leaving headroom for tar I/O. Same shape as
# reencode_heichole_pbs.sh, which ran this to completion.
NW=${NW:-32}
FFT=${FFT:-4}
STEP=$(( (TOT + NW - 1) / NW ))

echo "=== $SRC re-encode: $TOT shards, $NW workers x $STEP, ${FFT} x264 threads ==="
echo "arm: short_side=$SHORT (0 = native pixels, GOP only)  gop=16 crf=23"
echo "in : $IN"
echo "out: $OUT"
date

# Stale .tmp from an interrupted run would otherwise be mistaken for output.
rm -f "$OUT"/*.tmp 2>/dev/null || true

DONE_BEFORE=$(ls "$OUT"/*.tar 2>/dev/null | wc -l)
echo "already done: $DONE_BEFORE / $TOT (idempotent resume)"

pids=()
for i in $(seq 0 $((NW-1))); do
  lo=$(( i*STEP )); hi=$(( lo+STEP )); [ $hi -gt $TOT ] && hi=$TOT
  [ $lo -ge $TOT ] && break
  "$PY" scripts/reencode_source_reshard.py --input "$IN" --output "$OUT" \
      --short-side "$SHORT" --crf 23 --gop 16 --ffmpeg-threads "$FFT" \
      --shard-start "$lo" --shard-end "$hi" \
      > "$LOGDIR/w_${lo}_${hi}.log" 2>&1 &
  pids+=($!)
done
echo "launched ${#pids[@]} workers"

fail=0
for p in "${pids[@]}"; do
  wait "$p" || { echo "worker pid $p exited nonzero"; fail=1; }
done
echo "=== workers done (fail=$fail) ==="

NDONE=$(ls "$OUT"/*.tar 2>/dev/null | wc -l)
echo "output shards: $NDONE / $TOT  (this job added $(( NDONE - DONE_BEFORE )))"

# Finalize ONLY when complete. metadata.json is what the loader trusts for
# sample counts; writing it over a partial set would silently shrink the source.
if [ "$NDONE" -eq "$TOT" ]; then
  echo "=== finalize metadata.json ==="
  "$PY" scripts/reencode_source_reshard.py --input "$IN" --output "$OUT" --finalize
  echo "COMPLETE. Verify before swapping any config to $OUT:"
  echo "  1. sample count matches the original metadata.json"
  echo "  2. re-run the offline decode benchmark and confirm the predicted drop"
else
  echo "INCOMPLETE ($NDONE/$TOT) -- metadata.json NOT written, which is correct:"
  echo "a metadata.json over a partial shard set would silently shrink the source."
  echo "Resubmit this same job unchanged to resume."
fi
date
