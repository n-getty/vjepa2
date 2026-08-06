#!/usr/bin/env bash
# PBS launcher: GOP re-encode the sparse-keyframe sources, so a 16-frame
# scatter-seek stops walking hundreds of frames per sample.
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
#   --short-side 0   GOP only, pixels untouched. Correct for the sparse-GOP
#                    sources. Verified on a real cholec80 clip: 3.93 -> 0.43 s
#                    (9.2x), 1742 frames preserved, 20.9 -> 13.8 MB.
#   --short-side 512 also downscale. Needed ONLY by sitl_2026, which is already
#                    GOP-30, so its cost is pixels: g16 alone is 3.55 -> 2.35 s
#                    (1.5x) but 512p reaches 0.61 s (5.8x).
# Blanket downscaling would be wrong: cholec80 is 854x480, below the 512 target,
# so --short-side 512 would UPSCALE it -- more disk and slower decode.
#
# ONE SOURCE PER JOB, sized to the source. heichole (the precedent this is
# derived from) was 256 shards / 14 GB and fit one node; these are 100x that in
# bytes, so the fan-out is over nodes as well as cores:
#
#   source          shards   size   GOP   arm          predicted     suggested
#   cholec80           256    70G   249   native g16   3.04->0.35 s   8 nodes
#   surgvu24_clean    2000   320G   250   native g16   3.38->0.41 s  32 nodes
#   lemon              528   922G    96   native g16   2.77->0.63 s  16 nodes
#   sitl_2026         1454   551G    30   512p  g16    3.55->0.61 s  45 nodes
#
#   qsub -A AuroraGPT -q debug-scaling -l select=8 -l walltime=01:00:00 \
#        -l filesystems=home:flare -v SRC=cholec80 scripts/reencode_gop_pbs.sh
#
# Node counts above assume NW=32 workers/node; useful parallelism CAPS at
# TOT/NW nodes because a worker's unit of work is one shard (lemon's 528 shards
# cannot use more than 16 nodes at NW=32). The script clamps and says so.
#
# CPU-only -- no XPU, no CCL, no distributed bootstrap. mpiexec is used purely
# as a process launcher; ranks never talk to each other. They coordinate only by
# writing disjoint shard ranges into a shared output directory.
#
# Idempotent (existing output shards are skipped), sample COUNT is preserved, and
# a failed encode copies the ORIGINAL bytes through rather than dropping data. So
# a job that runs out of walltime is resumed by resubmitting it -- but
#
#   *** A RESUME MUST REUSE THE SAME select= AND NW= AS THE RUN IT RESUMES. ***
#
# Shards are safe at any layout (written .tmp then os.replace, so a killed shard
# leaves a .tmp the resume does not skip). The MANIFEST is not. Provenance lives
# in _partial_{lo}_{hi}.json, whose name is the shard range a proc owned, and
# the range is derived from nodes x NW. Resume at a different layout and the
# ranges are renamed and overlapping, so finalize() cannot match the new
# partials against the old ones -- it sums per_shard for sample_count and
# silently reports only the shards the RESUME touched. Measured on a 4-shard
# fixture: sample_count 4 against a truth of 8, with shard_count correct and
# nothing on stderr (fixed 0626c19: same-layout resumes now merge, and finalize
# hard-errors on all-shards-present-but-fewer-described). A complete corpus with
# a short manifest is the worst outcome here -- bad pixels get noticed, a bad
# manifest gets believed.
#
# SIZING, measured rather than predicted (cholec80 job 8740763, surgvu24
# 8740788): cost is ~23 s PER SAMPLE (cholec80 21.2, surgvu24 25.2) and
# bytes/sample does NOT predict it -- cholec80 is 3x the bytes per sample and
# FASTER. Size a job by samples/shard, not by GB:
#
#   source          samples/shard   -> ~single-shard wall at FFT=4
#   sitl_2026                 8.0      ~3 min
#   cholec80                 11.4      ~4 min
#   surgvu24_clean           20.2      ~8 min
#   lemon                   101.6      ~39 min      <-- against a 1 h cap
#
# A shard is INDIVISIBLE, so single-shard wall is a floor no node count lowers.
# When it approaches the cap, spend the node's threads on latency instead of
# throughput: NW=16 FFT=8 is the same 128 threads/node as the default NW=32
# FFT=4, but halves the time for any one shard. That is why lemon runs
# select=33 NW=16 FFT=8 (528 procs = exactly 1 shard each) rather than the
# 16 nodes the table below suggests.
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

NNODES=$(sort -u "$PBS_NODEFILE" | wc -l)
# Aurora compute node = 104 cores / 208 threads. 32 workers x 4 x264 threads =
# 128 threads, ~60% of the node, leaving headroom for tar I/O. Same shape as
# reencode_heichole_pbs.sh, which ran this to completion on one node.
NW=${NW:-32}
FFT=${FFT:-4}

NPROC=$(( NNODES * NW ))
if [ "$NPROC" -gt "$TOT" ]; then
  # A worker's unit of work is one shard, so more workers than shards leaves the
  # surplus idle -- harmless, but it means the extra nodes bought nothing and the
  # job should have asked for fewer. Say so rather than let it look like it scaled.
  echo "NOTE: $NPROC workers > $TOT shards. Useful parallelism caps at"
  echo "      $(( (TOT + NW - 1) / NW )) nodes at NW=$NW; $NNODES were allocated."
  NPROC=$TOT
fi
STEP=$(( (TOT + NPROC - 1) / NPROC ))
# Integer chunking rounds STEP up, so the last ranks can fall off the end: at
# 528 shards over 512 procs, STEP=2 and only 264 procs get work while 248 exit
# immediately. Coverage is still exact -- but the allocation is half idle, and a
# log line saying "512 procs" would hide that. Report what actually runs.
USED=$(( (TOT + STEP - 1) / STEP ))

echo "=== $SRC re-encode ==="
echo "shards : $TOT"
echo "layout : $NNODES nodes x $NW workers = $NPROC procs, $STEP shards each, ${FFT} x264 threads"
if [ "$USED" -lt "$NPROC" ]; then
  echo "         ...but only $USED procs get work (integer chunking at STEP=$STEP);"
  echo "         $(( NPROC - USED )) idle. $(( (USED + NW - 1) / NW )) nodes would do the same job."
fi
echo "arm    : short_side=$SHORT (0 = native pixels, GOP only)  gop=16 crf=23"
echo "in     : $IN"
echo "out    : $OUT"

# RESUME LAYOUT GUARD. Provenance lives in _partial_{lo}_{hi}.json and the range
# width is STEP, so resuming at a different STEP renames every partial. The old
# ones then describe shards that finalize() cannot match to the new ranges, and
# sample_count silently reports only what the resume touched -- a complete shard
# set with a short manifest. Guard on STEP rather than on nodes/NW: STEP is what
# the names are actually derived from, so a resume at half the nodes and double
# NW is legal and correctly passes.
PRIOR_STEP=""
for f in "$OUT"/_partial_*.json; do
  [ -e "$f" ] || break
  b=$(basename "$f" .json); b=${b#_partial_}
  PRIOR_STEP=$(( ${b#*_} - ${b%_*} ))
  break
done
if [ -n "$PRIOR_STEP" ] && [ "$PRIOR_STEP" -ne "$STEP" ]; then
  echo
  echo "FATAL: $OUT already holds partials at STEP=$PRIOR_STEP; this job would"
  echo "       write STEP=$STEP. Resuming at a different layout renames the"
  echo "       _partial_ ranges, and finalize() would then undercount"
  echo "       sample_count while reporting a complete shard set -- a corpus"
  echo "       that looks whole with a manifest that is short."
  echo
  echo "       Resubmit with the layout that produced the existing partials"
  echo "       (STEP=$PRIOR_STEP, i.e. $(( (TOT + PRIOR_STEP - 1) / PRIOR_STEP )) procs:"
  echo "       e.g. -l select=$(( ((TOT + PRIOR_STEP - 1) / PRIOR_STEP + NW - 1) / NW )) with NW=$NW),"
  echo "       or start clean by removing $OUT."
  exit 1
fi
date

# Stale .tmp from an interrupted run would otherwise be mistaken for output.
rm -f "$OUT"/*.tmp 2>/dev/null || true

DONE_BEFORE=$(ls "$OUT"/*.tar 2>/dev/null | wc -l)
echo "already done: $DONE_BEFORE / $TOT (idempotent resume)"

# Each rank derives its own disjoint shard range from its MPI rank id. Nothing is
# communicated; PALS_RANKID is read inside the spawned shell, not here.
# --cpu-bind depth --depth 4 gives each worker its own 4 hardware threads, which
# is what FFT=4 x264 threads expect; without it ffmpeg instances collide on core 0.
cat > "$PBS_O_WORKDIR/.reenc_worker_$SRC.sh" <<WORKER
#!/usr/bin/env bash
R=\${PALS_RANKID:-\${PMI_RANK:-\${OMPI_COMM_WORLD_RANK:-0}}}
lo=\$(( R * $STEP )); hi=\$(( lo + $STEP ))
[ \$hi -gt $TOT ] && hi=$TOT
[ \$lo -ge $TOT ] && exit 0
export PYTHONPATH="$ROOT:\$PYTHONPATH"
cd "$ROOT"
exec "$PY" scripts/reencode_source_reshard.py --input "$IN" --output "$OUT" \\
    --short-side $SHORT --crf 23 --gop 16 --ffmpeg-threads $FFT \\
    --shard-start \$lo --shard-end \$hi \\
    > "$LOGDIR/w_\${lo}_\${hi}.log" 2>&1
WORKER
chmod +x "$PBS_O_WORKDIR/.reenc_worker_$SRC.sh"

mpiexec -n "$NPROC" -ppn "$NW" --cpu-bind depth --depth 4 --no-vni \
    "$PBS_O_WORKDIR/.reenc_worker_$SRC.sh"
rc=$?
echo "=== mpiexec rc=$rc ==="

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
