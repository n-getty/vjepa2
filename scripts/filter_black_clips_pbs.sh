#!/usr/bin/env bash
# Source-side black-clip filter for surgvu24: bake the runtime min_clip_std
# reject into a surgvu24_clean copy so we stop re-decoding/re-rejecting ~19-27%
# dead clips every epoch (and stop copying them during staging).
#
# CPU-only (decord decode + numpy std, no GPU). Fans out N parallel workers over
# disjoint shard ranges on ONE node, then finalizes metadata. ~15.6s/shard
# single-thread x 2000 shards / NPROC workers. At NPROC=48 => ~11 min.
#
# Submit:
#   qsub /lus/flare/projects/ModCon/ngetty/vjepa2/scripts/filter_black_clips_pbs.sh
#
#PBS -N vjepa_blackfilter
#PBS -A AuroraGPT
#PBS -q debug
#PBS -l select=1:ncpus=104
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
RESHARD_ROOT=/flare/ModCon/ngetty/data/surg_vid_webdataset_resharded
# DATASET selects which source to filter; defaults to surgvu24 (original use).
# Override for the segmented sources: qsub -v DATASET=cholec80 ... / DATASET=grasp
DATASET="${DATASET:-surgvu24}"
INPUT="$RESHARD_ROOT/$DATASET"
OUTPUT="$RESHARD_ROOT/${DATASET}_clean"
PYTHON=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
NPROC=48

mkdir -p "$OUTPUT" /flare/ModCon/ngetty/logs
cd "$ROOT"
echo "JOB START: $(date) PBS_JOBID=${PBS_JOBID:-} NPROC=$NPROC"

# Total shard count.
NSHARDS=$("$PYTHON" -c "import glob; print(len(glob.glob('$INPUT/*.tar')))")
echo "input shards: $NSHARDS"
if [[ "$NSHARDS" -eq 0 ]]; then echo "no shards, abort" >&2; exit 1; fi

# Chunk size per worker (ceil).
CHUNK=$(( (NSHARDS + NPROC - 1) / NPROC ))
echo "chunk/worker: $CHUNK"

pids=()
for (( lo=0; lo<NSHARDS; lo+=CHUNK )); do
  hi=$(( lo + CHUNK )); (( hi > NSHARDS )) && hi=$NSHARDS
  log="$OUTPUT/_worker_${lo}_${hi}.log"
  "$PYTHON" "$ROOT/scripts/filter_black_clips_reshard.py" \
      --input "$INPUT" --output "$OUTPUT" \
      --min-clip-std 1.0 --fps 4 --frames-per-clip 16 \
      --shard-start "$lo" --shard-end "$hi" > "$log" 2>&1 &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "workers done, failed=$fail"
(( fail > 0 )) && { echo "inspect $OUTPUT/_worker_*.log" >&2; exit 1; }

# Merge partials -> metadata.json + reshard_summary.json.
"$PYTHON" "$ROOT/scripts/filter_black_clips_reshard.py" \
    --input "$INPUT" --output "$OUTPUT" --finalize

echo "JOB END: $(date)"
echo "=== output ==="
"$PYTHON" -c "import json; d=json.load(open('$OUTPUT/reshard_summary.json')); print(json.dumps(d, indent=2))"
