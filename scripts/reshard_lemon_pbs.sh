#!/usr/bin/env bash
# Reshard LEMON staging -> loader-ready fine shards, PARALLEL (the serial
# scripts/reshard_webdataset.py walltime-killed on 4162 tars / ~1.9 TB: its
# single-threaded index-then-write can't fit 1h). Fan N workers over disjoint
# source-tar subsets; each writes its own lemon-wII-*.tar shards; finalize
# merges metadata. See scripts/reshard_lemon_parallel.py.
#
# Submit (after regate):
#   qsub scripts/reshard_lemon_pbs.sh
#
#PBS -N lemon_reshard
#PBS -A AuroraGPT
#PBS -q debug-scaling
#PBS -l select=1:ncpus=104
#PBS -l walltime=01:00:00
#PBS -l filesystems=home:flare
#PBS -j oe
#PBS -o /flare/ModCon/ngetty/logs/

set -eo pipefail
umask 077

module load frameworks 2>/dev/null || module load frameworks

ROOT=/lus/flare/projects/ModCon/ngetty/vjepa2
DATA=/flare/ModCon/ngetty/data
STAGING="$DATA/surg_vid_webdataset_resharded/lemon_staging"
OUTPUT="$DATA/surg_vid_webdataset_resharded/lemon"
PYTHON="$(command -v python3)"
NWORK="${NWORK:-24}"
SHARDS_PER_WORKER="${SHARDS_PER_WORKER:-22}"   # ~24*22 = 528 total shards (>world_size 192)

mkdir -p "$OUTPUT" /flare/ModCon/ngetty/logs
cd "$ROOT"
echo "JOB START: $(date) PBS_JOBID=${PBS_JOBID:-} NWORK=$NWORK"
# fresh output
rm -f "$OUTPUT"/*.tar "$OUTPUT"/_wpartial_*.json "$OUTPUT"/metadata.json "$OUTPUT"/reshard_summary.json 2>/dev/null || true

pids=()
for (( w=0; w<NWORK; w++ )); do
  "$PYTHON" "$ROOT/scripts/reshard_lemon_parallel.py" \
      --staging "$STAGING" --output "$OUTPUT" \
      --n-workers "$NWORK" --worker-id "$w" \
      --shards-per-worker "$SHARDS_PER_WORKER" --seed 0 \
      > "$OUTPUT/_wlog_${w}.log" 2>&1 &
  pids+=($!)
done
fail=0; for pid in "${pids[@]}"; do wait "$pid" || fail=$((fail+1)); done
echo "workers done, failed=$fail"; (( fail>0 )) && { echo "see $OUTPUT/_wlog_*.log" >&2; exit 1; }

"$PYTHON" "$ROOT/scripts/reshard_lemon_parallel.py" \
    --staging "$STAGING" --output "$OUTPUT" --finalize

chmod 700 "$OUTPUT"; chmod 600 "$OUTPUT"/*.tar "$OUTPUT"/*.json 2>/dev/null || true
echo "JOB END: $(date)"
"$PYTHON" -c "import json;d=json.load(open('$OUTPUT/metadata.json'));print({k:d[k] for k in ['name','shard_count','sample_count']})"
stat -c '%A %U %n' "$OUTPUT"
echo "NOTE: next run scripts/filter_black_clips_pbs.sh -v DATASET=lemon"
